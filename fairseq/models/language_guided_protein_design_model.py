"""InstructPro2: instruction-following protein design with biological grounding.

Architecture (see Figure 1 of the paper):

  * a causally-masked text encoder over natural-language function descriptions,
    whose output is projected and prefixed to the protein token embeddings;
  * frozen biological encoders for ligands (ChemBERTa) and antigens (ESM-2),
    injected through zero-initialized gated cross-attention adapters placed
    every fourth decoder block;
  * a GPT-J-style autoregressive protein decoder initialized from
    ProGen2-large-BFD90;
  * an auxiliary text decoder that reconstructs the conditioning description
    from the decoder's hidden states at the text positions.
"""

import os

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers import AutoModel, AutoTokenizer
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map


logger = logging.get_logger(__name__)

TEXT_ENCODER = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
LIGAND_ENCODER = "seyonec/SMILES_tokenized_PubChem_shard00_160k"
ANTIGEN_ENCODER = "facebook/esm2_t12_35M_UR50D"

STAGE2_MODES = (
    "stage2",
    "stage2_frozen_decoder",
    "stage2_partial",
    "stage2_adapters_only",
)

# Modality type-embedding indices. Fixed, since they are baked into trained
# checkpoints.
MODALITY_LIGAND = 0
MODALITY_ANTIGEN = 1
NUM_MODALITIES = 2

# Historical mode names, accepted on load so that checkpoints written before
# the rename still resolve. Deprecated: new configs should use the names above.
_MODE_ALIASES = {
    "default": "stage1",
    "flamingo_full": "stage2",
    "flamingo": "stage2_frozen_decoder",
    "flamingo_partial": "stage2_partial",
    "flamingo_crossattn_only": "stage2_adapters_only",
}


class ProGenConfig(PretrainedConfig):
    model_type = "progen"

    def __init__(
        self,
        vocab_size=50400,
        n_positions=2048,
        n_ctx=2048,
        n_embd=4096,
        n_layer=28,
        n_head=16,
        rotary_dim=64,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        scale_attn_weights=True,
        gradient_checkpointing=False,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        text_embedding_dim=768,
        n_text_layer=3,
        ligand_embedding_dim=768,
        antigen_embedding_dim=480,  # ESM-2 35M
        architecture_mode="stage1",
        adapter_frequency=4,
        frozen_layer_start=0,
        frozen_layer_end=20,
        use_segment_rope=False,
        text_protein_rope_gap=0.0,
        text_rope_scale=1.0,
        **kwargs,
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.rotary_dim = rotary_dim
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.gradient_checkpointing = gradient_checkpointing
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.text_embedding_dim = text_embedding_dim
        self.n_text_layer = n_text_layer
        self.ligand_embedding_dim = ligand_embedding_dim
        self.antigen_embedding_dim = antigen_embedding_dim
        self.architecture_mode = _MODE_ALIASES.get(architecture_mode, architecture_mode)
        self.adapter_frequency = adapter_frequency
        self.frozen_layer_start = frozen_layer_start
        self.frozen_layer_end = frozen_layer_end
        self.use_segment_rope = use_segment_rope
        self.text_protein_rope_gap = text_protein_rope_gap
        self.text_rope_scale = text_rope_scale
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    @property
    def max_position_embeddings(self):
        return self.n_positions

    @property
    def hidden_size(self):
        return self.n_embd

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return self.n_layer


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    return torch.stack((-x2, x1), axis=-1).flatten(-2)


def fixed_pos_embedding(positions, dim):
    """positions: (1 or batch, seq_len) float/long tensor of rotary positions."""
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32) / dim)
    )
    sinusoid_inp = torch.einsum("bi,j->bij", positions.float(), inv_freq)
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def apply_rotary_pos_emb(x, sincos):
    """x: (batch, seq_len, num_heads, rotary_dim); sincos each (1|B, seq_len, dim/2)."""
    sin, cos = sincos
    sin = sin[:, :, None, :].repeat_interleave(2, -1)
    cos = cos[:, :, None, :].repeat_interleave(2, -1)
    return (x * cos) + (rotate_every_two(x) * sin)


class GatedCrossAttention(nn.Module):
    """Zero-initialized gated cross-attention adapter (Alayrac et al., 2022).

    The tanh(alpha) gate starts at zero, so the modality pathway contributes no
    signal at the beginning of Stage 2 and the pretrained decoder's computation
    is preserved exactly.
    """

    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.attn_pdrop,
            batch_first=True,
        )
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states, cross_attn_context, cross_attn_pad_mask=None,
                output_attentions=False):
        # Upcast the LayerNorm to fp32: in bf16 the variance computation can
        # lose enough precision to destabilize training.
        x_norm = F.layer_norm(
            hidden_states.float(),
            self.norm.normalized_shape,
            self.norm.weight.float() if self.norm.weight is not None else None,
            self.norm.bias.float() if self.norm.bias is not None else None,
            self.norm.eps,
        ).to(hidden_states.dtype)

        # A fully-masked key row makes softmax produce NaN, so unmask one
        # position for those rows and zero the output afterwards.
        safe_pad_mask = cross_attn_pad_mask
        all_masked = None
        if cross_attn_pad_mask is not None:
            all_masked = cross_attn_pad_mask.all(dim=1)
            if all_masked.any():
                safe_pad_mask = cross_attn_pad_mask.clone()
                safe_pad_mask[all_masked, 0] = False

        attn_out, attn_weights = self.cross_attn(
            query=x_norm,
            key=cross_attn_context,
            value=cross_attn_context,
            key_padding_mask=safe_pad_mask,
            need_weights=output_attentions,
        )

        if all_masked is not None and all_masked.any():
            attn_out[all_masked] = 0.0

        output = hidden_states + (torch.tanh(self.alpha) * attn_out)
        return output, attn_weights


class ProGenAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.register_buffer("masked_bias", torch.tensor(-1e9))
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_attention_heads")
        self.scale_attn = torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32)
        ).to(torch.get_default_dtype())
        self.qkv_proj = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.rotary_dim = config.rotary_dim

    def _split_heads(self, x, n_head, dim_head, mp_num):
        reshaped = x.reshape(x.shape[:-1] + (n_head // mp_num, dim_head))
        return reshaped.reshape(x.shape[:-2] + (-1,) + reshaped.shape[-1:])

    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        if len(tensor.shape) == 5:
            tensor = tensor.permute(0, 1, 3, 2, 4).contiguous()
        elif len(tensor.shape) == 4:
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
        else:
            raise ValueError("Input tensor rank should be 4 or 5")
        new_shape = tensor.size()[:-2] + (num_attention_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        query_length, key_length = query.size(-2), key.size(-2)
        if query_length == key_length:
            causal_mask = torch.tril(
                torch.ones((query_length, key_length), device=query.device, dtype=torch.bool)
            ).view(1, 1, query_length, key_length)
        else:
            # Incremental decoding: the single new query attends to all cached keys.
            causal_mask = torch.ones(
                (1, 1, query_length, key_length), device=query.device, dtype=torch.bool
            )

        query = query.to(torch.float32)
        key = key.to(torch.float32)

        attn_weights = torch.matmul(query, key.transpose(-1, -2)) / self.scale_attn
        attn_weights = torch.where(
            causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype)
        )
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights).to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        return torch.matmul(attn_weights, value), attn_weights

    def forward(self, hidden_states, attention_mask=None, layer_past=None,
                head_mask=None, use_cache=False, output_attentions=False,
                position_ids=None):
        qkv = self.qkv_proj(hidden_states)
        mp_num = 8
        qkv_split = qkv.reshape(qkv.shape[:-1] + (mp_num, -1))
        local_dim = self.head_dim * self.num_attention_heads // mp_num
        query, value, key = torch.split(qkv_split, local_dim, dim=-1)

        query = self._split_heads(query, self.num_attention_heads, self.head_dim, mp_num)
        key = self._split_heads(key, self.num_attention_heads, self.head_dim, mp_num)
        value = self._split_heads(value, self.num_attention_heads, self.head_dim, mp_num)
        value = value.permute(0, 2, 1, 3)

        cur_len = key.shape[1]
        offset = layer_past[0].shape[-2] if layer_past is not None else 0

        if position_ids is None:
            positions = torch.arange(
                offset, offset + cur_len, device=key.device, dtype=torch.float32
            )[None, :]
        else:
            positions = position_ids.to(device=key.device, dtype=torch.float32)
            assert positions.shape[-1] == cur_len, (
                f"position_ids length {positions.shape[-1]} != token count {cur_len}"
            )

        if self.rotary_dim is not None:
            k_rot, k_pass = key[:, :, :, : self.rotary_dim], key[:, :, :, self.rotary_dim:]
            q_rot, q_pass = query[:, :, :, : self.rotary_dim], query[:, :, :, self.rotary_dim:]
            sincos = fixed_pos_embedding(positions, self.rotary_dim)
            key = torch.cat([apply_rotary_pos_emb(k_rot, sincos), k_pass], dim=-1)
            query = torch.cat([apply_rotary_pos_emb(q_rot, sincos), q_pass], dim=-1)
        else:
            sincos = fixed_pos_embedding(positions, key.shape[-1])
            key = apply_rotary_pos_emb(key, sincos)
            query = apply_rotary_pos_emb(query, sincos)

        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)

        if layer_past is not None:
            key = torch.cat((layer_past[0], key), dim=-2)
            value = torch.cat((layer_past[1], value), dim=-2)

        present = (key, value) if use_cache else None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        attn_output = self._merge_heads(
            attn_output, self.num_attention_heads, self.head_dim
        )
        attn_output = self.resid_dropout(self.out_proj(attn_output))

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class ProGenMLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.n_embd
        self.fc_in = nn.Linear(embed_dim, intermediate_size)
        self.fc_out = nn.Linear(intermediate_size, embed_dim)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        return self.dropout(hidden_states)


class ProGenBlock(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = ProGenAttention(config)
        self.mlp = ProGenMLP(inner_dim, config)

        self.has_cross_attn = (
            config.architecture_mode in STAGE2_MODES
            and (layer_idx + 1) % config.adapter_frequency == 0
        )
        if self.has_cross_attn:
            self.cross_attn = GatedCrossAttention(config)

    def forward(self, hidden_states, layer_past=None, attention_mask=None,
                head_mask=None, use_cache=False, output_attentions=False,
                cross_attn_context=None, cross_attn_pad_mask=None, position_ids=None):
        residual = hidden_states
        hidden_states_norm = self.ln_1(hidden_states)

        attn_outputs = self.attn(
            hidden_states_norm,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            position_ids=position_ids,
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]

        # GPT-J-style parallel attention and feed-forward.
        feed_forward_hidden_states = self.mlp(hidden_states_norm)
        hidden_states = residual + attn_output + feed_forward_hidden_states

        cross_attn_weights = None
        if self.has_cross_attn and cross_attn_context is not None:
            hidden_states, cross_attn_weights = self.cross_attn(
                hidden_states, cross_attn_context, cross_attn_pad_mask,
                output_attentions=output_attentions,
            )

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        if output_attentions:
            outputs = outputs + (cross_attn_weights,)
        return outputs


class ProGenPreTrainedModel(PreTrainedModel):
    config_class = ProGenConfig
    base_model_prefix = "transformer"
    is_parallelizable = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class ProGenModel(ProGenPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [ProGenBlock(config, layer_idx=i) for i in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.rotary_dim = min(config.rotary_dim, config.n_ctx // config.num_attention_heads)
        self.init_weights()
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.h))
        self.model_parallel = True
        self.first_device = (
            "cpu" if "cpu" in self.device_map.keys()
            else "cuda:" + str(min(self.device_map.keys()))
        )
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        self.wte = self.wte.to(self.first_device)
        for k, v in self.device_map.items():
            for block in v:
                self.h[block] = self.h[block].to("cuda:" + str(k))
        self.ln_f = self.ln_f.to(self.last_device)

    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.wte = self.wte.to("cpu")
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")
        torch.cuda.empty_cache()

    def forward(self, input_ids=None, past_key_values=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None,
                cross_attn_context=None, cross_attn_pad_mask=None):
        output_attentions = (
            output_attentions if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds, not both")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("Specify one of input_ids or inputs_embeds")

        dev = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(
                past_length, input_shape[-1] + past_length, dtype=torch.long, device=dev
            ).unsqueeze(0).view(-1, input_shape[-1])

        if attention_mask is not None and attention_mask.dim() == 2:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask.to(dtype=self.dtype)) * -1e9

        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        hidden_states = inputs_embeds
        if token_type_ids is not None:
            hidden_states = hidden_states + self.wte(token_type_ids)
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                if layer_past is not None:
                    layer_past = tuple(p.to(hidden_states.device) for p in layer_past)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(
                            *inputs, use_cache, output_attentions,
                            cross_attn_context, cross_attn_pad_mask, position_ids,
                        )
                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states, None, attention_mask, head_mask[i],
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    cross_attn_context=cross_attn_context,
                    cross_attn_pad_mask=cross_attn_pad_mask,
                    position_ids=position_ids,
                )

            hidden_states = outputs[0]
            if use_cache:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                cross_idx = 3 if use_cache else 2
                all_cross_attentions = all_cross_attentions + (
                    outputs[cross_idx] if len(outputs) > cross_idx else None,
                )

            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states).view(*output_shape)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            res = [
                hidden_states, presents, all_hidden_states,
                all_self_attentions, all_cross_attentions,
            ]
            return tuple(v for v in res if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


@register_model("language_guided_protein_design_model")
class ProGenForCausalLM(ProGenPreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"h\.\d+\.attn\.masked_bias", r"h\.\d+\.attn\.bias", r"lm_head\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = ProGenModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.init_weights()

        n_text_layer = getattr(config, "n_text_layer", 3)

        # --- Text encoder and auxiliary text decoder (PubMedBERT) ---
        # The encoder is the first n_text_layer blocks of PubMedBERT; the
        # decoder reconstructs the conditioning description from the protein
        # decoder's hidden states at the text positions, and is initialized
        # from the last n_text_layer blocks.
        self.text_embedding_dim = config.text_embedding_dim
        self.text_tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER)
        bert_full = AutoModel.from_pretrained(TEXT_ENCODER)

        self.text_tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<FUNCTION>", "</FUNCTION>"]}
        )
        bert_full.resize_token_embeddings(len(self.text_tokenizer))

        self.text_embeddings = bert_full.embeddings
        self.text_encoder_layers = nn.ModuleList(bert_full.encoder.layer[:n_text_layer])
        self.text_mapping_layer = nn.Linear(config.text_embedding_dim, config.n_embd)
        self.text_decoder_projection = nn.Linear(config.n_embd, config.text_embedding_dim)
        self.text_decoder_blocks = nn.ModuleList(bert_full.encoder.layer[-n_text_layer:])
        self.text_decoder_ln = nn.LayerNorm(
            config.text_embedding_dim, eps=config.layer_norm_epsilon
        )
        self.text_lm_head = nn.Linear(config.text_embedding_dim, len(self.text_tokenizer))
        del bert_full

        # --- Frozen biological encoders ---
        self.ligand_encoder = AutoModel.from_pretrained(LIGAND_ENCODER)
        self.ligand_mapping_layer = nn.Linear(config.ligand_embedding_dim, config.n_embd)

        self.antigen_encoder = AutoModel.from_pretrained(ANTIGEN_ENCODER)
        self.antigen_mapping_layer = nn.Linear(config.antigen_embedding_dim, config.n_embd)

        # Per-modality learned scale s_m and type embedding e_m from Eq. 3.
        self.modality_type_embedding = nn.Embedding(NUM_MODALITIES, config.n_embd)
        nn.init.normal_(self.modality_type_embedding.weight, mean=0.0, std=1e-5)
        self.modality_scale = nn.ParameterDict({
            "ligand": nn.Parameter(torch.ones(1)),
            "antigen": nn.Parameter(torch.ones(1)),
        })

        for encoder in (self.ligand_encoder, self.antigen_encoder):
            for param in encoder.parameters():
                param.requires_grad = False

        self._apply_freezing(config)

        self.model_parallel = False
        self.device_map = None
        self.num_updates = 0

        # Zero-initialize the modality pathway so it contributes nothing at the
        # start of Stage 2, leaving the Stage 1 computation exactly preserved.
        for layer in (self.ligand_mapping_layer, self.antigen_mapping_layer):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        for block in self.transformer.h:
            if getattr(block, "cross_attn", None) is not None:
                nn.init.zeros_(block.cross_attn.cross_attn.out_proj.weight)
                if block.cross_attn.cross_attn.out_proj.bias is not None:
                    nn.init.zeros_(block.cross_attn.cross_attn.out_proj.bias)

        self.apply_lr_multipliers()

    # ------------------------------------------------------------------
    # Freezing
    # ------------------------------------------------------------------
    def _apply_freezing(self, config):
        mode = config.architecture_mode

        if mode == "stage1":
            # Pretraining trains the protein decoder, text encoder, text
            # projection and reconstruction head jointly; nothing is frozen
            # beyond the biological encoders, which never participate here.
            return

        def freeze(*modules):
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False

        def unfreeze(*modules):
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = True

        text_interface = (
            self.text_mapping_layer, self.text_decoder_projection,
            self.text_decoder_blocks, self.text_decoder_ln, self.text_lm_head,
        )

        if mode == "stage2":
            logger.info("Stage 2: decoder trainable; text and biological encoders frozen.")
            freeze(self.text_embeddings, self.text_encoder_layers,
                   self.ligand_encoder, self.antigen_encoder)
            unfreeze(self.transformer, self.lm_head)

        elif mode == "stage2_frozen_decoder":
            logger.info(
                "Stage 2 (frozen decoder): training adapters and the text interface only."
            )
            freeze(self.transformer, self.text_embeddings, self.text_encoder_layers,
                   self.ligand_encoder, self.antigen_encoder)
            unfreeze(*text_interface, self.lm_head)
            for name, param in self.transformer.named_parameters():
                if "cross_attn" in name or "ln_f" in name or "wte" in name:
                    param.requires_grad = True

        elif mode == "stage2_adapters_only":
            logger.info(
                "Stage 2 (adapters only): training cross-attention adapters, "
                "modality scales and type embeddings; everything else frozen."
            )
            for param in self.parameters():
                param.requires_grad = False
            for name, param in self.transformer.named_parameters():
                if "cross_attn" in name:
                    param.requires_grad = True
            unfreeze(self.modality_scale, self.modality_type_embedding)

        elif mode == "stage2_partial":
            n_layers = len(self.transformer.h)
            start, end = config.frozen_layer_start, config.frozen_layer_end
            logger.info(
                "Stage 2 (partial): freezing layers [%d, %d) of %d.", start, end, n_layers
            )
            freeze(self.text_embeddings, self.text_encoder_layers,
                   self.ligand_encoder, self.antigen_encoder, self.transformer)
            frozen_range = range(start, min(end, n_layers))
            for i, block in enumerate(self.transformer.h):
                if i not in frozen_range:
                    unfreeze(block)
            for name, param in self.transformer.named_parameters():
                if "cross_attn" in name:
                    param.requires_grad = True
            unfreeze(self.transformer.wte, self.transformer.ln_f, self.lm_head,
                     *text_interface)

        else:
            raise ValueError(
                f"Unknown architecture_mode {mode!r}. Expected 'stage1' or one of "
                f"{STAGE2_MODES}."
            )

    def set_num_updates(self, num_updates):
        """Track the training step count, as fairseq's Trainer expects."""
        self.num_updates = num_updates

    def apply_lr_multipliers(self, adapter_multiplier=2.0):
        """Tag each parameter with a `.lr_multiplier` used to build the optimizer.

        Cross-attention adapters, modality projections and modality scales are
        trained at `adapter_multiplier` times the base learning rate; every
        other parameter uses the base rate. Call once, after the freezing flags
        for the current architecture_mode are final and before the optimizer is
        constructed.
        """
        for name, param in self.named_parameters():
            is_adapter = (
                "cross_attn" in name
                or "modality_scale" in name
                or "modality_type_embedding" in name
            )
            param.lr_multiplier = adapter_multiplier if is_adapter else 1.0

    def _build_segment_rope_position_ids(self, text_len, protein_len, device):
        """Rotary positions that separate the text prefix from the protein.

        Protein tokens keep their natural positions 0..protein_len-1. Text
        tokens are placed at negative positions ending just before the protein,
        separated from it by `text_protein_rope_gap` and spread by
        `text_rope_scale`, so that the relative rotary distance between a
        protein token and the instruction does not grow with the instruction's
        length. With gap=0 and scale=1 this reduces to contiguous positions.
        """
        gap = self.config.text_protein_rope_gap
        scale = self.config.text_rope_scale
        protein_pos = torch.arange(protein_len, device=device, dtype=torch.float32)
        if text_len > 0:
            distance_from_boundary = text_len - torch.arange(
                text_len, device=device, dtype=torch.float32
            )
            text_pos = -(gap + scale * distance_from_boundary)
            return torch.cat([text_pos, protein_pos], dim=0).unsqueeze(0)
        return protein_pos.unsqueeze(0)

    def _uses_segment_rope(self):
        return (
            getattr(self.config, "use_segment_rope", False)
            and self.config.architecture_mode in STAGE2_MODES
        )

    def max_positions(self):
        return self.config.n_positions

    def log_adapter_gates(self, step, filepath="logs/adapter_gates.csv"):
        """Append the current tanh(alpha) gate value of each adapter to a CSV."""
        gate_values = [
            f"{torch.tanh(block.cross_attn.alpha).item():.6f}"
            for block in self.transformer.h
            if getattr(block, "cross_attn", None) is not None
        ]
        if not gate_values:
            return
        mode = "a" if os.path.exists(filepath) else "w"
        with open(filepath, mode) as f:
            if mode == "w":
                header = "Step," + ",".join(
                    f"Adapter_{i + 1}" for i in range(len(gate_values))
                )
                f.write(header + "\n")
            f.write(f"{step}," + ",".join(gate_values) + "\n")

    def resize_progen_embeddings(self, new_vocab_size):
        """Widen the token embedding matrix and LM head to `new_vocab_size`."""
        old_vocab_size = self.config.vocab_size
        if new_vocab_size <= old_vocab_size:
            return
        logger.info(
            "Resizing embeddings from %d to %d tokens.", old_vocab_size, new_vocab_size
        )

        old_embeddings = self.transformer.wte.weight.data
        self.transformer.wte = nn.Embedding(new_vocab_size, self.config.n_embd)
        self.transformer.wte.weight.data[:old_vocab_size] = old_embeddings
        self.transformer.wte.weight.data[old_vocab_size:].normal_(
            mean=0.0, std=self.config.initializer_range
        )

        old_head_weight = self.lm_head.weight.data
        old_head_bias = self.lm_head.bias.data if self.lm_head.bias is not None else None
        # Keep the bias term: the head is constructed with one in __init__, and
        # silently dropping it here would change the state-dict keys.
        self.lm_head = nn.Linear(
            self.config.n_embd, new_vocab_size, bias=old_head_bias is not None
        )
        self.lm_head.weight.data[:old_vocab_size] = old_head_weight
        self.lm_head.weight.data[old_vocab_size:].normal_(
            mean=0.0, std=self.config.initializer_range
        )
        if old_head_bias is not None:
            self.lm_head.bias.data[:old_vocab_size] = old_head_bias
            self.lm_head.bias.data[old_vocab_size:].zero_()

        self.config.vocab_size = new_vocab_size

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------
    def _get_causal_mask(self, seq_len, device):
        mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
        return ((1.0 - mask) * -1e9)[None, None, :, :]

    def _build_hybrid_mask(self, text_mask, ligand_mask, antigen_mask, protein_mask,
                           dtype, device):
        """Block-causal mask over [text | ligand | antigen | protein].

        Text attends causally within itself; each modality block attends to
        everything up to and including itself; protein tokens attend to all
        preceding blocks and causally within themselves.
        """
        batch_size = text_mask.size(0)
        L_T = text_mask.size(1) if text_mask is not None else 0
        L_L = ligand_mask.size(1) if ligand_mask is not None else 0
        L_A = antigen_mask.size(1) if antigen_mask is not None else 0
        L_P = protein_mask.size(1) if protein_mask is not None else 0
        L = L_T + L_L + L_A + L_P

        # The block structure is identical for every row, so build it once and
        # expand; only the padding mask varies per row.
        mask = torch.full((L, L), -1e9, device=device, dtype=dtype)
        if L_T > 0:
            mask[:L_T, :L_T] = self._get_causal_mask(L_T, device).squeeze()
        if L_L > 0:
            end = L_T + L_L
            mask[L_T:end, :end] = 0.0
        if L_A > 0:
            start, end = L_T + L_L, L_T + L_L + L_A
            mask[start:end, :end] = 0.0
        if L_P > 0:
            start = L_T + L_L + L_A
            mask[start:, :start] = 0.0
            mask[start:, start:] = self._get_causal_mask(L_P, device).squeeze()

        mask = mask.unsqueeze(0).expand(batch_size, -1, -1).clone()

        pad_masks = [
            m for m in (text_mask, ligand_mask, antigen_mask, protein_mask)
            if m is not None
        ]
        full_pad_mask = torch.cat(pad_masks, dim=1)
        mask = mask.masked_fill(
            full_pad_mask.unsqueeze(1).expand(-1, L, -1) == 0, -1e9
        )
        return mask.unsqueeze(1)

    def forward_text_encoder(self, text_input_ids, attention_mask=None):
        batch_size, seq_len = text_input_ids.size()
        hidden_states = self.text_embeddings(input_ids=text_input_ids)

        causal_mask = self._get_causal_mask(seq_len, text_input_ids.device)
        if attention_mask is not None:
            extended = (1.0 - attention_mask[:, None, None, :]) * -1e9
            final_mask = causal_mask + extended
        else:
            final_mask = causal_mask
        final_mask = final_mask.to(dtype=hidden_states.dtype)

        for layer_module in self.text_encoder_layers:
            hidden_states = layer_module(hidden_states, final_mask)[0]
        return hidden_states

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--pretrained-model-type", type=str, metavar="progen2-large-BFD90"
        )

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _encode_modality(self, tokens, pad_id, encoder, mapping_layer, scale_key,
                         type_index):
        """Encode one auxiliary modality and apply Eq. 3: s_m * W_c phi(c) + e_m."""
        mask = (tokens != pad_id).long()
        # Unmask the first position so a fully-padded row cannot produce NaN in
        # the encoder's own attention; the output is re-masked immediately after.
        safe_mask = mask.clone()
        safe_mask[:, 0] = 1

        raw = encoder(tokens, attention_mask=safe_mask).last_hidden_state
        emb = mapping_layer(raw) * mask.unsqueeze(-1)
        emb = emb * F.softplus(self.modality_scale[scale_key])

        type_ids = torch.full(
            emb.shape[:2], type_index, dtype=torch.long, device=emb.device
        )
        return emb + self.modality_type_embedding(type_ids), mask

    def forward(self, texts=None, seqs=None, ligands=None, antigens=None,
                input_ids=None, past_key_values=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, use_cache=False,
                output_attentions=None, output_hidden_states=None, return_dict=None):
        # Stage 1 checkpoints have no trained modality projections or adapters,
        # so auxiliary modalities are ignored even if present in the batch.
        if self.config.architecture_mode == "stage1":
            ligands, antigens = None, None

        text_pad_id = 0       # PubMedBERT [PAD]
        ligand_pad_id = 0     # ChemBERTa <pad>
        antigen_pad_id = 1    # ESM-2 <pad>
        protein_pad_id = 2    # Alphabet's <|pad|> index (see progen_vocab.py)

        if past_key_values is None:
            text_mask = (texts != text_pad_id).long() if texts is not None else None
            text_progen_embeds = (
                self.text_mapping_layer(
                    self.forward_text_encoder(texts, attention_mask=text_mask)
                )
                if texts is not None else None
            )

            modality_embeds, modality_masks = [], []
            if ligands is not None:
                emb, _ = self._encode_modality(
                    ligands, ligand_pad_id, self.ligand_encoder,
                    self.ligand_mapping_layer, "ligand", MODALITY_LIGAND,
                )
                modality_embeds.append(emb)
                modality_masks.append((ligands != ligand_pad_id).long())
            if antigens is not None:
                emb, _ = self._encode_modality(
                    antigens, antigen_pad_id, self.antigen_encoder,
                    self.antigen_mapping_layer, "antigen", MODALITY_ANTIGEN,
                )
                modality_embeds.append(emb)
                modality_masks.append((antigens != antigen_pad_id).long())

            protein_mask = (seqs != protein_pad_id).long()
            seq_embeddings = self.transformer.wte(seqs)

            cross_attn_context, cross_attn_pad_mask = None, None

            base_embeds = (
                [text_progen_embeds] if texts is not None else []
            ) + [seq_embeddings]
            combined_inputs = torch.cat(base_embeds, dim=1)
            combined_mask = self._build_hybrid_mask(
                text_mask, None, None, protein_mask,
                dtype=combined_inputs.dtype, device=combined_inputs.device,
            )

            if modality_embeds:
                cross_attn_context = torch.cat(modality_embeds, dim=1)
                cross_attn_pad_mask = torch.cat(modality_masks, dim=1) == 0

            text_len = text_progen_embeds.size(1) if texts is not None else 0

            if self._uses_segment_rope():
                position_ids = self._build_segment_rope_position_ids(
                    text_len, seq_embeddings.size(1), device=combined_inputs.device
                )
        else:
            combined_inputs = self.transformer.wte(seqs)
            combined_mask = attention_mask
            cross_attn_context, cross_attn_pad_mask = None, None
            text_len = 0
            text_mask = None

            if self._uses_segment_rope():
                # Incremental decoding: continue the protein's own position
                # numbering, skipping the cached text prefix.
                cached_text_len = texts.size(1) if texts is not None else 0
                protein_so_far = past_key_values[0][0].size(-2) - cached_text_len
                position_ids = torch.arange(
                    protein_so_far,
                    protein_so_far + seqs.size(1),
                    device=seqs.device,
                    dtype=torch.float32,
                ).unsqueeze(0)

        transformer_outputs = self.transformer(
            input_ids=None,
            past_key_values=past_key_values,
            attention_mask=combined_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=combined_inputs,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
            cross_attn_context=cross_attn_context,
            cross_attn_pad_mask=cross_attn_pad_mask,
        )

        hidden_states = transformer_outputs[0]

        idx = 1
        new_past_key_values = None
        if use_cache:
            new_past_key_values = transformer_outputs[idx]
            idx += 1
        if output_hidden_states:
            idx += 1
        self_attns = cross_attns = None
        if output_attentions:
            self_attns = transformer_outputs[idx]
            idx += 1
            if len(transformer_outputs) > idx:
                cross_attns = transformer_outputs[idx]

        if past_key_values is None:
            protein_hidden_out = hidden_states[:, text_len:, :]

            if texts is not None:
                # Auxiliary objective: reconstruct the conditioning description
                # from the decoder's hidden states at the text positions.
                curr = self.text_decoder_projection(hidden_states[:, :text_len, :])
                causal_mask = self._get_causal_mask(curr.size(1), curr.device)
                extended = (1.0 - text_mask[:, None, None, :]) * -1e9
                final_mask = (causal_mask + extended).to(dtype=curr.dtype)
                for layer in self.text_decoder_blocks:
                    curr = layer(curr, final_mask)[0]
                text_logits = self.text_lm_head(self.text_decoder_ln(curr))
            else:
                text_logits = None

            protein_logits = self.lm_head(protein_hidden_out)
        else:
            text_logits = None
            protein_logits = self.lm_head(hidden_states)

        if output_attentions:
            return (
                text_logits, protein_logits,
                new_past_key_values if use_cache else None,
                self_attns, cross_attns,
            )
        if use_cache:
            return text_logits, protein_logits, new_past_key_values
        return text_logits, protein_logits

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def forward_inference(self, texts, bos_id, eos_id, max_length,
                          ligands=None, antigens=None, prefix_tokens=None,
                          temperature=1.0, top_p=0.0, top_k=0,
                          output_attentions=None, **unused):
        device = texts.device
        batch_size = texts.size(0)

        if self.config.architecture_mode == "stage1":
            ligands, antigens = None, None

        if prefix_tokens is not None:
            seqs = prefix_tokens.to(device)
            max_length = max_length - seqs.size(1)
        else:
            seqs = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)

        unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)
        current_past_key_values = None
        current_attention_mask = None
        next_token = seqs
        outputs = None

        for _ in range(max_length):
            step_mask = (
                current_attention_mask[:, :, -1:, :]
                if current_attention_mask is not None else None
            )

            outputs = self.forward(
                texts=texts, ligands=ligands, antigens=antigens,
                seqs=next_token if current_past_key_values is not None else seqs,
                past_key_values=current_past_key_values,
                attention_mask=step_mask,
                use_cache=True,
                output_attentions=output_attentions,
            )

            protein_logits = outputs[1]
            current_past_key_values = outputs[2]
            next_token_logits = protein_logits[:, -1, :]

            if temperature != 1.0 and temperature > 0.0:
                next_token_logits = next_token_logits / temperature

            if top_k > 0 or top_p > 0.0:
                if top_k > 0:
                    kth = torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[next_token_logits < kth] = float("-inf")
                if top_p > 0.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    to_remove = cumulative > top_p
                    to_remove[..., 1:] = to_remove[..., :-1].clone()
                    to_remove[..., 0] = 0
                    next_token_logits[
                        to_remove.scatter(1, sorted_indices, to_remove)
                    ] = float("-inf")
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            next_token = (
                next_token * unfinished.unsqueeze(1)
                + eos_id * (~unfinished).unsqueeze(1)
            )
            seqs = torch.cat((seqs, next_token), dim=1)

            if current_attention_mask is None:
                t_mask = (texts != 0).long() if texts is not None else None
                p_mask = torch.ones(
                    (batch_size, seqs.size(1) - 1), dtype=torch.long, device=device
                )
                current_attention_mask = self._build_hybrid_mask(
                    t_mask, None, None, p_mask,
                    dtype=protein_logits.dtype, device=device,
                )

            # Grow the mask by one row and column for the token just emitted.
            B, _, L, _ = current_attention_mask.size()
            new_col = torch.full(
                (B, 1, L, 1), -1e9, device=device, dtype=current_attention_mask.dtype
            )
            current_attention_mask = torch.cat([current_attention_mask, new_col], dim=3)
            last_row = current_attention_mask[:, :, -1:, :-1]
            self_attend = torch.zeros(
                (B, 1, 1, 1), device=device, dtype=current_attention_mask.dtype
            )
            current_attention_mask = torch.cat(
                [current_attention_mask, torch.cat([last_row, self_attend], dim=3)],
                dim=2,
            )

            unfinished = unfinished & (next_token.squeeze(-1) != eos_id)

            # Degeneracy guard: stop a sequence that has emitted the same token
            # eight times in a row, which otherwise runs to max_length.
            if seqs.size(1) >= 8:
                last_8 = seqs[:, -8:]
                repeating = last_8.max(dim=1).values == last_8.min(dim=1).values
                unfinished = unfinished & ~repeating

            if unfinished.max() == 0:
                break

        if output_attentions and outputs is not None:
            return seqs, outputs[3], outputs[4]
        return seqs


@register_model_architecture(
    "language_guided_protein_design_model", "language_guided_protein_design_model"
)
def base_architecture(args):
    transformer_base_architecture(args)


@register_model_architecture(
    "language_guided_protein_design_model", "language_guided_protein_design_model_3B"
)
def language_guided_protein_design_model_3b(args):
    args.architecture_mode = getattr(args, "architecture_mode", "stage1")
    args.adapter_frequency = getattr(args, "adapter_frequency", 4)
    args.frozen_layer_start = getattr(args, "frozen_layer_start", 0)
    args.frozen_layer_end = getattr(args, "frozen_layer_end", 20)
    args.use_segment_rope = getattr(args, "use_segment_rope", False)
    args.text_protein_rope_gap = getattr(args, "text_protein_rope_gap", 0.0)
    args.text_rope_scale = getattr(args, "text_rope_scale", 1.0)
    base_architecture(args)
