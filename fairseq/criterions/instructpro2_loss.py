# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class InstructPro2Config(FairseqDataclass):
    design_factor: float = field(
        default=1.0,
        metadata={"help": "weight on the protein generation loss"},
    )
    text_loss_factor: float = field(
        default=0.5,
        metadata={
            "help": "weight on the auxiliary text reconstruction loss. The "
                    "paper uses 0.5 so that text reconstruction stays an "
                    "auxiliary signal and protein generation remains primary."
        },
    )


@register_criterion("instructpro2_loss", dataclass=InstructPro2Config)
class InstructPro2Loss(FairseqCriterion):
    """L = design_factor * L_protein + text_loss_factor * L_text."""

    def __init__(self, cfg: InstructPro2Config, task):
        super().__init__(task)
        self.design_factor = cfg.design_factor
        self.text_loss_factor = cfg.text_loss_factor

        self.pad_idx = task.target_dictionary.pad()
        self.sep_idx = task.target_dictionary.index("<SEP>")

        # PubMedBERT's pad_token_id is 0, matching the model's text_pad_id.
        # Cached here so it cannot drift from the model's own value.
        self.text_pad_idx = 0
        self.loss_text_fct = nn.CrossEntropyLoss(ignore_index=self.text_pad_idx)

    def forward(self, model, sample, reduce=True):
        texts = sample["texts"]
        seqs = sample["seqs"]
        ligands = sample.get("ligands", None)
        antigens = sample.get("antigens", None)

        sample_size = sample["ntokens"]
        batch_size = seqs.size(0)

        text_logits, protein_logits = model(
            texts=texts, seqs=seqs, ligands=ligands, antigens=antigens,
        )

        # --- Text reconstruction loss ---
        shift_text_logits = text_logits[..., :-1, :].contiguous()
        shift_text_labels = texts[..., 1:].contiguous()
        loss_text = self.loss_text_fct(
            shift_text_logits.view(-1, shift_text_logits.size(-1)),
            shift_text_labels.view(-1),
        )

        # --- Protein generation loss ---
        output_seqs = seqs[:, 1:]
        protein_logits_trunc = protein_logits[:, :-1, :]
        log_probs_all = F.log_softmax(protein_logits_trunc, dim=-1)

        loss_mask = (output_seqs != self.pad_idx).float()

        # Infilling loss mask: when the target contains <SEP>, everything up to
        # and including <SEP> is the given corrupted sequence, not a prediction
        # target. Applies to both Stage 1 span infilling and Stage 2 HCDR
        # generation, which share the same span-infilling format.
        if self.sep_idx != self.task.target_dictionary.unk():
            is_sep = output_seqs == self.sep_idx
            if is_sep.any():
                # Zero out every position at or before the first <SEP> in each
                # row. Rows without a <SEP> are left untouched.
                before_or_at_sep = is_sep.cumsum(dim=1) == 0
                has_sep = is_sep.any(dim=1, keepdim=True)
                prefix = (before_or_at_sep | is_sep) & has_sep
                loss_mask = loss_mask.masked_fill(prefix, 0.0)

        log_probs = -log_probs_all.gather(
            dim=-1, index=output_seqs.unsqueeze(-1)
        ).squeeze(-1)
        loss_protein = (log_probs * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)

        loss = (loss_protein * self.design_factor) + (loss_text * self.text_loss_factor)

        logging_output = {
            "loss": loss.data,
            "loss_text": loss_text.data,
            "loss_protein": loss_protein.data,
            "ntokens": sample_size,
            "nsentences": batch_size,
        }

        base_model = getattr(model, "module", model)

        # Diagnostics: learned per-modality scale and per-layer cross-attention
        # gate values. alpha_raw is the raw learnable scalar (initialized to 0);
        # alpha_gate is tanh(alpha_raw), the multiplier actually applied.
        if hasattr(base_model, "modality_scale"):
            for name in base_model.modality_scale.keys():
                logging_output[f"scale_{name}"] = F.softplus(
                    base_model.modality_scale[name]
                ).item()

        if hasattr(base_model, "transformer"):
            for i, block in enumerate(base_model.transformer.h):
                cross_attn = getattr(block, "cross_attn", None)
                if cross_attn is not None and cross_attn.alpha is not None:
                    raw_val = cross_attn.alpha.item()
                    logging_output[f"alpha_raw_layer{i}"] = raw_val
                    logging_output[f"alpha_gate_layer{i}"] = math_tanh(raw_val)

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        sample_size = sum(log.get("ntokens", 0) for log in logging_outputs)
        if sample_size == 0:
            return

        for key in ("loss", "loss_text", "loss_protein"):
            total = sum(log[key].cpu() * log["ntokens"] for log in logging_outputs)
            metrics.log_scalar(key, total / sample_size, round=3)
        metrics.log_scalar("sample_size", sample_size)

        if not logging_outputs:
            return

        first = logging_outputs[0]
        for key in first:
            if key.startswith("scale_"):
                metrics.log_scalar(key, first[key], round=4)
            elif key.startswith("alpha_raw_layer") or key.startswith("alpha_gate_layer"):
                metrics.log_scalar(key, first[key], round=6)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True


def math_tanh(x: float) -> float:
    return float(torch.tanh(torch.tensor(x)))
