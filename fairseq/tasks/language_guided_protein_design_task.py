# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from omegaconf import II

logger = logging.getLogger(__name__)


def load_protein_dataset(
    data_path,
    split,
    aa_dict,
    dataset_impl,
    left_pad,
    shuffle=True,
    pad_to_multiple=1,
):
    raw = data_utils.load_indexed_dataset(data_path, aa_dict, dataset_impl, split=split)
    logger.info("Loaded %d examples from %s [%s]", len(raw), data_path, split)
    return InstructPro2Dataset(
        raw,
        raw.sizes,
        aa_dict,
        left_pad=left_pad,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
        text_pad=0,
        ligand_pad=0,
        antigen_pad=1,
    )


@dataclass
class LanguageGuidedProteinDesignConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={"help": "path to the data directory for this split"},
    )
    pretrained_model_path: str = field(
        default="",
        metadata={"help": "directory containing the pretrained ProGen2 checkpoint"},
    )
    pretrained_model: str = field(
        default="progen2-large-BFD90",
        metadata={"help": "name of the pretrained ProGen2 checkpoint to initialize from"},
    )
    architecture_mode: str = field(
        default="stage1",
        metadata={
            "help": "stage1 (text-conditioned pretraining, no modality inputs) | "
                    "stage2 (full decoder trainable, encoders frozen) | "
                    "stage2_frozen_decoder | stage2_partial | stage2_adapters_only"
        },
    )
    adapter_frequency: int = field(
        default=4,
        metadata={"help": "insert a gated cross-attention adapter every N decoder blocks"},
    )
    frozen_layer_start: int = field(
        default=0,
        metadata={"help": "first frozen layer index, inclusive (stage2_partial only)"},
    )
    frozen_layer_end: int = field(
        default=20,
        metadata={"help": "last frozen layer index, exclusive (stage2_partial only)"},
    )
    use_segment_rope: bool = field(
        default=False,
        metadata={"help": "offset text rotary positions so they sit before the protein"},
    )
    text_protein_rope_gap: float = field(
        default=0.0,
        metadata={"help": "rotary gap inserted between the text prefix and the protein"},
    )
    text_rope_scale: float = field(
        default=1.0,
        metadata={"help": "multiplier on text-to-text rotary distance"},
    )
    left_pad: bool = field(
        default=False, metadata={"help": "pad sequences on the left"}
    )
    max_source_positions: int = field(
        default=1024, metadata={"help": "maximum number of tokens in a sequence"}
    )
    train_subset: str = II("dataset.train_subset")
    dataset_impl: str = field(
        default="instructpro2",
        metadata={"help": "InstructPro2 | InstructPro2_ligand | InstructPro2_antibody"},
    )
    generation: bool = field(
        default=False,
        metadata={"help": "generate sequences during validation for evaluation"},
    )
    max_generation_length: int = field(
        default=1024, metadata={"help": "maximum generated sequence length"}
    )
    temperature: float = field(
        default=1.0, metadata={"help": "sampling temperature (1.0 disables scaling)"}
    )
    top_p: float = field(
        default=0.0, metadata={"help": "nucleus sampling threshold; 0.0 uses greedy decoding"}
    )
    top_k: int = field(default=0, metadata={"help": "top-k sampling threshold"})


@register_task("language_guided_protein_design",
               dataclass=LanguageGuidedProteinDesignConfig)
class LanguageGuidedProteinDesignTask(FairseqTask):
    """Natural-language-guided protein design."""

    cfg: LanguageGuidedProteinDesignConfig

    def __init__(self, cfg: LanguageGuidedProteinDesignConfig, dictionary):
        super().__init__(cfg)
        self.aa_dict = dictionary

    @classmethod
    def setup_task(cls, cfg: LanguageGuidedProteinDesignConfig, **kwargs):
        return cls(cfg, Alphabet())

    def build_model(self, cfg):
        model = ProGenForCausalLM.from_pretrained(
            os.path.join(cfg.pretrained_model_path, cfg.pretrained_model),
            architecture_mode=getattr(cfg, "architecture_mode", "stage1"),
            adapter_frequency=getattr(cfg, "adapter_frequency", 4),
            frozen_layer_start=getattr(cfg, "frozen_layer_start", 0),
            frozen_layer_end=getattr(cfg, "frozen_layer_end", 20),
            use_segment_rope=getattr(cfg, "use_segment_rope", False),
            text_protein_rope_gap=getattr(cfg, "text_protein_rope_gap", 0.0),
            text_rope_scale=getattr(cfg, "text_rope_scale", 1.0),
        )
        # Widen the embedding matrix and LM head to cover the infilling tokens
        # (<SEP>, <M1>..<M30>) that the alphabet adds on top of ProGen2's vocab.
        model.resize_progen_embeddings(len(self.aa_dict))
        return model

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        self.datasets[split] = load_protein_dataset(
            self.cfg.data,
            split,
            self.aa_dict,
            dataset_impl=self.cfg.dataset_impl,
            left_pad=self.cfg.left_pad,
            shuffle=(split != "test"),
        )

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

        if not self.cfg.generation:
            return loss, sample_size, logging_output

        # NOTE: this returns a 5-tuple rather than fairseq's usual 3-tuple.
        # See the README: evaluation requires the matching trainer patch.
        with torch.no_grad():
            seqs = sample["seqs"]
            sep_token_id = self.aa_dict.get_idx("<SEP>")

            # Infilling (Stage 1 spans or Stage 2 HCDRs): <SEP> marks the end
            # of the given corrupted sequence, so everything up to and
            # including it is the generation prefix. Full-sequence generation
            # has no <SEP>, so decoding starts from BOS.
            sep_idx = (seqs[0] == sep_token_id).nonzero(as_tuple=True)[0]
            prefix_tokens = seqs[:, : sep_idx[0].item() + 1] if len(sep_idx) > 0 else None

            indexes = model.forward_inference(
                prefix_tokens=prefix_tokens,
                texts=sample["texts"],
                ligands=sample.get("ligands", None),
                antigens=sample.get("antigens", None),
                bos_id=self.aa_dict.bos(),
                eos_id=self.aa_dict.eos(),
                max_length=self.cfg.max_generation_length,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                top_k=self.cfg.top_k,
            )

            srcs = [self.aa_dict.string(seqs[i]) for i in range(seqs.size(0))]
            strings = [self.aa_dict.string(indexes[i]) for i in range(len(indexes))]
            return loss, sample["ntokens"], logging_output, strings, srcs

    def max_positions(self):
        return (self.cfg.max_source_positions, self.cfg.max_source_positions)

    @property
    def source_dictionary(self):
        return self.aa_dict

    @property
    def target_dictionary(self):
        return self.aa_dict
