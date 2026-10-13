# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Raw dataset readers for InstructPro2.

Three readers, one per training regime, all returning the same 4-tuple
``(text, seq, ligand, antigen)``:

    InstructPro2RawDataset      Stage 1 pretraining (mixed / no modality)
    InstructPro2LigandDataset   Stage 2 ligand-binding protein design
    InstructPro2AntibodyDataset Stage 2 antibody HCDR generation

Expected on-disk layout, per data directory::

    <data_dir>/train.jsonl
    <data_dir>/valid.jsonl
    <data_dir>/test.jsonl              (or test_seen.jsonl, test_unseen.jsonl, ...)
    <data_dir>/train_packed.pt         (optional, ligand only -- fast path)

Each JSONL line carries at minimum ``tokenized_instruction`` (a list of
PubMedBERT token ids) and ``target_protein`` (an amino-acid string, which may
contain <SEP> and <M1>..<M30> for infilling targets). ``ligand`` (a list of
SMILES strings) and ``antigen`` (a list of chain sequences) are optional.
"""

import json
import os
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from fairseq.dataclass.constants import DATASET_IMPL_CHOICES
from fairseq.file_io import PathManager
from . import FairseqDataset


LIGAND_ENCODER = "seyonec/SMILES_tokenized_PubChem_shard00_160k"
ANTIGEN_ENCODER = "facebook/esm2_t12_35M_UR50D"

# Truncation limits. Antigens are capped at ESM-2's positional limit; ligands
# at ChemBERTa's. Multi-chain antigens are joined with the ESM-2 EOS token and
# multiple ligands with "." (the SMILES multi-component separator).
MAX_ANTIGEN_TOKENS = 1024
MAX_LIGAND_TOKENS = 512


def get_available_dataset_impl():
    return list(map(str, DATASET_IMPL_CHOICES))


def make_dataset(path, impl, fix_lua_indexing=False, dictionary=None, sizes=None,
                 split="train"):
    if impl == "instructpro2_antibody" and InstructPro2AntibodyDataset.exists(path):
        return InstructPro2AntibodyDataset(path, dictionary, split=split)
    if impl == "instructpro2_ligand" and InstructPro2LigandDataset.exists(path):
        return InstructPro2LigandDataset(path, dictionary, split=split)
    if impl == "instructpro2" and InstructPro2RawDataset.exists(path):
        return InstructPro2RawDataset(path, dictionary, split=split)
    return None


def _split_filename(split):
    """Map a fairseq split name to a JSONL filename.

    ``valid`` is fairseq's name for the validation split; everything else maps
    straight through, which lets extra test splits (``test_unseen``, ...) be
    selected by name from the command line.
    """
    return f"{split}.jsonl"


def _encode_protein(aa_dict, target_sequence, reverse_order=False):
    return aa_dict.encode_line(
        target_sequence,
        add_if_not_exist=False,
        prepend_bos=True,
        append_eos=True,
        reverse_order=reverse_order,
    ).long()


def _encode_antigen(tokenizer, antigen_list):
    if not antigen_list:
        return None
    sep = getattr(tokenizer, "eos_token", "<eos>")
    joined = sep.join(antigen_list)
    ids = tokenizer(joined, truncation=True, max_length=MAX_ANTIGEN_TOKENS)["input_ids"]
    return torch.IntTensor(ids).long()


def _encode_ligand(tokenizer, ligand_list):
    if not ligand_list:
        return None
    joined = ".".join(ligand_list)
    ids = tokenizer(joined, truncation=True, max_length=MAX_LIGAND_TOKENS)["input_ids"]
    return torch.IntTensor(ids).long()


class _BaseRawDataset(FairseqDataset):
    """Shared plumbing: length bookkeeping and the fairseq dataset interface."""

    def __init__(self):
        self.sizes = np.array([], dtype=np.int64)
        self._len = 0

    def check_index(self, i):
        if i < 0 or i >= self._len:
            raise IndexError("index out of range")

    def __len__(self):
        return self._len

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)


class InstructPro2AntibodyDataset(_BaseRawDataset):
    """Stage 2 antibody dataset.

    Each row pairs a natural-language design blueprint with a heavy-chain
    target sequence (containing <SEP> and <M1>..<M3> for the three HCDR spans)
    and, where available, the antigen chains. Rows without a paired antigen
    are supported: ``antigen`` is simply None for those.
    """

    _antigen_tokenizer = None

    def __init__(self, path, dictionary, append_eos=True, reverse_order=False,
                 split="train"):
        super().__init__()
        self.reverse_order = reverse_order
        self.aa_dict = dictionary
        self.texts, self.seqs, self.antigens = [], [], []

        if InstructPro2AntibodyDataset._antigen_tokenizer is None:
            InstructPro2AntibodyDataset._antigen_tokenizer = AutoTokenizer.from_pretrained(
                ANTIGEN_ENCODER
            )
        self.antigen_tokenizer = InstructPro2AntibodyDataset._antigen_tokenizer

        self.read_data(path, split)
        self._len = len(self.seqs)

    def read_data(self, path, split):
        filepath = os.path.join(path, _split_filename(split))
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found at {filepath}")

        t0 = time.time()
        sizes = []
        with open(filepath, "r") as f:
            for line in f:
                data = json.loads(line.strip())

                self.texts.append(
                    torch.IntTensor(data["tokenized_instruction"]).long()
                )

                target = data.get("target_protein", data.get("output", ""))
                tokens = _encode_protein(self.aa_dict, target, self.reverse_order)
                self.seqs.append(tokens)
                sizes.append(len(tokens))

                self.antigens.append(
                    _encode_antigen(self.antigen_tokenizer, data.get("antigen", []))
                )

        self.sizes = np.array(sizes)
        print(
            f"Loaded {len(self.seqs):,} antibody examples from {filepath} "
            f"in {time.time() - t0:.1f}s",
            flush=True,
        )

    def __getitem__(self, i):
        self.check_index(i)
        return (self.texts[i], self.seqs[i], None, self.antigens[i])


class LigandPackedShard:
    """O(1) indexed view over pre-tokenized ligand data.

    The packed format stores every row's tokens end-to-end in flat tensors with
    an offsets array, which avoids re-tokenizing millions of rows at startup.
    """

    def __init__(self, packed: dict):
        self.text_tokens = packed["text_tokens"]
        self.text_offsets = packed["text_offsets"]
        self.seq_tokens = packed["seq_tokens"]
        self.seq_offsets = packed["seq_offsets"]
        self.ligand_tokens = packed["ligand_tokens"]
        self.ligand_offsets = packed["ligand_offsets"]
        self.sizes = packed["sizes"].numpy()

    def __len__(self):
        return len(self.sizes)

    def __getitem__(self, i):
        ts, te = int(self.text_offsets[i]), int(self.text_offsets[i + 1])
        ss, se = int(self.seq_offsets[i]), int(self.seq_offsets[i + 1])
        ls, le = int(self.ligand_offsets[i]), int(self.ligand_offsets[i + 1])
        return (
            self.text_tokens[ts:te].long(),
            self.seq_tokens[ss:se].long(),
            self.ligand_tokens[ls:le].long() if le > ls else None,
            None,  # no antigen in the ligand dataset
        )


class InstructPro2LigandDataset(_BaseRawDataset):
    """Stage 2 ligand-binding protein dataset.

    Prefers a packed binary (``<split>_packed.pt``) when present; otherwise
    falls back to tokenizing the JSONL at startup, which is slow on the full
    training split.
    """

    def __init__(self, path, dictionary, append_eos=True, reverse_order=False,
                 split="train"):
        super().__init__()
        self.reverse_order = reverse_order
        self.aa_dict = dictionary
        self._packed = None
        self.texts, self.seqs, self.ligands = [], [], []
        self.read_data(path, split)
        self._len = (
            len(self._packed) if self._packed is not None else len(self.seqs)
        )

    def read_data(self, path, split):
        packed_path = os.path.join(path, f"{split}_packed.pt")
        if os.path.exists(packed_path):
            t0 = time.time()
            self._packed = LigandPackedShard(torch.load(packed_path, map_location="cpu"))
            self.sizes = self._packed.sizes
            print(
                f"Loaded {len(self._packed):,} packed ligand examples from "
                f"{packed_path} in {time.time() - t0:.1f}s",
                flush=True,
            )
            return

        filepath = os.path.join(path, _split_filename(split))
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found at {filepath}")

        print(
            f"No packed file at {packed_path}; tokenizing {filepath} at startup. "
            f"Run the packing script first for the full training split.",
            flush=True,
        )
        ligand_tokenizer = AutoTokenizer.from_pretrained(LIGAND_ENCODER)

        sizes = []
        with open(filepath, "r") as f:
            for line in f:
                data = json.loads(line.strip())

                self.texts.append(
                    torch.IntTensor(data["tokenized_instruction"]).long()
                )

                target = data.get("target_protein", data.get("output", ""))
                tokens = _encode_protein(self.aa_dict, target, self.reverse_order)
                self.seqs.append(tokens)
                sizes.append(len(tokens))

                self.ligands.append(
                    _encode_ligand(ligand_tokenizer, data.get("ligand", []))
                )

        self.sizes = np.array(sizes)

    def __getitem__(self, i):
        self.check_index(i)
        if self._packed is not None:
            return self._packed[i]
        return (self.texts[i], self.seqs[i], self.ligands[i], None)


class InstructPro2RawDataset(_BaseRawDataset):
    """Stage 1 pretraining dataset.

    Handles the general case where a row may carry a ligand, an antigen, both,
    or neither, so the same reader serves pretraining and any mixed-modality
    split.
    """

    def __init__(self, path, dictionary, append_eos=True, reverse_order=False,
                 split="train"):
        super().__init__()
        self.reverse_order = reverse_order
        self.aa_dict = dictionary
        self.texts, self.seqs, self.ligands, self.antigens = [], [], [], []

        self.ligand_tokenizer = AutoTokenizer.from_pretrained(LIGAND_ENCODER)
        self.antigen_tokenizer = AutoTokenizer.from_pretrained(ANTIGEN_ENCODER)

        self.read_data(path, split)
        self._len = len(self.seqs)

    def read_data(self, path, split):
        filepath = os.path.join(path, _split_filename(split))
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found at {filepath}")

        t0 = time.time()
        sizes = []
        with open(filepath, "r") as f:
            for line in f:
                data = json.loads(line.strip())

                self.texts.append(
                    torch.IntTensor(data["tokenized_instruction"]).long()
                )

                target = data.get("target_protein", data.get("output", ""))
                tokens = _encode_protein(self.aa_dict, target, self.reverse_order)
                self.seqs.append(tokens)
                sizes.append(len(tokens))

                self.ligands.append(
                    _encode_ligand(self.ligand_tokenizer, data.get("ligand", []))
                )
                self.antigens.append(
                    _encode_antigen(self.antigen_tokenizer, data.get("antigen", []))
                )

        self.sizes = np.array(sizes)
        print(
            f"Loaded {len(self.seqs):,} examples from {filepath} "
            f"in {time.time() - t0:.1f}s",
            flush=True,
        )

    def __getitem__(self, i):
        self.check_index(i)
        return (self.texts[i], self.seqs[i], self.ligands[i], self.antigens[i])
