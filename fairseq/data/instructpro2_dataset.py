"""Collating wrapper around the raw InstructPro2 datasets in indexed_dataset.py.

Each underlying dataset yields a 4-tuple (text, seq, ligand, antigen). This
wrapper batches those, pads each stream with its own encoder's pad id, and
sorts by protein length as fairseq expects.
"""

import logging

import numpy as np
import torch

from fairseq.data import FairseqDataset, data_utils

logger = logging.getLogger(__name__)


def collate(
    samples,
    text_pad,
    aa_pad,
    eos_idx,
    ligand_pad=0,
    antigen_pad=1,  # ESM-2 uses 1 for <pad>
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    if len(samples) == 0:
        return {}

    def merge(key, pad_idx):
        # Substitute an empty tensor for missing rows rather than dropping
        # them, so row order stays aligned with `sort_order` below.
        data = [
            s[key] if s[key] is not None else torch.empty(0, dtype=torch.long)
            for s in samples
        ]
        return data_utils.collate_tokens(
            data,
            pad_idx,
            eos_idx,
            left_pad=left_pad,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    id = torch.LongTensor([s["id"] for s in samples])

    seqs = merge("seq", aa_pad)
    seq_lengths = torch.LongTensor(
        [s["seq"].ne(aa_pad).long().sum() for s in samples]
    )

    # fairseq expects batches sorted by descending source length.
    seq_lengths, sort_order = seq_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    seqs = seqs.index_select(0, sort_order)

    texts = merge("text", text_pad).index_select(0, sort_order)
    text_lengths = torch.LongTensor(
        [s["text"].ne(text_pad).long().sum() for s in samples]
    ).index_select(0, sort_order)

    def collate_modality(key, pad_val):
        if not any(s[key] is not None and len(s[key]) > 0 for s in samples):
            return None
        data = [
            s[key] if s[key] is not None else torch.empty(0, dtype=torch.long)
            for s in samples
        ]
        collated = data_utils.collate_tokens(
            data,
            pad_idx=pad_val,
            eos_idx=eos_idx,
            left_pad=left_pad,
            pad_to_multiple=pad_to_multiple,
        )
        return collated.index_select(0, sort_order)

    return {
        "id": id,
        "nsentences": len(samples),
        "ntokens": torch.sum(seq_lengths),
        "ntokens_text": torch.sum(text_lengths),
        "texts": texts,
        "seqs": seqs,
        "seq_lengths": seq_lengths,
        "ligands": collate_modality("ligand", ligand_pad),
        "antigens": collate_modality("antigen", antigen_pad),
    }


class InstructPro2Dataset(FairseqDataset):
    def __init__(
        self,
        text_protein,
        text_protein_sizes,
        aa_dict=None,
        left_pad=False,
        shuffle=True,
        text_pad=0,
        ligand_pad=0,
        antigen_pad=1,
        pad_to_multiple=1,
    ):
        self.text_protein = text_protein
        self.sizes = np.array(text_protein_sizes)
        self.aa_dict = aa_dict
        self.left_pad = left_pad
        self.shuffle = shuffle
        self.pad_to_multiple = pad_to_multiple

        self.text_pad = text_pad
        self.ligand_pad = ligand_pad
        self.antigen_pad = antigen_pad
        self.aa_pad = self.aa_dict.pad()

    def __getitem__(self, index):
        text_item, seq_item, ligand_item, antigen_item = self.text_protein[index]

        def to_tensor(item):
            if item is None:
                return None
            return item if torch.is_tensor(item) else torch.tensor(item, dtype=torch.long)

        return {
            "id": index,
            "text": to_tensor(text_item),
            "seq": to_tensor(seq_item),
            "ligand": to_tensor(ligand_item),
            "antigen": to_tensor(antigen_item),
        }

    def __len__(self):
        return len(self.text_protein)

    def collater(self, samples, pad_to_length=None):
        return collate(
            samples,
            self.text_pad,
            self.aa_pad,
            self.aa_dict.eos(),
            ligand_pad=self.ligand_pad,
            antigen_pad=self.antigen_pad,
            left_pad=self.left_pad,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )

    def num_tokens(self, index):
        return self.sizes[index]

    def num_tokens_vec(self, indices):
        return self.sizes[indices]

    def size(self, index):
        return (self.sizes[index], 0)

    def ordered_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        return indices[np.argsort(self.sizes[indices], kind="mergesort")]

    @property
    def supports_prefetch(self):
        return False
