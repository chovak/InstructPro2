import logging

import numpy as np
import torch
from fairseq.data import FairseqDataset, data_utils
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)
ligand_tokenizer = AutoTokenizer.from_pretrained("seyonec/SMILES_tokenized_PubChem_shard00_160k")


def collate(
    samples,
    text_pad,
    aa_pad,
    eos_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, pad_idx=1, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad=left_pad,
            move_eos_to_beginning=move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )
        
    def collate_liagnd(key):
        ligands = [s[key] for s in samples]
        ligand_inputs = ligand_tokenizer(ligands, padding=True, truncation=True, return_tensors='pt')
        return ligand_inputs

    id = torch.LongTensor([s["id"] for s in samples])
    seqs = merge(
        "seq",
        left_pad=left_pad,
        pad_idx=aa_pad  # 1
        )
    # sort by descending source length
    seq_lengths = torch.LongTensor(
        [s["seq"].ne(aa_pad).long().sum() for s in samples]
    )
    seq_lengths, sort_order = seq_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    seqs = seqs.index_select(0, sort_order)
    
    texts = merge(
        "text",
        left_pad=left_pad,
        pad_idx=text_pad  # 0
        )
    texts = texts.index_select(0, sort_order)
    
    ligands = collate_liagnd("ligand")
    ligands = {"input_ids": ligands["input_ids"].index_select(0, sort_order),
               "attention_mask": ligands["attention_mask"].index_select(0, sort_order)}

    ntokens = torch.sum(seq_lengths)

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "texts": texts,
        "ligands": ligands,
        "seqs": seqs,
        "seq_lengths": seq_lengths,
    }
    return batch


class LanguageGuidedProteinDataset(FairseqDataset):
    """
    A language guided protein dataset of torch.utils.data.Datasets.
    """

    def __init__(
        self,
        text_protein,
        text_protein_sizes,
        aa_dict=None,
        left_pad=False,
        shuffle=True,
        constraints=None,
        text_pad=0,
        num_buckets=0,
        pad_to_multiple=1,
    ):
        self.text_protein = text_protein
        self.sizes = np.array(text_protein_sizes)
        self.aa_dict = aa_dict
        self.left_pad = left_pad 
        self.shuffle = shuffle
        self.constraints = constraints
        self.buckets = None
        self.pad_to_multiple = pad_to_multiple
        self.text_pad = text_pad
        self.aa_pad = self.aa_dict.pad()

    def get_batch_shapes(self):
        return self.buckets

    def __getitem__(self, index):
        text_item, ligand_item, seq_item= self.text_protein[index]

        example = {
            "id": index,
            "text": text_item,
            "ligand": ligand_item,
            "seq": seq_item
            }

        return example

    def __len__(self):
        return len(self.text_protein)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        """
        res = collate(
            samples,
            self.text_pad,
            self.aa_pad,
            self.aa_dict.eos(),
            left_pad=self.left_pad,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )

        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.sizes[index]

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        sizes = self.sizes[indices]
        return sizes

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.sizes[index], 0
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            # sort by target length, then source length
            return indices[np.argsort(self.sizes[indices], kind="mergesort")]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind="mergesort")
            ]

    @property
    def supports_prefetch(self):
        return getattr(self.protein_complex, "supports_prefetch", False)

    def prefetch(self, indices):
        self.protein_complex.prefetch(indices)

    def filter_indices_by_size(self, indices, max_sizes):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        return data_utils.filter_paired_dataset_indices_by_size(
            self.sizes, None, indices, max_sizes,
        )
