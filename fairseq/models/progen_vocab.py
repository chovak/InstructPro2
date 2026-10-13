"""Amino-acid alphabet for InstructPro2.

Wraps the ProGen2 token inventory in a fairseq-Dictionary-compatible object.
Beyond the 20 standard amino acids plus ambiguity codes, the alphabet carries
the span-infilling tokens used by both training stages:

    <SEP>          separates the corrupted sequence from the recovered spans
    <M1> .. <M30>  sentinel tokens marking individual masked spans

The vocabulary is padded out to 51,200 entries with unused placeholders so the
embedding matrix and LM head keep a fixed, hardware-friendly width across
stages.
"""

import itertools
import re
from typing import List, Sequence, Tuple

import torch


# ProGen2 base token -> id. Offset by +2 below to make room for the two
# prepended null tokens, which keeps index 0 available as <unk>.
token2dict = {
    'F': 10, 'C': 7, 'N': 17, 'Q': 20, 'Z': 29, 'P': 19, 'Y': 28, '<|eos|>': 2,
    'B': 6, 'D': 8, 'O': 18, 'R': 21, 'V': 25, 'G': 11, 'X': 27, 'H': 12,
    'L': 15, 'K': 14, '2': 4, 'A': 5, 'S': 22, 'I': 13, 'M': 16, 'T': 23,
    'E': 9, 'W': 26, 'U': 24, '1': 3, '<|bos|>': 1, '<|pad|>': 0,
}

VOCAB_SIZE = 51200
NUM_SENTINELS = 30


class Alphabet(object):
    def __init__(
        self,
        prepend_toks: Sequence[str] = ("<null_0>", "<null_2>"),
        prepend_bos: bool = True,
        append_eos: bool = True,
    ):
        self.prepend_toks = list(prepend_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos

        self.all_toks = list(self.prepend_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}
        for token in token2dict:
            self.tok_to_idx[token] = token2dict[token] + 2

        self.idx_to_tok = {idx: tok for tok, idx in self.tok_to_idx.items()}

        for idx in range(2, len(self.idx_to_tok)):
            self.all_toks.append(self.idx_to_tok[idx])

        # Span-infilling tokens. Shared by Stage 1 span corruption and Stage 2
        # HCDR generation, so they must exist in both stages' dictionaries.
        special_tags = ["<SEP>"] + [f"<M{i}>" for i in range(1, NUM_SENTINELS + 1)]
        for tag in special_tags:
            self.add_symbol(tag)
        while len(self.all_toks) < VOCAB_SIZE:
            self.add_symbol(f"<unused_{len(self.all_toks)}>")

        self.unk_idx = self.tok_to_idx["<null_0>"]
        self.padding_idx = self.get_idx("<|pad|>")
        self.cls_idx = self.get_idx("<|bos|>")
        self.eos_idx = self.get_idx("<|eos|>")
        self.all_special_tokens = [
            "<null_0>", "<null_2>", "<|pad|>", "<|bos|>", "<|eos|>",
        ]
        self.unique_no_split_tokens = self.all_special_tokens + special_tags

    def __len__(self):
        return len(self.all_toks)

    def add_symbol(self, word):
        """Append a new symbol to the alphabet, returning its index."""
        if word not in self.tok_to_idx:
            idx = len(self.all_toks)
            self.all_toks.append(word)
            self.tok_to_idx[word] = idx
            self.idx_to_tok[idx] = word
            return idx
        return self.tok_to_idx[word]

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def index(self, tok):
        """Alias for get_idx, for drop-in compatibility with fairseq criterions."""
        return self.get_idx(tok)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def pad(self):
        return self.padding_idx

    def bos(self):
        return self.cls_idx

    def eos(self):
        return self.eos_idx

    def unk(self):
        return self.unk_idx

    def to_dict(self):
        return self.tok_to_idx.copy()

    def get_batch_converter(self, truncation_seq_length: int = None):
        return BatchConverter(self, truncation_seq_length)

    def _tokenize(self, text) -> List[str]:
        return text.split()

    def tokenize(self, text, **kwargs) -> List[str]:
        """Split `text` into tokens, keeping multi-character special tags intact.

        Adapted from the HuggingFace tokenization utilities.
        """

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.unique_no_split_tokens
                        else [token]
                    )
                    for token in tokenized_text
                )
            )

        return split_on_tokens(self.unique_no_split_tokens, text)

    def encode(self, text):
        return [self.tok_to_idx[tok] for tok in text]

    def encode_line(
        self,
        line,
        add_if_not_exist=True,
        consumer=None,
        prepend_bos=True,
        append_eos=True,
        reverse_order=False,
    ) -> torch.IntTensor:
        if isinstance(line, str):
            # Match any <tag> as one token, and any other single character.
            words = re.findall(r"<[^>]+>|.", line)
        else:
            words = line

        nwords = len(words)
        ids = torch.IntTensor(nwords + int(prepend_bos) + int(append_eos))
        ids.fill_(self.padding_idx)

        if prepend_bos:
            ids[0] = self.cls_idx

        seq_encoded = self.encode(words)
        seq = torch.tensor(seq_encoded, dtype=torch.int64)
        ids[int(prepend_bos): len(seq_encoded) + int(prepend_bos)] = seq

        if append_eos:
            ids[len(seq_encoded) + int(prepend_bos)] = self.eos_idx

        return ids

    def string(
        self,
        tensor,
        bpe_symbol=None,
        escape_unk=False,
        extra_symbols_to_ignore=None,
        unk_string=None,
        include_eos=False,
        separator="",
    ):
        """Convert a tensor of token indices back to an amino-acid string."""
        ignore = set(self.all_special_tokens)
        ignore.update(
            {self.eos(), self.bos(), self.unk(), self.pad(), self.get_idx("<null_1>")}
        )
        if extra_symbols_to_ignore:
            ignore.update(extra_symbols_to_ignore)

        return separator.join(
            self.get_tok(i) for i in tensor if utils.item(i) not in ignore
        )


class BatchConverter(object):
    """Convert a batch of (label, sequence-string) pairs into padded tensors."""

    def __init__(self, alphabet, truncation_seq_length: int = None):
        self.alphabet = alphabet
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        batch_size = len(raw_batch)
        batch_labels, seq_str_list = zip(*raw_batch)
        seq_encoded_list = [self.alphabet.encode(s) for s in seq_str_list]
        if self.truncation_seq_length:
            seq_encoded_list = [
                s[: self.truncation_seq_length] for s in seq_encoded_list
            ]
        max_len = max(len(s) for s in seq_encoded_list)
        tokens = torch.empty(
            (
                batch_size,
                max_len
                + int(self.alphabet.prepend_bos)
                + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels, strs = [], []

        for i, (label, seq_str, seq_encoded) in enumerate(
            zip(batch_labels, seq_str_list, seq_encoded_list)
        ):
            labels.append(label)
            strs.append(seq_str)
            offset = int(self.alphabet.prepend_bos)
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[i, offset: len(seq_encoded) + offset] = seq
            if self.alphabet.append_eos:
                tokens[i, len(seq_encoded) + offset] = self.alphabet.eos_idx

        return labels, strs, tokens
