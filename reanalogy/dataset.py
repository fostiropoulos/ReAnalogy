# TODO https://github.com/axiak/pyre2/
import re
import typing as ty
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from reanalogy.utils import Xeger
from filelock import FileLock

SPECIAL_TOKENS = ["BOS_TOKEN", "EOS_TOKEN", "PAD_TOKEN", "SEP_TOKEN", "UKN_TOKEN"]
SPECIAL_TOKENS_MAP = dict(zip(SPECIAL_TOKENS, range(len(SPECIAL_TOKENS))))
BOS_TOKEN = SPECIAL_TOKENS_MAP["BOS_TOKEN"]
EOS_TOKEN = SPECIAL_TOKENS_MAP["EOS_TOKEN"]
PAD_TOKEN = SPECIAL_TOKENS_MAP["PAD_TOKEN"]
SEP_TOKEN = SPECIAL_TOKENS_MAP["SEP_TOKEN"]
UKN_TOKEN = SPECIAL_TOKENS_MAP["UKN_TOKEN"]
MAX_SUB_SEQ_LEN = 64


def gen_regex(
    regex: str,
    num_samples=2,
    max_attempts=50,
    check_match=False,
):
    outputs = []
    for i in range(max_attempts):
        try:

            # TODO for some reason Xeger().xeger("\b") returns ->''
            random_match = Xeger().xeger(regex)
            if (
                (random_match in outputs)
                or len(random_match) > MAX_SUB_SEQ_LEN
                or len(random_match) < 2
            ):
                continue
            if check_match and re.match(regex, random_match) is None:
                raise RuntimeError
            outputs.append(random_match)
            if len(outputs) == num_samples:
                return outputs
        except Exception as e:
            pass

    return []


def str_encode(sub_seq: str) -> list[int]:
    return [ord(c) for c in sub_seq]


def bytes_encode(bytes_seq: list[int]) -> str:
    return "".join([chr(c) for c in bytes_seq])


class ReAnalogy(Dataset):
    def __init__(
        self,
        root_path: Path | str,
        split: ty.Literal["train", "val"] = "train",
        return_regex: bool = True,
        dataset_name: ty.Literal["reanalogy", "deep", "kb13"] = "reanalogy",
        n_examples: int = 5,
        filter: bool = False,
        check_match: bool = False,
    ) -> None:
        super().__init__()
        assert dataset_name in {"reanalogy", "deep", "kb13"}
        self.root_path = Path(root_path)
        assert self.root_path.exists(), f"{root_path} does not exist."
        from reanalogy.curation.filter import (
            load_deep_regex,
            load_kb13,
            load_reanalogy,
            read_bad_regex,
            load_vocab,
            BAD_REGEX,
        )

        if dataset_name == "reanalogy":
            dataset = load_reanalogy(self.root_path)
        elif dataset_name == "deep":
            dataset = load_deep_regex(self.root_path)
        elif dataset_name == "kb13":
            dataset = load_kb13(self.root_path)
        with FileLock(self.root_path.joinpath(f"{BAD_REGEX}.lock").as_posix()):
            self.bad_regex = read_bad_regex(self.root_path)
        np_dataset = np.array(list(set(dataset).difference(self.bad_regex)))
        self.return_regex = return_regex
        self.n_examples = n_examples
        self.max_seq_len = int((MAX_SUB_SEQ_LEN + 1) * (self.n_examples + 1) + 2 - 1)
        idxs = np.random.RandomState(seed=0).permutation(len(np_dataset))
        if split == "train":
            train_idxs = idxs[: int(len(idxs) * 0.8)]
            self.dataset = np_dataset[train_idxs]
        elif split == "val":
            eval_idxs = idxs[int(len(idxs) * 0.8) :]
            self.dataset = np_dataset[eval_idxs]
        else:
            raise NotImplementedError
        self.vocab_map, self.inv_vocab_map = load_vocab(self.root_path)
        self.vocab_size = len(self.vocab_map) + len(SPECIAL_TOKENS)
        self.filter = filter
        self.check_match = check_match

    def __len__(self):
        return len(self.dataset)

    def decode_regex(self, seq: torch.Tensor | np.ndarray):
        assert (
            self.return_regex
        ), f"Dataset was initialize with return_regex=`{self.return_regex}`"
        if len(seq.shape) == 1:
            seq = seq[None, :]
        out = [self._decode(_seq)[-1] for _seq in seq]
        if len(out) == 1:
            return out[0]
        else:
            return out

    def decode_examples(self, seq: torch.Tensor | np.ndarray):
        if len(seq.shape) == 1:
            seq = seq[None, :]
        if self.return_regex:
            out = [self._decode(_seq)[:-1] for _seq in seq]
        out = [self._decode(_seq) for _seq in seq]
        if len(out) == 1:
            return out[0]
        else:
            return out

    def score(self, regex: str | bytes, examples: np.ndarray):

        scores = []
        if isinstance(regex, bytes):
            regex = regex.decode("utf-8", errors="ignore")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            for example in examples:
                try:
                    search_match = re.search(regex, example)
                    if search_match is None:
                        scores.append(0)
                        continue
                    partial_match = search_match.endpos - search_match.pos
                    scores.append(1 - (len(example) - partial_match) / len(example))
                except Exception as e:
                    scores.append(0)

        return np.array(scores)

    def validate(self, regex: str, examples: np.ndarray):

        scores = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            for example in examples:
                try:
                    search_match = re.match(regex, example)
                    if search_match is None:
                        scores.append(0)
                        continue
                    scores.append(1)
                except Exception as e:
                    print(e)
                    scores.append(0)

        return np.array(scores)

    def _decode(self, seq: torch.Tensor | np.ndarray) -> list[str]:
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        assert len(seq.shape) == 1, "Only supports vector sequences"
        assert seq[0] == BOS_TOKEN
        text_seq: list[str] = []
        sub_seq: list[int] = []
        for token in seq[1:]:
            if token in {EOS_TOKEN, PAD_TOKEN}:
                text_seq.append(bytes_encode(sub_seq))
                break
            elif token == SEP_TOKEN:
                text_seq.append(bytes_encode(sub_seq))
                sub_seq = []
                continue
            elif token == UKN_TOKEN or token not in self.inv_vocab_map:
                sub_seq.append(ord("\u2400"))  # ␀ symbol
                continue
            sub_seq.append(self.inv_vocab_map[token])

        return text_seq

    def _concat_examples(self, examples: list[str]):
        seq = []
        for e in examples:
            seq += self._to_ord(e) + [SEP_TOKEN]
        return seq

    def _encode(self, regex: str, examples: list[str]) -> torch.Tensor:
        seq: list[int] = [BOS_TOKEN]
        seq += self._concat_examples(examples)
        if self.return_regex:
            seq += self._to_ord(regex)
        else:
            seq = seq[:-1]
        seq += [EOS_TOKEN]
        assert len(seq) < self.max_seq_len

        return torch.tensor(seq).long()

    def _pad_to_len(self, seq):
        assert len(seq) < self.max_seq_len
        x = torch.nn.functional.pad(
            seq,
            pad=(0, self.max_seq_len - len(seq)),
            value=PAD_TOKEN,
        )
        return x

    def _to_ord(self, str_seq: str) -> list[int]:
        return list(
            map(
                lambda x: self.vocab_map[x] if x in self.vocab_map else UKN_TOKEN,
                str_encode(str_seq),
            )
        )

    def __getitem__(self, index):
        regex = index
        regex = self.dataset[index]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            # replace regex by an example
            n_examples = self.n_examples if self.return_regex else self.n_examples + 1

            examples = gen_regex(
                regex,
                num_samples=n_examples,
                check_match=self.filter or self.check_match,
            )
            if self.filter:
                from reanalogy.curation.filter import BAD_REGEX, _append_bad_regex

                if len(examples) != n_examples:
                    with FileLock(
                        self.root_path.joinpath(f"{BAD_REGEX}.lock").as_posix()
                    ):
                        _append_bad_regex(self.root_path, regex)

            seq = self._encode(regex, examples)
            seq = self._pad_to_len(seq)
            return_dict = {"seq": seq, "regex": regex}

            return return_dict
