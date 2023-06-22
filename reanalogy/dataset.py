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
    regex,
    num_samples=2,
    max_attempts=50,
    check_match=False,
):
    outputs = []
    for i in range(max_attempts):
        try:

            # TODO for some reason Xeger().xeger("\b") returns ->''
            random_match = (
                Xeger().xeger(regex).encode("utf-8", errors="ignore")
            )  # .encode("utf-8", errors="ignore")
            if (random_match in outputs) or len(random_match) > MAX_SUB_SEQ_LEN:
                continue
            if check_match and isinstance(regex, str):
                regex = regex.encode("utf-8", errors="ignore")
            if check_match and re.match(regex, random_match) is None:
                raise RuntimeError
            outputs.append(random_match)
            if len(outputs) == num_samples:
                return outputs
        except:
            return None

    return []


class ReAnalogy(Dataset):
    def __init__(
        self,
        root_path: Path | str,
        split: ty.Literal["train", "val"] = "train",
        return_regex: bool = True,
        dataset_name: ty.Literal["reanalogy", "deep", "kb13"] = "reanalogy",
        n_examples: int = 5,
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
            dataset = np.array(load_deep_regex(self.root_path))
        elif dataset_name == "kb13":
            dataset = np.array(load_kb13(self.root_path))
        with FileLock(self.root_path.joinpath(f"{BAD_REGEX}.lock").as_posix()):

            bad_regex = read_bad_regex(self.root_path)
        dataset = np.array(list(set(dataset).difference(bad_regex)))
        self.return_regex = return_regex
        self.n_examples = n_examples
        self.max_seq_len = int((MAX_SUB_SEQ_LEN + 1) * (self.n_examples + 1) + 2 - 1)
        idxs = np.random.RandomState(seed=0).permutation(len(dataset))
        if split == "train":
            train_idxs = idxs[: int(len(idxs) * 0.8)]
            self.dataset = dataset[train_idxs]
        elif split == "val":
            eval_idxs = idxs[int(len(idxs) * 0.8) :]
            self.dataset = dataset[eval_idxs]
        else:
            raise NotImplementedError
        list_dataset: list[str] = self.dataset.tolist()
        self.dataset = np.array(list_dataset)
        self.vocab_map, self.inv_vocab_map = load_vocab(self.root_path)
        self.vocab_size = len(self.vocab_map) + len(SPECIAL_TOKENS)

    def __len__(self):
        return len(self.dataset)

    def _pad_to_len(self, seq):
        assert len(seq) < self.max_seq_len
        x = torch.nn.functional.pad(
            seq,
            pad=(0, self.max_seq_len - len(seq)),
            value=PAD_TOKEN,
        )
        return x

    def _encode(self, str_seq: str):
        return list(
            map(
                lambda x: self.vocab_map[x] if x in self.vocab_map else UKN_TOKEN,
                list(str_seq),
            )
        )

    def decode_regex(self, seq):
        assert (
            self.return_regex
        ), f"Dataset was initialize with return_regex=`{self.return_regex}`"
        if len(seq.shape) == 1:
            seq = seq[None, :]
        return [self._decode(_seq)[-1] for _seq in seq]

    def decode_examples(self, seq):
        if len(seq.shape) == 1:
            seq = seq[None, :]
        if self.return_regex:
            return [self._decode(_seq)[:-1] for _seq in seq]
        return [self._decode(_seq) for _seq in seq]

    def score(self, regex: str | bytes, examples: np.ndarray):

        scores = []
        if isinstance(regex, bytes):
            regex = regex.decode("utf-8", errors="ignore")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            for example in examples:
                try:
                    search_match = re.search(regex, example)
                    if search_match is None:
                        scores.append(0)
                        continue
                    partial_match = search_match.endpos - search_match.pos
                    scores.append(1 - (len(example) - partial_match) / len(example))
                except Exception as e:
                    print(e)
                    scores.append(0)

        return np.array(scores)

    def validate(self, regex: str | bytes, examples: np.ndarray):

        scores = []
        if isinstance(regex, bytes):
            regex = regex.decode("utf-8", errors="ignore")
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

    def _decode(self, seq: torch.Tensor | np.ndarray):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        assert len(seq.shape) == 1, "Only supports vector sequences"
        assert seq[0] == BOS_TOKEN
        text_seq = []
        sub_seq = []
        for token in seq[1:]:
            if token in {EOS_TOKEN, PAD_TOKEN}:
                text_seq.append(bytes(sub_seq).decode("utf-8"))
                break
            elif token == SEP_TOKEN:
                text_seq.append(bytes(sub_seq).decode("utf-8"))
                sub_seq = []
                continue
            elif token == UKN_TOKEN:
                sub_seq.append(max(self.inv_vocab_map.values()) + 1)
                continue
            sub_seq.append(self.inv_vocab_map[token])

        return text_seq

    def _concat_examples(self, examples):
        seq = []
        for e in examples:
            seq += self._encode(e) + [SEP_TOKEN]
        return seq

    def _build_seq(self, regex, examples):
        seq = [BOS_TOKEN]
        seq += self._concat_examples(examples)
        if self.return_regex:
            seq += self._encode(regex)
        else:
            seq = seq[:-1]
        seq += [EOS_TOKEN]
        assert len(seq) < self.max_seq_len

        return torch.tensor(seq).long()

    def __getitem__(self, index):
        regex = index
        regex = self.dataset[index].encode("utf-8", errors="ignore")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            # replace regex by an example
            n_examples = self.n_examples if self.return_regex else self.n_examples + 1
            examples = gen_regex(
                regex,
                num_samples=n_examples,
            )
            assert len(examples) > 0
            seq = self._build_seq(regex, examples)
            seq = self._pad_to_len(seq)
            return_dict = {"seq": seq, "examples": examples}
            if not self.return_regex:
                return_dict["regex"] = regex

            return return_dict
