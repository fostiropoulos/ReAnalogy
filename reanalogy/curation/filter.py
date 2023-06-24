import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock
from reanalogy.curation.parse import parse
from tqdm import tqdm
from urllib.request import urlretrieve

from reanalogy.dataset import MAX_SUB_SEQ_LEN, gen_regex, SPECIAL_TOKENS

from reanalogy import data_path

BAD_REGEX = "bad_regex.pkl"

VOCAB_FILE = "vocab.pkl"


def _make_vocab_map(vocab):
    n_special_tokens = len(SPECIAL_TOKENS)
    vocab = set([ord(c) for c in vocab if len(c) > 0] + [i for i in np.arange(256)])
    return dict(zip(vocab, np.arange(n_special_tokens, len(vocab) + n_special_tokens)))


def _inverse_vocab(vocab):
    return {v: k for k, v in vocab.items()}


def load_lingua_franca(root_path: Path) -> list[str]:

    p = root_path.joinpath("uniq-regexes-8.json")
    if not p.exists():
        # download
        urlretrieve(
            "https://raw.githubusercontent.com/fostiropoulos/ReAnalogy/dev/data/uniq-regexes-8.json",
            p.as_posix(),
        )

    text_file = p.read_text().split("}\n{")[1:-1]
    df = pd.DataFrame([json.loads("{" + s + "}") for s in text_file])
    linguage_franca = df["pattern"].values.tolist()
    # root_path.joinpath("LinguaFranca.txt").write_text(linguage_franca)
    return linguage_franca


def load_deep_regex(root_path: Path) -> list[str]:
    p = root_path.joinpath("deep-regex.txt")
    if not p.exists():
        urlretrieve(
            "https://raw.githubusercontent.com/fostiropoulos/ReAnalogy/dev/data/deep-regex.txt",
            p.as_posix(),
        )

    regex = p.read_text().split("\n")
    return regex


def load_reanalogy(root_path: Path) -> list[str]:
    p = root_path.joinpath(f"reanalogy.pkl")
    if not p.exists():
        urlretrieve(
            "https://github.com/fostiropoulos/ReAnalogy/raw/dev/data/reanalogy.pkl",
            p.as_posix(),
        )

    return [regex.replace("\\u", r"\u") for regex in pickle.load(p.open("rb"))]


def load_vocab(root_path: Path):

    p = root_path.joinpath(VOCAB_FILE)
    if not p.exists():
        urlretrieve(
            f"https://github.com/fostiropoulos/ReAnalogy/raw/dev/data/{VOCAB_FILE}",
            p.as_posix(),
        )

    vocab = pickle.load(p.open("rb"))
    vocab_map: dict[int, int] = _make_vocab_map(vocab)

    inv_vocab_map = _inverse_vocab(vocab_map)
    return vocab_map, inv_vocab_map


def make_reanalogy(root_path: Path) -> list[str]:
    lf = load_lingua_franca(root_path)
    reanalogy = parse(root_path)
    return lf + reanalogy + load_kb13(root_path) + load_deep_regex(root_path)


def load_kb13(root_path: Path) -> list[str]:
    p = root_path.joinpath("KB13.txt")
    if not p.exists():
        urlretrieve(
            f"https://github.com/fostiropoulos/ReAnalogy/raw/dev/data/KB13.txt",
            p.as_posix(),
        )

    regex = p.read_text().split("\n")
    return regex


def read_bad_regex(root_path: Path) -> list[str]:
    p = root_path.joinpath(BAD_REGEX)
    if not p.exists():
        # write_bad_regex(root_path, [])
        urlretrieve(
            f"https://github.com/fostiropoulos/ReAnalogy/raw/dev/data/{BAD_REGEX}",
            p.as_posix(),
        )

    regexes = pickle.load(p.open("rb"))
    return regexes


def write_bad_regex(root_path: Path, regexes: list[str]):
    p = root_path.joinpath(f".{BAD_REGEX}")
    pickle.dump(regexes, p.open("wb"))
    shutil.move(p, root_path.joinpath(BAD_REGEX))


def _append_bad_regex(root_path: Path, regex):
    bad_regex: list[str] = read_bad_regex(root_path)
    if regex not in bad_regex:
        bad_regex.append(regex)
        write_bad_regex(root_path, bad_regex)


def _remove_bad_regex(root_path: Path, regex: str):
    bad_regex = read_bad_regex(root_path)
    if regex in bad_regex:
        bad_regex.remove(regex)
        write_bad_regex(root_path, bad_regex)


def filter_regex(bad_regex_root_path: Path, regex: str, num_samples=12):

    with FileLock(bad_regex_root_path.joinpath(f"{BAD_REGEX}.lock").as_posix()):
        _append_bad_regex(bad_regex_root_path, regex)

        out = gen_regex(regex, num_samples=num_samples, check_match=True)
        if out is not None and len(out) > 0:
            _remove_bad_regex(bad_regex_root_path, regex)
            return regex


def filter_regexes(root_path: Path):
    ds = make_reanalogy(root_path)
    ds = [_ for _ in ds if len(str(_)) < MAX_SUB_SEQ_LEN and len(str(_)) > 5]
    with FileLock(root_path.joinpath(f"{BAD_REGEX}.lock").as_posix()):
        bad_regex = read_bad_regex(root_path)
    valid_regex = []
    set(ds)
    for regex in tqdm(set(ds).difference(bad_regex)):
        try:
            valid_regex.append(filter_regex(root_path, regex))
        except Exception as e:
            # raise e
            pass

    vocab = sorted(np.unique(list("".join(valid_regex))).tolist())
    pickle.dump(vocab, root_path.joinpath("vocab.pkl").open("wb"))
    pickle.dump(valid_regex, root_path.joinpath(f"reanalogy.pkl").open("wb"))


if __name__ == "__main__":
    filter_regexes(data_path)
    pass
