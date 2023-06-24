import tempfile
from pathlib import Path
import typing as ty

from tqdm import tqdm

from reanalogy.dataset import ReAnalogy
from torch.utils.data import DataLoader


def _test_dataset(
    tmp_path: Path | str, dataset: ty.Literal["reanalogy", "deep", "kb13"]
):

    ds = ReAnalogy(
        tmp_path,
        split="train",
        return_regex=True,
        dataset_name=dataset,
        n_examples=12,
        filter=True,
    )
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=10)
    for i, s in enumerate(tqdm(dl)):

        seqs = s["seq"]

        for j, seq in enumerate(seqs):

            idx = i + j
            regex = ds.dataset[idx]
            examples = ds.decode_examples(seq)[0]
            if ds.return_regex:
                assert ds.decode_regex(seq)[0] == str(regex)
            # NOTE >0 is because not all examples match due to
            # string encoding problems.
            assert ds.score(regex, examples).mean() > 0
            assert ds.validate(regex, examples).mean() > 0


def test_reanalogy(tmp_path: Path | str):
    _test_dataset(tmp_path, "reanalogy")


def test_deep(tmp_path: Path | str):
    _test_dataset(tmp_path, "deep")


def test_kb13(tmp_path: Path | str):
    _test_dataset(tmp_path, "kb13")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_path:
        bad_idxs = test_reanalogy(tmp_path)
