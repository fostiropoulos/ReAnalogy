from pathlib import Path
from reanalogy.dataset import ReAnalogy

from tqdm import tqdm
import tempfile
from torch.utils.data import DataLoader


def test_redata(tmp_path: Path | str):

    ds = ReAnalogy(
        tmp_path,
        split="train",
        return_regex=True,
        dataset_name="reanalogy",
        n_examples=5,
    )
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
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

        pass


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_path:
        bad_idxs = test_redata(tmp_path)
