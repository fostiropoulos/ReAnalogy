from pathlib import Path
import requests
from urllib.parse import quote
import json
import re
from reanalogy import data_path

import pandas as pd

import joblib
import time

DF_FILE_NAME = "crawled_regex.pk"
API_TOKEN = "xxxxx"


mem = joblib.Memory(Path.home().joinpath(".cache"), verbose=0)


SEARCH_TERM = [
    "re.compile",
    "re.match",
    "re.fullmatch",
    "re.split",
    "re.search",
    "re.findall",
    "re.finditer",
    "re.sub",
    "re.subn",
    "re.escape",
    "re.purge",
]


def init_df():
    search_params = []

    for q in SEARCH_TERM:
        for i in range(1, 30):
            search_params.append({"q": quote(q), "page": i, "fragments": None})
    df = pd.DataFrame(search_params)

    return df


def load_df(root_path: Path):
    p = root_path.joinpath(DF_FILE_NAME)
    if p.exists():
        return pd.read_pickle(p)
    else:
        df = init_df()
        save_df(root_path, df)
        return df


def save_df(root_path: Path, df: pd.DataFrame):
    df.to_pickle(root_path.joinpath(DF_FILE_NAME))


def parse_fragments(q, fragments):
    patterns = []
    if fragments is None:
        return None
    for delimeter in ['"', "'", '"""']:
        for f in re.findall(
            f"{q}\(r*{delimeter}[^{delimeter}]*{delimeter} *[,\)a-zA-Z\"']", fragments
        ):
            # regex_dataset.append(f)
            try:
                raw_regex = re.sub(f"{q}\(r*{delimeter}", "", f[: -len(delimeter) - 1])
                re.compile(raw_regex)
                patterns.append(raw_regex)
            except Exception as e:
                # print(f"error: {f} -> {raw_regex}")
                print(e)
    if len(patterns):

        return patterns
    else:
        return None



def make_regex_dataset(root_path: Path):

    headers = {
        "Accept": "application/vnd.github.text-match+json",
        "Authorization": f"Bearer {API_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    df = None
    while df is None or df.fragments.apply(lambda x: x is None).sum() > 0:
        df = load_df(root_path)
        regex_dataset = []
        for row in df.itertuples():
            if row.fragments is not None:
                continue
            params = {"q": row.q, "page": row.page}
            response = requests.get(
                "https://api.github.com/search/code", params=params, headers=headers
            )
            if response.status_code == 422:
                df.loc[row.Index, "fragments"] = str(response.text)
                continue
            elif response.status_code == 403:
                print(response.text)
                limit_reset = time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(int(response.headers["X-RateLimit-Reset"])),
                )
                seconds = int(response.headers["X-RateLimit-Reset"]) - time.time() + 5
                print(f"Sleeping for {seconds} until: {limit_reset}")
                time.sleep(seconds if seconds > 0 else 60)
                continue
            if response.status_code != 200:
                currently_done = df.fragments.apply(lambda x: x is not None).mean()
                print(f"progress: {currently_done:.2f}")
                time.sleep(60)
                continue
            fragments = [
                match["fragment"]
                for item in json.loads(response.text)["items"]
                for match in item["text_matches"]
            ]
            df.loc[row.Index, "fragments"] = str(fragments)
            save_df(root_path, df)
    return df


if __name__ == "__main__":
    make_regex_dataset(data_path)
