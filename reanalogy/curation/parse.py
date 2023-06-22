import re
from reanalogy.curation.crawl import SEARCH_TERM, load_df


def parse(root_path):

    # make_regex_dataset(data_path)
    df = load_df(root_path)

    def parse_fragment(fragments: str):
        if fragments is None:
            return None
        regxs = [
            i
            for fragment in eval(fragments)
            for i in re.split("\(|".join(SEARCH_TERM), fragment)
        ]
        characters = ["'", '"', '\\"', "\\'"]
        prefixes = ["r@", "@"]
        ptrns = []
        for regx in regxs:
            for prefix in prefixes:
                for ch in characters:
                    _prefix = prefix.replace("@", ch)
                    # some strings are not regexes
                    if regx.startswith(_prefix):
                        _postfix = "([^\\\\]@\)|[^\\\\]@,|[^\\\\]@ |[^\\\\]@.)".replace(
                            "@", ch
                        )
                        _ms = re.split(_postfix, regx)
                        _m = (
                            _ms[0] + _ms[1][0]
                            if len(_ms) > 1 and _ms[1][0] != ch
                            else ""
                        )
                        cr = _m[len(_prefix) :]
                        compiled = False
                        for x, y in [
                            ("", ""),
                            ("\\\\", "\\"),
                            ("\\", "\\\\"),
                            (")", "\\)"),
                            ("(", "\\()"),
                        ]:

                            try:
                                re.compile(cr.replace(x, y))
                                ptrns.append(cr)
                                compiled = True
                                break
                            except:
                                pass
                        if not compiled:
                            print(
                                f"-----------------\nerror: \n{regx} \n---------------\n {cr}"
                            )
        return ptrns

    regexes = df.fragments.apply(parse_fragment)
    clean_regexes = [
        j for i in regexes.dropna().values.tolist() for j in i if len(j) > 0
    ]

    return clean_regexes
