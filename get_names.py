import re
from collections import Counter
from typing import Dict, Iterable, Iterator

import pandas as pd


class WordTokenizer:
    def __init__(self):
        #digits included
        self.p = re.compile(r'\b\w+\b')
        #digits excluded
        self.b = re.compile(r'\b[^\d\W]+\b')

    def __call__(self, description: str) -> Iterator[str]:
        #upper for distinct column names
        return self.b.findall(description.upper())


def count(tokens: Iterable[str]) -> Dict[str, int]:
    return Counter(tokens)


tokenizer = WordTokenizer()
def count_tokens(description: str) -> Counter:
    return (count(tokenizer(description)))

def tokenize_except(feature: pd.Series, target: pd.Series) -> pd.DataFrame:
    return pd.concat([target,pd.DataFrame(feature.apply(count_tokens).to_list())], axis = 1)

