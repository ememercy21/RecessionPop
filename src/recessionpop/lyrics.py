"""NRC lexicon scoring of song lyrics."""

import re
from collections import Counter

import numpy as np
import pandas as pd

from . import config as cfg

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
RATE_COLS = [f"{e}_rate" for e in cfg.EMOTIONS]


def _score_one(text, lookup):
    tokens = TOKEN_RE.findall(str(text).lower())
    total = len(tokens)
    counts = dict.fromkeys(cfg.EMOTIONS, 0)
    matched = 0
    for word, n in Counter(tokens).items():
        assoc = lookup.get(word)
        if assoc is None:
            continue
        matched += n
        for emotion in cfg.EMOTIONS:
            counts[emotion] += assoc[emotion] * n

    row = {
        "nrc_token_count": total,
        "nrc_matched_tokens": matched,
        "nrc_coverage": matched / total if total else np.nan,
    }
    for emotion in cfg.EMOTIONS:
        row[f"{emotion}_rate"] = counts[emotion] / total if total else np.nan
    return row


def score_lyrics(lyrics, nrc):
    """Per-song emotion rates (share of tokens tagged with each NRC emotion).

    Songs with fewer than MIN_LYRIC_WORDS words, or no NRC-matched tokens, get NaN
    rates. lyric_sentiment = positive_rate - negative_rate.
    """
    df = lyrics.copy().reset_index(drop=True)
    df["word_count"] = df["lyrics"].str.split().str.len()
    df["lyric_valid"] = df["word_count"] >= cfg.MIN_LYRIC_WORDS

    lookup = nrc.set_index("word")[cfg.EMOTIONS].to_dict(orient="index")
    empty = {
        "nrc_token_count": 0,
        "nrc_matched_tokens": 0,
        "nrc_coverage": np.nan,
        **dict.fromkeys(RATE_COLS, np.nan),
    }
    scores = pd.DataFrame(
        [
            _score_one(text, lookup) if valid else empty
            for text, valid in zip(df["lyrics"], df["lyric_valid"])
        ]
    )
    df = pd.concat([df, scores], axis=1)

    df.loc[df["nrc_matched_tokens"] == 0, RATE_COLS] = np.nan
    df["lyric_sentiment"] = df["positive_rate"] - df["negative_rate"]
    return df.drop(columns="lyrics")
