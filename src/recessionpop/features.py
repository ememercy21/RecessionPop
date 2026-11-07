"""Song-level features -> rank-weighted weekly and monthly chart signals.

Every chart row is weighted by 101 - Rank, so #1 counts 100x more than #100.
A weekly feature value is the weighted mean over chart rows where that feature is
present; `<feature>_coverage` is the share of the week's total weight it covers.
"""

import numpy as np
import pandas as pd

from . import config as cfg

LYRIC_COLS = ["lyric_sentiment", "nrc_coverage"] + [f"{e}_rate" for e in cfg.EMOTIONS]
AUDIO_COLS = [
    "valence", "energy", "danceability", "tempo", "mode",
    "genre_engineered", "religious_music",
]


def build_song_features(song_matches, lyric_features, canonical):
    """One row per matched Billboard (Song, Artist) with lyric + audio features."""
    matched = song_matches.loc[song_matches["spotify_id"].notna()]
    return (
        matched.merge(
            lyric_features[["id"] + LYRIC_COLS],
            left_on="spotify_id", right_on="id", how="left", validate="many_to_one",
        )
        .drop(columns="id")
        .merge(
            canonical[["id"] + AUDIO_COLS],
            left_on="spotify_id", right_on="id", how="left", validate="many_to_one",
        )
        .drop(columns="id")
    )


def attach_to_chart(billboard, song_features):
    weekly = billboard.merge(
        song_features, on=["Song", "Artist"], how="left", validate="many_to_one"
    )
    weekly["raw_weight"] = 101 - weekly["Rank"]
    return weekly


def chart_coverage(billboard, song_matches, freq="M"):
    """Raw-row and rank-weighted share of chart rows matched to the 550K source."""
    df = billboard.merge(
        song_matches[["Song", "Artist", "spotify_id"]],
        on=["Song", "Artist"], how="left", validate="many_to_one",
    )
    df["raw_weight"] = 101 - df["Rank"]
    df["matched"] = df["spotify_id"].notna()
    df["matched_weight"] = df["raw_weight"] * df["matched"]
    period = df["Date"].dt.to_period(freq).dt.to_timestamp()
    out = df.groupby(period).agg(
        chart_rows=("Song", "size"),
        matched_rows=("matched", "sum"),
        total_rank_weight=("raw_weight", "sum"),
        matched_rank_weight=("matched_weight", "sum"),
    )
    out.index.name = "month" if freq == "M" else "period"
    out["row_match_rate"] = out["matched_rows"] / out["chart_rows"]
    out["rank_weighted_match_rate"] = out["matched_rank_weight"] / out["total_rank_weight"]
    return out


def weekly_signal(weekly_chart, features=cfg.FEATURES_TO_AGGREGATE):
    """Rank-weighted mean of each feature per chart week, plus its weight coverage."""
    g = weekly_chart["Date"]
    w = weekly_chart["raw_weight"]
    total = w.groupby(g).sum()

    out = {}
    for feat in features:
        x = weekly_chart[feat].astype(float)
        wv = w.where(x.notna())
        covered = wv.groupby(g).sum()
        num = (x * wv).groupby(g).sum()
        out[feat] = (num / covered).where(covered > 0)
        out[f"{feat}_coverage"] = (covered / total).where(total > 0)

    return pd.DataFrame(out).rename_axis("Date").reset_index()


def monthly_signal(weekly, start=cfg.ANALYSIS_START, end=cfg.ANALYSIS_END):
    monthly = (
        weekly.set_index("Date").resample("MS").mean()
        .rename_axis("month").reset_index()
    )
    return monthly.loc[monthly["month"].between(start, end)].reset_index(drop=True)


def genre_shares(weekly_chart, start=cfg.ANALYSIS_START, end=cfg.ANALYSIS_END):
    """Monthly rank-weighted share of matched chart weight by engineered genre."""
    df = weekly_chart.loc[weekly_chart["genre_engineered"].notna()].copy()
    df["month"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    shares = (
        df.pivot_table(index="month", columns="genre_engineered",
                       values="raw_weight", aggfunc="sum", fill_value=0)
    )
    shares = shares.div(shares.sum(axis=1), axis=0)
    shares = shares.loc[(shares.index >= start) & (shares.index <= end)]
    return shares.rename_axis(columns=None)
