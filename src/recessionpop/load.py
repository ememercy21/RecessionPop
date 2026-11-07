"""Raw-data loaders."""

from zipfile import ZipFile

import pandas as pd

from . import config as cfg


def load_billboard():
    """Weekly Hot 100 chart rows (Date, Song, Artist, Rank, ...)."""
    bb = pd.read_csv(cfg.BILLBOARD_PATH)
    bb["Date"] = pd.to_datetime(bb["Date"])
    bb["Rank"] = pd.to_numeric(bb["Rank"], errors="coerce")
    return bb


def load_song_metadata():
    """550K metadata + audio features, without lyrics. Cached as Parquet."""
    if cfg.SONG_METADATA_PATH.exists():
        return pd.read_parquet(cfg.SONG_METADATA_PATH)

    chunks = pd.read_csv(
        cfg.SONGS_PATH,
        usecols=cfg.SONG_METADATA_COLS,
        chunksize=cfg.CSV_CHUNKSIZE,
    )
    songs = pd.concat(chunks, ignore_index=True)
    cfg.ensure_dirs()
    songs.to_parquet(cfg.SONG_METADATA_PATH, index=False)
    return songs


def load_lyrics(spotify_ids):
    """Stream the 550K CSV and keep lyrics only for the given Spotify IDs."""
    spotify_ids = set(spotify_ids)
    kept = []
    for chunk in pd.read_csv(
        cfg.SONGS_PATH,
        usecols=["id", "lyrics"],
        chunksize=cfg.CSV_CHUNKSIZE,
    ):
        hit = chunk.loc[chunk["id"].isin(spotify_ids)]
        if not hit.empty:
            kept.append(hit)
    return (
        pd.concat(kept, ignore_index=True)
        .drop_duplicates(subset="id")
        .reset_index(drop=True)
    )


def load_nrc():
    """NRC word-level emotion lexicon as a wide word x emotion 0/1 table."""
    with ZipFile(cfg.NRC_PATH) as z, z.open(cfg.NRC_MEMBER) as f:
        long = pd.read_csv(
            f, sep="\t", header=None, names=["word", "emotion", "association"]
        )
    return (
        long.pivot(index="word", columns="emotion", values="association")
        .reset_index()
        .dropna(subset=["word"])
        .reset_index(drop=True)
    )


def _load_fred(path, value_col):
    df = pd.read_csv(path)
    df["month"] = pd.to_datetime(df["observation_date"])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    return df[["month", value_col]]


def load_umcsent():
    """University of Michigan Consumer Sentiment (FRED: UMCSENT), monthly."""
    return _load_fred(cfg.UMCSENT_PATH, "UMCSENT")


def load_mich_inflation():
    """University of Michigan 1-yr inflation expectations (FRED: MICH), monthly."""
    return _load_fred(cfg.MICH_INFLATION_PATH, "MICH")
