"""End-to-end data build: raw files -> data/processed/model_data.parquet.

Run from the project root:

    python -m recessionpop.pipeline            # reuse cached stages
    python -m recessionpop.pipeline --force    # rebuild everything
"""

import argparse
import time
from dataclasses import dataclass

import pandas as pd

from . import config as cfg
from .features import (
    attach_to_chart,
    build_song_features,
    chart_coverage,
    monthly_signal,
    weekly_signal,
)
from .genres import engineer_genres
from .load import (
    load_billboard,
    load_lyrics,
    load_nrc,
    load_song_metadata,
    load_umcsent,
)
from .lyrics import score_lyrics
from .matching import build_canonical_lookup, match_billboard_to_spotify
from .modeling import build_model_data


@dataclass
class PipelineOutputs:
    billboard: pd.DataFrame
    canonical: pd.DataFrame
    song_matches: pd.DataFrame
    song_features: pd.DataFrame
    weekly_chart: pd.DataFrame
    weekly_music: pd.DataFrame
    monthly_coverage: pd.DataFrame
    model_data: pd.DataFrame


def _log(msg, t0):
    print(f"[{time.perf_counter() - t0:6.1f}s] {msg}")


def _cached(path, build, force):
    if path.exists() and not force:
        return pd.read_parquet(path)
    df = build()
    df.to_parquet(path, index=False)
    return df


def run(force=False, verbose=True):
    cfg.ensure_dirs()
    t0 = time.perf_counter()
    log = (lambda m: _log(m, t0)) if verbose else (lambda m: None)

    billboard = load_billboard()
    log(f"Billboard: {len(billboard):,} weekly rows, "
        f"{billboard['Date'].min():%Y-%m-%d} to {billboard['Date'].max():%Y-%m-%d}")

    songs = _cached(
        cfg.ENGINEERED_METADATA_PATH,
        lambda: engineer_genres(load_song_metadata()),
        force,
    )
    log(f"550K metadata with engineered genres: {len(songs):,} rows")

    canonical = build_canonical_lookup(songs)
    log(f"Canonical title-artist lookup: {len(canonical):,} rows")

    song_matches = _cached(
        cfg.SONG_MATCHES_PATH,
        lambda: match_billboard_to_spotify(billboard, canonical),
        force,
    )
    log(f"Unique Billboard songs matched: {song_matches['spotify_id'].notna().sum():,} "
        f"/ {len(song_matches):,} ({song_matches['spotify_id'].notna().mean():.1%})")

    lyric_features = _cached(
        cfg.LYRIC_FEATURES_PATH,
        lambda: score_lyrics(load_lyrics(song_matches["spotify_id"].dropna()), load_nrc()),
        force,
    )
    log(f"Lyric features scored: {lyric_features['lyric_sentiment'].notna().sum():,} songs")

    song_features = build_song_features(song_matches, lyric_features, canonical)
    song_features.to_parquet(cfg.SONG_FEATURES_PATH, index=False)

    weekly_chart = attach_to_chart(billboard, song_features)
    weekly_music = weekly_signal(weekly_chart)
    weekly_music.to_parquet(cfg.WEEKLY_MUSIC_PATH, index=False)

    monthly_coverage = chart_coverage(billboard, song_matches)
    monthly_coverage.reset_index().to_parquet(cfg.MONTHLY_COVERAGE_PATH, index=False)

    monthly_music = monthly_signal(weekly_music)
    model_data = build_model_data(monthly_music, load_umcsent())
    model_data.to_parquet(cfg.MODEL_DATA_PATH, index=False)
    log(f"Model data: {len(model_data)} months, "
        f"{model_data['month'].min():%Y-%m} to {model_data['month'].max():%Y-%m}")

    return PipelineOutputs(
        billboard=billboard,
        canonical=canonical,
        song_matches=song_matches,
        song_features=song_features,
        weekly_chart=weekly_chart,
        weekly_music=weekly_music,
        monthly_coverage=monthly_coverage,
        model_data=model_data,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild cached stages")
    args = parser.parse_args()
    run(force=args.force)


if __name__ == "__main__":
    main()
