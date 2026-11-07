"""Project paths and study-design constants."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Raw inputs
BILLBOARD_PATH = DATA_RAW / "hot100.csv"
SONGS_PATH = DATA_RAW / "songs.csv"  # 550K Spotify songs (audio, lyrics, genres)
NRC_PATH = DATA_RAW / "NRC-Emotion-Lexicon.zip"
NRC_MEMBER = "NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
UMCSENT_PATH = DATA_RAW / "UMCSENT_consumersentiment.csv"
MICH_INFLATION_PATH = DATA_RAW / "MICH_inflation.csv"

# Cached intermediate outputs
SONG_METADATA_PATH = DATA_PROCESSED / "song_metadata.parquet"
ENGINEERED_METADATA_PATH = DATA_PROCESSED / "song_metadata_engineered.parquet"
SONG_MATCHES_PATH = DATA_PROCESSED / "billboard_song_matches.parquet"
LYRIC_FEATURES_PATH = DATA_PROCESSED / "lyric_features.parquet"
SONG_FEATURES_PATH = DATA_PROCESSED / "billboard_song_features.parquet"
WEEKLY_MUSIC_PATH = DATA_PROCESSED / "weekly_music.parquet"
MONTHLY_COVERAGE_PATH = DATA_PROCESSED / "monthly_coverage.parquet"
MODEL_DATA_PATH = DATA_PROCESSED / "model_data.parquet"

# Study design
ANALYSIS_START = pd.Timestamp("2001-01-01")
ANALYSIS_END = pd.Timestamp("2020-12-01")
TEST_START = pd.Timestamp("2017-01-01")

MIN_LYRIC_WORDS = 20
CSV_CHUNKSIZE = 50_000

# Columns read from the 550K CSV (lyrics are excluded; they are loaded separately)
SONG_METADATA_COLS = [
    "id", "name", "album_name", "artists",
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "duration_ms", "year", "genre", "popularity",
    "total_artist_followers", "avg_artist_popularity", "artist_ids",
    "niche_genres",
]

EMOTIONS = [
    "anger", "anticipation", "disgust", "fear", "joy",
    "negative", "positive", "sadness", "surprise", "trust",
]

# Song-level features aggregated to weekly/monthly chart signals
FEATURES_TO_AGGREGATE = [
    # Lyrics
    "lyric_sentiment", "positive_rate", "negative_rate",
    "anger_rate", "fear_rate", "joy_rate", "sadness_rate",
    # Audio
    "valence", "energy", "danceability", "tempo", "mode",
]


def ensure_dirs():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
