"""Deterministic Billboard -> 550K Spotify matching.

Matching runs once on unique (Song, Artist) pairs and is merged back to weekly rows.

Canonical lookup: the 550K source repeats recordings across albums/compilations,
so each normalised (title, primary artist) key is reduced to one row, preferring
(1) the most common rounded audio signature, (2) the highest popularity, (3) id.

Passes:
1. exact normalised title + exact normalised artist;
2. exact title + Billboard lead artist after dropping a "Featuring ..." suffix;
3. Spotify embeds the feature credit in the title: strip only that credit, require
   a Billboard featuring credit and overlap with a Spotify secondary artist, accept
   unique candidates plus 12 manually audited ambiguous cases.
"""

import pandas as pd

from .text import (
    normalize_text,
    parse_list,
    remove_featuring,
    remove_title_feature_credit,
)

KEY = ["title_key", "artist_key"]

LOOKUP_COLS = [
    "id", "name", "album_name", "artists", "primary_artist",
    "title_key", "artist_key", "year", "popularity",
    "valence", "energy", "danceability", "tempo", "mode", "duration_ms",
    "genre_original", "genre_engineered", "religious_music",
]

# Twelve of fourteen two-candidate Pass-3 cases were resolvable from collaborator
# credits / duplicate-recording metadata. Two "Or" cases stay unmatched on purpose.
PASS3_MANUAL_IDS = {
    "53Y0kdCa1CZ9gRqEuknfwy",  # Bang My Head
    "6yuvC80FcnVJNvC0DbXN9e",  # Come To Me
    "1MZtr7IH5qtjIkqrXj8WOJ",  # Essence
    "2npw9XuCQpHXM9ramYL4HR",  # Ima Boss
    "50DMJJpAeQv4fIpxZvQz2e",  # Monsters
    "31dgrKXIJ7FCxlENs8uxuo",  # Show Me Love
    "2eSJflipjhSKLExuSwuFrO",  # What's Luv?
    "1fLibtVj3MrNTwRZq5yUOs",  # Let's Go
    "7kQJCw0ZkvHgfJqRwPblmG",  # Neck & Wrist
    "5qqBsCBwzv0g0xb1kB1TGJ",  # Off The Books
    "66MsAs4nVj68t6va1lygIm",  # Shut Up
    "7x9nXsowok1JszkVztI5NI",  # Writing On The Wall
}


def build_canonical_lookup(songs):
    """One canonical 550K row per normalised (title, primary artist)."""
    songs = songs.copy()
    songs["primary_artist"] = songs["artists"].apply(parse_list).str[0]
    songs["title_key"] = songs["name"].map(normalize_text)
    songs["artist_key"] = songs["primary_artist"].map(normalize_text)

    lk = songs.loc[
        songs["title_key"].ne("") & songs["artist_key"].ne(""), LOOKUP_COLS
    ].copy()

    lk["sig_danceability"] = lk["danceability"].round(3)
    lk["sig_energy"] = lk["energy"].round(3)
    lk["sig_valence"] = lk["valence"].round(3)
    lk["sig_tempo"] = lk["tempo"].round(1)
    lk["sig_duration"] = lk["duration_ms"].div(1000).round()
    sig = ["sig_danceability", "sig_energy", "sig_valence", "sig_tempo", "sig_duration"]

    lk["candidate_count"] = lk.groupby(KEY, dropna=False)["id"].transform("size")
    lk["signature_count"] = lk.groupby(KEY + sig, dropna=False)["id"].transform("size")
    lk["signature_share"] = lk["signature_count"] / lk["candidate_count"]
    lk["popularity_sort"] = lk["popularity"].fillna(-1)

    canonical = (
        lk.sort_values(
            KEY + ["signature_count", "popularity_sort", "id"],
            ascending=[True, True, False, False, True],
        )
        .drop_duplicates(subset=KEY, keep="first")
        .drop(columns="popularity_sort")
        .reset_index(drop=True)
    )
    canonical["title_no_feature_key"] = (
        canonical["name"].map(remove_title_feature_credit).map(normalize_text)
    )
    assert not canonical.duplicated(KEY).any()
    assert not canonical["id"].duplicated().any()
    return canonical


def unique_billboard_songs(billboard):
    pairs = billboard[["Song", "Artist"]].drop_duplicates().reset_index(drop=True)
    pairs["title_key"] = pairs["Song"].map(normalize_text)
    pairs["artist_key"] = pairs["Artist"].map(normalize_text)
    return pairs


def _as_map(df, method):
    out = df[["Song", "Artist", "id"]].rename(columns={"id": "spotify_id"}).copy()
    out["match_method"] = method
    return out


def _pass1(bb_songs, canonical):
    merged = bb_songs.merge(
        canonical, on=KEY, how="left", indicator="_m", validate="many_to_one"
    )
    matched = _as_map(merged.loc[merged["_m"] == "both"], "exact_title_artist")
    remaining = merged.loc[merged["_m"] == "left_only", ["Song", "Artist"] + KEY].copy()
    return matched, remaining


def _pass2(remaining, canonical):
    remaining = remaining.copy()
    remaining["lead_artist_key"] = remaining["artist_key"].map(remove_featuring)
    has_feat = remaining["artist_key"].str.contains(r"\bfeaturing\b", regex=True, na=False)
    cand = remaining.loc[has_feat].merge(
        canonical,
        left_on=["title_key", "lead_artist_key"],
        right_on=KEY,
        how="left",
        validate="many_to_one",
        suffixes=("_billboard", "_spotify"),
    )
    return _as_map(
        cand.loc[cand["id"].notna()], "exact_title_lead_artist_after_featuring"
    )


def _pass3(bb_songs, pass12, canonical):
    remaining = (
        bb_songs.merge(
            pass12[["Song", "Artist", "spotify_id"]],
            on=["Song", "Artist"],
            how="left",
            validate="one_to_one",
        )
        .loc[lambda d: d["spotify_id"].isna(), ["Song", "Artist"] + KEY]
        .copy()
    )
    remaining["lead_artist_key"] = remaining["artist_key"].map(remove_featuring)

    cand = remaining.merge(
        canonical,
        left_on=["title_key", "lead_artist_key"],
        right_on=["title_no_feature_key", "artist_key"],
        how="inner",
        suffixes=("_billboard", "_spotify"),
    )
    secondary = cand["artists"].apply(parse_list).apply(
        lambda artists: [normalize_text(a) for a in artists][1:]
    )
    bb_artist = cand["artist_key_billboard"]
    overlap = [
        any(a and a in bb for a in sec) for sec, bb in zip(secondary, bb_artist)
    ]
    has_feat = bb_artist.str.contains(r"\bfeaturing\b", regex=True, na=False)

    strong = cand.loc[has_feat & pd.Series(overlap, index=cand.index)].copy()
    strong["candidate_rows"] = strong.groupby(["Song", "Artist"])["id"].transform("size")

    unique = _as_map(
        strong.loc[strong["candidate_rows"] == 1],
        "exact_title_after_feature_strip_lead_artist_secondary_overlap",
    )
    manual = _as_map(
        strong.loc[strong["id"].isin(PASS3_MANUAL_IDS)].drop_duplicates(
            ["Song", "Artist", "id"]
        ),
        "manual_ambiguous_feature_credit_review",
    ).reset_index(drop=True)
    assert len(manual) == len(PASS3_MANUAL_IDS), len(manual)
    return pd.concat([unique, manual], ignore_index=True)


def match_billboard_to_spotify(billboard, canonical):
    """Return every unique Billboard (Song, Artist) with spotify_id / match_method."""
    bb_songs = unique_billboard_songs(billboard)

    p1, remaining = _pass1(bb_songs, canonical)
    p2 = _pass2(remaining, canonical)
    pass12 = pd.concat([p1, p2], ignore_index=True)
    p3 = _pass3(bb_songs, pass12, canonical)

    match_map = pd.concat([p1, p2, p3], ignore_index=True)
    assert not match_map.duplicated(["Song", "Artist"]).any()

    return bb_songs[["Song", "Artist"]].merge(
        match_map, on=["Song", "Artist"], how="left", validate="one_to_one"
    )
