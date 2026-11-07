"""Genre engineering for the 550K Spotify metadata.

Rules (conservative by design):
- map the source's broad genre into the final taxonomy;
- let only audited, high-confidence niche tags vote for an override;
- keep the broad source genre when niche evidence is absent or tied;
- record religious content as a separate binary flag, not a primary genre.
"""

import pandas as pd

from .text import parse_list

FINAL_GENRES = [
    "Pop", "Indie", "Rock", "Hip-Hop", "R&B/Soul", "Country",
    "Electronic/Dance", "Folk", "Reggae/Dancehall", "Latin/Tropical",
    "Afrobeats/African Pop", "Jazz", "Classical/Musical", "Blues", "Other",
]

BASE_GENRE_MAP = {
    "Pop": "Pop",
    "Rock": "Rock",
    "Hip-Hop": "Hip-Hop",
    "R&B": "R&B/Soul",
    "Country": "Country",
    "Electronic": "Electronic/Dance",
    "Folk": "Folk",
    "Jazz": "Jazz",
    "Classical": "Classical/Musical",
    "Blues": "Blues",
}

GENRE_VOTE_TAGS = {
    "Reggae/Dancehall": {
        "calypso", "christian reggae", "dancehall", "dub", "lovers rock",
        "nz reggae", "ragga", "reggae", "reggae en español", "reggae pop",
        "reggae rock", "rocksteady", "roots reggae", "ska", "soca",
    },
    "Afrobeats/African Pop": {
        "afro adura", "afro r&b", "afro soul", "afrobeat", "afrobeats",
        "afrogospel", "afropiano", "afropop", "afroswing", "alté", "amapiano",
        "asakaa", "azonto", "bongo flava", "ghanaian hip hop", "highlife",
        "hiplife", "nigerian drill", "rumba congolaise",
    },
    "Latin/Tropical": {
        "bachata", "bolero", "bossa nova", "brazilian funk", "chilean mambo",
        "christian reggaeton", "corridos bélicos", "corridos tumbados",
        "cumbia", "cumbia norteña", "cumbia sonidera", "electro corridos",
        "electrocumbia", "forró", "forró tradicional", "funk carioca",
        "latin", "latin afrobeats", "latin alternative", "latin country",
        "latin dance", "latin folk", "latin folklore", "latin hip hop",
        "latin house", "latin indie", "latin jazz", "latin pop", "latin r&b",
        "latin rock", "mambo", "mariachi", "merengue", "mpb",
        "música mexicana", "new mpb", "norteño", "norteño-sax", "pagode",
        "pagode baiano", "ranchera", "reggaeton", "reggaeton chileno",
        "reggaeton mexa", "salsa", "samba", "sertanejo",
        "sertanejo tradicional", "sertanejo universitário", "tango",
        "trap latino", "urbano latino",
        # religious subgenres (religious_music flag is still set separately)
        "brazilian gospel", "pagode gospel", "sertanejo gospel",
    },
    "Indie": {
        "indie", "chinese indie", "german indie", "hindi indie",
        "indian indie", "indie québécois", "indonesian indie",
        "japanese indie", "lo-fi indie", "mexican indie", "pinoy indie",
        "taiwanese indie", "tamil indie", "vietnam indie",
    },
    "R&B/Soul": {
        "alternative r&b", "classic soul", "dark r&b", "funk", "indie r&b",
        "indie soul", "motown", "neo soul", "new jack swing",
        "northern soul", "philly soul", "quiet storm", "r&b", "retro soul",
        "soul", "trap soul",
        "christian r&b", "uk r&b", "funk gospel",
    },
    "Hip-Hop": {
        "alternative hip hop", "boom bap", "cloud rap", "crunk", "drill",
        "east coast hip hop", "emo rap", "experimental hip hop", "g-funk",
        "gangster rap", "hardcore hip hop", "hip hop", "melodic rap",
        "old school hip hop", "rap", "southern hip hop", "trap",
        "underground hip hop", "west coast hip hop",
        "christian hip hop", "grime", "uk grime", "chinese hip hop",
        "k-rap", "miami bass",
    },
    "Electronic/Dance": {
        "deep house", "disco", "drum and bass", "dubstep", "ebm", "edm",
        "electro house", "electronic", "eurodance", "future bass",
        "hardstyle", "house", "melodic bass", "progressive house",
        "synthpop", "tech house", "techno", "trance", "tropical house",
        "alternative dance", "electro swing", "synthwave", "witch house",
    },
    "Country": {
        "alt country", "americana", "bluegrass", "classic country",
        "country", "honky tonk", "outlaw country", "red dirt",
        "texas country", "traditional country",
        "country christian", "christian bluegrass", "southern gospel",
    },
    "Jazz": {
        "bebop", "big band", "cool jazz", "jazz", "jazz funk",
        "jazz fusion", "smooth jazz", "soul jazz", "swing music",
        "vocal jazz",
        "acid jazz", "experimental jazz", "christian jazz",
    },
    "Blues": {
        "blues", "classic blues", "country blues", "delta blues",
        "modern blues",
    },
    "Classical/Musical": {
        "baroque", "chamber music", "choral", "classical",
        "classical crossover", "contemporary classical", "musicals",
        "opera", "orchestral", "renaissance",
        "gregorian chant",
    },
    "Folk": {
        "anti-folk", "celtic folk", "folk", "folk pop", "indie folk",
        "singer-songwriter", "traditional folk",
        "christian folk", "celtic",
    },
    "Rock": {
        "alternative rock", "art rock", "blues rock", "classic rock",
        "death metal", "funk rock", "garage rock", "glam rock", "grunge",
        "hard rock", "hardcore", "indie rock", "industrial metal",
        "industrial rock", "metal", "metalcore", "pop punk", "post-grunge",
        "post-hardcore", "power metal", "progressive rock",
        "psychedelic rock", "psychobilly", "punk", "rock", "rock and roll",
        "ska punk", "skate punk", "soft rock", "yacht rock",
        "christian rock", "christian alternative rock", "southern rock",
        "celtic rock", "j-rock", "krautrock",
    },
    "Pop": {
        "art pop", "baroque pop", "bedroom pop", "dance pop", "electropop",
        "funk pop", "indie pop", "j-pop", "k-pop", "pop", "power pop",
        "soft pop",
        "christian pop", "pop worship", "dream pop", "acoustic pop",
    },
}

# Meaningful but too cross-genre to force a primary classification.
INHERIT_TAGS = {
    "industrial", "darkwave", "new wave", "riddim", "folk rock",
    "country rock", "gospel", "traditional gospel",
}

# Descriptor/context tags that never vote for a primary genre.
PRIMARY_GENRE_IGNORE_TAGS = {
    "christian", "christmas", "worship", "ccm", "hymns", "soundtrack",
    "children's music", "comedy",
}

RELIGIOUS_TAGS = {
    "christian", "christian alternative rock", "christian bluegrass",
    "christian folk", "christian hip hop", "christian jazz",
    "christian lo-fi", "christian pop", "christian r&b", "christian reggae",
    "christian reggaeton", "christian rock", "country christian", "ccm",
    "gospel", "traditional gospel", "southern gospel", "afrogospel",
    "brazilian gospel", "funk gospel", "pagode gospel", "sertanejo gospel",
    "worship", "pop worship", "hymns", "catholic music",
}


def _validate_rules():
    seen = {}
    for category, tags in GENRE_VOTE_TAGS.items():
        for tag in tags:
            seen.setdefault(tag, []).append(category)
    duplicates = {t: c for t, c in seen.items() if len(c) > 1}
    assert not duplicates, f"Tags vote for multiple genres: {duplicates}"

    vote_tags = set().union(*GENRE_VOTE_TAGS.values())
    assert not vote_tags & INHERIT_TAGS
    assert not vote_tags & PRIMARY_GENRE_IGNORE_TAGS
    assert not INHERIT_TAGS & PRIMARY_GENRE_IGNORE_TAGS


_validate_rules()


def _clean_tags(tags):
    return {str(t).lower().strip() for t in tags if pd.notna(t)}


def choose_genre_override(tags):
    """Return the single genre with the most voting tags, or None if absent/tied."""
    tag_set = _clean_tags(tags)
    scores = {cat: len(tag_set & cat_tags) for cat, cat_tags in GENRE_VOTE_TAGS.items()}
    best = max(scores.values())
    if best == 0:
        return None
    winners = [cat for cat, s in scores.items() if s == best]
    return winners[0] if len(winners) == 1 else None


def has_religious_tag(tags):
    return int(bool(_clean_tags(tags) & RELIGIOUS_TAGS))


def engineer_genres(songs):
    """Add genre_base, genre_engineered and religious_music columns."""
    songs = songs.copy()
    niche = songs["niche_genres"].apply(parse_list)

    songs["genre_original"] = songs["genre"]
    songs["genre_base"] = songs["genre"].map(BASE_GENRE_MAP).fillna("Other")
    songs["genre_override"] = niche.apply(choose_genre_override)
    songs["genre_engineered"] = songs["genre_override"].fillna(songs["genre_base"])
    songs["religious_music"] = niche.apply(has_religious_tag)

    unexpected = set(songs["genre_engineered"].dropna().unique()) - set(FINAL_GENRES)
    assert not unexpected, f"Unexpected genres: {unexpected}"
    return songs
