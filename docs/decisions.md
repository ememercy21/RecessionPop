# Decision log

These are the reasons behind the rules in `src/recessionpop/`. They came from the audits in the original notebooks, now kept in `archive/notebooks/`.

## 1. Timeline: Billboard dates rule

The 550K `year` field has obvious errors, such as modern recordings coded as 1900, so **Billboard `Date` is the only time axis**.

## 2. Genre engineering (`genres.py`)

- The broad source `genre` is useful but not enough on its own. For example, the source "Folk" category contained systematic reggae/dancehall misclassifications.
- **Only exact niche-tag matches can vote.** Substring matching produced false positives (`ska`→`skate punk`, `dub`→`dubstep`, `rap`→`trap`).
- If no tag votes, or the vote is tied, the broad source genre is kept.
- The 754 unique niche tags were sorted into four groups: **vote / inherit / ignore / fall back to source genre**. There are 273 voting tags. Each one votes for exactly one genre, and the vote, inherit and ignore sets don't overlap (this is checked at import time).
- Religious content is a **separate flag**, not a genre. For example, `christian hip hop` votes Hip-Hop *and* sets `religious_music = 1`.

## 3. Duplicate recordings (`matching.build_canonical_lookup`)

The 550K table repeats the same title + artist across albums, compilations and reissues. The canonical row is chosen by:

1. the most common rounded audio signature (danceability, energy, valence, tempo, duration);
2. then the highest Spotify popularity within that signature;
3. then the Spotify ID, as a stable tie-breaker.

Two test cases shaped this rule:

- *Drive*, The Cars: 70 candidate rows, and the dominant profile appeared 37 times. The rule picks the Heartbeat City recording.
- *Last Last*, Burna Boy: 22 rows, with the dominant profile appearing 21 times. The rule picks the Love, Damini recording.

Rounded values are used only to resolve duplicates. The analysis uses the original values.

## 4. Matching passes

| Pass | Rule | Matches |
|---|---|---|
| 1 | exact normalised title + exact normalised artist | 10,466 |
| 2 | exact title + Billboard lead artist after removing "Featuring …" | 798 |
| 3 | Spotify title with only its feature credit removed + a Billboard Featuring credit + overlap with a Spotify secondary artist, unique candidates only | 700 |
| 3m | manually audited ambiguous Pass-3 cases | 12 |
| **Total** | | **11,976 of 32,592 unique songs (36.7%)** |

Rejected ideas:

- **Title-only joins.** 22k unmatched songs blew up into 122k candidate rows, because titles like *Stay*, *Monster* and *Drive* belong to many unrelated artists.
- **Removing everything after `feat`.** This collapsed remixes and edits into the original (for example *Roses (feat. ROZES) – Loosid Remix* became *Roses*). That's dangerous because the project models *audio* features, and a remix can sound quite different. Only the feature-credit phrase is removed now.
- **Generic version stripping.** The Taylor Swift inspection found very few legitimate cases.

Manual Pass-3 cases: 12 of the 14 two-candidate songs could be resolved from collaborator credits (for example *Essence* → the row credited to both Justin Bieber and Tems). The other two are left unmatched on purpose, because Billboard itself credits "X Or Y" and the Spotify rows are separate versions:

- *Into You* — Fabolous Featuring Tamia Or Ashanti
- *Thinking 'Bout You* — Dustin Lynch Featuring Lauren Alaina Or MacKenzie Porter

## 5. Analysis window: 2001-01 to 2020-12

- The music signal is **rank-weighted** (`101 − Rank`). Missing a #1 hit matters 100 times more than missing #100, so coverage is judged on rank-weighted *monthly* coverage, not on raw row counts. An earlier 1999–2018 window was based on unweighted coverage and has been dropped.
- Over 2001–2020, monthly rank-weighted coverage has a mean of 54% and a minimum of 32%. Only 5 months fall below 40%.
- Coverage drops off sharply after 2020. Recent years are **not** forced in with weak or fuzzy matches. That's a reported limitation.

## 6. Lyrics

- Only songs with at least 20 words count as valid lyrics (3 of 11,974 fail).
- The scoring uses the NRC Emotion Lexicon v0.92, word level. Each emotion rate is the share of all tokens tagged with that emotion.
- Songs with no NRC-matched tokens get NaN rates rather than zero.
- `lyric_sentiment = positive_rate − negative_rate`.

## 7. Forecasting baseline

- UMCSENT is non-stationary in levels (ADF) and stationary in first differences.
- ARIMA models on differences barely beat (0,1,0), so the baseline is a **random walk**.
- Music models predict the *monthly change* from music signals at t−1, using training-period z-scores and OLS with no intercept. That predicted change is added to the random-walk forecast.
- Train: 2001–2016. Test: 2017–2020.
