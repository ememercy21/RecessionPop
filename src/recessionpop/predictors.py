"""Systematic out-of-sample screen of candidate music predictors.

Every candidate = feature x transform x lag. Each one gets the same test:

* Target: monthly change in UMCSENT. The benchmark forecast of the change is 0
  (random walk).
* Rolling origin: for every month t in the evaluation window, refit a
  no-intercept OLS of dUMCSENT on the (z-scored) predictor using only months < t,
  then forecast t. Nothing from month t or later is used.
* Clark-West (2007) test for nested models: one-sided, H1 = candidate beats RW,
  with Newey-West standard errors. Raw p-values are corrected across all
  candidates with Benjamini-Hochberg (FDR), because with hundreds of candidates
  a few will "win" by chance.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

LYRIC_FEATURES = [
    "lyric_sentiment", "positive_rate", "negative_rate",
    "anger_rate", "fear_rate", "joy_rate", "sadness_rate",
]
AUDIO_FEATURES = ["valence", "energy", "danceability", "tempo", "mode"]
TRANSFORMS = ("level", "d1", "d3", "dev12")
LAGS = (1, 3, 6)

# NBER business-cycle recessions (peak month through trough month)
RECESSIONS = [
    ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
    ("2020-02-01", "2020-04-01"),
]


def in_recession(months):
    months = pd.to_datetime(pd.Series(months))
    flag = pd.Series(False, index=months.index)
    for s, e in RECESSIONS:
        flag |= months.between(pd.Timestamp(s), pd.Timestamp(e))
    return flag.to_numpy()


def _transform(x, how):
    if how == "level":
        return x
    if how == "d1":
        return x.diff()
    if how == "d3":
        return x.diff(3)
    if how == "dev12":
        return x - x.rolling(12).mean()
    raise ValueError(how)


def build_panel(model_data, genre_shares=None, mich=None, top_genres=8):
    """Monthly frame of raw signals: music features, genre shares, controls, target."""
    df = model_data[["month", "UMCSENT", *LYRIC_FEATURES, *AUDIO_FEATURES]].copy()
    groups = {f: "lyrics" for f in LYRIC_FEATURES} | {f: "audio" for f in AUDIO_FEATURES}

    if genre_shares is not None:
        keep = genre_shares.mean().sort_values(ascending=False).index[:top_genres]
        g = genre_shares[keep].add_prefix("share_").rename_axis("month").reset_index()
        df = df.merge(g, on="month", how="left")
        groups |= {c: "genre" for c in g.columns if c != "month"}

    df["dUMCSENT"] = df["UMCSENT"].diff()
    df["ar_dUMCSENT"] = df["UMCSENT"]  # transformed to d1 below -> its own past change
    groups["ar_dUMCSENT"] = "control"

    if mich is not None:
        df = df.merge(mich, on="month", how="left")
        groups["MICH"] = "control"

    return df.sort_values("month").reset_index(drop=True), groups


def build_candidates(panel, groups, transforms=TRANSFORMS, lags=LAGS):
    """Columns x_{t-lag} for each feature/transform/lag, plus a metadata table."""
    cols, meta = {}, []
    for feat, group in groups.items():
        feat_transforms = ("d1",) if feat == "ar_dUMCSENT" else transforms
        for how in feat_transforms:
            base = _transform(panel[feat], how)
            for lag in lags:
                name = f"{feat}|{how}|L{lag}"
                cols[name] = base.shift(lag)
                meta.append({"candidate": name, "group": group, "feature": feat,
                             "transform": how, "lag": lag})
    X = pd.DataFrame(cols, index=panel.index)
    return X, pd.DataFrame(meta).set_index("candidate")


def _nw_lags(n):
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def clark_west(actual_change, pred_change):
    """One-sided Clark-West test of a nested model vs a zero-change random walk."""
    e1 = actual_change
    e2 = actual_change - pred_change
    f = e1**2 - (e2**2 - pred_change**2)
    fit = sm.OLS(f, np.ones_like(f)).fit(cov_type="HAC", cov_kwds={"maxlags": _nw_lags(len(f))})
    t = float(fit.tvalues[0])
    return t, float(1 - stats.norm.cdf(t))


def rolling_forecasts(y, X, months, start, min_train=60):
    """Expanding-window one-step forecasts of y (the change) from columns of X.

    X: DataFrame (n x k). Returns predicted change for test rows (NaN -> 0 = RW).
    A multi-column X is fitted jointly (block model).
    """
    y = np.asarray(y, float)
    Xv = np.asarray(X, float)
    if Xv.ndim == 1:
        Xv = Xv[:, None]
    months = pd.to_datetime(pd.Series(months)).to_numpy()
    test_idx = np.where(months >= np.datetime64(pd.Timestamp(start)))[0]
    ok = np.isfinite(y) & np.isfinite(Xv).all(axis=1)

    preds = np.zeros(len(test_idx))
    for j, t in enumerate(test_idx):
        tr = ok & (np.arange(len(y)) < t)
        if tr.sum() < min_train or not np.isfinite(Xv[t]).all():
            continue
        mu, sd = Xv[tr].mean(0), Xv[tr].std(0, ddof=1)
        sd[sd == 0] = 1
        Z = (Xv[tr] - mu) / sd
        beta, *_ = np.linalg.lstsq(Z, y[tr], rcond=None)
        preds[j] = ((Xv[t] - mu) / sd) @ beta
    return test_idx, preds


def screen(panel, X, meta, start="2008-01-01", exclude_covid=True):
    """Out-of-sample metrics for every candidate column in X."""
    y = panel["dUMCSENT"].to_numpy()
    months = panel["month"]
    rows, preds_all = [], {}
    for name in X.columns:
        idx, p = rolling_forecasts(y, X[name], months, start)
        preds_all[name] = p
        rows.append(_score(name, y[idx], p, months.iloc[idx], exclude_covid))
    res = pd.DataFrame(rows).set_index("candidate").join(meta)
    for w in ["full", "pre2020"]:
        res[f"q_{w}"] = multipletests(res[f"cw_p_{w}"], method="fdr_bh")[1]
    test_months = months.iloc[idx].reset_index(drop=True)
    return res.sort_values("cw_p_full"), pd.DataFrame(preds_all).assign(month=test_months)


def _score(name, actual, pred, months, exclude_covid=True):
    months = pd.to_datetime(pd.Series(months)).reset_index(drop=True)
    rec = in_recession(months)
    out = {"candidate": name}
    windows = {"full": np.ones(len(actual), bool),
               "pre2020": (months < "2020-01-01").to_numpy()}
    for w, m in windows.items():
        rmse_rw = np.sqrt(np.mean(actual[m] ** 2))
        rmse = np.sqrt(np.mean((actual[m] - pred[m]) ** 2))
        t, p = clark_west(actual[m], pred[m])
        out |= {f"rmse_ratio_{w}": rmse / rmse_rw, f"cw_t_{w}": t, f"cw_p_{w}": p}
    # loss differential split by business cycle (positive = candidate better)
    d = actual**2 - (actual - pred) ** 2
    out["gain_recession"] = d[rec].mean() if rec.any() else np.nan
    out["gain_expansion"] = d[~rec].mean()
    return out


def evaluate_block(panel, X, cols, start="2008-01-01", name=None):
    """Joint (multivariate) model over several candidate columns."""
    y = panel["dUMCSENT"].to_numpy()
    idx, p = rolling_forecasts(y, X[list(cols)], panel["month"], start)
    return _score(name or " + ".join(cols), y[idx], p, panel["month"].iloc[idx])


def granger_with_controls(panel, X, candidate, controls=("ar_dUMCSENT|d1|L1", "MICH|d1|L1"),
                          end=None):
    """In-sample HAC regression: does the candidate add to AR + inflation-expectation controls?"""
    df = pd.concat([panel[["month", "dUMCSENT"]], X[[candidate, *controls]]], axis=1).dropna()
    if end is not None:
        df = df.loc[df["month"] < end]
    Z = sm.add_constant(df[[candidate, *controls]])
    fit = sm.OLS(df["dUMCSENT"], Z).fit(cov_type="HAC", cov_kwds={"maxlags": _nw_lags(len(df))})
    return {"candidate": candidate, "coef": fit.params[candidate],
            "t": fit.tvalues[candidate], "p": fit.pvalues[candidate], "n": int(fit.nobs)}
