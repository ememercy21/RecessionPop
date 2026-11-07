"""Reverse direction: does consumer sentiment lead the music?

If music is a *lagging* indicator, past changes in UMCSENT should help forecast
changes in chart mood (lyrics, audio, genre mix) beyond the music series' own
history. Tests:

* Out-of-sample, rolling origin: baseline AR(2) on the music feature's own
  monthly change vs. baseline + a lagged sentiment change. Nested Clark-West
  test, Benjamini-Hochberg corrected across all feature x lag variants.
* In-sample Granger tests in *both* directions with HAC Wald F-tests, so the two
  directions are compared on identical footing.
* Lead/lag cross-correlations of the monthly changes.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .predictors import _nw_lags

SENTIMENT_VARIANTS = {
    "dU t-1": lambda u: u.diff().shift(1),
    "dU t-3": lambda u: u.diff().shift(3),
    "dU t-6": lambda u: u.diff().shift(6),
    "3-mo dU to t-1": lambda u: u.diff(3).shift(1),
}
AR_LAGS = 2


def music_features(groups):
    return [f for f, g in groups.items() if g != "control"]


def clark_west_nested(actual, pred_small, pred_big):
    """One-sided CW test: H1 = bigger nested model forecasts better."""
    f = (actual - pred_small) ** 2 - ((actual - pred_big) ** 2 - (pred_small - pred_big) ** 2)
    fit = sm.OLS(f, np.ones_like(f)).fit(cov_type="HAC", cov_kwds={"maxlags": _nw_lags(len(f))})
    t = float(fit.tvalues[0])
    return t, float(1 - stats.norm.cdf(t))


def _fit_predict(Ztr, ytr, zt):
    mu, sd = Ztr.mean(0), Ztr.std(0, ddof=1)
    sd[sd == 0] = 1
    A = np.column_stack([np.ones(len(Ztr)), (Ztr - mu) / sd])
    beta, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    return np.r_[1.0, (zt - mu) / sd] @ beta


def rolling_nested(y, base, extra, months, start, min_train=60):
    """Expanding-window forecasts of y from `base` and from `base + extra`.

    Both models use an intercept and the same training rows each month.
    """
    y = np.asarray(y, float)
    B = np.asarray(base, float)
    E = np.asarray(extra, float).reshape(len(y), -1)
    F = np.column_stack([B, E])
    months = pd.to_datetime(pd.Series(months)).to_numpy()
    ok = np.isfinite(y) & np.isfinite(F).all(axis=1)
    test_idx = np.where((months >= np.datetime64(pd.Timestamp(start))) & ok)[0]

    p_small, p_big = np.empty(len(test_idx)), np.empty(len(test_idx))
    for j, t in enumerate(test_idx):
        tr = ok & (np.arange(len(y)) < t)
        if tr.sum() < min_train:
            raise ValueError("not enough training rows")
        p_small[j] = _fit_predict(B[tr], y[tr], B[t])
        p_big[j] = _fit_predict(F[tr], y[tr], F[t])
    return test_idx, p_small, p_big


def _own_lags(x, k=AR_LAGS):
    d = x.diff()
    return np.column_stack([d.shift(i) for i in range(1, k + 1)])


def reverse_screen(panel, groups, start="2008-01-01"):
    """Sentiment -> music, out of sample, for every music feature x sentiment variant."""
    u = panel["UMCSENT"]
    rows = []
    for feat in music_features(groups):
        y = panel[feat].diff().to_numpy()
        base = _own_lags(panel[feat])
        for vname, make in SENTIMENT_VARIANTS.items():
            idx, ps, pb = rolling_nested(y, base, make(u).to_numpy(), panel["month"], start)
            a = y[idx]
            pre = (panel["month"].iloc[idx] < "2020-01-01").to_numpy()
            row = {"feature": feat, "group": groups[feat], "sentiment_predictor": vname}
            for w, m in {"full": np.ones(len(a), bool), "pre2020": pre}.items():
                rmse_s = np.sqrt(np.mean((a[m] - ps[m]) ** 2))
                rmse_b = np.sqrt(np.mean((a[m] - pb[m]) ** 2))
                t, p = clark_west_nested(a[m], ps[m], pb[m])
                row |= {f"rmse_ratio_{w}": rmse_b / rmse_s, f"cw_p_{w}": p}
            rows.append(row)
    res = pd.DataFrame(rows)
    for w in ["full", "pre2020"]:
        res[f"q_{w}"] = multipletests(res[f"cw_p_{w}"], method="fdr_bh")[1]
    return res.sort_values("cw_p_full").reset_index(drop=True)


def _granger(y, own, other, k):
    """HAC Wald F-test that k lags of `other` add to an intercept + k own lags."""
    df = pd.DataFrame({"y": y})
    for i in range(1, k + 1):
        df[f"own{i}"] = own.shift(i)
        df[f"oth{i}"] = other.shift(i)
    df = df.dropna()
    X = sm.add_constant(df.drop(columns="y"))
    fit = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": _nw_lags(len(df))})
    R = np.zeros((k, X.shape[1]))
    for i in range(k):
        R[i, list(X.columns).index(f"oth{i + 1}")] = 1
    w = fit.wald_test(R, scalar=True, use_f=True)
    return float(w.fvalue), float(w.pvalue)


def granger_both_ways(panel, groups, k=3, end=None):
    """In-sample Granger F-tests for music -> sentiment and sentiment -> music."""
    df = panel if end is None else panel.loc[panel["month"] < end]
    du = df["UMCSENT"].diff()
    rows = []
    for feat in music_features(groups):
        dm = df[feat].diff()
        f_fw, p_fw = _granger(du, du, dm, k)  # music -> sentiment
        f_rv, p_rv = _granger(dm, dm, du, k)  # sentiment -> music
        rows.append({"feature": feat, "group": groups[feat],
                     "F_music_to_sentiment": f_fw, "p_music_to_sentiment": p_fw,
                     "F_sentiment_to_music": f_rv, "p_sentiment_to_music": p_rv})
    res = pd.DataFrame(rows)
    res["q_music_to_sentiment"] = multipletests(res["p_music_to_sentiment"], method="fdr_bh")[1]
    res["q_sentiment_to_music"] = multipletests(res["p_sentiment_to_music"], method="fdr_bh")[1]
    return res.set_index("feature")


def cross_correlations(panel, groups, max_lag=6):
    """corr(dUMCSENT_t, dMusic_{t+k}); k > 0 means sentiment moves first."""
    du = panel["UMCSENT"].diff()
    out = {}
    for feat in music_features(groups):
        dm = panel[feat].diff()
        out[feat] = {k: du.corr(dm.shift(-k)) for k in range(-max_lag, max_lag + 1)}
    return pd.DataFrame(out).T.rename_axis(columns="k (months music lags sentiment)")


def power_check(panel, feature="valence", r2_grid=(0.02, 0.05, 0.10), reps=200,
                start="2008-01-01", seed=0):
    """Detection rate when a synthetic music change truly depends on dU_{t-1}.

    The synthetic target keeps the real feature's own dynamics (its actual monthly
    changes, standardised) and adds a planted sentiment component explaining
    `r2` of the variance, plus fresh noise each repetition.
    """
    rng = np.random.default_rng(seed)
    u = panel["UMCSENT"]
    du1 = u.diff().shift(1)
    z_du1 = ((du1 - du1.mean()) / du1.std()).to_numpy()
    dm = panel[feature].diff()
    z_dm = ((dm - dm.mean()) / dm.std()).to_numpy()
    base = _own_lags(panel[feature])  # own history as observed
    out = {}
    for r2 in r2_grid:
        hits = 0
        for _ in range(reps):
            noise = rng.standard_normal(len(z_dm))
            y = np.sqrt(r2) * z_du1 + np.sqrt(1 - r2) * (0.5 * z_dm + np.sqrt(0.75) * noise)
            idx, ps, pb = rolling_nested(y, base, du1.to_numpy(), panel["month"], start)
            hits += clark_west_nested(y[idx], ps, pb)[1] < 0.05
        out[f"true R² = {r2:.0%}"] = hits / reps
    return pd.Series(out, name="detection rate")
