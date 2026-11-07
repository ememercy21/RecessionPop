"""Forecasting UMCSENT with lagged music signals.

Baseline: random walk (next month = this month), justified by the ADF / ARIMA
diagnostics on the training window. Augmented models predict the monthly *change*
in UMCSENT from music signals in month t-1 (OLS, no intercept, train-only scaling)
and add it to the random-walk forecast.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from . import config as cfg

TARGET = "UMCSENT"


def build_model_data(monthly_music, target):
    return monthly_music.merge(target, on="month", how="inner", validate="one_to_one")


def split_mask(model_data, test_start=cfg.TEST_START):
    return model_data["month"] < test_start


def adf_report(series):
    stat, p, lags, nobs, crit, _ = adfuller(series.dropna(), autolag="AIC")
    return {"adf_stat": stat, "p_value": p, "lags": lags, "nobs": nobs,
            **{f"crit_{k}": v for k, v in crit.items()}}


def arima_comparison(series, orders=((0, 1, 0), (1, 1, 0), (2, 1, 0), (0, 1, 1), (1, 1, 1))):
    if isinstance(series.index, pd.DatetimeIndex):
        series = series.asfreq("MS")
    rows = []
    for order in orders:
        fit = ARIMA(series, order=order).fit()
        lb = acorr_ljungbox(fit.resid.dropna(), lags=[12, 24], return_df=True)
        rows.append({
            "order": order, "AIC": fit.aic, "BIC": fit.bic,
            "ljung_box_p12": lb.loc[12, "lb_pvalue"],
            "ljung_box_p24": lb.loc[24, "lb_pvalue"],
        })
    return pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)


def add_signals(model_data, features=("lyric_sentiment", "valence"), lag=1):
    """Train-period z-scores, a combined core_sentiment index, and lags."""
    df = model_data.copy()
    train = split_mask(df)
    for feat in features:
        mu, sd = df.loc[train, feat].mean(), df.loc[train, feat].std()
        df[f"{feat}_z"] = (df[feat] - mu) / sd
        df[f"{feat}_z_lag{lag}"] = df[f"{feat}_z"].shift(lag)

    df["core_sentiment"] = 0.5 * df["lyric_sentiment_z"] + 0.5 * df["valence_z"]
    df[f"core_sentiment_lag{lag}"] = df["core_sentiment"].shift(lag)
    df[f"{TARGET}_prev"] = df[TARGET].shift(1)
    df[f"{TARGET}_change"] = df[TARGET].diff()
    return df


def fit_change_model(df, predictors):
    """OLS of monthly UMCSENT change on predictors over the training window."""
    train = df.loc[split_mask(df), [f"{TARGET}_change", *predictors]].dropna()
    return sm.OLS(train[f"{TARGET}_change"], train[list(predictors)]).fit()


def _metrics(actual, pred):
    err = actual - pred
    return {"MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err**2)))}


def evaluate(df, models):
    """Out-of-sample MAE/RMSE for the random walk and each augmented model.

    models: {name: predictor list}. Returns (metrics table, test-period forecasts,
    fitted OLS results).
    """
    test = df.loc[~split_mask(df)].copy()
    test["random_walk"] = test[f"{TARGET}_prev"]

    rows = [{"model": "random_walk", **_metrics(test[TARGET], test["random_walk"])}]
    fits = {}
    for name, predictors in models.items():
        fit = fit_change_model(df, predictors)
        fits[name] = fit
        test[name] = test["random_walk"] + test[list(predictors)] @ fit.params
        rows.append({
            "model": name, **_metrics(test[TARGET], test[name]),
            "train_r2": fit.rsquared,
            "min_p_value": float(fit.pvalues.min()),
        })

    table = pd.DataFrame(rows).set_index("model")
    base = table.loc["random_walk", "RMSE"]
    table["RMSE_vs_rw_pct"] = (table["RMSE"] / base - 1) * 100
    return table, test, fits
