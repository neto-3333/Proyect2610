import numpy as np
import pandas as pd
import pytest
from analysis_core import *


def monthly(n=120):
    t = np.arange(n)
    rng = np.random.default_rng(12)
    return pd.Series(
        100 + 0.4 * t + 10 * np.sin(t * 2 * np.pi / 12) + rng.normal(0, 2, n),
        index=pd.period_range("2015-01", periods=n, freq="M"),
    )


def test_csv_decimal_and_duplicates():
    frame = read_csv(
        b"fecha;valor\n2020-01-01;1,5\n2020-01-01;2,5\n2020-03-01;4,0\n",
        sep=";",
        decimal=",",
    )
    with pytest.raises(ValueError, match="varias filas"):
        prepare_series(frame, "valor", "fecha")
    s = prepare_series(frame, "valor", "fecha", aggregate="Sumar")
    assert s.iloc[0] == 4 and np.isnan(s.iloc[1]) and len(s) == 3


def test_missing_bounds_and_long_runs():
    s = pd.Series([np.nan, 1, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan])
    out = fill_missing(s, "Interpolación lineal (interior)", 2)
    assert out.iloc[2] == 2 and out.iloc[4:7].isna().all()
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[-1])


def test_null_sum_stays_missing():
    f = pd.DataFrame({"date": ["2020-01-01", "2020-01-02"], "v": [np.nan, np.nan]})
    assert prepare_series(f, "v", "date", aggregate="Sumar").isna().all()


def test_diagnostics_no_calendar_compression():
    s = monthly()
    s.iloc[4] = np.nan
    result = diagnostics(s)
    assert "calendario" in result.loc[result.Prueba == "ADF", "Detalle"].iloc[0]


def test_forecast_trend_and_dates():
    s = pd.Series(
        np.arange(60, dtype=float),
        index=pd.period_range("2020-01", periods=60, freq="M"),
    )
    out, rank, metrics, winner, lb = forecast_series(s, 6, 12, "Tendencia lineal")
    np.testing.assert_allclose(out["Pronóstico"], np.arange(60, 66), atol=1e-8)
    assert metrics["MAE final"] < 1e-8 and out.index[0] == "2025-01"
    assert (out["Inferior 95 %"] <= out["Superior 95 %"]).all()


def test_forecast_automatic():
    out, rank, metrics, winner, lb = forecast_series(monthly(), 6, 12)
    assert len(out) == 6 and np.isfinite(out.to_numpy()).all()
    assert metrics["MAE final"] >= 0 and len(rank) >= 10


def test_no_future_imputation_leak():
    s = monthly(60)
    s.iloc[35:] = np.nan
    with pytest.raises(ValueError, match="faltantes|observados"):
        forecast_series(s, 6, 12, fill_method="Interpolación lineal (interior)")


def test_stl_reconstruction():
    s = monthly()
    result, _ = seasonal_adjust(s)
    np.testing.assert_allclose(result.Ajustada + result.Estacional, s)
    assert result.Ajustada.std() < result.Original.std()


def test_x13_real_binary():
    result, report = seasonal_adjust(monthly(), "X-13")
    assert len(result) == 120 and np.isfinite(result.to_numpy()).all()
    assert "x11" in report.lower()


def test_x13_rejects_daily():
    s = pd.Series(
        np.arange(100.0), index=pd.period_range("2020-01-01", periods=100, freq="D")
    )
    with pytest.raises(ValueError, match="mensual"):
        seasonal_adjust(s, "X-13")


def test_iso_dates_ignore_dayfirst():
    f = pd.DataFrame({"date": ["2024-02-01", "2024-03-01"], "v": [1, 2]})
    s = prepare_series(f, "v", "date", dayfirst=True)
    assert list(s.index.astype(str)) == ["2024-02", "2024-03"]


def test_x13_quarterly():
    t = np.arange(80)
    rng = np.random.default_rng(8)
    s = pd.Series(
        200 + 0.8 * t + 12 * np.sin(t * 2 * np.pi / 4) + rng.normal(0, 2, 80),
        index=pd.period_range("2000Q1", periods=80, freq="Q-DEC"),
    )
    result, report = seasonal_adjust(s, "X-13")
    assert len(result) == 80 and np.isfinite(result.to_numpy()).all()


def test_constant_series_reports_limitation():
    assert diagnostics(pd.Series([5.0] * 30)).iloc[0]["Prueba"] == "No aplicable"
