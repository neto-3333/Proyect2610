"""Statistical operations shared by the interface and tests."""

import io
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL

FREQUENCIES = {
    "Mensual": "M",
    "Trimestral": "Q-DEC",
    "Anual": "Y-DEC",
    "Semanal (domingo)": "W-SUN",
    "Diaria": "D",
    "Días hábiles (lunes–viernes)": "B",
}


def read_csv(data, sep=None, decimal=".", encoding="utf-8-sig"):
    frame = pd.read_csv(
        io.BytesIO(data), sep=sep, decimal=decimal, encoding=encoding, engine="python"
    )
    if frame.empty:
        raise ValueError("El archivo no contiene observaciones.")
    if len(frame) > 100000:
        raise ValueError("Esta versión admite hasta 100 000 filas.")
    return frame


def parse_dates(values, dayfirst=True):
    # ISO dates remain year-month-day even when local dates are day-first.
    text = values.astype(str)
    iso = text.str.match(r"^\d{4}-\d{2}-\d{2}(?:$|[ T])")
    result = pd.to_datetime(values, errors="coerce", dayfirst=dayfirst, format="mixed")
    result.loc[iso] = pd.to_datetime(values.loc[iso], errors="coerce", format="ISO8601")
    return result


def prepare_series(
    frame,
    value,
    date=None,
    freq="M",
    start="2020-01-01",
    aggregate="Rechazar",
    dayfirst=True,
):
    values = pd.to_numeric(frame[value], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    invalid = frame[value].notna() & values.isna()
    if invalid.any():
        raise ValueError(
            f"Hay {invalid.sum()} valores no numéricos o infinitos. Revisa el separador decimal."
        )
    if date:
        dates = parse_dates(frame[date], dayfirst)
        if dates.isna().any():
            raise ValueError(
                "Hay fechas vacías o inválidas; corrígelas antes de crear la serie."
            )
        index = pd.PeriodIndex(dates, freq=freq)
    else:
        index = pd.period_range(start, periods=len(values), freq=freq)
    series = pd.Series(values.to_numpy(), index=index, name=value).sort_index()
    if series.index.has_duplicates:
        if aggregate == "Rechazar":
            raise ValueError(
                "Hay varias filas en el mismo período. Selecciona un grupo o una regla de agregación."
            )
        grouped = series.groupby(level=0)
        series = grouped.sum(min_count=1) if aggregate == "Sumar" else grouped.mean()
    full = pd.period_range(series.index.min(), series.index.max(), freq=freq)
    if len(full) > 100000:
        raise ValueError(
            "El calendario resultante excede 100 000 períodos; revisa la frecuencia."
        )
    return series.reindex(full)


def fill_missing(series, method="Sin rellenar", max_gap=3):
    result = series.copy()
    if method == "Sin rellenar":
        return result
    missing = series.isna()
    groups = missing.ne(missing.shift()).cumsum()
    lengths = missing.groupby(groups).transform("sum")
    if method == "Interpolación lineal (interior)":
        candidate = series.interpolate(limit_area="inside")
    elif method == "Último valor (solo hacia adelante)":
        candidate = series.ffill()
    else:
        raise ValueError("Método desconocido")
    eligible = missing & (lengths <= max_gap)
    result.loc[eligible] = candidate.loc[eligible]
    return result


def descriptive(series):
    x = series.dropna()
    result = x.describe().to_dict()
    result.update(
        faltantes=int(series.isna().sum()),
        mediana=x.median(),
        varianza=x.var(),
        asimetría=x.skew(),
        curtosis=x.kurt(),
        IQR=x.quantile(0.75) - x.quantile(0.25),
    )
    return pd.Series(result, name="Valor")


def diagnostics(series):
    x = series.dropna()
    rows = []
    if len(x) < 8 or x.nunique() < 2:
        return pd.DataFrame(
            [
                {
                    "Prueba": "No aplicable",
                    "Detalle": "Se necesitan al menos 8 valores y variación.",
                }
            ]
        )
    for name, fn, null in [
        (
            "Normalidad (Shapiro–Wilk)",
            lambda: stats.shapiro(x),
            "Distribución normal; la dependencia temporal afecta la interpretación.",
        ),
        (
            "ADF",
            lambda: adfuller(x, autolag="AIC")[:2],
            "Raíz unitaria (modelo con constante).",
        ),
        (
            "KPSS",
            lambda: kpss(x, regression="c", nlags="auto")[:2],
            "Estacionariedad alrededor de un nivel.",
        ),
    ]:
        if name.startswith("Normalidad") and len(x) > 5000:
            rows.append(
                {"Prueba": name, "Detalle": "Omitida: más de 5 000 observaciones."}
            )
            continue
        if name in ("ADF", "KPSS") and len(x) > 5000:
            rows.append(
                {
                    "Prueba": name,
                    "Detalle": "Omitida: límite interactivo de 5 000 observaciones.",
                }
            )
            continue
        if name in ("ADF", "KPSS") and series.isna().any():
            rows.append(
                {
                    "Prueba": name,
                    "Detalle": "Resuelve faltantes sin comprimir el calendario.",
                }
            )
            continue
        try:
            with warnings.catch_warnings(record=True) as caught:
                stat, p = fn()
            rows.append(
                {
                    "Prueba": name,
                    "Estadístico": float(stat),
                    "p": float(p),
                    "H₀": null,
                    "Detalle": "Rechazo de H₀ al 5 %"
                    if p < 0.05
                    else "No se rechaza H₀ al 5 %",
                    "Avisos": "; ".join(str(w.message) for w in caught),
                }
            )
        except ValueError as exc:
            rows.append({"Prueba": name, "Detalle": str(exc)})
    return pd.DataFrame(rows)


def _fit_predict(y, horizon, spec):
    kind, order, seasonal = spec
    n = len(y)
    if kind == "Último valor":
        pred = np.repeat(y.iloc[-1], horizon)
        residual = np.diff(y)
        scale = np.sqrt(np.arange(1, horizon + 1))
    elif kind == "Estacional ingenuo":
        m = seasonal[-1]
        pred = np.resize(y.to_numpy()[-m:], horizon)
        residual = y.to_numpy()[m:] - y.to_numpy()[:-m]
        scale = np.sqrt(1 + np.arange(horizon) // m)
    elif kind == "Tendencia lineal":
        fit = stats.linregress(np.arange(n), y.to_numpy())
        future = np.arange(n, n + horizon)
        pred = fit.intercept + fit.slope * future
        residual = y.to_numpy() - (fit.intercept + fit.slope * np.arange(n))
        scale = np.sqrt(
            1
            + 1 / n
            + (future - (n - 1) / 2) ** 2 / np.sum((np.arange(n) - (n - 1) / 2) ** 2)
        )
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y.to_numpy(),
                order=order,
                seasonal_order=seasonal,
                trend="c" if order[1] == 0 and seasonal[1] == 0 else "n",
            ).fit(disp=False, maxiter=100)
        if not model.mle_retvals.get("converged", False):
            raise ValueError("El modelo no convergió")
        forecast = model.get_forecast(horizon)
        ci = np.asarray(forecast.conf_int(alpha=0.05))
        return (
            np.asarray(forecast.predicted_mean),
            ci[:, 0],
            ci[:, 1],
            np.asarray(model.resid)[max(model.loglikelihood_burn, 1) :],
            float(model.aicc),
        )
    sigma = np.std(residual, ddof=1) if len(residual) > 1 else 0.0
    band = 1.96 * sigma * scale
    return pred, pred - band, pred + band, residual, np.nan


def forecast_series(
    series,
    horizon=12,
    season=12,
    mode="Automático",
    fill_method="Sin rellenar",
    max_gap=3,
):
    if horizon < 1 or horizon > 60 or len(series) > 1500:
        raise ValueError(
            "Límites de esta versión: horizonte 1–60 y hasta 1 500 períodos para pronóstico."
        )
    if len(series) < 24:
        raise ValueError(
            "Se requieren al menos 24 períodos para esta validación temporal."
        )
    holdout = min(horizon, max(3, len(series) // 5))
    # A separate final block remains unseen during selection.
    cut = len(series) - holdout
    val_size = min(holdout, max(3, cut // 5))
    train_end = cut - val_size
    if train_end < 12:
        raise ValueError(
            "El historial no alcanza para entrenamiento, selección y evaluación."
        )
    specs = [("Último valor", None, None), ("Tendencia lineal", None, None)]
    if season >= 2 and train_end >= 2 * season:
        specs.append(("Estacional ingenuo", None, (0, 0, 0, season)))
    if mode != "Tendencia lineal":
        specs += [
            ("ARIMA", (p, d, q), (0, 0, 0, 0))
            for d in (0, 1)
            for p, q in ((0, 0), (1, 0), (0, 1), (1, 1))
        ]
        if 2 <= season <= 24 and train_end >= 3 * season:
            specs += [
                ("SARIMA", (p, d, 0), (1, 1, 0, season)) for p in (0, 1) for d in (0, 1)
            ]
    if mode == "Tendencia lineal":
        specs = [s for s in specs if s[0] == mode]
    if mode == "ARIMA / SARIMA":
        specs = [s for s in specs if s[0] in ("ARIMA", "SARIMA")]
    train = fill_missing(series.iloc[:train_end], fill_method, max_gap)
    if train.isna().any():
        raise ValueError(
            "Quedan faltantes en el entrenamiento; cambia la limpieza o revisa la serie."
        )
    target = series.iloc[train_end:cut]
    observed = target.notna().to_numpy()
    if observed.sum() < 3:
        raise ValueError(
            "Se requieren al menos 3 valores observados en el bloque de selección."
        )
    rows = []
    successes = []
    for spec in specs:
        name = f"{spec[0]} {spec[1] or ''} {spec[2] or ''}".strip()
        try:
            pred, _, _, _, aicc = _fit_predict(train, val_size, spec)
            mae = float(np.mean(np.abs(target.to_numpy()[observed] - pred[observed])))
            if not np.isfinite(mae):
                raise ValueError("Predicciones no finitas")
            rows.append(
                {
                    "Modelo": name,
                    "MAE selección": mae,
                    "AICc": aicc,
                    "Estado": "Evaluado",
                }
            )
            successes.append((mae, spec, name))
        except Exception as exc:
            rows.append({"Modelo": name, "Estado": f"Descartado: {exc}"})
    if not successes:
        raise ValueError("Ningún candidato pudo ajustarse.")
    _, best, name = min(successes, key=lambda r: r[0])
    past = fill_missing(series.iloc[:cut], fill_method, max_gap)
    if past.isna().any():
        raise ValueError("Quedan faltantes antes de la evaluación final.")
    test = series.iloc[cut:]
    mask = test.notna().to_numpy()
    if mask.sum() < 3:
        raise ValueError("Se requieren 3 valores observados en la evaluación final.")
    pred, _, _, _, _ = _fit_predict(past, holdout, best)
    errors = test.to_numpy()[mask] - pred[mask]
    lag = season if season >= 2 and len(past) > season else 1
    denom = np.mean(np.abs(past.to_numpy()[lag:] - past.to_numpy()[:-lag]))
    metrics = {
        "MAE final": float(np.mean(np.abs(errors))),
        "RMSE final": float(np.sqrt(np.mean(errors**2))),
        "MASE final": float(np.mean(np.abs(errors)) / denom) if denom > 0 else None,
        "Períodos evaluación": holdout,
        "Períodos selección": val_size,
    }
    complete = fill_missing(series, fill_method, max_gap)
    if complete.isna().any():
        raise ValueError("Quedan faltantes en la serie completa.")
    pred, low, high, resid, _ = _fit_predict(complete, horizon, best)
    future = pd.period_range(
        series.index[-1] + 1, periods=horizon, freq=series.index.freq
    )
    output = pd.DataFrame(
        {"Pronóstico": pred, "Inferior 95 %": low, "Superior 95 %": high},
        index=future.astype(str),
    )
    lb = (
        acorr_ljungbox(resid, lags=[min(10, max(1, len(resid) // 5))], return_df=True)
        if len(resid) > 5 and np.std(resid) > 0
        else pd.DataFrame()
    )
    return output, pd.DataFrame(rows), metrics, name, lb


def seasonal_adjust(series, method="STL", period=12):
    if len(series) > 10000:
        raise ValueError("El ajuste interactivo admite hasta 10 000 períodos.")
    if series.isna().any() or series.nunique() < 2:
        raise ValueError("Se necesita una serie completa con variación.")
    if method == "STL":
        if period < 2 or len(series) < 3 * period:
            raise ValueError("STL requiere aquí al menos tres ciclos completos.")
        result = STL(series.to_numpy(), period=period, robust=True).fit()
        return pd.DataFrame(
            {
                "Original": series.to_numpy(),
                "Ajustada": series.to_numpy() - result.seasonal,
                "Tendencia": result.trend,
                "Estacional": result.seasonal,
                "Irregular": result.resid,
            },
            index=series.index.astype(str),
        ), "STL robusto, aditivo."
    if series.index.freqstr not in ("M", "Q-DEC"):
        raise ValueError("X-13 admite aquí frecuencia mensual o trimestral calendario.")
    m = 12 if series.index.freqstr == "M" else 4
    if len(series) < 5 * m:
        raise ValueError("Este módulo requiere al menos cinco años para X-13.")
    import x13binary
    import subprocess
    import tempfile
    from pathlib import Path

    first = series.index[0]
    initial = first.month if m == 12 else first.quarter
    data = "\n".join(format(float(v), ".15g") for v in series)
    spec = f"series {{ period={m} start={first.year}.{initial} data=(\n{data}\n) }}\ntransform {{ function=auto }}\nautomdl {{ }}\noutlier {{ }}\nx11 {{ save=(d11 d12 d13) }}\n"
    with tempfile.TemporaryDirectory(prefix="serieclara_x13_") as tmp:
        root = Path(tmp)
        (root / "series.spc").write_text(spec)
        run = subprocess.run(
            [x13binary.find_x13_bin(), "series"],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=90,
        )
        tables = []
        for suffix in ("d11", "d12", "d13"):
            path = root / f"series.{suffix}"
            if not path.exists():
                errors = (
                    (root / "series_err.html").read_text(errors="replace")
                    if (root / "series_err.html").exists()
                    else ""
                )
                raise ValueError(
                    "X-13 no produjo componentes. "
                    + run.stdout[-2000:]
                    + errors[-1000:]
                )
            table = pd.read_csv(path, sep=r"\s+", skiprows=2, header=None)
            if len(table) != len(series):
                raise ValueError("X-13 devolvió un calendario incompleto.")
            tables.append(table.iloc[:, 1].to_numpy(dtype=float))
        report = spec + "\n\n" + run.stdout
        for path in sorted(root.glob("*.html")):
            report += "\n\n" + path.name + "\n" + path.read_text(errors="replace")
    return pd.DataFrame(
        {
            "Original": series.to_numpy(),
            "Ajustada": tables[0],
            "Tendencia": tables[1],
            "Irregular": tables[2],
        },
        index=series.index.astype(str),
    ), report
