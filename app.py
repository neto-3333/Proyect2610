import json
import hashlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from analysis_core import (
    FREQUENCIES,
    read_csv,
    prepare_series,
    fill_missing,
    descriptive,
    diagnostics,
    forecast_series,
    seasonal_adjust,
    acf,
    pacf,
    parse_dates,
)

st.set_page_config(
    page_title="Serie Clara · Análisis de datos", page_icon="📈", layout="wide"
)
st.title("Serie Clara")
st.caption(
    "De un CSV a estadísticas, pronósticos y series ajustadas. En español y con decisiones visibles."
)
st.info(
    "Tus archivos se procesan en la memoria del servidor que ejecuta esta aplicación. No se envían a APIs de IA. X-13 utiliza archivos temporales que se eliminan al terminar. Para datos confidenciales, ejecútala en tu propio equipo."
)


@st.cache_data
def example():
    rng = np.random.default_rng(42)
    t = np.arange(120)
    return pd.DataFrame(
        {
            "fecha": pd.date_range("2015-01-01", periods=120, freq="MS").strftime(
                "%Y-%m-%d"
            ),
            "ventas": np.round(
                100 + 0.6 * t + 15 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 3, 120),
                2,
            ),
        }
    )


with st.sidebar:
    st.header("1 · Tus datos")
    uploaded = st.file_uploader("Archivo CSV (máximo 10 MB)", type=["csv"])
    demo = st.checkbox("Usar ejemplo de ventas mensuales", value=uploaded is None)
    separator = st.selectbox("Separador", ["Detectar", ",", ";", "Tabulación", "|"])
    decimal = st.selectbox("Separador decimal", [".", ","])
    encoding = st.selectbox("Codificación", ["utf-8-sig", "latin-1", "utf-16"])
    st.download_button(
        "Descargar CSV de ejemplo",
        example().to_csv(index=False).encode(),
        "ejemplo_mensual.csv",
        "text/csv",
        on_click="ignore",
    )
try:
    if uploaded is not None and not demo:
        raw = read_csv(
            uploaded.getvalue(),
            None
            if separator == "Detectar"
            else "\t"
            if separator == "Tabulación"
            else separator,
            decimal,
            encoding,
        )
    elif demo:
        raw = example()
    else:
        st.write("Carga un archivo o activa el ejemplo para comenzar.")
        st.stop()
except Exception as exc:
    st.error(f"No se pudo leer el CSV: {exc}")
    st.stop()

st.subheader("2 · Revisa e interpreta el archivo")
st.dataframe(raw.head(30), width="stretch")
a, b, c = st.columns(3)
a.metric("Filas", len(raw))
b.metric("Filas repetidas", int(raw.duplicated().sum()))
c.metric("Celdas vacías", int(raw.isna().sum().sum()))
work = raw.copy()
remove = st.checkbox("Eliminar filas completamente idénticas", value=False)
if remove:
    work = work.drop_duplicates()
group = st.selectbox(
    "Columna para filtrar un grupo (opcional)", ["Sin filtro"] + list(work.columns)
)
if group != "Sin filtro":
    options = work[group].dropna().unique().tolist()
    if not options:
        st.error("La columna de grupos está vacía.")
        st.stop()
    selected = st.selectbox("Grupo", options)
    work = work[work[group] == selected]
numeric = work.select_dtypes(include="number").columns.tolist()
value = st.selectbox(
    "Variable numérica",
    list(work.columns),
    index=list(work.columns).index(numeric[0]) if numeric else 0,
)
x = pd.to_numeric(work[value], errors="coerce")
if (work[value].notna() & (x.isna() | ~np.isfinite(x))).any():
    st.error(
        "La variable contiene texto o valores infinitos. Revisa la columna y el separador decimal."
    )
    st.stop()

summary, temporal = st.tabs(
    ["Estadísticas y distribución", "Serie temporal y pronóstico"]
)
with summary:
    left, right = st.columns(2)
    left.dataframe(descriptive(x), width="stretch")
    if x.notna().any():
        right.plotly_chart(
            px.histogram(
                x.dropna(), x=value, nbins=25, title="Distribución de los valores"
            ),
            width="stretch",
        )
        st.plotly_chart(
            px.box(x.dropna(), y=value, title="Dispersión y posibles valores atípicos"),
            width="stretch",
        )
    st.caption(
        "La desviación estándar y la varianza son muestrales. Los puntos extremos se conservan: pueden ser observaciones válidas."
    )
    st.download_button(
        "Descargar tabla filtrada",
        work.to_csv(index=False).encode(),
        "tabla_limpia.csv",
        "text/csv",
        on_click="ignore",
    )
with temporal:
    date = st.selectbox(
        "Columna de fechas",
        ["Crear calendario"] + list(work.columns),
        index=1 if work.columns[0] != value else 0,
    )
    dayfirst = st.checkbox("Fechas ambiguas: día antes que mes", value=True)
    suggestion = None
    if date != "Crear calendario":
        try:
            dates = (
                parse_dates(work[date], dayfirst)
                .dropna()
                .sort_values()
                .drop_duplicates()
            )
            if len(dates) >= 3:
                suggestion = pd.infer_freq(dates)
        except (ValueError, TypeError):
            pass
    st.caption(
        f"Frecuencia detectada: {suggestion or 'no concluyente'}. Confirma la frecuencia; los huecos pueden impedir detectarla."
    )
    default = "Mensual"
    if suggestion:
        if suggestion.startswith(("Q", "QE", "QS")):
            default = "Trimestral"
        elif suggestion.startswith(("Y", "A")):
            default = "Anual"
        elif suggestion.startswith("W"):
            default = "Semanal (domingo)"
        elif suggestion == "D":
            default = "Diaria"
        elif suggestion == "B":
            default = "Días hábiles (lunes–viernes)"
    frequency = st.selectbox(
        "Frecuencia del calendario",
        list(FREQUENCIES),
        index=list(FREQUENCIES).index(default),
    )
    start = st.text_input("Fecha inicial (si creas calendario)", "2020-01-01")
    st.caption(
        "También puedes escribir fechas como 2020-01-01, 2020-04-01… para trimestres. El calendario hábil no incorpora festivos. No se admite aún frecuencia intradía."
    )
    aggregate = st.selectbox(
        "Varias filas por período", ["Rechazar", "Sumar", "Promediar"]
    )
    try:
        original = prepare_series(
            work,
            value,
            None if date == "Crear calendario" else date,
            FREQUENCIES[frequency],
            start,
            aggregate,
            dayfirst,
        )
    except Exception as exc:
        st.warning(str(exc))
        st.stop()
    method = st.selectbox(
        "Tratamiento opcional de faltantes",
        [
            "Sin rellenar",
            "Interpolación lineal (interior)",
            "Último valor (solo hacia adelante)",
        ],
    )
    gap = st.number_input("Longitud máxima del hueco que se puede rellenar", 1, 24, 3)
    clean = fill_missing(original, method, int(gap))
    imputed = original.isna() & clean.notna()
    st.write(
        f"Calendario: {len(original)} períodos · Faltantes originales: {original.isna().sum()} · Estimados: {imputed.sum()} · Pendientes: {clean.isna().sum()}"
    )
    st.caption(
        "La interpolación rellena huecos interiores. El último valor prolonga el nivel conocido y puede rellenar el extremo final. Las proyecciones futuras se calculan aparte. Los huecos mayores al límite permanecen vacíos."
    )
    audit = pd.DataFrame(
        {
            "original": original.to_numpy(),
            "valor": clean.to_numpy(),
            "estimado": imputed.to_numpy(),
        },
        index=original.index.astype(str),
    )
    st.plotly_chart(
        px.line(audit, y=["original", "valor"], title="Serie original y preparada"),
        width="stretch",
    )
    st.download_button(
        "Descargar serie y marcas de estimación",
        audit.to_csv(index_label="periodo").encode(),
        "serie_preparada.csv",
        "text/csv",
        on_click="ignore",
    )
    st.subheader("Pruebas estadísticas")
    st.dataframe(diagnostics(clean), width="stretch")
    st.caption(
        "No rechazar una hipótesis no demuestra que sea verdadera. ADF y KPSS usan constante; tendencias deterministas y cambios estructurales requieren análisis adicional. ARIMA no exige que la serie original tenga distribución normal. Las pruebas sobre valores imputados requieren cautela."
    )
    if not clean.isna().any() and len(clean) > 8 and clean.nunique() > 1:
        correlations = acf(clean, nlags=min(36, len(clean) // 3))
        st.plotly_chart(
            px.bar(
                x=np.arange(len(correlations)),
                y=correlations,
                labels={"x": "Rezago", "y": "ACF"},
                title="Autocorrelación",
            ),
            width="stretch",
        )
        partial = pacf(clean, nlags=min(36, len(clean) // 3), method="ywm")
        st.plotly_chart(
            px.bar(
                x=np.arange(len(partial)),
                y=partial,
                labels={"x": "Rezago", "y": "PACF"},
                title="Autocorrelación parcial",
            ),
            width="stretch",
        )
    season_default = {
        "Mensual": 12,
        "Trimestral": 4,
        "Anual": 1,
        "Semanal (domingo)": 52,
        "Diaria": 7,
        "Días hábiles (lunes–viernes)": 5,
    }[frequency]
    season = int(
        st.number_input("Períodos por ciclo estacional", 1, 366, season_default)
    )
    st.caption(
        "Mensual: 12; trimestral: 4; diaria: 7 si hay patrón semanal. Esta versión modela un solo ciclo. SARIMA automático se limita a ciclos de hasta 24 períodos por costo de cálculo."
    )
    st.subheader("Proyección")
    mode = st.selectbox("Método", ["Automático", "ARIMA / SARIMA", "Tendencia lineal"])
    horizon = int(st.number_input("Períodos futuros", 1, 60, 12))
    st.caption(
        "Búsqueda acotada de ARIMA/SARIMA, comparada por MAE en un bloque temporal. Un segundo bloque evalúa al ganador. AICc se informa como diagnóstico; no decide la selección. Los modelos sin convergencia se descartan."
    )
    if st.button("Calcular pronóstico", type="primary"):
        try:
            with st.spinner("Comparando modelos y evaluando el pronóstico…"):
                out, ranking, metrics, winner, lb = forecast_series(
                    original, horizon, season, mode, method, int(gap)
                )
            st.success(f"Modelo elegido: {winner}")
            st.json(metrics)
            fig = go.Figure()
            fig.add_scatter(x=audit.index, y=clean, mode="lines", name="Historial")
            fig.add_scatter(
                x=out.index,
                y=out["Superior 95 %"],
                mode="lines",
                line={"width": 0},
                name="Superior 95 %",
            )
            fig.add_scatter(
                x=out.index,
                y=out["Inferior 95 %"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                name="Intervalo 95 %",
            )
            fig.add_scatter(
                x=out.index, y=out["Pronóstico"], mode="lines", name="Pronóstico"
            )
            st.plotly_chart(fig, width="stretch")
            st.dataframe(out, width="stretch")
            st.dataframe(ranking, width="stretch")
            st.write(
                "Ljung–Box sobre residuos finales (exploratorio, sin ajuste por parámetros estimados):"
            )
            st.dataframe(lb)
            st.caption(
                "Intervalos basados en los supuestos del modelo; en referencias y tendencia son aproximaciones gaussianas. No incorporan incertidumbre por selección o imputación. La evaluación cubre un solo bloque final y no garantiza el desempeño futuro."
            )
            st.download_button(
                "Descargar pronóstico",
                out.to_csv(index_label="periodo").encode(),
                "pronostico.csv",
                "text/csv",
                on_click="ignore",
            )
            report = {
                "archivo_sha256": hashlib.sha256(
                    raw.to_csv(index=False).encode()
                ).hexdigest(),
                "variable": value,
                "columna_fecha": date,
                "fecha_inicial": start,
                "dia_primero": dayfirst,
                "grupo_columna": group,
                "grupo_valor": str(selected) if group != "Sin filtro" else None,
                "separador": separator,
                "decimal": decimal,
                "codificacion": encoding,
                "modelo": winner,
                "metricas": metrics,
                "frecuencia": frequency,
                "ciclo": season,
                "horizonte": horizon,
                "faltantes": method,
                "hueco_maximo": gap,
                "filas_duplicadas_eliminadas": remove,
                "agregacion": aggregate,
                "candidatos": json.loads(ranking.to_json(orient="records")),
            }
            st.download_button(
                "Descargar informe reproducible",
                json.dumps(report, ensure_ascii=False, indent=2).encode(),
                "informe.json",
                "application/json",
                on_click="ignore",
            )
        except Exception as exc:
            st.error(f"No se pudo pronosticar: {exc}")
    st.subheader("Ajuste estacional")
    adjustment = st.selectbox("Procedimiento", ["STL", "X-13ARIMA-SEATS (X-11)"])
    st.caption(
        "STL es una descomposición aditiva. X-13 se ejecuta directamente con descomposición X-11 y preajuste ARIMA automático; esta versión no reproduce especificaciones de INEGI ni incorpora su calendario de festivos. Admite meses y trimestres calendario, con al menos cinco años. Los resultados pueden revisarse al añadir datos."
    )
    if st.button("Calcular ajuste estacional"):
        try:
            with st.spinner("Calculando componentes…"):
                adjusted, report = seasonal_adjust(
                    clean, "STL" if adjustment == "STL" else "X-13", season
                )
            st.plotly_chart(
                px.line(adjusted, y=["Original", "Ajustada", "Tendencia"]),
                width="stretch",
            )
            st.dataframe(adjusted, width="stretch")
            st.download_button(
                "Descargar componentes",
                adjusted.to_csv(index_label="periodo").encode(),
                "ajuste_estacional.csv",
                "text/csv",
                on_click="ignore",
            )
            st.download_button(
                "Descargar especificación y diagnóstico",
                report.encode(),
                "diagnostico_estacional.txt",
                "text/plain",
                on_click="ignore",
            )
        except Exception as exc:
            st.error(f"No se pudo ajustar: {exc}")
