from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_demo_workflow():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(
        timeout=30
    )
    assert not app.exception and not app.error
    assert len(app.button) == 2
    next(w for w in app.selectbox if w.label == "Método").select("Tendencia lineal")
    app.button[0].click().run(timeout=30)
    assert not app.exception and not app.error
    assert any("Modelo elegido" in s.value for s in app.success)
    next(w for w in app.selectbox if w.label == "Procedimiento").select(
        "X-13ARIMA-SEATS (X-11)"
    )
    app.button[1].click().run(timeout=30)
    assert not app.exception and not app.error
