# Serie Clara

Aplicación de código abierto en español para explorar CSV, preparar series temporales, comparar pronósticos y ajustar estacionalidad. Incluye un ejemplo de ventas mensuales para probarla sin cargar datos.

## Inicio rápido

Probada en Linux x86_64 con Python 3.12. El ejecutable X-13 se instala con `x13binary`; la disponibilidad en otras plataformas depende de sus distribuciones.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Abre la dirección que muestre Streamlit en tu equipo. En el entorno de onboarding, las comprobaciones se realizan internamente y no hay una vista pública. No necesitas claves de API.

## Cómo usarla

1. Carga un CSV o utiliza el ejemplo incorporado. Confirma separador, decimal y codificación.
2. Revisa las filas. Opcionalmente elimina filas idénticas y filtra un grupo.
3. Escoge la variable numérica para ver estadísticas y distribución.
4. En la pestaña temporal, selecciona fechas o crea un calendario. Confirma frecuencia y regla de agregación.
5. Decide si rellenas faltantes y el tamaño máximo del hueco. Descarga la serie con las estimaciones marcadas.
6. Define ciclo y horizonte, calcula el pronóstico y descarga resultados e informe.
7. Si procede, aplica STL o X-13 y descarga componentes y diagnóstico.

Las fechas ISO (`2024-03-01`) se interpretan como año-mes-día. Para fechas locales ambiguas existe una opción día/mes. Para trimestres usa una fecha dentro de cada trimestre. Suma flujos (ventas) o promedia niveles solo cuando tenga sentido; la herramienta no decide esa semántica por ti.

## Funciones y métodos

- CSV hasta 10 MB y 100 000 filas; codificaciones UTF-8, Latin-1 y UTF-16.
- Media, mediana, desviación estándar y varianza muestrales, cuantiles, IQR, asimetría y curtosis.
- Histograma, caja, serie temporal y autocorrelación.
- Shapiro–Wilk (8 a 5 000 valores), ADF y KPSS con constante. No se comprime el calendario para aplicar pruebas de estacionariedad a datos incompletos.
- Frecuencias diaria, hábil lunes–viernes, semanal con cierre domingo, mensual, trimestral calendario y anual. La frecuencia inferida requiere confirmación.
- Limpieza opcional, agregación explícita y relleno de huecos cortos. Interpolación interior o último valor hacia adelante; no se extrapola el extremo inicial.
- Referencias de último valor y estacional, tendencia lineal y búsqueda acotada de ARIMA/SARIMA.
- STL aditivo robusto, con al menos tres ciclos.
- X-13ARIMA-SEATS 1.1 build 61: motor real, transformación automática, identificación ARIMA y detección de atípicos, descomposición **X-11**. Se requieren cinco años mensuales o trimestrales. No implementa aún la opción de descomposición SEATS ni calendarios festivos específicos.

### Evaluación del pronóstico

Se reserva un bloque final de tamaño `min(horizonte, max(3, n//5))`. Del historial anterior se reserva otro bloque de tamaño `min(bloque_final, max(3, historial//5))` para seleccionar por MAE. Los candidatos se ajustan al segmento previo; cada tratamiento de faltantes se calcula exclusivamente dentro del entrenamiento. Se evalúa solo contra valores realmente observados. Se necesitan al menos tres observaciones en cada bloque y doce períodos de entrenamiento.

El candidato ganador se evalúa en el bloque final y luego se reajusta a todo el historial. Se informan MAE, RMSE y MASE (indefinida cuando su denominador es cero), AICc por candidato y Ljung–Box exploratorio sobre residuos finales, sin ajuste por parámetros. Un resultado de prueba no garantiza validez del modelo.

La búsqueda considera `(p,d,q)` con `d` en 0–1 y `(p,q)` en (0,0), (1,0), (0,1), (1,1). Para SARIMA usa `(P,D,Q)=(1,1,0)`, `p` y `d` en 0–1 y `q=0`, con al menos tres ciclos de entrenamiento. No convergentes se descartan. «Automático» compara también referencias y tendencia; «ARIMA / SARIMA» limita la familia. No equivale a una búsqueda exhaustiva del mejor modelo posible.

Límites de pronóstico: 24 a 1 500 períodos, horizonte 1–60; SARIMA hasta ciclo 24. Series diarias con múltiples estacionalidades, semanales con ciclos anuales variables y datos intradía necesitan otros métodos. La estacionalidad no se detecta automáticamente: se propone un ciclo que el usuario puede cambiar.

Los intervalos del 95 % de ARIMA/SARIMA son condicionales al modelo. Las referencias y la tendencia usan aproximaciones gaussianas. No incluyen incertidumbre por selección, imputación o cambios estructurales. La evaluación temporal usa un solo bloque final; no implementa aún validación de origen móvil.

### Interpretación del ajuste estacional

Usar X-13 no reproduce las cifras de INEGI: hacen falta sus especificaciones y tratamientos de calendario. X-13 puede seleccionar transformación logarítmica; el componente irregular puede ser multiplicativo. El informe conserva especificación, salida del programa y diagnósticos HTML. Los componentes se pueden revisar al incorporar observaciones nuevas.

## Privacidad

Streamlit procesa los archivos en la memoria del servidor donde se ejecuta. La aplicación no los escribe en disco ni los envía a servicios de IA. X-13 recibe la serie seleccionada en un directorio temporal que se elimina al terminar. El ejemplo sintético es el único dato cacheado. Para información confidencial, utiliza una instalación local. El despliegue público necesita una revisión adicional de acceso y recursos.

## Desarrollo y pruebas

```bash
python -m pytest -q
```

Las pruebas cubren lectura decimal, duplicados por período, preservación de faltantes, fechas ISO, límites del relleno, separación temporal, pronóstico, reconstrucción STL, ejecución real de X-13 mensual/trimestral y recorrido de interfaz. Dependencias fijadas en `requirements.txt`.

Archivos principales: `app.py` (interfaz), `analysis_core.py` (cálculos), `tests/` (validación). No se incluyen datos personales ni credenciales.

## Próximas mejoras

- Validación temporal con varios orígenes y más modelos de referencia.
- Calendarios personalizados, múltiples ciclos y selección de términos deterministas.
- Opciones SEATS, efectos de calendario y diagnóstico de revisiones.
- Informes más accesibles y conservación de resultados entre cambios de controles.

Consulta `CONTRIBUTING.md` para colaborar. Licencia MIT para el código propio; las dependencias y el ejecutable X-13 conservan sus licencias respectivas.
