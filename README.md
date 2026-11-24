# Sistema de recomendación RestMex 2025

Aplicación local que recibe una preferencia turística en lenguaje natural y devuelve los cinco
Pueblos Mágicos más relacionados y los cinco menos relacionados. El índice se construye únicamente
con `data/Rest-Mex_2025_train_clean.csv`.

## Inicio rápido (para quien clona el repo)

Requisito: Docker Desktop. Los artefactos del índice ya vienen en `artifacts/`; no hace falta el
dataset ni el modelo E5.

```bash
git clone https://github.com/DyAndrade-AI/restmex-orbita.git
cd restmex-orbita
docker compose up --build -d
# abre http://localhost:4321  → “Ayúdanos a evaluar”
```

Para reconstruir el índice o correr el holdout necesitas el CSV de Rest-Mex 2025 (no se incluye por
licencia) en `data/Rest-Mex_2025_train_clean.csv`; `make build-semantic` descarga E5 a `models/`.

## Órbita: constelaciones de futuros

Además del ranking convencional, la interfaz incluye un modo experimental que trata una consulta
como varios **átomos de gusto**. Simula qué ocurriría si cada prioridad desapareciera y construye una
cartera de cinco futuros deliberadamente distintos:

- **Ancla:** la mayor afinidad directa.
- **Superviviente:** conserva una posición alta bajo cambios de prioridad.
- **Portal:** maximiza novedad con un umbral mínimo de plausibilidad.
- **Especialista:** lleva una faceta del viaje al extremo.
- **Comodín:** equilibra afinidad, resiliencia, cobertura y distancia.

El control de riesgo modifica cuánto puede alejarse el Portal. Para cada alternativa se resuelve un
contrafactual: qué porcentaje de atención tendría que trasladarse hacia una faceta para que esa
opción supere al Ancla. El proceso es determinista y sigue usando exclusivamente el train local.

## Validación sin humanos (usuarios simulados y sensibilidad)

`make validate-orbit` (≈3 min) responde las dos dudas obvias sobre lo experimental:

- **¿Sirven las preguntas de Strange Loop?** Con usuarios simulados que comparten la utilidad del
  modelo (caso optimista), el top 1 clásico acierta su destino ideal el 69 %, Strange Loop sin
  preguntas el 84 % y con 3 preguntas el 92 %. Cuando se rompe un supuesto (otra utilidad,
  respuestas con 30 % de error), la mejora con preguntas deja de ser significativa: su IC95
  incluye 0. Son usuarios simulados, no personas.
- **¿Dependen los roles de pesos elegidos a mano?** Con los 13 pesos perturbados ±50 % a la vez, la
  cartera de cinco futuros se conserva en un 90 % (Jaccard medio).

Las cifras exactas se generan desde el artefacto en
[docs/orbit_validation.md](docs/orbit_validation.md).
Detalle y límites en [docs/metodologia.md §8 bis](docs/metodologia.md). Después,
`make web-metrics` copia las cifras a `web/src/data/metrics.json`, que es lo que muestra la web.

## Web: video, evaluación y laboratorio ciego

- Dos videos: **2 min para todo público** (`web/public/media/orbita-2min.mp4`, el principal) y la
  versión **técnica de 7 min** con capítulos clicables (`orbita.mp4`). Fuentes en `video/`
  (`video/publico/` para el corto); se regeneran con `say` + Chrome headless + ffmpeg.
- La sección **Evaluación** muestra los cuatro niveles (regresión, holdout, usuarios simulados,
  personas reales) con sus límites.
- **Ayúdanos a evaluar** es el laboratorio ciego paso a paso: alias → consulta → un pueblo a la vez
  con cuatro botones grandes (o teclas 0–3) → revisión → enviar. El protocolo no cambia (pool
  completo, orden aleatorio por evaluador, escala 0–3, restricción); el conjunto por defecto es
  `test` y los demás quedan en “Opciones del equipo”.
- Producción en Railway (<https://web-production-d1f6e806.up.railway.app>): web + API léxica,
  definidas en `.railway/railway.ts`; los juicios se guardan en un volumen persistente
  (`judgments.jsonl`). Cada push a `main` despliega solo. Paso a paso y alternativas (Netlify,
  Hugging Face, Supabase) en [docs/despliegue.md](docs/despliegue.md).

## Documentación para la defensa

- [docs/metodologia.md](docs/metodologia.md): fórmulas, experimentos con IC95, Órbita, Strange
  Loop, convergencia de Monte Carlo y limitaciones.
- [docs/protocolo_evaluacion_humana.md](docs/protocolo_evaluacion_humana.md): cómo correr la
  evaluación humana ciega (laboratorio en la web, escala 0–3, particiones dev/calibration/test).
- `make experiments` y `make audit-multiverse` regeneran los resultados citados.

## Qué significa “5 mejores / 5 peores”

- **5 más afines (top):** los destinos cuyas reseñas del train mejor coinciden con la consulta.
- **5 menos afines (bottom):** los de menor afinidad temática. **No** son peores destinos: el
  bottom es exactamente el reverso del ranking completo y nunca se justifica con reseñas positivas.
- **Calidad percibida:** polaridad media con prior bayesiano (200 reseñas de la media global). Se
  muestra en cada tarjeta y **no entra al ranking**; así un pueblo con 700 reseñas no se castiga ni
  premia frente a uno con 45,000.
- **Verificación con corpus:** cada tarjeta indica qué % de reseñas del pueblo menciona cada
  término de la consulta frente a la mediana de los 40 pueblos. Quien evalúa puede comprobar a
  simple vista si una recomendación tiene respaldo.
- **Confianza:** si ninguna palabra existe en el vocabulario del train (modo léxico), la consulta
  se rechaza en lugar de devolver un orden arbitrario; si todo es exclusión (“sin playa, sin
  bosque”), también.

## Evaluación automática

| Fuente | Qué mide | Independencia |
|---|---|---|
| `eval/gold_queries.json` | 40 consultas con destinos aceptables y exclusiones | Seed heurístico; su autor vio resultados previos. Regresión, no evidencia no sesgada. Pendiente de adjudicación humana. |
| Holdout autosupervisado (`--holdout`) | Reseñas reservadas, fuera del índice, con nombres enmascarados, como consultas | No depende de consultas escritas a mano |
| Modo ciego en la web | 10 destinos mezclados (top + bottom), el evaluador marca 1/0 con el teclado | Juicio humano; se guarda con token HMAC del ranking mostrado |

```bash
make evaluate            # léxico: gold + popularidad/azar + ablación + robustez + adversarial
make evaluate-full       # además holdout sin fuga (reconstruye un índice en artifacts/holdout)
make evaluate-semantic   # híbrido con E5 (requiere que torch cargue)
make benchmark-api       # arranque en frío y p50/p95 reales por HTTP → artifacts/api_benchmark.json
make smoke-e2e           # API + Astro + flujo de etiquetas + accesibilidad básica
```

El reporte queda en `artifacts/evaluation_report.json` y en `docs/evaluation_report.md`. En la web,
la sección **Evaluación** ejecuta la rúbrica con un clic y ofrece el modo ciego: con 10 juicios por
consulta, una sesión toma unos 30–60 s en lugar de revisar listas a mano. Precision@5 se reporta
como hits/5; para consultas con pocos relevantes se usa además `capped_recall_at_5`
(hits / min(5, |R|)). El umbral de 0.70 solo se aplica a consultas con ≥5 relevantes.

## Arquitectura

El motor combina tres señales independientes:

1. **Perfil léxico por destino y categoría.** TF-IDF de palabras y bigramas sobre todas las reseñas.
   Cada pueblo tiene un perfil general y perfiles de atracciones, hoteles y restaurantes. La
   agregación es por promedio ponderado, así que Tulum no gana solo por tener más reseñas.
2. **Recuperación de evidencia.** Compara la consulta con una muestra estratificada de reseñas
   positivas de cada pueblo. Esta señal recupera actividades específicas que un promedio diluye.
3. **Perfil semántico multilingüe.** `multilingual-e5-base` entiende paráfrasis y consultas naturales.
   Sus pesos se guardan localmente para que la demostración no dependa de internet.

El ranking normaliza las tres señales entre los 40 candidatos y las combina. También detecta tipos
de intención (hospedaje, gastronomía o actividades), nombres de pueblos, regiones y exclusiones como
“sin playa”. Cada resultado muestra términos coincidentes y una reseña de respaldo del train.

## Instalación

Requiere Python 3.9 o posterior. Desde la raíz del proyecto:

```bash
make setup
make build-semantic
make test
make app
```

La primera construcción descarga el modelo semántico y lo conserva en `models/`. Después, tanto la
construcción como la aplicación pueden ejecutarse sin conexión. Para una instalación ligera sin el
modelo denso:

```bash
make build
make app
```

La interfaz queda disponible en `http://localhost:8501`.

## Frontend Astro

La experiencia web de Órbita vive en `web/` y consume el motor Python mediante una API local. Para
probarla, abre dos terminales desde la raíz del proyecto:

```bash
# Terminal 1: motor de recomendación
make web-api

# Terminal 2: frontend Astro
make web
```

Después visita `http://127.0.0.1:4321`. La página permite escribir una intención, ajustar el riesgo
creativo y abrir las cinco líneas temporales con afinidad, resiliencia, distancia, contrafactual y
evidencia del corpus. El build de producción se genera con `cd web && npm run build`.

### Strange Loop

El laboratorio Astro también ejecuta una simulación de 50,000 versiones posibles de las
preferencias. Cada versión asigna pesos distintos a los átomos de gusto y contiene variación de
experiencia mediante un modelo de elección probabilística reproducible. El frontend muestra:

- cuántos futuros convergen en cada destino;
- entropía de la decisión y líneas temporales sobrevivientes;
- la pregunta de sacrificio con mayor ganancia de información esperada;
- colapso interactivo del multiverso al responder;
- una trayectoria especulativa de tres viajes que representa cómo una experiencia podría cambiar
  la siguiente preferencia.

Estas trayectorias son herramientas de exploración y no predicciones psicológicas o causales.

## Docker

Dos servicios: Nginx sirve el frontend Astro y redirige `/api/*` al motor Python. **La imagen es
léxica a propósito** (`ORBITA_SEMANTIC=0`, sin torch ni el modelo de 1.1 GB) y fija las mismas
versiones de numpy/scipy/scikit-learn/joblib que generaron los artefactos; `/health` informa
`mode`, `semantic_status` y `compat_warnings`. Las etiquetas humanas persisten en el volumen
`orbita-labels`. La imagen usa el mismo `src/` que las pruebas: ya no hay copias en `docker/`.

```bash
docker compose up --build -d
```

Después abre `http://localhost:4321`. Para usar otro puerto público:

```bash
ORBITA_WEB_PORT=8080 docker compose up --build -d
```

Operación y diagnóstico:

```bash
docker compose ps
docker compose logs -f
docker compose down
```

El primer build descarga las dependencias de Python y construye Astro, por lo que puede tardar.

## API

Arranca primero en modo léxico (responde en segundos) y carga E5 en un hilo; cuando termina, las
respuestas cambian a `mode: híbrido`. Si la carga falla, `/health.semantic_status` muestra el motivo
y el léxico sigue sirviendo. Rutas: `GET /health`, `GET /api/gold`, `GET /api/evaluation`,
`POST /api/recommend`, `POST /api/evaluation/run` (409 si ya hay una en curso), `POST /api/labels`,
`POST /api/futures`, `POST /api/multiverse`. Los tokens de ranking caducan al reiniciar la API.

## Comandos de verificación

```bash
make demo       # tres consultas similares a una exposición
make evaluate   # rúbrica, métricas top-k, baselines y casos adversariales
make test       # pruebas unitarias y de integración
```

El archivo `artifacts/manifest.json` registra hash SHA-256 del CSV, número de filas, vocabulario,
semilla y versiones. Esto permite demostrar que no se usó ninguna partición distinta de train.

## Protocolo para la exposición

1. El día anterior, desconectar Wi-Fi y ejecutar `make demo`.
2. Iniciar `make app` y conservar una terminal abierta con el proceso.
3. Tener `make build` como contingencia: el modo léxico no requiere PyTorch ni el modelo semántico.
4. Explicar que “menos relacionado” significa menor similitud con la consulta, no menor calidad.
5. Guardar las tres consultas de evaluación y los resultados en una captura o en el reporte.

## Plan de trabajo hasta el 20/11/2026

- **Datos y evaluación:** auditoría del train, prevención de sesgo por volumen, diseño de consultas y
  registro de métricas.
- **Modelado:** perfiles TF-IDF, evidencia por reseña, embeddings E5, ajuste de pesos y análisis de
  errores.
- **Producto:** interfaz Streamlit, explicaciones, modo sin conexión y ensayo técnico.
- **Presentación:** narrativa del problema, arquitectura, resultados parciales, demostración y
  limitaciones.

Para el segundo avance del 13/11/2026 conviene mostrar el baseline léxico, tres consultas, el
manifiesto del train y un primer análisis de errores. La semana final debe dedicarse a calibrar pesos
con evaluación humana ciega y ensayar el arranque offline.

## Riesgo operativo: iCloud Drive

Si el proyecto vive en `~/Desktop` o `~/Documents` con iCloud “Optimizar almacenamiento”, macOS
puede desalojar archivos del `.venv`, `artifacts/` y `models/` (`ls -lO` muestra `dataless`). Cada
import los vuelve a descargar (minutos) y **sin Wi-Fi la aplicación no arranca**. Antes de la
exposición, mueve el proyecto fuera de las carpetas sincronizadas (por ejemplo `~/Proyectos`) o
márcalo como “Mantener descargado” en Finder, y repite `make benchmark-api` sin red.

Estado al 11/11/2026: el proyecto se movió a `~/Projects/SistemaRecomendacion` (con un enlace
simbólico en el Escritorio) y el `.venv` se recreó allí. Medido en Docker (léxico): arranque en
frío 2.0 s hasta `/health` y p95 de 0.42 s en 120 peticiones (`artifacts/api_benchmark_docker.json`).

## Límites conocidos

- El conjunto contiene reseñas, no juicios explícitos de relevancia consulta-destino. Por eso la
  métrica definitiva debe ser una evaluación humana ciega con consultas representativas.
- Los puntajes se interpretan dentro de cada consulta; no son probabilidades comparables entre
  consultas.
- Una reseña puede describir una mala experiencia. El constructor aumenta el peso de polaridades 4 y
  5, pero conserva algo de información de las demás para no perder temas.

