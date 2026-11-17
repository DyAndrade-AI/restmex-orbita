FROM python:3.11-slim

# Imagen léxica a propósito (sin torch ni el modelo de 1.1 GB); /health declara mode=léxico.

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ORBITA_HOST=0.0.0.0 \
    ORBITA_API_PORT=8765 \
    ORBITA_LABELS=/app/eval/labels/human_labels.jsonl \
    ORBITA_LAB=/app/eval/labels \
    ORBITA_SEMANTIC=0

WORKDIR /app

RUN groupadd --system orbita && useradd --system --gid orbita --home-dir /app orbita

COPY docker/requirements-api.txt ./requirements-api.txt
RUN python -m pip install --no-cache-dir -r requirements-api.txt

# Mismo código que se prueba localmente: no hay copias paralelas en docker/.
COPY src/restmex_recommender/ ./src/restmex_recommender/
COPY scripts/serve_api.py ./scripts/serve_api.py
COPY eval/gold_queries.json ./eval/gold_queries.json
# Semillas del laboratorio: al arrancar se copian al directorio de juicios (que puede ser un volumen).
COPY eval/query_bank.json eval/pools.json ./eval/seed/
COPY artifacts/vectorizer.joblib artifacts/lexical_profiles.npz artifacts/evidence_matrix.npz artifacts/town_term_rates.npz artifacts/bm25.npz artifacts/bm25.idf.npy artifacts/metadata.json artifacts/manifest.json ./artifacts/
COPY artifacts/evaluation_report.json ./artifacts/

RUN mkdir -p eval/labels && chown -R orbita:orbita /app
USER orbita

EXPOSE 8765

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
  CMD python -c "import json,urllib.request; json.load(urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=4))" || exit 1

# El banco de consultas se refresca en cada arranque; los pools congelados y los juicios no se pisan.
CMD ["sh", "-c", "mkdir -p \"$ORBITA_LAB\" && cp eval/seed/query_bank.json \"$ORBITA_LAB\"/ && cp -n eval/seed/pools.json \"$ORBITA_LAB\"/ ; exec python scripts/serve_api.py"]
