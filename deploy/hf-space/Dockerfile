# API de Órbita para Hugging Face Spaces (modo híbrido con E5).
# El Space solo contiene este Dockerfile y su README: el código se clona del repo público.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/tmp/hf ORBITA_HOST=0.0.0.0 ORBITA_API_PORT=7860 ORBITA_LAB=/app/eval

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* && useradd -m -u 1000 user

# Mismas versiones con que se calcularon los embeddings (torch CPU para no bajar CUDA).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 \
    && pip install joblib==1.5.3 numpy==2.0.2 scikit-learn==1.6.1 scipy==1.13.1 \
       sentence-transformers==5.1.2 transformers==4.57.6 huggingface_hub

# E5 dentro de la imagen: el Space arranca sin descargar 1 GB en cada reinicio.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('intfloat/multilingual-e5-base', local_dir='/opt/e5', allow_patterns=['*.json', '*.txt', 'model.safetensors', 'sentencepiece.bpe.model', '1_Pooling/*'])"

# Invalida la caché de esta capa cuando cambia main, para que "Factory rebuild" traiga lo último.
ARG REPO=DyAndrade-AI/restmex-orbita
ADD https://api.github.com/repos/${REPO}/commits/main /tmp/version.json
RUN git clone --depth 1 https://github.com/${REPO}.git /app \
    && rm -rf /app/.git /app/web /app/video /app/deploy \
    && mkdir -p /app/models && ln -s /opt/e5 /app/models/multilingual-e5-base \
    && chown -R user:user /app

USER user
WORKDIR /app
EXPOSE 7860
CMD ["python", "scripts/serve_api.py"]
