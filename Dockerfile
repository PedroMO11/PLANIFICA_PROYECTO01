# Imagen de SERVING. No incluye entrenamiento, datos crudos ni credenciales.
#
# Contrato de Cloud Run que esta imagen respeta:
#   - arquitectura linux/amd64 (construir con --platform linux/amd64)
#   - escucha en 0.0.0.0 y en el puerto de $PORT, 8080 por defecto
#   - sin estado durable dentro del contenedor: el ledger vive en el replay local
#
# Construccion:
#   docker build --platform linux/amd64 -t fraud-adaptive-serving .
# Ejecucion local:
#   docker run --rm -p 8080:8080 -v "$PWD/models/W30_T165:/app/package:ro" fraud-adaptive-serving

FROM python:3.12-slim-bookworm AS base

# libgomp es la dependencia nativa de LightGBM; sin ella el import falla en runtime,
# no en build, que es el peor momento para descubrirlo.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    PACKAGE_DIR=/app/package

WORKDIR /app

# Las dependencias se instalan antes de copiar el codigo para que un cambio en
# src/ no invalide la capa de dependencias.
COPY requirements-serving.lock ./
RUN pip install --no-cache-dir -r requirements-serving.lock

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY configs/ ./configs/
RUN pip install --no-cache-dir --no-deps -e .

# Usuario sin privilegios: un proceso que solo sirve inferencia no necesita root.
RUN useradd --create-home --uid 10001 serving \
    && chown -R serving:serving /app
USER serving

EXPOSE 8080

# Sin paquete montado el servicio arranca igual y responde 503 en /predict (C21):
# es preferible a fallar en el arranque y no poder diagnosticar por /health.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,os,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8080')+'/health').status==200 else 1)"

# sh -c para que $PORT se expanda en tiempo de ejecucion, no de build.
CMD ["sh", "-c", "python -m uvicorn --factory 'fraud_adaptive.serving:create_app' --host 0.0.0.0 --port ${PORT:-8080}"]
