# Runbook: despliegue manual en GCP

Guía para el integrante encargado del despliegue, con acceso a un proyecto de GCP y
permisos de Artifact Registry y Cloud Run.

Los comandos `gcloud` de este documento no se ejecutaron: no hay recursos creados en
la nube ni estimaciones de factura. Lo que está entregado y verificado en local es el
paquete del modelo, el servicio HTTP, el replay con su ledger, las pruebas de contrato
y la imagen Docker, construida y ejecutada (`linux/amd64`, 1,06 GB, HEALTHCHECK en
verde, paridad con el cálculo offline y 503 sin paquete). El paso 2 repite un build
ya comprobado.

La propuesta de despliegue del enunciado (sección 3.4) queda cubierta por el diseño,
este runbook y la demo local, de modo que la entrega no depende de publicar el
servicio.

## 0. Requisitos

| Verificación | Cómo |
|---|---|
| Proyecto y región | Por defecto `us-central1` |
| Facturación habilitada y límites revisados | Consola de GCP, sección Facturación |
| Permisos | `roles/artifactregistry.writer`, `roles/run.admin`, `roles/iam.serviceAccountUser` |
| Docker instalado | `docker --version` |
| Paquete del modelo | `models/<version_id>/` con `manifest.json` |

Variables usadas en los comandos:

```bash
PROJECT_ID=<tu-project-id>
REGION=us-central1
REPO=fraud-adaptive
SERVICE=fraud-adaptive-serving
VERSION=<version_id, por ejemplo E15_T165>
```

El paquete se obtiene con `python -m fraud_adaptive --run-id v3 package export --strategy E15`.

## 1. Verificar el paquete en local

```bash
python -m fraud_adaptive serve --package models/$VERSION --port 8080 &
curl -s localhost:8080/health | python -m json.tool
```

`/health` debe devolver `status: ok`, el `model_version` esperado y el
`package_hash`. Si devuelve `model_unavailable`, el paquete está incompleto y hay que
regenerarlo antes de seguir.

```bash
python -m pytest tests/test_serving_contract.py tests/test_replay_idempotency.py -q
```

## 2. Construir y publicar la imagen

```bash
gcloud artifacts repositories create $REPO \
  --repository-format=docker --location=$REGION \
  --description="Serving del sistema adaptativo de fraude"

gcloud auth configure-docker $REGION-docker.pkg.dev

IMAGE=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/serving

# Cloud Run ejecuta linux/amd64; en un Mac con Apple Silicon el build por defecto es arm64.
docker build --platform linux/amd64 -t $IMAGE:$VERSION .
docker push $IMAGE:$VERSION

docker inspect --format='{{index .RepoDigests 0}}' $IMAGE:$VERSION
```

El digest fija la imagen y debe coincidir con el que muestre Cloud Run tras el
despliegue. `.dockerignore` excluye `.env`, `~/.kaggle` y los datos crudos, así que
las credenciales no viajan en la imagen.

## 3. Crear el servicio

```bash
gcloud run deploy $SERVICE \
  --image=$IMAGE@<digest> \
  --region=$REGION \
  --platform=managed \
  --no-allow-unauthenticated \
  --ingress=internal \
  --port=8080 \
  --cpu=2 --memory=4Gi \
  --min-instances=0 --max-instances=1 \
  --concurrency=1 \
  --timeout=60s \
  --set-env-vars=PACKAGE_DIR=/app/package,LOG_FORMAT=json
```

- `--no-allow-unauthenticated` e `--ingress=internal` mantienen privado un endpoint
  que decide sobre pagos.
- `--max-instances=1` y `--concurrency=1` corresponden a la demo secuencial: el cupo
  lo garantiza el ledger del replay, y con varias instancias dos peticiones
  concurrentes podrían admitir revisiones de más.
- `--min-instances=0` evita pagar cómputo sin tráfico a cambio de cold start, que se
  mide por separado de la latencia en caliente.

```bash
URL=$(gcloud run services describe $SERVICE --region=$REGION --format='value(status.url)')
TOKEN=$(gcloud auth print-identity-token)
curl -s -H "Authorization: Bearer $TOKEN" $URL/health | python -m json.tool
```

## 4. Replay local contra el endpoint remoto

El scoring ocurre en la nube y el ledger sigue en la máquina local, que conserva el
estado del cupo y de las decisiones.

```bash
python -m fraud_adaptive replay \
  --package models/$VERSION \
  --endpoint $URL \
  --max-events 5000 \
  --fixtures
```

Comprobar que `ledger.excedio_capacidad` es `false`, que `idempotencia.aprobado` es
`true`, que aparecen las tres acciones (o que los fixtures rotulados cubren las
faltantes) y que se regenera `reports/replay_local.md`.

## 5. Promover o revertir

Este paso requiere aprobación humana registrada; los criterios están en
`deploy/promocion_rollback.md`.

```bash
# Promoción
gcloud run deploy $SERVICE --image=$IMAGE@<digest-nuevo> --region=$REGION

# Rollback a la revisión anterior
gcloud run services update-traffic $SERVICE --to-revisions=<revision-anterior>=100 --region=$REGION
```

`/health` debe informar el `model_version` y el `package_hash` esperados. La evidencia
económica de una versión requiere etiquetas maduras, que llegan 30 días después, así
que la demo sirve para verificar el funcionamiento y no para aprobar un modelo.

## 6. Cerrar la demo

```bash
gcloud run services delete $SERVICE --region=$REGION
gcloud artifacts repositories delete $REPO --location=$REGION
```

Conservar en local `runs/<run_id>/`, `reports/` y `models/<version>/`, y revisar el
consumo en la consola de facturación.

## 7. Escala, costos e integración

Los montos dependen de la cuenta, la región y el tráfico, así que el análisis es
cualitativo.

| Desafío | Factores | Decisión de diseño |
|---|---|---|
| Escala del serving | CPU y memoria por petición, cold start, tamaño del paquete, réplicas | Entrenamiento separado del serving, tiempos medidos en local y réplicas limitadas en la demo. Una medición local no permite comprometer un SLA |
| Reentrenamiento | Tamaño de la ventana, volumen, ancho del panel y lectura de datos | Cadencia de 15 días; E15 crece con el histórico, lo que conviene vigilar. Cloud Run Jobs queda como migración futura |
| Almacenamiento y observabilidad | Versiones retenidas y logs por evento | Como máximo dos paquetes activos y logs agregados sin series por `TransactionID` |
| Integración | Identidad incompleta o tardía, duplicados, orden y paridad de features | Contratos explícitos y pruebas de paridad. El prototipo asume identidad simultánea y features precomputadas |
| Concurrencia del cupo | Varios emisores o reintentos pueden admitir revisiones de más | Un único replay con ledger transaccional; el estado distribuido en Firestore queda como diseño futuro |
| Presupuesto | Créditos, precios y cuotas de la cuenta | El equipo revisa sus límites antes de crear recursos |

## 8. Fuera del alcance

- Terraform u otra infraestructura como código.
- Scripts que creen recursos automáticamente.
- Pub/Sub, Firestore, BigQuery y Cloud Scheduler, que figuran solo en el diagrama.
- Vertex AI Endpoints, descartado para esta demo por requerir configuración adicional.
- Estimación de factura mensual.

Sin Docker, el servicio corre de forma nativa con `python -m fraud_adaptive serve` y
el contrato es el mismo.
