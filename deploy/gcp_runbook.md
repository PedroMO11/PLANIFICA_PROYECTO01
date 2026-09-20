# Runbook: despliegue manual en GCP

**Escrito para:** el integrante del equipo encargado del despliegue (rol E), con
acceso a un proyecto de GCP y permisos de Artifact Registry y Cloud Run.

**Alcance.** Ningún comando de este documento se ejecutó. No se creó ningún
recurso, no se estimó ninguna factura y no se ejecutó `gcloud` desde el
repositorio. Lo que sí está entregado y verificado localmente: el paquete del
modelo, el `Dockerfile`, el servicio HTTP, el replay con su ledger y las pruebas
de contrato.

**La entrega académica no depende de que esto se despliegue.** Si el despliegue
falla o no hay presupuesto, la demo local es suficiente y la propuesta de §3.4
queda cubierta por el diseño, este runbook y la evidencia local.

---

## 0. Antes de empezar

| Verificación | Cómo |
|---|---|
| Proyecto y región elegidos | Perfil documental: `us-central1` (ajustable) |
| Billing habilitado y límites revisados | Consola de GCP → Facturación. **Revisa tus propios límites**: este documento no estima costos |
| Permisos | `roles/artifactregistry.writer`, `roles/run.admin`, `roles/iam.serviceAccountUser` |
| Docker instalado y autenticado | `docker --version` |
| Paquete del modelo disponible | `models/<version_id>/` con `manifest.json` |

Sustituye en todos los comandos:

```bash
PROJECT_ID=<tu-project-id>
REGION=us-central1
REPO=fraud-adaptive
SERVICE=fraud-adaptive-serving
VERSION=<version_id, p. ej. W30_T165>
```

---

## 1. Verificar el paquete localmente (antes de subir nada)

```bash
# El servicio arranca y reporta la versión activa
python -m fraud_adaptive serve --package models/$VERSION --port 8080 &
curl -s localhost:8080/health | python -m json.tool
```

`/health` debe devolver `status: ok`, el `model_version` esperado y el
`package_hash`. Si devuelve `model_unavailable`, **no continúes**: el paquete está
incompleto o corrupto.

```bash
# Contrato completo, cupo, idempotencia y rollback
python -m pytest tests/test_serving_contract.py tests/test_replay_idempotency.py -q
```

---

## 2. Construir y subir la imagen

```bash
gcloud artifacts repositories create $REPO \
  --repository-format=docker --location=$REGION \
  --description="Serving del sistema adaptativo de fraude"

gcloud auth configure-docker $REGION-docker.pkg.dev

IMAGE=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/serving

# --platform es obligatorio: Cloud Run solo ejecuta linux/amd64.
# En un Mac con Apple Silicon, omitirlo produce una imagen arm64 que falla al arrancar.
docker build --platform linux/amd64 -t $IMAGE:$VERSION .
docker push $IMAGE:$VERSION

# Anota el digest: fija la imagen y hace el despliegue reproducible.
docker inspect --format='{{index .RepoDigests 0}}' $IMAGE:$VERSION
```

**Verificación manual:** el digest anotado debe coincidir con el que muestre
Cloud Run tras el despliegue. Las credenciales no viajan en la imagen: el
`.dockerignore` excluye `.env`, `~/.kaggle` y los datos crudos.

---

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

**Por qué esta configuración y no otra:**

- `--no-allow-unauthenticated` e `--ingress=internal`: el endpoint decide sobre
  pagos. Nunca debe ser público.
- `--max-instances=1` y `--concurrency=1`: la demo es **secuencial**. El cupo lo
  garantiza el ledger local del replay, no el servicio. Con varias instancias, dos
  peticiones concurrentes podrían sobreadmitir revisiones.
- `--min-instances=0`: sin tráfico no se paga cómputo, a costa de cold start. El
  cold start se mide y se reporta **aparte** de la latencia warm.

Verificación:

```bash
gcloud run services describe $SERVICE --region=$REGION --format='value(status.url)'
TOKEN=$(gcloud auth print-identity-token)
curl -s -H "Authorization: Bearer $TOKEN" $URL/health | python -m json.tool
```

---

## 4. Ejecutar el replay local contra el endpoint remoto

El replay sigue siendo la **autoridad del estado**, aunque el scoring ocurra en la
nube. El ledger vive en tu máquina, no en el contenedor.

```bash
python -m fraud_adaptive replay \
  --package models/$VERSION \
  --endpoint $URL \
  --max-events 5000 \
  --fixtures
```

**Verificación manual:**

- `ledger.excedio_capacidad` = `false`
- `idempotencia.aprobado` = `true`
- las tres acciones aparecen en `acciones` (o los fixtures cubren las faltantes,
  claramente rotulados)
- `reports/replay_local.md` se regenera con la evidencia

---

## 5. Promover o volver a la versión anterior

**Solo tras aprobación humana registrada.** Lee `deploy/promocion_rollback.md`
antes de este paso.

```bash
# Promoción: desplegar el nuevo digest
gcloud run deploy $SERVICE --image=$IMAGE@<digest-nuevo> --region=$REGION

# Rollback: devolver el tráfico a la revisión anterior
gcloud run services update-traffic $SERVICE --to-revisions=<revision-anterior>=100 --region=$REGION
```

**Verificación:** `/health` debe informar el `model_version` y el `package_hash`
esperados. No apruebes un modelo por la apariencia de la demo: la evidencia
económica exige etiquetas maduras, que llegan 30 días después.

---

## 6. Cerrar la demo

```bash
gcloud run services delete $SERVICE --region=$REGION
gcloud artifacts repositories delete $REPO --location=$REGION
```

Conserva localmente: `runs/<run_id>/`, `reports/` y `models/<version>/`. Revisa
tu propio consumo en la consola de facturación.

---

## 7. Desafíos de escala, costo e integración

Análisis cualitativo. **No se estiman montos**: dependen de la cuenta, la región y
el tráfico real del equipo.

| Desafío | Causa y factor de costo | Decisión de diseño / responsable |
|---|---|---|
| **Escala de serving** | CPU y RAM por petición, cold start, tamaño del paquete, número de réplicas | Entrenamiento separado del serving; tiempos y memoria medidos localmente; el equipo limita réplicas en la demo. Una prueba local **no** permite prometer un SLA Perú–región |
| **Reentrenamiento** | W, volumen, ancho del panel y lectura de datos determinan la duración | Ventana fija y cadencia de 15 días; se adjunta el costo de cómputo local medido. Migrar a Cloud Run Jobs es trabajo futuro y manual |
| **Almacenamiento y observabilidad** | Versiones retenidas y logs por evento crecen con el volumen | Dos paquetes activos como máximo y logs agregados de baja cardinalidad. **Sin series etiquetadas por `TransactionID`**. La retención la define el equipo |
| **Integración** | Identidad incompleta o tardía, duplicados, orden y paridad de features | Contratos explícitos y pruebas locales de paridad offline/API. El prototipo asume identidad **simultánea** y features precomputadas; producción exige validación adicional |
| **Concurrencia del cupo** | Varios emisores o reintentos pueden sobreadmitir revisiones | Un solo replay con ledger transaccional. El estado distribuido (Firestore) es diseño futuro. **No publicar un endpoint abierto de decisión con cupo arbitrario** |
| **Presupuesto** | Créditos, precios y cuotas dependen de la cuenta y la región | El equipo verifica sus límites antes de crear recursos. Si no despliega, la entrega local cubre el alcance. **No se afirma costo cero** |

---

## 8. Qué NO está incluido

- Terraform ni infraestructura como código
- Scripts de creación automática de recursos
- Pub/Sub, Firestore, BigQuery o Cloud Scheduler (solo en el diagrama)
- Vertex AI Endpoints (descartado para el MVP: configuración adicional sin
  evidencia nueva para la rúbrica)
- Estimación de factura mensual

Si Docker no está disponible en tu máquina, el servicio corre de forma nativa
(`python -m fraud_adaptive serve`). En ese caso el `Dockerfile` se entrega **sin
build verificado**, y el primer paso manual es construirlo.
