# Deployment and Scaling

This document describes the deployment and scaling strategies for the machine learning pipeline, including the Model Registry, Preprocessing Service, and Inference Service.

## Containerization

All services are containerized using Docker for consistent deployment across environments.

### Model Registry

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY model_registry /app/model_registry
COPY models /app/models
COPY requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

ENV NAME ModelRegistry

CMD ["uvicorn", "model_registry.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Preprocessing Service

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY preprocessing_service /app/preprocessing_service
COPY src /app/src
COPY data/raw/customer_purchases.csv /app/data/raw/
COPY requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8001

ENV NAME PreprocessingService

CMD ["uvicorn", "preprocessing_service.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

### Inference Service

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY inference_service /app/inference_service
COPY src /app/src
COPY data/raw /app/data/raw
COPY requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8002

ENV NAME InferenceService

CMD ["uvicorn", "inference_service.main:app", "--host", "0.0.0.0", "--port", "8002"]
```

## Kubernetes Deployment

The services are deployed on Kubernetes for orchestration and management.

### Persistent Volume Claims

Three Persistent Volume Claims (PVCs) are defined for data persistence:

1. `models-pvc`: For storing trained models (1Gi)
2. `model-registry-data-pvc`: For Model Registry data (1Gi)
3. `raw-data-pvc`: For raw data storage (1Gi)

### Deployments

1. Model Registry Deployment
   - Image: `model-registry:latest`
   - Port: 8000
   - Resource limits: CPU 100m-500m
   - Volumes: `models-pvc`, `model-registry-data-pvc`

2. Inference Service Deployment
   - Image: `inference-service:latest`
   - Port: 8002
   - Resource limits: CPU 100m-500m
   - Environment variables:
     - `MODEL_REGISTRY_URL`: `http://model-registry:8000`
     - `PREPROCESSING_SERVICE_URL`: `http://preprocessing-service:8001`
   - Volumes: `models-pvc`

3. Preprocessing Service Deployment
   - Image: `preprocessing-service:latest`
   - Port: 8001
   - Resource limits: CPU 100m-500m

### Services

NodePort services are created for each deployment to enable external access:

1. Model Registry Service: NodePort 30000
2. Inference Service: NodePort 30002
3. Preprocessing Service: NodePort 30001

## Horizontal Pod Autoscaling

Horizontal Pod Autoscalers (HPAs) are configured for each service to automatically scale based on CPU utilization:

1. Model Registry HPA
   - Min replicas: 2
   - Max replicas: 10
   - Target CPU utilization: 50%

2. Inference Service HPA
   - Min replicas: 2
   - Max replicas: 10
   - Target CPU utilization: 50%

3. Preprocessing Service HPA
   - Min replicas: 2
   - Max replicas: 10
   - Target CPU utilization: 50%

## Deployment Steps

1. Build Docker images for each service:

   ```
   docker build -t model-registry:latest -f Dockerfile.model_registry .
   docker build -t preprocessing-service:latest -f Dockerfile.preprocessing_service .
   docker build -t inference-service:latest -f Dockerfile.inference_service .
   ```

   or it is easier to use docker-compose:

   ```
   docker-compose build --no-cache
   ```

2. Apply Kubernetes manifests:

   ```
   kubectl apply -f k8s-manifests.yaml
   ```

3. Apply HPA configurations:

   ```
   kubectl apply -f hpa.yaml
   ```

4. Verify deployments:

   ```
   kubectl get deployments
   kubectl get services
   kubectl get hpa
   ```
