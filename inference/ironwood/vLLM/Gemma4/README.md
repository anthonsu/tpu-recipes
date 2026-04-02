# Serve Gemma 4 IT with vLLM on Ironwood TPU

In this guide, we show how to serve Gemma 4 IT models (e.g., `google/gemma-4-31B-it`) with vLLM on Ironwood (TPU v7x) using GKE.

> Note: These setup instructions are **specifically for Gemma 4** on TPU and may not work for other models as it uses custom wheels and source builds for vllm and transformers.

## Verified Models

The following larger Gemma 4 models are verified for deployment on TPU.

### Verified Models

| Model | Parameters | Min TPUs (Chips) | HuggingFace |
| :---- | :---- | :---- | :---- |
| Gemma 4 31B IT | 31B | 4× | [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) |
| Gemma 4 26B-A4B IT (MoE) | 26B (4B active) | 4× | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) |



### Models Not Yet Verified for TPU

The smaller models are currently not verified for TPU deployment:
- Gemma 4 E2B IT
- Gemma 4 E4B IT

## Cluster Prerequisites

Before deploying the vLLM workload, ensure your GKE cluster is configured with the necessary networking and identity features.

### Define parameters

```bash
# Set variables
export CLUSTER_NAME=<YOUR_CLUSTER_NAME>
export PROJECT_ID=<YOUR_PROJECT_ID>
export REGION=<YOUR_REGION>
export ZONE=<YOUR_ZONE>
export NODEPOOL_NAME=<YOUR_NODEPOOL_NAME>
```

### Create nodepool

Create a TPU v7 (Ironwood) nodepool.

```bash
gcloud container node-pools create ${NODEPOOL_NAME} \
  --project=${PROJECT_ID} \
  --location=${REGION} \
  --node-locations=${ZONE} \
  --num-nodes=1 \
  --machine-type=tpu7x-standard-4t \
  --cluster=${CLUSTER_NAME}
```

## Deploy vLLM Workload on GKE

1. Configure kubectl to communicate with your cluster

    ```bash
    gcloud container clusters get-credentials ${CLUSTER_NAME} --location=${ZONE}
    ```

2. Create a Kubernetes Secret for Hugging Face credentials

    ```bash
    export HF_TOKEN=YOUR_TOKEN
    kubectl create secret generic hf-secret \
        --from-literal=hf_api_token=${HF_TOKEN}
    ```

3. Apply the vLLM manifest (you can use the provided `example.yaml` file in this directory or copy the YAML below into `vllm-tpu.yaml`):

> [!TIP]
> If you are using pre-built images optimized for Gemma 4 (which have the required custom wheels baked in), the environment variables for vision stability (`VLLM_WORKER_MULTIPROC_METHOD=fork`, etc.) are already set inside the image. If you are using a standard image, you must define them in the `env:` section of the container spec.

    ```yaml
    apiVersion: storage.k8s.io/v1
    kind: StorageClass
    metadata:
      name: hyperdisk-balanced-tpu
    provisioner: pd.csi.storage.gke.io
    parameters:
      type: hyperdisk-balanced
    reclaimPolicy: Delete
    volumeBindingMode: WaitForFirstConsumer
    allowVolumeExpansion: true
    ---
    apiVersion: v1
    kind: PersistentVolumeClaim
    metadata:
      name: hd-claim
    spec:
      storageClassName: hyperdisk-balanced-tpu
      accessModes:
        - ReadWriteOnce
      resources:
        requests:
          storage: 1000Gi
    ---
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: vllm-tpu
    spec:
      replicas: 1
      selector:
        matchLabels:
          app: vllm-tpu
      template:
        metadata:
          labels:
            app: vllm-tpu
        spec:
          nodeSelector:
            cloud.google.com/gke-tpu-accelerator: tpu7x
            cloud.google.com/gke-tpu-topology: 2x2x1
          containers:
          - name: vllm-tpu
            image: vllm/vllm-tpu:gemma4
            command: ["python3", "-m", "vllm.entrypoints.openai.api_server"]

            args:
            - --host=0.0.0.0
            - --port=8000
            - --seed=42
            - --tensor-parallel-size=8
            - --max-model-len=16384
            - --download-dir=/data
            - --no-enable-prefix-caching
            - --model=google/gemma-4-31B-it
            - --async-scheduling
            - --gpu-memory-utilization=0.90
            - --disable_chunked_mm_input
            - --enable-auto-tool-choice
            - --tool-call-parser=gemma4

            env:
            - name: HF_HOME
              value: /data
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-secret
                  key: hf_api_token
            - name: MODEL_IMPL_TYPE
              value: vllm
            ports:
            - containerPort: 8000
            resources:
              limits:
                google.com/tpu: '4' # Number of chips required by topology
              requests:
                google.com/tpu: '4'
            readinessProbe:
              tcpSocket:
                port: 8000
              initialDelaySeconds: 15
              periodSeconds: 10
            volumeMounts:
            - mountPath: "/data"
              name: data-volume
            - mountPath: /dev/shm
              name: dshm
          volumes:
          - emptyDir:
              medium: Memory
            name: dshm
          - name: data-volume
            persistentVolumeClaim:
              claimName: hd-claim
    ```

4. Apply the vLLM manifest

    ```bash
    kubectl apply -f vllm-tpu.yaml
    ```

5. Interact with the model using curl

    ```bash
    kubectl port-forward service/vllm-service 8000:8000

    curl http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "google/gemma-4-31B-it",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://images.unsplash.com/photo-1501594907352-04cda38ebc29?auto=format&fit=crop&w=800&q=80"
                            }
                        },
                        {
                            "type": "text", 
                            "text": "Describe the image above"
                        }
                    ]
                }
            ],
            "max_tokens": 300,
            "temperature": 0.0,
            "top_p": 1.0
        }'
    ```
