# Qwen3-Embedding-8B E2E Correctness Validation on TPUs

This guide provides step-by-step instructions for running end-to-end correctness validation of the **Qwen3-Embedding-8B** model on Google Cloud TPUs. The validation process ensures numerical parity between TPU and CPU baselines, specifically for long-context workloads.

## 1. Resource Provisioning

Choose the provisioning command corresponding to your target hardware:

### For v7x (Ironwood)
```bash
gcloud alpha compute tpus queued-resources create <RESOURCE_ID> \
    --node-id <NODE_ID> \
    --project <PROJECT_ID> \
    --zone <ZONE> \
    --accelerator-type v7x-8 \
    --runtime-version v2-alpha-tpuv7 \
    --reserved
```

### For v6e (Trillium)
```bash
gcloud compute tpus tpu-vm create <NODE_ID> \
    --project <PROJECT_ID> \
    --zone <ZONE> \
    --accelerator-type v6e-4 \
    --version v2-alpha-tpuv6e
```

## 2. Environment Setup
After connecting to your TPU VM, set up the environment. Select either Option A (prebuilt installation) or Option B (compiling from source).

```bash
# Set up uv and activate venv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv embed_env --python 3.12
source embed_env/bin/activate
```

**Option A:** Install official stable release

```bash
uv pip install vllm-tpu
```

**Option B:** Install from source code

```bash
# Clone the official tpu-inference repository
git clone https://github.com/vllm-project/tpu-inference.git
cd tpu-inference

# Sync to the validated LKG version of vLLM containing the StepPool chunked prefill fix
VLLM_COMMIT=$(cat .buildkite/vllm_lkg.version)
cd ..

# Install vLLM from the specified validated LKG commit
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout $VLLM_COMMIT
VLLM_TARGET_DEVICE="tpu" uv pip install -e .
cd ..

# Finalize tpu-inference installation
cd tpu-inference
uv pip install -e .
```

## 3. Create Validation Scripts
Create the following two scripts in your home directory (`~`) to execute the parity check.

### Create `embed_script.py`
This script uses the pooling runner to generate embeddings.

```python
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import random
import numpy as np

from vllm import LLM


def main():
    default_model = "Qwen/Qwen3-Embedding-8B"
    model = os.environ.get("DEV_MODEL", default_model)

    target_device = os.environ.get("VLLM_TARGET_DEVICE", "cpu").lower()
    tp_size = 2 if target_device == "tpu" else 1

    llm = LLM(
        model=model,
        runner="pooling",
        max_num_seqs=16,
        max_model_len=16384,
        max_num_batched_tokens=512,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=tp_size
    )

    base_inputs = [
        "Hello, my name is Alice.",
        "In today's fast-paced world, finding a balance between productivity "
        "and mindfulness has become more important than ever. As urban "
        "landscapes continue to evolve, people are looking for ways to "
        "reconnect with nature without losing the convenience of modern "
        "technology.",
        "最近の技術革新により、私たちの日常生活は劇的に変化しました。"
        "都市の風景は新旧の建築が入り混じり、静かな朝の光が窓から差し込む中で、"
        "人々はそれぞれの目的を持って歩き始めます。",
        "최근 기술의 발전과 함께 우리의 일상에는 많은 변화가 찾아왔습니다.",
    ]
    inputs = [text * 180 for text in base_inputs]

    results = llm.embed(inputs)

    report = dict(zip(base_inputs, [r.outputs.embedding for r in results]))

    output_file = "embed-output-tpu.json" if os.environ.get("VLLM_TARGET_DEVICE") == "tpu" else "embed-output-cpu.json"

    with open(output_file, "w") as f:
        json.dump(report, f, indent=4)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()

```

### Create `compare_precision.py`
This script calculates the Cosine Similarity between CPU and TPU output tensors.

```python
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import numpy as np
import datetime

def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def main():
    with open("embed-output-cpu.json", "r") as f:
        cpu_data = json.load(f)
    with open("embed-output-tpu.json", "r") as f:
        tpu_data = json.load(f)

    output_str = f"{'Text Content (Prefix)':<40} {'Cosine Similarity':<20}\n"
    output_str += "-" * 60 + "\n"

    for key in cpu_data:
        v1 = cpu_data[key]
        v2 = tpu_data.get(key)
        if v2 is None:
            output_str += f"{key[:37]+'...':<40} {'Not found in TPU':<20}\n"
            continue
        
        sim = cosine_similarity(v1, v2)
        output_str += f"{key[:37]+'...':<40} {sim:.10f}\n"

    print(output_str, end="")
    
    # Automatically log to file
    with open("precision_results.log", "a") as f:
        f.write(f"--- Run at {datetime.datetime.now()} ---\n")
        f.write(output_str)
        f.write("\n")

if __name__ == '__main__':
    main()

```

## 4. Generate Embeddings and Verify Numerical Parity
Run the validation suite sequentially:

```bash
# Run CPU Baseline
export VLLM_TARGET_DEVICE="cpu"
python ~/embed_script.py

# Run TPU Validation
export VLLM_TARGET_DEVICE="tpu"
python ~/embed_script.py

# Run Comparison
python ~/compare_precision.py
```

## 5. Expected Results
The Cosine Similarity metrics comparing Cloud TPUs against CPU reference implementations using 7K token sequences.

| Input Type | v7x Cosine Similarity | v6e Cosine Similarity |
|---|---|---|
| English (Short) | 0.9995546074 | 0.9996544555 |
| English (Long) | 0.9997845670 | 0.9997664366 |
| Japanese (Long) | 0.9997769296 | 0.9997792912 |
| Korean (Long) | 0.9997860590 | 0.9998040753 |
