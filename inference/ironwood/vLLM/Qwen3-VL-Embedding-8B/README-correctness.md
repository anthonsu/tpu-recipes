# Qwen3-VL-Embedding-8B E2E Correctness Validation on TPU Ironwood

This guide provides step-by-step instructions for running end-to-end correctness and multimodal validation of the **Qwen3-VL-Embedding-8B** model on Ironwood (TPU7x).

## 1. Resource Provisioning

Provision an Ironwood (v7x) TPU VM.
```bash
gcloud alpha compute tpus queued-resources create <RESOURCE_ID> \
    --node-id <NODE_ID> \
    --project <PROJECT_ID> \
    --zone <ZONE> \
    --accelerator-type v7x-8 \
    --runtime-version v2-alpha-tpuv7 \
    --reserved
```

## 2. Environment Setup

### TPU Environment Setup
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

# Sync to validated LKG version of vLLM
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

### GPU Environment Setup (Reference baseline machine)
Ensure your reference GPU (Blackwell/Hopper) VM is set up with the same relative vLLM version:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv embed_env --python 3.12
source embed_env/bin/activate

git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout a51376b3f05a2f74eac6ceeed7e52598b871a0fb
VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto
```

## 3. Create Validation Scripts
Create the scripts below inside your home directory (`~`).

### Create `embed_vl_script.py`
This script validates both multimodal inputs and pure text baselines.

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
import numpy as np
import torch
import multiprocessing as mp

# Select Mode: 'text'  or 'multimodal'
test_mode = os.environ.get("TEST_MODE", "text").lower()
target_device = os.environ.get("VLLM_TARGET_DEVICE", "cuda").lower()

os.environ["MODEL_IMPL_TYPE"] = "vllm"

if target_device == "tpu":
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"
elif target_device == "cuda":
    os.environ["VLLM_TARGET_DEVICE"] = "cuda"

def main():
    from vllm import LLM
    from vllm.multimodal.utils import fetch_image

    model_name = "Qwen/Qwen3-VL-Embedding-8B"
    
    # TPU uses TP=8
    # GPU uses TP=1
    tp_size = 8 if target_device == "tpu" else 1
    
    max_tokens = 13312

    print(f"Device: {target_device} | Mode: {test_mode} | TP: {tp_size}")

    # Initialize LLM with pooling runner
    llm = LLM(
        model=model_name,
        runner="pooling",
        max_model_len=16384,
        max_num_batched_tokens=max_tokens,
        enable_chunked_prefill=True,
        trust_remote_code=True,
        tensor_parallel_size=tp_size,
        dtype="bfloat16",
        gpu_memory_utilization=0.7
    )

    if test_mode == "multimodal":
        # Mode A: Full multimodal test
        image_placeholder = "<|image_pad|>"
        image_url = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/cat_snow.jpg"
        image = fetch_image(image_url)
        
        # Total tokens ~15,600 to force StepPool
        prompt = f"{image_placeholder}\nPlease analyze this image: " + "Test text. " * 1200
        inputs = [{"prompt": prompt, "multi_modal_data": {"image": image}}]
    else:
        # Mode B: Pure text test to isolate the Backbone and StepPool logic.
        # ~15,000 tokens to ensure StepPool is triggered
        prompt = "Backbone alignment test block. " * 1500 
        inputs = [{"prompt": prompt}]

    print(f"Executing {target_device} inference...")
    results = llm.embed(inputs)

    # Normalize output format for comparison
    report = {"embedding": results[0].outputs.embedding}
    output_file = f"embed-{test_mode}-{target_device}.json"
    
    with open(output_file, "w") as f:
        json.dump(report, f)

    print(f"Success! Results saved to {output_file}")

if __name__ == "__main__":
    main()

```

### Create `compare_precision.py`

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
import torch
import torch.nn.functional as F
import os

def load_emb(path):
    """Load embedding tensor from the JSON report file."""
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        data = json.load(f)
        # Standardize key access based on the latest report format
        if "embedding" in data:
            return torch.tensor(data["embedding"])
        elif "multimodal_step_pooling" in data:
            return torch.tensor(data["multimodal_step_pooling"])
        return None

def display_parity(mode, tpu_path, gpu_path):
    """Calculate and display cosine similarity for a device pair."""
    tpu_emb = load_emb(tpu_path)
    gpu_emb = load_emb(gpu_path)
    
    print(f"\n--- Result for [{mode.upper()}] Mode ---")
    if tpu_emb is None or gpu_emb is None:
        print(f"File status: Missing required data ({tpu_path} or {gpu_path})")
        return

    # Cosine Similarity Calculation
    # similarity = (A . B) / (||A|| ||B||)
    sim = F.cosine_similarity(tpu_emb.unsqueeze(0), gpu_emb.unsqueeze(0)).item()
    
    print(f"TPU File: {tpu_path}")
    print(f"GPU File: {gpu_path}")
    print(f"Similarity: {sim:.8f}")

def main():
    print("Numerical Alignment Summary (TPU vs GPU)")
    print("==================================================")
    
    display_parity("text", "embed-text-tpu.json", "embed-text-cuda.json")
    
    display_parity("multimodal", "embed-multimodal-tpu.json", "embed-multimodal-cuda.json")

if __name__ == "__main__":
    main()

```

## 4. Run Parity Comparison
Execute the following runs on both TPU and GPU devices to generate the outputs:

```bash
# Run TPU Pooling
TEST_MODE=text VLLM_TARGET_DEVICE=tpu python embed_vl_script.py
TEST_MODE=multimodal VLLM_TARGET_DEVICE=tpu python embed_vl_script.py

# Run GPU Pooling
TEST_MODE=text VLLM_TARGET_DEVICE=cuda python embed_vl_script.py
TEST_MODE=multimodal VLLM_TARGET_DEVICE=cuda python embed_vl_script.py

# Run Comparison
python ~/compare_precision.py
```

## 5. Expected Results
Expected cosine similarity on v7x (Ironwood) against GPU baselines:

```bash
Numerical Alignment Summary (TPU vs GPU)
==================================================

--- Result for [TEXT] Mode ---
TPU File: embed-text-tpu.json
GPU File: embed-text-cuda.json
Similarity: xxx

--- Result for [MULTIMODAL] Mode ---
TPU File: embed-multimodal-tpu.json
GPU File: embed-multimodal-cuda.json
Similarity: xxx

```

Below are the alignment metrics Ironwood(v7x) compared to Blackwell (B200) and Hopper (H200) targets:

| Input Type | Chunked Prefill* | B200 Cosine Similarity | H200 Cosine Similarity |
|---|---|---|---|
| Text | True | 0.99986249 | 0.999819165 |
| Multimodal | True | 0.99749935 | 0.99912983 |
| Text | False | 0.99986249 | 0.99985290 |
| Multimodal | False | 0.99749935 | 0.99850363 |

\*vLLM-TPU only chunks the text portion of multimodal prefill.
