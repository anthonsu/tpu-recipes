#!/bin/bash

# --- Environment Setup ---
# This script requires uv and a Python N/A virtual environment with xpk installed.
# If you haven't set up uv and the environment, please refer to the README.md.
first_input="$1"

UV_VENV_PATH=".local/bin/venv"

# Activate the virtual environment
source "${UV_VENV_PATH}/bin/activate"

# # Check if xpk is installed in the venv
# if ! pip show xpk &> /dev/null; then
#     echo "xpk not found in the virtual environment. Please install it by running:"
#     echo "pip install xpk==1.3.0"
#     exit 1
# fi

# Read the first command-line argument and store it in a descriptive variable

# Check if the user actually provided an argument
if [ -z "$first_input" ]; then
    echo "Error: No input provided."
    echo "Usage: ./run_recipe_Wan.sh <input>"
else
    echo "You ran the script with the input: $first_input"
fi
# --- End Environment Setup ---

# --- Configuration ---
# Before running this script, please modify the environment variables below
# to match your specific GCP project and cluster setup.
# ---


# XLA Flags
XLA_FLAGS="'\"'\"' \
  --xla_tpu_scoped_vmem_limit_kib=65536 \
  --xla_tpu_enable_async_collective_fusion=true \
  --xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true \
  --xla_tpu_enable_async_collective_fusion_multiple_steps=true \
  --xla_tpu_overlap_compute_collective_tc=true \
  --xla_enable_async_all_reduce=true'\"'\"'"

# MaxDiffusion Workload Overrides
COMMON_MAXDIFFUSION_ARGS="\
attention='\"'\"'flash'\"'\"' \
num_frames=81 \
width=1280 \
height=720 \
jax_cache_dir='\"'\"'${BASE_OUTPUT_DIR}/jax_cache/'\"'\"' \
skip_jax_distributed_system=False \
per_device_batch_size=0.25 \
ici_data_parallelism=2 \
ici_context_parallelism=4 \
allow_split_physical_axes=True \
flow_shift=5.0 \
enable_profiler=True \
run_name='\"'\"'${WORKLOAD_NAME}'\"'\"' \
output_dir='\"'\"'${BASE_OUTPUT_DIR}/'\"'\"' \
flash_min_seq_length=0 \
seed=118445 \
flash_block_sizes='\"'\"'{\"block_kv\":2048,\"block_kv_compute\":1024,\"block_kv_dkv\":2048,\"block_kv_dkv_compute\":1024,\"block_q\":3024,\"block_q_dkv\":3024,\"use_fused_bwd_kernel\":true}'\"'\"' \
base_output_directory='\"'\"'${BASE_OUTPUT_DIR}'\"'\"'"




case "$first_input" in
    "Wan2.1-T2V")
        echo "Starting the Wan2.1-T2V..."
        Wan2_1_T2V_ARGS="\
        model_name='\"'\"'wan2.1'\"'\"' \
        prompt='\"'\"'a japanese pop star young woman with black hair is singing with a smile. She is inside a studio with dim lighting and musical instruments.'\"'\"' \
        guidance_scale=5.0 \
        num_inference_steps=50"
        MAXDIFFUSION_ARGS="${COMMON_MAXDIFFUSION_ARGS} ${Wan2_1_T2V_ARGS}"
        BASE_YAML_CONFIG=${BASE_YAML_CONFIG_WAN_2_1_T2V}
        ;;
    "Wan2.1-I2V")
        echo "Starting the Wan2.1-I2V..."
        Wan2_1_I2V_ARGS="\
        model_name='\"'\"'wan2.1'\"'\"' \
        pretrained_model_name_or_path='\"'\"'Wan-AI/Wan2.1-I2V-14B-720P-Diffusers'\"'\"' \
        num_inference_steps=50"
        MAXDIFFUSION_ARGS="${COMMON_MAXDIFFUSION_ARGS} ${Wan2_1_I2V_ARGS}"
        BASE_YAML_CONFIG=${BASE_YAML_CONFIG_WAN_2_1_I2V}
        ;;
    "Wan2.2-T2V")
        echo "Starting the Wan2.2-T2V..."
        Wan2_2_T2V_ARGS="\
        model_name='\"'\"'wan2.2'\"'\"' \
        prompt='\"'\"'a japanese pop star young woman with black hair is singing with a smile. She is inside a studio with dim lighting and musical instruments.'\"'\"' \
        guidance_scale_low=3.0 \
        guidance_scale_high=4.0 \
        boundary_ratio=0.875 \
        num_inference_steps=40 \
        remat_policy='\"'\"'FULL'\"'\"'"
        MAXDIFFUSION_ARGS="${COMMON_MAXDIFFUSION_ARGS} ${Wan2_2_T2V_ARGS}"
        BASE_YAML_CONFIG=${BASE_YAML_CONFIG_WAN_2_2_T2V}
        ;;
    "Wan2.2-I2V")
        echo "Stopping the Wan2.2-I2V..."
        Wan2_2_I2V_ARGS="\
        model_name='\"'\"'wan2.2'\"'\"' \
        prompt="'\"'\"'a japanese pop star young woman with black hair is singing with a smile. She is inside a studio with dim lighting and musical instruments.'\"'\"'" \
        guidance_scale_low=3.0 \
        guidance_scale_high=4.0 \
        num_inference_steps=40 \
        remat_policy='\"'\"'FULL'\"'\"'"
        MAXDIFFUSION_ARGS="${COMMON_MAXDIFFUSION_ARGS} ${Wan2_2_I2V_ARGS}"
        BASE_YAML_CONFIG=${BASE_YAML_CONFIG_WAN_2_2_I2V}
        ;;
    *)
        # The asterisk (*) acts as the "else" or default catch-all
        echo "Error: Invalid input."
        echo "Please run as: ./run_recipe_wan_2_x.sh {Wan2.1-T2V|Wan2.1-I2V|Wan2.2-T2V|Wan2.2-I2V}"
        ;;
esac
echo ${SCRIPT_PATH}
echo ${MAXDIFFUSION_ARGS}

cmd="xpk workload create \
  --cluster=$CLUSTER_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --priority=medium \
  --max-restarts=0 \
  --tpu-type=$TPU_TPYE \
  --num-slices=1 \
  --docker-image="${WORKLOAD_IMAGE}" \
  --enable-debug-logs \
   \
  --workload="${WORKLOAD_NAME}" \
  --command='set -e && \
export ARTIFACT_DIR=${ARTIFACT_DIR} && \
export LIBTPU_INIT_ARGS=${XLA_FLAGS} && \
${COMMAND_PREFIX} && export HF_TOKEN=${HF_TOKEN} && \
  python ${SCRIPT_PATH}  \
  ${BASE_YAML_CONFIG} \
  ${MAXDIFFUSION_ARGS} \
  run_name='\"'\"'${WORKLOAD_NAME}'\"'\"''"

eval ${cmd}
