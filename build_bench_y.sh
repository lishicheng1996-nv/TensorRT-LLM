#!/bin/bash
#SBATCH --account=coreai_comparch_infbench
#SBATCH --partition=gb200
#SBATCH --job-name=build_bench_y
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fsw/coreai_comparch_infbench/shicli/TensorRT-LLM/build.log

set -euo pipefail

LUSTRE_DIR="/lustre/fsw/coreai_comparch_infbench/shicli"
IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc10"
MOUNTS="${LUSTRE_DIR}:/project"
TRTLLM_SRC="/project/TensorRT-LLM"
TRTLLM_ARCH="100-real"

HASH=$(git -C "${LUSTRE_DIR}/TensorRT-LLM" rev-parse --short HEAD)
SQSH="${LUSTRE_DIR}/trtllm_bench_y_feat_${HASH}.sqsh"

echo "============================================"
echo "Compile TensorRT-LLM (C++ + Python)"
echo "============================================"
echo "Image:      ${IMAGE}"
echo "Git hash:   ${HASH}"
echo "Output:     ${SQSH}"
echo "Time:       $(date)"
echo "Node:       ${SLURMD_NODENAME:-local}"
echo "Job:        ${SLURM_JOB_ID:-local}"
echo "============================================"

srun \
    -l \
    --account=coreai_comparch_infbench \
    --partition=batch \
    -n 1 -N 1 \
    --time=00:50:00 \
    --container-image="${IMAGE}" \
    --container-save="${SQSH}" \
    --container-mounts="${MOUNTS}" \
    bash -c "
cd ${TRTLLM_SRC}
echo '[build] Building C++ and wheel ...'
python scripts/build_wheel.py --cuda_architectures ${TRTLLM_ARCH} --build_type Release -j\$(nproc) 2>&1
echo '[build] Installing wheel ...'
pip install build/tensorrt_llm*.whl --force-reinstall --no-deps 2>&1
echo '[build] Setting up editable install ...'
pip install -e . 2>&1
echo '[build] Done.'
python -c 'import tensorrt_llm; print(f\"TRT-LLM version: {tensorrt_llm.__version__}\")'
"

echo "Saved: ${SQSH}"
echo "Build complete!  $(date)"
