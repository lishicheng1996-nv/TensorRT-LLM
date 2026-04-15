#!/bin/bash
#SBATCH --account=coreai_comparch_infbench
#SBATCH --partition=gb200
#SBATCH --job-name=build_noeditable
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/fsw/coreai_comparch_infbench/shicli/TensorRT-LLM/build_noeditable.log

set -euo pipefail

LUSTRE_DIR="/lustre/fsw/coreai_comparch_infbench/shicli"
BASE_IMAGE="${LUSTRE_DIR}/trtllm_bench_y_feat_cecae9806.sqsh"
MOUNTS="${LUSTRE_DIR}:/project"
TRTLLM_SRC="/project/TensorRT-LLM"

HASH=$(git -C "${LUSTRE_DIR}/TensorRT-LLM" rev-parse --short HEAD)
SQSH="${LUSTRE_DIR}/trtllm_bench_y_feat_${HASH}_noeditable.sqsh"

echo "============================================"
echo "Non-editable install (Python only)"
echo "============================================"
echo "Base image: ${BASE_IMAGE}"
echo "Git hash:   ${HASH}"
echo "Output:     ${SQSH}"
echo "Time:       $(date)"
echo "Node:       ${SLURMD_NODENAME:-local}"
echo "Job:        ${SLURM_JOB_ID:-local}"
echo "============================================"

srun \
    -l \
    -n 1 -N 1 \
    --container-image="${BASE_IMAGE}" \
    --container-save="${SQSH}" \
    --container-mounts="${MOUNTS}" \
    bash -c "
cd ${TRTLLM_SRC}
echo '[install] Installing non-editable (Python only, C++ already in base image) ...'
pip install . 2>&1
echo '[install] Done.'
python -c 'import tensorrt_llm; print(f\"TRT-LLM version: {tensorrt_llm.__version__}\"); print(f\"Location: {tensorrt_llm.__file__}\")'
"

echo "Saved: ${SQSH}"
echo "Build complete!  $(date)"
