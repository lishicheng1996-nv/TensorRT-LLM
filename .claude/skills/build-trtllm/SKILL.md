---
name: build-trtllm
description: Build TensorRT-LLM from a specific branch into a .sqsh container image via SLURM. Use when the user asks to build, compile, or create a container for TensorRT-LLM on a particular branch or commit.
argument-hint: <branch-name> [--arch ARCH] [--image IMAGE] [--account ACCOUNT] [--partition PARTITION]
user-invocable: true
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# Build TensorRT-LLM Branch

Build TensorRT-LLM from a specified branch into a `.sqsh` container image using SLURM.

## Arguments

Parse `$ARGUMENTS` for:
- **Branch name** (required): The branch to build (e.g. `feat/bench_y`, `main`, `rel/1.3`)
- `--arch ARCH`: CUDA architecture target (default: `100-real` for GB200). Examples: `90-real` (Hopper), `90-real;100-real` (both)
- `--image IMAGE`: Base container image (default: `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc10`)
- `--account ACCOUNT`: SLURM account (default: `coreai_comparch_infbench`)
- `--partition PARTITION`: SLURM partition for the SBATCH allocation (default: `gb200`)
- `--python-only`: Skip C++/CUDA rebuild, only reinstall Python package (use when only `.py` files changed)

## Workflow

### 1. Validate the Branch

Check that the TensorRT-LLM repo exists and the requested branch is available:

```bash
TRTLLM_DIR="/lustre/fsw/coreai_comparch_infbench/shicli/TensorRT-LLM"
```

- If the repo doesn't exist, inform the user and stop.
- Check if the branch exists locally: `git -C $TRTLLM_DIR branch --list <branch>`
- If not local, check remotes. The repo may have multiple remotes (origin, nvidia, etc.). Try fetching from any remote that has the branch:
  ```bash
  git -C $TRTLLM_DIR fetch <remote> <branch>
  ```
- If the branch is from `NVIDIA/TensorRT-LLM` and no `nvidia` remote exists, add it:
  ```bash
  git -C $TRTLLM_DIR remote add nvidia https://github.com/NVIDIA/TensorRT-LLM.git
  git -C $TRTLLM_DIR fetch nvidia <branch>
  ```
- Checkout the branch:
  ```bash
  git -C $TRTLLM_DIR checkout <branch>
  ```
  If the branch doesn't exist locally but was fetched, create a tracking branch:
  ```bash
  git -C $TRTLLM_DIR checkout -b <branch> <remote>/<branch>
  ```

### 2. Generate the Build Script

Get the short commit hash:
```bash
HASH=$(git -C $TRTLLM_DIR rev-parse --short HEAD)
```

Sanitize the branch name for use in filenames (replace `/` with `_`):
```bash
BRANCH_SAFE=$(echo "<branch>" | tr '/' '_')
```

Determine the build mode based on what changed:

**Full build** (C++/CUDA code changed — `.cpp`, `.cu`, `.h`, CMakeLists.txt, etc.):
- Use `build_wheel.py --clean` to recompile everything
- Walltime: ~2-3 hours

**Python-only build** (only `.py` files changed — no C++/CUDA):
- Use `pip install -e .` to reinstall the Python package without recompiling C++/CUDA
- Walltime: ~10-20 minutes
- Use `--python-only` flag or auto-detect by checking `git diff` for C++/CUDA file changes

To auto-detect, check the diff against the base image's commit:
```bash
git -C $TRTLLM_DIR diff --name-only <base_commit>..HEAD | grep -qE '\.(cpp|cu|cuh|h|hpp)$|CMakeLists'
```
If no C++/CUDA files changed, use Python-only mode.

Create the SLURM build script at `$TRTLLM_DIR/build_${BRANCH_SAFE}.sh`:

#### Full build template (C++/CUDA changes):
```bash
#!/bin/bash
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --job-name=build_<BRANCH_SAFE>
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --output=<TRTLLM_DIR>/build_<BRANCH_SAFE>.log

set -euo pipefail

LUSTRE_DIR="/lustre/fsw/coreai_comparch_infbench/shicli"
IMAGE="<IMAGE>"
MOUNTS="${LUSTRE_DIR}:/project"
TRTLLM_SRC="/project/TensorRT-LLM"
TRTLLM_ARCH="<ARCH>"

HASH=$(git -C "${LUSTRE_DIR}/TensorRT-LLM" rev-parse --short HEAD)
SQSH="${LUSTRE_DIR}/trtllm_<BRANCH_SAFE>_${HASH}.sqsh"

echo "============================================"
echo "Compile TensorRT-LLM <branch>"
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
    --account=<ACCOUNT> \
    --partition=batch \
    -n 1 -N 1 \
    --time=02:30:00 \
    --container-image="${IMAGE}" \
    --container-save="${SQSH}" \
    --container-mounts="${MOUNTS}" \
    bash -c "
cd ${TRTLLM_SRC}
rm -rf .venv-3.12
python3 ./scripts/build_wheel.py -G Ninja -a '${TRTLLM_ARCH}' --clean
pip uninstall -y tensorrt_llm
pip install build/tensorrt_llm-*.whl
"

echo "Saved: ${SQSH}"
echo "Build complete!  $(date)"
```

#### Python-only build template (no C++/CUDA changes):
```bash
#!/bin/bash
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --job-name=build_<BRANCH_SAFE>
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=<TRTLLM_DIR>/build_<BRANCH_SAFE>.log

set -euo pipefail

LUSTRE_DIR="/lustre/fsw/coreai_comparch_infbench/shicli"
IMAGE="<IMAGE>"
MOUNTS="${LUSTRE_DIR}:/project"
TRTLLM_SRC="/project/TensorRT-LLM"

HASH=$(git -C "${LUSTRE_DIR}/TensorRT-LLM" rev-parse --short HEAD)
SQSH="${LUSTRE_DIR}/trtllm_<BRANCH_SAFE>_${HASH}.sqsh"

echo "============================================"
echo "Install TensorRT-LLM <branch> (Python-only)"
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
    --account=<ACCOUNT> \
    --partition=batch \
    -n 1 -N 1 \
    --time=00:20:00 \
    --container-image="${IMAGE}" \
    --container-save="${SQSH}" \
    --container-mounts="${MOUNTS}" \
    bash -c "
cd ${TRTLLM_SRC}
pip install -e .
"

echo "Saved: ${SQSH}"
echo "Build complete!  $(date)"
```

Make it executable: `chmod +x $TRTLLM_DIR/build_${BRANCH_SAFE}.sh`

### 3. Important Notes

- The `gb200` partition on Lyris does **not** support `--gres=gpu:N`. Do NOT add `--gres` to either `#SBATCH` or `srun` lines.
- The `srun` uses `--partition=batch` (pyxis/enroot container execution partition) while the outer `#SBATCH` uses the GPU partition for allocation.
- The `.sqsh` output file is saved to `$LUSTRE_DIR/` (one level above the TensorRT-LLM repo).

### 4. Submit and Report

Submit the job:
```bash
sbatch $TRTLLM_DIR/build_${BRANCH_SAFE}.sh
```

Report to the user:
- The job ID
- The build log path: `$TRTLLM_DIR/build_${BRANCH_SAFE}.log`
- The expected output `.sqsh` path
- How to monitor: `tail -f $TRTLLM_DIR/build_${BRANCH_SAFE}.log`
- How to check status: `squeue -j <JOB_ID>`
