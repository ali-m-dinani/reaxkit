#!/bin/bash
#SBATCH --job-name=reaxkit_runtime_benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --account=open
#SBATCH --partition=basic
#SBATCH --output=reaxkit_runtime_benchmark_%j.out
#SBATCH --error=reaxkit_runtime_benchmark_%j.err
#SBATCH --export=ALL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

PYTHON_EXE="${PYTHON_EXE:-python}"
RESULT_DIR="${RESULT_DIR:-benchmark_results/$SLURM_JOB_ID}"
SCRATCH_ROOT="${SLURM_TMPDIR:-$RESULT_DIR/local_scratch}"
mkdir -p "$RESULT_DIR" "$SCRATCH_ROOT"

"$PYTHON_EXE" -m reaxkit.core.runtime.benchmark \
    --frames 1024 \
    --payload-mib 2 \
    --workers 1 2 4 "$SLURM_CPUS_PER_TASK" \
    --queue-multiplier 2 \
    --workspace "$SCRATCH_ROOT" \
    --output "$RESULT_DIR/runtime_benchmark.json"

/usr/bin/time -v "$PYTHON_EXE" -m pytest \
    tests/core/test_frame_pipeline.py \
    tests/core/test_runtime_benchmark.py \
    -q
