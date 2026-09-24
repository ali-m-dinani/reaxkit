#!/bin/bash
#SBATCH --job-name=reaxkit_polarity_perf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=open
#SBATCH --partition=basic
#SBATCH --output=reaxkit_polarity_perf_%j.out
#SBATCH --error=reaxkit_polarity_perf_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Activate the environment containing this revision before sbatch.
# The same xmolout and fort.7 inputs were preprocessed in job 55785915.
# Both analyses share a fresh workspace, matching their original order.
run_root="$SLURM_SUBMIT_DIR/reaxkit_workspace_perf_${SLURM_JOB_ID}"
mkdir "$run_root"
command -v reaxkit > "$run_root/executable.txt"
python -c 'from importlib.metadata import version; print(version("reaxkit"))' > "$run_root/version.txt"
python -c 'from reaxkit.engine.reaxff.io.stream_metrics import FrameWriteBatch; import inspect; print(inspect.getfile(FrameWriteBatch))' > "$run_root/reader_module.txt"

echo "Running formal-charge analysis..."
srun /usr/bin/time -v -o "$run_root/formal.time" \
    reaxkit --timing get-hbn-reference-polarization \
    --engine reaxff \
    --orthogonalize-reference \
    --replication 8 8 10 \
    --charge-source formal \
    --formal-charge Al=3 N=-3 \
    --frames 0:40001:3 \
    --project-root "$run_root"

echo "Running ReaxFF-charge analysis..."
srun /usr/bin/time -v -o "$run_root/reaxff.time" \
    reaxkit --timing get-hbn-reference-polarization \
    --engine reaxff \
    --orthogonalize-reference \
    --replication 8 8 10 \
    --charge-source reaxff \
    --frames 0:40001:3 \
    --project-root "$run_root"

echo "Results, both .time reports, and source/cache timings: $run_root"
echo "10x acceptance: elapsed <= 2479.8 s, with equivalent scientific outputs."
