#!/usr/bin/env bash
#SBATCH --job-name=angha_triplets
#SBATCH --account=naiss2025-22-449
#SBATCH --partition=alvis
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00               # ~1M files @ 32 workers; adjust if needed
#SBATCH --output=logs/triplets_%j.out
#SBATCH --error=logs/triplets_%j.err

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge
module load LLVM/16.0.6-GCCcore-12.3.0 Clang/16.0.6-GCCcore-12.3.0

# Python: use your venv / conda env; ensure tqdm is installed
# Uncomment whichever applies:
# source ~/venv/bin/activate
# module load Python/3.11.3-GCCcore-12.3.0 && pip install --quiet tqdm

export LLVM_BIN=/apps/Arch/software/Clang/16.0.6-GCCcore-12.3.0/bin

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
mkdir -p "${SCRIPT_DIR}/logs"

echo "Job  : $SLURM_JOB_ID"
echo "Node : $SLURMD_NODENAME"
echo "CPUs : $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"

python "${SCRIPT_DIR}/generate_triplets.py" \
    --bench   "${SCRIPT_DIR}/AnghaBench" \
    --output  "${SCRIPT_DIR}/triplets.jsonl" \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --mcpu    cascadelake \
    --resume

echo "End  : $(date)"
