#!/bin/bash
#SBATCH --job-name=kb-build
#SBATCH --partition=normal
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --output=%x-%j.out
# Replay exported games into training shards.
source "$KB_CODE/engine/scripts/distill/sherlock/common.sh"
python3 scripts/distill/build_dataset.py "$KB_DATA/games.jsonl.gz" "${SHARDS:-$KB_DATA/shards}" --workers "$SLURM_CPUS_PER_TASK"
