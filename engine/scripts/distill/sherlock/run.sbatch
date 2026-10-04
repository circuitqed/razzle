#!/bin/bash
#SBATCH --job-name=kb-run
#SBATCH --partition=owners,gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# Run one engine command on a GPU node: CMD (run from engine/, e.g. "python3 scripts/distill/opening_analysis.py ...").
source "$KB_CODE/engine/scripts/distill/sherlock/common.sh"
python3 razzle_fast/_build.py
eval "$CMD"
