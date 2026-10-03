# Shared setup for distillation jobs. Expects KB_CODE (code dir with engine/) and KB_DATA.
set -euo pipefail
ml purge
ml math py-pytorch/2.4.1_py312
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
python3 -c 'import torch, numpy; print("torch", torch.__version__, "numpy", numpy.__version__)'
# Run from node-local storage; razzle_fast is compiled per node (gcc -march=native).
WORK=${L_SCRATCH:-/tmp}/kb_$SLURM_JOB_ID
mkdir -p "$WORK"
cp -r "$KB_CODE/engine" "$WORK/"
cd "$WORK/engine"
