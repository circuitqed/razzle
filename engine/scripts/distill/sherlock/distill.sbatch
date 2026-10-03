#!/bin/bash
#SBATCH --job-name=kb-distill
#SBATCH --partition=owners
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=%x-%A_%a.out
# Array job: one student per line of ${STUDENTS:-$KB_DATA/students.txt} (shards: ${SHARDS:-$KB_DATA/shards})
#   <name> <distill.py args...>
# Preemption (owners) or time limit -> checkpoint, exit 1, requeue, resume.
source "$KB_CODE/engine/scripts/distill/sherlock/common.sh"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${STUDENTS:-$KB_DATA/students.txt}")
NAME=${LINE%% *}; ARGS=${LINE#* }
echo "student $NAME: $ARGS"
python3 scripts/distill/distill.py --data "${SHARDS:-$KB_DATA/shards}" --teacher "$KB_DATA/teacher.pt" \
    --out "$KB_DATA/runs/$NAME" $ARGS &
PID=$!
trap 'kill -USR1 $PID' USR1 TERM
wait $PID || { status=$?; wait $PID 2>/dev/null; exit $status; }
