#!/bin/bash
#SBATCH --job-name=kb-arena
#SBATCH --partition=owners,gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=%x-%j.out
# Runs every matchup in $MATCHES (one line of arena.py args each); results
# append to $KB_DATA/arena_results.jsonl. Matchups already recorded with the
# same label are skipped, so a requeued job continues where it left off.
source "$KB_CODE/engine/scripts/distill/sherlock/common.sh"
python3 razzle_fast/_build.py
RESULTS="$KB_DATA/arena_results.jsonl"
touch "$RESULTS"
while read -r LABEL ARGS; do
  [ -z "$LABEL" ] && continue
  if grep -q "\"label\": \"$LABEL\"" "$RESULTS"; then echo "skip $LABEL"; continue; fi
  echo "== $LABEL: $ARGS"
  OUT=$(python3 scripts/distill/arena.py $ARGS | tail -1)
  echo "$OUT" | python3 -c "import json,sys; r=json.load(sys.stdin); r['label']='$LABEL'; print(json.dumps(r))" | tee -a "$RESULTS"
done < "$MATCHES"
