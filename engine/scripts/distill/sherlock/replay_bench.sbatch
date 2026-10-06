#!/bin/bash
#SBATCH --job-name=kb-rpb
#SBATCH --partition=owners,gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# Replay + gates + benchmarks. LABEL, GAMES, START, REPLAY_ARGS, GATE_ITERS, DEEP_SET, BROAD_SET as in replay.sbatch.
source "$KB_CODE/engine/scripts/distill/sherlock/common.sh"
python3 razzle_fast/_build.py
OUT="$KB_DATA/replay3/$LABEL"; mkdir -p "$OUT"
python3 -u scripts/distill/replay_train.py --games "$GAMES" --model "$START" --out "$OUT" \
    --batch-games 2048 --batch-size 1024 --reuse 2 --workers 7 --device cuda $REPLAY_ARGS
RES="$KB_DATA/replay3/results.jsonl"; touch "$RES"
for it in $GATE_ITERS; do
  ck="$OUT/replay_iter_$it.pt"; [ -f "$ck" ] || { echo "missing $ck"; continue; }
  python3 scripts/distill/arena.py --a "$ck" --b "$START" --sims 256 --games 800 | tail -1 \
    | python3 -c "import json,sys; r=json.load(sys.stdin); r['label']='$LABEL'; r['iter']='$it'; r['kind']='gate'; print(json.dumps(r))" >> "$RES"
  python3 scripts/distill/benchmark.py --games "$DEEP_SET" --skip 45000 --limit 600 --label "$LABEL/$it/deep" "$ck" >> "$RES"
  python3 scripts/distill/benchmark.py --games "$BROAD_SET" --skip 340000 --limit 600 --label "$LABEL/$it/broad" "$ck" >> "$RES"
done
