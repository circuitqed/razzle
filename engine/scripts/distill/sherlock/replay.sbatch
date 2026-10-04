#!/bin/bash
#SBATCH --job-name=kb-replay
#SBATCH --partition=owners,gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out
# Offline replay of a self-play run under one trainer setting, then arena of
# selected checkpoints vs the run's starting network.
#   LABEL         output dir name under $KB_DATA/replay2/
#   GAMES         games .jsonl.gz (export_games.py)
#   START         starting checkpoint (.pt)
#   REPLAY_ARGS   extra replay_train.py args (e.g. "--lr 5e-5" or "--lr 2e-4 --ema 0.999")
#   GATE_ITERS    checkpoint iterations to arena (e.g. "050 100 173")
source "$KB_CODE/engine/scripts/distill/sherlock/common.sh"
python3 razzle_fast/_build.py
OUT="$KB_DATA/replay2/$LABEL"
mkdir -p "$OUT"
python3 -u scripts/distill/replay_train.py --games "$GAMES" --model "$START" --out "$OUT" \
    --batch-games 2048 --batch-size 1024 --reuse 2 --workers 7 --device cuda $REPLAY_ARGS
RESULTS="$KB_DATA/arena_results.jsonl"
touch "$RESULTS"
for it in $GATE_ITERS; do
  ck="$OUT/replay_iter_$it.pt"
  [ -f "$ck" ] || { echo "missing $ck"; continue; }
  L="replay2_${LABEL}_$it"
  OUTL=$(python3 scripts/distill/arena.py --a "$ck" --b "$START" --sims 256 --games 400 | tail -1)
  echo "$OUTL" | python3 -c "import json,sys; r=json.load(sys.stdin); r['label']='$L'; print(json.dumps(r))" | tee -a "$RESULTS"
done
