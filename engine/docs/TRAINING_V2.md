# Training v2: review findings, distillation, and the new self-play pipeline

October 2026. Supersedes the training settings in `TRAINING.md` where they conflict.

## 1. What limited the Pegasus / Gryphon runs

Verified against the stored Gryphon self-play data (346,583 games) and the code.

| # | Issue | Effect | Status |
|---|-------|--------|--------|
| 1 | LR schedule decayed by game count to ~1e-5 at 150k games | learning froze; "peaked at iter_250" ≈ 128k games | **fixed**: constant LR by default (`scripts/trainer.py`) |
| 2 | Replay buffer capped at 100k positions (~2 iterations) + 10 epochs per batch + value weight 12 | overfitting to the newest games | **fixed**: `CompactReplayBuffer` window 250k–3M positions, step-based training (`--reuse 2`), value weight 1.5 |
| 3 | Value targets: γ=0.99, λ=0.95 (discounted + self-bootstrapped) | compressed, inconsistent value scale | **fixed**: undiscounted outcomes by default |
| 4 | Network couldn't see the forced-pass rule (depends on the opponent's last knight move) | 8.3% of turns forced; 39.6% look identical but aren't | **fixed in v2 net**: input planes 7 (last knight dst) and 8 (forced pass) |
| 5 | Self-play ran with early termination on (MCTSConfig default) | ~29% of searches truncated → over-sharp policy targets | **fixed**: `selfplay_v2.py` full searches never early-stop; quick searches are value-only |
| 6 | Random-opening moves recorded as uniform policy targets (5.9% of positions) | taught the policy to be uniform in openings | **fixed**: trainer gives them policy weight 0 (value-only) |
| 7 | Mid-pass decisions got sims/10 fresh searches (17.8% of positions) | weak pass targets | **fixed**: tree reuse — continuations start with the turn's search |
| 8 | Left-right mirror symmetry unused | half the data | **fixed**: mirror augmentation in `NetworkTrainer` and distillation |
| 9 | `tournament.py` Elo never updated (return value dropped) | "best = iter_250" was noise; Pegasus kept improving slowly to iter_600 | fixed (uncommitted local change), and `scripts/distill/arena.py` replaces it |
| 10 | Training reset deleted all games and model files | lost Pegasus games/models | **fixed**: reset retires pending games, keeps files |
| 11 | vast.ai onstart copied `razzle_fast` into a nested dir | workers always ran the image's stale C engine | **fixed** in `train_distributed.py` |
| 12 | C `razzle_state_init` used old rules (pieces start eligible) | only affected callers using the C default start (arena) | **fixed** |

Bigger networks didn't help (Gryphon 24M ≈ Pegasus 2.5M at equal sims) because of 1–3, not capacity.

## 2. Distillation results (no new self-play)

Teacher: `pegasus_iter_600` (strongest: +54 Elo over iter_200, ≈ +90 over iter_250, ≈ Gryphon 344).
Data: all 346,583 stored Gryphon games → 15.6M positions. Students trained 100k steps
(batch 2048, ~30 min on an L40S): targets = ½ MCTS visits + ½ teacher policy, ½ outcome + ½ teacher value, mirror augmentation.

Arena vs teacher, 400 paired games, 256 sims each:

| Student | Cost vs teacher | Elo vs teacher |
|---------|-----------------|----------------|
| 96×12 (same size) | 1× | **+334** |
| 64×8 | ~0.3× | +266 |
| 64×6 | ~0.22× | +244 |
| 48×6 | ~0.13× | +200 (+449 at 4× sims) |
| 32×4 | ~0.04× | +91 |

A value-scaling diagnostic (teacher values ×1.25/1.5/2) only hurt, so the gain is genuine
learning from the large, cleaned dataset — not a calibration artifact.

## 3. Pipeline pieces (all in `engine/scripts/`)

- `distill/export_games.py` — DB → gzip JSONL (reads `training.db` or legacy `games.db`).
- `distill/build_dataset.py` — replay games → shards (bit-packed boards, sparse targets, flags, last-knight square).
- `distill/distill.py` — teacher → student on GPU; `--input-planes 9 --policy-head spatial` for v2; preemption-safe.
- `distill/arena.py` — batched multi-game head-to-head on the C engine; v1/v2 models; `--value-scale-*` diagnostic.
- `distill/calibration.py` — difficulty-ladder calibration (neighbour pairings + Bradley–Terry fit).
- `distill/sherlock/*.sbatch` — Slurm jobs. **Use sh04 nodes only** (`-C GPU_SKU:L40S` or `CLASS:SH4_*`):
  `$SCRATCH` reads hang on sh03 nodes (Oct 2026; reported symptoms in the session notes).
- `selfplay_v2.py` — new self-play worker (one process/GPU, many games, tree reuse, playout-cap randomization).
- `trainer.py` — compact window by default (`--legacy-buffer` for the old path).
- `train_distributed.py` — vast.ai launcher: `--worker v2 --branch <branch> --network-size medium_v2`.

## 4. Network v2 (`--network-size medium_v2`)

96 filters × 12 blocks, 9 input planes, spatial policy head (8 knight-jump planes + 8 directions × 7
distances, gathered into the unchanged 3137-action layout), 2.21M params. v1 models load unchanged.
The app's TypeScript inference supports any v1 size; **v2 needs app support** (9-plane tensor, spatial
head gather) before v2 models can ship in the app.

## 5. Where games live

See `server/persistence.py`: self-play games go to `data/training.db` (compressed) and a daily
`data/training_archive/selfplay_YYYYMMDD.jsonl.gz`. Offload finished days to Google Drive with
`scripts/archive_training_to_drive.sh [--delete-local]`. Games are the valuable resource — the
distillation above came entirely from stored games.

## 6. Starting a run (vast.ai)

1. `POST /training/reset` (keeps games/models; clears run bookkeeping).
2. Upload the bootstrapped starting network as `<run>_iter_000` (`TrainingAPIClient.upload_model`).
3. `python scripts/train_distributed.py --worker v2 --branch app-store-ready --network-size medium_v2
   --run-name <run> --simulations 800 --concurrency 96 --threshold 512 --trainer-extra "--reuse 2"`.
4. Gate checkpoints with `distill/arena.py` against the starting network.

Reliability (vast.ai): the launcher polls instance status every tick. Instances not `running`
within `--boot-timeout` (900 s; usually a stuck image pull) or that vanish are destroyed,
replaced, and their machine is appended to `scripts/vast_blacklist.json` (loaded by every
later run; commit it). A stalled trainer (games pending, no new model for 30–40 min) is
replaced too. Hosts under `--min-inet-down` Mbps (100) are skipped; at equal price faster
links win. Per-instance boot / first-game times go to `<output>/instance_timings.csv`.
Always pass `--max-hours`; stop with SIGINT (SIGTERM leaves instances running).

## 7. App difficulty ladder (Oct 2026)

20 levels in `webapp/src/utils/autoMatch.ts`, calibrated with ~25k arena games
(`scripts/distill/calibration.py`; results in Sherlock `$SCRATCH/kb/arena_results.jsonl`).
Levels 1–5 use `pegasus_iter_050` at 1–32 sims (gentle beginner steps); 6–20 use the
distilled students `distill_s32x4/s48x6/s64x8/s96x12` (`engine/output/models/`, served via
the API like any model, bundled in the iOS app). Displayed rating = 880 + calibrated Elo,
anchored so ~1500 ≈ an even game for a ~1500 chess player. Old levels 14–15 (4096/8192 sims)
were capped to ~1000 sims by the native 10 s budget anyway; old level 10 (p250@256) was
weaker than levels 8–9.
