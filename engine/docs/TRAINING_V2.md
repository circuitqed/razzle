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

1. Pick a new run name. Runs coexist: models, metrics and self-play games carry the run (the
   version prefix before `_iter_`), so no reset is needed. `GET /training/runs` lists them.
2. Upload the bootstrapped starting network as `<run>_iter_000` (`TrainingAPIClient.upload_model`).
   The latest upload's run is the *current* run: workers (no run given) play its latest model and
   the dashboard shows it. The trainer (`--run-name <run>`) asks for its own run explicitly: its
   latest model, only its games (`GET /training/games?run=`), its metrics, and trainer state
   stored as `<run>.trainer_state` / `<run>.replay_buffer`.
   `POST /training/reset?run=<run>` restarts one run (drops its model/metric/state records, retires
   its pending games); a plain reset only retires the pending queue. Files and games are never deleted.
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

## 8. Throughput profile and fixes (phoenix2, Oct 4 2026)

Measured on the live run (RTX 3060 workers, RTX 3090 trainer, 800-sim full / 160-sim quick searches).

**Self-play worker** (games/h per RTX 3060, 4.3-core host):

| Setup | games/h |
|---|---|
| 1 process, fp32 (as launched) | 5,650 |
| 2 processes | 7,975 |
| 3 processes | 9,833 |
| 3 processes, fp16 inference | **13,473** |

One process saturates one CPU core (Python tree bookkeeping) while the GPU idles ~half the
time; the other half is an fp32 forward pass (~30-50 ms per 768 positions). fp16 is 2.4x
faster (max |dp| 0.004, mean |dv| 0.0007, top move agrees 99.96%). Defaults now:
`selfplay_v2 --fp16`, `train_distributed --workers-per-instance 3`. Hosts need >= 4 CPU cores
for 3 processes. Processes share the model dir: downloads are atomic (temp file + rename).

**Trainer** (per 2,048-game iteration on the 3090): 120 s -> 29 s.

| Phase | before | after |
|---|---|---|
| game -> position conversion | 23 s (1 core) | 6 s (16 procs, packed chunks) |
| dense npz archive of every batch | 29 s | removed (games live on the server) |
| validation (all ~160k positions) | ~30 s | 1.4 s (16k subsample) |
| training | ~30 s, 2.6 steps/s | 21 s, 13 steps/s (vectorized sampler) |

Policy top-1/top-3 in the dashboard were computed over value-only positions too (no target)
and were meaningless before this fix; they now cover positions with a policy target only.

One 3090 trainer now absorbs ~150k+ games/h, i.e. ~10-12 workers at the new worker speed.
Follow-up measurements (same day):
- Profile with fp16 + 3 procs: the per-game Python loop is ~15% of a process; the forward
  pass dominates and the GPU runs at ~90%, i.e. workers are now **GPU-bound**.
- CUDA graphs (`selfplay_v2 --cuda-graphs`, padded batch buckets): outputs identical, but
  **-18% games/h** (12,356 -> 10,090) because padding adds GPU work and graphs are recaptured on
  every model update. Off by default.
- Leaf batching costs search quality: same net, equal sims, 1 leaf/round vs 8
  (`arena.py --leaf-batch-a 1 --leaf-batch-b 8`): **+34 Elo [10, 58] at 256 sims**, +14 [-20, 48]
  at 800 sims. Duplicate leaves within a batch: 3.2% at 160 sims, 0.3% at 800 (minor).
- Next levers: GPU throughput (TensorRT fp16/int8), and a batched C driver that makes
  1 leaf/game with ~8x more concurrent games cheap (quality gain mainly for quick searches).

## 9. phoenix2 outcome and diagnosis (Oct 4-5 2026)

phoenix2 (start = distilled v2 96x12 after the canary replay, LR 2e-4, reuse 2, 800/160 sims,
25% full searches) ran 267 iterations / 476k games and **did not get stronger**. Gates vs the
start (256 sims, 800 games, colour-corrected Elo = Bradley-Terry split of per-colour scores):
iter 72 -3, iter 152 -22, iter 267 -21. Paired-colour gates are diluted (about 2/3 of opening
pairs are won by the same colour), but the correction moves these numbers by only a few points.

Offline replays of the 354k phoenix2 games (`scripts/distill/sherlock/replay.sbatch`), colour-
corrected Elo vs the start at ~100k / 200k / 354k games:

| setting | | | |
|---|---|---|---|
| LR 2e-4 (live) | +2 | -16 | -13 |
| LR 5e-5 | +28 | 0 | +16 |
| LR 2e-5 | 0 | +4 | +4 |
| LR 2e-4 + EMA 0.999 | -77 | -61 | -29 (BN stats copied from the live net: implementation flaw) |

The data pipeline was verified identical to the distillation pipeline (planes, policies, legal
masks, value signs) on 200 games. Learning rate is not the problem. Open hypotheses: the
distilled start is already at/above what these self-play targets teach; the game's colour bias
(first player 65-79% in strong play) leaves value targets uninformative; network capacity.

Related measurements: first-player edge grows with skill (beginner 52%, ~1500 rating 63%,
superhuman 79%); moving second costs ~0.6 doublings of search at 32 sims, ~1.4 at 256, ~2.1 at
1024; dominant opening 72 / mirror 298 (`scripts/distill/opening_analysis.py`).

## 10. phoenix3 / phoenix4 and what they ruled out (Oct 5-6 2026)

**phoenix3** (800/160 sims, value target = 1/2 recorded search value + 1/2 outcome, LR 5e-5,
start = replay checkpoint +56 over phoenix2's start): gates vs its start -38 / -7 / -8.
Offline A/B on its 414k games: own search values -23/-25/-22, frozen teacher +2/-2/+19.
The current network's search values feed its own errors back into the value target.

**phoenix4** (every move a 3200-sim search, value = 1/2 frozen teacher + 1/2 outcome): gates
-77 (iter 25, 256 sims), -98 (iter 40, 1024 sims), -82 (iter 55). Not a colour bug (worse
with both colours) and not only shallow-search miscalibration (as bad at 1024 sims).
Benchmarks explain it:

| | deep-game positions (in-distribution) | phoenix2 positions (held out, broader) |
|---|---|---|
| start | move match 0.695, value MSE 0.526 | move match 0.764, value MSE 0.559 |
| iter 55 | 0.700, 0.494 | **0.738**, 0.565 |

The network specialises on the narrow positions of deep-search self-play and forgets broad
knowledge from distillation; policies also sharpen (EBF 3.43 -> 2.85). A frozen value anchor
only constrains positions that appear in training. Next candidates: also mix the teacher's
policy into the policy target (as distillation did), rehearse the broad distillation dataset
in every batch, more diverse self-play openings, and Gumbel root search (calibrated, Q-based
policy targets; see GUMBEL_SEARCH.md).

Tooling added: fixed deep-search benchmark (agreement with 3200-sim moves on 62,914 held-out
positions), gates on the trainer's GPU (no Sherlock queue), offline replay A/B on vast.ai.

## 11. v2 student ladder for the app (Oct 7-8 2026)

Seven v2 students (9-plane input, spatial policy head) were distilled from phoenix3_iter_000
on all 1.6M archived games (½ MCTS visits + ½ teacher policy; ½ outcome + ½ teacher value),
for 100k-250k steps. They were trained on vast.ai RTX 3090s (~$1.30 in total) and on Sherlock.

| Student | Matches the teacher's top move | Teacher KL | Value MSE vs teacher |
|---|---|---|---|
| 16x2 | 0.595 | 0.438 | 0.097 |
| 24x3 | 0.684 | 0.280 | 0.071 |
| 32x4 | 0.749 | 0.185 | 0.049 |
| 48x6 | 0.804 | 0.100 | 0.029 |
| 64x8 | 0.829 | 0.079 | 0.023 |
| 96x12 | 0.844 | 0.068 | 0.022 |
| 128x16 | 0.849 | 0.065 | 0.024 |

Calibration: 117 new 200-game matches (Sherlock H100/H200, about 1 h), fitted jointly with
the 111 earlier matches (Bradley-Terry; `scripts/distill/calibration.py fit`). The data is in
`docs/calibration/`. The displayed rating is 880 + Elo relative to pegasus_iter_050 at 1 sim.

- **v2 students beat v1 students of the same size.** v2 96x12 at 1024 sims rates 2225, against
  2107 for v1 s96x12.
- **128x16 is not worth it.** At equal sims it is only ~20-40 Elo above 96x12 (256: 2066 vs
  2025; 512: 2171 vs 2136; 1024: 2245 vs 2225). It costs about twice the compute, and a
  doubling of sims is worth ~100 Elo. The ladder stops at 96x12.
- **Very low sim counts are noisy.** 16x2 at 2 and at 4 sims rate about the same.
- The app ladder (`webapp/src/utils/autoMatch.ts` TIERS) uses the smallest network that
  reaches each level's rating, spanning 815 (16x2 @ 1 sim) to 2225 (96x12 @ 1024 sims), with
  steps of 56-98 Elo. Levels 1-5 cost 50-100x less per evaluation than the old pegasus_iter_050 levels.

### High-sim calibration and desktop levels (Oct 8 2026)

13 more 200-game matches compared v2 96x12, v2 128x16 and the teacher phoenix3_iter_000 at
2048, 4096 and 8192 sims, anchored to the 1024-sim configs. They were fitted jointly with
everything above (`docs/calibration/*_highsims.*`). The refit moved 96x12 @1024 by -11, so the
numbers below are shifted +11 to match the ladder.

| Config | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|
| v2 96x12 | 2225 | 2270 | 2306 | 2321 |
| v2 128x16 | 2242 | 2275 | 2310 | 2339 |
| teacher p3 | — | 2222 | — | 2295 |

- **Search depth levels off.** The gain is ~+45 for each doubling from 1024 sims, then ~+15
  from 4096 to 8192. About 80% of game pairs are decided by colour (P0 wins ~82%), so skill
  differences compress at the top.
- **128x16 ≈ 96x12** at equal sims (+5 to +18, within noise), for twice the compute.
- **The students out-search their teacher.** p3 @2048 loses to both students @2048 (by -42
  and -60). This is probably because the students' value heads are distilled from both
  outcomes and teacher values.
- The desktop-only levels (`desktopOnly` TIERS) are L21 96x12 @2048 (2270), L22 96x12 @4096
  (2305) and L23 128x16 @8192 (2340). Phones stop at L20.

## 12. Search options: virtual loss in Q, first-play urgency (Oct 9 2026)

`razzle_mcts_set_search_options` (C; off by default, so the original search is unchanged).
Arena, same net both sides (`phoenix3_iter_000`), leaf batch 8, A = option on, B = original:

| Option | Sims | Games | Elo A - B [95%] |
|---|---|---|---|
| virtual loss also lowers Q (`--vloss-q-a`) | 64 | 2000 | -97 [-113, -81] |
| virtual loss also lowers Q | 256 | 1000 | -70 [-93, -49] |
| FPU: unvisited Q = parent Q - 0.2·sqrt(visited prior) (`--fpu-a 0.2`) | 64 | 2000 | +43 [27, 58] |
| FPU 0.5 | 64 | 2000 | +51 [36, 67] |
| FPU 0.2 | 256 | 1000 | **+68 [47, 91]** |
| FPU 0.5 | 256 | 1000 | +58 [37, 80] |

Virtual loss in Q over-spreads each 8-leaf batch (duplicates were already rare). Unvisited
children at Q = 0 (a draw) were too optimistic in a decisive game: the search wasted visits on
low-prior moves. **selfplay_v2 now defaults to `--fpu 0.2`** (`--fpu -1` = old search). The
arena keeps the original search by default so earlier ratings stay comparable. The app's TS/
native search is separate and unchanged; porting FPU there would raise every ladder level and
needs a recalibration.
