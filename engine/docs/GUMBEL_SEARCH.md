# Gumbel root search (Gumbel AlphaZero / MuZero) — notes for Knightball

Reference: Danihelka, Guez, Schrittwieser, Silver, "Policy improvement by planning with Gumbel",
ICLR 2022. Open-source reference implementation: DeepMind `mctx` (JAX).

## The problem it solves

Standard AlphaZero search (PUCT) at the root keeps sending simulations to the move the prior
likes, and the training target is the root **visit counts**. Two consequences we measured:

- With few simulations (phones, low difficulty levels) and a confident prior, PUCT barely
  checks the second and third choices, so search adds little over the raw network.
- With many simulations, visit counts become nearly one-hot. Training on them makes the
  policy overconfident (phoenix4: EBF 3.43 -> 2.85 in 25 iterations).

## How it works

1. **Candidate moves.** Sample k root moves without replacement with the Gumbel-top-k trick:
   score(a) = g(a) + logit(a), g ~ Gumbel(0, 1), keep the k highest (k = 4-16 depending on the
   budget). The Gumbel noise replaces Dirichlet noise for exploration.
2. **Sequential halving.** Split the n simulations over log2(k) rounds. Each round gives every
   remaining candidate the same number of simulations, then keeps the better half by
   g(a) + logit(a) + sigma(q(a)), where sigma(q) = (c_visit + max_b N(b)) * c_scale * q
   (paper defaults c_visit = 50, c_scale = 0.1; q normalised to [0, 1]).
   This is the "explore the top 2-3 moves evenly" idea, done systematically.
3. **Move choice.** Play the surviving candidate with the highest g + logit + sigma(q). With
   g included this is a sample from an improved policy (self-play); without it, the best move
   (evaluation / app play).
4. **Policy target.** pi'(a) = softmax(logit(a) + sigma(completed_q(a))) over all legal moves,
   where completed_q uses the searched value for visited moves and a mixed value estimate
   (network value and visited Qs) for unvisited ones. It is smooth, Q-based, and provably an
   improvement on the prior in expectation, even with very few simulations.
5. **Below the root** (optional): choose the child that maximises pi'(a) - N(a) / (1 + sum N),
   which steers visits toward the improved policy deterministically instead of PUCT.

The paper reports Gumbel MuZero matching or beating MuZero in Go and chess with far fewer
simulations, and still improving over the raw network at 2-16 simulations, where standard
search often fails to.

## Why it matters here

- **Training:** calibrated policy targets, so no visit-count sharpening; good targets from
  fewer simulations per move, so cheaper games for the same signal.
- **App:** phones run short searches, where Gumbel's root allocation helps most, so the
  same network and time budget should give stronger play at every level.

## Implementation sketch

- C engine (`razzle_fast/core.c`): root-only Gumbel + sequential halving on top of the existing
  tree (interior nodes keep PUCT at first); a completed-Q policy-target export; deterministic
  mode for evaluation.
- Python: `Search` / `selfplay_v2` options to use it and record pi' instead of visit counts.
- TypeScript (`webapp/src/engine/mcts.ts`): the same root logic for app play.
- Knightball specifics: pass chains are extra decisions within a turn (each gets its own
  root search, as now); immediate wins stay a shortcut.

## How to test (offline, before touching training or the app)

1. Arena with the same network: Gumbel at n sims vs PUCT at n sims, n = 16, 64, 256, 1024
   (fixed colours and paired, colour-corrected Elo).
2. If it helps at small n: build a few thousand games with Gumbel targets and replay them
   (frozen value anchor), checking the deep-search benchmark **and** held-out phoenix2
   positions (the forgetting check phoenix4 failed).
3. Only then a vast.ai run, then the app.
