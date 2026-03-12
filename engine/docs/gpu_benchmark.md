# GPU Benchmark Results for Razzle Training

Date: 2026-03-12
Budget: ~$2.00 spent of $10.00 limit
Run ID: bm892

## Configuration

- Simulations: 2000 per move
- Network: medium (~2.4M params)
- MCTS batch size: 32
- Workers per instance: varies (1, 3, or 5)
- Measurement window: 420s (7 min) per config
- Docker image: `ghcr.io/circuitqed/razzle-worker:latest` + onstart C extension build
- FastMCTS: C extension for MCTS tree traversal (compiled at boot via `_build.py`)

## Results

| Config | $/hr | Games/hr | Games/$ | GPU% | CPU% | FastMCTS | Boot |
|--------|------|----------|---------|------|------|----------|------|
| **RTX 3060 x5** | $0.055 | 849 | **15,507** | 97% | 487% | Yes | 13s |
| **RTX 3060 x3** | $0.041 | 626 | **15,118** | n/a | n/a | n/a | 26s |
| RTX 3060 x1 | $0.052 | 309 | 5,981 | 28% | 97% | Yes | 15s |
| **RTX 4090 x5** | $0.368 | **1,714** | 4,661 | 98% | 506% | No | 13s |
| RTX 3090 x5 | $0.121 | 977 | 8,076 | 90% | 488% | Yes | 20s |
| RTX 4090 x3 | $0.315 | 1,003 | 3,182 | 90% | 297% | Yes | 15s |
| RTX 3090 x1 | $0.168 | - | - | - | - | - | TIMEOUT |
| RTX 3090 x3 | $0.139 | - | - | - | - | - | TIMEOUT |
| RTX 4090 x1 | $0.241 | - | - | - | - | - | TIMEOUT |

Sorted by games/$. RTX 3090 x1/x3 failed to boot (300s timeout); x5 needed 450s (8-min timeout).

## Key Findings

### 1. RTX 3060 is the clear value winner

RTX 3060 instances deliver **~15,000 games/$** — roughly **3-5x better value** than RTX 4090:

| GPU | Best games/$ | vs RTX 3060 |
|-----|-------------|-------------|
| RTX 3060 | 15,507 | 1.0x |
| RTX 3090 | 8,076 | 0.52x |
| RTX 4090 | 4,661 | 0.30x |

The RTX 4090 is faster in raw throughput (1,714 vs 849 games/hr) but costs ~7x more per hour. RTX 3090 lands in between but is still only half the value of RTX 3060.

### 2. More workers = better (up to GPU saturation)

| Workers | Games/hr | GPU% | Games/$ |
|---------|----------|------|---------|
| 1 | 309 | 28% | 5,981 |
| 3 | 626 | n/a | 15,118 |
| 5 | 849 | 97% | 15,507 |

Going from 1→3 workers nearly **doubles throughput** while the instance cost barely changes. Going from 3→5 adds another 36% throughput. GPU utilization at 5 workers is 97% — nearly saturated.

### 3. CPU is the bottleneck, not GPU

Each MCTS worker uses ~90-100% of one CPU core (single-threaded Python tree traversal). The GPU sits mostly idle waiting for the next batch. With 5 workers on RTX 3060:
- GPU: 97% utilized (good)
- Each worker: ~55% CPU (contending for 6 cores)
- Total CPU: 487% across 6 cores

The FastMCTS C extension helps but doesn't change the fundamental bottleneck — MCTS tree traversal is serial per worker.

### 4. RTX 4090 x5 did NOT use FastMCTS

Despite having the C extension compiled, the RTX 4090 x5 instance ran Python MCTS (possibly import failure). With FastMCTS enabled, RTX 4090 x5 would likely hit ~2,500+ games/hr. However, games/$ would still be ~3x worse than RTX 3060.

### 5. RTX 3090 is slow to boot but works

RTX 3090 x1/x3 failed with a 300s boot timeout, but x5 succeeded with an extended 8-min timeout (took 450s). Once running, RTX 3090 x5 delivered 977 games/hr with FastMCTS at $0.121/hr — decent but still only half the games/$ of RTX 3060. The slow boot times make RTX 3090 less practical for spot/ephemeral workloads.

## Recommendation

### For cost-efficient training (recommended)
**RTX 3060 with 3-5 workers per instance** at $0.04-0.06/hr

- 15,000+ games per dollar
- At $1/hr budget: ~18 instances = ~15,000 games/hr
- At $2/hr budget: ~36 instances = ~30,000 games/hr

Use 3 workers when instances have <4 CPU cores, 5 workers when they have 6+ cores.

### For maximum speed (money is no object)
**RTX 4090 with 5 workers** at $0.37/hr

- 1,714 games/hr per instance
- But only 4,661 games/$ (3.3x worse value than RTX 3060)

### Current training fleet
The active training run uses 20x RTX 3060 instances with 3 workers each:
- Cost: ~$1.25/hr ($30/day)
- Expected throughput: ~12,000 games/hr
- Value: ~15,000 games/$

## Scaling Advice

To double training speed at minimum cost:
1. **Add more RTX 3060 instances** ($0.04-0.06/hr each) — linear scaling
2. **Increase workers to 5** on instances with 6+ CPU cores — free 36% boost
3. **NOT** by upgrading to RTX 4090 — 3x worse $/game

The theoretical max with RTX 3060 on vast.ai is ~50-60 instances (limited by offer availability), giving ~25,000-30,000 games/hr at ~$3/hr.

## Instance Hardware Details

### RTX 3060 x3 (best value, lowest cost)
- GPU: RTX 3060 12GB, DLP: 12.5
- CPU: 9 cores @ 3.3GHz, 10GB RAM
- Cost: $0.041/hr

### RTX 3060 x5 (best value, highest throughput per instance)
- GPU: RTX 3060 12GB, DLP: 12.3
- CPU: 6 cores @ 3.5GHz, 22GB RAM
- Cost: $0.055/hr

### RTX 3090 x5
- GPU: RTX 3090 24GB, DLP: ~44
- CPU: ~14 cores, ~38GB RAM
- Cost: $0.121/hr
- Note: Boot time 450s (needs extended timeout)

### RTX 4090 x3
- GPU: RTX 4090 24GB, DLP: 100.3
- CPU: 26 cores @ 2.0GHz, 101GB RAM
- Cost: $0.315/hr

### RTX 4090 x5
- GPU: RTX 4090 24GB, DLP: 98.4
- CPU: 26 cores @ 2.5GHz, 101GB RAM
- Cost: $0.368/hr
