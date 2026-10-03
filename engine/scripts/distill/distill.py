#!/usr/bin/env python3
"""
Distill a teacher RazzleNet into a (usually smaller) student on stored self-play
positions (shards from build_dataset.py). No self-play needed.

Targets per position:
  policy = alpha * MCTS visit distribution + (1 - alpha) * teacher policy
           (teacher masked to legal moves; alpha forced to 0 on uniform
            random-opening targets, which carry no information)
  value  = beta * game outcome z + (1 - beta) * teacher value
           (teacher only where the game had no result)
Every sample is randomly mirrored left-right (the game is symmetric), and the
teacher is evaluated on the mirrored board so its targets match.

The whole dataset lives on the GPU (bit-packed boards + sparse CSR targets),
and batches are assembled there. Checkpoints are written atomically and the
job resumes from them, so it is safe on preemptible partitions (SIGTERM /
SIGUSR1 trigger a checkpoint before exit).

The student keeps the 7-plane input and RazzleNet layer types, so the result
exports with scripts/export_onnx.py and runs in the app unchanged.

Example:
  python3 distill.py --data $SCRATCH/kb/shards --teacher pegasus_iter_200.pt \
      --out $SCRATCH/kb/runs/s64x6 --filters 64 --blocks 6 --steps 100000
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from razzle.ai.network import RazzleNet, NetworkConfig, NUM_ACTIONS  # noqa: E402
from razzle.core.symmetry import MOVE_FLIP_MAP  # noqa: E402


# --------------------------------------------------------------------------
# Data

class GpuDataset:
    """All shards concatenated and resident on the GPU."""

    def __init__(self, data_dir: str, device: torch.device, val_mod: int):
        files = sorted(glob.glob(os.path.join(data_dir, 'shard_*.npz')))
        if not files:
            raise SystemExit(f'no shards in {data_dir}')
        parts = {k: [] for k in ('packed', 'pol_idx', 'pol_p', 'leg_idx', 'z', 'flags', 'game', 'lkd')}
        pol_len, leg_len = [], []
        for f in files:
            d = np.load(f)
            for k in parts:
                if k == 'lkd' and k not in d:      # shards built before v2 planes existed
                    parts[k].append(np.full(len(d['z']), -1, np.int8))
                else:
                    parts[k].append(d[k])
            pol_len.append(np.diff(d['pol_ptr']))
            leg_len.append(np.diff(d['leg_ptr']))
        cat = {k: np.concatenate(v) for k, v in parts.items()}
        pol_len = np.concatenate(pol_len)
        leg_len = np.concatenate(leg_len)

        def ptr(lengths):
            out = np.zeros(len(lengths) + 1, dtype=np.int64)
            np.cumsum(lengths, out=out[1:])
            return out

        t = lambda a, dt=None: torch.from_numpy(a if dt is None else a.astype(dt)).to(device)
        self.packed = t(cat['packed'])
        self.pol_ptr = t(ptr(pol_len))
        self.pol_idx = t(cat['pol_idx'])
        self.pol_p = t(cat['pol_p'])
        self.leg_ptr = t(ptr(leg_len))
        self.leg_idx = t(cat['leg_idx'])
        self.z = t(cat['z'], np.float32)
        self.flags = t(cat['flags'])
        self.lkd = t(cat['lkd'], np.int64)
        self.n = len(cat['z'])
        self.device = device

        is_val = (cat['game'] % val_mod) == 0
        self.train_rows = torch.from_numpy(np.nonzero(~is_val)[0]).to(device)
        self.val_rows = torch.from_numpy(np.nonzero(is_val)[0]).to(device)
        self.shifts = torch.arange(7, -1, -1, device=device, dtype=torch.uint8)
        self.flip = torch.from_numpy(MOVE_FLIP_MAP.astype(np.int64)).to(device)

    def _csr(self, ptr, idx, rows):
        starts = ptr[rows]
        lens = ptr[rows + 1] - starts
        r = torch.repeat_interleave(torch.arange(len(rows), device=self.device), lens)
        csum = torch.cumsum(lens, 0)
        offs = torch.arange(int(csum[-1]), device=self.device) - torch.repeat_interleave(csum - lens, lens)
        flat = starts[r] + offs
        return r, idx[flat].long(), flat

    def batch(self, rows: torch.Tensor, mirror: bool, planes: int = 7):
        b = len(rows)
        bits = (self.packed[rows].unsqueeze(-1) >> self.shifts) & 1          # [B, 49, 8]
        x = bits.reshape(b, -1)[:, :392].reshape(b, 7, 8, 7).float()
        if planes == 9:
            # v2 planes: 7 = opponent's last knight destination (one-hot), 8 = forced pass
            extra = torch.zeros(b, 2, 56, device=self.device)
            sq = self.lkd[rows]
            has = sq >= 0
            extra[has.nonzero(as_tuple=True)[0], 0, sq[has]] = 1.0
            extra[:, 1, :] = ((self.flags[rows] >> 2) & 1).float()[:, None]
            x = torch.cat([x, extra.view(b, 2, 8, 7)], dim=1)

        r, ci, flat = self._csr(self.pol_ptr, self.pol_idx, rows)
        p = torch.zeros(b, NUM_ACTIONS, device=self.device)
        p[r, ci] = self.pol_p[flat].float()

        r, li, _ = self._csr(self.leg_ptr, self.leg_idx, rows)
        legal = torch.zeros(b, NUM_ACTIONS, dtype=torch.bool, device=self.device)
        legal[r, li] = True

        if mirror:
            m = torch.rand(b, device=self.device) < 0.5
            if m.any():
                x[m] = x[m].flip(-1)
                p[m] = p[m][:, self.flip]
                legal[m] = legal[m][:, self.flip]

        return x, p, legal, self.z[rows], self.flags[rows]


# --------------------------------------------------------------------------
# Training

def amp(device: torch.device, dtype):
    """bf16 autocast on GPUs that support it; a no-op otherwise."""
    return torch.autocast(device.type, dtype=dtype or torch.bfloat16, enabled=dtype is not None)


def masked_log_softmax(logp: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logp.masked_fill(~legal, -1e9), dim=1)


def targets(teacher, x, p_mcts, legal, z, flags, alpha, beta):
    with torch.no_grad():
        t_logp, t_v, t_d = teacher(x[:, :teacher.config.num_input_planes])
        t_pol = masked_log_softmax(t_logp.float(), legal).exp()
        a = torch.full_like(z, alpha)
        a[(flags & 1).bool()] = 0.0                      # uniform random-opening targets
        p_tgt = a[:, None] * p_mcts + (1 - a[:, None]) * t_pol
        bta = torch.full_like(z, beta)
        bta[z == 0] = 0.0
        v_tgt = bta * z + (1 - bta) * t_v.float().squeeze(1)
    return p_tgt, v_tgt, t_d.float().squeeze(1), t_pol, t_v.float().squeeze(1)


@torch.no_grad()
def evaluate(student, teacher, ds, rows, args, amp_dtype, max_rows=60000):
    student.eval()
    rows = rows[:max_rows]
    sums = dict(n=0, ce_mcts=0.0, kl_teacher=0.0, top1_teacher=0.0, top1_mcts=0.0,
                v_mse_z=0.0, v_mse_teacher=0.0, t_ce_mcts=0.0, t_v_mse_z=0.0, t_top1_mcts=0.0)
    for i in range(0, len(rows), 4096):
        r = rows[i:i + 4096]
        x, p, legal, z, flags = ds.batch(r, mirror=False, planes=student.config.num_input_planes)
        with amp(x.device, amp_dtype):
            s_logp, s_v, _ = student(x)
        _, _, _, t_pol, t_v = targets(teacher, x, p, legal, z, flags, args.alpha, args.beta)
        s_lp = masked_log_softmax(s_logp.float(), legal)
        t_lp = torch.log(t_pol.clamp_min(1e-12))
        informative = ~(flags & 1).bool()               # skip uniform targets for MCTS metrics
        s_v = s_v.float().squeeze(1)
        n = len(r)
        sums['n'] += n
        sums['ce_mcts'] += float(-(p * s_lp).sum(1)[informative].sum()) / max(1, int(informative.sum())) * n
        sums['t_ce_mcts'] += float(-(p * t_lp).sum(1)[informative].sum()) / max(1, int(informative.sum())) * n
        sums['kl_teacher'] += float((t_pol * (t_lp - s_lp)).sum(1).sum())
        sums['top1_teacher'] += float((s_lp.argmax(1) == t_pol.argmax(1)).float().sum())
        sums['top1_mcts'] += float((s_lp.argmax(1) == p.argmax(1)).float()[informative].mean()) * n
        sums['t_top1_mcts'] += float((t_pol.argmax(1) == p.argmax(1)).float()[informative].mean()) * n
        sums['v_mse_z'] += float(((s_v - z) ** 2).sum())
        sums['t_v_mse_z'] += float(((t_v - z) ** 2).sum())
        sums['v_mse_teacher'] += float(((s_v - t_v) ** 2).sum())
    student.train()
    n = sums.pop('n')
    return {k: round(v / n, 5) for k, v in sums.items()}


def save_atomic(obj, path: Path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--teacher', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--filters', type=int, default=64)
    ap.add_argument('--blocks', type=int, default=6)
    ap.add_argument('--policy-filters', type=int, default=2)
    ap.add_argument('--value-filters', type=int, default=4)
    ap.add_argument('--value-hidden', type=int, default=256)
    ap.add_argument('--policy-hidden', type=int, default=0)
    ap.add_argument('--policy-head', choices=['fc', 'spatial'], default='fc')
    ap.add_argument('--input-planes', type=int, choices=[7, 9], default=7)
    ap.add_argument('--steps', type=int, default=100_000)
    ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--warmup', type=int, default=1000)
    ap.add_argument('--alpha', type=float, default=0.5, help='weight of MCTS visits in the policy target')
    ap.add_argument('--beta', type=float, default=0.5, help='weight of the game outcome in the value target')
    ap.add_argument('--value-weight', type=float, default=1.0)
    ap.add_argument('--val-mod', type=int, default=50, help='games with id %% val_mod == 0 are validation')
    ap.add_argument('--eval-every', type=int, default=2500)
    ap.add_argument('--ckpt-every', type=int, default=2500)
    ap.add_argument('--no-mirror', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    cuda = dev.type == 'cuda'
    if cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp_dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else None

    t0 = time.time()
    ds = GpuDataset(args.data, dev, args.val_mod)
    print(f'data: {ds.n:,} positions ({len(ds.train_rows):,} train / {len(ds.val_rows):,} val) '
          f'loaded in {time.time() - t0:.0f}s; device {torch.cuda.get_device_name() if cuda else 'cpu'}', flush=True)

    teacher = RazzleNet.load(args.teacher, device=str(dev)).to(dev).eval()
    for prm in teacher.parameters():
        prm.requires_grad_(False)

    cfg = NetworkConfig(num_filters=args.filters, num_blocks=args.blocks,
                        policy_filters=args.policy_filters, value_filters=args.value_filters,
                        value_hidden=args.value_hidden, policy_hidden=args.policy_hidden,
                        num_input_planes=args.input_planes, policy_head=args.policy_head)
    student = RazzleNet(cfg).to(dev)
    print(f'teacher {args.teacher}: {teacher.num_parameters():,} params | '
          f'student {args.filters}x{args.blocks}: {student.num_parameters():,} params', flush=True)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.wd)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))

    step = 0
    best = float('inf')
    ckpt_path = out / 'ckpt.pt'
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        student.load_state_dict(ck['student'])
        opt.load_state_dict(ck['opt'])
        step, best = ck['step'], ck.get('best', best)
        torch.set_rng_state(ck['rng_cpu'])
        if cuda and ck.get('rng_cuda') is not None:
            torch.cuda.set_rng_state(ck['rng_cuda'])
        print(f'resumed from step {step}', flush=True)
    else:
        torch.manual_seed(args.seed)
        (out / 'args.json').write_text(json.dumps(vars(args), indent=2))

    def checkpoint():
        save_atomic(dict(student=student.state_dict(), opt=opt.state_dict(), step=step, best=best,
                         rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state() if cuda else None), ckpt_path)

    stop = {'flag': False}

    def on_signal(signum, _frame):
        print(f'signal {signum}: checkpointing and exiting', flush=True)
        stop['flag'] = True
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGUSR1, on_signal)

    log = open(out / 'log.jsonl', 'a')
    student.train()
    n_train = len(ds.train_rows)
    t_last = time.time()
    while step < args.steps and not stop['flag']:
        for g in opt.param_groups:
            g['lr'] = lr_at(step)
        rows = ds.train_rows[torch.randint(0, n_train, (args.batch,), device=dev)]
        x, p, legal, z, flags = ds.batch(rows, mirror=not args.no_mirror, planes=args.input_planes)
        with amp(x.device, amp_dtype):
            p_tgt, v_tgt, d_tgt, _, _ = targets(teacher, x, p, legal, z, flags, args.alpha, args.beta)
            s_logp, s_v, s_d = student(x)
        pol_loss = -(p_tgt * s_logp.float()).sum(1).mean()
        val_loss = F.mse_loss(s_v.float().squeeze(1), v_tgt)
        diff_loss = F.mse_loss(s_d.float().squeeze(1), d_tgt)   # keep the (unused) head sane
        loss = pol_loss + args.value_weight * val_loss + 0.1 * diff_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
        opt.step()
        step += 1

        if step % 500 == 0:
            dt = time.time() - t_last
            t_last = time.time()
            print(f'step {step} loss {loss.item():.4f} (pol {pol_loss.item():.4f} val {val_loss.item():.4f}) '
                  f'lr {lr_at(step):.2e} {500 * args.batch / dt:,.0f} pos/s', flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            m = evaluate(student, teacher, ds, ds.val_rows, args, amp_dtype)
            m['step'] = step
            log.write(json.dumps(m) + '\n')
            log.flush()
            print('eval', m, flush=True)
            score = m['ce_mcts'] + m['v_mse_z']
            if score < best:
                best = score
                student.save(str(out / 'student_best.pt'))

        if step % args.ckpt_every == 0:
            checkpoint()

    checkpoint()
    if step >= args.steps:
        student.save(str(out / 'student_final.pt'))
        print(f'done: {step} steps in {time.time() - t0:.0f}s', flush=True)
    else:
        # Preempted: exit nonzero so Slurm --requeue reruns us (we resume from ckpt).
        sys.exit(1)


if __name__ == '__main__':
    main()
