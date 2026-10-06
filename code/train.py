"""
train.py — NEGATIVE CONTROL: a code-blind residual MLP decoder on the surface code.

This is NOT a competitive decoder and is not presented as one. It reads the detector
bits as a bare vector (no Tanner/detector-graph structure), predicts the logical
observable flip, and also emits a calibrated probability. It loses to MWPM by ~2 orders
of magnitude, which is the point: it measures what a structure-free model does under a
small fixed budget. See docs/p1_neural_controls.md.

Contract: prints 'logical_error_rate: <f>', 'decode_latency_us: <f>', 'ece: <f>',
          'brier: <f>', 'peak_vram_mb: <f>'.

Run: CUDA_VISIBLE_DEVICES=0 python train.py
"""
from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import prepare as P

FAMILY = "surface"
DISTANCE = 5              # measurable LER regime (MWPM d=5 LER ~7.8e-4) for a fair neural comparison
TIME_BUDGET_S = 120       # fast teacher baseline; raise toward P.TIME_BUDGET_S for the full study
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ResidualMLPDecoder(nn.Module):
    def __init__(self, n_det, width=512, depth=4, p_drop=0.1):
        super().__init__()
        self.inp = nn.Linear(n_det, width)
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
            nn.Dropout(p_drop), nn.Linear(width, width)) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        return self.head(h).squeeze(-1)   # logit of P(logical flip)


def ece_score(probs, labels, n_bins=15):
    """Calibration of P(logical flip): per prob-bin, |mean predicted prob - empirical rate|."""
    labels = labels.astype(np.float64)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0; n = len(probs)
    for i in range(n_bins):
        m = (probs > edges[i]) & (probs <= edges[i + 1])
        if m.sum() == 0:
            continue
        conf = probs[m].mean()            # mean predicted P(flip) in bin
        emp = labels[m].mean()            # empirical P(flip) in bin
        ece += (m.sum() / n) * abs(emp - conf)
    return float(ece)


def main():
    torch.manual_seed(0); np.random.seed(0)
    ds = P.generate_stim_dataset(FAMILY, DISTANCE)
    X = ds["detectors"].astype(np.float32)
    y = ds["observables"][:, 0].astype(np.float32)
    n = len(y); ntr = int(0.8 * n)
    perm = np.random.permutation(n)
    Xtr, ytr = X[perm[:ntr]], y[perm[:ntr]]
    Xva, yva = X[perm[ntr:]], y[perm[ntr:]]
    Xtr_t = torch.from_numpy(Xtr).to(DEVICE); ytr_t = torch.from_numpy(ytr).to(DEVICE)
    Xva_t = torch.from_numpy(Xva).to(DEVICE); yva_t = torch.from_numpy(yva).to(DEVICE)

    model = ResidualMLPDecoder(X.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    bs = 4096
    torch.cuda.reset_peak_memory_stats() if DEVICE == "cuda" else None

    t0 = time.time(); step = 0
    while time.time() - t0 < TIME_BUDGET_S:
        idx = torch.randint(0, ntr, (bs,), device=DEVICE)
        logit = model(Xtr_t[idx])
        loss = F.binary_cross_entropy_with_logits(logit, ytr_t[idx])
        opt.zero_grad(); loss.backward(); opt.step(); step += 1

    model.eval()
    with torch.no_grad():
        pv = torch.sigmoid(model(Xva_t)).float().cpu().numpy()
    yv = yva.astype(bool)
    pred = pv > 0.5
    ler = float(np.mean(pred != yv))
    ler_per_round = 1.0 - (1.0 - ler) ** (1.0 / P.ROUNDS) if 0 < ler < 1 else ler / P.ROUNDS
    ece = ece_score(pv, yv)
    brier = float(np.mean((pv - yva) ** 2))

    # latency: batched neural decode, us per round
    with torch.no_grad():
        _ = model(Xva_t[:512])   # warm
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        tl = time.time(); _ = model(Xva_t); torch.cuda.synchronize() if DEVICE == "cuda" else None
        lat_us = (time.time() - tl) / len(Xva_t) * 1e6 / P.ROUNDS
    peak_vram_mb = (torch.cuda.max_memory_allocated() / 1024**2) if DEVICE == "cuda" else 0.0

    # MWPM reference at the same distance (for context)
    import pymatching
    m = pymatching.Matching.from_detector_error_model(ds["dem"])
    mwpm = P.evaluate_ler(lambda d: m.decode_batch(d), FAMILY, DISTANCE)

    print("---")
    print(f"logical_error_rate: {ler:.8f}")
    print(f"ler_per_round:      {ler_per_round:.8e}")
    print(f"decode_latency_us:  {lat_us:.4f}")
    print(f"ece:                {ece:.6f}")
    print(f"brier:              {brier:.6f}")
    print(f"peak_vram_mb:       {peak_vram_mb:.1f}")
    print(f"train_steps:        {step}")
    print(f"val_shots:          {len(yv)}")
    print(f"mwpm_ref_ler:       {mwpm['logical_error_rate']:.8f}  (d={DISTANCE})")
    print(f"note: naive MLP baseline (negative control); MWPM is the surface-code reference.")

    # log an honest row alongside the MWPM baseline row
    import os
    tsv = os.path.join(os.environ.get(
        "P1_RESULT_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")),
        "results.tsv")
    with open(tsv, "a") as f:
        f.write(f"nn-mlp0\t{ler:.6e}\t{peak_vram_mb/1024:.3f}\trun\t"
                f"naive MLP teacher surface d={DISTANCE} (LER {ler:.2e} >> MWPM {mwpm['logical_error_rate']:.2e}); "
                f"ece={ece:.3f}\n")


if __name__ == "__main__":
    main()
