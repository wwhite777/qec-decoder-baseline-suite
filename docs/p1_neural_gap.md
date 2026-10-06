# P1 — Honest Assessment of the Neural QEC Baselines (the "strawman" gap)

> Status: a scoping note written in July 2026. Its effort figures are rough estimates and its
> causal statements are hypotheses (no ablations were run). Since then, Blue et al.
> (*Quantum* 10, 2149, 2026) reported a transformer decoder that beats BP-OSD on `[[72,12,6]]`
> under circuit-level noise.

**Verdict.** The neural decoders in this repository are **strawmen**. They lose to the classical references by wide margins in logical
error rate (about two orders of magnitude for the MLP, about 6x for the mixer), and no honest paper can present them as competitive
decoders. This document (a) states exactly how far off they are, (b) explains *why* they are
weak (it is architectural/setup, not a coincidence), and (c) scopes what a genuinely
competitive neural QEC baseline would take. It does **not** propose to hack the existing
models into looking good — a weak decoder dressed up as competitive would be worse than no
neural arm at all.

---

## 1. The current neural results (measured, from `results.tsv`)

| Neural model | Arm | Config | Neural LER | Reference LER | Gap |
|---|---|---|---|---|---|
| Naive MLP "teacher" | A: surface, circuit-level | `d=5`, `p=1e-3` | **8.27e-2** | MWPM **7.8e-4** | **~106x worse** |
| DeltaNet-style linear-attention mixer | B: BB code-capacity | bb72, `p=0.04` | **5.01e-1** | BP-OSD **7.94e-2** | **~6x worse (indicative; see below)** |

Supporting detail:
- The MLP (`nn-mlp0`) has ECE 0.073 and sits two orders of magnitude above MWPM — it is not
  in the same regime as the baseline at all.
- The MLP was trained on 80,000 shots of the frozen `d=5` set and scored on the other 20,000;
  on those same shots MWPM fails 14 times (7.0e-4) and always predicting "no flip" gives 0.209.
- The DeltaNet decoder (`nn-dnet0`) reaches only ~0.86 mean *per-logical-bit* accuracy on `k=12`
  bits and gets the whole 12-bit logical-flip vector wrong on about half of the shots (any bit
  wrong = failure). Its errors are correlated across bits (independent errors at 0.86 per bit
  would give an LER near 0.84). The run recorded its LER (0.50105) but not the shots it was
  scored on; 0.50105 is not a multiple of 1/50,000, so it was not the whole frozen set and the
  ratio to BP-OSD is indicative only.

These are honest negative results. The right thing to report is that **off-the-shelf
models, trained for minutes (120 s for the MLP, 300 s for the mixer) on a flattened syndrome,
do not decode these codes** — not to bury them or re-label them as wins.

---

## 2. Possible causes (hypotheses; no ablations were run)

Each control is a single run, so the causes below are hypotheses. Testing them needs
seed-repeated ablations of the input representation, the target and the training budget.

1. **No code structure in the model.** Both models ingest the syndrome as a bare bit vector
   (`bit_embed + pos_embed`, mean-pool, linear head). They are given **none** of the
   Tanner-graph / parity-check adjacency `H_Z`, no qubit<->check incidence, no code geometry.
   Competitive neural decoders are **graph/geometry-conditioned** (they run message passing
   on the code's Tanner graph or feed the model the stabiliser structure). The mixer sees the
   syndrome bits in a fixed order with learned positional embeddings, but it has to learn the
   syndrome-to-logical map without any code structure, from minutes of training.
2. **Wrong output target for BB.** The DeltaNet head predicts the `k`-bit **logical-flip
   vector** `L_Z e` directly and is scored as all-or-nothing over 12 bits. That is a very
   high-variance target; BP-OSD instead solves for a **physical coset representative** `ê` and
   *derives* the logical class. Predicting the joint 12-bit flip directly, without the
   physical-correction inductive bias, is close to hopeless at these code sizes.
3. **No real training budget / scale.** 300 s single-GPU with 96-dim width and a mean-pool is
   far below what published NN decoders use (AlphaQubit: large recurrent-transformer,
   pretrained on ~10^9 samples, then fine-tuned on device data). The comparison "300 s
   DeltaNet vs a mature BP-OSD" is not a fair decoder comparison; it is a probe of zero-shot
   architecture transfer, which the logs already (honestly) flag.
4. **Single-shot only for BB.** The neural bonus runs **code-capacity** BB (Arm B). The
   realistic decoding problem (Arm C, phenomenological / space-time) is where a learned
   decoder could plausibly add value, and it was never given a spatiotemporal model matched
   to that arm.
5. **No calibration/uncertainty machinery** — the MLP just reports an ECE; there is no temperature/conformal/evidential head that would make
   the "calibrated decoder" claim meaningful.

Conclusion: these are **lower bounds on what a naive setup does**, useful as sanity baselines
("a code-blind mixer fails"), but they are **not** the neural baseline the paper's thesis
needs.

---

## 3. What a genuinely competitive neural QEC baseline requires

Two credible targets, matched to the two families. Effort estimates assume the existing
frozen data/evaluators are reused (they are), a single L40S-class GPU, and one competent
engineer.

### 3a. Surface arm (Arm A) — AlphaQubit-style recurrent-transformer decoder
- **Architecture.** Per-round syndrome embedded with **detector-graph structure** (which
  detector belongs to which stabiliser / space-time location), a **recurrent** core
  (transformer or Mamba/SSM over the 25 rounds) that maintains a belief state across rounds,
  and a logical-observable head. This is the class that actually matches/beats MWPM on
  surface codes (Bausch et al., *Nature* 2024, "Learning high-accuracy error decoding for
  quantum processors"; "AlphaQubit").
- **Training.** Large synthetic corpus from Stim (the frozen `prepare.py` circuit generalizes
  to arbitrarily many shots) — realistically **10^7-10^9** syndrome samples, curriculum over
  `p`, plus optional fine-tuning on real device data. The 300 s fixed budget must be
  **dropped** for the baseline (it is a budget-matching knob, not a way to train a reference
  decoder); train to convergence (hours-to-days).
- **What "competitive" means here.** Match MWPM's LER at `d=7,9,11` and ideally beat it at
  larger `d` (the published AlphaQubit result). Report per-round LER with CIs, and **online**
  latency the same way as `code/p1_latency.py`.
- **Effort (rough estimate, not measured):** large, roughly weeks of engineering plus GPU-days
  for a faithful re-implementation and training run.

### 3b. BB/qLDPC arms (Arms B/C) — the honest hard case
Two realistic routes, both nontrivial:
- **(i) Ambiguity Clustering (AC)** — a recent *fast, accurate* qLDPC decoder
  (Wolanski & Barber / Riverlane, arXiv:2406.14527-class). It is a BP + clustering method,
  **not** a neural net, but it is the correct "strong fast baseline" the paper should compare
  against instead of (or alongside) a neural model. Reimplementing AC faithfully is **medium**
  effort (~1-2 engineer-weeks) and would immediately raise the bar above BP-OSD on latency.
- **(ii) A graph/message-passing neural decoder** matched to the Tanner graph — e.g. a MPNN
  over the BB Tanner graph (Maan-Paler-style, *npj QI* 11:78 2025) or the BB-specific ML
  decoder of Blue et al. (arXiv:2504.13043), and for Arm C a **spatiotemporal** GNN over the
  space-time `H_st` graph. This is a natural neural family to try, because it is
  code-structure-aware. **Large** effort
  (~3-4 engineer-weeks incl. getting it to actually match BP-OSD, which is not guaranteed).
- **Reality check.** Whether a learned decoder can beat BP-OSD on BB code-capacity LER is open;
  the honest headline for a neural qLDPC decoder is usually a **latency/throughput** or
  **calibration** win at matched LER, not an accuracy win. The paper should be scoped so the
  contribution survives a *tie* on LER (Pareto: LER-matched but faster/better-calibrated).

---

## 4. Recommendation for the paper

1. **Do not present the current MLP/DeltaNet as competitive decoders.** Either (a) cut them,
   or (b) keep them explicitly labelled as **negative controls** ("a code-blind sequence
   mixer under a 300 s budget fails to learn the decoding map — motivating code-structured
   models"), with the exact gaps from Section 1 stated.
2. **Add at least one strong non-trivial baseline** before any neural-vs-classical claim: on
   BB, that is **Ambiguity Clustering** (medium effort) as the fast reference beyond BP-OSD.
3. **If a neural decoder is to be a headline**, budget the AlphaQubit-style (surface) and/or
   Tanner-graph MPNN (BB) build honestly (rough estimate: weeks + GPU-days, no 300 s cap), and
   pre-register the target (LER, latency or calibration) before running.
4. **Every comparison stays per-arm** (see `docs/p1_arms.md`) and every LER carries its
   binomial CI (see `code/p1_intervals.py`), so a neural "win" cannot hide inside an unresolved rare-event
   point.

**Bottom line:** the classical baselines (MWPM, BP-OSD code-capacity + phenomenological) are
sound and now properly quantified (CIs + online latency). The neural side is currently a
strawman; making it competitive would likely take weeks of work and GPU-days (rough estimate),
and this document is the honest scoping of that gap rather than a fake fix.
