# P1 — Benchmark Arm Separation (noise models are NOT comparable across families)

**One-line summary.** The P1 experiments live in **three separate benchmark arms** that use
**different noise models** and therefore have **different, non-comparable physical
error-rate axes**. Any claim about "neural decoding" must be made **per arm**; a decoder that
wins on one arm has said nothing about the others, and LER numbers must never be compared
point-for-point across arms.

---

## The three arms

| Arm | Code family | Decoder reference | Noise model | `p` axis / config | Data source | Result CSV |
|-----|-------------|-------------------|-------------|-------------------|-------------|------------|
| **A. Surface (circuit-level)** | rotated surface code, `d = 5,7,9,11` | MWPM (`pymatching`) | **Circuit-level** depolarizing: `after_clifford_depolarization`, `before_measure_flip`, `after_reset_flip`, `before_round_data_depolarization`, all `= p`, over `ROUNDS = 25` rounds; decoded on the Stim **detector-error-model** (`decompose_errors=True`) | `p = 1e-3` (single pinned point) | Stim (`prepare.py`), `STIM_SEED=20260708`, 100k shots | `results/exp1/p1_mwpm_baseline.csv` |
| **B. BB (code-capacity)** | bivariate-bicycle `[[72,12,6]]`, `[[144,12,12]]` | BP-OSD (`ldpc`) | **Code-capacity**: single shot, i.i.d. X error `~Bernoulli(p)` on data qubits, **perfect** syndrome measurement, no time dimension | `p in {0.02,0.04,0.06,0.08}` | `prepare_qldpc.py`, `SEED=20260708`, 50k shots | `results/exp3/p1_qldpc_bposd.csv` |
| **C. BB (phenomenological)** | `[[72,12,6]]` reported (the code also supports `[[144,12,12]]`; no results reported) | BP-OSD over space-time `H_st` (`ldpc`) | **Phenomenological**: `T=d` noisy rounds, data-error rate `p`/round **and** measurement-flip rate `q=p`/round, + 1 final perfect readout; decode the `(T+1)*m` detector history | `p = q in {0.02,...,0.08}`, `T = 6` (reported; 2,000 shots per point) | `prepare_qldpc_phenom.py`, `SEED=20260709` | `results/exp3/p1_qldpc_phenom.csv` |

---

## Why the arms are not directly comparable

1. **Different physical error channels.** Arm A's `p` is a *circuit-level* gate/measurement/
   reset error injected at every operation across 25 rounds and then **decomposed** into
   detector-error-model edges; a single circuit fault can light up several detectors (and,
   via hook errors, corrupt several data qubits). Arm B's `p` is a *code-capacity* i.i.d.
   data-qubit flip with **noiseless** readout and no time. Arm C's `p` is a *phenomenological*
   per-round data flip **plus** an independent per-round measurement flip `q`. **The same
   numeric `p` means three different physical things.** A code-capacity `p=0.02` is far more
   benign than a circuit-level `p=0.02` would be, and Arm A is only ever evaluated at
   `p=1e-3`. Plotting or ranking LERs across arms on a shared `p` axis is meaningless.

2. **Different information reaching the decoder.** Arm A/C decoders reason over a
   **space-time** detector graph (time-correlated syndromes); Arm B decoders see a **single**
   perfect syndrome (no time). The decoding *problem* — not just its difficulty — differs.

3. **Graphlike matching does not apply directly to BB codes.** The surface arm's reference
   (MWPM) is the standard strong graphlike decoder: the surface code's detector error model
   decomposes into graphlike edges. In the BB check matrices each qubit is in three checks, so
   matching cannot be applied to them directly, and BP-OSD is the reference.
   There is therefore **no single classical baseline** spanning the arms; each arm has its
   own bar to beat.

4. **Different logical-failure bookkeeping.** Arm A scores a logical **observable** flip from
   the Stim circuit (`observables`); Arms B/C score `L_Z @ residual != 0` against a clean CSS
   logical basis. These are both "did the logical qubit survive" but are computed from
   different objects on different codes.

---

## Rules for any cross-family / neural-decoding claim

- **State the arm.** Every LER, latency, threshold, or "beats BP-OSD/MWPM" statement must
  name its arm (A/B/C) and its exact `(code, p, [q, T, rounds])`. No bare "our decoder
  achieves LER X".
- **Compare only within an arm.** A neural decoder is compared to **the reference decoder of
  the same arm on the same frozen val set** (same seed, same shots) — MWPM for A, BP-OSD
  (code-capacity) for B, BP-OSD (space-time) for C. Cross-arm LER comparisons are prohibited.
- **"Cross-family transfer" must be per-arm.** A surface <-> BB transfer claim is legitimate
  **only** as: train once, then evaluate **separately** on each family's own arm against that
  arm's own baseline. A single aggregate "cross-family LER" is not a valid quantity.
- **Latency comparisons must use the same measurement mode.** Online per-shot latency
  (`p1_latency.py`) vs online; batch throughput vs batch. Never compare a GPU batch-throughput
  number to a CPU online-latency number (see the latency note).
- **Circuit-level BB is a missing arm.** There is currently **no** circuit-level BB arm
  (Stim BB syndrome-extraction circuit + DEM + BP-OSD). Until it exists, the surface
  circuit-level result (Arm A) cannot stand in for "circuit-level qLDPC", and the honest
  framing is: surface = circuit-level, BB = code-capacity/phenomenological. This is the
  single biggest apples-to-oranges risk in the paper and must be stated explicitly.

---

## Recommended figure/table hygiene

- Never place Arm A and Arm B/C LERs on the same axis. Use **separate panels** per arm, each
  labelled with its noise model.
- Put the **noise model in every caption** ("circuit-level, `p=1e-3`, 25 rounds" vs
  "code-capacity, single-shot" vs "phenomenological, `T=d`, `q=p`").
- Report every LER with its exact binomial CI (`p1_intervals.py`) and its failure count; flag
  under-resolved points (e.g. bb144 code-capacity `p=0.02`: 2/50000) as bounds, not values.
