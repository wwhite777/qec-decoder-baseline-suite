# Neural negative controls: records and hypotheses

Two code-blind learned decoders were trained in July 2026 and are reported as **negative controls**, not as
competitive decoders. Each is a single recorded run; neither run saved its weights or per-shot predictions, so neither
can be re-scored or compared with the paired test of `code/harness.py`.

## Records

| Control | Setting | Recorded LER | Reference on the same shots | Notes |
|---|---|---|---|---|
| Residual MLP (4 blocks, width 512) on the flattened detection events | Arm A: surface code, `d = 5`, `p = 1e-3` | 0.08265 on 20,000 held-out shots (1,653 failures; 95% interval [7.89e-2, 8.66e-2]) | MWPM: 14 failures on the same shots, 7.0e-4 (95% interval [3.8e-4, 1.2e-3]; low count) | trained 120 s on the other 80,000 shots of the frozen `d = 5` set (NumPy seed 0 permutation); ratio of point estimates about 118x. On all 100,000 shots MWPM gives 7.8e-4 (unpaired context only). Always predicting "no flip" on the held-out shots gives 0.209. |
| Gated delta-rule sequence mixer (learned positional embeddings, mean-pool, linear head over 12 logical bits) | Arm B: `[[72,12,6]]`, code capacity, `p = 0.04` | 0.50105 (mean per-logical-bit accuracy about 0.86) | BP-OSD on the full frozen 50,000-shot set: 7.94e-2 (not the same shots, see notes) | trained 300 s on 200,000 freshly sampled shots. The run did not record which shots it was scored on; 0.50105 is not a multiple of 1/50,000, so it was not the whole frozen set, and the ratio (about 6x) is indicative only. |

## Possible reasons for the gap (untested hypotheses)

No ablations were run, so the following are hypotheses, not findings:

1. Neither model is given the code structure. The mixer reads the syndrome bits in a fixed order with learned
   positional embeddings, but has no Tanner-graph adjacency or qubit–check incidence.
2. The mixer predicts the joint 12-bit logical-flip vector directly and is scored all-or-nothing. BP-OSD solves for a
   physical correction and derives the logical class.
3. The training budgets (120 s and 300 s) are far smaller than those of published learned decoders. For example,
   AlphaQubit was pretrained on up to 2 x 10^9 synthetic samples and then finetuned on experimental data.

Testing these would need seed-repeated ablations of the input representation, the target and the training budget, with
per-shot predictions saved so that the paired test applies.

## Candidate routes to a competitive learned decoder

These are directions, not results:
- An AlphaQubit-style recurrent transformer with detector-graph structure for the surface code.
- A Tanner-graph message-passing or spatiotemporal network for the BB codes. Alternatively, a recurrent transformer, as in
  Blue et al. (*Quantum* 10, 2149, 2026), which reports a transformer that outperforms BP-OSD on `[[72,12,6]]` under
  circuit-level noise.
- Ambiguity Clustering (Wolanski and Barber, arXiv:2406.14527), a fast classical qLDPC decoder on CPUs. It is another
  reference to add.

Whatever a new decoder wins on (LER, latency or calibration), compare it with the arm's reference:
- with the paired test on the same frozen shots;
- with single-frame call latency measured as in `code/harness.py`, one frame per call, including any host–device transfer.
