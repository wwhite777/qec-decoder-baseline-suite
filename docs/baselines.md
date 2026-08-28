# P1 frozen baselines — freeze BEFORE modeling; never change protocol mid-study

| Baseline | Role | Open-source |
|---|---|---|
| PyMatching / Sparse Blossom (MWPM) | surface-code standard | `pip install pymatching` (arXiv:2105.13082) |
| BP-OSD | qLDPC standard | `pip install ldpc bposd` (Roffe et al.) |
| Ambiguity Clustering | qLDPC fast decoder | ref implementation, arXiv:2406.xxxx |
| AlphaQubit / AlphaQubit-2 | NN decoder ref | Nature 2024 / arXiv:2512.07737 (reproduce where possible) |
| Transformer-QEC | transferable transformer | arXiv:2311.16082 |
| Maan–Paler MPNN | qLDPC message-passing | npj QI 11:78 (2025) |
| Blue et al. BB ML decoder | bivariate-bicycle | arXiv:2504.13043 |
| Stim | data generator (not a decoder) | `pip install stim sinter` (Quantum 5:497) |

Only the first three rows are *reference decoders* in this artifact: MWPM is the Arm-A
reference, BP-OSD is the Arm-B/C reference, and Stim is the data generator, not a decoder.
The learned decoders are listed as literature reference points; none of them is
re-implemented or claimed here (see `p1_neural_gap.md`).
