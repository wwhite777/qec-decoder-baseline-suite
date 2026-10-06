# Reference decoders and related decoders

| Decoder or tool | Role in this repository | Source |
|---|---|---|
| MWPM (PyMatching, Sparse Blossom) | **Implemented reference** for Arm A (surface code) | `pip install pymatching`; Higgott and Gidney, *Quantum* 9, 1600 (2025), arXiv:2303.15933 |
| BP-OSD (`ldpc`) | **Implemented reference** for Arms B and C (BB codes) | `pip install ldpc`; Roffe et al., *Phys. Rev. Research* 2, 043423 (2020) |
| Stim | Data generator, not a decoder | `pip install stim`; Gidney, *Quantum* 5, 497 (2021) |
| Ambiguity Clustering | Literature alternative for qLDPC codes (not implemented here) | Wolanski and Barber, arXiv:2406.14527 |
| AlphaQubit and its real-time successor | Learned surface-code decoders (literature only) | Bausch et al., *Nature* 635, 834 (2024); Senior et al., arXiv:2512.07737 |
| Transformer-QEC | Learned decoder (literature only) | Wang et al., arXiv:2311.16082 |
| Learned message passing for qLDPC codes | Learned decoder (literature only) | *npj Quantum Information* 11, 78 (2025), doi:10.1038/s41534-025-01033-w |
| Blue et al. | Learned decoder for BB codes under circuit-level noise (literature only) | *Quantum* 10, 2149 (2026), arXiv:2504.13043 |

Only MWPM and BP-OSD are implemented and used as references here: MWPM for Arm A, and BP-OSD for Arms B and C
(`code/harness.py`, `reference_for`). Stim generates the data. The other rows are literature reference points; none of
them is re-implemented or evaluated in this repository. The two neural negative controls that were trained here are
described in `p1_neural_controls.md`.
