# Fixed-sample runs and latency sessions: protocol

This protocol was written and frozen on 2026-10-06, before any of the runs in `results/v3/`. The configuration files
are part of it:

| file | sha256 |
|---|---|
| `code/v3/configs_fixed_sample.jsonl` (34 configurations) | `d43f8c2595e5aabe70cbc188040b209b75104447853d280fade2b564abb3d1e9` |
| `code/v3/configs_latency.jsonl` (9 configurations) | `8ced9e169ba11a640887718f8faba7890d7d103c13294391d854e4262b3d859c` |

Short pilot runs, on their own seeds, sized the shot counts and checked throughput. They are not pooled with anything
and are not reported.

## Fixed-sample runs

- Each of the 34 configurations has its number of shots `N` fixed in advance. There is no early stopping and no
  extension after looking at a result.
- Every run draws fresh shots from its own seed (20261006000 to 20261006027). These seeds are never reused from the
  original frozen validation sets or from the pilots.
- Configurations:
  - Arm A: rotated surface code, 25 rounds, `d` in {5, 7, 9, 11} x `p` in {0.001, 0.002, 0.003, 0.005}.
  - Arm B: `[[144,12,12]]` at `p = 0.02`, the point with 2 failures in the frozen set.
  - Arm C: `[[72,12,6]]`, `T = 6`, `p = q` in {0.002, 0.005, 0.01, 0.015, 0.02}; and `[[144,12,12]]`, `T = 12`,
    `p` in {0.005, 0.01, 0.02}.
  - Paired decoder variants on shared seeds, so that the variants decode identical frames: `osd7` (OSD_CS of order 7, the
    reference), `osd0` (OSD_0) and `bp` (BP only). The settings are Arm B `[[72,12,6]]` at `p = 0.04` (`N = 100,000`)
    and Arm C `[[72,12,6]]`, `T = 6`, at `p = 0.01` (`N = 50,000`) and `p = 0.02` (`N = 20,000`).
- Arm C frames come from a vectorised sampler with the same law as `code/prepare_qldpc_phenom.py`. On the first 50 frames
  of every run it checks that the sampled detectors equal `H_st` applied to the sampled fault vector.

## Reporting rules

- Each configuration is reported as `f/N`, with the exact two-sided, equal-tailed 95% Clopper–Pearson interval.
  - `f = 0`: the interval's upper end is reported, stated as such. The one-sided 95% limit is mentioned once.
  - `f < 30`: the point is flagged as low-count. The estimate is shown with its interval.
- Every planned configuration is reported, including zero-failure and unfavourable ones.
- Fixed-sample results are reported separately from the frozen-set results and are never pooled with them.
- Paired comparisons:
  - Exact McNemar test: a two-sided binomial test on the discordant shots `b` (variant fails, reference succeeds) and
    `c` (the reverse).
  - The family is {osd0 vs osd7, bp vs osd7} x {B at `p = 0.04`, C at `p = 0.01`, C at `p = 0.02`}: six tests,
    Holm-adjusted at alpha = 0.05.
  - Reported: `b`, `c`, `N`, `delta = (b - c)/N`, and the raw and Holm-adjusted p-values.
  - No claim is made from whether marginal intervals overlap.
- No threshold is estimated. The curves describe the finite-size behaviour of the measured configurations.

## Latency sessions

- 9 configurations x 5 sessions.
  - Each session is a fresh process pinned to one CPU core, with the configuration order shuffled per session.
  - Frames are fresh, drawn from a session-specific seed: 500 untimed warm-up calls, then 10,000 timed calls of one
    frame each.
  - The timed region is the decoder call alone. Sampling, decoder construction and scoring are outside it.
- Reported:
  - per session: `n`, mean, p50, p90, p95, p99 and max;
  - distribution-free (order-statistic) 95% intervals for p50, p95 and p99, which assume i.i.d. timings;
  - the range across sessions;
  - for BP-OSD, the number of calls on which BP did not converge, and the share of the slowest 5% of calls that did not
    converge;
  - the load average at the start and end of each configuration, and the CPU model.
- The sessions ran after the fixed-sample runs had finished. The machine is shared, so absolute times are specific to
  the host and its load.

## Not changed

- The frozen-set results of the original study stay reported as measured.
- The neural controls cannot be re-scored, because their weights were not saved.
- There is no circuit-level BB setting.

## Scripts

| script | role |
|---|---|
| `code/v3/run_config.py` | one fixed-sample run: counts, interval, packed per-shot failure vector |
| `code/v3/driver.py` | runs every configuration of a JSONL file in a process pool |
| `code/v3/latency_session.py` | one latency session |
| `code/v3/paired_tests.py` | the six paired tests from the failure vectors, with Holm's adjustment |
| `code/v3/export_v3_results.py` | collects run and session outputs into `results/v3/` with `SHA256SUMS` |

These copies differ from the ones that produced `results/v3/` only in path handling: the default repository root, and
the interpreter the driver calls. Two configurations re-run from these copies (`A_d5_p0.005` and
`pair_C_bb72_T6_p0.02_osd7`) reproduced their recorded failure vectors exactly with the package versions listed in
`results/v3/fixed_sample_runs.csv`.
