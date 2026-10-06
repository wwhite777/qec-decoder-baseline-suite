# P1 — Phenomenological Bivariate-Bicycle (BB) Decoding: Full Specification

**Scope.** This document fully specifies the *phenomenological* (multi-round,
measurement-noisy) decoding benchmark for the bivariate-bicycle qLDPC codes, exactly as
implemented in `prepare_qldpc_phenom.py` (frozen data/contract) and consumed by
`baseline_bposd_phenom.py` (BP-OSD baseline). It is written to be precise and to match the
code line-for-line; where the code and this spec would disagree, the code is authoritative
and this spec is the bug.

Everything below is the **Z-sector** decoding problem: the Z-type checks `H_Z` detect
**X data errors**. By CSS symmetry the X-sector (X-checks detecting Z errors) is analogous
and is not separately modelled. This mirrors the code-capacity contract in
`prepare_qldpc.py`, which is the single source of truth for `H_X`, `H_Z`, and the clean
logical basis `L_Z`.

---

## 1. Codes and their algebra (inherited, unchanged)

The BB codes are built in `prepare_qldpc.build_bb_code` and *reused verbatim* by the
phenomenological module (`from prepare_qldpc import build_bb_code, css_logicals, ...`). No
algebra is duplicated in the phenom file.

Over the group `Z_l x Z_m` with cyclic-shift matrices `x = S_l (x) I_m` and
`y = I_l (x) S_m` (Kronecker products; `S_n[i,(i+1) mod n]=1`):

```
A = x^3 + y + y^2   (mod 2)
B = y^3 + x + x^2   (mod 2)
H_X = [ A | B ]                     (m x n)
H_Z = [ B^T | A^T ]                 (m x n)
n = 2*l*m   physical qubits
m = l*m     Z-checks (= number of rows of H_Z)
k = n - rank(H_X) - rank(H_Z)       logical qubits (verified at import)
```

Instances used:

| name  | (l, m) | code         | n   | m (Z-checks) | k  | T (rounds) |
|-------|--------|--------------|-----|--------------|----|------------|
| bb72  | (6, 6) | `[[72,12,6]]`  | 72  | 36           | 12 | 6          |
| bb144 | (12,6) | `[[144,12,12]]`| 144 | 72           | 12 | 12         |

The CSS commutation `H_X H_Z^T = 0 (mod 2)` and `n, k` are checked directly at import via
`code_params()`. The distance `d` is the **published literature value** for these exact
polynomials (Bravyi et al., *Nature* 627, 2024; arXiv:2308.07915) — it is **not**
recomputed here (min-weight logical search needs ILP/GAP). `T = d` is chosen per code
(`ROUNDS = {"bb72": 6, "bb144": 12}`), the standard fault-tolerant choice of decoding as
many noisy rounds as the distance.

`L_Z` is a clean logical basis: `L_Z = ker(H_X) / rowspace(H_Z)`, exactly `k` rows, computed
once by `css_logicals()` (via `ldpc.mod2.kernel` + a rank-raising quotient). It is the
**same** `L_Z` used by the code-capacity contract, so the logical failure criterion is
identical across the two BB arms.

---

## 2. Noise model (phenomenological)

Per shot we simulate `T` rounds of **noisy** syndrome extraction followed by **one final
perfect** readout. Frozen config: `P_SWEEP = [0.02, 0.04, 0.06, 0.08]`, measurement-flip
rate `q := p` (`Q_EQ_P = True`, balanced model), `SEED = 20260709`,
`DEFAULT_SHOTS = 20000`. Two independent noise processes:

1. **Data errors** — before round `r`'s measurement (`r = 0 .. T-1`), each of the `n` data
   qubits independently picks up an X error with probability `p`:
   `e_r ~ Bernoulli(p)^n`, `e_r in F2^n`.
   Errors **accumulate** over rounds: the physical X error present at round `r` is the
   running XOR `cum_r = e_0 ^ e_1 ^ ... ^ e_r`.

2. **Measurement errors** — each round's `m` syndrome-bit measurements are independently
   flipped with probability `q`: `f_r ~ Bernoulli(q)^m`, `f_r in F2^m`, `r = 0 .. T-1`.

The **measured** syndrome in round `r` is the true syndrome of the accumulated data error,
corrupted by that round's measurement flips:

```
s_r = H_Z @ (e_0 ^ ... ^ e_r) ^ f_r  = H_Z @ cum_r ^ f_r        (mod 2),   r = 0..T-1
```

In code (`generate_dataset`): `cum = (cum + e[r]) % 2`, then
`s_meas[r] = ((HZi @ cum) % 2) ^ f[r]`.

**Final perfect readout.** After the `T` noisy rounds, one noiseless stabiliser
measurement is performed (no `f` term):

```
s_T = H_Z @ (e_0 ^ ... ^ e_{T-1}) = H_Z @ cum_{T-1}   (mod 2)
```

In code: `s_perfect = (HZi @ cum) % 2` with `cum` at its final value `cum_{T-1}`. This is
the standard phenomenological **temporal boundary condition**: without it, the last round's
measurement flips and any late data errors would be undetectable. The perfect final round
"closes" the space-time detector graph in time.

---

## 3. Detectors (space-time syndrome the decoder actually sees)

The decoder does **not** see the raw measured syndromes `s_r`; it sees **detectors** =
XOR of temporally-consecutive syndromes (a detector equals 0 in the absence of noise, which
is what makes it a valid parity check on the noise variables). There are `(T+1)` detector
blocks of `m` bits each, hence `(T+1)*m` detector bits:

```
D_0 = s_0                = H_Z e_0 ^ f_0
D_r = s_r ^ s_{r-1}      = H_Z e_r ^ f_r ^ f_{r-1}      (r = 1 .. T-1)
D_T = s_T ^ s_{T-1}      = f_{T-1}                        (final perfect round, no data/e term)
```

Derivation of `D_r` for `1 <= r <= T-1`:
`s_r ^ s_{r-1} = (H_Z cum_r ^ f_r) ^ (H_Z cum_{r-1} ^ f_{r-1})`
`= H_Z (cum_r ^ cum_{r-1}) ^ f_r ^ f_{r-1} = H_Z e_r ^ f_r ^ f_{r-1}`
(since `cum_r ^ cum_{r-1} = e_r`).
Derivation of `D_T`: `s_T ^ s_{T-1} = H_Z cum_{T-1} ^ (H_Z cum_{T-1} ^ f_{T-1}) = f_{T-1}`.

In code (`generate_dataset`):
```
D[0] = s_meas[0]
D[r] = s_meas[r] ^ s_meas[r-1]     for r = 1..T-1
D[T] = s_perfect ^ s_meas[T-1]
detectors = D.reshape(-1)          # length (T+1)*m, block-major in time
```

The ground-truth label stored per shot is the **total accumulated data error**
`err_total = cum_{T-1} = e_0 ^ ... ^ e_{T-1}` (uint8, length `n`). This is the quantity the
decoder must ultimately correct — not the individual per-round `e_r`.

---

## 4. Space-time parity-check matrix `H_st`

The unknowns are the concatenation of all per-round data errors and all per-round
measurement flips, ordered:

```
unknown vector  u = [ e_0 .. e_{T-1} | f_0 .. f_{T-1} ]   in F2^(T*n + T*m)
                     \___ T*n cols __/ \___ T*m cols __/
```

`H_st` is the `(T+1)*m x (T*n + T*m)` sparse binary matrix such that
`H_st @ u = detectors (mod 2)`. Its row-block `d` (for `d = 0 .. T`) encodes detector `D_d`
above. Built by `build_spacetime_check(HZ, T)`:

- **e-columns** (block `r`, an `m x n` sub-block): equals `H_Z` iff `d < T and r == d`,
  else the `m x n` zero block. (Detector `D_d` for `d<T` contains `H_Z e_d`; the perfect
  final detector `D_T` has no data term.)
- **f-columns** (block `r`, an `m x m` sub-block): the `m x m` identity `I_m` iff that
  round's flip appears in detector `D_d`, else zero. Concretely:
  - for `d < T`: `I_m` at `r == d` (the `f_d` term) and, if `d >= 1`, also at `r == d-1`
    (the `f_{d-1}` term);
  - for `d == T`: `I_m` only at `r == T-1` (the lone `f_{T-1}` term of `D_T`).

This reproduces exactly the three detector equations in Section 3. The rows are stacked with
`scipy.sparse.vstack`; each row-block is `hstack([e-parts ...] + [f-parts ...])`. The module
self-check prints `H_st.shape`, its GF(2) rank, and the dimension bookkeeping
(`det_bits = (T+1)*m`, `err_vars = T*n + T*m`).

**Per-column channel prior.** BP needs a prior fault probability per column
(`channel_prior(n, m, T, p, q)`): `p` on all `T*n` data (e) columns and `q` on all `T*m`
measurement (f) columns (floored at `1e-12` to avoid log(0)). Since `q = p` in the frozen
model these are equal in magnitude, but the code keeps them separate so an unbalanced model
(`q != p`) is a one-line change.

---

## 5. How BP-OSD consumes the syndrome history

`baseline_bposd_phenom.make_bposd_phenom` builds one `ldpc.BpOsdDecoder` over the sparse
`H_st` with `channel_probs = channel_prior(...)` and hyperparameters **identical** to the
code-capacity baseline (min-sum BP, `ms_scaling_factor = 0.625`, serial schedule,
`max_iter = 50`, `osd_method = "OSD_CS"`, `osd_order = 7`) so the two BB arms differ only in
the noise model, not the decoder tuning.

Decoding one shot:

1. Input: the length-`(T+1)*m` detector vector for that shot (the full syndrome **history**,
   flattened block-major in time).
2. `corr = dec.decode(detectors)` returns a length-`(T*n + T*m)` correction over the full
   unknown vector `u` — i.e. it jointly infers **every per-round data error and every
   measurement flip** that best explains the whole space-time syndrome (BP over `H_st`, with
   an OSD post-process on the columns BP is least sure about when BP does not converge).
3. Project to the physical data correction: take the first `T*n` entries (the `e` part),
   reshape to `(T, n)`, and XOR over rounds:
   `Ê = XOR_r ê_r` (`np.bitwise_xor.reduce(e_dec, axis=0)`), a length-`n` net X-correction
   on the data qubits. The measurement-flip estimates `f̂_r` are inferred but discarded — they
   are nuisance variables that only help explain the syndrome.

The batched `decode_fn` loops this over shots (one `dec.decode` call per shot; **no batch
vectorization** inside `ldpc`, which is exactly why the online-latency script measures it
one shot at a time).

---

## 6. The logical test (failure criterion)

Ground-truth total data error `E_true = err_total = cum_{T-1}`. Residual after correction:
`R = E_true ^ Ê`. A shot is a **logical failure** iff the residual anticommutes with any
Z-logical:

```
logical failure  <=>  L_Z @ R != 0  (mod 2)     (any of the k components nonzero)
```

Implemented in `prepare_qldpc_phenom._logical_failures`:
`la = (resid @ L_Z^T) % 2; fail = any(la != 0 along the k axis)`. This is the **identical**
criterion (same `L_Z`) as the code-capacity contract, so "did the logical qubit survive" is
measured consistently across both BB arms. It correctly ignores stabiliser-equivalent
residuals (a residual in `rowspace(H_Z)` commutes with all of `L_Z` and is *not* a failure).

LER `= failures / shots`. `evaluate_ler_phenom` returns `{LER, failures, shots, p, q, T,
code}`; the baseline additionally reports Wilson 95% lo/hi (and this rigor pass adds exact
Clopper-Pearson intervals — see `p1_intervals.py`).

---

## 7. Determinism, caching, and reported latency

- The whole phenom val set is a **frozen artifact**: fixed `SEED = 20260709`,
  `np.random.default_rng(SEED)`, cached to
  `~/.cache/ququ_p1_qec/phenom_<code>_T{T}_p{p}_q{q}_s{shots}_seed..._<md5>.npz`. Re-running
  reuses the cache byte-for-byte, so BP-OSD and any neural decoder are evaluated on
  **exactly the same** detector/label arrays.
- `measure_latency_us` in the frozen contract times a **whole batch** and divides by shots →
  that is amortized throughput **per multi-round shot** (divide by `T` for per-round).
  For the operationally meaningful **online** latency see `p1_latency.py`, which times each
  shot's `dec.decode` separately and reports p50/p95/p99 (the OSD fallback gives a heavy
  right tail).

---

## 8. What this model is NOT (honest boundary)

This is **phenomenological**, not **circuit-level**. It models data errors + measurement
errors round-by-round with a perfect temporal boundary, which is strictly harder than
code-capacity and much closer to reality. It does **not** model:

- a full Stim BB **syndrome-extraction circuit** with the explicit ancilla-qubit CNOT
  schedule,
- **hook errors** (a single ancilla fault spreading to multiple data qubits through the
  measurement circuit),
- gate/idle **depolarizing** noise, correlated `Y` errors, or leakage,
- the space-time **decomposition** of circuit faults into a detector-error-model
  (`decompose_errors=True`), which is what the surface-code arm uses via Stim.

Consequently the phenom BB arm and the circuit-level surface arm are **separate benchmark
arms with different noise models and different physical error-rate axes** and must not be
compared point-for-point. See `p1_arms.md`. Circuit-level BB (Stim DEM + BP-OSD over the
circuit detectors) is the explicit, larger next step and is intentionally out of scope here.
