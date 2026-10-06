"""Fixed-N logical-error-rate run for one benchmark configuration (the fixed-sample runs in results/v3).

Each run draws a FRESH validation set from its own seed (never pooled with pilots or with the frozen 2026-07 sets),
decodes every shot with the arm's reference decoder (or a named variant), and writes:
  <out>/<id>.json   counts, exact two-sided equal-tailed 95% Clopper-Pearson interval, seeds, versions, timing, host
  <out>/<id>.fail.npy   packed per-shot failure indicators (np.packbits), for paired comparisons
Arms (definitions identical to the paper and to code/prepare*.py in the repository):
  A  stim rotated_memory_z, rounds=25, all four noise parameters = p; PyMatching on the DEM (decompose_errors=True)
  B  BB code capacity: i.i.d. X errors ~ Bernoulli(p) on data, syndrome H_Z e; BP-OSD on H_Z
  C  BB phenomenological: T noisy rounds (data p, measurement q=p) + final perfect readout; BP-OSD on H_st
Decoder variants (Arms B/C): osd7 = OSD_CS order 7 (reference), osd0 = OSD_0, bp = BP only (ldpc BpDecoder).
Usage: python code/v3/run_config.py '<json config>' <out_dir>   (one line of code/v3/configs_fixed_sample.jsonl)
"""
import hashlib, json, os, platform, socket, sys, time
import numpy as np
import scipy.sparse as sp
from scipy.stats import beta

REPO = os.environ.get("P1_REPO", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "code"))
BP = dict(max_iter=50, bp_method="minimum_sum", ms_scaling_factor=0.625, schedule="serial")


def cp(f, n):
    lo = 0.0 if f == 0 else float(beta.ppf(0.025, f, n - f + 1))
    hi = 1.0 if f == n else float(beta.ppf(0.975, f + 1, n - f))
    return lo, hi


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def make_bp_decoder(H, variant, **kw):
    from ldpc import BpOsdDecoder, BpDecoder
    if variant == "bp":
        return BpDecoder(sp.csr_matrix(H), **BP, **kw)
    order = {"osd7": ("OSD_CS", 7), "osd0": ("OSD_0", 0)}[variant]
    return BpOsdDecoder(sp.csr_matrix(H), **BP, osd_method=order[0], osd_order=order[1], **kw)


def run_arm_a(c):
    import stim, pymatching
    p, d, N, seed, chunk = c["p"], c["d"], c["N"], c["seed"], c.get("chunk", 100_000)
    circ = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=25,
                                  after_clifford_depolarization=p, before_measure_flip_probability=p,
                                  after_reset_flip_probability=p, before_round_data_depolarization=p)
    m = pymatching.Matching.from_detector_error_model(circ.detector_error_model(decompose_errors=True))
    sampler = circ.compile_detector_sampler(seed=seed)
    fails = np.zeros(N, dtype=bool)
    done = 0
    while done < N:
        k = min(chunk, N - done)
        det, obs = sampler.sample(k, separate_observables=True)
        pred = m.decode_batch(det)
        fails[done:done + k] = pred[:, 0].astype(bool) != obs[:, 0].astype(bool)
        done += k
    return fails, {"stim_seed": seed, "chunk": chunk, "num_detectors": circ.num_detectors}


def run_arm_b(c):
    import prepare_qldpc as Q
    code, p, N, seed, chunk = c["code"], c["p"], c["N"], c["seed"], c.get("chunk", 50_000)
    HX, HZ = Q.build_bb_code(code)
    _, LZ = Q.css_logicals(HX, HZ)
    dec = make_bp_decoder(HZ, c.get("decoder", "osd7"), error_rate=p)
    rng = np.random.default_rng(seed)
    n = HX.shape[1]
    fails = np.zeros(N, dtype=bool)
    unsat = 0
    done = 0
    LZt, HZt = LZ.T.astype(np.int64), HZ.T.astype(np.int64)
    while done < N:
        k = min(chunk, N - done)
        err = (rng.random((k, n)) < p).astype(np.uint8)
        syn = (err.astype(np.int64) @ HZt % 2).astype(np.uint8)
        ehat = np.empty_like(err)
        for i in range(k):
            ehat[i] = dec.decode(syn[i])
        unsat += int(((ehat.astype(np.int64) @ HZt % 2) != syn).any(axis=1).sum())
        fails[done:done + k] = (((err ^ ehat).astype(np.int64) @ LZt) % 2).any(axis=1)
        done += k
    return fails, {"numpy_seed": seed, "chunk": chunk, "syndrome_unsatisfied": unsat}


def run_arm_c(c):
    import prepare_qldpc as Q
    import prepare_qldpc_phenom as PH
    code, p, N, seed, chunk = c["code"], c["p"], c["N"], c["seed"], c.get("chunk", 20_000)
    T = c["T"]
    q = p
    HX, HZ = Q.build_bb_code(code)
    _, LZ = Q.css_logicals(HX, HZ)
    m, n = HZ.shape
    Hst = PH.build_spacetime_check(HZ, T)
    chan = PH.channel_prior(n, m, T, p, q)
    from ldpc import BpOsdDecoder, BpDecoder
    v = c.get("decoder", "osd7")
    if v == "bp":
        dec = BpDecoder(sp.csr_matrix(Hst), channel_probs=list(chan), **BP)
    else:
        order = {"osd7": ("OSD_CS", 7), "osd0": ("OSD_0", 0)}[v]
        dec = BpOsdDecoder(sp.csr_matrix(Hst), channel_probs=list(chan), **BP, osd_method=order[0], osd_order=order[1])
    rng = np.random.default_rng(seed)
    HZi = HZ.astype(np.int64)
    LZt = LZ.T.astype(np.int64)
    fails = np.zeros(N, dtype=bool)
    done = 0
    while done < N:
        k = min(chunk, N - done)
        # vectorised phenomenological sampling, identical in law to prepare_qldpc_phenom.generate_dataset
        e = (rng.random((k, T, n)) < p).astype(np.int64)
        f = (rng.random((k, T, m)) < q).astype(np.uint8)
        cum = np.cumsum(e, axis=1) % 2                                  # [k, T, n] accumulated data error
        s_meas = ((cum @ HZi.T) % 2).astype(np.uint8) ^ f               # [k, T, m]
        s_perf = ((cum[:, -1, :] @ HZi.T) % 2).astype(np.uint8)          # [k, m]
        D = np.empty((k, T + 1, m), dtype=np.uint8)
        D[:, 0] = s_meas[:, 0]
        D[:, 1:T] = s_meas[:, 1:] ^ s_meas[:, :-1]
        D[:, T] = s_perf ^ s_meas[:, -1]
        det = D.reshape(k, -1)
        etot = cum[:, -1, :].astype(np.uint8)
        if done == 0:   # self-test: the sampled detectors equal H_st applied to the sampled fault vector
            for i in range(min(50, k)):
                x = np.concatenate([e[i].reshape(-1).astype(np.uint8), f[i].reshape(-1)])
                assert np.array_equal((Hst @ x) % 2, det[i]), "space-time sampler inconsistent with H_st"
        corr = np.empty_like(etot)
        for i in range(k):
            x = dec.decode(det[i])
            corr[i] = np.bitwise_xor.reduce(x[:T * n].reshape(T, n).astype(np.uint8), axis=0)
        fails[done:done + k] = (((etot ^ corr).astype(np.int64) @ LZt) % 2).any(axis=1)
        done += k
    return fails, {"numpy_seed": seed, "chunk": chunk, "T": T, "q": q, "sampler": "vectorised (same law as prepare_qldpc_phenom)"}


def main():
    cfg = json.loads(sys.argv[1])
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    import stim, pymatching, ldpc
    t0 = time.time()
    load0 = os.getloadavg()
    fails, extra = {"A": run_arm_a, "B": run_arm_b, "C": run_arm_c}[cfg["arm"]](cfg)
    dt = time.time() - t0
    f, N = int(fails.sum()), int(fails.size)
    lo, hi = cp(f, N)
    packed = np.packbits(fails)
    fpath = os.path.join(out, cfg["id"] + ".fail.npy")
    np.save(fpath, packed)
    res = {"config": cfg, "shots": N, "failures": f, "ler": f / N, "cp95_lo": lo, "cp95_hi": hi,
           "fail_vector_sha256": hashlib.sha256(packed.tobytes()).hexdigest(), "seconds": dt,
           "loadavg_start": load0, "loadavg_end": os.getloadavg(), "host": socket.gethostname(), "cpu": cpu_model(),
           "versions": {"python": platform.python_version(), "stim": stim.__version__, "pymatching": pymatching.__version__,
                        "ldpc": ldpc.__version__, "numpy": np.__version__}, **extra}
    with open(os.path.join(out, cfg["id"] + ".json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"{cfg['id']}: {f}/{N} = {f / N:.4e} CP95 [{lo:.3e}, {hi:.3e}] in {dt:.1f}s")


if __name__ == "__main__":
    main()
