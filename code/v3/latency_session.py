"""One latency session (fresh process): single-frame decoder-call wall-clock latency for every config, in a session-specific
random order, with the process pinned to one CPU core.

Per config: draw 10,500 fresh frames from the config's task distribution (session-specific seed), run 500 untimed warm-up
calls on frames 0..499, then time frames 500..10,499 one call at a time with time.perf_counter (the timing boundary is
the decoder call alone: no sampling, no decoder construction, no scoring inside the timed region). For BP-OSD decoders the
`converge` flag is read after each call (outside the timed region).
Usage: python code/v3/latency_session.py <session_index> code/v3/configs_latency.jsonl <out_dir> <core>
(Linux only: os.sched_setaffinity pins the process to <core>.)
"""
import json, os, platform, random, socket, sys, time
import numpy as np
import scipy.sparse as sp
from scipy.stats import binom

REPO = os.environ.get("P1_REPO", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "code"))
BP = dict(max_iter=50, bp_method="minimum_sum", ms_scaling_factor=0.625, schedule="serial")
WARM, TIMED = 500, 10_000


def qci(x_sorted, q, alpha=0.05):
    """Distribution-free (order-statistic) 95% CI for the q-quantile; assumes i.i.d. stationary timings."""
    n = len(x_sorted)
    lo_r = max(1, int(binom.ppf(alpha / 2, n, q)))
    hi_r = min(n, int(binom.ppf(1 - alpha / 2, n, q)) + 1)
    return float(x_sorted[lo_r - 1]), float(x_sorted[hi_r - 1])


def frames_and_decoder(c, seed, nframes):
    if c["arm"] == "A":
        import stim, pymatching
        p, d = c["p"], c["d"]
        circ = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=25,
                                      after_clifford_depolarization=p, before_measure_flip_probability=p,
                                      after_reset_flip_probability=p, before_round_data_depolarization=p)
        m = pymatching.Matching.from_detector_error_model(circ.detector_error_model(decompose_errors=True))
        det, _ = circ.compile_detector_sampler(seed=seed).sample(nframes, separate_observables=True)
        return det.astype(np.uint8), m.decode, None
    import prepare_qldpc as Q
    from ldpc import BpOsdDecoder, BpDecoder
    HX, HZ = Q.build_bb_code(c["code"])
    v = c.get("decoder", "osd7")
    rng = np.random.default_rng(seed)
    if c["arm"] == "B":
        n = HX.shape[1]
        err = (rng.random((nframes, n)) < c["p"]).astype(np.uint8)
        frames = (err.astype(np.int64) @ HZ.T.astype(np.int64) % 2).astype(np.uint8)
        H, kw = HZ, {"error_rate": c["p"]}
    else:
        import prepare_qldpc_phenom as PH
        T, p = c["T"], c["p"]
        m_, n = HZ.shape
        e = (rng.random((nframes, T, n)) < p).astype(np.int64)
        f = (rng.random((nframes, T, m_)) < p).astype(np.uint8)
        cum = np.cumsum(e, axis=1) % 2
        s_meas = ((cum @ HZ.T.astype(np.int64)) % 2).astype(np.uint8) ^ f
        s_perf = ((cum[:, -1, :] @ HZ.T.astype(np.int64)) % 2).astype(np.uint8)
        D = np.empty((nframes, T + 1, m_), dtype=np.uint8)
        D[:, 0] = s_meas[:, 0]; D[:, 1:T] = s_meas[:, 1:] ^ s_meas[:, :-1]; D[:, T] = s_perf ^ s_meas[:, -1]
        frames = D.reshape(nframes, -1)
        H, kw = PH.build_spacetime_check(HZ, T), {"channel_probs": list(PH.channel_prior(n, m_, T, p, p))}
    if v == "bp":
        dec = BpDecoder(sp.csr_matrix(H), **BP, **kw)
    else:
        meth, order = {"osd7": ("OSD_CS", 7), "osd0": ("OSD_0", 0)}[v]
        dec = BpOsdDecoder(sp.csr_matrix(H), **BP, osd_method=meth, osd_order=order, **kw)
    return frames, dec.decode, dec


def main():
    s, cfg_path, out, core = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
    os.sched_setaffinity(0, {core})
    os.makedirs(out, exist_ok=True)
    cfgs = [json.loads(l) for l in open(cfg_path) if l.strip()]
    order = list(range(len(cfgs)))
    random.Random(1000 + s).shuffle(order)
    import stim, pymatching, ldpc
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "")
    for pos, i in enumerate(order):
        c = cfgs[i]
        seed = 20261006500 + 100 * s + i
        frames, decode, dec = frames_and_decoder(c, seed, WARM + TIMED)
        for k in range(WARM):
            decode(frames[k])
        t = np.empty(TIMED); conv = np.ones(TIMED, dtype=bool)
        load0 = os.getloadavg()
        for k in range(TIMED):
            fr = frames[WARM + k]
            t0 = time.perf_counter(); decode(fr); t[k] = (time.perf_counter() - t0) * 1e6
            if dec is not None and hasattr(dec, "converge"):
                conv[k] = bool(dec.converge)
        load1 = os.getloadavg()
        xs = np.sort(t)
        summ = {"config": c, "session": s, "position_in_session": pos, "seed": seed, "core": core, "cpu": cpu,
                "host": socket.gethostname(), "loadavg_start": load0, "loadavg_end": load1, "n": TIMED,
                "warmup": WARM, "mean_us": float(t.mean()), "max_us": float(t.max()),
                "versions": {"python": platform.python_version(), "stim": stim.__version__,
                             "pymatching": pymatching.__version__, "ldpc": ldpc.__version__, "numpy": np.__version__}}
        for q in (0.5, 0.9, 0.95, 0.99):
            summ[f"p{int(q * 100)}_us"] = float(np.percentile(t, q * 100))
        for q in (0.5, 0.95, 0.99):
            summ[f"p{int(q * 100)}_ci95_us"] = qci(xs, q)
        if dec is not None and hasattr(dec, "converge"):
            summ["bp_not_converged"] = int((~conv).sum())
            if (~conv).any():
                top = t >= np.percentile(t, 95)
                summ["share_of_slowest_5pct_not_converged"] = float((~conv[top]).mean())
                summ["p50_us_not_converged"] = float(np.percentile(t[~conv], 50))
            summ["p50_us_converged"] = float(np.percentile(t[conv], 50)) if conv.any() else None
        np.savez_compressed(os.path.join(out, f"s{s}_{c['id']}.npz"), t_us=t.astype(np.float32), converged=conv)
        with open(os.path.join(out, f"s{s}_{c['id']}.json"), "w") as fh:
            json.dump(summ, fh, indent=1)
        print(f"session {s} {c['id']}: p50 {summ['p50_us']:.1f} p99 {summ['p99_us']:.1f} us", flush=True)


if __name__ == "__main__":
    main()
