"""
prepare.py — FROZEN experimental contract for Arm A (surface code, circuit-level noise).

Stim circuit-level-noise data generation + generic decode-based LER / latency
evaluation. Color / BB-qLDPC families are out of scope here (the BB arms live in
prepare_qldpc.py and prepare_qldpc_phenom.py). This file is FROZEN: it defines the
pinned circuit, seed and shot count, and must not be edited when adding a decoder.

Open-source: stim (Gidney), pymatching (MWPM), ldpc/bposd (BP-OSD), sinter.
"""
from __future__ import annotations
import os, time, hashlib
import numpy as np

# --- FROZEN pinned config --------------------------------------------------
CODE_FAMILIES = ["surface", "color", "bb_qldpc"]
PINNED_DISTANCES = {"surface": [5, 7, 9, 11], "color": [5, 7, 9], "bb_qldpc": [6, 10]}
PHYSICAL_ERROR_RATE = 1e-3          # p, pinned
ROUNDS = 25                         # syndrome-extraction rounds per shot, pinned
TIME_BUDGET_S = 300                 # fixed training wall-clock per trial (neural decoder)
LATENCY_TARGET_US = 1.0             # student must reach < 1 us/round (AlphaQubit-2 ref)
PRIMARY_METRIC = "logical_error_rate"
PRIMARY_DIRECTION = "lower"
STIM_SEED = 20260708                # pinned so data is a frozen artifact
DEFAULT_SHOTS = 100_000             # val shots (transparency: report failure count too)

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "ququ_p1_qec")
os.makedirs(CACHE_DIR, exist_ok=True)


def make_circuit(code_family: str, distance: int):
    """Circuit-level-noise memory experiment at the pinned p / rounds."""
    import stim
    if code_family != "surface":
        raise NotImplementedError(f"{code_family}: add color/BB circuit + BP-OSD (TODO).")
    p = PHYSICAL_ERROR_RATE
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=distance, rounds=ROUNDS,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p)


def generate_stim_dataset(code_family: str, distance: int, shots: int = DEFAULT_SHOTS):
    """FROZEN Stim syndrome dataset (cached npz keyed by config+seed). Returns dict with
    detectors (bool [shots, n_det]), observables (bool [shots, n_obs]), and the DEM."""
    key = f"{code_family}_d{distance}_p{PHYSICAL_ERROR_RATE}_r{ROUNDS}_s{shots}_seed{STIM_SEED}"
    tag = hashlib.md5(key.encode()).hexdigest()[:10]
    path = os.path.join(CACHE_DIR, f"{key}_{tag}.npz")
    circuit = make_circuit(code_family, distance)
    dem = circuit.detector_error_model(decompose_errors=True)
    if os.path.exists(path):
        z = np.load(path)
        return {"detectors": z["det"], "observables": z["obs"], "dem": dem, "circuit": circuit}
    sampler = circuit.compile_detector_sampler(seed=STIM_SEED)
    det, obs = sampler.sample(shots, separate_observables=True)
    np.savez_compressed(path, det=det, obs=obs)
    return {"detectors": det, "observables": obs, "dem": dem, "circuit": circuit}


def evaluate_ler(decode_fn, code_family: str, distance: int, shots: int = DEFAULT_SHOTS) -> dict:
    """GROUND-TRUTH LER on the frozen val set. `decode_fn(detectors)->predicted observables`.
    Returns per-shot LER, per-round LER, and the raw failure count (for statistics)."""
    ds = generate_stim_dataset(code_family, distance, shots)
    det, obs = ds["detectors"], ds["observables"]
    pred = np.asarray(decode_fn(det)).reshape(obs.shape[0], -1)[:, 0].astype(bool)
    truth = obs[:, 0].astype(bool)
    fails = int(np.sum(pred != truth))
    ler = fails / len(truth)
    ler_per_round = 1.0 - (1.0 - ler) ** (1.0 / ROUNDS) if 0 < ler < 1 else ler / ROUNDS
    return {"logical_error_rate": ler, "ler_per_round": ler_per_round,
            "failures": fails, "shots": len(truth)}


def measure_latency_us(decode_fn, code_family: str, distance: int, shots: int = 20_000) -> float:
    """Real steady-state decode latency (wall-clock), reported as us PER ROUND."""
    ds = generate_stim_dataset(code_family, distance, shots)
    det = ds["detectors"]
    decode_fn(det[:512])                       # warm up
    t0 = time.time()
    decode_fn(det)
    dt = time.time() - t0
    return (dt / len(det)) * 1e6 / ROUNDS      # us per round
