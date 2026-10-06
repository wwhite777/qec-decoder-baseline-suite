"""Run every config of a JSONL file through run_config.py in a process pool (one subprocess per config).
Usage: python code/v3/driver.py code/v3/configs_fixed_sample.jsonl <out_dir> <max_workers>   -> writes <out_dir>/logs/<id>.log and DRIVER_DONE"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
cfgs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
out, W = sys.argv[2], int(sys.argv[3])
os.makedirs(os.path.join(out, "logs"), exist_ok=True)
PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
def one(c):
    log = os.path.join(out, "logs", c["id"] + ".log")
    with open(log, "w") as fh:
        r = subprocess.run([PY, os.path.join(HERE, "run_config.py"), json.dumps(c), out], stdout=fh, stderr=subprocess.STDOUT)
    return c["id"], r.returncode
t0 = time.time()
with ThreadPoolExecutor(max_workers=W) as ex:
    res = list(ex.map(one, cfgs))
bad = [r for r in res if r[1] != 0]
with open(os.path.join(out, "DRIVER_DONE"), "w") as fh:
    json.dump({"configs": len(cfgs), "failed": bad, "seconds": time.time() - t0}, fh)
print(f"done {len(cfgs)} configs, {len(bad)} failed, {time.time()-t0:.0f}s")
