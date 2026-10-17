#!/usr/bin/env python
"""ONE command for the whole experiment (LinuxONE or anywhere).

    python run_all.py --quick      about 2-3 minutes: 2 scenarios, 5 seeds, a few hundred days
    python run_all.py              full: 3 scenarios x 50 seeds, 3,000 simulated days, both models

Steps: environment -> tests -> scenarios + plan search -> early-warning data (parallel timing 1/2/4/8 workers) ->
calibration + patient-flow model (skipped with a message if the cleaned NHAMCS file is missing) -> early-warning model.
Writes results/run.log and results/linuxone_run.json (on an s390x machine) or results/local_run.json (anywhere else:
nothing is labelled LinuxONE unless it really ran there). Safe to rerun. Never deletes files.
A failed step is recorded and the run continues. Use --all-tests to also run the data-pipeline tests (they need downloaded data).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PY = sys.executable
NHAMCS = ROOT / "data" / "processed" / "nhamcs_2022_cleaned.csv"
CORE_TESTS = ["tests/test_simulator.py", "tests/test_scenarios.py", "tests/test_optimizer_v1.py",
              "tests/test_simulator_additions.py", "tests/test_patient_flow.py", "tests/test_early_warning.py",
              "tests/test_dashboard_v2.py"]


class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")      # append: reruns never delete old logs

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


def sh(cmd: list, log: Logger, timeout: int = 3600) -> tuple:
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        out, code = (p.stdout or "") + (p.stderr or ""), p.returncode
    except subprocess.TimeoutExpired:
        out, code = "TIMEOUT", 124
    tail = "\n".join(out.strip().splitlines()[-12:])
    log(tail)
    return code, round(time.time() - t0, 1), out


def env_info() -> dict:
    def run(c):
        try:
            return subprocess.run(c, capture_output=True, text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    info = {"uname_m": platform.machine(), "uname_a_system": platform.platform(), "nproc": os.cpu_count(), "python": platform.python_version(),
            "byteorder": sys.byteorder, "lscpu_summary": None, "packages": {}}
    lscpu = run(["lscpu"])
    if lscpu:
        keep = ("Architecture", "Byte Order", "CPU(s)", "Model name", "Thread(s) per core", "Core(s) per socket", "Socket(s)", "Hypervisor vendor")
        info["lscpu_summary"] = {l.split(":", 1)[0].strip(): l.split(":", 1)[1].strip() for l in lscpu.splitlines() if ":" in l and l.split(":", 1)[0].strip() in keep}
    for m in ("numpy", "pandas", "scipy", "sklearn", "xgboost", "matplotlib", "streamlit", "pytest"):
        try:
            info["packages"][m] = __import__(m).__version__
        except Exception:
            info["packages"][m] = None
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--all-tests", action="store_true")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    a = ap.parse_args(argv)
    RESULTS.mkdir(exist_ok=True)
    log = Logger(RESULTS / "run.log")
    t_all = time.time()
    log(f"===== run_all {'QUICK' if a.quick else 'FULL'}  {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
    env = env_info()
    log(f"machine {env['uname_m']} | cores {env['nproc']} | python {env['python']} | byteorder {env['byteorder']}")
    steps, status = {}, {}

    def step(name, cmd, **kw):
        log(f"\n--- {name}: {' '.join(str(c) for c in cmd[1:])}")
        code, secs, out = sh(cmd, log, **kw)
        steps[name], status[name] = secs, ("ok" if code == 0 else f"exit {code}")
        log(f"[{name}] {status[name]} in {secs}s")
        return code, out

    tests = ["-m", "pytest", "-q", "-p", "no:cacheprovider", "--no-header"] + (["tests"] if a.all_tests else [t for t in CORE_TESTS if (ROOT / t).exists()])
    code, out = step("tests", [PY] + tests)
    summary = [l for l in out.splitlines() if " passed" in l or " failed" in l or " error" in l]
    env["tests_summary"] = summary[-1].strip() if summary else None

    step("scenarios_and_plan_search", [PY, "scripts/run_scenarios.py"] + (["--quick"] if a.quick else ["--seeds", "50"]))
    days = 60 if a.quick else 750
    step("early_warning_data", [PY, "scripts/generate_days.py", "--days", str(days), "--workers", str(a.workers)])

    # serial vs parallel timing for the data generation (honest: may not help on small jobs)
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT))
    t0 = time.time()
    try:
        import generate_days
        timing = generate_days.benchmark(days=15 if a.quick else 200, worker_list=(1, 2, 4, 8))
    except Exception as err:     # timing must never stop the run
        timing, status["parallel_timing"] = [], f"failed: {err}"
    steps["parallel_timing"] = round(time.time() - t0, 1)
    for r in timing:
        log(f"  workers {r['workers']}: {r['seconds']}s for {r['days']} days (speed-up {r['speedup_vs_1_worker']}x)")

    if NHAMCS.exists():
        step("calibration", [PY, "scripts/calibrate.py"] + (["--quick"] if a.quick else []))
        step("patient_flow_model", [PY, "scripts/train_patient_flow.py"] + (["--quick"] if a.quick else []))
    else:
        log(f"\nSKIPPED patient_flow_model and calibration: {NHAMCS} not found (the cleaned data is not in git")
        status["patient_flow_model"] = status["calibration"] = "skipped: data file missing"
    step("early_warning_model", [PY, "scripts/train_early_warning.py"] + (["--quick"] if a.quick else []))
    rec = {**env, "mode": "quick" if a.quick else "full", "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds_per_step": steps,
           "step_status": status, "parallel_timing": timing, "total_seconds": round(time.time() - t_all, 1),
           "workers_used": a.workers,
           "note": ("Timing is wall-clock on this machine. More workers are not always faster for small jobs; compare the table, do not assume.")}
    on_linuxone = platform.machine() == "s390x"
    out_name = "linuxone_run.json" if on_linuxone else "local_run.json"
    (RESULTS / out_name).write_text(json.dumps(rec, indent=2), encoding="utf-8")
    log(f"\nWrote results/{out_name}" + ("" if on_linuxone else "  (this machine is not s390x, so it is NOT written as linuxone_run.json)"))
    plots = ROOT / "scripts" / "make_plots.py"
    if plots.exists():                      # after the run record exists, so the timing chart can use it
        step("demo_results", [PY, "scripts/make_demo_results.py"])
        step("plots", [PY, "scripts/make_plots.py"])
        rec["seconds_per_step"], rec["step_status"] = steps, status
        rec["total_seconds"] = round(time.time() - t_all, 1)
        (RESULTS / out_name).write_text(json.dumps(rec, indent=2), encoding="utf-8")
    log(f"Total {rec['total_seconds']}s. Step status: {status}")
    return 0 if all(v == "ok" or v.startswith("skipped") for v in status.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
