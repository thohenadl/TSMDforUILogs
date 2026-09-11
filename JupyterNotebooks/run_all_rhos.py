"""
Launch EX1's experiment() for multiple rho values in parallel, each as its
own OS process, instead of re-running 00_EX1.ipynb four times sequentially.

Each rho value writes to its own output CSV (see experiment.py), so the runs
are independent and safe to run concurrently. BLAS thread pools are pinned to
1 per process to avoid CPU oversubscription across the concurrent processes.

Usage:
    cd JupyterNotebooks
    python run_all_rhos.py
    python run_all_rhos.py --rho 0.7 --rho 0.9
    python run_all_rhos.py --log-limit 2000
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_RHOS = [0.6, 0.7, 0.8, 0.9]
SCRIPT_DIR = Path(__file__).resolve().parent

CHILD_SCRIPT_TEMPLATE = """
from experiment import experiment
experiment(
    {result_file_name!r},
    {rho!r},
    log_limit={log_limit},
    pre_filtering={pre_filtering},
    safety_margin_factor={safety_margin_factor},
    encoding_method={encoding_method},
    core_threshold={core_threshold},
    l_max_default={l_max_default},
)
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rho", type=float, action="append", dest="rhos",
                         help="rho value to run (repeatable). Defaults to 0.6, 0.7, 0.8, 0.9.")
    parser.add_argument("--log-limit", type=int, default=2000000)
    parser.add_argument("--pre-filtering", type=lambda s: s.lower() == "true", default=False)
    parser.add_argument("--safety-margin-factor", type=int, default=2)
    parser.add_argument("--encoding-method", type=int, default=0)
    parser.add_argument("--core-threshold", type=float, default=0.8)
    parser.add_argument("--l-max-default", type=int, default=75)
    args = parser.parse_args()
    if not args.rhos:
        args.rhos = DEFAULT_RHOS
    return args


def result_file_name_for(rho: float) -> str:
    rho_value = str(rho).split(".")[1]
    return f"validation_experiment_results_word2vec_safety2_olap08_rho0{rho_value}_enc0_core08.csv"


def main():
    args = parse_args()

    child_env = os.environ.copy()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        child_env[var] = "1"

    processes = []
    for rho in args.rhos:
        result_file_name = result_file_name_for(rho)
        child_script = CHILD_SCRIPT_TEMPLATE.format(
            result_file_name=result_file_name,
            rho=rho,
            log_limit=args.log_limit,
            pre_filtering=args.pre_filtering,
            safety_margin_factor=args.safety_margin_factor,
            encoding_method=args.encoding_method,
            core_threshold=args.core_threshold,
            l_max_default=args.l_max_default,
        )
        log_path = SCRIPT_DIR / f"run_rho_{str(rho).split('.')[1]}.log"
        log_file = open(log_path, "w")
        print(f"Starting rho={rho} -> {result_file_name} (log: {log_path.name})")
        proc = subprocess.Popen(
            [sys.executable, "-c", child_script],
            cwd=SCRIPT_DIR,
            env=child_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        processes.append((rho, proc, log_path, log_file))

    print(f"\nLaunched {len(processes)} process(es). Waiting for completion...")
    print("Tail an individual run with: tail -f <log file>\n")

    results = []
    for rho, proc, log_path, log_file in processes:
        exit_code = proc.wait()
        log_file.close()
        results.append((rho, exit_code, log_path))

    print("\n--- Summary ---")
    for rho, exit_code, log_path in results:
        status = "OK" if exit_code == 0 else f"FAILED (exit {exit_code})"
        print(f"rho={rho}: {status}  [{log_path}]")

    if any(exit_code != 0 for _, exit_code, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
