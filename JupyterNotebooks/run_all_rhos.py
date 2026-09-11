"""
Launch EX1's experiment() for multiple rho values in parallel, each as its
own OS process, instead of re-running 00_EX1.ipynb four times sequentially.

Each rho value writes to its own output CSV (see experiment.py), so the runs
are independent and safe to run concurrently. BLAS thread pools are pinned to
1 per process to avoid CPU oversubscription across the concurrent processes.

While running, a self-updating status block shows progress both by log count
and by event count (log count alone is misleading, since a handful of huge
logs can dominate total runtime).

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
import threading
from pathlib import Path

import pandas as pd

DEFAULT_RHOS = [0.6, 0.7, 0.8, 0.9]
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "../logs/smartRPA/202511-results"
VALIDATION_DATA_PATH = SCRIPT_DIR / "../logs/smartRPA/202511-update/validationLogInformation.csv"
STATUS_POLL_SECONDS = 5

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


def read_validation_totals() -> tuple[int, int]:
    validation_data = pd.read_csv(VALIDATION_DATA_PATH)
    return len(validation_data), int(validation_data["logLength"].sum())


def read_progress(output_csv_path: Path) -> tuple[int, int]:
    """Returns (logs processed, events processed) for one rho's output CSV.

    Read failures are treated as "no progress yet" rather than raised, since
    the writing process rewrites the whole file on every save and a read can
    land mid-write.
    """
    try:
        df = pd.read_csv(output_csv_path)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return 0, 0
    processed_logs = len(df)
    processed_events = int(df["logLength"].fillna(0).sum()) if "logLength" in df.columns else 0
    return processed_logs, processed_events


def format_status_lines(rho_outputs, total_logs: int, total_events: int) -> list[str]:
    lines = []
    agg_logs = 0
    agg_events = 0
    for rho, output_csv_path in rho_outputs:
        processed_logs, processed_events = read_progress(output_csv_path)
        agg_logs += processed_logs
        agg_events += processed_events
        log_pct = (processed_logs / total_logs * 100) if total_logs else 0.0
        event_pct = (processed_events / total_events * 100) if total_events else 0.0
        lines.append(
            f"  rho={rho:<4} logs {processed_logs:>4}/{total_logs:<4} ({log_pct:5.1f}%)   "
            f"events {processed_events:>12,}/{total_events:<12,} ({event_pct:5.1f}%)"
        )
    total_logs_all = total_logs * len(rho_outputs)
    total_events_all = total_events * len(rho_outputs)
    agg_log_pct = (agg_logs / total_logs_all * 100) if total_logs_all else 0.0
    agg_event_pct = (agg_events / total_events_all * 100) if total_events_all else 0.0
    lines.append(
        f"  {'TOTAL':<8} logs {agg_logs:>4}/{total_logs_all:<4} ({agg_log_pct:5.1f}%)   "
        f"events {agg_events:>12,}/{total_events_all:<12,} ({agg_event_pct:5.1f}%)"
    )
    return lines


def run_status_monitor(rho_outputs, total_logs: int, total_events: int, stop_event: threading.Event):
    """Redraws an in-place status block every STATUS_POLL_SECONDS until stop_event is set."""
    if not sys.stdout.isatty():
        return  # avoid emitting cursor-control garbage into redirected/piped output
    n_lines = len(rho_outputs) + 1

    def draw():
        for line in format_status_lines(rho_outputs, total_logs, total_events):
            sys.stdout.write("\033[K" + line + "\n")
        sys.stdout.flush()

    draw()
    while not stop_event.wait(STATUS_POLL_SECONDS):
        sys.stdout.write(f"\033[{n_lines}A")
        draw()
    sys.stdout.write(f"\033[{n_lines}A")
    draw()


def main():
    args = parse_args()

    child_env = os.environ.copy()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        child_env[var] = "1"

    total_logs, total_events = read_validation_totals()

    processes = []
    rho_outputs = []
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
        rho_outputs.append((rho, RESULTS_DIR / result_file_name))

    print(f"\nLaunched {len(processes)} process(es). Waiting for completion...")
    print(f"Tracking progress against {total_logs} logs / {total_events:,} events per rho run.")
    print("Tail an individual run with: tail -f <log file>\n")

    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=run_status_monitor, args=(rho_outputs, total_logs, total_events, stop_event), daemon=True,
    )
    monitor_thread.start()

    results = []
    for rho, proc, log_path, log_file in processes:
        exit_code = proc.wait()
        log_file.close()
        results.append((rho, exit_code, log_path))

    stop_event.set()
    monitor_thread.join(timeout=STATUS_POLL_SECONDS + 2)

    print("\n--- Summary ---")
    for rho, exit_code, log_path in results:
        status = "OK" if exit_code == 0 else f"FAILED (exit {exit_code})"
        print(f"rho={rho}: {status}  [{log_path}]")

    if any(exit_code != 0 for _, exit_code, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
