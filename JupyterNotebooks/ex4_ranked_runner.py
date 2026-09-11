"""
EX4 runner: parameter-free ranked discovery pipeline across the SmartRPA 2025
validation set.

All design decisions and their paper-facing justification live in
``EX4_design_decisions.md`` next to this file. Keep that file in sync.

Outputs (under ``../logs/smartRPA/202511-results/``):
    ex4_ranked_results.csv          — one row per log (summary + AUPRC)
    ex4_ranked_results_motifs.csv   — one row per discovered motif
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Ensure relative validation-data path in experiment.py resolves correctly.
os.chdir(_THIS_DIR)

from experiment import experiment_ranked, T_MAX_SKIP_DEFAULT


if __name__ == "__main__":
    experiment_ranked(
        target_filename="ex4_ranked_results.csv",
        t_max_skip=T_MAX_SKIP_DEFAULT,
        printing=True,
    )
