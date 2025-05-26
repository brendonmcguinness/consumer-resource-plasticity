#!/usr/bin/env python3
"""Test script to sequentially run one_acc_comm and figure scripts (3b, 4, 5)
to verify that changes to utils.py and Community.py don’t break any of the
analysis/figure-generation pipelines."""

import sys
import os
import runpy
import matplotlib

# Use non-interactive backend for figure scripts
#matplotlib.use("Agg")

# Ensure the project root (where utils.py and Community.py live) is on PYTHONPATH
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# List of scripts to test
SCRIPTS = [
    "one_acc_comm.py",
    "fig3b_script.py",
    "figure4_script.py",
    "fig5_script.py",
]

def run_script(script_name):
    path = os.path.join(SCRIPT_DIR, script_name)
    print(f"Running {script_name}...")
    runpy.run_path(path, run_name="__main__")
    print(f"Completed {script_name}.n")

def main():
    for script in SCRIPTS:
        run_script(script)
    print("All scripts ran successfully without errors.")

if __name__ == "__main__":
    main()
