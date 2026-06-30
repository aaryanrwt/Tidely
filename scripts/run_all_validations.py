import os
import subprocess
import sys


def run_cmd(cmd):
    print("\n======================================")
    print(f"Running: {cmd}")
    print("======================================")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED: {cmd}")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)
    else:
        print(result.stdout)
        print(f"SUCCESS: {cmd}")


if __name__ == "__main__":
    print("Starting Global Phase 10 Regression Suite...")

    # 1. Run Unit Tests (Fuzzing, Hypothesis, API Contracts)
    run_cmd("python -m pytest tests/")

    # 2. Run QA Suite 1, 2, 3, 4, 5 scripts
    scripts = [
        "examples/massive_scale.py",
        "examples/qa3_ds1_ds2.py",
        "examples/qa3_ds3_ds4.py",
        "examples/qa3_ds5_ds6.py",
        "examples/qa4_ecommerce.py",
        "examples/qa5_retail.py",
    ]

    for script in scripts:
        if os.path.exists(script):
            run_cmd(f"python {script}")
        else:
            print(f"Warning: Script {script} not found, skipping.")

    print("\n[SUCCESS] GLOBAL REGRESSION SUITE PASSED SUCCESSFULLY.")
