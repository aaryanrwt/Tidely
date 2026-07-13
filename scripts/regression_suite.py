# Report 12: Regression Benchmark Suite
import subprocess
import sys

def run_regression():
    print("Running Tidely scientific validation regression check...")
    res = subprocess.run(["python", "scripts/run_scientific_validation.py"], capture_output=True, text=True)
    if res.returncode == 0:
        print(res.stdout)
        print("REGRESSION CHECK: PASSED")
        sys.exit(0)
    else:
        print("REGRESSION CHECK: FAILED")
        print(res.stdout)
        print(res.stderr)
        sys.exit(1)

if __name__ == "__main__":
    run_regression()
