import json

import pandas as pd

import tidely as td


def cleaning_validation():
    with open("qa_ds1_baseline.json") as f:
        baseline = json.load(f)

    csv_path = baseline["csv_path"]
    df = pd.read_csv(csv_path)

    # Run Tidely Clean
    result = td.clean(df)

    # We want to extract the actions that were applied.
    # We can parse them from the audit log or summary.
    # Alternatively, just save the summary report string.
    report = {"summary": result.summary()}

    with open("qa_ds1_cleaning_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("Saved Tidely cleaning report to qa_ds1_cleaning_report.json")


if __name__ == "__main__":
    cleaning_validation()
