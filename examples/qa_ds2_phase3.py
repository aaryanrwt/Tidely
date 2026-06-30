import json

import pandas as pd

import tidely as td


def cleaning_validation():
    df = pd.read_csv("qa_ds2_temp.csv")

    # Run Tidely Clean
    result = td.clean(df)

    report = {"summary": result.summary()}

    with open("qa_ds2_cleaning_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("Saved Tidely cleaning report to qa_ds2_cleaning_report.json")


if __name__ == "__main__":
    cleaning_validation()
