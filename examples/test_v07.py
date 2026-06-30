import os
import sys

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import pandas as pd

import tidely as td


def main():
    # Load Titanic dataset
    csv_path = os.path.join(os.path.dirname(__file__), "../titanic.csv")
    df = pd.read_csv(csv_path)

    # Inject bad data to test new semantic normalizers
    df["contact_email"] = "JOHN.DOE@GMAIL.COM  "
    df["signup_date"] = "2023/12/31"
    df["is_active"] = "Yes"

    # The Core API workflow
    print("--- 1. td.clean() ---")
    result = td.clean(df)

    print("\n\n--- 2. Explainability Summary ---")
    print(result.summary())

    print("\n\n--- 3. td.validate() ---")
    schema = {
        "PassengerId": "int",
        "Survived": "int",
        "contact_email": "str",
        "is_active": "bool",
        "signup_date": "datetime",
    }

    is_valid = td.validate(result.df, schema)
    print(f"Dataset passes strict schema validation? {is_valid}")

    print("\n\n--- 4. result.export() ---")
    out_csv = "cleaned_titanic.csv"
    out_html = "cleaned_report.html"
    result.export(out_csv)
    result.export(out_html)
    print(f"Exported to {out_csv} and {out_html}")

    # Cleanup exports
    os.remove(out_csv)
    os.remove(out_html)


if __name__ == "__main__":
    main()
