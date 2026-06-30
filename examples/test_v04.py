import os
import sys

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Add src to path so we can import tidely without installing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import pandas as pd

import tidely as td


def main():
    # Load Titanic dataset as a test
    csv_path = os.path.join(os.path.dirname(__file__), "../titanic.csv")
    df = pd.read_csv(csv_path)

    print(
        "Original memory usage:", df.memory_usage(deep=True).sum() / (1024 * 1024), "MB"
    )

    # Intentionally add a duplicate row and a bad email column to test the intelligence
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    df["contact_email"] = "JOHN.DOE@GMAIL.COM  "

    # Run the magic command
    result = td.clean(df)

    # Output the summary
    print("\n\n--- Testing Result ---")
    print(result.summary())

    # Output the dataframe
    print("\n\n--- Returned DataFrame ---")
    print(result.df.head())


if __name__ == "__main__":
    main()
