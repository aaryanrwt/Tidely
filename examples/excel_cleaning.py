"""Tidely Example 2: Excel Cleaning
This script demonstrates reading from an Excel file (requires openpyxl).
"""

import pandas as pd

import tidely as td


def main():
    # Load an Excel file using Pandas
    # Ensure you have installed openpyxl: pip install openpyxl
    try:
        df = pd.read_excel("financial_report.xlsx", sheet_name="Actuals")
    except FileNotFoundError:
        print("Please create 'financial_report.xlsx' to run this example.")
        return

    print("Cleaning Enterprise Excel Export...")

    # Tidely handles Excel DataFrames seamlessly
    result = td.clean(df)

    # Missing financial data (e.g. 'Tax' or 'Trade Spend') is safely
    # ignored to prevent financial corruption.
    print(result.summary())

    # Export back to Excel
    result.df.to_excel("financial_report_cleaned.xlsx", index=False)
    print("Saved clean data to financial_report_cleaned.xlsx")


if __name__ == "__main__":
    main()
