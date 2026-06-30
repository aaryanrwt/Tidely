import pandas as pd

import tidely as td


def stress_test():
    df = pd.read_csv(
        "C:\\Users\\Aaryan Rawat\\.cache\\kagglehub\\datasets\\mirzahasnine\\loan-data-set\\versions\\1\\loan_train.csv"
    )

    print(f"Dependents Before:\n{df['Dependents'].value_counts(dropna=False)}")

    result = td.clean(df)

    print(f"\nDependents After:\n{result.df['Dependents'].value_counts(dropna=False)}")


if __name__ == "__main__":
    stress_test()
