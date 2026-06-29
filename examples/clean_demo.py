"""Basic example demonstrating Tidely Explainable Cleaning."""

import polars as pl

import tidely as td


def main() -> None:
    # 1. Create a messy dataset
    df = pl.DataFrame(
        {
            "id": [1, 2, 2, 4, 5],
            "email": [
                "alice@example.com",
                " BOB@EXAMPLE.COM ",
                "invalid-email",
                "david@test.org",
                None,
            ],
            "phone": [
                "(555) 123-4567",
                "555-987-6543",
                "not a phone",
                "+1234567890",
                None,
            ],
            "age": [25, 30, 30, None, 40],
            "category": ["A", "B", "A", "B", "A"],  # Low cardinality string
        }
    )

    print("--- Original DataFrame ---")
    print(df)
    print("\n")

    print("Generating Clean Plan...")

    # 2. Generate the plan
    plan = td.plan(df)

    # 3. Show the explainable plan
    plan.show()

    # 4. Execute the cleaning!
    clean_df = plan.execute(dry_run=False)

    print("\n--- Cleaned DataFrame ---")
    print(clean_df)

    print("\n--- Audit Log ---")
    for log in plan.audit_log:
        print(
            f"[{log['status']}] {log['category']} -> {log['action']} (Affected: {log['rows_affected']} rows)"
        )


if __name__ == "__main__":
    main()
