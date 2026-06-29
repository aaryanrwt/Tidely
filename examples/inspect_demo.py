"""Basic example demonstrating Tidely dataset inspection."""

import polars as pl

import tidely as td


def main() -> None:
    # 1. Create a slightly messy dataset
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
            "signup_date": [
                "2023-01-01",
                "01/02/2023",
                "2023-01-03",
                None,
                "2023-01-05",
            ],
            "age": [25, 30, 30, None, 40],
        }
    )

    print("Running Tidely Inspection...")

    # 2. Inspect the dataset
    profile = td.inspect(df)

    # 3. Render the visual report
    profile.show()


if __name__ == "__main__":
    main()
