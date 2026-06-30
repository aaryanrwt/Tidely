"""Cleaning Executor for applying strictly vectorized AutoFixes."""

from typing import Any


class CleaningExecutor:
    """Applies AutoFix commands using vectorized Pandas/Polars operations.

    Python loops (e.g. iterrows or apply with lambda) are strictly forbidden
    in this execution layer to guarantee performance.
    """

    def __init__(self):
        pass

    def execute(self, df: Any, autofixes: list[dict[str, Any]]) -> Any:
        """Executes the given autofix commands on the dataframe.

        Args:
            df: The DataFrame.
            autofixes: The list of actions to apply.

        Returns:
            The mutated DataFrame. (Mutates in place where possible).
        """
        import pandas as pd

        is_pandas = isinstance(df, pd.DataFrame)

        for action in autofixes:
            atype = action["type"]
            col = action.get("column")

            if atype == "dedup_rows":
                if is_pandas:
                    df.drop_duplicates(inplace=True)
                else:
                    df = df.unique()

            elif atype == "normalize_email" and col:
                if is_pandas:
                    df[col] = df[col].astype(str).str.lower().str.strip()
                else:
                    import polars as pl

                    df = df.with_columns(
                        pl.col(col).cast(pl.String).str.to_lowercase().str.strip_chars()
                    )

            elif atype == "to_categorical" and col:
                if is_pandas:
                    df[col] = df[col].astype("category")
                else:
                    import polars as pl

                    df = df.with_columns(pl.col(col).cast(pl.Categorical))

            elif atype == "downcast_numeric" and col:
                if is_pandas:
                    # Use pandas pd.to_numeric for safe downcasting
                    df[col] = pd.to_numeric(df[col], downcast="integer")
                    df[col] = pd.to_numeric(df[col], downcast="float")
                else:
                    pass  # Polars often handles this cleanly or we can do pl.col(col).shrink_dtype()

            elif atype == "impute_median" and col:
                if is_pandas:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    import polars as pl

                    df = df.with_columns(pl.col(col).fill_null(strategy="median"))

            elif atype == "impute_mode" and col:
                if is_pandas:
                    mode_val = (
                        df[col].mode().iloc[0]
                        if not df[col].mode().empty
                        else "Unknown"
                    )
                    df[col] = df[col].fillna(mode_val)
                else:
                    df = df.fill_null("Unknown")

            elif atype == "normalize_date" and col:
                if is_pandas:
                    df[col] = pd.to_datetime(df[col], errors="coerce")

            elif atype == "normalize_boolean" and col:
                if is_pandas:
                    df[col] = (
                        df[col]
                        .astype(str)
                        .str.lower()
                        .map(
                            {
                                "true": True,
                                "yes": True,
                                "1": True,
                                "y": True,
                                "t": True,
                                "false": False,
                                "no": False,
                                "0": False,
                                "n": False,
                                "f": False,
                            }
                        )
                        .fillna(df[col])
                    )  # Fallback to original if not matched

            elif atype == "clean_string" and col:
                if is_pandas:
                    # Remove multiple spaces, strip, remove non-printable characters
                    df[col] = (
                        df[col]
                        .astype(str)
                        .str.replace(r"\s+", " ", regex=True)
                        .str.strip()
                    )
                    df[col] = df[col].str.replace(r"[^\x20-\x7E]", "", regex=True)

        return df
