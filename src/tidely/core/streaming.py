"""Out-of-core streaming and DuckDB execution engines for large datasets."""

import os
from typing import Any, cast

import polars as pl
import pyarrow as pa

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]

from tidely.core.clean_engine import RepairPlan


class StreamingEngine:
    """Out-of-core and memory-efficient execution engine using DuckDB and Polars streaming."""

    @staticmethod
    def clean_with_duckdb(
        plan_obj: RepairPlan,
        source: Any,
        columns: list[str],
        format_name: str,
        output_filepath: str | None = None,
    ) -> Any:
        """Executes the cleaning plan using DuckDB.

        Args:
            plan_obj: The RepairPlan instance containing actions to execute.
            source: A file path string, Pandas DataFrame, Polars DataFrame, or Arrow Table.
            columns: The list of column names in the dataset.
            format_name: The format description (e.g. 'csv', 'parquet', 'pandas').
            output_filepath: Optional path to write output directly (file-to-file).

        Returns:
            The cleaned DataFrame (Polars, Pandas, or LazyFrame wrapper).
        """
        if duckdb is None:
            raise ImportError(
                "DuckDB is required for DuckDB execution mode but is not installed."
            )

        conn = duckdb.connect()

        # Handle in-memory DataFrames by registering them
        registered_name = "raw_dataset"
        if not isinstance(source, str):
            try:
                import pandas as pd

                if isinstance(source, pd.DataFrame):
                    conn.register(registered_name, source)
                elif isinstance(source, pl.DataFrame):
                    conn.register(registered_name, source.to_arrow())
                elif isinstance(source, pa.Table):
                    conn.register(registered_name, source)
                else:
                    conn.register(registered_name, source)
                source_identifier = registered_name
            except Exception as e:
                raise RuntimeError(
                    f"Failed to register in-memory dataset in DuckDB: {e}"
                ) from e
        else:
            source_identifier = source

        # Compile the plan into a single optimized CTE query
        sql_query = plan_obj.compile_to_sql(source_identifier, columns)

        # File-to-file COPY execution if output path is provided
        if output_filepath:
            ext = os.path.splitext(output_filepath)[1].lower()
            copy_format = "PARQUET" if ext == ".parquet" else "CSV"
            copy_sql = (
                f"COPY ({sql_query}) TO '{output_filepath}' (FORMAT {copy_format})"
            )
            try:
                conn.execute(copy_sql)
                # Return a Polars LazyFrame pointing to the cleaned file so it doesn't load into RAM
                if copy_format == "PARQUET":
                    lf = pl.scan_parquet(output_filepath)
                else:
                    lf = pl.scan_csv(output_filepath)

                # Rename the LazyFrame columns back to original empty strings if needed
                rename_map = {}
                for idx, c in enumerate(columns):
                    if c == "":
                        rename_map[f"_unnamed_{idx}"] = ""
                if rename_map:
                    lf = lf.rename(rename_map)
                return lf
            except Exception as e:
                raise RuntimeError(
                    f"DuckDB out-of-core COPY operation failed: {e}"
                ) from e

        # Otherwise execute and return in-memory
        try:
            res_df = conn.execute(sql_query).pl()
            # Rename back to original empty strings if needed
            rename_map = {}
            for idx, c in enumerate(columns):
                if c == "":
                    rename_map[f"_unnamed_{idx}"] = ""
            if rename_map and hasattr(res_df, "rename"):
                res_df = res_df.rename(rename_map)
            return res_df
        except Exception as e:
            raise RuntimeError(f"DuckDB SQL query execution failed: {e}") from e

    @staticmethod
    def clean_chunked_streaming(
        plan_obj: RepairPlan,
        filepath: str,
        columns: list[str],
        format_name: str,
        chunk_size: int = 50000,
    ) -> Any:
        """Cleans a dataset out-of-core by streaming chunks sequentially to minimize RAM usage.

        Args:
            plan_obj: The RepairPlan instance containing actions to execute.
            filepath: Path to the raw CSV or Parquet file.
            columns: The list of column names in the dataset.
            format_name: File format description ('csv_lazy', 'parquet_lazy', etc.).
            chunk_size: Size of each memory chunk (number of rows).

        Returns:
            A Polars LazyFrame pointing to the cleaned temporary file.
        """
        temp_out = filepath + ".cleaned.tmp"
        if os.path.exists(temp_out):
            try:
                os.remove(temp_out)
            except Exception:
                pass

        ext = os.path.splitext(filepath)[1].lower()

        if ext == ".csv":
            lf = pl.scan_csv(filepath)
            first = True
            for chunk_df in lf.collect_batches(chunk_size=chunk_size):
                # Run plan on chunk
                for action in plan_obj.actions:
                    try:
                        chunk_df = action.rule_fn(chunk_df)
                    except Exception:
                        pass

                # Append to output CSV
                with open(temp_out, "ab" if not first else "wb") as f:
                    chunk_df.write_csv(f, include_header=first)
                first = False

            # Scan cleaned CSV
            return pl.scan_csv(temp_out)

        elif ext == ".parquet":
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(filepath)  # type: ignore[no-untyped-call]

            # Create Parquet writer
            writer = None
            try:
                # Iterate row groups
                for i in range(pf.num_row_groups):
                    table = pf.read_row_group(i)  # type: ignore[no-untyped-call]
                    chunk_df = cast(pl.DataFrame, pl.from_arrow(table))

                    # Run plan on chunk
                    for action in plan_obj.actions:
                        try:
                            chunk_df = action.rule_fn(chunk_df)
                        except Exception:
                            pass

                    # Write PyArrow batch
                    out_table = chunk_df.to_arrow()
                    if writer is None:
                        writer = pq.ParquetWriter(temp_out, out_table.schema)  # type: ignore[no-untyped-call]
                    writer.write_table(out_table)  # type: ignore[no-untyped-call]
            finally:
                if writer is not None:
                    writer.close()  # type: ignore[no-untyped-call]

            # Scan cleaned Parquet
            return pl.scan_parquet(temp_out)

        else:
            # Fall back to standard lazy evaluation if unsupported file format
            pl_lazy = (
                pl.scan_parquet(filepath)
                if ext == ".parquet"
                else pl.scan_csv(filepath)
            )
            # We can't apply plan directly on LazyFrame if some rules use eager functions,
            # so we collect and clean (best effort)
            df = pl_lazy.collect()
            for action in plan_obj.actions:
                df = action.rule_fn(df)
            return df
