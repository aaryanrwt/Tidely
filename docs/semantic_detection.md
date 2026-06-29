# Semantic Detection

Tidely's core intelligence comes from its **Semantic Engine**. Rather than relying on simple pandas `dtypes` (which classify everything as `object`), Tidely attempts to infer the real-world business meaning of your data.

## Supported Semantic Types

The engine currently detects and classifies the following types:
- **Date**: Standard `YYYY-MM-DD` as well as US `MM/DD/YYYY H:MM:SS` and ISO 8601 timestamps.
- **Email**: Standard email addresses (`user@domain.com`).
- **URL**: Web URLs (`https://...`).
- **Currency**: Monetary values prefixed with symbols (`$`, `€`, `£`, `¥`).
- **Boolean**: Yes/No, True/False, T/F, Y/N, 1/0.
- **SSN**: US Social Security Numbers.
- **IPv4**: Standard IP addresses.
- **Categorical**: Low cardinality text columns.
- **Numeric**: Floats and Integers.
- **ID/Key**: High cardinality unique text or integer keys (e.g. `InvoiceNo`, `CustomerID`).

## How Confidence Works (Match Rate)

Tidely uses a rigid confidence threshold (default 95%). 
If a column contains 1,000 rows, and 960 rows natively match the `Date` regular expression, Tidely assigns the semantic type `Date` with a `match_rate` of 0.96.

If the match rate falls below the internal safety threshold, Tidely falls back to a safer, generic type (like `Categorical` or `Text`) to prevent accidental destruction of unstructured data.
