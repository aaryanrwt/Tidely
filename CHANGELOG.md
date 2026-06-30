# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0-beta] - 2026-06-30

### Added
- Native Attribute-Relation File Format (ARFF) parsing engine with nominal and type casting.
- `"DNA Sequence"` semantic checker matching nucleotide strings.
- Multi-sheet and robust worksheet fallbacks for educational Excel files.

### Improved
- Added robust column-level fallback for Pandas-to-Polars conversions when mixed types fail PyArrow type inference.
- Refined primary key deduplication logic to ignore descriptive string columns (e.g. columns with "name", "title", "text", "description").

### Fixed
- Unicode cleaning engine updated to preserve non-ASCII, foreign alphabets, and emojis (preserving movie reviews/text datasets).
- Fixed `replace_null_placeholders` to correctly convert custom string null representations (`?`, `N/A`, `NaN`) to true nulls.
- Moved row deduplication step to the end of the pipeline to prevent post-cleaning duplicate rows.

### Performance
- Fully vectorized parsing and cleaning pipelines with low peak RAM footprints and high throughput (~100 ms average latency).

### Validation
- 100% pass rate on UCI (Wine, Adult, Splice), Kaggle (IMDb, VGG16, Pokemon), and local mixed datasets validation campaigns.

## [1.0.0] - Public Beta Release

### Added
- Complete rewrite of the semantic detection engine to support cross-domain pattern matching.
- Memory optimizer module for massive-scale Pandas/Polars downcasting.
- Zero-corruption deterministic guarantees across duplicate drops and imputation.
- Expanded Datetime parser supporting US formats (`MM/DD/YYYY H:MM`).
- Validation Suite V: Full ERP Excel compatibility (`.xlsx`).
- Strict Hypothesis property-fuzzing and regression framework.
- Business Logic protection warning system for nulls in critical identifiers.
- Official documentation site and tutorials.
