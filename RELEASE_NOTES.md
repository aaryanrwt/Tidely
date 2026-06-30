# Tidely v1.3.0 Beta: Release Notes

Tidely v1.3.0 Beta introduces extensive reliability, performance, and format compatibility updates, validated across multiple industry datasets (UCI Machine Learning repository, Kaggle campaigns, government records, and educational data).

## 🚀 Highlights in v1.3.0
- **Native ARFF Format Support:** Introducing a zero-dependency, pure-Python Attribute-Relation File Format (ARFF) parser supporting numeric, real, and nominal types.
- **DNA Semantic Protection:** Added semantic nucleotide pattern recognition (`"DNA Sequence"`) to protect biological sequences from casing transformations or normalizations.
- **Robust Mixed-Type Load Fallback:** Prevents loading crashes on dataframes with mixed-type columns (e.g. whitespace strings inside numeric values) by automatically falling back to safe string coercion.

## 🛠️ Performance & Scalability
- Optimized memory downcasting footprints by up to 61% utilizing Polars lazy expressions and vectorized typing boundaries.
- Zero-corruption deterministic guarantees across duplicate drops, date formatting, and text character encodings.

## 🐛 Bug Fixes
- **Unicode Preservation:** Updated the text normalization filter to target only non-printable C0/C1 control codes, preserving foreign alphabets, accented letters, and emojis.
- **Target Key Safeguard:** Skip primary-key deduplication on descriptive columns (e.g. columns with name/description keywords) to prevent accidental record drops.
- **Null Replacements:** Cleaned up missing-value converters to accurately map string representations of nulls (`?`, `N/A`, `NaN`) to true missing cells.

## 📦 Installation
```bash
pip install tidely==1.3.0b2
```

## 📖 Known Limitations & Next Steps
- Nested JSON flattening is supported, but advanced auto-relational flattening of deeply nested arrays is scheduled for future milestones.
- Out-of-core streaming files are under active testing.
