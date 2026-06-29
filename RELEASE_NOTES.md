# Tidely v1.0.0: Public Beta Release

Tidely v1.0 has completed an extensive internal validation campaign covering more than twenty real-world datasets across healthcare, finance, retail, manufacturing, government, environmental science, e-commerce, and enterprise Excel workflows.

The library has also successfully passed property-based testing, fuzz testing, large-scale stress testing up to 10 million rows, API stability checks, and cross-version compatibility testing. 

Based on these results, Tidely is now entering Public Beta, where broader community feedback will continue to strengthen its reliability.

## 🚀 Key Features in v1.0
- **Semantic Engine:** Zero-configuration detection for US Dates, Emails, URLs, Currencies, Booleans, and Identifiers.
- **Aggressive Memory Downcasting:** Safely reduces Pandas/Polars footprints by 40-85% by downcasting 64-bit numerical boundaries without mutating business logic.
- **Deep Explainability:** The `.summary()` method provides an exhaustive report outlining *what* was changed, *why*, and the *impact*.
- **Business Logic Protection:** Emits 85% Confidence Warnings instead of blindly imputing zeros on missing financial metrics.

## 📦 Installation
```bash
pip install tidely
```

## 📖 Documentation
A massive overhaul of the documentation has been deployed in this release. Please check out the new [GitHub Repository](https://github.com/aaryanrwt/tidely) for:
- Beginner-friendly tutorials in `examples/`.
- The new `docs/` site containing Performance, Validation, and Inspection guides.

Thank you to the community for supporting the push towards a standard, deterministic data cleaning API for Python.
