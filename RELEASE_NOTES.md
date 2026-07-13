# Tidely v1.4.3 Release Notes

Tidely v1.4.3 is a validation and scientific verification release that demonstrates that Tidely produces results equivalent to—or statistically better than—a carefully engineered traditional data cleaning reference pipeline.

## ✅ What's New in v1.4.3

- **Scientific Validation Engine**: Automatically run comparative tests between raw datasets, a traditional Pandas/NumPy cleaning pipeline, and the Tidely engine.
- **Statistical Quality Safeguards**: Integrated Population Stability Index (PSI), Kolmogorov-Smirnov (KS) test, and Jensen-Shannon (JS) Distance calculations to certify that no unintentional data drift occurred.
- **Downstream ML Readiness Validation**: End-to-end evaluation pipeline that trains ML classifiers (Logistic Regression, Random Forest, Gradient Boosting) on raw vs traditional vs Tidely-cleaned datasets, asserting zero predictive performance degradation.
- **Fine-Grained Auditing**: Exposes exact cell-level changes (including raw, traditional, and Tidely values) alongside explanations of chosen statistical heuristics.

## 🔧 What's Fixed & Improved in v1.4.3

- **Robust Casting Semantics**: Updated currency/salary normalization checks to handle non-numeric values gracefully under Pandas, matching the resilience of Tidely's internal SQL `TRY_CAST` semantics.

## 📦 Installation
```bash
pip install tidely==1.4.3
```

## 📖 Verification Status
All 62 unit tests and the newly introduced scientific validation regression suite pass successfully with zero failures.
