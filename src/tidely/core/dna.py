"""Dataset DNA Profiler to classify dataset domain and infer likely ML tasks."""


class DatasetDNA:
    """Represents the dataset's domain classification, entities, and likely ML tasks."""

    def __init__(
        self,
        domain: str,
        confidence: float,
        entities: list[str],
        ml_tasks: list[str],
    ) -> None:
        """Initialize DatasetDNA.

        Args:
            domain: Inferred domain name (e.g., E-commerce, HR, Finance).
            confidence: Domain classification confidence score (0.0 to 1.0).
            entities: Business entities found in the columns.
            ml_tasks: Recommended ML tasks that can be trained on this dataset.
        """
        self.domain = domain
        self.confidence = confidence
        self.entities = entities
        self.ml_tasks = ml_tasks


# Domain keywords mapper
DOMAIN_KEYWORDS = {
    "E-commerce": {
        "words": [
            "order",
            "price",
            "amount",
            "revenue",
            "quantity",
            "discount",
            "sku",
            "product",
            "cart",
            "transaction",
            "customer_id",
            "sale",
            "checkout",
        ],
        "entities": ["Customers", "Orders", "Revenue", "Products", "Transactions"],
        "ml_tasks": [
            "Churn Prediction",
            "Sales Forecasting",
            "Customer Lifetime Value (LTV)",
            "Recommendation Engine",
        ],
    },
    "Healthcare": {
        "words": [
            "patient",
            "doctor",
            "visit",
            "diagnosis",
            "blood_pressure",
            "bp",
            "systolic",
            "diastolic",
            "heart_rate",
            "temperature",
            "symptom",
            "medication",
            "prescription",
            "pulse",
            "vital",
            "clinic",
        ],
        "entities": ["Patients", "Doctors", "Appointments", "Vitals", "Diagnoses"],
        "ml_tasks": [
            "Patient Readmission Prediction",
            "Disease Classification",
            "Length of Stay (LOS) Forecasting",
        ],
    },
    "Finance": {
        "words": [
            "transaction_id",
            "bank",
            "account",
            "balance",
            "credit",
            "debit",
            "iban",
            "swift",
            "portfolio",
            "ticker",
            "stock",
            "trade",
            "shares",
            "revenue",
            "profit",
            "loss",
            "card_number",
        ],
        "entities": ["Accounts", "Transactions", "Credit Cards", "Ledger Logs"],
        "ml_tasks": [
            "Fraud Detection",
            "Credit Risk Scoring",
            "Stock Price Forecasting",
            "Anomaly Detection",
        ],
    },
    "HR": {
        "words": [
            "employee",
            "salary",
            "hire_date",
            "department",
            "manager",
            "performance",
            "attendance",
            "leave",
            "termination",
            "job_title",
            "onboarding",
            "recruitment",
        ],
        "entities": ["Employees", "Salaries", "Departments", "Job Roles"],
        "ml_tasks": [
            "Employee Attrition (Churn)",
            "Talent Performance Forecasting",
            "Salary Range Recommendation",
        ],
    },
}


def infer_dataset_dna(columns: list[str]) -> DatasetDNA:
    """Analyzes column names to classify the dataset domain, entities, and ML tasks.

    Args:
        columns: List of column names in the dataset.

    Returns:
        DatasetDNA: Inferred Dataset DNA profile.
    """
    scores: dict[str, int] = dict.fromkeys(DOMAIN_KEYWORDS, 0)
    matched_words: dict[str, list[str]] = {domain: [] for domain in DOMAIN_KEYWORDS}

    # Normalize columns
    cols_lower = [c.lower() for c in columns]

    for col in cols_lower:
        for domain, data in DOMAIN_KEYWORDS.items():
            for word in data["words"]:
                # Matches either exact word or sub-segment (e.g. 'total_revenue' matches 'revenue')
                if word in col:
                    scores[domain] += 1
                    matched_words[domain].append(col)

    # Find highest matching domain
    best_domain = max(scores, key=scores.get)  # type: ignore
    best_score = scores[best_domain]

    if best_score == 0:
        # Fallback to Generic
        return DatasetDNA(
            domain="Generic Tabular",
            confidence=0.5,
            entities=["Records"],
            ml_tasks=[
                "Classification",
                "Regression",
                "Clustering",
                "Anomaly Detection",
            ],
        )

    # Compute a simple confidence ratio based on number of columns matched
    total_cols = len(columns)
    # The more matched columns, the higher the confidence, capping at 0.95
    base_confidence = min(0.5 + (best_score / total_cols) * 0.5, 0.95)

    # Get domain details
    domain_info = DOMAIN_KEYWORDS[best_domain]

    return DatasetDNA(
        domain=best_domain,
        confidence=base_confidence,
        entities=domain_info["entities"],
        ml_tasks=domain_info["ml_tasks"],
    )
