"""Tests for the Dataset DNA classification engine."""

from tidely.core.dna import infer_dataset_dna


def test_dna_ecommerce() -> None:
    """Infers E-commerce DNA correctly."""
    cols = ["customer_id", "order_amount", "discount_percentage", "sku", "product_name"]
    dna = infer_dataset_dna(cols)
    assert dna.domain == "E-commerce"
    assert dna.confidence >= 0.5
    assert "Orders" in dna.entities
    assert "Churn Prediction" in dna.ml_tasks


def test_dna_healthcare() -> None:
    """Infers Healthcare DNA correctly."""
    cols = ["patient_id", "doctor_name", "systolic_bp", "diastolic_bp", "pulse"]
    dna = infer_dataset_dna(cols)
    assert dna.domain == "Healthcare"
    assert "Patients" in dna.entities
    assert "Disease Classification" in dna.ml_tasks


def test_dna_finance() -> None:
    """Infers Finance DNA correctly."""
    cols = ["account_number", "balance_usd", "credit_amount", "ticker_symbol"]
    dna = infer_dataset_dna(cols)
    assert dna.domain == "Finance"
    assert "Accounts" in dna.entities


def test_dna_hr() -> None:
    """Infers HR DNA correctly."""
    cols = ["employee_id", "salary_grade", "hire_date", "department_name"]
    dna = infer_dataset_dna(cols)
    assert dna.domain == "HR"
    assert "Employees" in dna.entities


def test_dna_generic() -> None:
    """Fallback to generic DNA when columns are unrelated."""
    cols = ["col_a", "col_b", "col_c"]
    dna = infer_dataset_dna(cols)
    assert dna.domain == "Generic Tabular"
    assert "Classification" in dna.ml_tasks
