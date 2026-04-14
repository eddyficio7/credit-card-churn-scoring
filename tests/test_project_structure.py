"""Tests for the initial project structure."""

from pathlib import Path

from churn_scoring.config import get_settings


def test_project_directories_exist() -> None:
    """Check that the main project directories exist."""
    expected_directories = [
        Path("src/churn_scoring"),
        Path("tests"),
        Path("notebooks"),
        Path("configs"),
        Path("data/raw"),
        Path("data/processed"),
        Path("models"),
    ]

    for directory in expected_directories:
        assert directory.exists()
        assert directory.is_dir()


def test_get_settings_returns_default_paths() -> None:
    """Check that default settings are loaded correctly."""
    settings = get_settings()

    assert settings.environment == "development"
    assert settings.data_raw_path == Path("data/raw/credit_card_customers.csv")
    assert settings.model_dir == Path("models")
