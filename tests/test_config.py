"""Tests for the Tidely configuration system."""

import pytest

from tidely.core.config import (
    EngineConfig,
    PipelineConfig,
    RuleConfig,
    TidelyConfig,
)
from tidely.core.errors import ConfigurationError


def test_default_config_instantiation() -> None:
    """Config should initialize with sensible defaults."""
    config = TidelyConfig()
    assert config.version == "1.0"
    assert isinstance(config.engine, EngineConfig)
    assert config.engine.lazy is True
    assert config.engine.streaming is False
    assert isinstance(config.pipeline, PipelineConfig)
    assert len(config.pipeline.rules) == 0


def test_yaml_serialization(tmp_path) -> None:  # type: ignore
    """Config should serialize and deserialize to/from YAML correctly."""
    yaml_file = tmp_path / "config.yaml"

    config = TidelyConfig(
        version="1.1",
        engine=EngineConfig(lazy=False, streaming=True, memory_map=False),
        pipeline=PipelineConfig(
            rules=[
                RuleConfig(name="drop_duplicates", params={"subset": ["email"]}),
                RuleConfig(name="fill_missing", params={"column": "age", "value": 0}),
            ]
        ),
    )

    config.to_yaml(str(yaml_file))
    assert yaml_file.exists()

    loaded_config = TidelyConfig.from_yaml(str(yaml_file))
    assert loaded_config.version == "1.1"
    assert loaded_config.engine.lazy is False
    assert loaded_config.engine.streaming is True
    assert loaded_config.engine.memory_map is False

    assert len(loaded_config.pipeline.rules) == 2
    assert loaded_config.pipeline.rules[0].name == "drop_duplicates"
    assert loaded_config.pipeline.rules[0].params == {"subset": ["email"]}
    assert loaded_config.pipeline.rules[1].name == "fill_missing"
    assert loaded_config.pipeline.rules[1].params == {"column": "age", "value": 0}


def test_toml_serialization(tmp_path) -> None:  # type: ignore
    """Config should serialize and deserialize to/from TOML correctly."""
    toml_file = tmp_path / "config.toml"

    config = TidelyConfig(
        version="1.2",
        engine=EngineConfig(lazy=True, streaming=False),
        pipeline=PipelineConfig(
            rules=[
                RuleConfig(name="drop_duplicates", params={"subset": ["id"]}),
            ]
        ),
    )

    config.to_toml(str(toml_file))
    assert toml_file.exists()

    loaded_config = TidelyConfig.from_toml(str(toml_file))
    assert loaded_config.version == "1.2"
    assert loaded_config.engine.lazy is True
    assert len(loaded_config.pipeline.rules) == 1
    assert loaded_config.pipeline.rules[0].name == "drop_duplicates"
    assert loaded_config.pipeline.rules[0].params == {"subset": ["id"]}


def test_config_errors(tmp_path) -> None:  # type: ignore
    """Invalid configurations and missing files should raise ConfigurationError."""
    # File not found
    with pytest.raises(ConfigurationError):
        TidelyConfig.from_yaml(str(tmp_path / "missing.yaml"))

    with pytest.raises(ConfigurationError):
        TidelyConfig.from_toml(str(tmp_path / "missing.toml"))

    # Invalid YAML syntax
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("engine:\n  lazy: [unclosed list", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        TidelyConfig.from_yaml(str(bad_yaml))

    # Invalid TOML syntax
    bad_toml = tmp_path / "bad.toml"
    bad_toml.write_text("engine = {lazy = ", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        TidelyConfig.from_toml(str(bad_toml))

    # Valid YAML syntax but invalid schema
    schema_bad_yaml = tmp_path / "schema_bad.yaml"
    schema_bad_yaml.write_text("engine:\n  lazy: 'not_a_boolean'\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        TidelyConfig.from_yaml(str(schema_bad_yaml))
