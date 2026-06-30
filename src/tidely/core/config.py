"""Configuration management system for Tidely using Pydantic."""

import tomllib
from typing import Any

from pydantic import BaseModel, Field

from tidely.core.errors import ConfigurationError

try:
    import yaml
except ImportError:
    yaml = None


class EngineConfig(BaseModel):
    """Configuration options for the execution engine."""

    model_config = {"extra": "forbid"}

    lazy: bool = Field(
        default=True, description="Enable lazy evaluation if supported by the engine."
    )
    streaming: bool = Field(
        default=False, description="Enable streaming data processing for large files."
    )
    memory_map: bool = Field(
        default=True, description="Use memory mapping for reading files."
    )
    parallel: bool = Field(
        default=True, description="Enable multi-threaded parallel execution."
    )


class RuleConfig(BaseModel):
    """Configuration for a specific cleaning or validation rule."""

    model_config = {"extra": "forbid"}

    name: str = Field(..., description="Name or registration alias of the rule.")
    params: dict[str, Any] = Field(
        default_factory=dict, description="Parameters passed to the rule constructor."
    )


class PipelineConfig(BaseModel):
    """Configuration of the cleaning pipeline."""

    model_config = {"extra": "forbid"}

    rules: list[RuleConfig] = Field(
        default_factory=list, description="Ordered list of rules to execute."
    )


class TidelyConfig(BaseModel):
    """Root configuration for a Tidely run or pipeline."""

    model_config = {"extra": "forbid"}

    version: str = Field(default="1.0", description="Configuration schema version.")
    engine: EngineConfig = Field(
        default_factory=EngineConfig, description="Execution engine configurations."
    )
    pipeline: PipelineConfig = Field(
        default_factory=PipelineConfig, description="Pipeline configuration rules."
    )

    @classmethod
    def from_yaml(cls, path: str) -> "TidelyConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            TidelyConfig: Parsed configuration instance.

        Raises:
            ConfigurationError: If file not found, YAML invalid, or schema invalid.
        """
        if yaml is None:
            raise ConfigurationError(
                "PyYAML is not installed. Install PyYAML to parse YAML configs."
            )
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                raise ConfigurationError(
                    f"YAML content in {path} must be a dictionary."
                )
            return cls.model_validate(data)
        except FileNotFoundError as e:
            raise ConfigurationError(f"Config file not found: {path}") from e
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse YAML file {path}: {e}") from e
        except Exception as e:
            raise ConfigurationError(
                f"Invalid configuration schema in {path}: {e}"
            ) from e

    @classmethod
    def from_toml(cls, path: str) -> "TidelyConfig":
        """Load configuration from a TOML file.

        Args:
            path: Path to the TOML file.

        Returns:
            TidelyConfig: Parsed configuration instance.

        Raises:
            ConfigurationError: If file not found, TOML invalid, or schema invalid.
        """
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
            return cls.model_validate(data)
        except FileNotFoundError as e:
            raise ConfigurationError(f"Config file not found: {path}") from e
        except Exception as e:
            raise ConfigurationError(f"Failed to parse TOML file {path}: {e}") from e

    def to_yaml(self, path: str) -> None:
        """Serialize configuration to a YAML file.

        Args:
            path: Target file path.

        Raises:
            ConfigurationError: If serialization or write fails.
        """
        if yaml is None:
            raise ConfigurationError(
                "PyYAML is not installed. Install PyYAML to dump YAML configs."
            )
        try:
            data = self.model_dump()
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to write configuration to YAML: {e}"
            ) from e

    def to_toml(self, path: str) -> None:
        """Serialize configuration to a TOML file.

        Args:
            path: Target file path.

        Raises:
            ConfigurationError: If serialization or write fails.
        """
        # A simple helper since standard tomllib is read-only, but we can write
        # simple TOML structure manually or raise if too complex. Since
        # TidelyConfig is nested, manual serialization is cleaner without
        # adding tomli-w.
        try:
            # Write a clean, human-readable nested format
            data = self.model_dump()
            lines = [f'version = "{data["version"]}"', ""]

            # Engine section
            lines.append("[engine]")
            for k, v in data["engine"].items():
                val = str(v).lower() if isinstance(v, bool) else repr(v)
                lines.append(f"{k} = {val}")
            lines.append("")

            # Pipeline rules section
            lines.append("[pipeline]")
            lines.append("# Rules are represented as an array of tables")
            for rule in data["pipeline"]["rules"]:
                lines.append("[[pipeline.rules]]")
                lines.append(f'name = "{rule["name"]}"')
                if rule["params"]:
                    import json

                    parts = [
                        f"{pk} = {json.dumps(pv)}" for pk, pv in rule["params"].items()
                    ]
                    lines.append(f"params = {{ {', '.join(parts)} }}")
                lines.append("")

            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception as e:
            raise ConfigurationError(
                f"Failed to write configuration to TOML: {e}"
            ) from e
