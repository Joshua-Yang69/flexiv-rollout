from __future__ import annotations

from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar

T = TypeVar("T")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file.

    PyYAML is intentionally imported lazily so modules can still be imported on
    machines where only syntax checks are being run.
    """
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load rollout .yml configs.") from exc

    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping at top level.")
    return data


def deep_get(mapping: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def dataclass_from_mapping(cls: type[T], data: Mapping[str, Any] | None) -> T:
    """Construct a dataclass from a mapping, ignoring unknown keys."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass type")
    data = data or {}
    field_names = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs = {key: value for key, value in data.items() if key in field_names}
    return cls(**kwargs)

