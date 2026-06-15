"""Config loading for the planner.

The project intentionally keeps runtime dependencies at zero. This parser only
supports the YAML subset used by the bundled configs: nested mappings, scalar
strings, ints, floats, booleans, and nulls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_default_configs(config_dir: Path = CONFIG_DIR) -> Dict[str, Dict[str, Any]]:
    return {
        "hardware": load_yaml(config_dir / "hardware_profiles.yaml"),
        "models": load_yaml(config_dir / "model_profiles.yaml"),
        "pricing": load_yaml(config_dir / "pricing_profiles.yaml"),
        "workloads": load_yaml(config_dir / "workload_profiles.yaml"),
    }


def load_yaml(path: Path) -> Dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return parse_yaml_mapping(lines)


def parse_yaml_mapping(lines: Iterable[str]) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: list[Tuple[int, Dict[str, Any]]] = [(-1, root)]

    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if raw_line.startswith("\t"):
            raise ValueError(f"Tabs are not supported in YAML at line {line_number}")

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        if ":" not in stripped:
            raise ValueError(f"Expected key/value mapping at line {line_number}: {raw_line}")

        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key:
            raise ValueError(f"Empty key at line {line_number}")

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"Invalid indentation at line {line_number}")

        parent = stack[-1][1]
        if raw_value == "":
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(raw_value)

    return root


def assumption_value(mapping: Dict[str, Any], key: str) -> Any:
    value = mapping[key]
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def _parse_scalar(raw_value: str) -> Any:
    if raw_value in {"null", "Null", "NULL", "~"}:
        return None
    if raw_value in {"true", "True", "TRUE"}:
        return True
    if raw_value in {"false", "False", "FALSE"}:
        return False
    if (raw_value.startswith('"') and raw_value.endswith('"')) or (
        raw_value.startswith("'") and raw_value.endswith("'")
    ):
        return raw_value[1:-1]
    try:
        return int(raw_value)
    except ValueError:
        pass
    try:
        return float(raw_value)
    except ValueError:
        return raw_value
