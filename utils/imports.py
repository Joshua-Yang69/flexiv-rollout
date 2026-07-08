from __future__ import annotations

import importlib
from types import ModuleType


def optional_import(module_name: str, install_hint: str | None = None) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        message = f"Optional dependency '{module_name}' is required for this device/model."
        if install_hint:
            message += f" {install_hint}"
        raise RuntimeError(message) from exc

