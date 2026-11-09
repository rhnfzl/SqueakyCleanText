"""Load isolated benchmark adapters without their optional model packages."""

import importlib.util
import sys
from pathlib import Path

ADAPTERS = Path(__file__).resolve().parents[1] / "benchmarks" / "compare" / "adapters"


def adapter_module(name):
    sys.path.insert(0, str(ADAPTERS))
    try:
        spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), ADAPTERS / name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(ADAPTERS))
