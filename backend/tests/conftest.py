import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["ADOBEPS_BACKEND"] = "stub"   # tests never download models


@pytest.fixture(autouse=True)
def _tmp_outputs(tmp_path, monkeypatch):
    """Point outputs/logs/cache at a temp dir so tests don't litter the project."""
    import yaml

    from core import config as cfgmod

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["backend"] = "stub"
    for k in ("outputs", "logs", "comparisons", "cache"):
        cfg["paths"][k] = str(tmp_path / k)
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("ADOBEPS_CONFIG", str(p))
    cfgmod.get_config.cache_clear()
    yield
    cfgmod.get_config.cache_clear()
