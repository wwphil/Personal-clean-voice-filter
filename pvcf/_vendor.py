"""Bootstrap vendored model sources without triggering heavy package __init__.

The upstream `wesep` and `wespeaker` packages import CLI/training dependencies
(tqdm, kaldiio, s3prl, ...) at package import time. We only need model classes,
so we register bare package skeletons — correct `__path__`, no `__init__`
execution — for exactly the packages whose `__init__` is heavy:

- `wesep`          (its __init__ imports the CLI: hub/tqdm/requests/silero)
- `wespeaker`      (its __init__ imports the CLI: kaldiio/hub)
- `wespeaker.frontend` (imports s3prl, whisper, w2vbert frontends — unused by us)

Submodules (wesep.models, wespeaker.models, ...) load naturally via the
skeletons' `__path__`, executing their own (light) `__init__` files.
`s3prl` is stubbed because `frontend/s3prl.py` imports it at module level.

Usage:
    from pvcf._vendor import ensure_wesep, ensure_wespeaker
    ensure_wesep()
    ensure_wespeaker()
    from wesep.models import get_model
    from wesep.utils.checkpoint import load_pretrained_model
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor"


def _skeleton(name: str, path: Path) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]  # type: ignore[return-value]
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]  # marks it as a package for the path finder
    sys.modules[name] = mod
    return mod


def _stub_s3prl() -> None:
    if "s3prl" in sys.modules:
        return
    s3 = types.ModuleType("s3prl")
    s3_nn = types.ModuleType("s3prl.nn")
    s3_nn.Featurizer = object  # type: ignore[attr-defined]
    s3_nn.S3PRLUpstream = object  # type: ignore[attr-defined]
    s3.util = types.SimpleNamespace(  # type: ignore[attr-defined]
        download=types.SimpleNamespace(set_dir=lambda *a, **k: None)
    )
    sys.modules["s3prl"] = s3
    sys.modules["s3prl.nn"] = s3_nn


def ensure_wesep() -> None:
    _skeleton("wesep", VENDOR / "wesep" / "wesep")


def ensure_wespeaker() -> None:
    _skeleton("wespeaker", VENDOR / "WeSpeaker" / "wespeaker")
    _skeleton("wespeaker.frontend", VENDOR / "WeSpeaker" / "wespeaker" / "frontend")
    _stub_s3prl()
