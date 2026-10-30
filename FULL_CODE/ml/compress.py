"""Compressed variants of the PardaDB GLiNER model: one build CLI and one loader for every format.

Build a variant from the fine-tuned float32 checkpoint::

    .venv/bin/python compress.py int8-weight --src models/pardadb-pii --dst models/pardadb-pii-int8w
    .venv/bin/python compress.py --list

Formats (results in ``reports/compression.json``):

* ``fp32``, ``fp16`` (this module): plain copy, and float16 safetensors upcast to float32 at load.
* ``int8-weight`` (chosen), ``int8-weight-dyn`` (``quant_int8.py``): per-channel int8 safetensors.
* ``onnx-*`` (``quant_onnx.py``): ONNX export, int8 via ONNX Runtime dynamic quantization.

A variant is a directory holding everything needed to load it (weights, ``gliner_config.json``,
tokenizer files) plus ``compression.json``, a small manifest::

    {"format": "fp16", "module": "compress", "source": "models/pardadb-pii", "weights": ["model.safetensors"]}

``format`` picks the loader and ``module`` names the ``ml/`` module that registers it. ``weights`` lists
the files that hold the parameters (used for size reporting). A directory without ``compression.json``
is a plain fp32 GLiNER checkpoint. Formats live in plugin modules named ``quant_*.py``, which call
:func:`register` at import; this module imports them on demand, so adding a format needs no edit here.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from gliner import GLiNER  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

MODULE = "compress"  # import name used in manifests, also when run as a script
log = logging.getLogger(MODULE)

MANIFEST = "compression.json"
WEIGHTS = "model.safetensors"
SOURCE_DIR = ROOT / "models" / "pardadb-pii"  # fine-tuned float32 checkpoint (ml/train.py)
CHOSEN_DIR = ROOT / "models" / "pardadb-pii-int8w"  # variant chosen in reports/compression.json
# Files that are rewritten by a build rather than copied from the source checkpoint.
_WEIGHT_SUFFIXES = (".safetensors", ".onnx", ".bin", ".pt")
FP16_MAX = 65504.0

BuildFn = Callable[[Path, Path], None]
LoadFn = Callable[[Path, str], Any]


@dataclass(frozen=True)
class Format:
    """A storage format: how to build a variant directory and how to load it back."""

    name: str
    build: BuildFn
    load: LoadFn
    description: str = ""


FORMATS: dict[str, Format] = {}


def register(name: str, build: BuildFn, load: LoadFn, description: str = "") -> None:
    """Register a format. Re-registering the same name replaces it (safe on module reload)."""
    FORMATS[name] = Format(name, build, load, description)


def import_plugins() -> None:
    """Import every ``ml/quant_*.py`` so the formats they register become available."""
    for path in sorted(ROOT.glob("quant_*.py")):
        importlib.import_module(path.stem)


def read_manifest(model_dir: Path) -> dict[str, Any]:
    """Return ``compression.json`` for a variant, or the implicit fp32 manifest if it is absent."""
    path = Path(model_dir) / MANIFEST
    if not path.is_file():
        return {"format": "fp32", "module": MODULE, "weights": [WEIGHTS]}
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(dst: Path, fmt: str, module: str, src: Path, weights: list[str], **extra: Any) -> None:
    """Write ``compression.json`` describing a built variant."""
    manifest = {"format": fmt, "module": module, "source": str(src), "weights": weights, **extra}
    (Path(dst) / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def copy_support_files(src: Path, dst: Path) -> list[str]:
    """Copy config and tokenizer files (everything except weights and the manifest) from src to dst."""
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(Path(src).iterdir()):
        if path.is_file() and path.name != MANIFEST and not path.name.endswith(_WEIGHT_SUFFIXES):
            shutil.copy2(path, dst / path.name)
            copied.append(path.name)
    return copied


def to_fp16(name: str, t: torch.Tensor) -> torch.Tensor:
    """Cast a floating-point tensor to contiguous float16, refusing values outside the float16 range."""
    if t.numel() and t.abs().max().item() > FP16_MAX:
        raise ValueError(f"{name} has values outside the float16 range")
    return t.to(torch.float16).contiguous()


def current_rss_mb() -> float:
    """Resident set size of this process in MB (1e6 bytes), read from /proc; 0.0 where unavailable."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) * 1024 / 1e6, 1)
    except OSError:
        pass
    return 0.0


def default_model_dir() -> Path:
    """The chosen compressed variant if it has been built, else the float32 checkpoint."""
    return CHOSEN_DIR if (CHOSEN_DIR / MANIFEST).is_file() else SOURCE_DIR


def _resolve(fmt: str, module: str | None = None) -> Format:
    """Find a registered format, importing its plugin module (or every ``ml/quant_*.py``) if needed."""
    if fmt not in FORMATS and module:
        importlib.import_module(module)
    if fmt not in FORMATS:
        import_plugins()
    if fmt not in FORMATS:
        raise KeyError(f"unknown format {fmt!r}; registered: {sorted(FORMATS)}")
    return FORMATS[fmt]


def load_model(model_dir: str | Path, device: str = "cpu", threads: int | None = None) -> Any:
    """Load any variant directory as a ready-to-run, eval-mode model (float32 compute).

    Args:
        model_dir: A plain GLiNER checkpoint or a directory written by :func:`build`.
        device: Torch device; the int8 and ONNX formats target ``"cpu"``.
        threads: If set, ``torch.set_num_threads(threads)`` before loading; ONNX loaders read it back.

    Returns:
        An object with GLiNER's ``inference(texts, labels, threshold=..., batch_size=...)`` API.
    """
    if threads:
        torch.set_num_threads(threads)
    model_dir = Path(model_dir)
    manifest = read_manifest(model_dir)
    fmt = _resolve(manifest["format"], manifest.get("module"))
    log.info("loading %s as %s", model_dir, fmt.name)
    return fmt.load(model_dir, device)


def build(fmt: str, src: str | Path, dst: str | Path) -> Path:
    """Build variant ``fmt`` of the checkpoint at ``src`` into ``dst`` and return ``dst``."""
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        raise ValueError("dst must differ from src")
    dst.mkdir(parents=True, exist_ok=True)
    _resolve(fmt).build(src, dst)
    log.info("built %s -> %s", fmt, dst)
    return dst


def load_torch(model_dir: Path, device: str) -> GLiNER:
    """Load a safetensors GLiNER checkpoint of any float dtype, cast to float32 on read."""
    model = GLiNER.from_pretrained(str(model_dir), local_files_only=True, dtype=torch.float32, map_location=device)
    return model.eval()


def _build_fp32(src: Path, dst: Path) -> None:
    """Copy the checkpoint unchanged (the reference variant)."""
    copy_support_files(src, dst)
    shutil.copy2(src / WEIGHTS, dst / WEIGHTS)
    write_manifest(dst, "fp32", MODULE, src, [WEIGHTS])


def _build_fp16(src: Path, dst: Path) -> None:
    """Store every floating-point tensor as float16 safetensors; other tensors are kept as they are."""
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(src / WEIGHTS), framework="pt") as f:
        metadata = f.metadata()
        for key in f.keys():  # noqa: SIM118 - safe_open handles are not iterable
            t = f.get_tensor(key)
            tensors[key] = to_fp16(key, t) if t.is_floating_point() else t.contiguous()
    save_file(tensors, str(dst / WEIGHTS), metadata=metadata)
    copy_support_files(src, dst)
    write_manifest(dst, "fp16", MODULE, src, [WEIGHTS], compute_dtype="float32")


register("fp32", _build_fp32, load_torch, "original float32 safetensors")
register("fp16", _build_fp16, load_torch, "float16 safetensors, upcast to float32 at load")


def dir_size_mb(path: Path) -> float:
    """Total size of the files under ``path`` in MB (1e6 bytes)."""
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


def main() -> None:
    """CLI: ``compress.py FORMAT --src DIR --dst DIR``, or ``compress.py --list``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import_plugins()
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("format", nargs="?", choices=sorted(FORMATS))
    ap.add_argument("--src", type=Path, default=SOURCE_DIR, help="float32 GLiNER checkpoint (default: %(default)s)")
    ap.add_argument("--dst", type=Path, help="output variant directory")
    ap.add_argument("--list", action="store_true", help="list the registered formats and exit")
    args = ap.parse_args()
    if args.list:
        for fmt in sorted(FORMATS.values(), key=lambda f: f.name):
            log.info("%-16s %s", fmt.name, fmt.description)
        return
    if not (args.format and args.dst):
        ap.error("FORMAT and --dst are required (or use --list)")
    dst = build(args.format, args.src, args.dst)
    log.info("%s: %.1f MB on disk (1 MB = 1e6 bytes)", dst, dir_size_mb(dst))


if __name__ == "__main__":
    # Plugins do ``import compress``; point that name at this module so they share one FORMATS registry.
    sys.modules.setdefault(MODULE, sys.modules[__name__])
    main()
