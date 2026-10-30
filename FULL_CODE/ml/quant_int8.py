"""INT8 weight-only variants of the PardaDB GLiNER model (plugin for ``compress.py``).

Every large weight matrix (all ``nn.Linear`` weights, the LSTM ``weight_ih``/``weight_hh`` matrices and
the token-embedding table) is stored as int8 with one float16 scale per output channel (row)::

    scale[r] = max(|W[r, :]|) / 127          (rounded to float16 first)
    Q[r, c]  = round(W[r, c] / scale[r])      clamped to [-127, 127]

Biases, LayerNorm weights and other small tensors are stored as float16. Two formats are registered:

* ``int8-weight``: dequantized to float32 at load, so compute (and accuracy) matches the fp32 model.
* ``int8-weight-dyn``: the same file on disk; at load the encoder's attention output and GeGLU input
  Linear layers (``attn.Wo``, ``mlp.Wi``) are swapped for torch dynamic-quantized int8 Linear layers
  (int8 GEMM on CPU) and the token embedding stays int8 in memory. ``attn.Wqkv``, the GeGLU output
  projection ``mlp.Wo``, the span scorer, projections and LSTM stay float32: int8 activation
  quantization of their inputs costs accuracy (see ``DYN_SKIP``), which is where an earlier
  all-Linear dynamic quantization lost it.

    .venv/bin/python compress.py int8-weight --src models/pardadb-pii --dst models/pardadb-pii-int8w
    .venv/bin/python compress.py int8-weight-dyn --src models/pardadb-pii --dst models/pardadb-pii-int8w-dyn
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import compress
import torch
from gliner import GLiNER
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

log = logging.getLogger(__name__)

WEIGHTS = "model.int8.safetensors"
QSUFFIX = ".qint8"  # int8 matrix, stored under "<param name>.qint8"
SSUFFIX = ".qscale"  # float16 per-row scales, stored under "<param name>.qscale"
QMAX = 127
MIN_NUMEL = 16_384  # 2-D tensors smaller than this (e.g. the 3x3072 output layer) stay float16
EMBEDDINGS = "token_rep_layer.bert_layer.model.embeddings.tok_embeddings"
ENCODER = "token_rep_layer.bert_layer"  # Linear layers under this prefix are dynamic-quantized in -dyn
# Encoder Linear layers kept float32 in -dyn because int8 activations cost accuracy there. Bank F1 with only
# that layer type dynamic-quantized (320-text subset; 0.988 with none): mlp.Wo 0.901 (outlier activations after
# GeGLU), attn.Wqkv 0.977, attn.Wo 0.988, mlp.Wi 0.993. All four together: bank F1 0.81 on the full test set.
DYN_SKIP = ("mlp.Wo", "attn.Wqkv")


def quantize_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row int8 quantization of a 2-D tensor; returns ``(int8 values, float16 scales)``."""
    w = w.float()
    scale = (w.abs().amax(dim=1) / QMAX).to(torch.float16)
    scale[scale == 0] = 1.0  # all-zero rows: any scale works, avoid dividing by zero
    q = torch.round(w / scale.float()[:, None]).clamp_(-QMAX, QMAX).to(torch.int8)
    return q.contiguous(), scale.contiguous()


def dequantize_rows(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`quantize_rows`, in float32."""
    return q.float() * scale.float()[:, None]


def _quantizable(name: str, t: torch.Tensor, int8_embeddings: bool) -> bool:
    """Whether a checkpoint tensor is stored as per-row int8 (large 2-D float weights)."""
    if not (t.is_floating_point() and t.dim() == 2 and t.numel() >= MIN_NUMEL):
        return False
    return int8_embeddings or not name.startswith(EMBEDDINGS)


def quantize_checkpoint(src_file: Path, dst_file: Path, int8_embeddings: bool = True) -> dict[str, Any]:
    """Write an int8 weight-only copy of ``src_file`` (fp32 safetensors) to ``dst_file``.

    Returns statistics for the manifest: how many tensors were quantized and the worst relative
    reconstruction error (max |W - W'| / max |W| over each quantized tensor).
    """
    tensors: dict[str, torch.Tensor] = {}
    n_int8 = n_fp16 = 0
    worst = 0.0
    with safe_open(str(src_file), framework="pt") as f:
        for key in f.keys():  # noqa: SIM118 - safe_open handles are not iterable
            t = f.get_tensor(key)
            if _quantizable(key, t, int8_embeddings):
                q, scale = quantize_rows(t)
                peak = t.abs().max().item()
                if peak > 0:
                    worst = max(worst, (dequantize_rows(q, scale) - t.float()).abs().max().item() / peak)
                tensors[key + QSUFFIX], tensors[key + SSUFFIX] = q, scale
                n_int8 += 1
            elif t.is_floating_point():
                tensors[key] = compress.to_fp16(key, t)
                n_fp16 += 1
            else:
                tensors[key] = t.contiguous()
    save_file(tensors, str(dst_file), metadata={"format": "pt", "quantization": "int8 symmetric per-row"})
    return {"int8_tensors": n_int8, "fp16_tensors": n_fp16, "max_rel_error": round(worst, 5)}


def iter_dequantized(path: Path, device: str = "cpu") -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(parameter name, float32 tensor)`` for a file written by :func:`quantize_checkpoint`.

    Tensors are read and dequantized one at a time, so the full float32 state dict never exists at once.
    """
    with safe_open(str(path), framework="pt", device=device) as f:
        for key in f.keys():  # noqa: SIM118 - safe_open handles are not iterable
            if key.endswith(SSUFFIX):
                continue
            if key.endswith(QSUFFIX):
                name = key[: -len(QSUFFIX)]
                yield name, dequantize_rows(f.get_tensor(key), f.get_tensor(name + SSUFFIX))
            else:
                t = f.get_tensor(key)
                yield key, t.float() if t.is_floating_point() else t


def load_into(net: nn.Module, path: Path) -> None:
    """Copy the dequantized weights of ``path`` into ``net`` in place, checking names and shapes."""
    targets = net.state_dict()
    loaded = set()
    with torch.no_grad():
        for name, t in iter_dequantized(path):
            if name not in targets or targets[name].shape != t.shape:
                raise ValueError(f"{path.name}: {name} {tuple(t.shape)} does not fit the model")
            targets[name].copy_(t)
            loaded.add(name)
    missing = sorted(set(targets) - loaded)
    if missing:
        raise ValueError(f"{path.name} lacks {len(missing)} tensors, e.g. {missing[:3]}")


def _build(src: Path, dst: Path, fmt: str, int8_embeddings: bool, compute: str) -> None:
    """Quantize ``src/model.safetensors`` into ``dst`` and write the support files and manifest."""
    stats = quantize_checkpoint(src / compress.WEIGHTS, dst / WEIGHTS, int8_embeddings=int8_embeddings)
    compress.copy_support_files(src, dst)
    compress.write_manifest(
        dst,
        fmt,
        __name__,
        src,
        [WEIGHTS],
        scheme="int8 symmetric per output channel, float16 scales; biases and norms float16",
        embeddings="int8 per row" if int8_embeddings else "float16",
        compute_dtype=compute,
        **stats,
    )
    log.info("%s: %s", fmt, stats)


def build_int8_weight(src: Path, dst: Path) -> None:
    """Build the ``int8-weight`` variant (int8 weights on disk, float32 compute)."""
    _build(src, dst, "int8-weight", int8_embeddings=True, compute="float32")


def build_int8_weight_dyn(src: Path, dst: Path) -> None:
    """Build the ``int8-weight-dyn`` variant (same file; the loader keeps the encoder in int8)."""
    _build(src, dst, "int8-weight-dyn", True, "dynamic int8 for encoder attn.Wo and mlp.Wi, float32 elsewhere")


def _gliner_class(model_dir: Path) -> type:
    """The concrete GLiNER class for this checkpoint, made to build the model without reading weights.

    GLiNER's standard load path builds a randomly initialised model and then loads a state dict into it.
    Returning an empty state dict (with the default ``strict=False``) leaves only the model allocated;
    :func:`load_int8_weight` then streams the int8 weights in, so peak memory stays near one model copy.
    """
    config = json.loads((model_dir / "gliner_config.json").read_text(encoding="utf-8"))
    base = GLiNER._get_gliner_class(GLiNER._config_from_dict(config))

    def resolve_model_file(cls: type, directory: Path, variant: str | None) -> tuple[Path, None]:
        return Path(directory) / WEIGHTS, None

    def no_state_dict(cls: type, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        return {}

    return type(
        f"Int8Weight{base.__name__}",
        (base,),
        {"_resolve_model_file": classmethod(resolve_model_file), "_load_state_dict": classmethod(no_state_dict)},
    )


def load_int8_weight(model_dir: Path, device: str = "cpu") -> Any:
    """Load an int8 weight-only variant as an eval-mode GLiNER computing in float32."""
    model_dir = Path(model_dir)
    cls = _gliner_class(model_dir)
    model = cls.from_pretrained(
        model_id=str(model_dir),
        model_dir=model_dir,
        local_files_only=True,
        map_location="cpu",
        dtype=torch.float32,
        low_cpu_mem_usage=False,
    )
    model.__class__ = cls.__mro__[1]  # the stock GLiNER class; the loader subclass is only needed while loading
    load_into(model.model, model_dir / WEIGHTS)
    model.model.to(device)
    return model.eval()


class Int8Embedding(nn.Module):
    """Embedding table kept as int8 rows plus float16 scales; rows are dequantized on lookup."""

    def __init__(self, weight: torch.Tensor, padding_idx: int | None = None) -> None:
        super().__init__()
        q, scale = quantize_rows(weight)
        self.register_buffer("qweight", q)
        self.register_buffer("scale", scale.float())
        self.padding_idx = padding_idx
        self.num_embeddings, self.embedding_dim = q.shape

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Look up rows of ``ids`` and return them in float32."""
        return self.qweight[ids].float() * self.scale[ids].unsqueeze(-1)


def _set_module(root: nn.Module, name: str, module: nn.Module) -> None:
    """Replace the submodule at dotted path ``name``."""
    parent, _, child = name.rpartition(".")
    setattr(root.get_submodule(parent) if parent else root, child, module)


def load_int8_weight_dyn(model_dir: Path, device: str = "cpu") -> Any:
    """Load as :func:`load_int8_weight`, then run most encoder Linear layers and the embedding in int8."""
    if device != "cpu":
        raise ValueError("int8-weight-dyn uses CPU int8 kernels; load int8-weight for other devices")
    model = load_int8_weight(model_dir, device)
    net = model.model
    encoder = net.get_submodule(ENCODER)
    # Per-channel qconfig: the weights already sit on a per-row int8 grid, so requantizing them is lossless;
    # the error comes from quantizing activations, which is why DYN_SKIP layers stay float32.
    qconfig = torch.ao.quantization.per_channel_dynamic_qconfig
    spec = {
        name: qconfig
        for name, mod in encoder.named_modules()
        if isinstance(mod, nn.Linear) and not name.endswith(DYN_SKIP)
    }
    torch.ao.quantization.quantize_dynamic(encoder, spec, dtype=torch.qint8, inplace=True)
    emb = net.get_submodule(EMBEDDINGS)
    _set_module(net, EMBEDDINGS, Int8Embedding(emb.weight.detach(), emb.padding_idx))
    return model.eval()


compress.register("int8-weight", build_int8_weight, load_int8_weight, "int8 per-channel weights, float32 compute")
compress.register(
    "int8-weight-dyn",
    build_int8_weight_dyn,
    load_int8_weight_dyn,
    "int8 per-channel weights; encoder attn.Wo/mlp.Wi run as dynamic int8 on CPU",
)
