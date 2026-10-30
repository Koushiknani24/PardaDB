"""ONNX export of the PardaDB GLiNER model with int8 dynamic quantization, run by ONNX Runtime on CPU.

The float32 checkpoint is traced with GLiNER's own exporter (``export_to_onnx``, opset 19) and then
quantized with ``onnxruntime.quantization.quantize_dynamic``: weights are stored as int8, activations
are quantized on the fly at inference. Pre- and post-processing (tokenisation, span decoding,
thresholding) stay GLiNER's: a variant is loaded with ``GLiNER.from_pretrained(..., load_onnx_model=True)``,
so its predictions compare directly with the torch model.

Formats registered here (the variant-directory contract is in ``compress.py``):

* ``onnx-int8-fp16`` (fastest on CPU): token embeddings, LSTM and span scorer are int8; the transformer
  encoder weights are stored as float16 and cast to float32 when the session is built. 58.6 MB of
  weights, bank/golden F1 0.988/0.935 (float32: 0.984/0.936), 1.5x faster than torch on CPU.
* ``onnx-int8``: int8 everywhere except the ten MLP down-projections (``layers.N/mlp/Wo``, whose GeGLU
  inputs have outlier activations that per-tensor dynamic quantization clips). Smallest usable variant.
* ``onnx-int8-qkv``: as ``onnx-int8``, with the attention QKV projections also kept in float32.
* ``onnx-int8-head``: as ``onnx-int8-fp16`` but with the encoder stored in float32.
* ``onnx-fp32``: the exported graph without quantization, to check the export itself.

Tried and dropped: every weight int8 with per-channel scales (the textbook recipe) scored F1 0.0, because
per-channel scales on the QKV projections alone break the model (measurement kept in
``reports/compression/onnx-int8-all.json``). Encoder int8 sensitivity was measured per sub-module: QKV and
MLP down-projection lose accuracy, the attention output and MLP up-projection do not.
Results are in ``reports/compression/onnx-*.json``.

Build and benchmark::

    .venv/bin/python compress.py onnx-int8-fp16 --src models/pardadb-pii --dst models/pardadb-pii-onnx-int8-fp16
    .venv/bin/python benchmark.py --variant onnx-int8-fp16 --model-dir models/pardadb-pii-onnx-int8-fp16 \\
        --out reports/compression/onnx-int8-fp16.json --threads 4
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import compress
import numpy as np
import onnx
import onnxruntime as ort
import torch
from gliner import GLiNER
from onnx import helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic

log = logging.getLogger(__name__)

OPSET = 19
FP32_FILE = "model.onnx"
INT8_FILE = "model.int8.onnx"
# Labels used for the dummy batch that is traced; the exported graph accepts any label set.
EXPORT_LABELS = ["person", "address", "organization"]
# Op types whose constant weight input quantize_dynamic can turn into int8.
QUANT_OPS = ("MatMul", "Gather", "LSTM")
ENCODER = r"^/core/token_rep_layer/"


@dataclass(frozen=True)
class Policy:
    """Which weights of the exported graph become int8.

    Attributes:
        keep_fp32: Regex over ONNX node names; matching weight nodes are left in float32.
        fp16_storage: Store the weights that are not int8 as float16 behind a ``Cast`` node. ONNX Runtime
            folds the cast when it builds the session, so they are computed in float32 as before.
    """

    keep_fp32: str | None
    fp16_storage: bool = False


POLICIES: dict[str, Policy] = {
    "onnx-int8": Policy(keep_fp32=r"/mlp/Wo/"),
    "onnx-int8-qkv": Policy(keep_fp32=r"/(mlp/Wo|attn/Wqkv)/"),
    "onnx-int8-head": Policy(keep_fp32=ENCODER + r".*/MatMul$"),
    "onnx-int8-fp16": Policy(keep_fp32=ENCODER + r".*/MatMul$", fp16_storage=True),
}


def _export_fp32(src: Path, out_dir: Path) -> Path:
    """Trace the float32 checkpoint at ``src`` to ``out_dir/model.onnx`` with GLiNER's exporter."""
    model = compress.load_torch(src, "cpu")
    paths = model.export_to_onnx(out_dir, onnx_filename=FP32_FILE, quantize=False, opset=OPSET, labels=EXPORT_LABELS)
    del model
    return Path(paths["onnx_path"])


def weight_nodes(onnx_path: Path) -> list[str]:
    """Names of the nodes that quantize_dynamic can quantize: MatMul/Gather/LSTM with a constant weight."""
    graph = onnx.load(str(onnx_path), load_external_data=False).graph
    constants = {t.name for t in graph.initializer}
    return [n.name for n in graph.node if n.op_type in QUANT_OPS and any(i in constants for i in n.input)]


def store_fp16(onnx_path: Path, node_names: list[str]) -> int:
    """Rewrite the float32 weights of ``node_names`` in ``onnx_path`` as float16 initializers plus a Cast.

    The Cast output keeps the original tensor name, so every consumer is unchanged. Returns the number
    of tensors converted.
    """
    model = onnx.load(str(onnx_path))
    graph = model.graph
    inits = {t.name: t for t in graph.initializer}
    wanted = set(node_names)
    casts = []
    for node in graph.node:
        if node.name not in wanted:
            continue
        for name in node.input:
            tensor = inits.pop(name, None)  # pop: a weight shared by two nodes is converted once
            if tensor is None or tensor.data_type != onnx.TensorProto.FLOAT:
                continue
            half = numpy_helper.from_array(numpy_helper.to_array(tensor).astype(np.float16), name + "_fp16")
            graph.initializer.remove(tensor)
            graph.initializer.append(half)
            casts.append(helper.make_node("Cast", [half.name], [name], name=name + "_cast", to=onnx.TensorProto.FLOAT))
    for i, cast in enumerate(casts):  # constants first keeps the graph topologically sorted
        graph.node.insert(i, cast)
    onnx.save(model, str(onnx_path))
    return len(casts)


def quantize(fp32_path: Path, out_path: Path, policy: Policy) -> dict[str, object]:
    """Write an int8 copy of ``fp32_path`` to ``out_path`` following ``policy``; return manifest details."""
    nodes = weight_nodes(fp32_path)
    keep = [n for n in nodes if policy.keep_fp32 and re.search(policy.keep_fp32, n)]
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(out_path),
        op_types_to_quantize=list(QUANT_OPS),
        weight_type=QuantType.QInt8,
        per_channel=False,  # one scale per tensor: per-channel scales broke the QKV projections
        nodes_to_exclude=keep,
    )
    log.info("quantized %d of %d weight nodes to int8 (%d kept float)", len(nodes) - len(keep), len(nodes), len(keep))
    fp16_tensors = store_fp16(out_path, keep) if policy.fp16_storage else 0
    return {
        "weight_type": "QInt8",
        "activations": "dynamic uint8, per tensor",
        "per_channel": False,
        "keep_fp32_regex": policy.keep_fp32,
        "int8_nodes": len(nodes) - len(keep),
        "float_nodes": keep,
        "float_storage": "float16, cast to float32 at session load" if policy.fp16_storage else "float32",
        "fp16_tensors": fp16_tensors,
    }


def _build(src: Path, dst: Path, fmt: str) -> None:
    """Export ``src`` to ONNX, quantize it as format ``fmt`` prescribes, and write a variant into ``dst``."""
    policy = POLICIES.get(fmt)
    with tempfile.TemporaryDirectory(prefix="onnx-export-") as tmp:
        fp32_path = _export_fp32(src, Path(tmp))
        if policy is None:
            weights, extra = FP32_FILE, {}
            shutil.copy2(fp32_path, dst / weights)
        else:
            weights = INT8_FILE
            extra = quantize(fp32_path, dst / weights, policy)
    compress.copy_support_files(src, dst)
    compress.write_manifest(dst, fmt, __name__, src, [weights], opset=OPSET, runtime="onnxruntime", **extra)


def load(model_dir: Path, device: str = "cpu") -> GLiNER:
    """Load an ONNX variant as a GLiNER object backed by an ONNX Runtime CPU session.

    The session uses ``torch.get_num_threads()`` intra-op threads (``compress.load_model`` sets that from
    ``--threads``) and full graph optimisation.
    """
    if device != "cpu":
        raise ValueError("ONNX variants are built for the CPU execution provider")
    weights = compress.read_manifest(model_dir)["weights"][0]
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = torch.get_num_threads()
    options.inter_op_num_threads = 1
    return GLiNER.from_pretrained(
        str(model_dir),
        local_files_only=True,
        load_onnx_model=True,
        onnx_model_file=weights,
        session_options=options,
        map_location="cpu",
    )


def _builder(fmt: str) -> compress.BuildFn:
    """Bind ``fmt`` into a ``build(src, dst)`` function for the registry."""

    def build(src: Path, dst: Path) -> None:
        _build(src, dst, fmt)

    return build


_DESCRIPTIONS = {
    "onnx-int8": "ONNX Runtime CPU; int8 weights except the float32 MLP down-projections",
    "onnx-int8-qkv": "ONNX Runtime CPU; as onnx-int8, plus float32 attention QKV projections",
    "onnx-int8-head": "ONNX Runtime CPU; int8 embeddings, LSTM and span scorer, float32 encoder",
    "onnx-int8-fp16": "ONNX Runtime CPU; int8 embeddings, LSTM and span scorer, encoder stored float16",
    "onnx-fp32": "ONNX Runtime CPU; float32 export, no quantization",
}
for _fmt, _text in _DESCRIPTIONS.items():
    compress.register(_fmt, _builder(_fmt), load, _text)
