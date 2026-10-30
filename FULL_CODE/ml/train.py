"""Fine-tune GLiNER-PII edge for PardaDB on the RTX 3050 (CUDA, fp16).

.venv/bin/python train.py --max-steps 1500
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from gliner import GLiNER  # noqa: E402


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compose(examples, rate, rng, max_tokens=256):
    """Stitch a share of examples together so the model also sees multi-sentence text."""
    out = []
    for ex in examples:
        if rng.random() >= rate:
            out.append(ex)
            continue
        tokens, ner = [], []
        for part in [ex] + rng.sample(examples, rng.choice([1, 2])):
            if tokens and len(tokens) + len(part["tokenized_text"]) > max_tokens:
                break
            off = len(tokens)
            tokens += part["tokenized_text"]
            ner += [[s + off, e + off, label] for s, e, label in part["ner"]]
        out.append({"tokenized_text": tokens, "ner": ner})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="knowledgator/gliner-pii-edge-v1.0")
    ap.add_argument("--name", default="pardadb-pii")
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--others-lr", type=float, default=8e-5)
    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    train = compose(read_jsonl(ROOT / "dataset" / "train.jsonl"), 0.25, rng)
    dev = read_jsonl(ROOT / "dataset" / "dev.jsonl")
    random.shuffle(train)
    if not torch.cuda.is_available():
        raise SystemExit("training needs a CUDA GPU (fp16); none was found")
    out_dir = ROOT / "models" / "checkpoints" / args.name
    print(f"train={len(train)} dev={len(dev)} gpu={torch.cuda.get_device_name(0)} out={out_dir}", flush=True)

    model = GLiNER.from_pretrained(args.base)
    training_args = model.create_training_args(
        output_dir=str(out_dir),
        learning_rate=args.lr,
        others_lr=args.others_lr,
        weight_decay=0.01,
        others_weight_decay=0.01,
        focal_loss_alpha=0.75,
        focal_loss_gamma=2.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.06,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=1,
        max_steps=args.max_steps,
        save_steps=args.eval_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        fp16=True,
        dataloader_num_workers=2,
        report_to="none",
        seed=args.seed,
    )
    started = time.time()
    trainer = model.train_model(train_dataset=train, eval_dataset=dev, training_args=training_args)
    final = ROOT / "models" / args.name
    trainer.model.save_pretrained(str(final), safe_serialization=True)  # safetensors, no pickle
    print(f"done in {(time.time() - started) / 60:.1f} min -> {final}", flush=True)


if __name__ == "__main__":
    main()
