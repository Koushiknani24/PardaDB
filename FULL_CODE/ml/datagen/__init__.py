"""Synthetic PII training data generator (our team's pre-built code, written before Day 1).

India-first (60%) with global coverage (40%); every identifier is synthetic
and checksum-valid where the real format has a checksum; a held-out set of
name components never appears in train/dev text so the test split measures
generalisation to unseen names. Standard library only.

    python -m datagen.generate --n 20000 --seed 7 --split train --out data/train.jsonl
"""
