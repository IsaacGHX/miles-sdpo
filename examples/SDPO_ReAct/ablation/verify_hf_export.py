"""Sanity-check an HF model dir produced by tools/convert_torch_dist_to_hf.py.

Run: python3 examples/SDPO_ReAct/ablation/verify_hf_export.py <hf_dir>

Checks the things that a megatron->HF conversion can silently get wrong, i.e.
the ones that produce a dir that *looks* complete but loads to garbage:

  1. embedding/lm_head row count == config.json's vocab_size. Megatron pads the
     vocab for tensor parallelism; if --vocab-size was not passed (or was wrong)
     the padding survives into the safetensors and disagrees with config.json.
  2. every shard referenced by the index actually exists, and the index's
     total_size matches the shards on disk.
  3. the tokenizer/chat-template assets --origin-hf-dir was supposed to copy
     are present -- a dir missing chat_template.jinja silently serves with the
     wrong prompt format.

Exits non-zero with a list of problems, so a launcher can gate a Hub push on it.
"""

import json
import os
import struct
import sys

ASSETS = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json")
EMBED_KEYS = ("lm_head.weight", "model.language_model.embed_tokens.weight", "model.embed_tokens.weight")


def safetensors_shape(path: str, key: str) -> list[int]:
    with open(path, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len))
    return header[key]["shape"]


def main(out: str) -> int:
    problems: list[str] = []

    for asset in ASSETS:
        if not os.path.exists(os.path.join(out, asset)):
            problems.append(f"missing asset {asset}")

    index_path = os.path.join(out, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print(f"FAIL: no {index_path}")
        return 1
    index = json.load(open(index_path))
    weight_map = index["weight_map"]
    declared = index["metadata"]["total_size"]
    print(f"tensors: {len(weight_map)}  declared total_size: {declared / 1024**3:.2f} GiB")

    shards = sorted(set(weight_map.values()))
    on_disk = 0
    for shard in shards:
        p = os.path.join(out, shard)
        if not os.path.exists(p):
            problems.append(f"index references missing shard {shard}")
            continue
        on_disk += os.path.getsize(p)
    print(f"shards: {len(shards)}  bytes on disk: {on_disk / 1024**3:.2f} GiB")
    # Shards carry a JSON header on top of the tensor bytes, so on-disk is
    # slightly larger; a big shortfall means a truncated/incomplete write.
    if on_disk < declared:
        problems.append(f"shards total {on_disk} < index total_size {declared} (truncated write?)")

    cfg_path = os.path.join(out, "config.json")
    vocab_size = None
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        vocab_size = cfg.get("text_config", cfg).get("vocab_size")
    print(f"config vocab_size: {vocab_size}")

    found_embed = False
    for key in EMBED_KEYS:
        if key not in weight_map:
            continue
        found_embed = True
        shape = safetensors_shape(os.path.join(out, weight_map[key]), key)
        print(f"  {key} {shape}")
        if vocab_size is not None and shape[0] != vocab_size:
            problems.append(f"{key} rows {shape[0]} != config vocab_size {vocab_size} (pass --vocab-size)")
    if not found_embed:
        problems.append(f"none of {EMBED_KEYS} present in the index")

    if problems:
        print("FAIL:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("OK: shapes agree with config.json, all shards present, assets copied")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
