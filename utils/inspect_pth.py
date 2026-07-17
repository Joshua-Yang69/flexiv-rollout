"""
Quick .pth checkpoint inspector for ViTacDreamer / ACT checkpoints.
Usage:
    python inspect_pth.py path/to/checkpoint.pth [--full]

    --full   print every tensor key instead of grouped prefixes
"""

import sys
import argparse
from collections import defaultdict
from pathlib import Path

import torch


# ── helpers ────────────────────────────────────────────────────────────────────

def _fmt_shape(t):
    return "x".join(str(d) for d in t.shape) if t.numel() > 0 else "scalar"


def _total_params(state_dict):
    return sum(t.numel() for t in state_dict.values() if isinstance(t, torch.Tensor))


def _detect_model_type(state_dict):
    v2_markers = {"visual_mask_token", "memory_encoder", "masked_visual_embed"}
    if any(any(m in k for m in v2_markers) for k in state_dict):
        return "ViTacDreamerV2 (stage2)"
    return "ViTacDreamer (stage1/v1)"


def _detect_encoder_only(state_dict):
    decoder_prefixes = (
        "decoder.", "tactile_flow_decoder.",
        "depth_delta_decoder.", "marker_flow_decoder.",
    )
    return not any(k.startswith(p) for k in state_dict for p in decoder_prefixes)


def _group_by_prefix(state_dict, depth=2):
    groups = defaultdict(list)
    for k, v in state_dict.items():
        parts = k.split(".")
        prefix = ".".join(parts[:depth])
        groups[prefix].append((k, v))
    return groups


# ── main ───────────────────────────────────────────────────────────────────────

def inspect(path: str, full: bool = False):
    path = Path(path)
    print(f"\n{'='*60}")
    print(f"  {path.name}")
    print(f"  size: {path.stat().st_size / 1e6:.1f} MB")
    print(f"{'='*60}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # ── top-level keys ─────────────────────────────────────────────────────────
    if isinstance(ckpt, dict):
        top_keys = list(ckpt.keys())
        print(f"\n[Top-level keys]  {top_keys}")
    else:
        print(f"\n[Type] {type(ckpt).__name__} (not a dict — treating as bare state_dict)")
        ckpt = {"model_state_dict": ckpt}
        top_keys = list(ckpt.keys())

    # ── config ─────────────────────────────────────────────────────────────────
    config = ckpt.get("config") or ckpt.get("args") or ckpt.get("cfg")
    if config:
        print(f"\n[Config]")
        if isinstance(config, dict):
            for k, v in sorted(config.items()):
                print(f"  {k}: {v}")
        else:
            print(f"  {config}")

    # ── training metadata ──────────────────────────────────────────────────────
    for meta_key in ("epoch", "step", "best_val_loss", "global_step", "optimizer_state_dict"):
        if meta_key in ckpt and meta_key != "optimizer_state_dict":
            print(f"\n[{meta_key}]  {ckpt[meta_key]}")

    if "optimizer_state_dict" in ckpt:
        print(f"\n[optimizer_state_dict]  present (skipped)")

    # ── state dict ─────────────────────────────────────────────────────────────
    state_dict = (
        ckpt.get("model_state_dict")
        or ckpt.get("state_dict")
        or ckpt.get("model")
        or (ckpt if all(isinstance(v, torch.Tensor) for v in ckpt.values()) else None)
    )

    if state_dict is None:
        print("\n[WARNING] no state_dict found under known keys")
        return

    total = _total_params(state_dict)
    model_type = _detect_model_type(state_dict)
    encoder_only = _detect_encoder_only(state_dict)

    print(f"\n[Model type]      {model_type}")
    print(f"[Encoder-only]    {encoder_only}")
    print(f"[Total params]    {total:,}  ({total/1e6:.1f} M)")
    print(f"[Keys]            {len(state_dict)}")

    if full:
        print(f"\n[All keys]")
        for k, v in state_dict.items():
            print(f"  {k:70s}  {_fmt_shape(v):>20s}  {v.dtype}")
    else:
        print(f"\n[Key groups (depth=2)]")
        groups = _group_by_prefix(state_dict, depth=2)
        for prefix, items in sorted(groups.items()):
            n_params = sum(t.numel() for _, t in items)
            print(f"  {prefix:50s}  {len(items):3d} tensors  {n_params/1e6:6.2f} M")

    # ── special tensors useful for config recovery ─────────────────────────────
    probes = {
        "encoder.task_embed.weight":     "num_tasks = shape[0]",
        "encoder.temporal_slot_embed":   "history_len = shape[1]",
        "encoder.delta_t_embed.weight":  "max_delta_t = shape[0]-1",
    }
    found_any = False
    for key, note in probes.items():
        if key in state_dict:
            if not found_any:
                print(f"\n[Config-recoverable tensors]")
                found_any = True
            t = state_dict[key]
            print(f"  {key:45s}  shape={list(t.shape)}  → {note}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a .pth checkpoint")
    parser.add_argument("path", help="Path to .pth file")
    parser.add_argument("--full", action="store_true", help="Print every tensor key")
    args = parser.parse_args()
    inspect(args.path, full=args.full)
