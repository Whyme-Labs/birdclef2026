"""Weight-average V194_iter2 + R2_fold1 → single V2S checkpoint.

V199 prep: keep V137's per-model compute budget by collapsing 2 V2S models into 1.
Mirrors Wortsman 2022 model-soup approach used in V177 (which preserved V137=0.941).
"""
import argparse
from pathlib import Path
import torch
from src.models_v2 import SEDModelV2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+",
                    default=["checkpoints/v194_ns_iter2/best_fold0.pt",
                             "checkpoints/effv2s_r2_fold0/best_fold0.pt"])
    ap.add_argument("--weights", nargs="+", type=float,
                    default=[0.5, 0.5])
    ap.add_argument("--out", default="checkpoints/v199_soup/best_fold0.pt")
    ap.add_argument("--backbone", default="tf_efficientnetv2_s.in21k")
    args = ap.parse_args()

    assert len(args.ckpts) == len(args.weights)
    assert abs(sum(args.weights) - 1.0) < 1e-6, "weights must sum to 1"

    print("Loading checkpoints...")
    states = []
    for p in args.ckpts:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        s = ck.get("model_state_dict", ck)
        states.append(s)
        print(f"  {p}: {len(s)} keys")

    # Use first state's keys; warn on missing in others
    keys = list(states[0].keys())
    missing_per_state = [[k for k in keys if k not in s] for s in states]
    for i, miss in enumerate(missing_per_state):
        if miss:
            print(f"  WARN state {i} missing {len(miss)} keys (first 3: {miss[:3]})")

    # Weight-average tensors that exist in ALL states
    avg = {}
    for k in keys:
        if all(k in s for s in states):
            tensors = [s[k].float() for s in states]
            if not all(t.shape == tensors[0].shape for t in tensors):
                print(f"  WARN shape mismatch on {k}; skipping (using first)")
                avg[k] = states[0][k]
                continue
            stacked = torch.stack(tensors)
            wts = torch.tensor(args.weights, dtype=stacked.dtype).view(-1, *([1] * (stacked.ndim - 1)))
            avg[k] = (stacked * wts).sum(0)
        else:
            avg[k] = states[0][k]
    print(f"Souped {len(avg)} parameters")

    # Validate by instantiating model + loading
    model = SEDModelV2(backbone=args.backbone, num_classes=234)
    missing, unexpected = model.load_state_dict(avg, strict=False)
    print(f"Sanity load: missing={len(missing)} unexpected={len(unexpected)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": avg}, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
