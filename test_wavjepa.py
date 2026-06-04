import torch
from transformers import AutoModel
m = AutoModel.from_pretrained("labhamlet/wavjepa-nat-base", trust_remote_code=True)
m.eval()
print("Type:", type(m).__name__)
print("Params:", sum(p.numel() for p in m.parameters()) / 1e6, "M")

# Try mono first (duplicate to 2ch)
x = torch.randn(2, 2, 16000 * 5)  # B=2, 2-channel, 5s @ 16kHz
print("Trying input shape:", x.shape)
with torch.no_grad():
    out = m(x)
print("out type:", type(out).__name__)
if isinstance(out, tuple):
    print("Tuple length:", len(out))
    for i, t in enumerate(out):
        if isinstance(t, torch.Tensor):
            print(f"  [{i}]: tensor {t.shape} dtype {t.dtype}")
        else:
            print(f"  [{i}]: {type(t).__name__}")
