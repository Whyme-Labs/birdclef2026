import torch
from transformers import AutoModel
model = AutoModel.from_pretrained('DBD-research-group/Bird-MAE-Base', trust_remote_code=True)
model.eval()
print("Type:", type(model).__name__)
print("Params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
x = torch.randn(2, 1, 512, 128)  # T x F per config img_size_x=512, img_size_y=128
with torch.no_grad():
    out = model(x)
print("out type:", type(out).__name__)
if isinstance(out, torch.Tensor):
    print("tensor shape:", out.shape)
else:
    print("dir:", [a for a in dir(out) if not a.startswith('_')][:15])
    for k in ['last_hidden_state', 'pooler_output', 'hidden_states', 'logits']:
        if hasattr(out, k):
            v = getattr(out, k)
            print(f"  {k}:", v.shape if hasattr(v, 'shape') else type(v))
