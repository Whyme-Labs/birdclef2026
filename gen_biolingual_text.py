import pandas as pd
import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

tax = pd.read_csv("data/taxonomy.csv")
print(f"Taxonomy: {len(tax)} species")

prompts = []
for _, row in tax.iterrows():
    sci, cn, cls = row["scientific_name"], row["common_name"], row["class_name"]
    if cls == "Aves":
        prompts.append(f"Sound of {cn}, {sci}, a bird")
    elif cls == "Amphibia":
        prompts.append(f"Sound of {cn}, {sci}, a frog")
    elif cls == "Reptilia":
        prompts.append(f"Sound of {cn}, {sci}, a reptile")
    elif cls == "Insecta":
        prompts.append(f"Sound of {cn}, {sci}, an insect")
    elif cls == "Mammalia":
        prompts.append(f"Sound of {cn}, {sci}, a mammal")
    else:
        prompts.append(f"Sound of {cn}, {sci}")

print("Sample prompts:")
for p in prompts[:5]: print(f"  - {p}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
model = ClapModel.from_pretrained("davidrrobinson/BioLingual").to(device)
processor = ClapProcessor.from_pretrained("davidrrobinson/BioLingual")
model.eval()

all_embs = []
with torch.no_grad():
    for i in range(0, len(prompts), 32):
        batch = prompts[i:i+32]
        inputs = processor(text=batch, return_tensors="pt", padding=True).to(device)
        text_feats = model.get_text_features(**inputs)
        text_feats = torch.nn.functional.normalize(text_feats, dim=-1)
        all_embs.append(text_feats.cpu().numpy())

text_embs = np.concatenate(all_embs, axis=0)
print(f"Text embeddings: {text_embs.shape}")
np.save("kaggle_model/biolingual_text_embs.npy", text_embs)
tax[["primary_label"]].to_csv("kaggle_model/biolingual_species_order.csv", index=False)

sim = text_embs @ text_embs.T
np.fill_diagonal(sim, np.nan)
print(f"Text-text cos sim: mean={np.nanmean(sim):.4f}, std={np.nanstd(sim):.4f}, max={np.nanmax(sim):.4f}")
