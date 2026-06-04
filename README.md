# BirdCLEF+ 2026 — Whyme Labs

**Final: 289 / 4243 (top 6.8%, 🥉 bronze). Public LB 0.950 · Private LB 0.942.**

Multi-taxa bioacoustic classification (234 species — birds, amphibians, mammals, reptiles, insects) in Brazilian Pantanal soundscapes. CPU-only Kaggle notebooks, 90-minute runtime, macro-averaged ROC-AUC.

This repository is unusual: it is **not** a winning recipe. It is a rigorous, measurement-driven record of *why a strong shared public baseline (0.950) could not be beaten with one consumer GPU plus a small cloud budget* — and a reusable framework for testing improvement hypotheses cheaply, before spending scarce submission slots.

📄 **Read the full account:** [`WORKING_NOTE.md`](WORKING_NOTE.md) (our CLEF working-note draft).

---

## TL;DR of findings

The public field converged on a single pipeline — **Perch v2 embeddings → LightProtoSSM → distilled SED → taxonomy smoothing** — and ~888 of 4243 teams tied at exactly **0.950**. We treated that plateau as an object of study and falsified, *with measurements*, every lever available to a small team:

| Direction | Result | Why it failed |
|---|---|---|
| Own-trained SED (focal / soundscape) | 0.931 / 0.933 | Distilled copy of the public SED; capped below it |
| `compare_to_teacher` gate | student 0.91 vs public **0.99** | …but the 0.99 is **leaked** (public SED trained on the held-out files) |
| Foundation-model probes (Bird-MAE / AudioMAE) | beats public on **0/27** classes | strictly dominated → blending only dilutes |
| Rare-taxon (frog/insect) specialist | beats public on **0/16** rare classes | public SED already 0.99 on rare taxa; external data doesn't exist for them |
| Post-hoc tricks (BirdNET, co-occurrence, genus-proxy, …) | 0.948–0.950 | disturb a tightly-calibrated equilibrium |
| **Diverse-CNN SED ensemble** | **+0.021** gain (0.954 vs 0.933) | real gain, but **undeployable**: 90-min runtime can't fit the extra models |
| Ensemble two full public pipelines | blocked | **platform limits**: ~1 MB notebook cap + 90-min CPU |

**The viewpoint:** in the foundation-model era the binding constraint for a small team is not modeling skill but *platform physics* + the strength of large shared pretrained components. The gap from 0.950 to 0.96 is paid in pre-training compute, not inference-time cleverness.

---

## What's in here (our own code)

| File | What it is |
|---|---|
| [`WORKING_NOTE.md`](WORKING_NOTE.md) | The full working note (CLEF best-writeup submission). |
| `sed_modal.py` | Cloud (Modal) training + **measurement-gate harness**: `train_fold`, `compare_to_teacher`, `eval_sed_ensemble`, `eval_birdmae_probe`, `eval_rare_taxa_specialist`, ONNX export. The methodological contribution. |
| `eval_ensemble_local.py` | Local (single-GPU) ensemble-vs-single gate — proves the +0.021 diversity gain without leak. |
| `export_ensemble_local.py` | Export trained SED checkpoints to ONNX matching the public SED I/O. |
| `pseudo_label_soundscapes.py` | Noisy-student pseudo-labeling of soundscapes with the public 5-fold SED. |
| `train_sed_v2.py`, `train_sed_local.py` | Own SED trainers (focal + soundscape-native). |
| `build_*_probes.py` | Foundation-model linear-probe gates (Bird-MAE, AudioMAE, BioLingual, WavJEPA, MiMo, …). |
| `analyze_probe_orthogonality.py` | Per-class orthogonality analysis used in §4.3 of the note. |
| `src/` | SED model (timm backbone + GeM freq pool + attention temporal pooling), losses (ASL/focal/distill), augmentations. |

**Not included** (see `.gitignore`): the competition data (Kaggle's), large model weights/ONNX (regenerable), and the forked public 0.950 notebooks — those belong to their original authors (credited below) and should be obtained from the originals.

## Credits

The 0.950 public baseline is the work of the Kaggle community, not us. In particular: **Perch v2** (Google bird-vocalization-classifier); the **EoS / LightProtoSSM / distilled-SED / `TAX_SMOOTHING`** lineage (nina2025, pilkwang, tuckerarrants, hideyukizushi, chaneyma, and others). This repo's contribution is the measurement framework and the negative-results map layered on top.

## Reproducing the gates

The measurement gates run on [Modal](https://modal.com) (cloud GPU) or locally. Each costs cents and answers one hypothesis before a submission slot is spent:

```bash
# Train a diverse SED + compare against the public distilled SED on a clean hold-out
modal run sed_modal.py::train_fold --fold 0 --epochs 24 --backbone convnext_small.fb_in22k_ft_in1k --tag convnext
modal run sed_modal.py::compare_to_teacher --fold 0          # leak-aware caveat in §4.2 of the note

# Does an architecturally-diverse ensemble beat its best single member? (the +0.021 result)
modal run sed_modal.py::eval_sed_ensemble                    # or: python eval_ensemble_local.py

# Is a foundation-model probe orthogonal to the public SED? (the 0/27, 0/16 results)
modal run sed_modal.py::eval_rare_taxa_specialist --model-id audiomae
```

## License

Code released under CC0-1.0 (public domain). Competition data and upstream public models retain their own licenses.
