# EasyChair submission — paste-ready package

Everything CLEF 2026 / EasyChair will ask for, ready to copy. Do this in **your own browser** (you need to log in + the PDF on your disk).

> **Camera-ready status (post-review).** The paper was **accepted** (both reviews: accept). The camera-ready in `paper/main.pdf` is built with the **official `ceurart.cls`** (checked in) and addresses all reviewer requests: official CEUR-WS template, the two mandatory citations, the "measurement-gate → local cross-validation" terminology, and an explicit leak-control description (see `paper/README.md`). Upload `paper/main.pdf` as the camera-ready, plus the signed CEUR copyright form.

## Steps

1. **The final PDF is already built** with the official template: **`paper/main.pdf`**. To rebuild: `pdflatex main → bibtex main → pdflatex main → pdflatex main` (see `paper/README.md`). No Overleaf round-trip is needed — `ceurart.cls` is in the repo.
2. **Go to** https://www.easychair.org/my/conference?conf=clef2026 → log in / sign up.
3. **New submission** → choose the **LifeCLEF Lab** → tick **Task 2 – BirdCLEF+**.
4. Paste the fields below, upload the PDF, Submit.

## Fields to paste

**Title**
```
Mapping the Public Ceiling: Negative Results and a Platform-Constraint Hypothesis for BirdCLEF+ 2026
```

**Author** (add yourself as an individual if you want personal credit; otherwise team)
```
Name: Wei Meng Soh
Email: wmhy.tech@gmail.com
Affiliation: Whyme Labs (independent)
Country: Malaysia
```

**Abstract** (plain text)
```
Most competition working notes describe a winning recipe. This note deliberately does the opposite, and argues the inversion is useful: a systematic, measurement-driven account of why a strong shared public baseline could not be beaten with consumer-scale resources, and what that reveals about the structure of modern bioacoustic competitions. By mid-competition a single architectural pattern - a frozen Perch v2 embedding model, a Light-ProtoSSM sequence head, a distilled Sound-Event-Detection (SED) CNN, and a taxonomy-smoothing post-processor - had been openly shared and converged on a public-leaderboard score of 0.950, where roughly 890 of 4243 teams scored in the [0.950, 0.951) band. We treat this plateau as an object of study rather than a target. We built a measurement-gate framework that reduces each improvement hypothesis to a cheap offline quantity and tests it before spending one of five daily submission slots, and used it to test every lever available to a small team: own-trained SEDs, foundation-model linear probes, rare-taxon specialists, post-hoc calibration tricks, and architecturally-diverse model ensembles. Each was neutral or negative. Two observations we found practically consequential: (1) the public components' apparent near-perfect held-out accuracy is optimistic by construction - they are fit on the only in-domain labelled audio available for validation - which makes held-out macro-AUC an unreliable basis for ranking a challenger against the baseline; and (2) the binding constraint for a small team is not modeling skill but platform limits - a ~1 MB notebook-size cap and a 90-minute CPU runtime budget - which physically forbid the one technique with genuine headroom (ensembling multiple full foundation-model pipelines). On a single held-out split, architecturally-diverse CNN ensembling improved macro-AUC by +0.021 over its best member, yet that ensemble could not be deployed within the runtime envelope. We close with a hypothesis, offered explicitly as conjecture since the leaders' solutions are private: a public pipeline assembled from large pretrained components plus a community-discovered post-processing trick is a remarkably hard floor, and we conjecture the residual gap to the leaders is paid largely in up-front training compute rather than inference-time cleverness. All code and the measurement-gate harness are released openly.
```

**Keywords** (one per line in EasyChair)
```
bioacoustics
BirdCLEF
foundation models
negative results
ensemble diversity
knowledge distillation
data leakage
```

**PDF:** `paper/main.pdf` (official CEURART build).

## Notes
- Final standing: 289 / 4243, bronze. Public 0.950 / Private 0.942.
- The note includes an AI-workflow section + the required Generative-AI disclosure (organizers explicitly asked for both).
- After acceptance (June 24), upload camera-ready + signed CEUR copyright form by July 6.
