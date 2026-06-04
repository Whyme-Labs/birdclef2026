# Mapping the Public Ceiling: A Measurement-Driven Study of Why Foundation-Model Pipelines Resist Improvement in BirdCLEF+ 2026

**Whyme Labs**
BirdCLEF+ 2026 — Final rank 289 / 4243 (top 6.8%, bronze). Public LB 0.950, Private LB 0.942.

---

## Abstract

Most competition working notes describe a winning recipe. This note describes the opposite and argues that it is more useful: a systematic, measurement-driven account of *why a strong shared public baseline could not be beaten with consumer-scale resources*, and what that reveals about the structure of modern bioacoustic competitions.

By mid-competition, a single architectural pattern — a frozen Perch v2 embedding model, a LightProtoSSM sequence head, a distilled Sound Event Detection (SED) CNN, and a taxonomy-smoothing post-processor — had been openly shared and converged on a public-leaderboard score of **0.950**, where roughly **888 of 4243 teams piled up in an exact tie**. We treat this 0.950 plateau as an object of study. We built a cheap *measurement-gate* framework that tests each improvement hypothesis offline (≈ US$0.50 on cloud GPU) *before* spending one of five daily submission slots, and used it to falsify, with numbers rather than intuition, every lever available to a small team: own-trained SEDs, foundation-model linear probes, rare-taxon specialists, post-hoc calibration tricks, and architecturally-diverse model ensembles. Two findings are, to our knowledge, not previously documented for this setting: (1) the public distilled SED's apparent near-perfect held-out accuracy is an artifact of **label leakage** into the public training soundscapes, which silently invalidates the most natural offline comparison; and (2) the binding constraints on a small team are not modeling skill but **platform limits** — a ~1 MB notebook-size cap and a 90-minute CPU runtime budget — which physically forbid the one technique with genuine headroom (ensembling multiple full foundation-model pipelines). We confirm that architecturally-diverse CNN ensembling produces a clean +0.021 macro-AUC gain *in principle*, but cannot be deployed within the runtime budget. We conclude with a viewpoint: in the foundation-model era, a public pipeline assembled from large pretrained components plus a community-discovered post-processing trick is a remarkably hard floor, and the gap to the top is paid in pre-training compute, not inference-time cleverness.

---

## 1. Introduction

BirdCLEF+ 2026 asked participants to identify which of 234 species (birds, amphibians, mammals, reptiles, insects) vocalize in 1-minute soundscapes from the Brazilian Pantanal, scored by macro-averaged ROC-AUC (skipping classes absent from the hidden test). Submissions are CPU-only Kaggle notebooks with a **90-minute** runtime over ~600 hidden test recordings, no internet, and a five-submissions-per-day quota.

The competition exhibited an unusually sharp *public-kernel monoculture*. A sequence of openly-shared notebooks ("EoS.x", and forks thereof) raised the public ceiling in discrete community-wide jumps — 0.928 → 0.948 → 0.949 → **0.950** — each jump driven by a single new mechanism that everyone immediately adopted. At the final standings, **11 teams scored ≥ 0.96**, ~42 ≥ 0.955, ~252 > 0.951, and roughly **888 teams tied at exactly 0.950**: the population of participants running the shared pipeline essentially unchanged.

Our team entered this monoculture late and spent the competition asking a single disciplined question: *can a small team, on one consumer GPU plus a small cloud budget, add anything to the public 0.950 pipeline that survives contact with the hidden test?* We answer it negatively, but with rigor: every hypothesis was reduced to a measurable quantity and tested before committing a submission. The contribution of this note is the **map** — what we tested, the numbers, and the structural reasons each path failed — together with a reusable methodology for resource-constrained competitors.

## 2. Anatomy of the public 0.950 pipeline

The converged public solution is a four-stage rank ensemble:

1. **Perch v2** (Google's bird-vocalization foundation model) produces a 1536-d embedding and 234-logit score per 5-second window. This is the single most expensive stage at inference and the dominant signal.
2. **LightProtoSSM** — a small (≈0.75–5.8 M-parameter) bidirectional selective-state-space sequence head with prototype classification and cross-window attention — is trained (or loaded pre-trained) on Perch embeddings of the labeled soundscapes. It contributes ~60% of the final rank blend.
3. **Distilled SED** — a 5-fold EfficientNet-B0 sound-event-detection CNN on 256×313 log-mel input, distilled from a stronger teacher ensemble, contributing ~40%.
4. **`TAX_SMOOTHING`** — the final-jump mechanism (0.949 → 0.950): each species' probability is pulled a small amount (α_genus≈0.15, α_class≈0.05) toward the mean of its genus- and class-siblings, exploiting taxonomic correlation among co-occurring Pantanal species.

Every public 0.950 notebook is a scalar-tuned variation of this stack. We verified this directly: forks differing in post-processing (BirdNET sidecars, PCEN, OOF gating, iterative smoothing) and even a different ProtoSSM implementation all returned exactly 0.950 or below.

## 3. Methodology: measurement gates before submission slots

The scarce resource is not compute; it is **submission slots** (five per day, each scored only after a multi-hour hidden-test rerun) and **information** (the public held-out set is tiny and, as we show, leaked). Our central methodological move was to never spend a slot on a blind hypothesis. For each idea we built an offline *gate* on cloud GPU costing pennies:

- **`compare_to_teacher`** — train a candidate model with a clean file-level hold-out of 13 labeled soundscapes; run the public distilled SED on the *same* held-out files; report both macro-AUCs and the per-class win count.
- **`eval_*_probe` / `eval_rare_taxa`** — train linear probes on frozen foundation-model embeddings, evaluate against the public SED *sliced to specific taxa*, and measure orthogonality (how many classes the candidate beats the public model on).
- **`eval_sed_ensemble`** — measure whether an ensemble's macro-AUC exceeds its best single member (the decorrelation test), independent of any leaked teacher.

This framework let us *falsify five distinct research directions for a total cloud cost under US$5*, each of which would otherwise have consumed one-to-many submission slots and a day of waiting. We regard the gate methodology itself as a transferable contribution for small teams in any code-competition with scarce submissions.

## 4. Systematic negative results

### 4.1 Own-trained SED cannot match the distilled public SED

We trained EfficientNet-B0 SEDs three ways: focal-recording-only (private LB **0.931**), and a noisy-student variant on 10,592 pseudo-labeled soundscapes (private **0.933**). Both regressed ~0.017 below the 0.950 baseline when blended in. The `compare_to_teacher` gate then made the reason precise: on a clean 13-file hold-out the public SED scored **0.9937** versus our student's **0.91**. A single own-trained CNN on one GPU simply cannot reach a model distilled from a strong teacher ensemble — and, because our students were trained on the public SED's own pseudo-labels, they are *its distilled copies*, structurally capped below it.

### 4.2 The held-out leak: a measurement trap

The `compare_to_teacher` number above (public SED 0.9937 on held-out) is **misleading**, and recognizing why is, we believe, an original and practically important observation. The public distilled SED and ProtoSSM are trained *on the labeled soundscapes* — including the 13 files we "held out." Their 0.99 held-out score is therefore **leaked**; their true performance on unseen data is far lower (the full pipeline scores 0.950 on the hidden test, and 0.942 private). Our own models, genuinely held out, are scored fairly. **Any offline comparison of a candidate against a public component on the labeled soundscapes is silently biased in the public component's favor.** This single fact dissolves most "the public model is perfect everywhere" conclusions, and it is the reason held-out validation is nearly useless in this competition for ranking candidates against the shared baseline.

### 4.3 Foundation-model probes are strictly dominated

We trained linear probes on Bird-MAE-Base and AudioMAE embeddings (general-audio and bird-domain masked autoencoders) and measured orthogonality against the public SED. On the 27 held-out classes with positives, the Bird-MAE probe beat the public SED on **0/27**; every rank-blend weight monotonically *reduced* macro-AUC. There is no decorrelated signal to add — adding a strictly-dominated source can only dilute a calibrated ensemble.

### 4.4 Rare-taxon specialists: a plausible premise, falsified

We hypothesized that the bird-tuned pipeline (Perch and a bird-mel SED) would be weak on the rare *Amphibia/Insecta* classes — stationary frog and insect textures, acoustically alien to bird syllables — and that a texture-appropriate model (AudioMAE) could win there. The gate, sliced to 16 measurable rare classes (954 held-out positives), refuted it: the public SED scored **0.9912** on exactly these rare taxa (it had learned them from the labeled soundscapes), and our specialists beat it on **0/16**. We also found that external audio cannot rescue the truly data-starved classes: of 53 classes with < 10 training recordings, Xeno-Canto (a bird archive) covered only 1, and 28 of them — including the `sonXX` insect "morphotypes" — are competition-internal labels with no scientific name and therefore no external data anywhere.

### 4.5 Post-hoc tricks dilute a tight equilibrium

Genus-proxy boosting, sonotype mirroring, per-taxon temperature, BirdNET 3-way blending, cross-class co-occurrence boosting, and iterative second-pass smoothing were each layered on the 0.950 base. Every one returned 0.948–0.950 (i.e., neutral-to-negative). The public pipeline's calibration (priors, rank-aware scaling, power optimization, threshold tuning, taxonomy smoothing) is a tight equilibrium; post-hoc reorderings disturb it more than they help. The community's own +0.001 jump came not from *adding* a trick but from `TAX_SMOOTHING`, a mechanism applied *inside* the calibrated chain at the correct scale.

### 4.6 Diverse-CNN ensembling: a real gain that cannot be deployed

The single reliable technique in the field is ensembling architecturally-diverse CNNs. We trained four diverse SED backbones — EfficientNetV2-S, ConvNeXt-Small, EfficientNet-B3, SEResNeXt50 — on cloud GPU. The `eval_sed_ensemble` gate confirmed the textbook result cleanly and *without leak*: best single member **0.9332**, four-model ensemble **0.9542** — a **+0.021** decorrelation gain. This was the most promising signal of the entire effort.

It still could not break 0.950, for two independent reasons that are the crux of this note. First, *deployment runtime*: adding our four SEDs to the public pipeline (5 + 4 = 9 SED inferences) exceeded the 90-minute budget and timed out; even 7 SEDs timed out. Second, *substitution cost*: swapping two public SEDs for two of ours (holding the count at 5, runtime-safe) scored **0.945** — our individually-weaker-but-diverse models lose more public strength than their diversity recovers, because on the *hidden* test (no leak) the public distilled SED is genuinely strong. The ensemble gain is real; the platform and the strength of the shared component jointly forbid realizing it.

## 5. The platform-constraint finding

The deeper lesson generalizes beyond SEDs. The one strategy with real headroom over a converged baseline is to **ensemble multiple full, diverse pipelines**. We attempted to combine two genuinely different public 0.950 pipelines (distinct ProtoSSM implementations and auxiliary models) in a single notebook. This is blocked by *platform physics*, not modeling:

- **Notebook-size cap** (~1 MB): the two full pipelines concatenated to 1.08 MB and were rejected; a single pipeline is 0.77 MB.
- **Runtime cap** (90 min CPU): each pipeline's Perch pass nearly fills the budget; two passes time out.

These limits explain the monoculture: heavy foundation-model pipelines cannot be ensembled at inference within the platform, so the public field is *structurally* funneled into a single 0.950 cluster. The teams above 0.95 escape this exactly one way — by **pre-training their own lightweight models** that are individually competitive yet cheap enough to run several within 90 minutes. The gap from 0.950 to 0.96 is therefore paid in *up-front training compute and time*, not in a missing inference-time trick. A small team arriving late, on one consumer GPU, cannot buy that.

## 6. What actually moved, and final picks

The only positive movement available to us was tracking the community's `TAX_SMOOTHING` jump to 0.950 and selecting robust final submissions. We confirmed all five public 0.950 variants and, recognizing the ~888-team tie, selected two maximally-divergent 0.950 pipelines to hedge the public→private shift. The shift proved mild and uniform (public 0.950 → private **0.942** across our picks; interestingly the older EoS.8 base held 0.942 while the newer EoS.9 dropped to 0.941 — a small robustness signal favoring the more battle-tested calibration). Final standing: **289 / 4243, bronze**.

## 7. Discussion

Three takeaways we believe are non-obvious and worth recording:

1. **Held-out leakage can invert your conclusions.** When public components are trained on the only labeled in-domain data available for validation, offline comparisons systematically flatter them. Resource-constrained teams should treat such comparisons as lower bounds on their own models' relative quality, not as verdicts.

2. **Cheap offline gates beat blind submissions.** Reducing each hypothesis to a sub-dollar measurement before spending a scarce, slow submission slot let us explore five research directions in the time a blind approach would have spent on one. We recommend this as standard practice for code-competitions with daily quotas and multi-hour scoring.

3. **In the foundation-model era, the binding constraint is often the platform, not the idea.** The single highest-value technique (pipeline ensembling) and a genuine +0.021 ensemble gain were both real and both undeployable under a 1 MB / 90-minute envelope. The public ceiling is not a wall of cleverness to be out-thought; it is a floor set by large shared pretrained components, and the only door through it is pre-training compute that a late, small entrant does not have.

None of this diminishes the competition — it sharpens the question of where effort actually pays, and it argues that the most honest contribution a mid-pack team can make is a rigorous map of the territory that the winners' recipes leave unspoken.

## 8. Conclusion

We finished 289/4243 (bronze) at private 0.942, tied into the public 0.950 monoculture and unable to exceed it. The value of this note is not a score but a *negative result delivered with measurements*: a falsification of five improvement directions, the identification of a held-out-leak trap, a reusable measurement-gate methodology, and a platform-constraint explanation for why public foundation-model pipelines form such hard ceilings. We release all code, the cloud training/evaluation harness, and the measurement gates openly, in the hope that the map is useful to the next small team deciding where to spend its five daily submissions.

---

### Reproducibility & code

All training (cloud GPU), the measurement-gate harness (`compare_to_teacher`, `eval_sed_ensemble`, probe/rare-taxon gates), pseudo-labeling, ONNX export, and kernel-build scripts are released at: *[repository link]*. The public baseline components are credited to their original authors (Perch v2 / Google; the EoS / ProtoSSM / distilled-SED / TAX_SMOOTHING community lineage).
