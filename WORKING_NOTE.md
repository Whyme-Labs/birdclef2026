# Mapping the Public Ceiling: A Measurement-Driven Study of Why a Shared Foundation-Model Pipeline Resists Improvement in BirdCLEF+ 2026

**Whyme Labs**
*Working note — BirdCLEF+ 2026 (LifeCLEF / CLEF 2026).*
Final standing: **289 / 4243** (top 6.8%, bronze). Public LB **0.950**, Private LB **0.942**.

> Formatting note: this is the content draft. Camera-ready will be reflowed into the CEUR-WS single-column working-note template; figures/tables are referenced inline below.

---

## Abstract

Most competition working notes describe a winning recipe. This note deliberately does the opposite, and argues the inversion is useful: a systematic, measurement-driven account of *why a strong shared public baseline could not be beaten with consumer-scale resources*, and what that reveals about the structure of modern bioacoustic competitions.

By mid-competition a single architectural pattern — a frozen Perch v2 embedding model, a Light-ProtoSSM sequence head, a distilled Sound-Event-Detection (SED) CNN, and a taxonomy-smoothing post-processor — had been openly shared and converged on a public-leaderboard score of **0.950**, where roughly **890 of 4243 teams tied to three decimal places**. We treat this plateau as an object of study rather than a target. We built a cheap *measurement-gate* framework that reduces each improvement hypothesis to an offline quantity costing pennies of cloud GPU, and tests it *before* spending one of five daily submission slots. With it we falsified — with numbers, not intuition — every lever available to a small team: own-trained SEDs, foundation-model linear probes, rare-taxon specialists, post-hoc calibration tricks, and architecturally-diverse model ensembles. Two observations we found practically important and under-discussed for this setting: (1) the public components' apparent near-perfect held-out accuracy is *optimistic by construction* — they are fit on the only in-domain labelled audio available for validation — which silently inverts the most natural offline comparison and makes held-out validation an unreliable basis for ranking a challenger against the baseline; and (2) the binding constraint for a small team is not modeling skill but **platform limits** — a ~1 MB notebook-size cap and a 90-minute CPU runtime budget — which physically forbid the one technique with genuine headroom (ensembling multiple full foundation-model pipelines). We confirm that architecturally-diverse CNN ensembling yields a clean, leak-free **+0.021** macro-AUC gain in principle, yet cannot be deployed within the runtime envelope. We conclude with a viewpoint, offered as a conjecture supported by our measurements: in the foundation-model era a public pipeline assembled from large pretrained components plus a community-discovered post-processing trick is a remarkably hard floor, and the residual gap to the leaders is paid mostly in pre-training compute, not inference-time cleverness. All code and the measurement-gate harness are released openly.

---

## 1. Introduction

BirdCLEF+ 2026 asked participants to identify which of 234 species — birds, amphibians, mammals, reptiles, and insects — vocalize in 1-minute soundscapes from the Brazilian Pantanal, an important conservation-monitoring task. Submissions are CPU-only Kaggle notebooks evaluated on a hidden set of ~600 recordings, scored by macro-averaged ROC-AUC that skips classes with no true positives in the test set; the runtime budget is **90 minutes**, internet is disabled, and each team may submit five times per day.

The competition exhibited an unusually sharp *public-kernel monoculture*. A lineage of openly-shared notebooks ("Ensemble-of-Solutions", EoS.x, and their forks) raised the public ceiling in discrete, community-wide jumps — 0.928 → 0.948 → 0.949 → **0.950** — each jump driven by one new mechanism that the field adopted within days. At the close, the public leaderboard showed **11 teams ≥ 0.96**, ~42 ≥ 0.955, ~252 > 0.951, and roughly **890 teams at exactly 0.950**: the large population running the shared pipeline essentially unchanged.

We entered this monoculture late and spent the competition on one disciplined question: *can a small team, on a single consumer GPU plus a small cloud budget, add anything to the public 0.950 pipeline that survives contact with the hidden test?* Our answer is negative, but delivered with rigor — every hypothesis was reduced to a measured quantity and gated before a submission was spent. The contribution of this note is therefore not a model but a **map**: what we tried, the numbers, the structural reason each path failed, and a reusable methodology for resource-constrained competitors. We believe such negative results, reported with measurements, are scarce and valuable precisely because winners' notes leave the surrounding terrain unspoken.

## 2. Related work

BirdCLEF has run annually since 2014; recent editions converged on transfer-learning from large audio/bird foundation models followed by sequence modeling and heavy ensembling of diverse CNNs (the consistent ingredient in winning solutions across 2023–2025). Our public baseline reflects this: Perch v2 [Google bird-vocalization-classifier] as the embedding backbone, a ProtoSSM-style selective-state-space head over its embeddings, and a distilled SED CNN. Our own-model attempts draw on noisy-student self-training (pseudo-labeling unlabeled soundscapes with a teacher, then training a student) and on foundation-model linear probing (Bird-MAE, AudioMAE). Two themes we engage critically are (i) knowledge distillation's ceiling — a student trained on a teacher's outputs is bounded by that teacher — and (ii) ensemble diversity — that decorrelated architectures, averaged, beat any single member. Our contribution relative to this literature is empirical and negative: we quantify *where each of these reliably-useful techniques stops working* under this competition's specific data-leakage and platform constraints.

## 3. Anatomy of the public 0.950 pipeline

The converged public solution is a four-stage rank ensemble:

1. **Perch v2** — Google's bird-vocalization foundation model — produces a 1536-d embedding and 234-logit score per 5-second window. It is the single most expensive inference stage and the dominant signal.
2. **Light-ProtoSSM** — a small (≈0.75–5.8 M-parameter) bidirectional selective-state-space head with prototype classification and cross-window attention — is fit (or loaded pre-trained) on Perch embeddings of the labelled soundscapes, contributing ~60% of the rank blend.
3. **Distilled SED** — a 5-fold EfficientNet-B0 sound-event detector on 256×313 log-mel input, distilled from a stronger teacher ensemble, contributing ~40%.
4. **`TAX_SMOOTHING`** — the final-jump mechanism (0.949 → 0.950): each species' score is pulled slightly toward the mean of its genus- and class-siblings (α_genus ≈ 0.15, α_class ≈ 0.05), exploiting taxonomic correlation among co-occurring Pantanal taxa.

We verified directly that every public 0.950 notebook is a scalar-tuned variant of this stack: forks differing in post-processing (BirdNET sidecars, PCEN, out-of-fold gating, iterative smoothing) and even one with a different ProtoSSM implementation all returned exactly 0.950 or below.

## 4. Method: measurement gates before submission slots

The scarce resources here are not FLOPs but **submission slots** (five per day, each scored only after a multi-hour hidden-test rerun) and **trustworthy validation signal** (the labelled audio is tiny and, as §5.2 shows, optimistically biased toward the public models). Our central methodological move was to never spend a slot on an unmeasured hypothesis. For each idea we built an offline *gate* on cloud GPU (Modal) or the local GPU, costing pennies:

- **`compare_to_teacher`** — train a candidate SED with a clean *file-level* hold-out of 13 labelled soundscapes; run the public distilled SED on the *same* held-out files; report both macro-AUCs and the per-class win count.
- **`eval_*_probe` / `eval_rare_taxa`** — train linear probes on frozen foundation-model embeddings; evaluate against the public SED *sliced to specific taxa*; measure orthogonality (how many classes the candidate strictly beats the baseline on).
- **`eval_sed_ensemble`** — measure whether an ensemble's macro-AUC exceeds its best single member (a leak-free decorrelation test, independent of any public teacher).

This let us *falsify five distinct research directions for a total cloud spend under US$5*, each of which would otherwise have consumed one-to-many submission slots and a day of latency. We regard the gate methodology as a transferable contribution for any code-competition with daily quotas and slow scoring.

## 5. Systematic negative results

Table 1 consolidates the experiments. Each row was decided by a gate (offline macro-AUC, leak-aware) and/or a leaderboard submission.

**Table 1. Levers tested against the public 0.950 baseline.**

| Lever | Measured outcome | Verdict | Mechanism of failure |
|---|---|---|---|
| Own SED, focal-trained | LB 0.931 | −0.019 | distilled copy of a weaker teacher |
| Own SED, soundscape noisy-student | LB 0.933 | −0.017 | bounded by the public SED's own pseudo-labels |
| `compare_to_teacher` (clean 13-file hold-out) | student 0.910 vs public **0.994** | — | but the 0.994 is leak-optimistic (§5.2) |
| Bird-MAE / AudioMAE linear probe | beats public on **0/27** classes | dominated | no orthogonal signal to blend |
| Rare-taxon (frog/insect) specialist | beats public on **0/16** rare classes | dominated | public SED already 0.991 on rare taxa |
| Post-hoc tricks (BirdNET, co-occurrence, genus-proxy, iterative smooth) | LB 0.948–0.950 | neutral/neg. | perturb a tight calibrated equilibrium |
| **Diverse-CNN SED ensemble** (4 archs) | **0.954** vs best single **0.933** | **+0.021** | real gain — but undeployable (§6) |
| Diverse-ensemble *swapped* into pipeline | LB 0.945 | −0.005 | our members individually < public SED on hidden test |
| Diverse-ensemble *added* (7 or 9 SEDs) | runtime timeout | — | 90-min CPU budget exceeded |

### 5.1 Own-trained SED cannot match the distilled public SED

We trained EfficientNet-B0 SEDs two ways: focal-recording-only (private LB **0.931**) and a noisy-student variant on 10,592 pseudo-labelled soundscapes → 127,896 soft-labelled windows (private **0.933**). Both regressed ~0.017–0.019 below the baseline when blended in. The gate made the cause precise: on a clean 13-file hold-out the public SED scored **0.994** versus our student's **0.910**. A single own-trained CNN on one GPU cannot reach a model distilled from a strong teacher ensemble — and, because our students learn from the public SED's *own* pseudo-labels, they are its distilled copies, structurally capped below it.

### 5.2 The validation trap: optimistic held-out for public components

The gate number above (public SED 0.994 on hold-out) is **misleading**, and seeing why is, in our experience, the single most useful methodological lesson of the competition. The public Light-ProtoSSM is *provably* fit on the labelled soundscapes (it trains on them in-kernel), and the distilled SED was plausibly distilled with them in scope; either way, the public pipeline's 0.994 on a held-out subset of those same soundscapes is *optimistic by construction*, whereas our genuinely-held-out student is scored fairly. The gap between the public components' ~0.99 on labelled audio and the pipeline's 0.950 / 0.942 on the hidden test corroborates this optimism. The practical consequence is sharp: **any offline comparison of a challenger against a public component on the labelled soundscapes is biased in the baseline's favour**, so held-out macro-AUC cannot rank a small team's model against the shared pipeline. We therefore lean on *leak-free relative* gates (e.g. ensemble-vs-best-single, §5.6) wherever an absolute comparison to the baseline is untrustworthy. (Whether the precise mechanism is label leakage or in-domain over-fit plus distribution shift, the methodological conclusion is identical.)

### 5.3 Foundation-model probes are strictly dominated

Linear probes on Bird-MAE-Base and AudioMAE embeddings beat the public SED on **0/27** held-out classes with positives; every rank-blend weight monotonically *reduced* macro-AUC. A strictly-dominated source cannot improve a calibrated ensemble — it can only dilute it.

### 5.4 Rare-taxon specialists: a plausible premise, falsified

We hypothesized the bird-tuned pipeline would be weak on rare *Amphibia/Insecta* — stationary frog/insect textures unlike bird syllables — and that a texture-oriented model (AudioMAE) could win there. Sliced to 16 measurable rare classes (954 held-out positives), the gate refuted it: the public SED scored **0.991** on exactly these taxa (it had learned them from the labelled soundscapes), and our specialists beat it on **0/16** (Bird-MAE 0.876, AudioMAE 0.799 macro-AUC on the slice). External audio cannot rescue the data-starved tail either: of 53 classes with < 10 training recordings, Xeno-Canto (a bird archive) covered 1, and 28 — including the `sonXX` insect "morphotypes" — are competition-internal labels with no scientific name and thus no external data anywhere.

### 5.5 Post-hoc tricks perturb a tight equilibrium

Genus-proxy boosting, sonotype mirroring, per-taxon temperature, BirdNET 3-way blending, cross-class co-occurrence boosting, and iterative second-pass smoothing each layered onto the 0.950 base returned 0.948–0.950 (neutral-to-negative). The public pipeline's calibration chain (priors, rank-aware scaling, power optimization, threshold tuning, taxonomy smoothing) is a tight equilibrium; post-hoc reorderings disturb it more than they help. Tellingly, the community's own last +0.001 came not from *adding* a trick but from `TAX_SMOOTHING` applied *inside* the calibrated chain at the right scale.

### 5.6 Diverse-CNN ensembling: a real gain that cannot be deployed

The most reliable technique in the field is ensembling architecturally-diverse CNNs. We trained four diverse SED backbones — EfficientNetV2-S, ConvNeXt-Small, EfficientNet-B3, SEResNeXt50 — and the leak-free `eval_sed_ensemble` gate confirmed the textbook result: best single member **0.9332**, four-model ensemble **0.9542**, a **+0.021** decorrelation gain. This was the strongest positive signal of the entire effort, and it is *fair* (genuine hold-out, no leak).

It still could not break 0.950, for two independent reasons that are the crux of this note. First, *deployment runtime*: adding the four SEDs to the public pipeline (5 + 4 = 9, or even 5 + 2 = 7 inferences) exceeded the 90-minute budget and timed out. Second, *substitution cost*: swapping two public SEDs for two of ours (holding the count at 5, runtime-safe) scored **0.945** — our individually-weaker-but-diverse members lose more public strength than their diversity recovers, because on the *hidden* test (no leak) the public distilled SED is genuinely strong, not merely leak-flattered. The ensemble gain is real; the platform and the strength of the shared component jointly forbid realizing it. (The four backbones were, moreover, only half-trained when a cloud-budget cap froze them; a fully-trained ensemble would be stronger still, but the runtime wall is independent of model quality.)

## 6. The platform-constraint finding

The deeper lesson generalizes beyond SEDs. The one strategy with real headroom over a converged baseline is to **ensemble multiple full, diverse pipelines**. We attempted to combine two genuinely different public 0.950 pipelines (distinct ProtoSSM implementations and auxiliary models) in one notebook. This is blocked by *platform physics*, not modeling:

- **Notebook-size cap (~1 MB).** The two full pipelines concatenated to 1.08 MB and were rejected at upload; a single pipeline is 0.77 MB.
- **Runtime cap (90 min CPU).** Each pipeline's Perch pass nearly fills the budget; two passes time out — the same wall that killed our additive-SED ensembles.

These limits are, we argue, the structural reason for the monoculture: heavy foundation-model pipelines cannot be ensembled at inference inside the platform, so the public field is *funnelled* into a single 0.950 cluster. The teams above 0.95 escape this in one way — by **pre-training their own lightweight models** that are individually competitive yet cheap enough to run several within 90 minutes — paying the cost up front in training compute. A late, small entrant on one consumer GPU cannot buy that. We offer this as a conjecture consistent with all our measurements (the winners' methods are not public); it is the most parsimonious explanation we found for why a 0.950 floor is so hard and a 0.96 ceiling so populated by well-resourced teams.

## 7. Threats to validity

We state the limits of these conclusions plainly. (i) Our offline gates use a single fold and a small (13-file, 32-class-with-positives) hold-out; absolute numbers are noisy, which is exactly why we rely on *relative* and *orthogonality* signals rather than absolute deltas. (ii) The leak claim (§5.2) is partly inferential for the SED — we prove the ProtoSSM trains on labelled soundscapes but infer the SED's training scope from the 0.99-vs-0.95 optimism gap; the methodological conclusion holds under either leak or distribution-shift. (iii) The diverse ensemble was half-trained, so its absolute 0.954 understates its ceiling; this strengthens, not weakens, the deployment-wall argument. (iv) Our characterization of the >0.95 leaders is a conjecture — their solutions are private. (v) All scores are this competition's metric (masked macro-AUC) on its specific data; the *platform-constraint* and *validation-trap* lessons should transfer to similar code-competitions, but the exact thresholds will not.

## 8. What moved, and final standing

The only positive movement available to us was tracking the community's `TAX_SMOOTHING` jump to 0.950 and choosing robust final submissions. Recognizing the ~890-team public tie, we selected two maximally-divergent 0.950 pipelines to hedge the public→private shift. The shift proved mild and uniform (0.950 → **0.942** across our picks); a small robustness signal favoured the more battle-tested calibration (the older EoS.8 base held 0.942 while the newer EoS.9 fell to 0.941). Final standing: **289 / 4243, bronze**.

## 9. Discussion

Three takeaways we believe are non-obvious and worth recording:

1. **Optimistic held-out can invert your conclusions.** When public components are fit on the only in-domain labelled data available for validation, offline comparisons systematically flatter them. Resource-constrained teams should treat such comparisons as *lower bounds* on their own models' relative quality, and prefer leak-free relative gates.
2. **Cheap offline gates dominate blind submissions.** Reducing each hypothesis to a sub-dollar measurement before spending a scarce, slow submission slot let us explore five directions in the budget a blind approach spends on one. We recommend this as default practice under daily-quota, slow-scoring regimes.
3. **In the foundation-model era the binding constraint is often the platform, not the idea.** The highest-value technique (pipeline ensembling) and a genuine +0.021 ensemble gain were both real and both undeployable under a 1 MB / 90-minute envelope. A public ceiling built from large shared pretrained components is not a wall of cleverness to out-think; it is a floor, and the door through it is pre-training compute a late, small team does not have.

None of this diminishes the competition — it sharpens the question of where effort actually pays, and it suggests the most honest contribution a mid-pack team can make is a rigorous map of the terrain the winners' recipes leave unspoken.

## 10. Conclusion

We finished 289/4243 (bronze) at private 0.942, absorbed into the public 0.950 monoculture and unable to exceed it. The value of this note is not a score but a *negative result delivered with measurements*: a falsification of five improvement directions, identification of a held-out-validation trap, a reusable measurement-gate methodology, and a platform-constraint explanation for why public foundation-model pipelines form such hard ceilings. We release all code, the cloud training/evaluation harness, and the gates openly, in the hope the map helps the next small team decide where to spend its five daily submissions.

---

### Reproducibility & code

All model training (cloud GPU), the measurement-gate harness (`compare_to_teacher`, `eval_sed_ensemble`, the probe and rare-taxon gates), soundscape pseudo-labeling, ONNX export, and kernel-build scripts are released at: *[repository link — to be inserted]*. The public-baseline components are the work of the Kaggle community and are credited to their original authors: Perch v2 (Google); the EoS / Light-ProtoSSM / distilled-SED / `TAX_SMOOTHING` lineage (nina2025, pilkwang, tuckerarrants, hideyukizushi, chaneyma, and others). This note's own contribution is the measurement framework and the negative-results map layered on top.

### Acknowledgements

We thank the BirdCLEF+ 2026 organizers and the Kaggle community whose open notebooks constitute the baseline studied here.
