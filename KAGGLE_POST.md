# What we learned by *not* beating 0.950 — a negative-results map for a small team 🪶

First: this is not a winner's solution. We finished **289 / 4243 (bronze, public 0.950 / private 0.942)**, which means we landed exactly where ~890 other teams landed — squarely inside the public 0.950 tie. We didn't add a single thing on top of the shared baseline that survived the hidden test.

So why a write-up at all? Because almost every post on this forum comes from someone who *succeeded*, and the terrain that winners' recipes leave unspoken — the stuff that *doesn't* work, measured — is genuinely useful and rarely shared. We treated the public 0.950 plateau as an object to study rather than a number to chase, and we tried to be disciplined about measuring every lever before spending a submission slot. This is the map we came back with.

## Credit where it's overwhelmingly due 🙏

The 0.950 we "achieved" is the **community's** work, not ours. The pipeline we ran is the openly-shared lineage that the whole field converged on:

- **Perch v2** (Google) — the frozen bird-vocalization embedding backbone that does the heavy lifting.
- **Light-ProtoSSM** — the selective-state-space head over Perch embeddings (~60% of the rank blend).
- **Distilled SED** — the 5-fold EfficientNet-B0 sound-event detector (~40%).
- **`TAX_SMOOTHING`** — the taxonomy-smoothing post-processor that moved the field from 0.949 → 0.950.

This was built in public, in discrete community-wide jumps (0.928 → 0.948 → 0.949 → 0.950), by **nina2025, pilkwang, tuckerarrants, hideyukizushi, chaneyma, and many others** in the EoS / Light-ProtoSSM / distilled-SED / TAX_SMOOTHING lineage. Thank you. Our entire contribution sits *on top of* your floor.

## The question we actually spent the competition on

By mid-competition a single architectural pattern had been openly shared and ~890 of 4243 teams were tied in the [0.950, 0.951) band. So our question wasn't "how do we win" — it was the humbler one:

> Can a small team, on one consumer GPU plus a few dollars of cloud, add *anything* to the public 0.950 pipeline that survives the hidden test?

Answer, honestly: **no.** But we built a little "measure-before-you-submit" harness — five daily slots and multi-hour rescoring make blind submissions expensive — and reduced each idea to a cheap offline gate first. Here's what every lever returned.

## The results table

| Lever we tried | Measured outcome | Verdict |
|---|---|---|
| Own SED, focal-trained | LB **0.931** | −0.019 (distilled copy of a weaker teacher) |
| Own SED, soundscape noisy-student | LB **0.933** | −0.017 (bounded by the public SED's own pseudo-labels) |
| Bird-MAE / AudioMAE linear probe | beats public on **0/27** classes | strictly dominated, nothing to blend |
| Rare-taxon (frog/insect) specialist | beats public on **0/16** rare classes | dominated — public SED already 0.991 there |
| Post-hoc tricks (BirdNET, co-occurrence, genus-proxy, iterative smooth) | LB **0.948–0.950** | neutral-to-negative |
| **Diverse-CNN SED ensemble** (4 archs) | **0.954** vs best single **0.933** (single split) | **+0.021** held-out — but undeployable ⬇️ |
| Diverse ensemble *swapped* into pipeline | LB **0.945** | −0.005 |
| Diverse ensemble *added* (7 or 9 SEDs) | runtime **timeout** | can't fit the 90-min CPU budget |

Every original direction was neutral or negative. The one technique with a real, clean offline gain (+0.021) was the one we physically could not ship.

## The 3 things most worth sharing

**1. The held-out *validation trap*.** This one bit us early and reframed everything. The public components are fit on the labelled soundscapes — the *only* in-domain labelled audio that exists for validation. So when you compare your challenger model against a public component on a held-out slice of those same soundscapes, the comparison is **optimistically biased toward the baseline by construction**. On a clean 13-file hold-out the public SED scored **0.994**; the full pipeline scores 0.950 / 0.942 on the hidden test — that ~0.04 gap *is* the optimism. The practical fix: don't gate on absolute held-out AUC vs the baseline. Gate on **leak-aware *relative* signals** instead — ensemble-vs-best-single-member, or per-class win counts — where both sides are held out equally. (Textbook leakage pitfall, sure; the point is it silently invalidates the *most natural* validation a competitor reaches for in this comp.)

**2. The platform limits probably *cause* the monoculture.** The one strategy with genuine headroom over a converged baseline is ensembling several full, diverse pipelines. We tried to put two complete 0.950 pipelines in one notebook. It's blocked by physics, not modeling:
- **~1 MB notebook-size cap** — two full pipelines concatenated to 1.08 MB and were rejected at upload (one pipeline is 0.77 MB).
- **90-min CPU runtime cap** — a single Perch pass nearly fills the budget; two passes time out. Same wall that killed our additive-SED ensembles.

Heavy foundation-model pipelines simply can't be ensembled at inference inside the envelope. That funnels everyone who forks the heavy shared pipeline into one 0.950 cluster. It's likely *a* structural reason the field converged so hard.

**3. Diverse-CNN ensembling genuinely gains — but we couldn't deploy it.** Four diverse SED backbones (EffNetV2-S, ConvNeXt-S, EffNet-B3, SEResNeXt50), with the 13 hold-out files excluded from *all* training including pseudo-label generation, gave a leak-controlled **+0.021** over the best single member on that split. The textbook diversity effect, observed cleanly. And then it timed out, or — swapped in at constant count — dropped to 0.945 (confounded: members were only half-trained when a budget cap froze them, and blend weights weren't re-tuned). The firm conclusion is narrow but solid: *we could not realize the diversity gain within the platform envelope.*

## A guess about the leaders (clearly marked as a guess)

Their solutions are private, so this is **conjecture**, not a finding. Our most parsimonious read: teams above 0.95 escape the ensembling funnel by **pre-training their own lighter models** — individually competitive, yet cheap enough to run several within 90 minutes. They pay the cost up front in training compute that a late, small team on one consumer GPU can't match. This matches the empirical prior that BirdCLEF 2023–2025 winners consistently ran *ensembles of several independently-trained CNNs* inside the budget. But we genuinely don't know what the 11 teams ≥ 0.96 did, and we're not claiming otherwise.

## Code & the long version

Everything — cloud training/eval harness, the measurement gates (`compare_to_teacher`, `eval_sed_ensemble`, the probe and rare-taxon gates), soundscape pseudo-labeling, ONNX export, kernel-build scripts — is open:

➡️ **https://github.com/Whyme-Labs/birdclef2026**

The full CLEF working note (with the threats-to-validity section, single-split caveats, and the longer argument) is in the repo too.

## Open question for the forum

The one thing we couldn't crack: **did anyone actually fit the diversity gain inside the 90-minute CPU budget? If so, how?** Lighter pre-trained backbones, quantization, a cheaper-than-Perch embedding, smarter window subsampling? We'd love to hear what let you run several diverse models where we could only afford one. And if you found a *different* reason for the 0.950 convergence than the platform-physics one, we'd genuinely like to be wrong about it.

Thanks again to everyone who built the baseline in the open — it was a privilege to study it. 🐸🐦
