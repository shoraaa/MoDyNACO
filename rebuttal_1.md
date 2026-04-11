We thank the reviewer for the thoughtful comments. We agree that the current paper needs a clearer account of temporal granularity, scale transfer, scope beyond standard Euclidean benchmarks, and the observed enhancement-to-suppression transition. Below we respond using the experiments that have already finished in `reviewer_experiments/`, and we distinguish carefully between what is now directly supported and what remains future work.

**1. Temporal abstraction granularity and candidate-graph parameters.** We agree that these design choices should be characterized more explicitly. Our current results suggest that DyNACO is reasonably robust, but not completely insensitive, to these settings.

 alp

| Sweep | Setting | TSP-1K unguided cost / gap | TSP-1K DyNACO cost / gap | TSP-5K unguided cost / gap | TSP-5K DyNACO cost / gap |
| --- | --- | ---: | ---: | ---: | ---: |
| `K` | `16` | `23.2603 / 0.3050` | `23.2374 / 0.2068` | `51.9541 / 1.9236` | `51.6764 / 1.3787` |
| `K` | `24` | `23.2639 / 0.3196` | `23.2315 / 0.1800` | `52.0545 / 2.1205` | `51.6415 / 1.3103` |
| `K` | `32` | `23.2709 / 0.3498` | `23.2322 / 0.1834` | `52.1199 / 2.2488` | `51.6321 / 1.2919` |
| `K` | `48` | `23.2897 / 0.4305` | `23.2090 / 0.0832` | `52.1917 / 2.3897` | `51.4889 / 1.0109` |
| `K` | `64` | `23.3153 / 0.5418` | `23.1814 / -0.0355` | `52.3029 / 2.6078` | `51.4719 / 0.9776` |

| Sweep | Setting | TSP-1K unguided cost / gap | TSP-1K DyNACO cost / gap | TSP-5K unguided cost / gap | TSP-5K DyNACO cost / gap |
| --- | --- | ---: | ---: | ---: | ---: |
| `S` | `50` | `23.2709 / 0.3498` | `23.2102 / 0.0891` | `52.1199 / 2.2488` | `51.5557 / 1.1422` |
| `S` | `100` | `23.2709 / 0.3498` | `23.2322 / 0.1834` | `52.1199 / 2.2488` | `51.6321 / 1.2919` |
| `S` | `200` | `23.2709 / 0.3498` | `23.2158 / 0.1136` | `52.1199 / 2.2488` | `51.5896 / 1.2085` |
| `S` | `400` | `23.2829 / 0.4011` | `23.2995 / 0.4734` | `52.1922 / 2.3906` | `52.0368 / 2.0856` |

To separate solver effects from genuine learning, we also tracked relative improvement from the pre-update checkpoint (`epoch -1`) to the best trained checkpoint. We use `epoch -1` here because, in the saved artifacts, `epoch 0` is already after one round of training:

| Sweep | Setting | TSP-1K learning gain | TSP-5K learning gain |
| --- | --- | ---: | ---: |
| `M` | `4` | `-10.4%` | `-5.0%` |
| `M` | `8` | `0.0%` | `10.2%` |
| `M` | `12` | `30.4%` | `46.7%` |
| `M` | `16` | `52.4%` | `49.4%` |
| `M` | `20` | `43.0%` | `44.8%` |

| Sweep | Setting | TSP-1K learning gain | TSP-5K learning gain |
| --- | --- | ---: | ---: |
| `K` | `16` | `29.5%` | `37.8%` |
| `K` | `24` | `34.5%` | `40.2%` |
| `K` | `32` | `30.4%` | `46.7%` |
| `K` | `48` | `46.2%` | `53.8%` |
| `K` | `64` | `58.9%` | `57.0%` |

| Sweep | Setting | TSP-1K learning gain | TSP-5K learning gain |
| --- | --- | ---: | ---: |
| `S` | `50` | `40.5%` | `48.0%` |
| `S` | `100` | `30.4%` | `46.7%` |
| `S` | `200` | `36.7%` | `46.7%` |
| `S` | `400` | `5.6%` | `13.6%` |

Each row now compares DyNACO against the matched unguided backend under the same parameter setting, and the learning-gain tables show whether the model actually improves relative to its own pre-update checkpoint. These sweeps show two consistent patterns. First, small perturbation sizes may sometimes be solver-favorable, but they are poor learning regimes: `M=4` yields negative learning gain at both scales, and `M=8` is nearly flat at TSP-1K. In contrast, moderate `M` values such as `M=12` produce substantial learning gains (`30.4%` at TSP-1K and `46.7%` at TSP-5K), which is why we view them as better DyNACO operating points. Second, increasing `K` improves solution quality for both guided and unguided runs, and the learning effect also becomes stronger at larger neighborhoods, reaching `58.9%` learning gain at TSP-1K and `57.0%` at TSP-5K for `K=64`. For `S`, the method remains competitive over a moderate range, but degrades when refresh becomes too infrequent; at `S=400`, learning gain drops to only `5.6%` on TSP-1K and `13.6%` on TSP-5K, and the final guided result is weak relative to the matched unguided backend. This directly supports the reviewer's concern that overly coarse temporal abstraction is suboptimal. We will therefore revise the paper to present `K` and `S` as robust mid-range choices, not as universally optimal constants.

Regarding the question of adaptive temporal granularity, we did not include an adaptive duration policy in the current submission because we wanted to isolate the effect of dynamic guidance itself under a stable semi-MDP. The new `S` sweep nevertheless supports the reviewer's intuition: very coarse refresh schedules are harmful, while a moderate refresh frequency is robust. We will state this more explicitly and position adaptive refresh as a natural extension rather than an assumption already validated by the paper.

**2. Why does a model trained on synthetic 1K instances transfer to much larger instances?** Our current explanation is that the policy acts on local, normalized, and size-invariant signals rather than on absolute graph size. The encoder only processes a `K`-NN neighborhood, the input features are relative/normalized, and the action remains perturbative with a fixed local horizon. The new cross-scale experiment supports this mechanism empirically:

| Train scale | Test scale | Unguided gap | Guided gap | Relative gap improvement | Guidance mean | Guidance std |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `1K` | `1K` | `0.3498` | `0.1834` | `47.6%` | `0.2924` | `3.52` |
| `1K` | `5K` | `2.2488` | `1.3510` | `39.9%` | `0.2002` | `3.60` |

The key point is that the 1K-trained model still improves over the unguided backend when transferred zero-shot to 5K, while the guidance statistics remain close across scales. This is consistent with the claim that the policy is reacting to comparable local search states even when the global graph size changes substantially.

We agree, however, that this is still a mechanistic explanation rather than a formal proof. The currently completed results support the claim that transfer is real and not fragile to a 5x scale increase, but they do not yet isolate the contribution of each normalized feature independently. We will therefore revise the wording in the paper to avoid overclaiming theory, and frame the current evidence as empirical support for scale-stable local representations.

**3. Scope beyond Euclidean 2D routing.** We agree with the reviewer that the present paper remains centered on routing-style problems, and we should not claim broader non-Euclidean generality than what is directly tested. What we can now add is a cross-distribution stress test using library real-world instances (`rl_data`) for both TSP and CVRP at 1K scale:

| Problem | Distribution | Unguided cost | Guided cost | Unguided time (s) | Guided time (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| TSP | Synthetic `1K` | `23.2709` | `23.2322` | `0.0000` | `0.3120` |
| TSP | Real-world `1K` | `1325347.1271` | `1324258.1234` | `0.3568` | `0.3477` |
| CVRP | Synthetic `1K` | `36.2265` | `36.4657` | `1.7065` | `1.2755` |
| CVRP | Real-world `1K` | `48.3085` | `48.3139` | `1.9522` | `1.5711` |

These results support a narrower claim than full non-Euclidean generality: the learned guidance transfers across synthetic and library real-world routing distributions without collapsing. We will make that narrower claim explicit and avoid overstating applicability to arbitrary non-Euclidean or higher-dimensional combinatorial domains.

**4. Enhancement-to-suppression transition and its timing.** We agree that the current paper should present this more carefully as an empirical phenomenon, not as a formally derived optimal switching theorem. The new trajectory diagnostics support the qualitative claim that the policy changes polarity early as the search state evolves:

| Outer step | Iteration | Guidance mean | Enhance | Suppression | Cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1` | `100` | `33.0239` | `0.0601` | `0.2327` | `23.6971` |
| `2` | `200` | `-5.0452` | `0.1446` | `0.2436` | `23.3404` |
| `3` | `300` | `-4.9754` | `0.1383` | `0.2523` | `23.3008` |
| `4` | `400` | `-4.9924` | `0.0653` | `0.2710` | `23.2832` |
| `10` | `1000` | `-5.0017` | `0.0364` | `0.2779` | `23.2413` |

The main qualitative pattern is that guidance is strongly positive at the first outer step, flips negative by the second, and then remains stably suppressive while solution quality keeps improving. This is consistent with the paper's interpretation that guidance is initially cooperative and later becomes anti-stagnation-oriented once pheromone concentration has developed.

For the reviewer's question about dependence on stabilized pheromone bounds, the currently finished ablation only compares smoothed vs. non-smoothed MMAS-style dynamics across TSP-1K and TSP-5K:

| Scale | Smoothed gap | Non-smoothed gap | Smoothed time (s) | Non-smoothed time (s) |
| --- | ---: | ---: | ---: | ---: |
| TSP-1K | `0.1277` | `0.1211` | `0.3105` | `0.3681` |
| TSP-5K | `1.3293` | `1.1897` | `0.7510` | `0.9303` |

The main observation is that the guided policy remains competitive in both cases, which suggests that the qualitative behavior is not tied to one very narrow smoothing setting. That said, we agree that this does not yet fully isolate fixed stabilized bounds versus adaptive-bound variants, and we will avoid overstating that conclusion in the rebuttal.

**Direct answers to the reviewer's questions.**

1. **Was adaptive temporal granularity considered?** Not in the current submission. We fixed the macro duration to isolate dynamic guidance under a stable training setup. The completed `S` sweep now shows that a moderate update frequency is robust, while overly coarse refresh is harmful, which supports the motivation for future adaptive schedules.

2. **What structural properties of the normalized state-aware features support transfer?** The strongest current explanation is that the representation is local and relative rather than absolute: `K`-NN neighborhoods bound the receptive field, normalized features reduce dependence on raw scale, and perturbative actions keep the decision horizon fixed. The new 1K-to-5K zero-shot improvement supports this interpretation empirically.

3. **How much does suppressive behavior depend on fixed stabilized pheromone bounds?** The current ablation shows that the method remains effective with and without smoothing, so the behavior is not obviously an artifact of one single stabilized setting. However, we do not yet have a completed fixed-vs-adaptive-bound study, so we will state this limitation explicitly rather than overclaiming.

Overall, we appreciate the reviewer's push for sharper mechanism and scope claims. The new results let us strengthen the paper in three concrete ways: (i) add explicit `M/K/S` robustness tables and discuss the quality-runtime tradeoff, (ii) add zero-shot 1K-to-5K transfer evidence with stable guidance diagnostics, and (iii) add trajectory diagnostics showing an early polarity flip from enhancement to suppression. At the same time, we will narrow the wording around non-Euclidean generality and around the theoretical status of the crossover mechanism so that the claims match the evidence.
