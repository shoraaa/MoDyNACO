I'll answer as a world-famous machine learning for combinatorial optimization researcher with a PhD in neural search and a Best Paper–level track record at top data mining conferences

**TL;DR**: These weaknesses are rebuttable, but not by rhetoric alone. The safest rebuttal strategy is to separate what you can defend **from current evidence** versus what would require **new experiments**. The most important additional experiments are: adaptive macro-duration ablation, scale-transfer diagnostics, fixed-vs-adaptive pheromone-bound sensitivity, and at least one non-Euclidean stress test. Below I give you a reviewer-facing rebuttal structure, plus exactly what experiments are needed for each point.

Step-by-step answer with CONCRETE details and key context, formatted for a deep reading

## Overall strategy for rebuttal

Your rebuttal should not try to “win” every point. It should do three things well:

1. **Acknowledge the limitation precisely**, without sounding defensive.
2. **State what the current paper already shows**, using the evidence you have.
3. **Identify targeted follow-up experiments**, ideally small and fast, that directly answer the reviewer’s concern.

For this set of comments, the main danger is overclaiming mechanism without direct evidence. So the tone should be:

* “We agree this is an important question.”
* “Our current evidence suggests X.”
* “We will add Y clarification / Z ablation.”

That will read much stronger than trying to sound fully certain.

---

## Point-by-point rebuttal with what needs to be experimented

## 1. Ambiguity about temporal abstraction granularity and candidate graph parameters

This is a fair criticism. Right now, your paper presents (S), (K), and related backend choices as effective, but not yet fully characterized across domains.

### What you can already say

You can say that:

* the framework does **not** require changing the policy architecture across TSP and CVRP,
* only the backend feasibility logic changes,
* the same high-level temporal abstraction works across scales from 1K to 100K,
* this suggests the framework is not brittle, even if optimal hyperparameters remain problem-dependent.

That is a reasonable defense. But it does **not** answer “what is the right (S)?” or “how sensitive is the method?”

### What experiments are needed

You need a **hyperparameter sensitivity study** centered on:

* macro-action duration (S),
* candidate graph size (K),
* perturbation size (M),
* possibly outer-step count (H) under fixed total budget.

The strongest experiment is:

### Experiment A: Fixed-budget temporal granularity sweep

Hold total inner iterations fixed, for example (H \times S = 1000), and compare:

* (H=100, S=10)
* (H=50, S=20)
* (H=20, S=50)
* (H=10, S=100)
* (H=5, S=200)

Measure:

* final gap,
* area-under-cost-curve,
* runtime,
* guidance drift between outer steps,
* pheromone entropy/convergence rate.

This directly answers whether finer or coarser abstraction is better, and whether the “right” (S) tracks pheromone convergence speed.

### Experiment B: Adaptive temporal granularity

A lightweight adaptive rule is enough for rebuttal:

* trigger guidance refresh when pheromone entropy drops by a threshold,
* or when incumbent improvement stalls for (r) iterations,
* or when mean KL change in pheromone field exceeds/falls below a threshold.

Then compare:

* best fixed (S),
* simple adaptive (S).

You do **not** need a learned option-duration policy for rebuttal. That is too ambitious. A rule-based adaptive schedule already answers the reviewer’s question.

### Experiment C: Candidate graph sensitivity

Sweep (K \in {16, 32, 64}), maybe also backup-neighbor size, and report:

* quality,
* runtime,
* transfer behavior,
* whether the dynamic policy degrades gracefully.

This is especially important because the reviewer asked about “candidate graph construction parameters,” not just temporal abstraction.

### Suggested rebuttal language

> We agree that the optimal temporal abstraction granularity and candidate-graph configuration are important design questions. In the current paper, our main claim is robustness rather than full parameter optimality: the same policy architecture and training framework transfer across TSP and CVRP and across 1K–100K scales with fixed backend settings. To further address this point, we will add a fixed-budget ablation over ((H,S)) and a sensitivity analysis over candidate size (K), as well as a simple adaptive-refresh variant based on pheromone entropy / incumbent stagnation. These experiments directly test whether guidance updates should track the solver’s convergence rate rather than remain fixed.

---

## 2. Insufficient explanation for 1K synthetic to 86x larger real-world generalization

This is also fair. You currently give a plausible explanation, but not a fully convincing one.

Right now your explanation is mainly architectural:

* local (K)-NN neighborhoods,
* normalized features,
* fixed perturbation horizon,
* scale-invariant representation.

That is a good start, but reviewers want evidence that these are the actual reasons.

### What you can already say

You already have some support:

* cross-scale zero-shot transfer table,
* real-world TSPLIB/CVRPlib results,
* normalized distance and relative pheromone features,
* local architecture independent of full graph size.

So your current logic is not empty. It is just under-validated.

### What experiments are needed

### Experiment D: Size-transfer decomposition ablation

This is the most important experiment for this point.

Take the 1K-trained model and systematically remove the hypothesized invariances:

1. replace normalized distance with raw distance,
2. replace relative pheromone features with raw pheromone,
3. remove incumbent-relative/topological features,
4. increase perturbation horizon with (N) versus keep it fixed,
5. possibly compare local (K)-NN encoder with a less local/globalized variant if feasible.

Evaluate zero-shot on 1K, 5K, 10K, and real-world instances.

If transfer collapses when normalization is removed, you have direct evidence for your explanation.

### Experiment E: Density / scale stress test

Generate matched synthetic test sets where you vary:

* number of nodes,
* spatial density,
* clustering level,
* coordinate scaling.

Then show that normalized features preserve performance under geometric rescaling and density changes.

This directly answers the reviewer’s mention of “instance sizes and node densities.”

### Experiment F: Representation diagnostics

You do not need full theory here. A strong empirical diagnostic is enough:

* measure feature distributions across scales,
* measure embedding distribution similarity across scales,
* measure whether the policy logit statistics remain stable across 1K, 10K, 100K, and TSPLIB.

For example:

* histograms of normalized distance,
* pheromone CV,
* logit mean/variance,
* top-k edge rank overlap.

If the state representation is truly scale-stable, these distributions should not drift wildly.

### Suggested rebuttal language

> We agree that our current explanation of zero-shot scale transfer is more mechanistic than formal. Our present evidence is that transfer is enabled by three size-invariant design choices: (i) local (K)-NN processing, whose receptive field does not grow with (N); (ii) normalized state features such as relative distance and relative pheromone statistics; and (iii) a perturbative action space with fixed horizon (M), which avoids the length-(N) decision horizon of constructive methods. To strengthen this point, we will add transfer ablations that remove these invariances one at a time (e.g., replacing normalized with raw features) and evaluate the 1K-trained model across larger synthetic and real-world scales.

---

## 3. Focus on Euclidean 2D problems only

This is probably the hardest weakness to fully neutralize, because it is genuinely a scope limitation.

You should not over-defend it. Instead, frame it as a deliberate first focus:

* dynamic guidance within iterative ACO,
* large-scale routing problems,
* Euclidean 2D is the standard large-scale testbed.

Then show at least one extension beyond that regime if possible.

### What experiments are needed

You need at least one of these:

### Experiment G: Non-Euclidean edge weights on the same topology

This is the fastest high-value rebuttal experiment.

Take the same node sets and replace Euclidean costs with:

* random asymmetric or symmetric metric costs,
* graph shortest-path distances,
* road-network shortest-path distances if available,
* perturbed distance matrices not derivable from raw coordinates.

Then test whether the same framework still improves the unguided ACO backend.

This is much easier than moving to a completely different COP.

### Experiment H: Higher-dimensional Euclidean routing

Run on Euclidean TSP/CVRP in 3D or higher-dimensional coordinates.

This is not as strong as a truly non-Euclidean domain, but it directly addresses “higher dimensional.”

### Experiment I: Another ACO-friendly domain

If you have bandwidth, a very small extension to:

* ATSP,
* SOP,
* Orienteering,
* Prize-collecting TSP,
* graph-based routing without Euclidean geometry

would be strong.

But for rebuttal, I would prioritize G over I, because it is more feasible.

### Suggested rebuttal language

> We agree that the current empirical scope is centered on Euclidean routing, which is both the dominant benchmark regime for large-scale TSP/CVRP and the setting in which prior neural-guided ACO work is evaluated. Our goal in this paper is to establish the core principle of dynamic guidance under iterative pheromone dynamics. That said, the framework itself does not fundamentally require 2D Euclidean structure beyond the current backend and feature construction. To probe this boundary, we will add a stress test on non-Euclidean / higher-dimensional cost structures and report whether dynamic guidance continues to improve the corresponding unguided perturbation-based backend.

---

## 4. Lack of theory for the enhancement-to-suppression crossover timing

This one should be handled carefully. I would **not** promise a full theoretical derivation unless you already have one. That is too risky.

The reviewer is asking for a theory of when the policy should switch from cooperative to suppressive behavior relative to pheromone concentration. You currently provide a plausible narrative, but not an optimal-control derivation.

### What you can already say

You can say:

* the paper presents this as an empirical phenomenon, not a claimed theorem,
* the crossover correlates with rising pheromone concentration / reduced diversity,
* trajectory-aware training allows the policy to exploit whichever mode lowers future cost.

That is defensible.

### What experiments are needed

### Experiment J: Correlate crossover with measurable search-state statistics

Track, across outer steps:

* pheromone entropy,
* coefficient of variation of pheromone,
* incumbent improvement rate,
* ant-solution diversity,
* edge-overlap concentration.

Then show that the enhancement-to-suppression crossover occurs near a reproducible threshold in one or more of these statistics.

This will not be a derivation of optimal timing, but it gives a much stronger explanatory account.

### Experiment K: Intervention experiment

Artificially manipulate the search state:

* start from uniform pheromone,
* start from partially concentrated pheromone,
* start from highly concentrated pheromone,
* perhaps inject noise or restarts.

Then observe whether the learned policy immediately changes polarity.

If it does, that strongly supports the claim that the crossover is state-dependent, not just clock-time dependent.

### Experiment L: Compare with analytically motivated schedules

Construct simple hand-designed baselines:

* always-enhancing,
* always-suppressive,
* phase-switch based on entropy threshold,
* phase-switch based on iteration count.

Then compare to learned DyNACO.

This lets you say:

* the transition is not arbitrary,
* fixed-time switching is weaker than state-conditioned switching.

### Suggested rebuttal language

> We appreciate this point. Our current paper presents the enhancement-to-suppression transition as an empirically observed behavior, rather than a formally derived optimal switching theorem. What we can show is that the crossover is tightly coupled to measurable search-state statistics such as pheromone concentration and diversity collapse, which the policy directly observes through the state-aware features. To strengthen this analysis, we will add a diagnostic correlating the crossover point with pheromone entropy / concentration and an intervention study in which we initialize the solver from states with different degrees of pheromone saturation.

---

## Direct answers to the reviewer’s specific questions

## Q1. Was adaptive temporal granularity considered?

A good answer is:

> Not in the current paper, where we intentionally use a fixed macro-action duration to isolate the effect of dynamic guidance itself and keep the semi-MDP stable. We agree that adaptive duration is a natural extension, especially because pheromone convergence speed varies across scales and problems. We will add a fixed-budget ((H,S)) sweep and a simple adaptive-refresh variant based on pheromone entropy / incumbent stagnation to evaluate whether variable-duration macro-actions provide further gains.

That is better than just saying “future work.”

## Q2. What structural properties of the normalized features enable transfer?

A good answer is:

> The key property is that the state representation is relative rather than absolute. Distances are normalized by local neighborhood scale, pheromone is represented through relative log-levels and local dispersion rather than raw magnitude, and incumbent structure is encoded through adjacency/topology indicators that do not depend on graph size. Together with a local (K)-NN encoder and a fixed-horizon perturbative action space, this makes both the input statistics and the decision horizon approximately size-invariant. We agree that this mechanism should be validated more directly, and we will add transfer ablations replacing normalized features with raw ones.

That is precise and grounded.

## Q3. Does suppressive anti-stagnation behavior depend on fixed pheromone bounds?

A good answer is:

> This is an important question. Our current ablation already shows that stabilized fixed bounds improve DyNACO relative to standard MMAS-style dynamics, suggesting that scale-stable inputs help learning. However, we agree that the present analysis does not fully isolate whether the suppressive behavior itself depends on fixed bounds or whether it would also emerge under adaptive-bound variants. We will add a sensitivity study comparing fixed stabilized bounds against traditional adaptive-bound updates and measure both performance and the learned enhancement/suppression statistics.

This is honest and strong.

---

## The exact additional experiments I would prioritize

If rebuttal time is short, do these in order:

### Highest priority

1. **Fixed-budget ((H,S)) sweep**
2. **1K-model transfer ablation with normalized vs raw features**
3. **Fixed-bound vs adaptive-bound study, including suppression statistics**
4. **Crossover-state correlation with pheromone entropy/diversity**

### Second priority

5. **Candidate graph size (K) sensitivity**
6. **Simple adaptive macro-duration rule**
7. **One non-Euclidean or higher-dimensional stress test**

If you can only afford one new domain experiment, do the **non-Euclidean cost stress test**.

---

## A compact rebuttal draft you can adapt

> We thank the reviewer for these thoughtful questions. We agree that several of these issues concern the boundary of the current paper rather than the validity of the central contribution.
>
> **On temporal granularity and candidate-graph design.** In the current paper we intentionally fix the macro-action duration and candidate-graph size to isolate the contribution of dynamic guidance and to keep the outer semi-MDP stable across scales. Empirically, the same configuration transfers across TSP/CVRP and 1K–100K scales, which suggests robustness, though not full optimality. To address this more directly, we will add a fixed-budget ablation over ((H,S)), a sensitivity analysis over (K), and a simple adaptive-refresh variant based on pheromone entropy / incumbent stagnation.
>
> **On 1K-to-86x transfer.** Our current explanation is that transfer is enabled by three scale-invariant design choices: local (K)-NN processing, normalized state features (relative distance / relative pheromone statistics), and a perturbative action space with fixed horizon (M). We agree that this mechanism should be validated more directly. We will therefore add transfer ablations that replace normalized features with raw ones and measure zero-shot degradation across larger synthetic and real-world instances.
>
> **On Euclidean scope.** We agree that the current experimental scope is centered on Euclidean routing, which is the standard regime for large-scale neural-guided TSP/CVRP. Our goal here is to establish the principle of dynamic neural guidance under iterative pheromone dynamics. To probe generality beyond this regime, we will add a non-Euclidean / higher-dimensional stress test and report whether DyNACO still improves the corresponding unguided perturbation-based backend.
>
> **On the enhancement-to-suppression crossover.** We view this as an empirical phenomenon induced by state-conditioned dynamic guidance rather than a formally derived optimal switching theorem. What we can test directly is whether the crossover aligns with measurable search-state statistics such as pheromone entropy, concentration, and diversity collapse. We will add this diagnostic, along with an intervention study starting from solver states with different levels of pheromone saturation.

---

## My blunt assessment

These are good reviewer comments. None of them is fatal. But they do expose exactly where the paper is currently weakest:

* parameter sensitivity,
* mechanism of transfer,
* scope beyond Euclidean routing,
* explanation of the phase transition.

So for rebuttal, I would avoid grand theoretical claims and instead promise **surgical empirical additions** that directly answer each concern.

The rebuttal becomes much stronger if you present these not as vague future work, but as:

* one fixed-budget ablation,
* one transfer-mechanism ablation,
* one bound-sensitivity study,
* one out-of-regime stress test.

That would make the paper feel much more complete.

I can turn this into a polished **conference-style rebuttal paragraph set** next.

