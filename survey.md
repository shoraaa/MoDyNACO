## Contents
- Index Terms:
- I Introduction
- II Backgrounds and Proposed Taxonomy
  - II-A The Formulation of Vehicle Routing Problem
  - II-B Categories of Heuristics
    - II-B 1 Construction-based Methods
    - II-B 2 Improvement-based Methods
  - II-C Generation Paradigms
    - II-C 1 Autoregressive Paradigm
    - II-C 2 Non-Autoregressive Paradigm
  - II-D Major Learning Paradigms
    - II-D 1 Supervised Learning
    - II-D 2 Reinforcement Learning
  - II-E Proposed Taxonomy and Statistics
- III Construction-based Methods
  - III-A Single-Stage Methods
    - III-A 1 Appending
    - III-A 2 Insertion
  - III-B Two-Stage Methods
- IV Single-Solution-Based Methods For Improvement
  - IV-A Small Neighborhood Methods
    - IV-A 1 Immediate Search
    - IV-A 2 Sequential Search
  - IV-B Large Neighborhood Methods
    - IV-B 1 Direct LNS
      - IV-B 1a Unrestricted Direct LNS
      - IV-B 1b Restricted Direct LNS
    - IV-B 2 Indirect LNS
- V Population-Based Methods For Improvement
- VI Experimental Studies
  - VI-A Selected Methods for Comparative Evaluation
  - VI-B Experiment on Conventional Evaluation Pipeline
    - VI-B 1 Experimental Purpose
    - VI-B 2 Experimental Settings
      - VI-B 2a Problem and Instance Setting
      - VI-B 2b Metrics and Inference
    - VI-B 3 Performance Evaluation
  - VI-C Experiment on the Proposed Evaluation Pipeline
    - VI-C 1 Experimental Purpose
    - VI-C 2 Experimental Settings
      - VI-C 2a Problem and Instance Setting
      - VI-C 2b Metrics
      - VI-C 2c Inference
    - VI-C 3 Performance Evaluation
  - VI-D Discussions
    - VI-D 1 Advantages of the Proposed Evaluation Pipeline
    - VI-D 2 Principles for Method Selection
    - VI-D 3 Does Deep Learning Truly Help in NRSs?
- VII Challenges, Frontier Strategies, and Future Directions
  - VII-A In-problem Generalization
  - VII-B Cross-problem Generalization
- VIII Conclusions
- References
- Appendix A Experiment Details
  - A-A Adopted Resources
  - A-B Detailed Results of the Proposed Pipeline

## Abstract

Abstract Neural routing solvers (NRSs) that leverage deep learning to tackle vehicle routing problems have demonstrated notable potential for practical applications. By learning implicit heuristic rules from data, NRSs replace the handcrafted counterparts in classic heuristic frameworks, thereby reducing reliance on costly manual design and trial-and-error adjustments. This survey makes two main contributions: (1) The heuristic nature of NRSs is highlighted, and existing NRSs are reviewed from the perspective of heuristics. A hierarchical taxonomy based on heuristic principles is further introduced. (2) A generalization-focused evaluation pipeline is proposed to address limitations of the conventional pipeline. Comparative benchmarking of representative NRSs across both pipelines uncovers a series of previously unreported gaps in current research.

###### Index Terms:

## I Introduction

The vehicle routing problem (VRP) [25, 22] is a classic combinatorial optimization problem (COP) that seeks cost-minimizing routes for serving geographically distributed customers under specific constraints. Its scientific significance and broad practical impact have been demonstrated across various fields [128], such as transportation [24], logistics [45], and manufacturing [2]. As an NP-hard problem [70], VRPs cannot be solved to optimality in polynomial time, which has driven decades of research into heuristic algorithms to obtain high-quality approximations within acceptable computation time [69, 122]. However, designing effective heuristics requires substantial domain expertise and careful manual tuning, which poses significant challenges for real-world applications.

Efforts to automate heuristic design for combinatorial optimization have long been underway. A prominent direction is algorithm selection [113], which leverages features across problem instances to choose the most suitable algorithm for a given one. This idea has been extended to portfolio-based methods [51, 39], where a set of complementary algorithms is maintained and selectively applied to maximize performance for different instances [51, 39]. Another established paradigm is algorithm configuration, which aims to optimize algorithm performance for a target problem by automatically tuning parameters and combining modules [53, 41]. Despite their advances, these approaches remain confined to manually specified components and predefined parameter ranges, thus unable to discover or integrate novel algorithmic elements, which fundamentally limits potential performance gains.

**TABLE I: Comparisons of Existing Surveys for NRSs**
| Exisiting Surveys | Unified Perspective^† | Algorithm-Level Perspective^‡ | Description |
| --- | --- | --- | --- |
| Learning<br>Perspective | ✓ | $\times$ | • Model Structures: GNN, Transformer, etc. [126, 127, 104, 114, 123, 1];<br>• Learning Paradigms: SL, RL, etc. [127, 8, 13, 11, 56, 95];<br>• Generation Paradigms: AR, NAR, etc. [104, 56];<br>• Reliance on Learned Modules: E2E, Hybrid, etc. [8, 66, 11, 114, 21, 1]. |
| Hybrid<br>Perspective<br>(Heuristic + Others) | $\times$ | ✓ | • Residual Categories: Grouping NRSs with failed-to-identify heuristic categories as<br>“Predict” [136] / “Hybrid” [84, 118] / “Non-DRL” [146] / “Decomposition” [6];<br>• Self-Contradiction: Treating iterative GNN-based NRSs as construction-based [71, 133, 153];<br>• Scope Limitations: Only concerning RL-based [98, 134, 74, 156] or Transformer-based [3] NRSs. |
| Heuristic<br>Perspective (Ours) | ✓ | ✓ | • Hierarchical Taxonomy: Detailing how solutions are constructed / improved;<br>• Progression Identification: Tracing NRSs from traditional heuristics;<br>• Category-Specific Insights: Transferring from heuristics to NRSs with corresponding categories. |

A recent and transformative development that can address this limitation in heuristic design automation is the emergence of neural routing solvers (NRSs). NRSs leverage deep learning (DL) models to learn implicit heuristic rules from data [65], replacing their handcrafted counterparts within heuristic frameworks. Their advantages over traditional heuristics primarily lie in two aspects: (1) reducing reliance on manual design by learning from data rather than manual trial-and-error tuning [8], and (2) enabling GPU-accelerated parallel computation for problem solving.

The growing literature on NRSs has been partially reviewed in several surveys, yet a significant gap remains. Surveys organized from a learning perspective [126, 127, 8, 66, 104, 13, 11, 114, 21, 123, 1, 95, 56] typically structure the field around DL techniques related to specific components, which cannot capture algorithmic structure and behavior of NRSs. Other surveys adopting a hybrid perspective [146, 6, 84, 136, 118, 71, 133, 153, 98, 134, 74, 156, 3] often introduce secondary attributes to define heuristic categories or restrict attention to NRSs with such attributes, leading to (1) incomplete coverage, forcing a residual “others” category, (2) ambiguous classification, resulting in self-contradictory taxonomies, or (3) limited scope, omitting NRSs without chosen attributes from discussion. Overall, these surveys lack a unified algorithm-level perspective on the field. A comparative summary is provided in Table [I](https://arxiv.org/html/2602.21761v1#S1.T1).

This survey of NRSs makes two main contributions: (1) a unified algorithm-level review from the perspective of heuristics, and (2) a generalization-focused evaluation pipeline.

Unified Algorithm-Level Review NRSs are inherently heuristic algorithms powered by DL models. Building on this understanding, a hierarchical taxonomy of NRSs is proposed from the perspective of heuristics, organized by how NRSs construct or improve solutions. This perspective clarifies the relationships among NRSs and highlights their progression from traditional heuristics. Furthermore, category-specific insights from heuristics are introduced to corresponding NRSs.

Generalization-Focused Evaluation Pipeline A new evaluation pipeline is proposed to address limitations of the conventional one, and representative NRSs are benchmarked under both pipelines. The key focus of the proposed pipeline is zero-shot in-problem generalization, a critical indicator of current progress in NRSs. Results under the new evaluation pipeline reveal that many NRSs are outperformed by simple construction-based heuristics such as nearest neighbor and random insertion, indicating that the conventional evaluation pipeline tends to be overly optimistic. In addition, two major challenges in the field are discussed, *i.e.*, in-problem and cross-problem generalization, and related suggestions are provided.

The rest of this survey is organized as follows:

- •
Section [II](https://arxiv.org/html/2602.21761v1#S2) introduces essential preliminaries and details of the proposed NRS taxonomy.
- •
Section [III](https://arxiv.org/html/2602.21761v1#S3), [IV](https://arxiv.org/html/2602.21761v1#S4), and [V](https://arxiv.org/html/2602.21761v1#S5) respectively review and analyze the categories of NRSs identified in the taxonomy.
- •
Section [VI](https://arxiv.org/html/2602.21761v1#S6) proposes a new evaluation pipeline and benchmarks representative NRSs in terms of zero-shot in-problem generalization, where the results reveal previously unreported limitations in NRS development.
- •
Section [VII](https://arxiv.org/html/2602.21761v1#S7) outlines key research challenges and provides corresponding suggestions for future work.
- •
Section [VIII](https://arxiv.org/html/2602.21761v1#S8) concludes this survey.

## II Backgrounds and Proposed Taxonomy

### II-A The Formulation of Vehicle Routing Problem

This subsection introduces a three-index vehicle flow formulation [124] for the Capacitated VRP (CVRP), which is extendable to different VRP variants. The formulation is defined on a complete directed graph $\mathcal{G}=(\mathcal{V},\mathcal{A})$, where the node set $\mathcal{V}=\{0,1,\ldots,n\}$ includes a depot (node 0) and $n$ customers. Each arc $(i,j)\in\mathcal{A}$ is associated with a travel cost coefficient $c_{ij}>0$. The fleet consists of $K$ vehicles, each with a homogeneous capacity $C$. Each customer $i\in V\setminus\{0\}$ has a demand $d_{i}>0$, while the depot has $d_{0}=0$. To capture the routing decisions, two sets of binary variables are used: $x_{ij}^{k}$ indicates whether vehicle $k$ traverses arc $(i,j)$, and $y_{i}^{k}$ indicates whether customer $i$ is served by vehicle $k$. The problem formulation can be defined as follows:

$$ min $\displaystyle\sum_{i\in V}\sum_{j\in V}c_{ij}\sum_{k=1}^{K}x_{ij}^{k}$ (1) s.t. $\displaystyle\sum_{k=1}^{K}y_{i}^{k}=1\quad\forall i\in V\backslash\{0\},$ (2) $\displaystyle\sum_{k=1}^{K}y_{0}^{k}=K,$ (3) $\displaystyle\sum_{j\in V}x_{ij}^{k}=\!\sum_{j\in V}x_{ji}^{k}=\!y_{i}^{k}\quad\forall i\!\in\!V,k\!\in\!\{1,\ldots,K\},$ (4) $\displaystyle\sum_{i\in V}d_{i}y_{i}^{k}\leq C\quad\forall k\in\{1,\ldots,K\},$ (5) $\displaystyle\sum_{i\in S}\sum_{j\notin S}x_{ij}^{k}\geq y_{h}^{k}\quad\forall S\subseteq V\backslash\{0\},h\in S,k\in\{1,\ldots,K\},$ (6) $\displaystyle y_{i}^{k}\in\{0,1\}\quad\forall i\in V,k\in\{1,\ldots,K\},$ (7) $\displaystyle x_{ij}^{k}\in\{0,1\}\quad\forall i,j\in V,k\in\{1,\ldots,K\}.$ (8) $$

In this formulation, objective ([1](https://arxiv.org/html/2602.21761v1#S2.E1)) minimizes the total travel cost. Constraint ([2](https://arxiv.org/html/2602.21761v1#S2.E2)) ensures that each customer is visited exactly once. Constraint ([3](https://arxiv.org/html/2602.21761v1#S2.E3)) requires all $K$ vehicles to depart from the depot, and constraint ([4](https://arxiv.org/html/2602.21761v1#S2.E4)) enforces that a vehicle must arrive at and depart from the same customer. Constraints ([5](https://arxiv.org/html/2602.21761v1#S2.E5)) and ([6](https://arxiv.org/html/2602.21761v1#S2.E6)) impose capacity limit and route connectivity for each vehicle $k$, respectively. Finally, constraints ([7](https://arxiv.org/html/2602.21761v1#S2.E7)) and ([8](https://arxiv.org/html/2602.21761v1#S2.E8)) specify the binary nature of the decision variables. This formulation explicitly identifies vehicle-arc assignments, facilitating the incorporation of additional constraints (*e.g.*, time windows [124]) and accommodating asymmetric cases. For undirected graphs, directed arc variables $x_{ij}^{k}$ can be replaced with edge variables $x_{e}^{k}$, where $e\in E$ denotes an undirected edge.

### II-B Categories of Heuristics

Traditional heuristics for solving VRPs can be primarily classified into two categories: construction-based methods and improvement-based methods [23].

#### II-B 1 Construction-based Methods

As shown in Figure [1](https://arxiv.org/html/2602.21761v1#S2.F1), construction-based methods generate a complete solution from scratch. They can be further divided by whether solutions are generated directly on the original graph or on decomposed subgraphs. Specifically, single-stage methods [128] work on the original graph using simple strategies such as nearest neighbor, insertion [112], and sweep algorithm [36]. In contrast, two-stage methods [124] decompose the problem into different stages, typically separating customer assignment to vehicles from node sequencing within each route.

#### II-B 2 Improvement-based Methods

Improvement-based methods, also illustrated in Figure [1](https://arxiv.org/html/2602.21761v1#S2.F1), iteratively refine one or more complete solutions during the optimization process. They can be further divided based on the number of solutions involved during the search process. Specifically, single-solution-based methods [32, 128] focus on refining one solution by exploring its neighborhood with a small or large size. In contrast, population-based methods [101, 58] maintain a population of candidate solutions, and leverage collective information to guide the search towards promising regions.

Figure: Figure 1: Hierarchical structure of the heuristic taxonomy. There are two main heuristic categories: the non-iterative construction-based methods and the iterative improvement-based methods. Construction-based methods can be further divided into single-stage and two-stage methods, based on whether the original graph is decomposed. Improvement-based methods can be further split into single-solution-based and population-based methods, depending on the number of solutions maintained during the improvement process.
Refer to caption: https://arxiv.org/html/2602.21761v1/x1.png

### II-C Generation Paradigms

NRSs mainly adopt two generation paradigms to select elements (*i.e.*, nodes or edges): the Autoregressive (AR) and Non-autoregressive (NAR) approaches.

#### II-C 1 Autoregressive Paradigm

In this paradigm, nodes or edges are generated sequentially, with each new element conditioned on previous ones. This sequential dependency mimics a step-by-step decision-making process, which tends to yield high-quality solutions but at the cost of slower inference speed. In NRSs, the AR paradigm is well-suited to both construction-based methods that incrementally append or insert a node to a partial solution [65], and improvement-based methods that iteratively apply a local search move to refine a solution [137].

#### II-C 2 Non-Autoregressive Paradigm

In contrast, the NAR paradigm generates all elements concurrently in a single forward pass. This massively parallel strategy can significantly improve computational efficiency, though it may compromise solution quality due to the simplified independence assumption among elements. In NRSs, the NAR methods typically generate a probability distribution represented as a heatmap, over all candidate edges in the solution [57]. The final solution is then generated through guided stepwise edge selection, and additional refinement steps may be applied afterwards.

### II-D Major Learning Paradigms

To acquire heuristic rules from data, NRSs primarily rely on two learning paradigms: Supervised Learning (SL) and Reinforcement Learning (RL).

#### II-D 1 Supervised Learning

SL trains a model on a dataset of input-label pairs to learn the mapping [131, 88, 27]. In NRSs, the typical goal is to imitate decisions made by an expert solver, such as predicting the next node to add using a known optimal solution as the label. While this approach enables efficient learning from high-quality data, its performance is inherently bounded by the quality of the labeled data and is unlikely to surpass the expert solver it imitates.

#### II-D 2 Reinforcement Learning

RL formulates solving VRPs as a sequential decision-making process [65, 67, 151]. A solver agent learns a policy to get a high-quality solution by taking actions (*e.g.*, selecting nodes) based on a given state (*e.g.*, the current partial solution) and receiving a scalar reward (*e.g.*, the negative tour length) as feedback at the end. The objective is to learn a policy that maximizes the cumulative reward. This paradigm is well-suited for NRSs as it does not require pre-solved instances, allowing the agent to explore and potentially surpass any known strategy. Nevertheless, RL-based methods may suffer from issues such as sparse rewards and high memory overhead from storing full trajectories.

### II-E Proposed Taxonomy and Statistics

This survey proposes a taxonomy of NRSs from the perspective of heuristics. As shown in Figure [2](https://arxiv.org/html/2602.21761v1#S2.F2), this taxonomy classifies NRSs into a multi-level hierarchy based on solution construction or improvement strategies rooted in the classical heuristic taxonomy. It naturally accommodates NRSs with different generation paradigms (AR or NAR) and learning paradigms (SL or RL) across categories. Figure [2](https://arxiv.org/html/2602.21761v1#S2.F2) further reports the statistics of all 344 NRSs across hierarchical levels.

For NRSs, the heuristic taxonomy structure outlined in Section [II-B](https://arxiv.org/html/2602.21761v1#S2.SS2) admits finer distinctions within several subcategories. For example, single-stage NRSs for construction can be split into appending and insertion variants, depending on how nodes or edges are incorporated into a partial solution. Similarly, single-solution NRSs for improvement can be categorized by neighborhood size into small and large neighborhood methods, where the latter aligns with the traditional Large Neighborhood Search (LNS) heuristics. Details of different categories are provided in Section [III](https://arxiv.org/html/2602.21761v1#S3), [IV](https://arxiv.org/html/2602.21761v1#S4), and [V](https://arxiv.org/html/2602.21761v1#S5).

Figure: Figure 2: Hierarchical structure of the proposed NRS taxonomy. Each subcategory is presented with the proportion of existing studies. The statistics are obtained from Google Scholar between January 1, 2015, and November 17, 2025. The paper list is further filtered by content relevance and supplemented with relevant experience. Finally, there are a total of 439 papers, including 344 methods across various categories, as well as other related studies such as surveys and benchmarks. Note that an NRS may contribute to the counts of multiple subcategories, due to the adoption of multiple inference strategies.
Refer to caption: https://arxiv.org/html/2602.21761v1/x2.png

## III Construction-based Methods

In NRSs, construction-based methods build solutions incrementally from scratch. Similar to classical construction-based heuristics, they can be further split into single-stage and two-stage methods. The key distinction is whether the (partial) solutions are constructed directly on the original graphs or on subgraphs created by a separate decomposition stage.

Figure: (a) Appending
Refer to caption: https://arxiv.org/html/2602.21761v1/x3.png

### III-A Single-Stage Methods

Single-stage methods generate complete solutions from scratch without problem decomposition, with representative methods presented in Table [II](https://arxiv.org/html/2602.21761v1#S3.T2). Currently, most methods employ an appending strategy, sequentially adding selected elements to the end of the partial solution. However, alternative popular construction strategies in traditional heuristics, such as insertion, remain largely unexplored in NRSs. Therefore, although single-stage methods constitute the most active direction in current NRS research according to Figure [2](https://arxiv.org/html/2602.21761v1#S2.F2), their design space and potential have not yet been fully explored.

#### III-A 1 Appending

As illustrated in Figure [3](https://arxiv.org/html/2602.21761v1#S3.F3), nodes or edges are sequentially attached to the end of a partial solution in appending methods. Related inference strategies include greedy appending, sampling [7, 110], beam search [100, 57, 19], (restricted) dynamic programming (DP) [64], and Monte Carlo tree search (MCTS) based appending [143]. Given that the appending position is predetermined, the core of these methods lies in stepwise element selection. Traditional heuristics such as nearest neighbor and sweep algorithms [36] rely on greedy rules based on Cartesian distance or polar angle. Corresponding NRSs replace them with learned ones.

**TABLE II: Representative Construction-based Single-stage NRSs**
| Tertiary | Generation | Solvable | Backbone | Learning | Method | Year | Remarks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Category | Paradigm | VRPs | Paradigm |  |  |  |  |
| Appending | AR | TSP | LSTM | SL | Ptr-Net [131] | 2015 | The first NRS. |
| RL | PN-RL [7] | 2017 | The first RL-based NRS. |  |  |  |  |
| TSP, CVRP, | Transformer | RL | AM [65] | 2019 | The first NRS with the Transformer |  |  |
| OP, SDVRP, |  |  | encoder-decoder model. |  |  |  |  |
| (S)PCTSP | MDAM [141] | 2021 | A multi-decoder framework with re-embedding. |  |  |  |  |
| (A)TSP | Transformer | RL | MatNet [68] | 2021 | Matrix Encoding Network. |  |  |
| TSP, CVRP, | Transformer | RL | Sym-NCO [62] | 2022 | Training scheme with symmetricities. |  |  |
| PCTSP, OP |  |  |  |  |  |  |  |
| PDP | Transformer | RL | MAPDP [158] | 2022 | A multi-agent RL-based NRS for PDP. |  |  |
| MOTSP, MOCVRP | Transformer | RL | P-MOCO [80] | 2022 | A multi-objective NRS |  |  |
|  |  |  |  |  | with a preference-conditioned model. |  |  |
| (A)TSP, CVRP, OP | Transformer | SL | BQ [27] | 2023 | Decoder-only structure. |  |  |
| TSP, CVRP | Transformer | SL | LEHD [88] | 2023 | Light-Encoder Heavy-Decoder structure. |  |  |
| (Greedy) |  |  |  |  |  |  |  |
| SIL [89] | 2025 | Self-improved Training. |  |  |  |  |  |
| (Greedy) |  |  |  |  |  |  |  |
| RL | POMO [67] | 2020 | Parallel multiple rollouts. |  |  |  |  |
| ELG [33] | 2024 | Ensemble of local and global policies. |  |  |  |  |  |
| INViT [28] | 2024 | Distance-based search space reduction. |  |  |  |  |  |
| ICAM [151] | 2025 | Distance-biased Attention. |  |  |  |  |  |
| L2R [152] | 2025 | Learning-based search space reduction. |  |  |  |  |  |
| min-max VRPs | Transformer | RL | DPN [149] | 2024 | Decoupling tasks in the encoder for min-max VRPs. |  |  |
| (Variants of) | Transformer | RL | MTPOMO [83] | 2024 | A multi-task generalizable NRS. |  |  |
| CVRP, VRPTW | MVMoE [154] | 2024 | An MoE-based NRS for multi-attribute VRPs. |  |  |  |  |
| OVRP, VRPB | CaDA [72] | 2025 | A constraint-prompted dual-attention mechanism. |  |  |  |  |
| VRPL | ReLD [50] | 2025 | Enhancing the Light Decoder for generalization. |  |  |  |  |
| ATSP, CVRP, | Transformer | SL | GOAL [26] | 2025 | A generalist NRS with a single backbone plus |  |  |
| CVRPTW, (S)OP, | problem-specific adapters. |  |  |  |  |  |  |
| PCTSP, OVRP, |  |  |  |  |  |  |  |
| SDCVRP, TRP |  |  |  |  |  |  |  |
| NAR | TSP | GCN | SL | GCN [57] | 2019 | An NAR NRS with graph ConvNet. |  |
| AGNN | RL | DIMES [110] | 2022 | Proposing a differentiable parameterization |  |  |  |
|  |  | (Greedy) |  | of the solution space. |  |  |  |
| TSP, CVRP | GNN | GFlowNet | AGFN [147] | 2025 | A GFlowNet-based construction-based NRS. |  |  |
| Insertion | AR | TSP | GNN | RL | S2V-DQN [59] | 2017 | A GNN-based insertion NRS. |
| TSP, CVRP | Transformer | SL | L2C-Insert [90] | 2025 | An AR SL-based insertion NRS. |  |  |
|  |  |  | (Greedy) |  |  |  |  |
| NAR | TSP | U-Net | SL | DMPP [40] | 2022 | An NAR NRS with image-based diffusion models. |  |
| AGNN | SL | DIFUSCO [121] | 2023 | An NAR NRS with graph-based diffusion models. |  |  |  |

In AR appending methods, learned rules sequentially select the next node to append based on the current solution state. Consequently, this approach has driven efforts to improve the model’s ability for state representation and reasoning based on the current state. The first NRS Ptr-Net [131] employs an attention-based pointer mechanism for stepwise node selection. PN-RL [7] introduces RL into NRSs and adopts active search to fine-tune on individual test instances. AM [65] incorporates the Transformer-based encoder-decoder architecture, which improves state representation ability. Subsequently, POMO [67] extends this work by leveraging multiple trajectories with different starting nodes to enhance exploration.

**TABLE III: Representative Construction-based Two-stage NRSs**
| Role of the | Generation | Solvable | Backbone | Learning | Method | Year | Remarks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| First Stage | Paradigm | VRPs | Paradigm |  |  |  |  |
| Scale Reduction | AR | TSP | CNN, | RL | H-TSP [102] | 2023 | A two-stage NRS capable for TSP instances with |
|  |  |  | Transformer |  |  |  | 10K nodes. |
| Scale Reduction; | AR | CVRP | Transformer | RL | TAM-AM [49] | 2023 | A two-stage NRS capable for VRP instances with |
| Constraint Handling |  |  |  |  |  |  | over 5K nodes. |

To mitigate interference from irrelevant information of visited nodes during stepwise selections, some methods periodically re-embed feasible nodes [103, 140]. This alternating process of re-encoding for updated embeddings and decoding for node selection inevitably incurs substantial computational cost, yet enables more accurate state representation. A more direct alternative shifts the computational burden to a stronger decoder that performs stepwise dynamic node re-embedding. As a result, the original Heavy Encoder and Light Decoder (HELD) structure is replaced by a Light Encoder Heavy Decoder (LEHD) or even a Decoder-only structure. Methods adopting this design, such as BQ [27] and LEHD [88], demonstrate improved generalization ability. These two methods typically rely on SL to achieve high sample efficiency.

For VRPs with common distance-related objectives, the optimal next node to append is often close to the current partial solution. This observation has inspired two distinct strategies. The first is to restrict candidate selection at each step to a local neighborhood based on distance, which dramatically reduces the computational complexity and difficulty of node selection with little loss in solution quality. The second is to explicitly incorporate node-wise distance information into specific modules, thereby enhancing the model’s ability to assess the current state. Examples of the former local policy approach include [33, 135, 28, 27, 37, 152]. Particularly, ELG [33] introduces an auxiliary local policy on polar-coordinate features in addition to the regular global policy, while INViT [28] aggregates multi-scale neighborhood information through nested local views. The latter distance-enhanced modeling strategy is demonstrated in [55, 116, 73, 33, 135, 151, 50, 152]. Specifically, ICAM [151] introduces a distance-based adaptation function within the attention mechanism to better capture spatial relationships.

AR appending remains an active research area in NRSs, with a notable advantage that lies in learning a relatively simple stepwise node-selection policy. However, like traditional construction-based heuristics, these methods generate solutions from scratch, where suboptimal selections in early steps inevitably impact the quality of subsequent decisions. The solution quality can be further improved by iteratively refining solution segments using the same inference mechanism, though at the cost of increased computational overhead. Such refinement strategies fall under the restricted direct LNS subcategory, which is discussed further in Section [IV-B1b](https://arxiv.org/html/2602.21761v1#S4.SS2.SSS1.P2).

In contrast, NAR appending methods like GCN [57] and DIMES [110] select elements in a single pass guided by a predicted heatmap. While this approach enables faster inference, the static nature of heatmap cannot account for the influence of dynamic masked elements or the evolving partial solution, thereby gradually distorting the guidance information and leading to suboptimal performance. To mitigate this limitation, one potential direction is to develop an inference process that dynamically updates the heatmap during element selection. Another plausible direction is to adopt iterative refinement, thereby converting these methods into improvement-based approaches. In such cases, techniques such as population-based strategies or local search can be applied to refine the solutions (see more details in Section [IV](https://arxiv.org/html/2602.21761v1#S4) and [V](https://arxiv.org/html/2602.21761v1#S5)).

#### III-A 2 Insertion

Within single-stage methods, insertion remains a notable yet underexplored alternative to the prevalent appending paradigm. As illustrated in Figure [3](https://arxiv.org/html/2602.21761v1#S3.F3), insertion methods can place unvisited nodes into arbitrary positions of the partial solution, rather than only at the end. This flexibility introduces two coupled decisions, namely, which nodes to insert and where to insert them. While having higher time complexity, insertion can mitigate error accumulation inherent in appending by allowing corrections in subsequent steps.

Only a few studies have attempted to learn insertion policies. Among AR insertion approaches, S2V-DQN [59] selects nodes with the highest predicted values and inserts them at the minimum-cost positions for TSP. Besides, L2C-Insert [90] selects unvisited nodes via a nearest-neighbor rule and learn to determine the insertion positions. A few NAR insertion methods, such as DIFUSCO [121], incorporate greedy edge insertion guided by predefined priority scores as one inference strategy. These initial efforts, however, only scratch the surface of insertion. Future research could investigate the design of more effective joint policies that explicitly model node-position interactions, while balancing computational overhead with the opportunity to repair earlier suboptimal decisions.

### III-B Two-Stage Methods

Two-stage methods are designed to address different challenges separately in each stage. The widely used “cluster-first route-second” strategy [36, 29] first groups customers into feasible clusters based on constraints and then sequences nodes within each cluster (*i.e.*, subgraph) by solving a set of smaller TSPs. It can significantly reduce the problem scale and allows the second stage to focus on sequencing. This strategy has been adopted by a few NRSs as shown in Table [III](https://arxiv.org/html/2602.21761v1#S3.T3). For example, TAM-AM [49] partitions a large-scale VRP into clusters of small-scale TSPs in the first stage, and then applies a single-stage solver such as AM for each TSP in the second stage. Other methods, such as H-TSP [102], solely target the scaling challenge of TSPs by decomposition, which generate open-loop tours per cluster and then connect them to form a complete solution. By leveraging existing TSP solvers, these methods essentially transfer the core challenge of problem solving to the graph-partitioning step. Furthermore, related improvement-based methods with iterative redivision and refinement are discussed in Section [IV-B1b](https://arxiv.org/html/2602.21761v1#S4.SS2.SSS1.P2).

Figure: (a) Small Neighborhood
Refer to caption: https://arxiv.org/html/2602.21761v1/x5.png

## IV Single-Solution-Based Methods For Improvement

Single-solution-based methods iteratively improve a complete solution by exploring its neighborhood, which is a specific subset of feasible solutions reachable from the current solution through specific modifications.

### IV-A Small Neighborhood Methods

**TABLE IV: Representative Improvement-based Single-solution-based Small Neighborhood NRSs**
| Quaternary | Generation | Solvable | Backbone | Learning | Method | Year | Remarks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Category | Paradigm | VRPs | Paradigm |  |  |  |  |
| Immediate | AR | CVRP | LSTM | RL | NeuRewriter [15] | 2019 | Improvement with separate policies to select node-pairs. |
| TSP, CVRP | Transformer | RL | LIH [137] | 2021 | Improvement with a single policy to select node-pairs. |  |  |
| DACT [93] | 2021 | Improvement with cyclic positional encoding. |  |  |  |  |  |
| TSP | GCN, FiLM, | RL | Neural-3-OPT [119] | 2021 | Improvement with the 3-opt operator. |  |  |
|  | LSTM |  |  |  |  |  |  |
| PDP | Transformer | RL | NCS [63] | 2024 | An improvement-based NRS for PDP. |  |  |
| NAR | TSP | GNN | SL | RGLS [52] | 2022 | Predicting regret for guided local search. |  |
|  | AR | TSP, | Transformer, | RL | NeuOpt [92] | 2023 | Improvement with flexible k-opt. |
|  | CVRP | GRU |  |  |  |  |  |
|  |  | TSP, PDP, |  |  |  |  |  |
|  |  | CVRP, | SGN | SL, UL | NeuroLKH [142] | 2021 | Introducing DL to LKH. |
|  |  | CVRPTW |  |  |  |  |  |
| Sequential |  | TSP | GCRN | SL | Att-GCN [30] | 2021 | Introducing MCTS-k-opt to NAR NRSs. |
|  | NAR | AGNN | RL | DIMES [110] | 2022 | Proposing a differentiable parameterization of |  |
|  |  |  |  | (MCTS-k-opt) |  | the solution space. |  |
|  |  | AGNN | SL | DIFUSCO [121] | 2023 | An NAR NRS with graph-based diffusion models. |  |
|  |  |  |  | (MCTS-k-opt) |  |  |  |
|  |  | SAG | UL | UTSP [99] | 2023 | A NAR UL-based NRS. |  |
|  |  | / | / | SoftDist [138] | 2024 | A critique of DL-output-heatmap-MCTS-k-opt paradigm. |  |

As presented in Figure [4](https://arxiv.org/html/2602.21761v1#S3.F4), small neighborhood methods explore neighborhoods with limited sizes defined by local search operators. Based on whether moves are decomposed, they are categorized into immediate and sequential search. Immediate search relies on simple operators such as swap and 2-opt. In contrast, sequential search employs more complex operators like k-opt (k$>$2), where each move is typically decomposed into a sequence of steps to mitigate decision complexity.

#### IV-A 1 Immediate Search

There are typically two steps in each iteration of immediate search methods: (1) selecting a few nodes or edges, and (2) performing a single move via a local search operator. Learned rules in these methods primarily focus on the selection step, implemented either autoregressively, such as choosing nodes or edges with an agent, or non-autoregressively, such as generating a heatmap to guide iterative node pair or edge selection.

AR immediate search methods select moves via learned rules rather than handcrafted distance-based ones. For example, NeuRewriter [15] uses two interrelated learned rules to separately select two nodes for a local search move, while LIH [137] and DACT [93] employ a single learned rule to select node pairs. DACT additionally addresses challenges related to positional encoding. In contrast, Neural-3-OPT [119] learns separate rules to remove and reconnect three edges for each 3-opt move. NAR immediate search methods, such as RGLS [52], use heatmaps predicted by learned rules to guide the improvement process. For example, regret values can be predicted for all edges to steer the improvement process in guided local search (GLS) [35].

Benefiting from fine-grained local search operators, immediate search methods typically perform well on small-scale instances. Nevertheless, their limited neighborhood size makes them prone to local optima and less effective on large-scale problems. Therefore, the development of such methods has encountered a bottleneck in recent years.

#### IV-A 2 Sequential Search

Sequential search methods typically employ k-opt operators with k$>$2 to expand the search neighborhood for discovering better solutions. However, increasing k would lead to exponential growth in neighborhood size and, consequently, in computational complexity. A plausible strategy to address this is to decompose a k-opt move into a sequence of basic moves, which treats the improvement process as a Markov Decision Process.

There are various strategies to select a basic move at each step. To begin with, AR methods learn rules for stepwise basic move selections. For example, NeuOpt [92] dynamically adjusts k to balance coarse- and fine-grained search. Besides, NAR methods prioritize basic moves based on per-edge values in heatmaps, and can be further split by whether the heatmap is static or updated during inference. (1) NAR methods with static heatmaps typically take advanced heuristic algorithms with k-opt, such as LKH [43], as their backbone. For example, NeuroLKH [142] replaces LKH’s handcrafted edge-preference prediction rule with a learned one to determine edge candidate sets and search priorities. (2) NAR methods with dynamic heatmaps often utilize MCTS to iteratively update the heatmaps for guiding the k-opt search. In particular, Att-GCN [30] merges multiple heatmaps from small-scale subgraphs to generate the heatmap for a large-scale instance. DIMES [110] incorporates an extra meta-learning-based fine-tuning stage to improve performance. DIFUSCO [121] introduces a graph-based diffusion framework for modeling the explicit node or edge selection, while UTSP [99] eliminates the need for costly labeled datasets via unsupervised learning. Nevertheless, SoftDist [138] critically re-evaluates the heatmap-MCTS-k-opt paradigm, particularly questioning the effectiveness of DL-based heatmap generation. This finding highlights fundamental limitations of the current paradigm, underscoring the need for more principled studies.

**TABLE V: Representative Improvement-based Single-solution-based Large Neighborhood NRSs**
| Quaternary | Generation | Solvable | Backbone | Learning | Method | Year | Remarks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Category | Paradigm | VRPs | Paradigm |  |  |  |  |
|  | AR | CVRP, | Transformer | RL | NLNS [48] | 2020 | LNS with two handcrafted destroy and |
|  | SDVRP |  |  |  |  | one learned repair criteria. |  |
| Unrestricted | CVRP, | GAT, GRU | RL | EGATE [34] | 2020 | LNS with one destroy and one repair criteria |  |
| Direct LNS | CVRPTW |  |  |  |  | learned by a single model. |  |
|  | TSP, CVRP | Transformer | SL | L2C-Insert [90] | 2025 | LNS with one handcrafted destroy and |  |
|  |  |  |  | (Iteration) |  | one learned repair criteria. |  |
|  | CVRP | Transformer | RL | L2I [87] | 2020 | ILS with both small and large neighborhood search. |  |
|  | NAR | TSP | Transformer | SL | GenSCO [77] | 2025 | ILS with a generation process for local search. |
|  | AR | TSP, CVRP | Transformer | SL | LEHD [88] | 2023 | Iterative random reconstructions of partial solutions |
|  | (RRC) |  | via appending. |  |  |  |  |
|  | SIL [89] | 2025 | Iterative parallel reconstructions of partial solutions |  |  |  |  |
|  | (PRC) |  | via appending and related iterative training without labels. |  |  |  |  |
|  | DRHG [75] | 2025 | LNS with restricted ranges (outside hypernodes). |  |  |  |  |
|  | TSP, CVRP, | Transformer | RL | LCP [61] | 2021 | Iterative re-decompositions and revisions. |  |
| Restricted | PCTSP |  |  |  |  |  |  |
| Direct LNS | (A)TSP, OP, |  |  |  |  |  |  |
|  | CVRP, OVRP, | AGNN, | RL | UDC [150] | 2024 | Considering the negative impact of sub-optimal |  |
|  | (S)PCTSP, | Transformer |  |  |  | dividing policies. |  |
|  | min-max mTSP |  |  |  |  |  |  |
|  | / | CVRP(TW),<br>VRPMPD | Transformer | RL | L2D [76] | 2021 | Iterative subproblem selection and optimization. |
|  | LSTM, | RL | RBG [157] | 2022 | Iterative re-partitioning, merging, and re-solving. |  |  |
|  | Transformer |  |  |  |  |  |  |
|  | (A)TSP, CVRP, | GNN, | RL | GLOP [145] | 2024 | An NRS with both NAR and AR paradigms. |  |
|  |  | PCTSP | Transformer |  |  |  |  |
| Indirect LNS | NAR | TSP | AGNN | SL | T2T [78] | 2023 | Integrating local search in diffuse-and-denoise. |
| Fast T2T [79] | 2024 | Mapping from different noise levels to the optima. |  |  |  |  |  |

### IV-B Large Neighborhood Methods

Large neighborhood methods are grounded in the LNS heuristics [115], which explore broader solution regions to escape local optima while maintaining manageable computational complexity [108]. Corresponding NRSs learn different rules to either enhance classical LNS components, such as destroy and repair criteria for perturbation, or to automate the criterion selection. Beyond refining classical LNS, NRSs also introduce novel paradigms such as search in auxiliary latent spaces. These approaches are identified as direct LNS when searching directly on the original solution representation, and indirect LNS when conducted in an auxiliary space.

#### IV-B 1 Direct LNS

Direct LNS methods search directly on the original decision space. They can be further categorized by the flexibility of allowed modifications: (a) unrestricted direct LNS permits modifications anywhere in the solution sequence, whereas (b) restricted direct LNS limits modifications to certain predefined positions of the solution.

**TABLE VI: Representative Improvement-based Population-based NRSs**
| Search | Generation | Solvable | Backbone | Learning | Method | Year | Remarks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Space | Paradigm | VRPs | Paradigm |  |  |  |  |
| Continuous | NAR | TSP, CVRP | GRU | SL | CVAE-Opt [46] | 2021 | Latent space search with DE. |
| Transformer | RL | COMPASS [14] | 2023 | Latent space search with CMA-ES. |  |  |  |
| Discrete | NAR | (PC)TSP, (S)OP, | GNN | RL | DeepACO [144] | 2023 | Learning heuristic measures in ACO with RL. |
| CVRP(TW) | GFlowNet | GFACS [60] | 2025 | Learning heuristic measures in ACO with GFlowNet. |  |  |  |

##### IV-B 1a Unrestricted Direct LNS

Unrestricted Direct LNS methods are typically built upon the classic destroy-and-repair paradigm of the LNS heuristic, where neighborhoods are implicitly defined by the destroy and repair criteria. In each iteration, the destroy step removes multiple nodes from the complete solution, and the repair step reinserts them sequentially back into the solution for potential improvement. This approach offers two key advantages: (1) computational scalability [35], because the number of nodes removed and reinserted (*i.e.*, the perturbation strength) is independent of instance size; and (2) solution quality [94], as even a small set of nodes, when destroyed and repaired under effective criteria, can lead to promising improvement.

Current related NRSs typically focus on learning effective destroy or repair criteria. For example, NLNS [48] incorporates two handcrafted destroy criteria and a learned repair criterion, while EGATE [34] employs a single learned rule to both select nodes for removal and determine reinsertion sequences. When applied iteratively, L2C-Insert  [90] can also be regarded as an LNS variant when its learned insertion rule is treated as the repair criterion, complemented by a handcrafted destroy step. The Iterated Local Search (ILS) heuristic [117] further extends LNS by interleaving between large neighborhood perturbation to escape the local region and fine-grained local search to refine the solution. In particular, L2I [87] integrates DL into ILS to select both local search operators and destroy or repair criteria. GenSCO [77] perturbs solutions via successive 2-opt moves, as commonly used in heuristics [105], and then refines them using a rectified flow model.

The effectiveness of LNS heuristics relies not only on well-designed destroy and repair criteria, but also on rules for controlling the perturbation strength, adapting the criteria, determining the insertion orders, and designing more complex acceptance criteria [86, 12, 107, 35]. However, the current NRSs have focused predominantly on learning destroy and repair criteria, leaving other critical rules still largely handcrafted. Therefore, a key future research direction is to automate the design of these rules and to thoroughly investigate their interactions. This holistic design principle is crucial for advancing both this subcategory and NRSs more broadly.

##### IV-B 1b Restricted Direct LNS

After the destroy step, typical LNS heuristics encounter scenarios involving partial solutions and unvisited nodes, identical to those faced in construction-based methods. Recent studies have therefore drawn inspiration from single-stage and two-stage construction-based methods to develop new iterative approaches adhering to LNS principles. Some of them iteratively reconstruct partial solutions with single-stage strategies. In contrast, others adopt iterative versions of two-stage methods, which repeatedly partition the problem and solve subproblems with existing NRSs or heuristics. Though not explicitly framed in classical heuristics, these paradigms can be regarded as position-restricted destroy-and-repair and thus a subcategory of LNS methods.

For the extensions of single-stage methods, the appending LEHD [88] can use a flexible Random Re-Construct (RRC) approach to refine a sampled partial solution at each iteration. From the perspective of LNS, RRC, and its parallel version Parallel local Re-Construction (PRC), destroy random node sequences and adopt the learned appending rule as the repair criterion. Integrating this iterative improvement approach into SL training can further reduce the reliance on high-quality solutions [88, 89, 106] and even enable direct training on large-scale instances [89]. In addition, DRHG [75] treats partial solutions as hypernodes, which are sequentially appended with unconnected nodes during the repair process.

Instead of designing more powerful subsolvers, the extensions of two-stage methods, such as the so-called “hierarchical search” [61, 157], “divide-and-conquer approach” [17, 145, 150], and “learning-augmented local search” [76], focus on developing appropriate strategies that leverage existing NRSs or heuristics to achieve better overall performance. For example, LCP [61] employs a seeder policy to generate candidate solutions, which are then optimized in parallel by a reviser that iteratively decomposes and reconstructs them. RBG [157] decomposes a complete solution into non-overlapping regions, each containing several routes. This division is iteratively updated by a learned rewriter that selects regions to split or merge, after which a generator then generates the routes for the updated regions. Both GLOP [145] and UDC [150] initially generate a heatmap for decomposition. GLOP partitions the original problem once into sub-TSPs, in which divide-and-conquer steps are further applied. In contrast, UDC employs iterative subproblem re-divisions, and the subproblems are not limited to TSP.

These methods have gained popularity owing to their ability to extend existing construction-based NRSs through iterative refinement. However, like two-stage construction-based methods, the subsolvers generally lack global information, which potentially leads to premature convergence. Moreover, they are often presented merely as extensions of construction-based NRSs, without explicitly acknowledging their LNS nature, leading to insufficient attention to holistic algorithm design. A systematic analysis of LNS heuristics could inspire more principled designs. Promising future research directions include dynamically controlling subproblem sizes, similar to adaptive perturbation degree control in LNS, to balance exploration and exploitation, and identifying suboptimal partial solutions for further improvement while preserving promising ones.

#### IV-B 2 Indirect LNS

LNS methods can be generalized to operate in an auxiliary space rather than the original decision space. The auxiliary space is often continuous, enabling gradient-based methods to guide the search. Moreover, operations performed in this space can simultaneously modify multiple parts of a solution, bypassing the sequential node-by-node selection and positioning required in the original decision space. It enables more extensive solution adjustment compared to classical destroy-and-repair perturbations, which typically modify only a small number of edges.

A typical example is to use diffusion models for solving TSPs in an NAR manner [78, 79]. During inference, the forward noising process gradually increases the confidence of extra edges, which turns a feasible solution into an infeasible one with more edges. Conversely, the reverse denoising process decreases the confidence of redundant edges to recover a feasible solution. The stochastic nature of diffusion allows applying an iterative noising-denoising process to produce diverse solutions. Incorporating effective guidance, such as the gradient feedback in T2T for denosing [78], can further improve the solution quality. A subsequent work, Fast T2T [79], further accelerates denoising via consistency modeling.

Although indirect LNS methods differ from classical LNS, heuristics can still offer valuable insights. Current implementations typically employ a fixed noise schedule during inference. However, as stated earlier, adaptive perturbation strength is crucial for balancing fine-grained search and escape from local optima [35]. Therefore, adaptively adjusting re-noising levels based on search progress could be helpful. Additionally, the greedy decoding is often suboptimal, and more well-designed inference strategies deserve greater attention as in other NAR NRSs. Finally, heatmap-guided search is not the only possible paradigm for indirect LNS. Further work is expected to explore alternative auxiliary search spaces.

## V Population-Based Methods For Improvement

Population-based methods maintain and evolve a set of candidate solutions, leveraging collective information from the entire set to guide search [129, 58, 5]. In NRSs, these methods can be implemented either by operating directly on the discrete solution space of the original problem, or by transforming solutions into a continuous latent space for optimization, as illustrated in Table [VI](https://arxiv.org/html/2602.21761v1#S4.T6).

For methods that work in the discrete solution space, DeepACO [144] and GFACS [60] enhance the classic Ant Colony Optimization (ACO) by replacing handcrafted heuristic measures for edges (*e.g.*, inverting the length) with learned scoring rules. Unlike various NAR construction-based NRSs confined to TSP and Maximum Independent Set (MIS), these approaches inherit the flexibility of meta-heuristics, which can tackle a broader range of COPs.

For methods that work in continuous latent space, CVAE-Opt [46] utilizes a Variational Autoencoder (VAE) model to learn the distribution of high-quality solutions, then evolves a population in the latent space via differential evolution (DE) [109]. Besides, COMPASS [14] parametrizes a continuous policy distribution and applies Covariance Matrix Adaptation Evolution Strategy (CMA-ES) [42] to search.

Like classic domain-agnostic meta-heuristics, population-based methods exhibit inherent robustness for problems with complex search spaces. A promising future direction is to adapt them to problems with dynamic environments, where traditional population-based heuristics have demonstrated strong suitability [35]. In addition, given the successful DL-based enhancement of the single-solution-based LKH [43] (as discussed in Section [IV-A2](https://arxiv.org/html/2602.21761v1#S4.SS1.SSS2)), powerful population-based algorithms like HGS [129] could likewise be integrated with DL techniques to develop more competitive NRSs.

## VI Experimental Studies

This section investigates the in-problem performance of representative NRSs, with a focus on their zero-shot generalization ability, a topic of significant interest in recent years. The conventional evaluation pipeline is first applied, which emphasizes scalability on synthetic instances and yields promising results. Nevertheless, this pipeline suffers from notable limitations, including a narrow range of test distributions, conflated in- and out-of-distribution comparisons, and inconsistent inference settings. Therefore, a generalization-focused evaluation pipeline is introduced for single-model performance across diverse benchmark instances, with unified inference and complementary metrics. Experimental results under this new pipeline reveal that NRSs trained on narrowly distributed data may be outperformed by even simple construction heuristics such as nearest neighbor and random insertion. This contrast suggests that the conventional pipeline can systematically lead to overly optimistic conclusions. Building on these findings, the advantages of the proposed pipeline are discussed, and principles for method selection are outlined. In particular, learning is argued to remain crucial for NRSs, even when their performance falls short of prior expectations. The implementation details of the experimental studies are available in [https://github.com/CIAM-Group/NRS_Survey](https://github.com/CIAM-Group/NRS_Survey).

### VI-A Selected Methods for Comparative Evaluation

The comparative evaluation incorporates two groups of methods: classical and SOTA heuristics that serve as baselines, and representative NRSs. The selected heuristics, chosen for their efficiency or effectiveness, are briefly introduced below.

- •
Nearest Neighbor A classic construction-based heuristic. At each step, the nearest node to the last node of the partial solution is selected for appending.
- •
Random Insertion A classic construction-based heuristic. At each step, a randomly selected node is inserted at the position that minimizes the increase in cost.
- •
LKH-3 [44]  A single-solution-based SOTA heuristic for TSP, widely adopted as a baseline in prior works.
- •
HGS [130]  A population-based SOTA heuristic for CVRP, widely adopted as a baseline in prior works.
- •
AILS-II [96]  A single-solution-based SOTA heuristic for CVRP, rarely adopted as a baseline in prior works.

The selected NRSs comprehensively cover all categories in the proposed taxonomy and are listed in Table [VII](https://arxiv.org/html/2602.21761v1#S6.T7). All inference experiments of NRSs are uniformly conducted on a single NVIDIA GeForce RTX 3090 GPU with 24GB of memory. Specifically, 20 cores of the Intel(R) Xeon(R) Gold 6348 CPU @ 2.60GHz and 40 GB of memory are allocated to each NAR NRS (GFACS, GenSCO, and Fast T2T) for potential calculations on the CPU.

**TABLE VII: Selected NRSs for Comparative Evaluation**
| Category | Method |  |  |  |
| --- | --- | --- | --- | --- |
| Primary | Secondary | Tertiary | Quaternary |  |
| Construction | Single-stage | Appending | / | BQ [27] |
| LEHD^† [88] |  |  |  |  |
| SIL^† [89] |  |  |  |  |
| ICAM [151] |  |  |  |  |
| ELG [33] |  |  |  |  |
| INViT [28] |  |  |  |  |
| L2R [152] |  |  |  |  |
| DGL [139] |  |  |  |  |
| ReLD [50] |  |  |  |  |
| Insertion | / | L2C-Insert^† [90] |  |  |
| Two-stage | / | / | H-TSP [102] |  |
| Improvement | Single-solution | Small | Immediate | DACT [93] |
| Neighborhood | Sequential | NeuOpt [92] |  |  |
|  | Unrestricted | L2C-Insert^‡ [90] |  |  |
|  | Direct LNS | GenSCO [77] |  |  |
| Large | Restricted<br>Direct LNS | LEHD^‡ [88] |  |  |
| Neighborhood | SIL^‡ [89] |  |  |  |
|  | DRHG [75] |  |  |  |
|  | Indirect LNS | Fast T2T [79] |  |  |
| Population | / | / | GFACS [60] |  |

### VI-B Experiment on Conventional Evaluation Pipeline

**TABLE VIII: Experimental Results of Conventional Evaluation Pipeline**
| Method | TSP 100 | TSP 1K | TSP 10K |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Gap | Time | Gap | Time | Gap | Time |  |
| LKH-3 | 0.000% | 10.97m | 0.000% | 5.69m | 0.000% | 49.34m |
| Nearest Neighbor | 24.722% | 6.72s | 25.022% | 0.97s | 23.864% | 2.29s |
| Random Insertion | 9.672% | 2.04s | 13.096% | 0.46s | 13.966% | 4.64s |
| ^↑BQ greedy | 0.348% | 1.13m | 2.294% | 1.19m | / | / |
| ^↑LEHD^∗ greedy | 0.576% | 26.84s | 3.116% | 1.64m | / | / |
| ^∥SIL^∗ greedy | / | / | 1.952% | 29.12s | 4.061% | 6.06m |
| ^↑ICAM aug $\times 8$ | 0.147% | 44.66s | 1.647% | 3.93m | / | / |
| ^↑ELG aug $\times 8$ | 0.224% | 3.02m | / | / | / | / |
| ^↑INViT-3V aug^† | 1.419% | 32.44m | 5.154% | 5.52m | 6.678% | 1.27h |
| ^↑L2R greedy | / | / | 4.494% | 6.48s | 4.824% | 1.07m |
| ^↑DGL aug^† | 0.609% | 14.16m | 2.714% | 1.42m | 6.792% | 10.61m |
| ^↑L2C-Insert^∗ greedy | 0.458% | 1.24m | 4.756% | 32.98s | 7.760% | 1.11m |
| ^∥H-TSP | / | / | 6.673% | 46.59s | 8.329% | 50.92s |
| ^∥DACT T=10K | 0.379% | 2.05h | / | / | / | / |
| ^∥NeuOpt T=10K | 0.018% | 1.47h | / | / | / | / |
| ^↑L2C-Insert^∗ T=1K | 0.0001% | 12.01h | 0.485% | 1.21h | 2.086% | 15.86m |
| ^∥GenSCO 2-opt | 0.0003% | 1.96m | 0.033% | 6.76m | / | / |
| ^↑LEHD^∗ RRC1K | 0.002% | 2.36h | 0.729% | 7.49h | / | / |
| ^∥SIL^∗ PRC1K | / | / | 0.375% | 3.47h | 1.824% | 5.19h |
| ^↑DRHG T=1K | 0.0003% | 7.12h | 0.420% | 3.89h | 1.802% | 1.05h |
| ^∥Fast T2T^‡ Ts=5, Tg=5 | 0.030% | 37.29m | 0.589% | 9.03m | / | / |
| ^∥GFACS T=10, K=100 | / | / | 2.615% | 3.14h | / | / |

#### VI-B 1 Experimental Purpose

This pipeline generally evaluates NRSs on synthetic instances with specific scales, node distributions, and optional constraint tightness [91]. Among these aspects, scalability is the most widely studied one and is also the primary focus of this experiment. It is important to note, however, that scalability is not equivalent to generalization, which will be discussed in detail in Section [VI-D1](https://arxiv.org/html/2602.21761v1#S6.SS4.SSS1).

#### VI-B 2 Experimental Settings

##### VI-B 2a Problem and Instance Setting

TSP unrelated to constraint tightness is considered due to the lack of a unified setting in the literature. For scale and node distribution, the evaluation follows common practice by testing on uniformly distributed instances at scales of 100, 1K, and 10K. All instances are drawn from the generated datasets of SIL [89].

##### VI-B 2b Metrics and Inference

Two metrics are reported for each method: the optimality gap (Gap) and the total inference time (Time). Specifically, the optimality gap measures the discrepancy between the obtained solutions and the best-known solutions, provided by the LKH-3 heuristic, as is common practice in AM [65]. For NRSs, the released implementations and pretrained models are adopted. Note that each NRS is evaluated only under a specific configuration on instances with corresponding sizes reported in the original studies. Results for unreported conditions are denoted by “/”.

#### VI-B 3 Performance Evaluation

According to Table [VIII](https://arxiv.org/html/2602.21761v1#S6.T8), NRSs exhibit promising performance under the conventional pipeline. For construction-based NRSs, all of them outperform simple heuristics (nearest neighbor and random insertion) within their respective categories. Specifically, ICAM achieves strong in- and out-of-distribution results. Besides, L2R maintains competitive performance while reducing inference time by approximately an order of magnitude. For large-scale instances with 10K nodes, where only a few construction-based methods are evaluated, SIL (Greedy) delivers the best performance. In contrast, H-TSP generally underperforms single-stage counterparts, falling short of the expected two-stage advantages on larger instances. For improvement-based NRSs, most of them achieve competitive performance close to that of the advanced heuristic LKH-3. For example, GenSCO with 2-opt achieves strong in-distribution results within a short runtime. In addition, among the limited NRSs tested at 10K, DRHG performs best, achieving slightly better performance than SIL (PRC) while using only about one-fifth of its inference time. Nevertheless, GFACS is outperformed by several construction-based methods (BQ, SIL (Greedy), and ICAM) at the scale of 1K.

### VI-C Experiment on the Proposed Evaluation Pipeline

#### VI-C 1 Experimental Purpose

The conventional evaluation pipeline has several limitations. First, its testing distributions are limited in scope, typically restricted to specific scales and node distributions [155]. This restricted coverage poorly represents real-world scenarios. Moreover, the parameterized synthetic instance generators can bias performance toward certain training distributions. Second, it does not distinguish the evaluation of single-model generalization performance (*i.e.*, one model applied to all test instances) and multi-model in-distribution performance (*i.e.*, separate models trained and tested per problem scale). Finally, the inference settings are typically inconsistent across different methods. In short, this pipeline inherently favors NRSs whose DL models overfit to the training distribution and report multi-model in-distribution performance, therefore introducing systematic evaluation bias.

To address these issues, a new evaluation pipeline is introduced. It centers on the zero-shot in-problem generalization, which has been the primary focus of advanced NRSs in recent years and therefore serves as a representative indicator of progress in the field. Under this pipeline, NRSs are benchmarked on diverse instances that more faithfully reflect the irregular conditions of real-world production and logistics. Distributional biases inherent in synthetically generated instances, particularly those with uniformly distributed nodes, are avoided. In addition, the inference settings are consistently standardized across all evaluated NRSs.

**TABLE IX: Experimental Results of the Proposed Evaluation Pipeline**
|  | Method | (0,1K) | [1K, 10K) | [10K, 100K] | Total |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Gap | Time | Solved | Gap | Time | Solved | Gap | Time | Solved | Gap | Solved |  |
| TSP | Nearest Neighbor | 25.29% | 0.01s | 69/69 | 26.66% | 0.29s | 109/109 | 25.01% | 22.60s | 50/50 | 25.88% | 228/228 |
| Random Insertion | 10.60% | 0.00s | 69/69 | 15.32% | 0.05s | 109/109 | 16.37% | 8.93s | 50/50 | 14.12% | 228/228 |  |
| LKH-3^↓ t=n/3, runs=1 | 0.00% | 7.88s | 69/69 | 0.01% | 631.34s | 109/109 | 0.08% | 14600.50s | 50/50 | 0.03% | 228/228 |  |
| LKH-3^↓ t=n/3, runs=1 | 0.00% | 9.25s | 69/69 | 0.01% | 600.36s | 109/109 | 0.05% | 10800.24s | 50/50 | 0.02% | 228/228 |  |
| BQ | 5.00% | 2.51s | 68/69 | 19.03% | 22.74s | 92/109 | 52.00% | 187.81s | 4/50 | 14.02% | 164/228 |  |
| LEHD^∗ greedy | 4.85% | 1.01s | 69/69 | 20.13% | 68.27s | 106/109 | 49.35% | 1386.01s | 11/50 | 16.19% | 186/228 |  |
| SIL^∗ greedy | 8.64% | 1.69s | 69/69 | 9.83% | 17.69s | 109/109 | 11.11% | 430.73s | 50/50 | 9.75% | 228/228 |  |
| ICAM | 6.53% | 0.25s | 69/69 | 16.62% | 21.57s | 109/109 | 21.34% | 1050.33s | 19/50 | 13.54% | 197/228 |  |
| ELG | 6.05% | 0.63s | 69/69 | 18.14% | 88.12s | 108/109 | 21.65% | 940.35s | 6/50 | 13.70% | 183/228 |  |
| INViT-3V | 7.93% | 2.77s | 69/69 | 12.08% | 49.03s | 109/109 | 11.52% | 1079.50s | 42/50 | 10.67% | 220/228 |  |
| L2R | 5.89% | 1.60s | 69/69 | 9.22% | 15.55s | 109/109 | 8.52% | 153.11s | 50/50 | 8.06% | 228/228 |  |
| DGL | 6.53% | 1.17s | 69/69 | 11.32% | 11.67s | 109/109 | 11.14% | 58.62s | 25/50 | 9.67% | 203/228 |  |
| L2C-Insert^∗ greedy | 4.39% | 1.51s | 69/69 | 18.12% | 15.34s | 109/109 | 30.94% | 145.17s | 50/50 | 16.77% | 228/228 |  |
| H-TSP | 6.16% | 0.61s | 36/69 | 11.62% | 3.15s | 100/109 | 12.29% | 21.44s | 40/50 | 10.65% | 176/228 |  |
| DACT T=1K | 16.37% | 39.84s | 69/69 | 26.58% | 261.73s | 83/109 | / | / | 0/50 | 21.94% | 152/228 |  |
| NeuOpt T=1K | 19.90% | 81.22s | 46/69 | / | / | 0/109 | / | / | 0/50 | 19.90% | 46/228 |  |
| L2C-Insert^∗ T=1K | 1.08% | 381.55s | 69/69 | 9.80% | 479.98s | 109/109 | 29.15% | 615.19s | 50/50 | 11.41% | 228/228 |  |
| GenSCO | 14.56% | 23.19s | 68/69 | 35.46% | 677.31s | 104/109 | 35.17% | 14304.70s | 25/50 | 28.21% | 197/228 |  |
| LEHD^∗ RRC1K | 1.73% | 498.40s | 69/69 | 10.87% | 1634.51s | 109/109 | 24.02% | 2769.86s | 9/50 | 8.13% | 187/228 |  |
| SIL^∗ PRC1K | 0.80% | 883.87s | 69/69 | 2.58% | 2880.46s | 109/109 | 4.55% | 4933.45s | 50/50 | 2.47% | 228/228 |  |
| DRHG T=1K | 0.10% | 769.53s | 69/69 | 1.46% | 2857.55s | 109/109 | 4.46% | 3004.28s | 50/50 | 1.71% | 228/228 |  |
| Fast T2T Ts=10, Tg=10 | 10.46% | 1.41s | 45/69 | / | / | 0/109 | / | / | 0/50 | 10.46% | 45/228 |  |
| GFACS^† T=100, K=100 | 31.64% | 166.94s | 66/69 | 86.77% | 2601.13s | 22/109 | / | / | 0/50 | 45.42% | 88/228 |  |
| GFACS^‡ T=100, K=100 | 0.72% | 174.21s | 69/69 | 3.76% | 9142.93s | 83/109 | / | / | 0/50 | 2.38% | 152/228 |  |
| CVRP | Nearest Neighbor | 21.17% | 0.03s | 99/99 | 15.18% | 1.08s | 5/5 | 11.80% | 14.63s | 6/6 | 20.39% | 110/110 |
| Random Insertion | 75.00% | 0.00s | 36/99 | / | / | 0/5 | / | / | 0/6 | 75.00% | 36/110 |  |
| HGS t=n/3 | 0.29% | 111.24s | 99/99 | 3.59% | 1428.41s | 5/5 | 7.86% | 6926.41s | 6/6 | 0.85% | 110/110 |  |
| AILS-II t=n/3 | 0.57% | 133.95s | 99/99 | 1.58% | 1388.42s | 5/5 | 1.58% | 5646.48s | 6/6 | 0.68% | 110/110 |  |
| BQ | 8.87% | 3.63s | 99/99 | 20.28% | 39.97s | 5/5 | 41.52% | 202.69s | 5/6 | 10.89% | 109/110 |  |
| LEHD^∗ greedy | 11.25% | 1.53s | 98/99 | 19.22% | 99.43s | 5/5 | 32.80% | 852.02s | 2/6 | 12.04% | 105/110 |  |
| SIL^∗ greedy | 40.04% | 2.48s | 65/99 | 16.09% | 26.86s | 5/5 | 10.81% | 146.83s | 6/6 | 36.16% | 76/110 |  |
| ICAM | 5.00% | 0.42s | 99/99 | 11.69% | 32.32s | 5/5 | / | / | 0/6 | 5.32% | 104/110 |  |
| ELG | 8.03% | 1.29s | 99/99 | 18.51% | 30.21s | 5/5 | 29.38% | 133.08s | 2/6 | 8.93% | 106/110 |  |
| INViT-3V | 13.15% | 4.72s | 99/99 | 19.03% | 77.33s | 5/5 | 23.91% | 496.25s | 5/6 | 13.91% | 109/110 |  |
| L2R | 8.16% | 2.49s | 99/99 | 11.62% | 23.99s | 5/5 | 11.08% | 97.12s | 6/6 | 8.48% | 110/110 |  |
| DGL | 15.27% | 2.22s | 99/99 | 17.96% | 22.60s | 5/5 | 18.69% | 78.64s | 5/6 | 15.55% | 109/110 |  |
| ReLD | 4.10% | 0.41s | 99/99 | 10.22% | 5.29s | 5/5 | 11.27% | 28.87s | 3/6 | 4.58% | 107/110 |  |
| L2C-Insert^∗ greedy | 6.87% | 2.73s | 99/99 | 22.37% | 616.75s | 5/5 | 49.41% | 5525.71s | 2/6 | 8.40% | 106/110 |  |
| DACT T=1K | 16.42% | 246.51s | 74/99 | 17.70% | 479.82s | 1/5 | / | / | 0/6 | 16.44% | 75/110 |  |
| NeuOpt T=1K | 26.93% | 571.14s | 36/99 | / | / | 0/5 | / | / | 0/6 | 26.93% | 36/110 |  |
| L2C-Insert^∗ T=1K | 3.21% | 344.05s | 99/99 | 18.87% | 6166.21s | 5/5 | 44.29% | 32754.86s | 2/6 | 4.72% | 106/110 |  |
| LEHD^∗ RRC1K | 3.58% | 796.15s | 99/99 | 11.73% | 2043.74s | 5/5 | 21.98% | 2820.28s | 2/6 | 4.32% | 106/110 |  |
| SIL^∗ PRC1K | 21.38% | 1307.97s | 99/99 | 8.28% | 3471.69s | 5/5 | 7.40% | 4251.88s | 6/6 | 20.02% | 110/110 |  |
| DRHG T=1K | 11.11% | 1114.60s | 99/99 | 17.95% | 2529.12s | 5/5 | 16.95% | 5376.37s | 6/6 | 11.74% | 110/110 |  |
| GFACS^† T=100, K=100 | 36.83% | 437.38s | 99/99 | 34.09% | 9654.33s | 3/5 | / | / | 0/6 | 36.75% | 102/110 |  |
| GFACS^‡ T=100, K=100 | 2.60% | 405.81s | 99/99 | 7.65% | 14884.94s | 4/5 | / | / | 0/6 | 2.80% | 103/110 |  |

#### VI-C 2 Experimental Settings

##### VI-C 2a Problem and Instance Setting

The proposed evaluation pipeline assesses NRSs on representative TSP and CVRP. The test instances are drawn from benchmarks and challenge sets, covering diverse data distributions, with scales in $(0,100\text{K]}$, and specific edge-weight types (EUC_2D or CEIL_2D) to ensure integer Euclidean distance matrices. All selected instances have available best known solutions (BKS) and do not impose additional constraints, such as fixed route numbers or duration limits. The composition of the test instances is detailed as follows.

- •
TSPLIB [111]  a famous dataset with TSP instances from various sources. 77 EUC_2D instances and 4 CEIL_2D are included. Note that the EUC_2D instance $linhp318$ is excluded due to a fixed-edge constraint.
- •
National a dataset with 27 EUC_2D TSP instances for countries, based on data from the National Imagery and Mapping Agency. All the instances are included.
- •
VLSI a dataset with 102 EUC_2D TSP instances of industrial applications of the very large-scale integration design from the Bonn Institute. Note that 4 instances (SRA104815, ARA238025, LRA498378, LRB744710) with over 100K nodes are excluded.
- •
Dataset of The 8th DIMACS Implementation Challenge (TSP) a dataset comprises a selection of instances from the TSPLIB library, supplemented by generated instances. To avoid redundant instances and to satisfy distributional diversity and edge-weight-type consistency, only the 22 generated EUC_2D instances with clustered nodes are included. Note that the instance C316k.0 with over 100K clustered nodes is excluded.
- •
CVRPLIB [125]  a famous dataset with 14 sets of CVRP instances from several academic literature and real-world applications. The library encompasses the adopted open-source instances in the 12th DIMACS Implementation Challenge (CVRP). 100 EUC_2D instances of Set X [125] and 10 of Set AGS [4] are included.

##### VI-C 2b Metrics

The solvers are evaluated from the following three perspectives:

- •
Effectiveness a solver’s ability to maintain high performance across out-of-distribution instances. It is measured by the average gap relative to the BKSs.
- •
Efficiency a solver’s ability to solve the instances in a reasonable time. It is measured by the average computational time a solver requires to output the solutions.
- •
Reliability a solver’s ability to successfully solve instances within the current scope. It is measured by the number of instances a solver can handle before failure, where “failure” encompasses Out-of-Memory (OOM) errors, performance breakdowns (*i.e.*, gaps exceeding 100% [18, 89]), or timeouts (per-instance runtime beyond 36,000s for NRSs).

Results are reported separately for three instance scale groups: small ((0,1K)), medium ([1K,10K)], and large ([10K,100K]). The overall aggregated results are also provided. Results for unsolvable conditions are denoted by “/”.

##### VI-C 2c Inference

For advanced heuristics, the termination criterion follows common practice in the heuristic literature [96], where the time budget is set proportional to the instance size. To align with the inference time of NRSs, this multiple is set to one-third. Furthermore, the runtime required to reach the current best solution or the BKS is reported. Results are averaged over 10 independent runs.

For NRSs, to facilitate a direct and fair comparison, all methods are evaluated with greedy inference. In other words, special decoding strategies (*e.g.*, beam search) are deliberately excluded, while data augmentation and fine-tuning techniques (*e.g.*, active search [7, 47]) are deactivated. Unless otherwise specified, additional operator-based local search processes (*e.g.*, 2-opt) are also disabled to preserve experimental fairness and prevent possible shifts in categories of NRSs. All other configurations are kept at their method-specific defaults. For improvement-based methods, the number of iterations is set as the maximum value specified in the original configurations. No additional training is conducted during evaluation. Instead, all publicly available pretrained models (trained on instances with specific scales and uniform node distribution) are tested, and the reported result for each NRS corresponds to the best-performing one, selected by prioritizing reliability first and effectiveness second.
Complete results are provided in Tables [XII](https://arxiv.org/html/2602.21761v1#A1.T12) and [XIII](https://arxiv.org/html/2602.21761v1#A1.T13) in Appendix [A](https://arxiv.org/html/2602.21761v1#A1).

#### VI-C 3 Performance Evaluation

Overall, the results presented in Table [IX](https://arxiv.org/html/2602.21761v1#S6.T9) lead to conclusions fundamentally different from those under the conventional evaluation pipeline.

Overall Performance of NRSs Under the proposed evaluation pipeline, NRSs generally underperform SOTA heuristics in both effectiveness and efficiency, with the gap widening as the problem size increases. Even with comparable runtime, improvement-based NRSs still fall short of SOTA heuristics across all scales. In terms of reliability, only a few NRSs (L2R, SIL (PRC), and DRHG) can successfully solve all TSP and CVRP instances, among which L2R is the only construction-based method. For the remaining methods, only SIL (Greedy) and L2C-Insert (with both greedy and iterative inference) manage to solve every TSP instance. Notably, all successful cases discussed above benefit from techniques for search space reduction (discussed in Section [VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1)). These results indicate a narrow solvable range of current NRSs.

Performance of Construction-based NRSs The performance of construction-based NRSs is less encouraging than that indicated by the conventional evaluation pipeline. In terms of effectiveness, many single-stage NRSs underperform simple heuristics from the same subcategory. For CVRP, the effectiveness of appending NRS SIL (Greedy) deteriorates at small and medium scales, whereas BQ, LEHD (Greedy), ELG, INViT, and DGL degrade on medium- and large-scale instances. All of these NRSs fall short of the nearest neighbor heuristic. Similarly, for TSP, the insertion NRS L2C-Insert (Greedy) is outperformed by random insertion on medium- and large-scale instances. Nevertheless, a few methods stand out: L2R achieves strong effectiveness and reliability on both problems, and ReLD attains competitive effectiveness on CVRP despite limited reliability on large instances. In terms of efficiency, L2C-Insert (Greedy) runs slower than other single-stage methods on CVRP because its released implementation evaluates all unvisited nodes, rather than restricting attention to the nearest one as described in the paper. As the only two-stage NRS, H-TSP benefits from its architecture to achieve inference times comparable to those of the nearest neighbor heuristic while maintaining stable effectiveness on TSP. Nevertheless, its reliability is not strong even in small-scale instances.

Performance of Improvement-based NRSs The performance of improvement-based NRSs is mixed. Among single-solution-based NRSs, LNS methods (especially the direct ones) generally exhibit superior effectiveness and reliability compared to the small neighborhood counterparts, consistent with their recognized advantages of escaping local optima. For example, DRHG approaches LKH-3’s effectiveness on TSP across scales. Nevertheless, a few LNS methods exhibit effectiveness deterioration on large-scale instances (LEHD (RRC) and L2C-Insert (Iteration) for both, DRHG for CVRP), and SIL (PRC) degrades on small-scale CVRP instances. In all these deterioration cases, they perform worse than at least one of the two simple construction-based heuristics.
Besides, L2C-Insert (Iteration) remains inefficient as in its construction-based version. In addition, GenSCO and Fast T2T underperform other large neighborhood NRSs across all evaluation aspects. Fast T2T adopts a distance-based insertion strategy similar to random insertion and achieves comparable effectiveness, suggesting it fails to effectively leverage information from out-of-distribution instances. Besides, GenSCO exhibits a performance drop using greedy decoding without explicitly incorporating distance information. This observation suggests that, for the distance-driven TSP, additional spatial bias during element selection remains important for current NAR NRSs. Lastly, the two small-neighborhood DACT and NeuOpt incur high computational cost in AR node-pair selection and insufficient convergence from per-step sampling, resulting in limited effectiveness and reliability.

For population-based NRSs, two variants of GFACS are evaluated: (1) the original version, which returns the best solution over the entire run, and (2) a variant, which disables local search in the final iteration and outputs the best solution from that iteration. The latter variant aligns with the inference settings of other NRSs, and follows the original study’s motivation that local search primarily facilitates convergence during training. The variant shows degraded effectiveness and reliability, indicating that edge weights, shaped by learned edge-preference rules and iterative population dynamics, still provide insufficient guidance for constructing high-quality solutions for instances across diverse distributions.

### VI-D Discussions

#### VI-D 1 Advantages of the Proposed Evaluation Pipeline

Compared with the conventional evaluation pipeline, the proposed pipeline offers several advantages in the following aspects:

- •
Purpose of Evaluation The proposed pipeline is designed specifically to assess the zero-shot generalization performance of NRSs. It enforces a consistent single-model evaluation across methods, thereby enhancing comparability and strengthening the validity of conclusions. In contrast, the conventional pipeline primarily focuses on scalability without clearly specifying whether evaluation is conducted under a single- or multi-model condition. Consequently, this ambiguity makes fair comparisons difficult, as some results reflect generalization from a single model, while others report purely in-distribution performance of multiple models, each evaluated only on its matched distribution.
- •
Instance Selection The proposed pipeline draws test instances from well-known benchmarks and challenge sets, rather than ad-hoc synthetic distributions. It thus enables a more comprehensive evaluation across diverse distributions while mitigating generator-induced distribution shifts that could bias results, preserving fairness and comparability. Notably, although prior work sometimes reports results on TSPLIB or CVRPLIB, the instances are often restricted to selected scale ranges or specific subsets, which may introduce selection bias. In contrast, the proposed pipeline uses a broader instance pool, providing a more robust assessment.
- •
Inference and Metrics For inference, the pipeline enforces a greedy decoding setting and avoids arbitrary add-on enhancements or parameter tuning on specific instances, thereby ensuring a more equitable comparison across methods. For the metrics, in addition to effectiveness, the proposed pipeline explicitly introduces reliability as a complementary metric, enabling a more comprehensive evaluation of algorithmic performance.

#### VI-D 2 Principles for Method Selection

To ensure fair and informative comparisons, two principles for selecting NRSs and baseline heuristics are followed.

In-category Comparison of NRSs The primary goal of NRS experiments is to demonstrate the effectiveness of the learned heuristics. However, NRSs from different categories may rely on distinct heuristic frameworks, each requiring different levels of domain knowledge and computational resources. Therefore, cross-category comparisons may fail to accurately reflect the specific contribution of a DL model to the overall performance. For this reason, comparisons and conclusions are restricted as much as possible to NRSs and traditional heuristics that belong to the same category.

Baseline Heuristic Selection Results under the proposed pipeline highlighted a performance gap between many NRSs and traditional heuristics, contradicting several existing claims that such NRSs can outperform SOTA heuristic methods [77, 89, 157]. In many of those studies, comparisons are conducted either under settings that disadvantage heuristics or against relatively weak heuristic baselines. For the former cases, when heuristics are allowed a shorter initial period [31] under the same time budget, they can achieve better effectiveness and efficiency on large-scale instances, as shown in Table [IX](https://arxiv.org/html/2602.21761v1#S6.T9). For the latter cases, widely-used CVRP baselines in the NRS literature, such as HGS and LKH-3, are not selected for comparison in recent heuristics literature [96, 97, 20]. Advanced heuristic methods, such as AILS-II, can achieve more competitive performance on medium- and large-scale instances where NRSs are often claimed to outperform traditional heuristics. The evidence above indicates that heuristic baselines in most NRS literature lag behind the SOTA. Accordingly, baseline heuristics in this experiment are selected from the SOTA heuristic literature and appropriately configured to enable a more informative comparison.

#### VI-D 3 Does Deep Learning Truly Help in NRSs?

The experimental results suggest that DL does contribute to good NRS performance. Under the conventional evaluation pipeline, most NRSs explicitly designed for generalization outperform their handcrafted heuristic counterparts within the same category and framework on uniformly distributed instances on different scales. Notably, across both pipelines, appending methods like ReLD, ICAM, and L2R, which incorporate distance information as an auxiliary bias, can achieve stronger generalization performance than the nearest neighbor heuristic. This outperformance suggests that DL models can extract useful implicit knowledge complementary to explicit distance information, demonstrating generalization potential.

On the other hand, the current generalization capability of NRSs remains limited. Under the proposed pipeline, all NRSs demonstrate lower-than-expected effectiveness and, at times, worse reliability. In several cases, their effectiveness even falls below that of simple construction-based heuristics, suggesting that rules learned solely from instances with the uniform node distribution fail to transfer robustly across diverse distributions.

Taken together, these results support a cautiously optimistic conclusion. DL can indeed capture implicit knowledge and yield measurable gains in solving various routing problems. The less favorable performance under the proposed pipeline likely stems from overfitting due to a narrow training distribution, rather than a fundamental limitation of NRSs. In addition, the algorithmic frameworks of current NRSs, especially improvement-based ones, are often simpler than those of advanced heuristics, which may also constrain performance. Therefore, NRSs retain clear research value and foreseeable potential for further performance gain in in-problem generalization. Within the same heuristic framework, DL holds the promise of discovering rules that outperform or complement handcrafted designs, thereby further improving overall performance. Moreover, since few-shot adaptation can rapidly align learned implicit knowledge with a target distribution, NRSs offer a viable pathway to practical deployment by combining general-purpose knowledge with distribution-specific patterns to serve a wide range of applications.

## VII Challenges, Frontier Strategies, and Future Directions

The increasing demand for NRSs to perform effectively and reliably in real-world settings has brought significant challenges in generalization. This section elaborates on these challenges, discusses the strategies explored in the recent literature, and outlines potential future research directions.

### VII-A In-problem Generalization

In-problem generalization refers to maintaining stable performance across instances from different data distributions of a single problem, including but not limited to variations in scale, node distribution, and constraint tightness. It is influenced by both how models extract and utilize information and by the distribution shift between the training and test data. Correspondingly, related studies are analyzed from two complementary perspectives: model design and data distribution.

Model Design Strategies for improving in-problem generalization through model design follow two main lines. On the one hand, a few appending methods [103, 141, 27, 88, 89, 28, 152] and restricted direct LNS methods [75] allocate more attention layers to dynamically capture relationships between the partial solution and remaining nodes, and among the remaining nodes themselves. On the other hand, some appending methods [55, 116, 73, 33, 135, 151] incorporate node-wise distances into specific model modules, given VRPs’ typical distance-based objectives. Both strategies enable more informed node selection, albeit with different trade-offs. The former incurs higher memory and runtime overhead due to stepwise re-embedding, while the latter relies more on handcrafted designs and is applicable only to distance-based objectives. These limitations highlight the need for future architectures that jointly improve state representation and inference efficiency without restricting to specific objectives.

Data Distribution Existing strategies related to data distribution generally employ two strategies. The first strategy pre-processes test instances to resemble the training distribution. For example, certain two-stage methods [49, 102], restricted direct LNS methods [88, 89, 76, 61, 157, 17, 145, 150], and sequential search methods [30] decompose the original problem into subproblems with scales comparable to those of the training data. Besides, statically [110, 121, 78, 79, 82, 30, 143, 99] or dynamically [33, 135, 28, 27] reducing the search space [120] has been proven effective across categories. Coordinate normalization can further align the test distribution with the training distributions [145, 150, 30, 28, 152, 16]. However, they often prioritize locally optimal partial solutions, which can trap the complete solution in local optima and degrade final performance. Incorporating global information during search may help mitigate this issue. The second strategy diversifies the training data, *e.g.*, by incorporating instances with different scales or node distributions [151, 135, 155, 59, 54, 50, 132]. Given that NRSs are typically trained on narrow distributions and may overfit, as our experiments suggest, enriching the training data with diverse distributions may already yield substantial gains. Nevertheless, identifying representative distributions and designing effective training strategies remain challenging. Neither the factors shaping data distributions nor the mechanisms through which distributions affect model performance are yet systematically understood. A thorough investigation of these issues is therefore needed in future work.

### VII-B Cross-problem Generalization

Most NRSs train a specialized model for each problem. This “one problem, one model” paradigm is inefficient because it ignores the structural similarities across VRPs. Therefore, a promising research direction is to develop a general-purpose solver that can handle multiple VRPs without costly problem-specific engineering or retraining from scratch.
Existing strategies primarily adapt established DL techniques to the AR appending methods. For example, a recent trend for developing general VRP solvers is multi-task learning [83, 154, 72, 9, 85, 38, 148], where a single model is trained and tested on a set of VRP variants with combinations of predefined attributes. However, the observed generalization is at best limited to variants with novel attribute combinations and, at worst, amounts to in-domain performance. This reliance on a predefined attribute set fundamentally restricts generalization, since the attributes in real-world problems cannot be fully anticipated or enumerated in advance. Other strategies employ the model with a shared backbone and problem-specific adapters for different VRPs [81, 26]. While this design enables flexible fine-tuning, it introduces unavoidable limitations. Particularly, related constraint handling remains inherently problem-specific and tied to the adapter design, preventing zero-shot application to unseen problems. Furthermore, this design incurs additional computational overhead during fine-tuning.

To outline potential pathways toward zero-shot generalization for unseen VRPs, two promising future research directions are highlighted: input representations and constraint handling. (1) For input representations, existing methods largely rely on fixed-length attribute or problem vectors, which inherently limit the range of solvable problems. Accordingly, moving beyond attribute-predefined designs toward more general input representations is thus a critical step for broader applicability. (2) For constraint handling, step-wise masking in AR single-stage methods can enforce hard constraints but introduce manual intervention and is inapplicable to certain problems [10] (*e.g.*, TSPTW). To address both issues, a promising direction is to develop intervention-free constraint-handling mechanisms, especially for cross-problem settings.

## VIII Conclusions

This survey systematically reviews neural routing solvers (NRSs) from the perspective of heuristics. They are identified as heuristic algorithms in which DL-learned rules replace handcrafted ones. A hierarchical taxonomy is introduced based on how solutions are constructed or improved. This perspective enables consistent analysis of connections and developmental trends among NRSs, and naturally links their designs to established heuristic principles within corresponding categories.

Besides, a generalization-focused evaluation pipeline is proposed to address limitations of the conventional one, and representative NRSs are benchmarked under both pipelines. Results under the new pipeline show that NRSs trained on a narrow range of instance distributions can be outperformed by simple construction-based heuristics such as nearest neighbor and random insertion, indicating that the conventional pipeline can lead to overly optimistic conclusions. These findings motivate further discussion of the new pipeline’s advantages, principles for method selection, and the role of DL in NRSs despite current performance gaps. Finally, two central challenges in the field, *i.e.*, in-problem and cross-problem generalization, are analyzed. Related prevailing strategies are summarized, and several directions for future work are outlined.

## References

- [1]
E. Alanzi and M. E. B. Menai (2025)
Solving the traveling salesman problem with machine learning: a review of recent advances and challenges.
Artificial Intelligence Review 58 (9), pp. 267.
Cited by: [TABLE I](https://arxiv.org/html/2602.21761v1#S1.T1.3.3.4.1.1),
[§I](https://arxiv.org/html/2602.21761v1#S1.p4.1).
- [2]
J. Alegre, M. Laguna, and J. Pacheco (2007)
Optimizing the periodic pick-up of raw materials for a manufacturer of auto parts.
European Journal of Operational Research 179 (3), pp. 736–746.
Cited by: [§I](https://arxiv.org/html/2602.21761v1#S1.p1.1).
- [3]
I. Araya, O. Rojas, M. Vásquez, G. Marín, and L. Robles (2026)
What makes a transformer solve the tsp? a component-wise analysis.
Preprints.
Cited by: [TABLE I](https://arxiv.org/html/2602.21761v1#S1.T1.4.4.4.1.1),
[§I](https://arxiv.org/html/2602.21761v1#S1.p4.1).
- [4]
F. Arnold, M. Gendreau, and K. Sörensen (2019)
Efficiently solving very large-scale routing problems.
Computers & Operations Research 107, pp. 32–42.
Cited by: [5th item](https://arxiv.org/html/2602.21761v1#S6.I5.i5.p1.1).
- [5]
T. Back (1996)
Evolutionary algorithms in theory and practice: evolution strategies, evolutionary programming, genetic algorithms.
Oxford university press.
Cited by: [§V](https://arxiv.org/html/2602.21761v1#S5.p1.1).
- [6]
R. Bai, X. Chen, Z. Chen, T. Cui, S. Gong, W. He, X. Jiang, H. Jin, J. Jin, G. Kendall, et al. (2023)
Analytics and machine learning in vehicle routing research.
International Journal of Production Research 61 (1), pp. 4–30.
Cited by: [TABLE I](https://arxiv.org/html/2602.21761v1#S1.T1.4.4.4.1.1),
[§I](https://arxiv.org/html/2602.21761v1#S1.p4.1).
- [7]
I. Bello, H. Pham, Q. V. Le, M. Norouzi, and S. Bengio (2017)
Neural combinatorial optimization with reinforcement learning.
International conference on learning representations workshop.
Cited by: [§III-A1](https://arxiv.org/html/2602.21761v1#S3.SS1.SSS1.p1.1),
[§III-A1](https://arxiv.org/html/2602.21761v1#S3.SS1.SSS1.p2.1),
[TABLE II](https://arxiv.org/html/2602.21761v1#S3.T2.1.4.2),
[¶VI-C2c](https://arxiv.org/html/2602.21761v1#S6.SS3.SSS2.P3.p2.1).
- [8]
Y. Bengio, A. Lodi, and A. Prouvost (2021)
Machine learning for combinatorial optimization: a methodological tour d’horizon.
European Journal of Operational Research 290 (2), pp. 405–421.
Cited by: [TABLE I](https://arxiv.org/html/2602.21761v1#S1.T1.3.3.4.1.1),
[§I](https://arxiv.org/html/2602.21761v1#S1.p3.1),
[§I](https://arxiv.org/html/2602.21761v1#S1.p4.1).
- [9]
F. Berto, C. Hua, N. G. Zepeda, A. Hottung, N. Wouda, L. Lan, J. Park, K. Tierney, and J. Park (2024)
Routefinder: towards foundation models for vehicle routing problems.
ICML 2024 Workshop on Foundation Models in the Wild.
Cited by: [§VII-B](https://arxiv.org/html/2602.21761v1#S7.SS2.p1.1).
- [10]
J. Bi, Y. Ma, J. Zhou, W. Song, Z. Cao, Y. Wu, and J. Zhang (2024)
Learning to handle complex constraints for vehicle routing problems.
Advances in Neural Information Processing Systems 37, pp. 93479–93509.
Cited by: [§VII-B](https://arxiv.org/html/2602.21761v1#S7.SS2.p2.1).
- [11]
A. Bogyrbayeva, M. Meraliyev, T. Mustakhov, and B. Dauletbayev (2024)
Machine learning to solve vehicle routing problems: a survey.
IEEE Transactions on Intelligent Transportation Systems 25 (6), pp. 4754–4772.
Cited by: [TABLE I](https://arxiv.org/html/2602.21761v1#S1.T1.3.3.4.1.1),
[§I](https://arxiv.org/html/2602.21761v1#S1.p4.1).
- [12]
J. Brandão (2020)
A memory-based iterated local search algorithm for the multi-depot open vehicle routing problem.
European Journal of Operational Research 284 (2), pp. 559–571.
Cited by: [¶IV-B1a](https://arxiv.org/html/2602.21761v1#S4.SS2.SSS1.P1.p3.1).
- [13]
Q. Cappart, D. Chételat, E. B. Khalil, A. Lodi, C. Morris, and P. Veličković (2023)
Combinatorial optimization and reasoning with graph neural networks.
Journal of Machine Learning Research 24 (130), pp. 1–61.
Cited by: [TABLE I](https://arxiv.org/html/2602.21761v1#S1.T1.3.3.4.1.1),
[§I](https://arxiv.org/html/2602.21761v1#S1.p4.1).
- [14]
F. Chalumeau, S. Surana, C. Bonnet, N. Grinsztajn, A. Pretorius, A. Laterre, and T. Barrett (2023)
Combinatorial optimization with policy adaptation using latent space search.
Advances in Neural Information Processing Systems 36, pp. 7947–7959.
Cited by: [TABLE VI](https://arxiv.org/html/2602.21761v1#S4.T6.1.4.3),
[§V](https://arxiv.org/html/2602.21761v1#S5.p3.1).
- [15]
X. Chen and Y. Tian (2019)
Learning to perform local rewriting for combinatorial optimization.
Advances in Neural Information Processing Systems 32.
Cited by: [§IV-A1](https://arxiv.org/html/2602.21761v1#S4.SS1.SSS1.p2.1),
[TABLE IV](https://arxiv.org/html/2602.21761v1#S4.T4.1.3.6).
- [16]
Y. Chen, R. Chen, F. Luo, and Z. Wang (2025)
Improving generalization of neural combinatorial optimization for vehicle routing problems via test-time projection learning.
Advances in Neural Information Processing Systems,.
Cited by: [§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p3.1).
- [17]
H. Cheng, H. Zheng, Y. Cong, W. Jiang, and S. Pu (2023)
Select and optimize: learning to solve large-scale tsp instances.
In International Conference on Artificial Intelligence and Statistics,
pp. 1219–1231.
Cited by: [¶IV-B1b](https://arxiv.org/html/2602.21761v1#S4.SS2.SSS1.P2.p3.1),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p3.1).
- [18]
I. Choi, W. Shin, S. Cho, and H. Kim (2025)
Towards generalizable multi-policy optimization with self-evolution for job scheduling.
Advances in Neural Information Processing Systems,.
Cited by: [3rd item](https://arxiv.org/html/2602.21761v1#S6.I6.i3.p1.1).
- [19]
J. Choo, Y. Kwon, J. Kim, J. Jae, A. Hottung, K. Tierney, and Y. Gwon (2022)
Simulation-guided beam search for neural combinatorial optimization.
Advances in Neural Information Processing Systems 35, pp. 8760–8772.
Cited by: [§III-A1](https://arxiv.org/html/2602.21761v1#S3.SS1.SSS1.p1.1).
- [20]
J. Christiaens and G. Vanden Berghe (2020)
Slack induction by string removals for vehicle routing problems.
Transportation Science 54 (2), pp. 417–433.
Cited by: [§VI-D2](https://arxiv.org/html/2602.21761v1#S6.SS4.SSS2.p3.1).
- [21]
K. Chung, C. Lee, and Y. Tsang (2025)
Neural combinatorial optimization with reinforcement learning in industrial engineering: a survey.
Artificial Intelligence Review 58 (5), pp. 130.
Cited by: [TABLE I](https://arxiv.org/html/2602.21761v1#S1.T1.3.3.4.1.1),
[§I](https://arxiv.org/html/2602.21761v1#S1.p4.1).
- [22]
G. Clarke and J. W. Wright (1964)
Scheduling of vehicles from a central depot to a number of delivery points.
Operations Research 12 (4), pp. 568–581.
Cited by: [§I](https://arxiv.org/html/2602.21761v1#S1.p1.1).
- [23]
W.J. Cook, W.H. Cunningham, W.R. Pulleyblank, and A. Schrijver (1997)
Combinatorial optimization.
A Wiley-Interscience publication, Wiley.
External Links: ISBN 9780471558941,
LCCN 97035774
Cited by: [§II-B](https://arxiv.org/html/2602.21761v1#S2.SS2.p1.1).
- [24]
J. Cordeau and G. Laporte (2003)
A tabu search heuristic for the static multi-vehicle dial-a-ride problem.
Transportation Research Part B: Methodological 37 (6), pp. 579–594.
Cited by: [§I](https://arxiv.org/html/2602.21761v1#S1.p1.1).
- [25]
G. B. Dantzig and J. H. Ramser (1959)
The truck dispatching problem.
Management Science 6 (1), pp. 80–91.
Cited by: [§I](https://arxiv.org/html/2602.21761v1#S1.p1.1).
- [26]
D. Drakulic, S. Michel, and J. Andreoli (2025)
GOAL: a generalist combinatorial optimization agent learning.
International Conference on Learning Representations.
Cited by: [TABLE II](https://arxiv.org/html/2602.21761v1#S3.T2.1.29.4.1),
[§VII-B](https://arxiv.org/html/2602.21761v1#S7.SS2.p1.1).
- [27]
D. Drakulic, S. Michel, F. Mai, A. Sors, and J. Andreoli (2024)
Bq-nco: bisimulation quotienting for efficient neural combinatorial optimization.
Advances in Neural Information Processing Systems 36.
Cited by: [TABLE X](https://arxiv.org/html/2602.21761v1#A1.T10.1.5.1),
[§II-D1](https://arxiv.org/html/2602.21761v1#S2.SS4.SSS1.p1.1),
[§III-A1](https://arxiv.org/html/2602.21761v1#S3.SS1.SSS1.p3.1),
[§III-A1](https://arxiv.org/html/2602.21761v1#S3.SS1.SSS1.p4.1),
[TABLE II](https://arxiv.org/html/2602.21761v1#S3.T2.1.14.4),
[TABLE VII](https://arxiv.org/html/2602.21761v1#S6.T7.6.9.5),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p2.1),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p3.1).
- [28]
H. Fang, Z. Song, P. Weng, and Y. Ban (2024)
INViT: a generalizable routing problem solver with invariant nested view transformer.
In International Conference on Machine Learning,
pp. 12973–12992.
Cited by: [TABLE X](https://arxiv.org/html/2602.21761v1#A1.T10.1.10.1),
[§III-A1](https://arxiv.org/html/2602.21761v1#S3.SS1.SSS1.p4.1),
[TABLE II](https://arxiv.org/html/2602.21761v1#S3.T2.1.21.1),
[TABLE VII](https://arxiv.org/html/2602.21761v1#S6.T7.6.12.1),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p2.1),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p3.1).
- [29]
M. L. Fisher and R. Jaikumar (1981)
A generalized assignment heuristic for vehicle routing.
Networks 11 (2), pp. 109–124.
Cited by: [§III-B](https://arxiv.org/html/2602.21761v1#S3.SS2.p1.1).
- [30]
Z. Fu, K. Qiu, and H. Zha (2021)
Generalize a small pre-trained model to arbitrarily large tsp instances.
In Proceedings of the AAAI Conference on Artificial Intelligence,
Vol. 35, pp. 7474–7482.
Cited by: [§IV-A2](https://arxiv.org/html/2602.21761v1#S4.SS1.SSS2.p2.1),
[TABLE IV](https://arxiv.org/html/2602.21761v1#S4.T4.1.15.6.1.1),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p3.1).
- [31]
Z. Fu, S. Sun, J. Ren, T. Yu, H. Zhang, Y. Liu, L. Huang, X. Yan, and P. Lu (2023)
A hierarchical destroy and repair approach for solving very large-scale travelling salesman problem.
arXiv preprint arXiv:2308.04639.
Cited by: [§VI-D2](https://arxiv.org/html/2602.21761v1#S6.SS4.SSS2.p3.1).
- [32]
B. Funke, T. Grünert, and S. Irnich (2005)
Local search for vehicle routing and scheduling problems: review and conceptual integration.
Journal of Heuristics 11 (4), pp. 267–306.
Cited by: [§II-B2](https://arxiv.org/html/2602.21761v1#S2.SS2.SSS2.p1.1).
- [33]
C. Gao, H. Shang, K. Xue, D. Li, and C. Qian (2024)
Towards generalizable neural solvers for vehicle routing problems via ensemble with transferrable local policy.
In International Joint Conference on Artificial Intelligence,
pp. 6914–6922.
Cited by: [TABLE X](https://arxiv.org/html/2602.21761v1#A1.T10.1.9.1),
[§III-A1](https://arxiv.org/html/2602.21761v1#S3.SS1.SSS1.p4.1),
[TABLE II](https://arxiv.org/html/2602.21761v1#S3.T2.1.20.1),
[TABLE VII](https://arxiv.org/html/2602.21761v1#S6.T7.6.11.1),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p2.1),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p3.1).
- [34]
L. Gao, M. Chen, Q. Chen, G. Luo, N. Zhu, and Z. Liu (2020)
Learn to design the heuristics for vehicle routing problem.
arXiv preprint arXiv:2002.08539.
Cited by: [¶IV-B1a](https://arxiv.org/html/2602.21761v1#S4.SS2.SSS1.P1.p2.1),
[TABLE V](https://arxiv.org/html/2602.21761v1#S4.T5.1.5.5).
- [35]
M. Gendreau, J. Potvin, et al. (2010)
Handbook of metaheuristics.
Vol. 2, Springer.
Cited by: [§IV-A1](https://arxiv.org/html/2602.21761v1#S4.SS1.SSS1.p2.1),
[¶IV-B1a](https://arxiv.org/html/2602.21761v1#S4.SS2.SSS1.P1.p1.1),
[¶IV-B1a](https://arxiv.org/html/2602.21761v1#S4.SS2.SSS1.P1.p3.1),
[§IV-B2](https://arxiv.org/html/2602.21761v1#S4.SS2.SSS2.p3.1),
[§V](https://arxiv.org/html/2602.21761v1#S5.p4.1).
- [36]
B. E. Gillett and L. R. Miller (1974)
A heuristic algorithm for the vehicle-dispatch problem.
Operations Research 22 (2), pp. 340–349.
Cited by: [§II-B1](https://arxiv.org/html/2602.21761v1#S2.SS2.SSS1.p1.1),
[§III-A1](https://arxiv.org/html/2602.21761v1#S3.SS1.SSS1.p1.1),
[§III-B](https://arxiv.org/html/2602.21761v1#S3.SS2.p1.1).
- [37]
Y. L. Goh, Z. Cao, Y. Ma, Y. Dong, M. H. Dupty, and W. S. Lee (2024)
Hierarchical neural constructive solver for real-world tsp scenarios.
In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining,
pp. 884–895.
Cited by: [§III-A1](https://arxiv.org/html/2602.21761v1#S3.SS1.SSS1.p4.1).
- [38]
Y. L. Goh, Z. Cao, Y. Ma, J. Zhou, M. H. Dupty, and W. S. Lee (2025)
SHIELD: multi-task multi-distribution vehicle routing solver with sparsity and hierarchy.
International Conference of Machine Learning.
Cited by: [§VII-B](https://arxiv.org/html/2602.21761v1#S7.SS2.p1.1).
- [39]
C. P. Gomes and B. Selman (2001)
Algorithm portfolios.
Artificial Intelligence 126 (1-2), pp. 43–62.
Cited by: [§I](https://arxiv.org/html/2602.21761v1#S1.p2.1).
- [40]
A. Graikos, N. Malkin, N. Jojic, and D. Samaras (2022)
Diffusion models as plug-and-play priors.
Advances in Neural Information Processing Systems 35, pp. 14715–14728.
Cited by: [TABLE II](https://arxiv.org/html/2602.21761v1#S3.T2.1.40.5).
- [41]
T. Guo, Y. Mei, M. Zhang, H. Zhao, K. Cai, and W. Du (2025)
Learning-aided neighborhood search for vehicle routing problems.
IEEE Transactions on Pattern Analysis and Machine Intelligence 47 (7), pp. 5930–5944.
Cited by: [§I](https://arxiv.org/html/2602.21761v1#S1.p2.1).
- [42]
N. Hansen (2016)
The cma evolution strategy: a tutorial.
arXiv preprint arXiv:1604.00772.
Cited by: [§V](https://arxiv.org/html/2602.21761v1#S5.p3.1).
- [43]
K. Helsgaun (2000)
An effective implementation of the lin–kernighan traveling salesman heuristic.
European Journal of Operational Research 126 (1), pp. 106–130.
Cited by: [§IV-A2](https://arxiv.org/html/2602.21761v1#S4.SS1.SSS2.p2.1),
[§V](https://arxiv.org/html/2602.21761v1#S5.p4.1).
- [44]
K. Helsgaun (2017)
An extension of the lin-kernighan-helsgaun tsp solver for constrained traveling salesman and vehicle routing problems.
Roskilde: Roskilde University 12, pp. 966–980.
Cited by: [TABLE X](https://arxiv.org/html/2602.21761v1#A1.T10.1.2.1),
[3rd item](https://arxiv.org/html/2602.21761v1#S6.I1.i3.p1.1).
- [45]
V. C. Hemmelmayr, J. Cordeau, and T. G. Crainic (2012)
An adaptive large neighborhood search heuristic for two-echelon vehicle routing problems arising in city logistics.
Computers & Operations Research 39 (12), pp. 3215–3228.
Cited by: [§I](https://arxiv.org/html/2602.21761v1#S1.p1.1).
- [46]
A. Hottung, B. Bhandari, and K. Tierney (2021)
Learning a latent search space for routing problems using variational autoencoders.
In International Conference on Learning Representations,
Cited by: [TABLE VI](https://arxiv.org/html/2602.21761v1#S4.T6.1.3.6),
[§V](https://arxiv.org/html/2602.21761v1#S5.p3.1).
- [47]
A. Hottung, Y. Kwon, and K. Tierney (2022)
Efficient active search for combinatorial optimization problems.
International Conference on Learning Representations.
Cited by: [¶VI-C2c](https://arxiv.org/html/2602.21761v1#S6.SS3.SSS2.P3.p2.1).
- [48]
A. Hottung and K. Tierney (2020)
Neural large neighborhood search for the capacitated vehicle routing problem.
In European Conference on Artificial Intelligence,
pp. 443–450.
Cited by: [¶IV-B1a](https://arxiv.org/html/2602.21761v1#S4.SS2.SSS1.P1.p2.1),
[TABLE V](https://arxiv.org/html/2602.21761v1#S4.T5.1.3.6).
- [49]
Q. Hou, J. Yang, Y. Su, X. Wang, and Y. Deng (2023)
Generalize learned heuristics to solve large-scale vehicle routing problems in real-time.
In International Conference on Learning Representations,
Cited by: [§III-B](https://arxiv.org/html/2602.21761v1#S3.SS2.p1.1),
[TABLE III](https://arxiv.org/html/2602.21761v1#S3.T3.1.5.6),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p3.1).
- [50]
Z. Huang, J. Zhou, Z. Cao, and Y. Xu (2025)
Rethinking light decoder-based solvers for vehicle routing problems.
International Conference on Learning Representations.
Cited by: [TABLE X](https://arxiv.org/html/2602.21761v1#A1.T10.1.13.1),
[§III-A1](https://arxiv.org/html/2602.21761v1#S3.SS1.SSS1.p4.1),
[TABLE II](https://arxiv.org/html/2602.21761v1#S3.T2.1.28.2),
[TABLE VII](https://arxiv.org/html/2602.21761v1#S6.T7.6.15.1),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p3.1).
- [51]
B. A. Huberman, R. M. Lukose, and T. Hogg (1997)
An economics approach to hard computational problems.
Science 275 (5296), pp. 51–54.
Cited by: [§I](https://arxiv.org/html/2602.21761v1#S1.p2.1).
- [52]
B. Hudson, Q. Li, M. Malencia, and A. Prorok (2022)
Graph neural network guided local search for the traveling salesperson problem.
In International Conference on Learning Representations,
Cited by: [§IV-A1](https://arxiv.org/html/2602.21761v1#S4.SS1.SSS1.p2.1),
[TABLE IV](https://arxiv.org/html/2602.21761v1#S4.T4.1.9.5).
- [53]
F. Hutter, H. H. Hoos, K. Leyton-Brown, and T. Stützle (2009)
ParamILS: an automatic algorithm configuration framework.
Journal of Artificial Intelligence Research 36, pp. 267–306.
Cited by: [§I](https://arxiv.org/html/2602.21761v1#S1.p2.1).
- [54]
Y. Jiang, Y. Wu, Z. Cao, and J. Zhang (2022)
Learning to solve routing problems via distributionally robust optimization.
In Proceedings of the AAAI Conference on Artificial Intelligence,
Vol. 36, pp. 9786–9794.
Cited by: [§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p3.1).
- [55]
Y. Jin, Y. Ding, X. Pan, K. He, L. Zhao, T. Qin, L. Song, and J. Bian (2023)
Pointerformer: deep reinforced multi-pointer transformer for the traveling salesman problem.
In Proceedings of the AAAI Conference on Artificial Intelligence,
Vol. 37, pp. 8132–8140.
Cited by: [§III-A1](https://arxiv.org/html/2602.21761v1#S3.SS1.SSS1.p4.1),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p2.1).
- [56]
Y. Jin, X. Yan, S. Liu, and X. Wang (2024)
A unified framework for combinatorial optimization based on graph neural networks.
arXiv preprint arXiv:2406.13125.
Cited by: [TABLE I](https://arxiv.org/html/2602.21761v1#S1.T1.3.3.4.1.1),
[§I](https://arxiv.org/html/2602.21761v1#S1.p4.1).
- [57]
C. K. Joshi, T. Laurent, and X. Bresson (2019)
An efficient graph convolutional network technique for the travelling salesman problem.
arXiv preprint arXiv:1906.01227.
Cited by: [§II-C2](https://arxiv.org/html/2602.21761v1#S2.SS3.SSS2.p1.1),
[§III-A1](https://arxiv.org/html/2602.21761v1#S3.SS1.SSS1.p1.1),
[§III-A1](https://arxiv.org/html/2602.21761v1#S3.SS1.SSS1.p6.1),
[TABLE II](https://arxiv.org/html/2602.21761v1#S3.T2.1.33.5).
- [58]
J. Kennedy (2006)
Swarm intelligence.
In Handbook of Nature-inspired and Innovative Computing: Integrating Classical Models with Emerging Technologies,
pp. 187–219.
Cited by: [§II-B2](https://arxiv.org/html/2602.21761v1#S2.SS2.SSS2.p1.1),
[§V](https://arxiv.org/html/2602.21761v1#S5.p1.1).
- [59]
E. Khalil, H. Dai, Y. Zhang, B. Dilkina, and L. Song (2017)
Learning combinatorial optimization algorithms over graphs.
Advances in Neural Information Processing Systems 30.
Cited by: [§III-A2](https://arxiv.org/html/2602.21761v1#S3.SS1.SSS2.p2.1),
[TABLE II](https://arxiv.org/html/2602.21761v1#S3.T2.1.37.6),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p3.1).
- [60]
M. Kim, S. Choi, H. Kim, J. Son, J. Park, and Y. Bengio (2025)
Ant colony sampling with gflownets for combinatorial optimization.
In International Conference on Artificial Intelligence and Statistics,
pp. 469–477.
Cited by: [TABLE X](https://arxiv.org/html/2602.21761v1#A1.T10.1.21.1),
[TABLE VI](https://arxiv.org/html/2602.21761v1#S4.T6.1.6.3),
[§V](https://arxiv.org/html/2602.21761v1#S5.p2.1),
[TABLE VII](https://arxiv.org/html/2602.21761v1#S6.T7.6.22.4).
- [61]
M. Kim, J. Park, et al. (2021)
Learning collaborative policies to solve np-hard routing problems.
Advances in Neural Information Processing Systems 34, pp. 10418–10430.
Cited by: [¶IV-B1b](https://arxiv.org/html/2602.21761v1#S4.SS2.SSS1.P2.p3.1),
[TABLE V](https://arxiv.org/html/2602.21761v1#S4.T5.1.16.5),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p3.1).
- [62]
M. Kim, J. Park, and J. Park (2022)
Sym-nco: leveraging symmetricity for neural combinatorial optimization.
Advances in Neural Information Processing Systems 35, pp. 1936–1949.
Cited by: [TABLE II](https://arxiv.org/html/2602.21761v1#S3.T2.1.9.4.1).
- [63]
D. Kong, Y. Ma, Z. Cao, T. Yu, and J. Xiao (2024)
Efficient neural collaborative search for pickup and delivery problems.
IEEE Transactions on Pattern Analysis and Machine Intelligence 46 (12), pp. 11019–11034.
Cited by: [TABLE IV](https://arxiv.org/html/2602.21761v1#S4.T4.1.8.4).
- [64]
W. Kool, H. van Hoof, J. Gromicho, and M. Welling (2022)
Deep policy dynamic programming for vehicle routing problems.
In International Conference on Integration of Constraint Programming, Artificial Intelligence, and Operations Research,
pp. 190–213.
Cited by: [§III-A1](https://arxiv.org/html/2602.21761v1#S3.SS1.SSS1.p1.1).
- [65]
W. Kool, H. Van Hoof, and M. Welling (2019)
Attention, learn to solve routing problems!.
International Conference on Learning Representations.
Cited by: [§I](https://arxiv.org/html/2602.21761v1#S1.p3.1),
[§II-C1](https://arxiv.org/html/2602.21761v1#S2.SS3.SSS1.p1.1),
[§II-D2](https://arxiv.org/html/2602.21761v1#S2.SS4.SSS2.p1.1),
[§III-A1](https://arxiv.org/html/2602.21761v1#S3.SS1.SSS1.p2.1),
[TABLE II](https://arxiv.org/html/2602.21761v1#S3.T2.1.5.4),
[¶VI-B2b](https://arxiv.org/html/2602.21761v1#S6.SS2.SSS2.P2.p1.1).
- [66]
J. Kotary, F. Fioretto, P. Van Hentenryck, and B. Wilder (2021)
End-to-end constrained optimization learning: a survey.
International Joint Conference on Artificial Intelligence.
Cited by: [TABLE I](https://arxiv.org/html/2602.21761v1#S1.T1.3.3.4.1.1),
[§I](https://arxiv.org/html/2602.21761v1#S1.p4.1).
- [67]
Y. Kwon, J. Choo, B. Kim, I. Yoon, Y. Gwon, and S. Min (2020)
Pomo: policy optimization with multiple optima for reinforcement learning.
Advances in Neural Information Processing Systems 33, pp. 21188–21198.
Cited by: [§II-D2](https://arxiv.org/html/2602.21761v1#S2.SS4.SSS2.p1.1),
[§III-A1](https://arxiv.org/html/2602.21761v1#S3.SS1.SSS1.p2.1),
[TABLE II](https://arxiv.org/html/2602.21761v1#S3.T2.1.19.2).
- [68]
Y. Kwon, J. Choo, I. Yoon, M. Park, D. Park, and Y. Gwon (2021)
Matrix encoding networks for neural combinatorial optimization.
Advances in Neural Information Processing Systems 34, pp. 5138–5149.
Cited by: [TABLE II](https://arxiv.org/html/2602.21761v1#S3.T2.1.8.4).
- [69]
G. Laporte (1992)
The vehicle routing problem: an overview of exact and approximate algorithms.
European Journal of Operational Research 59 (3), pp. 345–358.
Cited by: [§I](https://arxiv.org/html/2602.21761v1#S1.p1.1).
- [70]
J. K. Lenstra and A. R. Kan (1981)
Complexity of vehicle routing and scheduling problems.
Networks 11 (2), pp. 221–227.
Cited by: [§I](https://arxiv.org/html/2602.21761v1#S1.p1.1).
- [71]
B. Li, G. Wu, Y. He, M. Fan, and W. Pedrycz (2022)
An overview and experimental study of learning-based optimization algorithms for the vehicle routing problem.
IEEE/CAA Journal of Automatica Sinica 9 (7), pp. 1115–1138.
Cited by: [TABLE I](https://arxiv.org/html/2602.21761v1#S1.T1.4.4.4.1.1),
[§I](https://arxiv.org/html/2602.21761v1#S1.p4.1).
- [72]
H. Li, F. Liu, Z. Zheng, Y. Zhang, and Z. Wang (2024)
CaDA: cross-problem routing solver with constraint-aware dual-attention.
International Conference on Machine Learning.
Cited by: [TABLE II](https://arxiv.org/html/2602.21761v1#S3.T2.1.27.2),
[§VII-B](https://arxiv.org/html/2602.21761v1#S7.SS2.p1.1).
- [73]
J. Li, Y. Ma, Z. Cao, Y. Wu, W. Song, J. Zhang, and Y. M. Chee (2023)
Learning feature embedding refiner for solving vehicle routing problems.
IEEE Transactions on Neural Networks and Learning Systems 35 (11), pp. 15279–15291.
Cited by: [§III-A1](https://arxiv.org/html/2602.21761v1#S3.SS1.SSS1.p4.1),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p2.1).
- [74]
K. Li, T. Zhang, R. Wang, W. Qin, H. He, and H. Huang (2021)
Research reviews of combinatorial optimization methods based on deep reinforcement learning.
Acta Automatica Sinica 47 (11), pp. 2521–2537.
Cited by: [TABLE I](https://arxiv.org/html/2602.21761v1#S1.T1.4.4.4.1.1),
[§I](https://arxiv.org/html/2602.21761v1#S1.p4.1).
- [75]
K. Li, F. Liu, Z. Wang, and Q. Zhang (2025)
Destroy and repair using hyper-graphs for routing.
In Proceedings of the AAAI Conference on Artificial Intelligence,
Vol. 39, pp. 18341–18349.
Cited by: [TABLE X](https://arxiv.org/html/2602.21761v1#A1.T10.1.19.1),
[¶IV-B1b](https://arxiv.org/html/2602.21761v1#S4.SS2.SSS1.P2.p2.1),
[TABLE V](https://arxiv.org/html/2602.21761v1#S4.T5.1.15.2),
[TABLE VII](https://arxiv.org/html/2602.21761v1#S6.T7.6.20.2),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p2.1).
- [76]
S. Li, Z. Yan, and C. Wu (2021)
Learning to delegate for large-scale vehicle routing.
Advances in Neural Information Processing Systems 34, pp. 26198–26211.
Cited by: [¶IV-B1b](https://arxiv.org/html/2602.21761v1#S4.SS2.SSS1.P2.p3.1),
[TABLE V](https://arxiv.org/html/2602.21761v1#S4.T5.1.22.6),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p3.1).
- [77]
Y. Li, L. Chen, H. Wang, R. Wang, and J. Yan (2025)
Generation as search operator for test-time scaling of diffusion-based combinatorial optimization.
In Advances in Neural Information Processing Systems,
Cited by: [TABLE X](https://arxiv.org/html/2602.21761v1#A1.T10.1.18.1),
[¶IV-B1a](https://arxiv.org/html/2602.21761v1#S4.SS2.SSS1.P1.p2.1),
[TABLE V](https://arxiv.org/html/2602.21761v1#S4.T5.1.10.6.1.1),
[§VI-D2](https://arxiv.org/html/2602.21761v1#S6.SS4.SSS2.p3.1),
[TABLE VII](https://arxiv.org/html/2602.21761v1#S6.T7.6.19.3).
- [78]
Y. Li, J. Guo, R. Wang, and J. Yan (2024)
T2T: from distribution learning in training to gradient search in testing for combinatorial optimization.
Advances in Neural Information Processing Systems 36.
Cited by: [§IV-B2](https://arxiv.org/html/2602.21761v1#S4.SS2.SSS2.p2.1),
[TABLE V](https://arxiv.org/html/2602.21761v1#S4.T5.1.27.6),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p3.1).
- [79]
Y. Li, J. Guo, R. Wang, H. Zha, and J. Yan (2024)
Fast t2t: optimization consistency speeds up diffusion-based training-to-testing solving for combinatorial optimization.
In Advances in Neural Information Processing Systems,
Cited by: [TABLE X](https://arxiv.org/html/2602.21761v1#A1.T10.1.20.1),
[§IV-B2](https://arxiv.org/html/2602.21761v1#S4.SS2.SSS2.p2.1),
[TABLE V](https://arxiv.org/html/2602.21761v1#S4.T5.1.28.1),
[TABLE VII](https://arxiv.org/html/2602.21761v1#S6.T7.6.21.3),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p3.1).
- [80]
X. Lin, Z. Yang, and Q. Zhang (2022)
Pareto set learning for neural multi-objective combinatorial optimization.
International Conference on Learning Representations.
Cited by: [TABLE II](https://arxiv.org/html/2602.21761v1#S3.T2.1.12.4).
- [81]
Z. Lin, Y. Wu, B. Zhou, Z. Cao, W. Song, Y. Zhang, and S. Jayavelu (2024)
Cross-problem learning for solving vehicle routing problems.
In International Joint Conference on Artificial Intelligence,
pp. 6958–6966.
Cited by: [§VII-B](https://arxiv.org/html/2602.21761v1#S7.SS2.p1.1).
- [82]
A. Lischka, J. Wu, R. Basso, M. H. Chehreghani, and B. Kulcsár (2024)
Less is more-on the importance of sparsification for transformers and graph neural networks for tsp.
arXiv preprint arXiv:2403.17159.
Cited by: [§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p3.1).
- [83]
F. Liu, X. Lin, Z. Wang, Q. Zhang, T. Xialiang, and M. Yuan (2024)
Multi-task learning for routing problem with cross-problem zero-shot generalization.
In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining,
pp. 1898–1908.
Cited by: [TABLE II](https://arxiv.org/html/2602.21761v1#S3.T2.1.25.4),
[§VII-B](https://arxiv.org/html/2602.21761v1#S7.SS2.p1.1).
- [84]
S. Liu, Y. Zhang, K. Tang, and X. Yao (2023)
How good is neural combinatorial optimization? a systematic evaluation on the traveling salesman problem.
IEEE Computational Intelligence Magazine 18 (3), pp. 14–28.
Cited by: [TABLE I](https://arxiv.org/html/2602.21761v1#S1.T1.4.4.4.1.1),
[§I](https://arxiv.org/html/2602.21761v1#S1.p4.1).
- [85]
S. Liu, Z. Cao, S. Feng, and Y. Ong (2025)
A mixed-curvature based pre-training paradigm for multi-task vehicle routing solver.
In International Conference on Machine Learning,
Cited by: [§VII-B](https://arxiv.org/html/2602.21761v1#S7.SS2.p1.1).
- [86]
H. R. Lourenço, O. C. Martin, and T. Stützle (2019)
Iterated local search: framework and applications.
Handbook of Metaheuristics, pp. 129–168.
Cited by: [¶IV-B1a](https://arxiv.org/html/2602.21761v1#S4.SS2.SSS1.P1.p3.1).
- [87]
H. Lu, X. Zhang, and S. Yang (2019)
A learning-based iterative method for solving vehicle routing problems.
In International Conference on Learning Representations,
Cited by: [¶IV-B1a](https://arxiv.org/html/2602.21761v1#S4.SS2.SSS1.P1.p2.1),
[TABLE V](https://arxiv.org/html/2602.21761v1#S4.T5.1.9.5.1.1).
- [88]
F. Luo, X. Lin, F. Liu, Q. Zhang, and Z. Wang (2023)
Neural combinatorial optimization with heavy decoder: toward large scale generalization.
Advances in Neural Information Processing Systems 36, pp. 8845–8864.
Cited by: [TABLE X](https://arxiv.org/html/2602.21761v1#A1.T10.1.6.1),
[§II-D1](https://arxiv.org/html/2602.21761v1#S2.SS4.SSS1.p1.1),
[§III-A1](https://arxiv.org/html/2602.21761v1#S3.SS1.SSS1.p3.1),
[TABLE II](https://arxiv.org/html/2602.21761v1#S3.T2.1.15.4),
[¶IV-B1b](https://arxiv.org/html/2602.21761v1#S4.SS2.SSS1.P2.p2.1),
[TABLE V](https://arxiv.org/html/2602.21761v1#S4.T5.1.11.6.1.1),
[TABLE VII](https://arxiv.org/html/2602.21761v1#S6.T7.1.1.1.1.1),
[TABLE VII](https://arxiv.org/html/2602.21761v1#S6.T7.5.5.1.1.1),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p2.1),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p3.1).
- [89]
F. Luo, X. Lin, Y. Wu, Z. Wang, T. Xialiang, M. Yuan, and Q. Zhang (2025)
Boosting neural combinatorial optimization for large-scale vehicle routing problems.
In International Conference on Learning Representations,
Cited by: [TABLE X](https://arxiv.org/html/2602.21761v1#A1.T10.1.7.1),
[TABLE II](https://arxiv.org/html/2602.21761v1#S3.T2.1.17.1),
[¶IV-B1b](https://arxiv.org/html/2602.21761v1#S4.SS2.SSS1.P2.p2.1),
[TABLE V](https://arxiv.org/html/2602.21761v1#S4.T5.1.13.2.1.1),
[3rd item](https://arxiv.org/html/2602.21761v1#S6.I6.i3.p1.1),
[¶VI-B2a](https://arxiv.org/html/2602.21761v1#S6.SS2.SSS2.P1.p1.1),
[§VI-D2](https://arxiv.org/html/2602.21761v1#S6.SS4.SSS2.p3.1),
[TABLE VII](https://arxiv.org/html/2602.21761v1#S6.T7.2.2.1.1.1),
[TABLE VII](https://arxiv.org/html/2602.21761v1#S6.T7.6.6.1),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p2.1),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p3.1).
- [90]
F. Luo, X. Lin, M. Zhong, F. Liu, Z. Wang, J. Sun, and Q. Zhang (2025)
Learning to insert for constructive neural vehicle routing solver.
Advances in Neural Information Processing Systems,.
Cited by: [TABLE X](https://arxiv.org/html/2602.21761v1#A1.T10.1.14.1),
[§III-A2](https://arxiv.org/html/2602.21761v1#S3.SS1.SSS2.p2.1),
[TABLE II](https://arxiv.org/html/2602.21761v1#S3.T2.1.38.4),
[¶IV-B1a](https://arxiv.org/html/2602.21761v1#S4.SS2.SSS1.P1.p2.1),
[TABLE V](https://arxiv.org/html/2602.21761v1#S4.T5.1.7.5),
[TABLE VII](https://arxiv.org/html/2602.21761v1#S6.T7.3.3.1.1.1),
[TABLE VII](https://arxiv.org/html/2602.21761v1#S6.T7.4.4.1.1.1).
- [91]
F. Luo, Y. Wu, Z. Zheng, and Z. Wang (2025)
Rethinking neural combinatorial optimization for vehicle routing problems with different constraint tightness degrees.
Advances in Neural Information Processing Systems,.
Cited by: [§VI-B1](https://arxiv.org/html/2602.21761v1#S6.SS2.SSS1.p1.1).
- [92]
Y. Ma, Z. Cao, and Y. M. Chee (2024)
Learning to search feasible and infeasible regions of routing problems with flexible neural k-opt.
Advances in Neural Information Processing Systems 36.
Cited by: [TABLE X](https://arxiv.org/html/2602.21761v1#A1.T10.1.17.1),
[§IV-A2](https://arxiv.org/html/2602.21761v1#S4.SS1.SSS2.p2.1),
[TABLE IV](https://arxiv.org/html/2602.21761v1#S4.T4.1.10.6),
[TABLE VII](https://arxiv.org/html/2602.21761v1#S6.T7.6.18.3).
- [93]
Y. Ma, J. Li, Z. Cao, W. Song, L. Zhang, Z. Chen, and J. Tang (2021)
Learning to iteratively solve routing problems with dual-aspect collaborative transformer.
Advances in Neural Information Processing Systems 34, pp. 11096–11107.
Cited by: [TABLE X](https://arxiv.org/html/2602.21761v1#A1.T10.1.16.1),
[§IV-A1](https://arxiv.org/html/2602.21761v1#S4.SS1.SSS1.p2.1),
[TABLE IV](https://arxiv.org/html/2602.21761v1#S4.T4.1.5.1),
[TABLE VII](https://arxiv.org/html/2602.21761v1#S6.T7.6.17.5).
- [94]
S. T. W. Mara, R. Norcahyo, P. Jodiawan, L. Lusiantoro, and A. P. Rifai (2022)
A survey of adaptive large neighborhood search algorithms and applications.
Computers & Operations Research 146, pp. 105903.
Cited by: [¶IV-B1a](https://arxiv.org/html/2602.21761v1#S4.SS2.SSS1.P1.p1.1).
- [95]
M. S. Martins, J. Sousa, and S. Vieira (2025)
A systematic review on reinforcement learning for industrial combinatorial optimization problems..
Applied Sciences 15 (3).
Cited by: [TABLE I](https://arxiv.org/html/2602.21761v1#S1.T1.3.3.4.1.1),
[§I](https://arxiv.org/html/2602.21761v1#S1.p4.1).
- [96]
V. R. Máximo, J. Cordeau, and M. C. Nascimento (2024)
AILS-ii: an adaptive iterated local search heuristic for the large-scale capacitated vehicle routing problem.
INFORMS Journal on Computing 36 (4), pp. 974–986.
Cited by: [TABLE X](https://arxiv.org/html/2602.21761v1#A1.T10.1.4.1),
[5th item](https://arxiv.org/html/2602.21761v1#S6.I1.i5.p1.1),
[¶VI-C2c](https://arxiv.org/html/2602.21761v1#S6.SS3.SSS2.P3.p1.1),
[§VI-D2](https://arxiv.org/html/2602.21761v1#S6.SS4.SSS2.p3.1).
- [97]
V. R. Máximo and M. C. Nascimento (2021)
A hybrid adaptive iterated local search with diversification control to the capacitated vehicle routing problem.
European Journal of Operational Research 294 (3), pp. 1108–1119.
Cited by: [§VI-D2](https://arxiv.org/html/2602.21761v1#S6.SS4.SSS2.p3.1).
- [98]
N. Mazyavkina, S. Sviridov, S. Ivanov, and E. Burnaev (2021)
Reinforcement learning for combinatorial optimization: a survey.
Computers & Operations Research 134, pp. 105400.
Cited by: [TABLE I](https://arxiv.org/html/2602.21761v1#S1.T1.4.4.4.1.1),
[§I](https://arxiv.org/html/2602.21761v1#S1.p4.1).
- [99]
Y. Min, Y. Bai, and C. P. Gomes (2024)
Unsupervised learning for solving the travelling salesman problem.
Advances in Neural Information Processing Systems 36.
Cited by: [§IV-A2](https://arxiv.org/html/2602.21761v1#S4.SS1.SSS2.p2.1),
[TABLE IV](https://arxiv.org/html/2602.21761v1#S4.T4.1.20.5),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p3.1).
- [100]
M. Nazari, A. Oroojlooy, L. Snyder, and M. Takác (2018)
Reinforcement learning for solving the vehicle routing problem.
Advances in Neural Information Processing Systems 31.
Cited by: [§III-A1](https://arxiv.org/html/2602.21761v1#S3.SS1.SSS1.p1.1).
- [101]
F. Neri, C. Cotta, and P. Moscato (2011)
Handbook of memetic algorithms.
Vol. 379, Springer.
Cited by: [§II-B2](https://arxiv.org/html/2602.21761v1#S2.SS2.SSS2.p1.1).
- [102]
X. Pan, Y. Jin, Y. Ding, M. Feng, L. Zhao, L. Song, and J. Bian (2023)
H-tsp: hierarchically solving the large-scale traveling salesman problem.
In Proceedings of the AAAI Conference on Artificial Intelligence,
Vol. 37, pp. 9345–9353.
Cited by: [TABLE X](https://arxiv.org/html/2602.21761v1#A1.T10.1.15.1),
[§III-B](https://arxiv.org/html/2602.21761v1#S3.SS2.p1.1),
[TABLE III](https://arxiv.org/html/2602.21761v1#S3.T3.1.3.6),
[TABLE VII](https://arxiv.org/html/2602.21761v1#S6.T7.6.16.4),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p3.1).
- [103]
B. Peng, J. Wang, and Z. Zhang (2019)
A deep reinforcement learning algorithm using dynamic attention model for vehicle routing problems.
In Artificial Intelligence Algorithms and Applications: 11th International Symposium,
pp. 636–650.
Cited by: [§III-A1](https://arxiv.org/html/2602.21761v1#S3.SS1.SSS1.p3.1),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p2.1).
- [104]
Y. Peng, B. Choi, and J. Xu (2021)
Graph learning for combinatorial optimization: a survey of state-of-the-art.
Data Science and Engineering 6 (2), pp. 119–141.
Cited by: [TABLE I](https://arxiv.org/html/2602.21761v1#S1.T1.3.3.4.1.1),
[§I](https://arxiv.org/html/2602.21761v1#S1.p4.1).
- [105]
P. H. V. Penna, A. Subramanian, and L. S. Ochi (2013)
An iterated local search heuristic for the heterogeneous fleet vehicle routing problem.
Journal of Heuristics 19 (2), pp. 201–232.
Cited by: [¶IV-B1a](https://arxiv.org/html/2602.21761v1#S4.SS2.SSS1.P1.p2.1).
- [106]
J. Pirnay and D. G. Grimm (2024)
Self-improvement for neural combinatorial optimization: sample without replacement, but improvement.
Transactions on Machine Learning Research.
Cited by: [¶IV-B1b](https://arxiv.org/html/2602.21761v1#S4.SS2.SSS1.P2.p2.1).
- [107]
D. Pisinger and S. Ropke (2018)
Large neighborhood search.
In Handbook of Metaheuristics,
pp. 99–127.
Cited by: [¶IV-B1a](https://arxiv.org/html/2602.21761v1#S4.SS2.SSS1.P1.p3.1).
- [108]
D. Pisinger and S. Ropke (2019)
Large neighborhood search.
Handbook of Metaheuristics, pp. 99–127.
Cited by: [§IV-B](https://arxiv.org/html/2602.21761v1#S4.SS2.p1.1).
- [109]
K. Price (2006)
Differential evolution: a practical approach to global optimization.
Springer Science & Business Media.
Cited by: [§V](https://arxiv.org/html/2602.21761v1#S5.p3.1).
- [110]
R. Qiu, Z. Sun, and Y. Yang (2022)
Dimes: a differentiable meta solver for combinatorial optimization problems.
Advances in Neural Information Processing Systems 35, pp. 25531–25546.
Cited by: [§III-A1](https://arxiv.org/html/2602.21761v1#S3.SS1.SSS1.p1.1),
[§III-A1](https://arxiv.org/html/2602.21761v1#S3.SS1.SSS1.p6.1),
[TABLE II](https://arxiv.org/html/2602.21761v1#S3.T2.1.34.3),
[§IV-A2](https://arxiv.org/html/2602.21761v1#S4.SS1.SSS2.p2.1),
[TABLE IV](https://arxiv.org/html/2602.21761v1#S4.T4.1.16.5.1.1),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p3.1).
- [111]
G. Reinelt (1991)
TSPLIB—a traveling salesman problem library.
ORSA Journal on Computing 3 (4), pp. 376–384.
Cited by: [1st item](https://arxiv.org/html/2602.21761v1#S6.I5.i1.p1.1).
- [112]
J. Renaud, F. F. Boctor, and J. Ouenniche (2000)
A heuristic for the pickup and delivery traveling salesman problem.
Computers & Operations Research 27 (9), pp. 905–916.
Cited by: [§II-B1](https://arxiv.org/html/2602.21761v1#S2.SS2.SSS1.p1.1).
- [113]
J. R. Rice (1976)
The algorithm selection problem.
In Advances in Computers,
Vol. 15, pp. 65–118.
Cited by: [§I](https://arxiv.org/html/2602.21761v1#S1.p2.1).
- [114]
R. Shahbazian, L. D. P. Pugliese, F. Guerriero, and G. Macrina (2024)
Integrating machine learning into vehicle routing problem: methods and applications.
IEEE Access 12 (), pp. 93087–93115.
Cited by: [TABLE I](https://arxiv.org/html/2602.21761v1#S1.T1.3.3.4.1.1),
[§I](https://arxiv.org/html/2602.21761v1#S1.p4.1).
- [115]
P. Shaw (1998)
Using constraint programming and local search methods to solve vehicle routing problems.
In International Conference on Principles and Practice of Constraint Programming,
pp. 417–431.
Cited by: [§IV-B](https://arxiv.org/html/2602.21761v1#S4.SS2.p1.1).
- [116]
J. Son, M. Kim, H. Kim, and J. Park (2023)
Meta-sage: scale meta-learning scheduled adaptation with guided exploration for mitigating scale shift on combinatorial optimization.
In International Conference on Machine Learning,
pp. 32194–32210.
Cited by: [§III-A1](https://arxiv.org/html/2602.21761v1#S3.SS1.SSS1.p4.1),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p2.1).
- [117]
T. Stützle (1999)
Local search algorithms for combinatorial problems: analysis, improvements, and new applications.
Cited by: [¶IV-B1a](https://arxiv.org/html/2602.21761v1#S4.SS2.SSS1.P1.p2.1).
- [118]
J. Sui, S. Ding, X. Huang, Y. Yu, R. Liu, B. Xia, Z. Ding, L. Xu, H. Zhang, C. Yu, et al. (2025)
A survey on deep learning-based algorithms for the traveling salesman problem.
Frontiers of Computer Science 19 (6), pp. 1–30.
Cited by: [TABLE I](https://arxiv.org/html/2602.21761v1#S1.T1.4.4.4.1.1),
[§I](https://arxiv.org/html/2602.21761v1#S1.p4.1).
- [119]
J. Sui, S. Ding, R. Liu, L. Xu, and D. Bu (2021)
Learning 3-opt heuristics for traveling salesman problem via deep reinforcement learning.
In Asian Conference on Machine Learning,
pp. 1301–1316.
Cited by: [§IV-A1](https://arxiv.org/html/2602.21761v1#S4.SS1.SSS1.p2.1),
[TABLE IV](https://arxiv.org/html/2602.21761v1#S4.T4.1.6.4).
- [120]
Y. Sun, X. Li, and A. Ernst (2021)
Using statistical measures and machine learning for graph reduction to solve maximum weight clique problems.
IEEE Transactions on Pattern Analysis and Machine Intelligence 43 (5), pp. 1746–1760.
Cited by: [§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p3.1).
- [121]
Z. Sun and Y. Yang (2023)
Difusco: graph-based diffusion solvers for combinatorial optimization.
Advances in Neural Information Processing Systems 36, pp. 3706–3731.
Cited by: [§III-A2](https://arxiv.org/html/2602.21761v1#S3.SS1.SSS2.p2.1),
[TABLE II](https://arxiv.org/html/2602.21761v1#S3.T2.1.41.3),
[§IV-A2](https://arxiv.org/html/2602.21761v1#S4.SS1.SSS2.p2.1),
[TABLE IV](https://arxiv.org/html/2602.21761v1#S4.T4.1.18.5.1.1),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p3.1).
- [122]
L. Tang, T. Li, Y. Meng, and J. Liu (2025)
Searching in symmetric solution space for permutation-related optimization problems.
IEEE Transactions on Pattern Analysis and Machine Intelligence 47 (8), pp. 7036–7052.
Cited by: [§I](https://arxiv.org/html/2602.21761v1#S1.p1.1).
- [123]
P. Tao and L. Chen (2025)
Combinatorial optimization: from deep learning to large language models.
Science China Mathematics 68, pp. 2519–2537.
Cited by: [TABLE I](https://arxiv.org/html/2602.21761v1#S1.T1.3.3.4.1.1),
[§I](https://arxiv.org/html/2602.21761v1#S1.p4.1).
- [124]
P. Toth and D. Vigo (2002)
The vehicle routing problem.
SIAM.
Cited by: [§II-A](https://arxiv.org/html/2602.21761v1#S2.SS1.p1.17),
[§II-A](https://arxiv.org/html/2602.21761v1#S2.SS1.p3.5),
[§II-B1](https://arxiv.org/html/2602.21761v1#S2.SS2.SSS1.p1.1).
- [125]
E. Uchoa, D. Pecin, A. Pessoa, M. Poggi, T. Vidal, and A. Subramanian (2017)
New benchmark instances for the capacitated vehicle routing problem.
European Journal of Operational Research 257 (3), pp. 845–858.
Cited by: [5th item](https://arxiv.org/html/2602.21761v1#S6.I5.i5.p1.1).
- [126]
M. Veres and M. Moussa (2019)
Deep learning for intelligent transportation systems: a survey of emerging trends.
IEEE Transactions on Intelligent transportation systems 21 (8), pp. 3152–3168.
Cited by: [TABLE I](https://arxiv.org/html/2602.21761v1#S1.T1.3.3.4.1.1),
[§I](https://arxiv.org/html/2602.21761v1#S1.p4.1).
- [127]
N. Vesselinova, R. Steinert, D. F. Perez-Ramirez, and M. Boman (2020)
Learning combinatorial optimization on graphs: a survey with applications to networking.
IEEE Access 8 (), pp. 120388–120416.
Cited by: [TABLE I](https://arxiv.org/html/2602.21761v1#S1.T1.3.3.4.1.1),
[§I](https://arxiv.org/html/2602.21761v1#S1.p4.1).
- [128]
T. Vidal, T. G. Crainic, M. Gendreau, and C. Prins (2013)
Heuristics for multi-attribute vehicle routing problems: a survey and synthesis.
European Journal of Operational Research 231 (1), pp. 1–21.
Cited by: [§I](https://arxiv.org/html/2602.21761v1#S1.p1.1),
[§II-B1](https://arxiv.org/html/2602.21761v1#S2.SS2.SSS1.p1.1),
[§II-B2](https://arxiv.org/html/2602.21761v1#S2.SS2.SSS2.p1.1).
- [129]
T. Vidal, T. G. Crainic, M. Gendreau, and C. Prins (2014)
A unified solution framework for multi-attribute vehicle routing problems.
European Journal of Operational Research 234 (3), pp. 658–673.
Cited by: [§V](https://arxiv.org/html/2602.21761v1#S5.p1.1),
[§V](https://arxiv.org/html/2602.21761v1#S5.p4.1).
- [130]
T. Vidal (2022)
Hybrid genetic search for the cvrp: open-source implementation and swap* neighborhood.
Computers & Operations Research 140, pp. 105643.
Cited by: [TABLE X](https://arxiv.org/html/2602.21761v1#A1.T10.1.3.1),
[4th item](https://arxiv.org/html/2602.21761v1#S6.I1.i4.p1.1).
- [131]
O. Vinyals, M. Fortunato, and N. Jaitly (2015)
Pointer networks.
Advances in Neural Information Processing Systems 28.
Cited by: [§II-D1](https://arxiv.org/html/2602.21761v1#S2.SS4.SSS1.p1.1),
[§III-A1](https://arxiv.org/html/2602.21761v1#S3.SS1.SSS1.p2.1),
[TABLE II](https://arxiv.org/html/2602.21761v1#S3.T2.1.3.6).
- [132]
C. Wang, Z. Yu, S. McAleer, T. Yu, and Y. Yang (2024)
Asp: learn a universal neural solver!.
IEEE Transactions on Pattern Analysis and Machine Intelligence 46 (6), pp. 4102–4114.
Cited by: [§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p3.1).
- [133]
F. Wang, Q. He, and S. Li (2024)
Solving combinatorial optimization problems with deep neural network: a survey.
Tsinghua Science and Technology 29 (5), pp. 1266–1282.
Cited by: [TABLE I](https://arxiv.org/html/2602.21761v1#S1.T1.4.4.4.1.1),
[§I](https://arxiv.org/html/2602.21761v1#S1.p4.1).
- [134]
Q. Wang and C. Tang (2021)
Deep reinforcement learning for transportation network combinatorial optimization: a survey.
Knowledge-Based Systems 233, pp. 107526.
Cited by: [TABLE I](https://arxiv.org/html/2602.21761v1#S1.T1.4.4.4.1.1),
[§I](https://arxiv.org/html/2602.21761v1#S1.p4.1).
- [135]
Y. Wang, Y. Jia, W. Chen, and Y. Mei (2025)
Distance-aware attention reshaping for enhancing generalization of neural solvers.
IEEE Transactions on Neural Networks and Learning Systems 36 (10), pp. 18900–18914.
Cited by: [§III-A1](https://arxiv.org/html/2602.21761v1#S3.SS1.SSS1.p4.1),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p2.1),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p3.1).
- [136]
X. Wu, D. Wang, L. Wen, Y. Xiao, C. Wu, Y. Wu, C. Yu, D. L. Maskell, and Y. Zhou (2024)
Neural combinatorial optimization algorithms for solving vehicle routing problems: a comprehensive survey with perspectives.
arXiv preprint arXiv:2406.00415.
Cited by: [TABLE I](https://arxiv.org/html/2602.21761v1#S1.T1.4.4.4.1.1),
[§I](https://arxiv.org/html/2602.21761v1#S1.p4.1).
- [137]
Y. Wu, W. Song, Z. Cao, J. Zhang, and A. Lim (2021)
Learning improvement heuristics for solving routing problems.
IEEE Transactions on Neural Networks and Learning Systems 33 (9), pp. 5057–5069.
Cited by: [§II-C1](https://arxiv.org/html/2602.21761v1#S2.SS3.SSS1.p1.1),
[§IV-A1](https://arxiv.org/html/2602.21761v1#S4.SS1.SSS1.p2.1),
[TABLE IV](https://arxiv.org/html/2602.21761v1#S4.T4.1.4.4).
- [138]
Y. Xia, X. Yang, Z. Liu, Z. Liu, L. Song, and J. Bian (2024)
Position: rethinking post-hoc search-based neural approaches for solving large-scale traveling salesman problems.
In International Conference on Machine Learning,
pp. 54178–54190.
Cited by: [§IV-A2](https://arxiv.org/html/2602.21761v1#S4.SS1.SSS2.p2.1),
[TABLE IV](https://arxiv.org/html/2602.21761v1#S4.T4.1.21.5).
- [139]
Y. Xiao, Y. Wu, R. Cao, D. Wang, Z. Cao, P. Zhao, Y. Li, Y. Zhou, and Y. Jiang (2025)
DGL: dynamic global-local information aggregation for scalable vrp generalization with self-improvement learning.
In International Joint Conference on Artificial Intelligence,
pp. 1–9.
Cited by: [TABLE X](https://arxiv.org/html/2602.21761v1#A1.T10.1.12.1),
[TABLE VII](https://arxiv.org/html/2602.21761v1#S6.T7.6.14.1).
- [140]
L. Xin, W. Song, Z. Cao, and J. Zhang (2020)
Step-wise deep learning models for solving routing problems.
IEEE Transactions on Industrial Informatics 17 (7), pp. 4861–4871.
Cited by: [§III-A1](https://arxiv.org/html/2602.21761v1#S3.SS1.SSS1.p3.1).
- [141]
L. Xin, W. Song, Z. Cao, and J. Zhang (2021)
Multi-decoder attention model with embedding glimpse for solving vehicle routing problems.
In Proceedings of the AAAI Conference on Artificial Intelligence,
Vol. 35, pp. 12042–12049.
Cited by: [TABLE II](https://arxiv.org/html/2602.21761v1#S3.T2.1.7.2),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p2.1).
- [142]
L. Xin, W. Song, Z. Cao, and J. Zhang (2021)
Neurolkh: combining deep learning model with lin-kernighan-helsgaun heuristic for solving the traveling salesman problem.
Advances in Neural Information Processing Systems 34, pp. 7472–7483.
Cited by: [§IV-A2](https://arxiv.org/html/2602.21761v1#S4.SS1.SSS2.p2.1),
[TABLE IV](https://arxiv.org/html/2602.21761v1#S4.T4.1.13.6).
- [143]
Z. Xing and S. Tu (2020)
A graph neural network assisted monte carlo tree search approach to traveling salesman problem.
IEEE Access 8, pp. 108418–108428.
Cited by: [§III-A1](https://arxiv.org/html/2602.21761v1#S3.SS1.SSS1.p1.1),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p3.1).
- [144]
H. Ye, J. Wang, Z. Cao, H. Liang, and Y. Li (2024)
DeepACO: neural-enhanced ant systems for combinatorial optimization.
Advances in Neural Information Processing Systems 36.
Cited by: [TABLE VI](https://arxiv.org/html/2602.21761v1#S4.T6.1.5.6),
[§V](https://arxiv.org/html/2602.21761v1#S5.p2.1).
- [145]
H. Ye, J. Wang, H. Liang, Z. Cao, Y. Li, and F. Li (2024)
Glop: learning global partition and local construction for solving large-scale routing problems in real-time.
In Proceedings of the AAAI Conference on Artificial Intelligence,
Vol. 38, pp. 20284–20292.
Cited by: [¶IV-B1b](https://arxiv.org/html/2602.21761v1#S4.SS2.SSS1.P2.p3.1),
[TABLE V](https://arxiv.org/html/2602.21761v1#S4.T5.1.25.5),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p3.1).
- [146]
C. Zhang, Y. Wu, Y. Ma, W. Song, Z. Le, Z. Cao, and J. Zhang (2023)
A review on learning to solve combinatorial optimisation problems in manufacturing.
IET Collaborative Intelligent Manufacturing 5 (1), pp. e12072.
Cited by: [TABLE I](https://arxiv.org/html/2602.21761v1#S1.T1.4.4.4.1.1),
[§I](https://arxiv.org/html/2602.21761v1#S1.p4.1).
- [147]
N. Zhang, J. Yang, Z. Cao, and X. Chi (2025)
Adversarial generative flow network for solving vehicle routing problems.
International Conference on Learning Representations.
Cited by: [TABLE II](https://arxiv.org/html/2602.21761v1#S3.T2.1.36.4).
- [148]
Y. Zheng, F. Luo, Z. Wang, Y. Wu, and Y. Zhou (2025)
MTL-kd: multi-task learning via knowledge distillation for generalizable neural vehicle routing solver.
Advances in Neural Information Processing Systems,.
Cited by: [§VII-B](https://arxiv.org/html/2602.21761v1#S7.SS2.p1.1).
- [149]
Z. Zheng, S. Yao, Z. Wang, X. Tong, M. Yuan, and K. Tang (2024)
Dpn: decoupling partition and navigation for neural solvers of min-max vehicle routing problems.
International Conference on Machine Learning.
Cited by: [TABLE II](https://arxiv.org/html/2602.21761v1#S3.T2.1.24.4).
- [150]
Z. Zheng, C. Zhou, T. Xialiang, M. Yuan, and Z. Wang (2024)
UDC: a unified neural divide-and-conquer framework for large-scale combinatorial optimization problems.
Advances in Neural Information Processing Systems 37, pp. 6081–6125.
Cited by: [¶IV-B1b](https://arxiv.org/html/2602.21761v1#S4.SS2.SSS1.P2.p3.1),
[TABLE V](https://arxiv.org/html/2602.21761v1#S4.T5.1.19.5),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p3.1).
- [151]
C. Zhou, X. Lin, Z. Wang, X. Tong, M. Yuan, and Q. Zhang (2024)
Instance-conditioned adaptation for large-scale generalization of neural combinatorial optimization.
arXiv preprint arXiv:2405.01906.
Cited by: [TABLE X](https://arxiv.org/html/2602.21761v1#A1.T10.1.8.1),
[§II-D2](https://arxiv.org/html/2602.21761v1#S2.SS4.SSS2.p1.1),
[§III-A1](https://arxiv.org/html/2602.21761v1#S3.SS1.SSS1.p4.1),
[TABLE II](https://arxiv.org/html/2602.21761v1#S3.T2.1.22.1),
[TABLE VII](https://arxiv.org/html/2602.21761v1#S6.T7.6.10.1),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p2.1),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p3.1).
- [152]
C. Zhou, X. Lin, Z. Wang, and Q. Zhang (2025)
Learning to reduce search space for generalizable neural routing solver.
arXiv preprint arXiv:2503.03137.
Cited by: [TABLE X](https://arxiv.org/html/2602.21761v1#A1.T10.1.11.1),
[§III-A1](https://arxiv.org/html/2602.21761v1#S3.SS1.SSS1.p4.1),
[TABLE II](https://arxiv.org/html/2602.21761v1#S3.T2.1.23.1),
[TABLE VII](https://arxiv.org/html/2602.21761v1#S6.T7.6.13.1),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p2.1),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p3.1).
- [153]
F. Zhou, A. Lischka, B. Kulcsar, J. Wu, M. H. Chehreghani, and G. Laporte (2025)
Learning for routing: a guided review of recent developments and future directions.
Transportation Research Part E: Logistics and Transportation Review 202, pp. 104278.
External Links: ISSN 1366-5545
Cited by: [TABLE I](https://arxiv.org/html/2602.21761v1#S1.T1.4.4.4.1.1),
[§I](https://arxiv.org/html/2602.21761v1#S1.p4.1).
- [154]
J. Zhou, Z. Cao, Y. Wu, W. Song, Y. Ma, J. Zhang, and C. Xu (2024)
MVMoE: multi-task vehicle routing solver with mixture-of-experts.
Proceedings of Machine Learning Research 235, pp. 61804–61824.
Cited by: [TABLE II](https://arxiv.org/html/2602.21761v1#S3.T2.1.26.2),
[§VII-B](https://arxiv.org/html/2602.21761v1#S7.SS2.p1.1).
- [155]
J. Zhou, Y. Wu, W. Song, Z. Cao, and J. Zhang (2023)
Towards omni-generalizable neural methods for vehicle routing problems.
In International Conference on Machine Learning,
pp. 42769–42789.
Cited by: [§VI-C1](https://arxiv.org/html/2602.21761v1#S6.SS3.SSS1.p1.1),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p3.1).
- [156]
Z. Zong, T. Feng, J. Wang, T. Xia, and Y. Li (2025)
Deep reinforcement learning for demand-driven services in logistics and transportation systems: a survey.
ACM Transactions on Knowledge Discovery from Data 19 (4), pp. 1–42.
Cited by: [TABLE I](https://arxiv.org/html/2602.21761v1#S1.T1.4.4.4.1.1),
[§I](https://arxiv.org/html/2602.21761v1#S1.p4.1).
- [157]
Z. Zong, H. Wang, J. Wang, M. Zheng, and Y. Li (2022)
Rbg: hierarchically solving large-scale routing problems in logistic systems via reinforcement learning.
In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining,
pp. 4648–4658.
Cited by: [¶IV-B1b](https://arxiv.org/html/2602.21761v1#S4.SS2.SSS1.P2.p3.1),
[TABLE V](https://arxiv.org/html/2602.21761v1#S4.T5.1.23.4),
[§VI-D2](https://arxiv.org/html/2602.21761v1#S6.SS4.SSS2.p3.1),
[§VII-A](https://arxiv.org/html/2602.21761v1#S7.SS1.p3.1).
- [158]
Z. Zong, M. Zheng, Y. Li, and D. Jin (2022)
Mapdp: cooperative multi-agent reinforcement learning to solve pickup and delivery problems.
In Proceedings of the AAAI Conference on Artificial Intelligence,
Vol. 36, pp. 9980–9988.
Cited by: [TABLE II](https://arxiv.org/html/2602.21761v1#S3.T2.1.11.4).

## Appendix A Experiment Details

### A-A Adopted Resources

The sources and possible licenses of the adopted methods and benchmark instances are summarized in Table [X](https://arxiv.org/html/2602.21761v1#A1.T10) and [XI](https://arxiv.org/html/2602.21761v1#A1.T11). All of them are open-sourced and available for academic use.

**TABLE X: Sources of the Adopted Methods**
| Method | Link | License |
| --- | --- | --- |
| LKH-3 [44] | [http://webhotel4.ruc.dk/~keld/research/LKH-3/](http://webhotel4.ruc.dk/~keld/research/LKH-3/) | Available for academic research use |
| HGS [130] | [https://github.com/vidalt/HGS-CVRP](https://github.com/vidalt/HGS-CVRP) | MIT License |
| AILS-II [96] | [https://github.com/INFORMSJoC/2023.0106](https://github.com/INFORMSJoC/2023.0106) | MIT License |
| BQ [27] | [https://github.com/naver/bq-nco](https://github.com/naver/bq-nco) | CC BY-NC-SA 4.0 license |
| LEHD [88] | [https://github.com/CIAM-Group/NCO_code/tree/main/single_objective/LEHD](https://github.com/CIAM-Group/NCO_code/tree/main/single_objective/LEHD) | MIT License |
| SIL [89] | [https://github.com/CIAM-Group/SIL](https://github.com/CIAM-Group/SIL) | MIT License |
| ICAM [151] | [https://github.com/CIAM-Group/ICAM](https://github.com/CIAM-Group/ICAM) | MIT License |
| ELG [33] | [https://github.com/gaocrr/ELG](https://github.com/gaocrr/ELG) | MIT License |
| INViT [28] | [https://github.com/Kasumigaoka-Utaha/INViT](https://github.com/Kasumigaoka-Utaha/INViT) | MIT License |
| L2R [152] | [https://github.com/CIAM-Group/L2R](https://github.com/CIAM-Group/L2R) | MIT License |
| DGL [139] | [https://github.com/wuyuesong/DGL](https://github.com/wuyuesong/DGL) | Available for academic research use |
| ReLD [50] | [https://github.com/ziweileonhuang/reld-nco](https://github.com/ziweileonhuang/reld-nco) | MIT License |
| L2C-Insert [90] | [https://github.com/CIAM-Group/L2C_Insert](https://github.com/CIAM-Group/L2C_Insert) | MIT License |
| H-TSP [102] | [https://github.com/Learning4Optimization-HUST/H-TSP](https://github.com/Learning4Optimization-HUST/H-TSP) | MIT License |
| DACT [93] | [https://github.com/yining043/VRP-DACT](https://github.com/yining043/VRP-DACT) | MIT License |
| NeuOpt [92] | [https://github.com/yining043/NeuOpt](https://github.com/yining043/NeuOpt) | MIT License |
| GenSCO [77] | [https://github.com/Thinklab-SJTU/GenSCO](https://github.com/Thinklab-SJTU/GenSCO) | Available for academic research use |
| DRHG [75] | [https://github.com/CIAM-Group/DRHG](https://github.com/CIAM-Group/DRHG) | Available for academic research use |
| FastT2T [79] | [https://github.com/Thinklab-SJTU/Fast-T2T](https://github.com/Thinklab-SJTU/Fast-T2T) | MIT license |
| GFACS [60] | [https://github.com/ai4co/gfacs](https://github.com/ai4co/gfacs) | MIT license |

**TABLE XI: Sources of the Adopted Benchmarks**
| Benchmark | Instance | BKS |
| --- | --- | --- |
| TSPLIB | [http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) | [http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/STSP.html](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/STSP.html) |
| National | [https://www.math.uwaterloo.ca/tsp/world/countries.html](https://www.math.uwaterloo.ca/tsp/world/countries.html) | [https://www.math.uwaterloo.ca/tsp/world/summary.html](https://www.math.uwaterloo.ca/tsp/world/summary.html) |
| VLSI | [https://www.math.uwaterloo.ca/tsp/vlsi/index.html](https://www.math.uwaterloo.ca/tsp/vlsi/index.html) | [https://www.math.uwaterloo.ca/tsp/vlsi/summary.html](https://www.math.uwaterloo.ca/tsp/vlsi/summary.html) |
| 8th DIMACS | [http://dimacs.rutgers.edu/archive/Challenges/TSP/download.html](http://dimacs.rutgers.edu/archive/Challenges/TSP/download.html) | [http://webhotel4.ruc.dk/~keld/research/LKH/DIMACS_results.html](http://webhotel4.ruc.dk/~keld/research/LKH/DIMACS_results.html) |
| Implementation Challenge | [http://dimacs.rutgers.edu/archive/Challenges/TSP/opts.html](http://dimacs.rutgers.edu/archive/Challenges/TSP/opts.html) |  |
| CVRPLIB | [https://galgos.inf.puc-rio.br/cvrplib/index.php/en/instances](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/instances) | Provided in the corresponding .vrp files of the instances. |

### A-B Detailed Results of the Proposed Pipeline

The detailed results of the proposed pipeline, including NRSs with all available models, are presented in Table [XII](https://arxiv.org/html/2602.21761v1#A1.T12) and Table [XIII](https://arxiv.org/html/2602.21761v1#A1.T13), respectively.

**TABLE XII: Detailed Experimental Results of the Proposed Evaluation Pipeline for TSP**
| Method | (0,1K) | [1K, 10K) | [10K, 100K] | Total |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gap | Time | Solved | Gap | Time | Solved | Gap | Time | Solved | Gap | Solved |  |
| Nearest Neighbor | 25.29% | 0.01s | 69/69 | 26.66% | 0.29s | 109/109 | 25.01% | 22.60s | 50/50 | 25.88% | 228/228 |
| Random Insertion | 10.60% | 0.00s | 69/69 | 15.32% | 0.05s | 109/109 | 16.37% | 8.93s | 50/50 | 14.12% | 228/228 |
| LKH-3^↓ t=n/3, runs=1 | 0.00% | 7.88s | 69/69 | 0.01% | 631.34s | 109/109 | 0.08% | 14600.50s | 50/50 | 0.03% | 228/228 |
| LKH-3^↓ t=n/3, runs=1 | 0.00% | 9.25s | 69/69 | 0.01% | 600.36s | 109/109 | 0.05% | 10800.24s | 50/50 | 0.02% | 228/228 |
| BQ | 5.00% | 2.51s | 68/69 | 19.03% | 22.74s | 92/109 | 52.00% | 187.81s | 4/50 | 14.02% | 164/228 |
| LEHD^∗ greedy | 4.85% | 1.01s | 69/69 | 20.13% | 68.27s | 106/109 | 49.35% | 1386.01s | 11/50 | 16.19% | 186/228 |
| SIL^∗ greedy (1K) | 6.07% | 1.72s | 69/69 | 10.29% | 17.31s | 106/109 | 20.71% | 434.58s | 47/50 | 11.18% | 222/228 |
| SIL^∗ greedy (5K) | 9.68% | 1.71s | 69/69 | 11.06% | 17.83s | 109/109 | 18.75% | 422.26s | 50/50 | 12.33% | 228/228 |
| SIL^∗ greedy (10K) | 6.11% | 1.72s | 69/69 | 9.99% | 17.87s | 109/109 | 20.85% | 414.78s | 50/50 | 11.20% | 228/228 |
| SIL^∗ greedy (50K) | 9.86% | 1.71s | 69/69 | 9.40% | 17.87s | 109/109 | 12.48% | 427.04s | 50/50 | 10.21% | 228/228 |
| SIL^∗ greedy (100K) | 8.64% | 1.69s | 69/69 | 9.83% | 17.69s | 109/109 | 11.11% | 430.73s | 50/50 | 9.75% | 228/228 |
| ICAM | 6.53% | 0.25s | 69/69 | 16.62% | 21.57s | 109/109 | 21.34% | 1050.33s | 19/50 | 13.54% | 197/228 |
| ELG | 6.05% | 0.63s | 69/69 | 18.14% | 88.12s | 108/109 | 21.65% | 940.35s | 6/50 | 13.70% | 183/228 |
| INViT-3V | 7.93% | 2.77s | 69/69 | 12.08% | 49.03s | 109/109 | 11.52% | 1079.50s | 42/50 | 10.67% | 220/228 |
| L2R | 5.89% | 1.60s | 69/69 | 9.22% | 15.55s | 109/109 | 8.52% | 153.11s | 50/50 | 8.06% | 228/228 |
| DGL | 6.53% | 1.17s | 69/69 | 11.32% | 11.67s | 109/109 | 11.14% | 58.62s | 25/50 | 9.67% | 203/228 |
| L2C-Insert^∗ greedy | 4.39% | 1.51s | 69/69 | 18.12% | 15.34s | 109/109 | 30.94% | 145.17s | 50/50 | 16.77% | 228/228 |
| H-TSP (1K) | 6.16% | 0.67s | 36/69 | 12.26% | 3.24s | 100/109 | 12.74% | 21.93s | 40/50 | 11.12% | 176/228 |
| H-TSP (2K) | 6.26% | 0.64s | 36/69 | 11.83% | 3.23s | 100/109 | 12.35% | 21.42s | 40/50 | 10.81% | 176/228 |
| H-TSP (5K) | 6.16% | 0.61s | 36/69 | 11.62% | 3.15s | 100/109 | 12.29% | 21.44s | 40/50 | 10.65% | 176/228 |
| H-TSP (10K) | 6.10% | 0.61s | 36/69 | 11.75% | 3.13s | 100/109 | 12.29% | 21.37s | 40/50 | 10.72% | 176/228 |
| DACT T=1K (20) | 24.54% | 39.47s | 69/69 | 26.85% | 260.55s | 83/109 | / | / | 0/50 | 25.80% | 152/228 |
| DACT T=1K (50) | 17.49% | 39.68s | 69/69 | 26.69% | 259.77s | 83/109 | / | / | 0/50 | 22.52% | 152/228 |
| DACT T=1K (100) | 16.37% | 39.84s | 69/69 | 26.58% | 261.73s | 83/109 | / | / | 0/50 | 21.94% | 152/228 |
| NeuOpt T=1K (20) | 35.76% | 533.78s | 7/69 | / | / | 0/109 | / | / | 0/50 | 35.76% | 7/228 |
| NeuOpt T=1K (50) | 20.70% | 138.26s | 27/69 | / | / | 0/109 | / | / | 0/50 | 20.70% | 27/228 |
| NeuOpt T=1K (100) | 12.44% | 104.05s | 36/69 | / | / | 0/109 | / | / | 0/50 | 12.44% | 36/228 |
| NeuOpt T=1K (200) | 19.90% | 81.22s | 46/69 | / | / | 0/109 | / | / | 0/50 | 19.90% | 46/228 |
| L2C-Insert^∗ T=1K | 1.08% | 381.55s | 69/69 | 9.80% | 479.98s | 109/109 | 29.15% | 615.19s | 50/50 | 11.41% | 228/228 |
| GenSCO (100) | 14.56% | 23.19s | 68/69 | 35.46% | 677.31s | 104/109 | 35.17% | 14304.70s | 25/50 | 28.21% | 197/228 |
| GenSCO (500) | 19.64% | 25.57s | 42/69 | 48.66% | 563.68s | 68/109 | 52.89% | 17964.48s | 14/50 | 39.31% | 124/228 |
| GenSCO (1K) | 24.11% | 27.76s | 29/69 | 19.27% | 268.60s | 53/109 | 90.42% | 24542.75s | 2/50 | 22.64% | 84/228 |
| LEHD^∗ RRC1K | 1.73% | 498.40s | 69/69 | 10.87% | 1634.51s | 109/109 | 24.02% | 2769.86s | 9/50 | 8.13% | 187/228 |
| SIL^∗ PRC1K (1K) | 0.63% | 903.72s | 69/69 | 2.89% | 3250.51s | 109/109 | 7.29% | 5366.67s | 50/50 | 3.17% | 228/228 |
| SIL^∗ PRC1K (5K) | 0.86% | 888.48s | 69/69 | 2.86% | 2889.38s | 109/109 | 6.85% | 5046.36s | 50/50 | 3.13% | 228/228 |
| SIL^∗ PRC1K (10K) | 0.55% | 891.35s | 69/69 | 2.63% | 2898.09s | 109/109 | 6.36% | 4962.70s | 50/50 | 2.82% | 228/228 |
| SIL^∗ PRC1K (50K) | 0.80% | 883.45s | 69/69 | 2.56% | 2879.65s | 109/109 | 5.11% | 5009.26s | 50/50 | 2.59% | 228/228 |
| SIL^∗ PRC1K (100K) | 0.80% | 883.87s | 69/69 | 2.58% | 2880.46s | 109/109 | 4.55% | 4933.45s | 50/50 | 2.47% | 228/228 |
| DRHG T=1K | 0.10% | 769.53s | 69/69 | 1.46% | 2857.55s | 109/109 | 4.46% | 3004.28s | 50/50 | 1.71% | 228/228 |
| Fast T2T Ts=10, Tg=10 (50) | 15.72% | 1.42s | 45/69 | / | / | 0/109 | / | / | 0/50 | 15.72% | 45/228 |
| Fast T2T Ts=10, Tg=10 (100) | 10.46% | 1.41s | 45/69 | / | / | 0/109 | / | / | 0/50 | 10.46% | 45/228 |
| Fast T2T Ts=10, Tg=10 (500) | 20.68% | 1.39s | 45/69 | / | / | 0/109 | / | / | 0/50 | 20.68% | 45/228 |
| Fast T2T Ts=10, Tg=10 (1K) | 18.55% | 1.40s | 45/69 | / | / | 0/109 | / | / | 0/50 | 18.55% | 45/228 |
| Fast T2T Ts=10, Tg=10 (10K) | 43.74% | 1.41s | 45/69 | / | / | 0/109 | / | / | 0/50 | 43.74% | 45/228 |
| GFACS^† T=100, K=100 (200) | 31.64% | 166.94s | 66/69 | 86.77% | 2601.13s | 22/109 | / | / | 0/50 | 45.42% | 88/228 |
| GFACS^† T=100, K=100 (500) | 35.06% | 175.46s | 64/69 | 86.43% | 2590.34s | 16/109 | / | / | 0/50 | 45.33% | 80/228 |
| GFACS^† T=100, K=100 (1K) | 42.05% | 175.18s | 64/69 | 84.88% | 2766.93s | 10/109 | / | / | 0/50 | 47.84% | 74/228 |
| GFACS^‡ T=100, K=100 (200) | 0.72% | 174.21s | 69/69 | 3.76% | 9142.93s | 83/109 | / | / | 0/50 | 2.38% | 152/228 |
| GFACS^‡ T=100, K=100 (500) | 0.80% | 209.26s | 69/69 | 3.31% | 6718.88s | 65/109 | / | / | 0/50 | 2.02% | 134/228 |
| GFACS^‡ T=100, K=100 (1K) | 0.93% | 225.39s | 69/69 | 3.63% | 7545.88s | 66/109 | / | / | 0/50 | 2.25% | 135/228 |

**TABLE XIII: Detailed Experimental Results of the Proposed Evaluation Pipeline for CVRP**
| Method | (0,1K) | [1K, 10K) | [10K, 100K] | Total |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gap | Time | Solved | Gap | Time | Solved | Gap | Time | Solved | Gap | Solved |  |
| Nearest Neighbor | 21.17% | 0.03s | 99/99 | 15.18% | 1.08s | 5/5 | 11.80% | 14.63s | 6/6 | 20.39% | 110/110 |
| Random Insertion | 75.00% | 0.00s | 36/99 | / | / | 0/5 | / | / | 0/6 | 75.00% | 36/110 |
| HGS t=n/3 | 0.29% | 111.24s | 99/99 | 3.59% | 1428.41s | 5/5 | 7.86% | 6926.41s | 6/6 | 0.85% | 110/110 |
| AILS-II t=n/3 | 0.57% | 133.95s | 99/99 | 1.58% | 1388.42s | 5/5 | 1.58% | 5646.48s | 6/6 | 0.68% | 110/110 |
| BQ | 8.87% | 3.63s | 99/99 | 20.28% | 39.97s | 5/5 | 41.52% | 202.69s | 5/6 | 10.89% | 109/110 |
| LEHD^∗ greedy | 11.25% | 1.53s | 98/99 | 19.22% | 99.43s | 5/5 | 32.80% | 852.02s | 2/6 | 12.04% | 105/110 |
| SIL^∗ greedy (1K) | 43.24% | 2.62s | 58/99 | 21.73% | 27.17s | 5/5 | 15.39% | 146.85s | 6/6 | 39.26% | 69/110 |
| SIL^∗ greedy (5K) | 42.70% | 2.49s | 62/99 | 18.34% | 26.60s | 5/5 | 12.38% | 147.91s | 6/6 | 38.54% | 73/110 |
| SIL^∗ greedy (10K) | 46.94% | 2.54s | 62/99 | 20.78% | 27.08s | 5/5 | 12.62% | 148.55s | 6/6 | 42.32% | 73/110 |
| SIL^∗ greedy (50K) | 40.04% | 2.48s | 65/99 | 16.09% | 26.86s | 5/5 | 10.81% | 146.83s | 6/6 | 36.16% | 76/110 |
| SIL^∗ greedy (100K) | 40.44% | 2.54s | 60/99 | 18.32% | 26.86s | 5/5 | 10.95% | 148.69s | 6/6 | 36.39% | 71/110 |
| ICAM | 5.00% | 0.42s | 99/99 | 11.69% | 32.32s | 5/5 | / | / | 0/6 | 5.32% | 104/110 |
| ELG | 8.03% | 1.29s | 99/99 | 18.51% | 30.21s | 5/5 | 29.38% | 133.08s | 2/6 | 8.93% | 106/110 |
| INViT-3V | 13.15% | 4.72s | 99/99 | 19.03% | 77.33s | 5/5 | 23.91% | 496.25s | 5/6 | 13.91% | 109/110 |
| L2R | 8.16% | 2.49s | 99/99 | 11.62% | 23.99s | 5/5 | 11.08% | 97.12s | 6/6 | 8.48% | 110/110 |
| DGL | 15.27% | 2.22s | 99/99 | 17.96% | 22.60s | 5/5 | 18.69% | 78.64s | 5/6 | 15.55% | 109/110 |
| ReLD | 4.10% | 0.41s | 99/99 | 10.22% | 5.29s | 5/5 | 11.27% | 28.87s | 3/6 | 4.58% | 107/110 |
| L2C-Insert^∗ greedy | 6.87% | 2.73s | 99/99 | 22.37% | 616.75s | 5/5 | 49.41% | 5525.71s | 2/6 | 8.40% | 106/110 |
| DACT T=1K (20) | 18.90% | 284.37s | 64/99 | / | / | 0/5 | / | / | 0/6 | 18.90% | 64/110 |
| DACT T=1K (50) | 16.42% | 338.69s | 54/99 | / | / | 0/5 | / | / | 0/6 | 16.42% | 54/110 |
| DACT T=1K (100) | 16.42% | 246.51s | 74/99 | 17.70% | 479.82s | 1/5 | / | / | 0/6 | 16.44% | 75/110 |
| NeuOpt T=1K (20) | 79.48% | 3440.62s | 6/99 | / | / | 0/5 | / | / | 0/6 | 79.48% | 6/110 |
| NeuOpt T=1K (50) | 74.85% | 2953.52s | 7/99 | / | / | 0/5 | / | / | 0/6 | 74.85% | 7/110 |
| NeuOpt T=1K (100) | 46.93% | 868.55s | 24/99 | / | / | 0/5 | / | / | 0/6 | 46.93% | 24/110 |
| NeuOpt T=1K (200) | 26.93% | 571.14s | 36/99 | / | / | 0/5 | / | / | 0/6 | 26.93% | 36/110 |
| L2C-Insert^∗ T=1K | 3.21% | 344.05s | 99/99 | 18.87% | 6166.21s | 5/5 | 44.29% | 32754.86s | 2/6 | 4.72% | 106/110 |
| LEHD^∗ RRC1K | 3.58% | 796.15s | 99/99 | 11.73% | 2043.74s | 5/5 | 21.98% | 2820.28s | 2/6 | 4.32% | 106/110 |
| SIL^∗ PRC1K (1K) | 23.67% | 1291.37s | 99/99 | 12.59% | 3405.06s | 5/5 | 11.46% | 4254.08s | 6/6 | 22.50% | 110/110 |
| SIL^∗ PRC1K (5K) | 21.58% | 1314.59s | 99/99 | 10.12% | 3487.80s | 5/5 | 8.60% | 4324.59s | 6/6 | 20.35% | 110/110 |
| SIL^∗ PRC1K (10K) | 22.16% | 1297.48s | 99/99 | 9.40% | 3442.22s | 5/5 | 8.54% | 4223.01s | 6/6 | 20.84% | 110/110 |
| SIL^∗ PRC1K (50K) | 21.38% | 1307.97s | 99/99 | 8.28% | 3471.69s | 5/5 | 7.40% | 4251.88s | 6/6 | 20.02% | 110/110 |
| SIL^∗ PRC1K (100K) | 22.24% | 1315.62s | 99/99 | 9.13% | 3471.46s | 5/5 | 8.02% | 4267.11s | 6/6 | 20.87% | 110/110 |
| DRHG T=1K | 11.11% | 1114.60s | 99/99 | 17.95% | 2529.12s | 5/5 | 16.95% | 5376.37s | 6/6 | 11.74% | 110/110 |
| GFACS^† T=100, K=100 (200) | 36.83% | 437.38s | 99/99 | 34.09% | 9654.33s | 3/5 | / | / | 0/6 | 36.75% | 102/110 |
| GFACS^† T=100, K=100 (500) | 61.11% | 606.05s | 78/99 | 86.40% | 1701.41s | 1/5 | / | / | 0/6 | 61.43% | 79/110 |
| GFACS^† T=100, K=100 (1K) | 67.68% | 476.95s | 64/99 | / | / | 0/5 | / | / | 0/6 | 67.68% | 64/110 |
| GFACS^‡ T=100, K=100 (200) | 2.60% | 405.81s | 99/99 | 7.65% | 14884.94s | 4/5 | / | / | 0/6 | 2.80% | 103/110 |
| GFACS^‡ T=100, K=100 (500) | 2.48% | 588.86s | 99/99 | 5.28% | 10935.70s | 3/5 | / | / | 0/6 | 2.56% | 102/110 |
| GFACS^‡ T=100, K=100 (1K) | 2.55% | 511.39s | 99/99 | 5.50% | 11283.58s | 3/5 | / | / | 0/6 | 2.63% | 102/110 |
