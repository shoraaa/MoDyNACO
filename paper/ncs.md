Here is the transcribed and properly formatted Markdown version of the provided PDF document.

# Efficient Neural Collaborative Search for Pickup and Delivery Problems

**Detian Kong, Yining Ma, Zhiguang Cao, Tianshu Yu, and Jianhua Xiao**

**Abstract**—In this paper, we first introduce an efficient Neural Neighborhood Search (N2S) approach for solving pickup and delivery problems (PDPs) by iteratively improving an initial solution. In N2S, we design a Synthesis Attention based encoder that allows the vanilla self-attention to synthesize various features regarding a route solution. We also exploit two customized decoders that automatically learn to perform removal and reinsertion of a pickup-delivery node pair to tackle the precedence constraint. Moreover, a diversity enhancement scheme is leveraged to ameliorate the performance during the inference. Then, on top of N2S, we propose the Neural Collaborative Search (NCS) approach which introduces a (lightweight) construction model as a complement to the *improvement* model (i.e. neighborhood search) in N2S, so as to boost the performance further. In NCS, the improvement model and the construction model are jointly trained mainly via a shared-critic mechanism. Meanwhile, the construction model strengthens the improvement model through upgraded curriculum learning, and the improvement model strengthens the construction model through imitation learning. Our N2S and NCS are both generic, and extensive experiments on two canonical PDP variants show that they can produce state-of-the-art results among existing neural methods. Remarkably, our N2S and NCS could surpass the well-known LKH3 solver on the more constrained PDP variant. Our code is available at: https://github.com/dtkon/PDP-NCS.

**Index Terms**—Learning to optimize, deep reinforcement learning, attention mechanism, pickup and delivery, neighborhood search

---

## 1 INTRODUCTION

EFFICIENT neighborhood search functions as a vital component of powerful heuristics for solving pickup and delivery problems (PDPs) [1]. Typically, it involves an iterative search process which transforms a solution into another candidate in its current neighborhood, hopefully in an efficient way. Designing the neighborhood and search rules usually determines the efficiency and even the success of a solver. However, they are often problem-specific and *manually engineered* with a lot of trial and error, which need to redesign when changes occur such as in constraints or objectives. These limitations may hinder the applications to the realistic problems from the rapidly evolving industries.

On the other hand, recent neural methods for vehicle routing problems (VRPs) have emerged as promising alternatives to traditional heuristics (e.g., [2]). They are usually faster, and more importantly, could automate the design of heuristics for new variants where no hand-crafted rule is available [3]. However, the prevailing neural methods mainly focus on travelling salesman problem (TSP) or capacitated vehicle routing problem (CVRP), where efficient neural solvers for PDPs are rarely studied. The PDP is ubiquitous in logistics, robotics, meal-delivery services, etc [1], which optimizes the route for pickup-delivery requests and is characterised by the precedence constraint (pickup before delivery). Despite the early attempt in [4], which learns a neural *construction* method to build a PDP solution (route) in seconds, it leaves a considerable gap to traditional heuristics in solution quality. To reduce the gap, we first propose an efficient *Neural Neighborhood Search* (N2S) approach for PDPs, based on a novel Transformer styled policy network with the encoder-decoder structure to iteratively improve an initial solution and a diversity enhancement scheme for more efficient inference. Then on top of N2S, we propose *Neural Collaborative Search* (NCS) to boost the performance further, by incorporating a *construction* model as the complement to the *improvement* model in N2S. The two models are jointly and collaboratively learned via a shared-critic mechanism, where the construction model provides more suitable initial solutions for the improvement model through an upgraded curriculum learning, and the improvement model provides high-quality solutions for the construction model to imitate.

**Neural Neighborhood Search (N2S).** To our knowledge, the most similar existing policy network to our N2S is the DACT in [2], both of which are essentially kind of *neural improvement* methods. DACT learns to encode the current solution and transform it into another one using the *2-opt*, *insert*, or *swap* decoder. Among them, the *2-opt* decoder, which considers reversing a segment of the solution (route), performed the best for TSP and CVRP. However, it is not suitable for PDPs since the precedence constraint can be easily violated by segment inversion. Though the *insert* decoder performed well on small-scale PDPs, which consider removing and reinserting a single node, its performance significantly drops on larger-scale or highly constrained PDPs (see Section 6). Conversely, our N2S tackles the precedence constraint more efficiently by allowing a pair of pickup-delivery nodes to be simultaneously operated in the neighborhood search through two customized *removal* and *reinsertion* decoders as shown in Fig. 1.

Another challenge lies in the design of the encoders. It was well revealed in [2] that the vanilla Transformer encoder [5] failed to correctly encode route solutions since the embeddings of node features (i.e., coordinates) and node positional features (i.e., node positions in a solution sequence) involve two different *aspects* of a route solution which are not directly compatible during encoding. They thus proposed the dual-aspect collaborative attention (DAC-Att) to learn dual representations for each feature aspect. In this paper, we propose a simple yet powerful *Synthesis Attention* (*Synth-Att*) where the attention scores from various types of node feature embeddings can be synthesized to attain a comprehensive and informative representation. It not only has the potential to encode more aspects than DAC-Att, but also reduces the computation costs while reserving competitive performance. Additionally, we design a diversity enhancement scheme to further ameliorate the performance. The proposed N2S was trained through the actor-critic based reinforcement learning [2], with a curriculum learning mechanism that relies on the policy network itself to increase the difficulty of exploration during training.

**Neural Collaborative Search (NCS).** In order to further unleash the power of N2S, we propose a Neural Collaborative Search (NCS) approach, which introduces a lightweight *construction* model (neural network) as a complement to the *improvement* model in N2S (see Fig. 5). In NCS, the two models are trained simultaneously and collaboratively via a shared-critic mechanism to promote each other. On the one hand, the construction model enhances the improvement model through an upgraded curriculum learning strategy that yields more suitable initial solutions as the starting search point of the improvement model during training, thus fostering its exploration ability and better search scope towards more optimal regions. On the other hand, the improvement model enhances the construction model through an imitation learning loss to accelerate the convergence, and its critic network is also shared to derive the critic of the construction model, saving much computation overhead. Finally, the best solution from the improvement model is retrieved as the eventual output.

Our proposed N2S and NCS are evaluated on two canonical problems in the PDP family, i.e., the pickup and delivery travelling salesman problem (PDTSP) and its variant with the last-in-first-out constraint (PDTSP-LIFO) to verify our design. Experimental results show that our N2S outperforms the state-of-the-art learning-based baselines, and becomes the *first* neural method to surpass the well-known LKH3 solver [6] when solving the synthesized PDP instances. Our NCS further strengthens such superiority and advantage by outstripping N2S.

The contributions of the paper are summarized as follows: 1) we present an efficient N2S approach for PDPs, which learns to perform removal and reinsertion of a pickup-delivery node pair automatically; 2) we propose Synth-Att, which allows the vanilla self-attention mechanism to synthesize node relationships from various feature embeddings in a simple way, and achieves superior expressiveness with much fewer computation costs in comparison to DAC-Att; 3) we explore a diversity enhancement scheme, which further makes our N2S the first neural method with almost no domain knowledge to surpass the LKH3 solver on the synthesized PDP instances (when there are no significant distribution shift w.r.t training ones); 4) we propose a neural collaborative search (NCS) approach with a construction model as the a complement to the improvement model in N2S, and further improve the performance over N2S. All these advantages highlight the potential of our proposed N2S and NCS as powerful neural approaches for solving new PDP variants without much need for manual trial and error.

The reminders of this paper are as follows. Section 2 briefly reviews various types of methods for VRPs including PDPs. Section 3 introduces the preliminary of the PDP and attention mechanism. Section 4 elaborates the MDP formulation of the problem, and the neural network structure and training algorithm of N2S. Section 5 presents the NCS approach, including the MDP formulation and policy network for the introduced construction model, and the collaborative training algorithm. The experimental results and analysis are given in Section 6. Finally, Section 7 summarizes our work and points out the future research direction.

## 2 RELATED WORK

### 2.1 Neural Methods for VRPs
We classify recent *neural* methods into *construction* and *improvement* ones. The *construction* methods, e.g., [7], [8], [9], learn a distribution of selecting nodes to autoregressively build solutions from scratch. Despite being fast, they lack abilities to search (near-)optimal solutions, even if armed with sampling (e.g., [9]), local search (e.g., [10]), or Monte-Carlo tree search (e.g., [11]). Among them, POMO [3] which explored diverse rollouts and data augments is recognized as the best construction method. Differently, *improvement* methods often hinge on a neighborhood search procedure such as *node swap* in [12], *ruin-and-repair* in [13], and *2-opt* in [14]. The work in [2] extended the Transformer styled model of [14] to Dual-Aspect Collaborative Transformer (DACT), and achieved state-of-the-art performance, which was also competitive to the *hybrid* neural methods, e.g., the ones combined with differential evolution [15] and dynamic programming [16]. Despite the success of the above methods for CVRP or TSP, they are not verified on the precedence-constrained PDPs. Though the first attempt was made in [4] to learn a construction solver for PDPs by introducing the heterogeneous attention to [9], the resulting solution qualities are still far from the optimality.

### 2.2 Neighbourhood Search for PDPs
Various heuristics based on neighborhood search have been proposed for PDPs. For PDTSP, the *k-interchange* neighborhood was studied in [17]. A *ruin-and-repair* neighborhood was later proposed in [18], and further extended in [19] with multiple perturbation methods. Aside from PDTSP, other PDP variants were also investigated, normally solved by designing new problem-specific neighborhoods [1], e.g., five neighborhoods were proposed in [20] to tackle the PDTSP with handling costs. Among them, the PDTSP-LIFO attracts much attention due to its largely constrained search space. To solve it, additional neighborhoods such as *double-bridge* and *shake* were introduced in [21]. Neighborhoods with tree structures were further proposed in [22]. Different from the above PDP solvers, the well-known LKH3 solver [6] combines neighborhood restriction strategies to achieve a more effective local search, which could solve various VRPs with superior performance. Recently, LKH3 was extended to tackle several PDP variants including PDTSP and PDTSP-LIFO, and delivered comparable performance to [19] and [22] on PDTSP and PDTSP-LIFO, respectively. Also given the open-sourced nature, we use LKH3 as the benchmark heuristic.

### 2.3 Collaborative and Hybrid Methods for VRPs
Recent studies highlight the potential of learning collaborative and hybrid methods for VRPs. The work in [23] proposed to combine the neural construction method with a traditional neighborhood search, which takes the output of the construction model as the initial solution for local search to yield the final VRP solution. The learning collaborative policies (LCP) in [10] considered dividing the search process into *seed stage* and *revise stage* for VRPs. Two different construction networks were employed to perform the respective tasks, with the initial solution constructed in the seed phase and (re)-constructed in the revised phase. Nevertheless, this method is still much less efficient than POMO aforementioned. In [24], a neural constructor and a neural perturbator were independently designed, which were directly stacked together during the inference. However, the above works failed to train the respective methods in an integrated and collaborative way, thus preventing effective information sharing and holding back the eventual performance. Furthermore, none of them is designed to tackle the pickup and delivery problems as studied in this paper.

## 3 PRELIMINARIES

### 3.1 PDP Formulation
We define the studied PDPs over a graph $G = (V, E)$, where nodes in $V = P \cup D \cup \{0\}$ represent locations and edges in $E$ represent connections between locations. With $n$ one-to-one pickup-delivery requests, an PDP instance contains $|V| = 2n + 1$ different locations, where node $0$ is depot, node set $P = \{1^+, 2^+, ..., n^+\}$ is referred to as pickup nodes, and node set $D = \{1^-, 2^-, ..., n^-\}$ is referred to as delivery nodes. Each pickup node $i^+$ has a number of goods to be transported to its pairing delivery node $i^-$. The objective is to find the shortest Hamiltonian cycle to fulfil all requests. In this paper, we consider two representative PDP variants, i.e., PDTSP and PDTSP-LIFO. The solution $\delta$ is defined as a cyclic sequence $(x_0, ..., x_{2n+1})$, where $x_0$ and $x_{2n+1}$ are the depot, and the rest is a permutation of nodes in $P \cup D$. For PDTSP, such permutation is under the *precedence* constraint that requires each pickup $i^+$ to be visited before its pairing delivery $i^-$. For PDTSP-LIFO, the *last-in-first-out* constraint is further imposed which requires loading and unloading to be executed in the corresponding order. This implies that unloading at a delivery node is allowed if and only if the goods is at the top of the stack. In Fig. 2, we present two example solutions with $n = 3$ and $|V| = 7$. The two solutions are both feasible to PDTSP, however, solution $\delta_2$ in Fig. 2(b) is infeasible to PDTSP-LIFO as the goods from $1^+$ is *NOT* at the top of the stack when it needs to be delivered at $1^-$.

### 3.2 Self-Attention Mechanism
The self-attention mechanism, initially presented in [5], allows the neural network to learn to capture the complex relationships among the elements in the input sequence. The rationale behind it is similar to the retrieval system, where a query is used to search for relevant information, and the returned results are the values associated with the matching keys. Formally, given an input sequence $\{x_1, ..., x_n\}$, the above query, key and value in self-attention can be computed as linear transformations of the input elements as follows,
$$ q_i = x_i W^Q, \; k_i = x_i W^K, \; v_i = x_i W^V, \quad (1) $$
where $W^Q \in \mathbb{R}^{d \times d_q}, W^K \in \mathbb{R}^{d \times d_k}, W^V \in \mathbb{R}^{d \times d_v}$ are trainable parameter matrices (usually $d_q = d_k = d_v$). The attention scores are then computed, normalized, and transformed into output as
$$ o_i = \sum_j \hat{s}_{i,j} v_j, \; \text{ where } \hat{s}_{i,j} = \text{Softmax}(\frac{q_i k_j^T}{\sqrt{d_k}}). \quad (2) $$
In multi-head attention (MHA), the attention mechanism is divided into multiple heads, which allows the model to capture different aspects of the input sequence simultaneously. The process of self-attention calculation is repeated for each head, with unique parameter matrices ($W_m^Q, W_m^K$, and $W_m^V$) assigned to head $m$. Given the output $o_{i,m}$ from each head, the eventual outputs of MHA are achieved by concatenating them together and then applying a linear transformation using a trainable matrix $W^O$ as
$$ O_i = \text{Concat}(o_{i,1}, o_{i,2}, ..., o_{i,m}) W^O . \quad (3) $$

## 4 NEURAL NEIGHBORHOOD SEARCH
We introduce our Neural Neighborhood Search (N2S) approach. The main component of N2S is an improvement model guided by a policy network that takes the current solution (or an initial solution), the features of a PDP instance, and the past actions as inputs, and then outputs an action to induce a search for a potentially better solution from the neighborhood. In this section, we first present the Markov Decision Process (MDP) formulation for N2S. Next, we elaborate on the design of the policy network. Finally, we give the training and inference algorithm.

### 4.1 MDP Formulation
We define the process of solving PDPs by our N2S as a Markov Decision Process $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma)$ as follows.

**State $\mathcal{S}$.** At time step $t$, the state is defined to include, 1) features of the current solution $\delta_t$, 2) action history, and 3) objective value of the best incumbent solution, i.e.,
$$ s_t = \{ \{l(x)\}_{x \in V}, \{p_t(x)\}_{x \in V}, \mathcal{H}(t, K), f(\delta_t^*) \} \, , \quad (4) $$
where $\delta_t$ is described from two aspects following [2]: $l(x)$ contains 2-dim coordinates of node $x$ (i.e., *node features*) and $p_t(x)$ indicates the index position of $x$ in $\delta_t$ (i.e., *node positional feature*); $\mathcal{H}(t, K)$ stores the most recent $K$ actions at time step $t$ if any; and $f(\cdot)$ denotes the objective function and $\delta_t^* = \arg\min_{\delta_\tau \in \{\delta_0, ..., \delta_t\}} f(\delta_\tau)$.

**Action $\mathcal{A}$.** With action $a_t = \{(i^+, i^-), (j, k)\}$ where $j, k \in V \setminus \{i^+, i^-\}$, the agent removes node pair $(i^+, i^-)$, and then reinserts node $i^+$ and $i^-$ after node $j$ and $k$, respectively.

**State Transition $\mathcal{T}$.** We utilize a deterministic transition rule to perform $a_t$. An example is illustrated in Fig. 1.

**Reward $\mathcal{R}$.** The reward function is defined as $r_t = f(\delta_t^*) - \min \left[ f(\delta_{t+1}), f(\delta_t^*) \right]$ which is the immediate reduced cost w.r.t. $f(\delta_t^*)$. The N2S agent aims to maximize the expected total reduced cost w.r.t. $\delta_0$ with a discount factor $\gamma < 1$.

### 4.2 Policy Network

#### 4.2.1 Encoder and Synth-Att
Given state $s = \{ \{l(x)\}_{x \in V}, \{p(x)\}_{x \in V}, \mathcal{H}(t, K), f(\delta^*) \}$, the N2S encoder takes $\{l(x)\}_{x \in V}$ and $\{p(x)\}_{x \in V}$ as inputs to learn embeddings for representing the current solution. Following DACT, we first project these raw features into two sets of embeddings, i.e., node feature embeddings (NFEs) $\{h_i\}_{i=0}^{|V|}$ and positional feature embeddings (PFEs) $\{g_i\}_{i=0}^{|V|}$. Different from [2], we treat NFEs as the primary set of embeddings whereas PFEs as auxiliary ones.

**NFEs.** We define $h_i$ as the linear projection of its node features $l(x_i)$ for any $x_i \in V$ with output dimension $d_h = 128$.

**PFEs.** By extending the absolute positional encoding in the vanilla Transformer [5], the cyclic positional encoding (CPE) was proposed in [2], which enables Transformer to encode cyclic sequences (as our PDP solutions) more accurately. The PFE $g_i$ with output dimension $d_g = 128$ is initialized by CPE as follows,
$$
g_i^{(d)} =
\begin{cases}
\sin(\omega_d \cdot (z(i) \bmod \frac{4\pi}{\omega_d}) - \frac{2\pi}{\omega_d}), \text{ if d is even} \\
\cos(\omega_d \cdot (z(i) \bmod \frac{4\pi}{\omega_d}) - \frac{2\pi}{\omega_d}), \text{ if d is odd}
\end{cases}
\quad (5)
$$
where superscript $d$ of $g_i^{(d)}$ refers to the $d$-th dimension of $g_i$, and the scalar $z(i)$ as well as the angular frequency $\omega_d = \frac{2\pi}{T_d}$ are defined according to Eq. (6) and Eq. (7), respectively.
$$ z(i) = \frac{i}{|V|} \frac{2\pi}{\omega_d} \left\lfloor \frac{|V|}{2\pi/\omega_d} \right\rfloor , \quad (6) $$
$$
T_d \!=\!\! \begin{cases}
\!\frac{3 \lfloor d/3 \rfloor + 1}{d_g} \!\left( |V| \!\!-\!\! |V|^{\frac{1}{\lfloor d_g/2 \rfloor}} \right) \!+\! |V|^{\frac{1}{\lfloor d_g/2 \rfloor}}, \! \text{if } \! d \! < \! \lfloor \frac{d_g}{2} \rfloor \\
\hfill |V|, \hfill \text{otherwise}
\end{cases} \quad (7)
$$
According to [2], directly fusing the two sets of embeddings (i.e., $h_i + g_i$) may cause undesired noises to the vanilla self-attention. As shown in Fig. 3(a), they thus proposed the DAC-Att in a way that each embedding set independently computes attention scores and shares them with the other one to learn dual-aspect representations. Different from it, we propose a simple and generic mechanism by incorporating a multilayer perception (MLP). As shown in Fig. 3(b), besides the original self-attention scores (the orange squares), multiple auxiliary attention scores learned from other feature embeddings (the blue squares) are leveraged and fed into an element-wise MLP, which allows it to synthesize heterogeneous attention relationships into comprehensive ones. We call it **Synthesis Attention (Synth-Att)**. It is able to not only leverage more attention scores from various feature embeddings, but also achieve competitive performance to DAC-Att with less computation costs. Below, we present more details.

**Auxiliary Attention Scores.** In our N2S, PFEs are used to generate multi-head auxiliary attention scores as follows,
$$ \alpha_{i,j,m}^{\text{aux}} = \left( g_i W_m^{Q_{\text{aux}}} \right) \left( g_j W_m^{K_{\text{aux}}} \right)^T, \quad (8) $$
where $W_m^{Q_{\text{aux}}} \in \mathbb{R}^{d_g \times d_q}, W_m^{K_{\text{aux}}} \in \mathbb{R}^{d_g \times d_k}$ are trainable matrices for each head $m$. We set $m = 4$ and $d_q = d_k = d_g/m$.

**Syn-Att.** The Syn-Att is defined as follows,
$$ \tilde{h}_i = \textbf{Syn-Att}(W^Q, W^K, W^V, W^O, \text{MLP}). \quad (9) $$
In specific, it first computes the multi-head self-attention scores $\alpha_{i,j,m}^{\text{self}}$ for NFEs based on the trainable matrices $W_m^Q \in \mathbb{R}^{d_h \times d_q}$ and $W_m^K \in \mathbb{R}^{d_h \times d_k}$ for head $m$ using Eq. (10) as
$$ \alpha_{i,j,m}^{\text{self}} = \left( h_i W_m^Q \right) \left( h_j W_m^K \right)^T. \quad (10) $$
Thereafter, the attention scores $\alpha^{\text{aux}}$ and $\alpha^{\text{self}}$ are fed into a three-layer MLP with structure $(2m \times 2m \times m)$ to compute the synthesized multi-head attention scores as follows,
$$ \alpha_{i,j,1}^{\text{Synth}}, ..., \alpha_{i,j,m}^{\text{Synth}} = \!\text{MLP}\!\left( \alpha_{i,j,1}^{\text{self}}, ..., \alpha_{i,j,m}^{\text{self}}, \alpha_{i,j,1}^{\text{aux}}, ..., \alpha_{i,j,m}^{\text{aux}} \right). \quad (11) $$
The scores are then normalized to $\tilde{\alpha}_{i,j,m}$ through Softmax, which are further used to calculate the attention values for each head as Eq. (12). Finally, the outputs are given by Eq. (13) with trainable matrix $W^O \in \mathbb{R}^{md_v \times d_h} \; (d_v = d_h/m)$.
$$ \text{head}_{i,m} = \sum_{j=1}^{|V|} \tilde{\alpha}_{i,j,m} \left( h_j W_m^V \right), \quad (12) $$
$$ \tilde{h}_i = \text{Concat}\left[ \text{head}_{i,1}, ..., \text{head}_{i,m} \right] W^O. \quad (13) $$

**N2S Encoder.** We stack $L$ ($L=3$) encoders, each of which is the same as the Transformer encoder, except that the vanilla multi-head self-attention is replaced with our multi-head Synth-Att and we use the same instance normalization layer as [2]. Note that the auxiliary attention scores $\alpha_{i,j,m}^{\text{aux}}$ are only computed once and shared among all stacked encoders to reduce computation costs.

#### 4.2.2 Decoder
The N2S decoder first adopts the max-pooling layer in [14] to aggregate the global representation of all embeddings into each individual one as follows,
$$ \hat{h}_i = \tilde{h}_i^{(L)} W_h^{\text{Local}} + \max \left[ \{ \tilde{h}_i^{(L)} \}_{i=1}^{|V|} \right] W_h^{\text{Global}}. \quad (14) $$

**Node-Pair Removal Decoder.** Given the enhanced embeddings $\{ \hat{h}_i \}_{i=1}^{|V|}$ and the set $\mathcal{H}(t, K)$, the *removal* decoder outputs a categorical distribution over $n$ requests for removal action. In specific, it first computes a score $\lambda_i$ for each $x_i \in V$ indicating the closeness between node $x_i$ and its neighbors as
$$
\lambda_i = (\hat{h}_{\text{pred}(x_i)} W_\lambda^Q)(\hat{h}_i W_\lambda^K)^T + (\hat{h}_i W_\lambda^Q)(\hat{h}_{\text{succ}(x_i)} W_\lambda^K)^T \\
- (\hat{h}_{\text{pred}(x_i)} W_\lambda^Q)(\hat{h}_{\text{succ}(x_i)} W_\lambda^K)^T, \quad (15)
$$
where $\text{pred}(x_i)$ and $\text{succ}(x_i)$ refer to the first predecessor and the second successor nodes of $x_i$ (the second successor works better empirically), respectively, and $W_\lambda^Q \in \mathbb{R}^{d_h \times d_h}, W_\lambda^K \in \mathbb{R}^{d_h \times d_h}$. We use multi-head technique to obtain $\lambda_{i,1}$ to $\lambda_{i,m}$. Then the decoder aggregates the scores for each pickup-delivery pair $(i^+, i^-)$ based on a three-layer $\text{MLP}_\lambda$,
$$
\tilde{\Lambda}_{(i^+, i^-)} = \text{MLP}_\lambda(\lambda_{i^+,1}, ..., \lambda_{i^+,m}, \lambda_{i^-,1}, ..., \lambda_{i^-,m}, \\
c(i), \mathbf{1}_{\text{last}(1)=i}, \mathbf{1}_{\text{last}(2)=i}, \mathbf{1}_{\text{last}(3)=i}), \quad (16)
$$
where the MLP structure is $(2m+4, 32, 32, 1)$, scalar $c(i)$ counts the frequency of request $(i^+, i^-)$ being selected for removal in the past $K$ steps, and $\mathbf{1}_{\text{last}(t)=i'}$ is a binary variable indicating whether request $i'$ was selected at the $t$-th last step. An activation layer $\hat{\Lambda} = C \cdot \text{Tanh}(\tilde{\Lambda})$ is then applied ($C = 6$), followed by Softmax to normalize the distribution which is then used to sample a node pair $(i^+, i^-)$ as the removal action.

**Node-Pair Reinsertion Decoder.** Given a request $(i^+, i^-)$ for removal, the *reinsertion* decoder outputs the joint distribution that reinserts the two nodes back to the solution. We first define two scores $\mu^p(x_\alpha, x_\beta)$ and $\mu^s(x_\alpha, x_\beta)$ for a node $x_\alpha$ indicating the degree of preference of accepting a node $x_\beta$ as its new predecessor and successor nodes, respectively,
$$
\mu^p[x_\alpha, x_\beta] = (\hat{h}_\alpha W_\mu^{Q_p})(\hat{h}_\beta W_\mu^{K_p})^T, \\
\mu^s[x_\alpha, x_\beta] = (\hat{h}_\alpha W_\mu^{Q_s})(\hat{h}_\beta W_\mu^{K_s})^T, \quad (17)
$$
where $W_\mu^{Q_p}, W_\mu^{Q_s} \in \mathbb{R}^{d_h \times d_h}$, and $W_\mu^{K_p}, W_\mu^{K_s} \in \mathbb{R}^{d_h \times d_h}$. Again, we use multiple heads. Based on the scores, the decoder predicts the distribution of reinserting node $i^+$ after node $j$, and reinserting node $i^-$ after node $k$ using $\text{MLP}_\mu$,
$$
\tilde{\mu}[j, k] = \text{MLP}_\mu(\mu_1^p[i^+, \text{succ}(j)], ..., \mu_m^p[i^+, \text{succ}(j)], \\
\mu_1^p[i^-, \text{succ}(k)], ..., \mu_m^p[i^-, \text{succ}(k)], \quad (18) \\
\mu_1^s[i^+, j], ..., \mu_m^s[i^+, j], \mu_1^s[i^-, k], ..., \mu_m^s[i^-, k]),
$$
where the MLP structure is $(4m, 32, 32, 1)$. Note that here $\text{pred}(\cdot)$ and $\text{succ}(\cdot)$ should be considered in the new solution where nodes $i^+, i^-$ have already been removed. Afterwards, $\hat{\mu} = C \cdot \text{Tanh}(\tilde{\mu})$ is applied and infeasible choices are masked as $-\infty$ before normalizing by Softmax. Finally, a node pair $(j, k)$, as the *reinsertion* action, is sampled according to the resulting distribution to indicate the positions of reinserting the node-pair $(i^-, i^+)$ back to the solution.

---
**Algorithm 1 n-step PPO with CL strategy**
**Input:** policy $\pi_\theta$, critic $v_\phi$, PPO clipping threshold $\varepsilon$, learning rate $\eta_\theta, \eta_\phi$, learning rate decay $\beta$, epochs $E$, batches $B$, mini-batch $\kappa$, training steps $T_{\text{train}}$, CL scalar $\rho$
1: **for** $e = 1$ to $E$ **do**
2:     **for** $b = 1$ to $B$ **do**
3:         Generate training data $\mathcal{D}_b$ on the fly;
4:         Initialize random solutions $\{\delta_{-1}\}$ to $\mathcal{D}_b$;
5:         Improve $\{\delta_{-1}\}$ to $\{\delta_0\}$ via $\pi_\theta$ for $T = e/\rho$ steps;
6:         Set initial state $s_0$ based on $\{\delta_0\}$ and Eq. (4); $t \leftarrow 0$;
7:         **while** $t < T_{\text{train}}$ **do**
8:             Get $\{(s_\tau, a_\tau, r_\tau)\}_{\tau=t}^{t+n-1}$ where $a_\tau \sim \pi_\theta(a_\tau | s_\tau)$;
9:             $t \leftarrow t + n$, $\pi_{\text{old}} \leftarrow \pi_\theta$, $v_{\text{old}} \leftarrow v_\phi$;
10:            **for** $k = 1$ to $\kappa$ **do**
11:                $\hat{R}_t = v_\phi(s_t)$;
12:                **for** $\tau \in \{t - 1, ..., t - n\}$ **do**
13:                    $\hat{R}_\tau \leftarrow r_\tau + \gamma \hat{R}_{\tau+1}$;
14:                    $\hat{A}_\tau \leftarrow \hat{R}_\tau - v_\phi(s_\tau)$;
15:                **end for**
16:                Compute RL loss $J_{RL}(\theta)$ using Eq. (19) and clipped critic loss $L_{BL}(\phi)$ using Eq. (20);
17:                $\theta \leftarrow \theta + \eta_\theta \nabla J_{RL}(\theta)$;
18:                $\phi \leftarrow \phi - \eta_\phi \nabla L_{BL}(\phi)$;
19:            **end for**
20:        **end while**
21:    **end for**
22:    $\eta_\theta \leftarrow \beta \eta_\theta$, $\eta_\phi \leftarrow \beta \eta_\phi$;
23: **end for**
---

### 4.3 Training Algorithm
The training algorithm for N2S is the same as [2], namely the proximal policy optimization (PPO), where it learns a policy $\pi_\theta$ (our N2S) with the help of a critic $v_\phi$ and leverages a curriculum learning (CL) strategy for better convergence. Specifically, as presented in Algorithm 1, we train N2S for $E$ epochs and $B$ batches per epoch, where the training dataset $\mathcal{D}_b$ is randomly generated on the fly with uniform distribution (line 3). The initial states (solutions) for training are yielded by the CL strategy in each training batch (lines 4-6). Then we follow the original n-step PPO to train our N2S, where we present the reinforcement learning loss function in Eq. (19), and the baseline loss function of the critic in Eq. (20), respectively. Here, we clip the estimated value $\hat{v}(s_t)$ around the previous value estimates using $v_\phi^{\text{clip}}(s_\tau) = \text{clip}\left[ v_\phi(s_\tau), v_{\text{old}}(s_\tau) - \varepsilon, v_{\text{old}}(s_\tau) + \varepsilon \right]$. Below we present details of the critic network and the employed CL strategy.

$$
J_{RL}(\theta) = \frac{1}{n|\mathcal{D}_b|} \sum_{\mathcal{D}_b} \sum_{\tau=t}^{t+n} \min \left( \frac{\pi_\theta(a_\tau | s_\tau)}{\pi_{\text{old}}(a_\tau | s_\tau)} \hat{A}_\tau, \right. \nonumber \\
\left. \text{clip}\left[ \frac{\pi_\theta(a_\tau | s_\tau)}{\pi_{\text{old}}(a_\tau | s_\tau)}, 1 - \varepsilon, 1 + \varepsilon \right] \hat{A}_\tau \right), \quad (19)
$$
$$
L_{BL}(\phi) = \frac{1}{n|\mathcal{D}_b|} \sum_{\mathcal{D}_b} \sum_{\tau=t}^{t+n} \max \left( \left| v_\phi(s_\tau) - \hat{R}_\tau \right| , \right. \nonumber \\
\left. \left| v_\phi^{\text{clip}}(s_\tau) - \hat{R}_\tau \right| \right)^2. \quad (20)
$$

---
**Algorithm 2 N2S-A Inference**
**Input:** Instance $\mathcal{I}$ with size $|V|$, policy $\pi_\theta$, maximum step $T$
1: **for** $i = 1, ..., \lfloor \frac{1}{2} |V| \rfloor$ **do**
2:     $\mathcal{I}_i \leftarrow \mathcal{I}$;
3:     $\mathcal{A}_i \leftarrow \textbf{RandomShuffle}([\text{flip-x-y, 1-x, 1-y, rotate}])$;
4:     **for** each augment method $j \in \mathcal{A}_i$ **do**
5:         $\varrho_j \leftarrow \textbf{RandomConfig}(j)$;
6:         $\mathcal{I}_i \leftarrow \text{perform augment } j \text{ on } \mathcal{I}_i \text{ with config } \varrho_j$;
7:     **end for**
8: **end for**
9: Solve all instances and $\mathcal{I}_i$ in parallel with $\pi_\theta$ for $T$ steps;
10: **return** the best solution found among all $\mathcal{I}_i$
---

**Critic Network.** Given the embeddings $\{ \tilde{h}_i^{(L)} \}_{i=0}^{|V|}$ from policy network $\pi_\theta$, the critic network first enhances them by a vanilla multi-head attention layer (with $m = 4$ heads) to get $\{y_i\}_{i=0}^{|V|}$. The enhanced embeddings are then fed into a mean-pooling layer [14] to aggregate the global representation of all embeddings into each individual one as
$$ \hat{y}_i = y_i W_v^{\text{Local}} + \text{mean} \left[ \{y_i\}_{i=1}^{|V|} \right] W_v^{\text{Global}}, \quad (21) $$
where we use trainable matrices $W_v^{\text{Local}}, W_v^{\text{Global}} \in \mathbb{R}^{d_h \times \frac{d_h}{2}}$. Lastly, the state value $v(s_t)$ is output by a four-layer MLP in Eq. (22) with structure $(129, 128, 64, 1)$. Here, $f(\delta^*)$ is added as an additional input to $\text{MLP}_v$.
$$ v(s_t) = \text{MLP}_v \left( \max \left[ \{\hat{y}_i\}_{i=1}^{|V|} \right], \text{mean} \left[ \{\hat{y}_i\}_{i=1}^{|V|} \right], f(\delta^*) \right). \quad (22) $$

**Curriculum Learning.** Following [2] and [25], we leverage the curriculum learning (CL) strategy (in lines 4 to 6) to ameliorate the training further. In specific, it improves the randomly generated solutions $\{\delta_{-1}\}$ to $\{\delta_0\}$ by running the current policy $\pi_\theta$ for $T = e/\rho$ steps, which means that we exploit the evolved policy itself to yield better and better initial solutions as the training progresses. The improved solutions $\{\delta_0\}$ with higher quality (thus harder to improve) are used to initialize the first state $s_0$. In such a way, the hardness level of neighborhood search is gradually increased per epoch in the context of CL.

### 4.4 Diversity Enhancement during Inference
To be more resistant to local minima, we further equip our N2S with an augmentation-based inference scheme, which leads to N2S-A in Algorithm 2. Such idea was originally explored in [3] for a neural *construction* method, and we extend it to an *improvement* one. The rationale is that an instance $\mathcal{I}$ can be transformed into different ones for searching while reserving the same optimal solution, e.g., rotating all locations of nodes by $\pi/2$ radian. For an instance of size $|V|$, our N2S-A performs $\lfloor \frac{1}{2}|V| \rfloor$ augments, each of which is generated by sequentially applying four preset invariant transformation operations with different orders and different configurations as listed in Table 1. Note that although the mentioned transformations are conducted on instances defined in the Euclidean space, we believe that such an idea has favourable potential to be also exploited in non-Euclidean space, as long as there are proper invariant transformations for coordinates in the target space. Meanwhile, we use $K = |V|$ for training and $\hat{K} = \lfloor \frac{1}{2}|V| \rfloor$ for inference. This is because we found that when a specific $K$ in $\mathcal{H}(t, K)$ is used for training, a smaller $K$ during inference can improve the diversity of the solutions, thus better eventual performance.

**TABLE 1: Descriptions of the four invariant transformations.**

| Transformations | Formulations | Configurations |
| :--- | :--- | :--- |
| flip-x-y | $(\hat{x}, \hat{y}) = (y, x)$ | perform or skip |
| 1-x | $(\hat{x}, \hat{y}) = (1 - x, y)$ | perform or skip |
| 1-y | $(\hat{x}, \hat{y}) = (x, 1 - y)$ | perform or skip |
| rotate | $\begin{pmatrix} \hat{x} \\ \hat{y} \end{pmatrix} = \begin{pmatrix} x\cos\vartheta - y\sin\vartheta \\ x\sin\vartheta + y\cos\vartheta \end{pmatrix}$ | $\vartheta \in \{0, \pi/2, \pi, 3\pi/2\}$ |


## 5 NEURAL COLLABORATIVE SEARCH
As established, the N2S approach exploits an improvement model to search for better solutions with a self-dependent curriculum learning strategy. To further unleash the potential, in this section, we propose the Neural Collaborative Search (NCS) approach on top of N2S. In NCS, a lightweight construction model is introduced to complement the improvement model in N2S. The two models work collaboratively to solve the PDPs, with the construction model yielding the initial solution that the improvement model iteratively refines.

To achieve such collaboration, an intuitive way is to directly connect the two models that are trained independently. However, it would be far from optimal in doing so. On the one hand, for the improvement model that starts searching from a random initial solution, it may perform inferior to search from the solution yielded by the construction model. On the other hand, the solutions yielded by the improvement model that might be valuable for the construction model, could not be fully used. In our NCS, we build a bridge between the construction model and improvement model to achieve collaborative training through a shared-critic mechanism, where they could naturally further promote each other with an upgraded curriculum learning strategy and an imitation learning loss function, respectively. The basic rationale of our NCS is illustrated in Fig. 5.

In the following, we first define the MDP for the construction model, then present the design of its policy network, and finally elaborate on how to train the construction model and the improvement model collaboratively in NCS. Note that to save notations or symbols, some of them for NCS may be the same as those used for N2S, e.g., we have used $s$ to denote the state of the improvement model in N2S and we continue to use it to denote the state of the construction model in NCS.

### 5.1 MDP Formulation
**State $\mathcal{S}$.** The state $s_t$ represents the instance information and the partial solution constructed up to time step $t$. The former includes the coordinates of the depot and all customer nodes, denoted as $l(x)$ with $x \in V$, and the latter includes the set of nodes visited till time step $t$-1 and the one visited exactly at time step $t$-1. At time step 0, there is no last visited node, so $s_0 = \{l(x)\}_{x \in V}$.
**Action $\mathcal{A}$.** The action $a_t$ of the construction model represents the next node to visit where the infeasible nodes will be filtered out.
**State Transition $\mathcal{T}$.** Based on $a_t$, the partial solution will append the newly visited node and lead to the next state $s_{t+1}$. The transition will be terminated when all nodes have been visited once, and the vehicle will return to the depot.
**Reward $\mathcal{R}$.** The goal is to construct a route with minimal travel length. Therefore, the reward of the whole action trajectory is set to the negative of the route length after the route is completed.
**Policy $\mathcal{P}$.** The stochastic policy $\pi_{\theta'}$ selects one node to visit (i.e., $a_t$) at each time step $t$. This process will be repeated until all customer nodes are served before returning to the depot. The final solution $\delta = (a_0, a_1, \dots, a_{|V|-1}, a_0)$ can be described as a permutation of actions. Note that for PDPs, $a_0 = x_0$ is always the depot. The stochastic policy $\pi_{\theta'}$ for constructing a solution $\delta$ given a problem instance $\{l(x)\}_{x \in V}$ can be factorized and parameterized by $\theta'$ as
$$ \pi_{\theta'} (\delta | \{l(x)\}_{x \in V}) = \pi_{\theta'}(\delta | s_0) = \prod_{\tau=0}^{|V|-1} \pi_{\theta'}(a_\tau | s_\tau). \quad (23) $$

### 5.2 Policy Network
Compared to the improvement policy in N2S, the network for the construction policy is less complicated since it plays an auxiliary role in our NCS, where it mainly constructs a route as the initial solution to be improved by the improvement policy. In this sense, we leverage an encoder-decoder structured policy network for PDPs (as illustrated in Fig. 4) following the Attention Model [9]. The input to the construction policy includes all nodes in a PDP instance and the partially constructed solution, and it outputs a probability distribution for the action which is used to select the next node to visit.

#### 5.2.1 Encoder
The input to the encoder network is the coordinates of the $2n+1$ nodes in the Euclidean space. Given the property of the PDP, a representation enhancement is adopted, where each pickup node is concatenated with its paring delivery node [4], so the pickup node will be represented as a 4-dimension vector rather than 2-dimension. And three linear projection layers are used for the depot node, concatenated pickup nodes, and delivery nodes, respectively. Each linear projection layer encodes a 2-dimension (4-dimension for the concatenated pickup node) coordinate vector $l(x_i)$ to a $d$-dimension embedding $h_i^{(0)}$ with $d=128$.
**Attention layer.** Given the embeddings from the linear projection, they will be updated and promoted through $N$ attention layers, each consisting of two sub-layers, i.e., multi-head self-attention (MHA) and feed-forward (FF), respectively. Each sub-layer has a skip-connection [26] and a layer normalization (LN) [27] (different from the batch normalization [28] used in Attention Model [9], we found LN works better). The above process could be briefly expressed as follows,
$$ \hat{h}_i = \text{LN}^l(h_i^{(l-1)} + \text{MHA}_i^l(h_i^{(l-1)}, (h_0^{(l-1)}, \dots, h_{2n}^{(l-1)})), \quad (24) $$
$$ h_i^{(l)} = \text{LN}(\hat{h}_i + \text{FF}^l(\hat{h}_i)), \quad (25) $$
where $l$ ($l \in \{1, 2, \dots, N\}$) refers to the index of the layer. Different layers keep their own parameters and do not share with others. Each MHA sub-layer uses $M=8$ heads and FF sub-layer has one hidden layer with 512-dimension.

After passing through the $N$ ($N=3$) attention layers, the representation for each node will be transformed into a $d$-dimension embedding $h_i^{(N)}$. Then the graph embedding will be calculated as the mean of these node embeddings as
$$ \bar{h}^{(N)} = \frac{1}{2n + 1} \sum_{i=0}^{2n} h_i^{(N)}. \quad (26) $$

#### 5.2.2 Decoder
Decoder aims to derive a probability distribution, based on which the action (or node) could be sequentially determined. It has three primary inputs, including 1) a ($2 * d$)-dimension context embedding $h_c^{(N)}$, which contains the graph embedding $\bar{h}^{(N)}$ as well as the embedding $h_{a_{t-1}}^{(N)}$ of the last node $a_{t-1}$ visited at time step $t$, i.e., $h_c^{(N)} = \text{Concat}(\bar{h}^{(N)}, h_{a_{t-1}}^{(N)})$. Note that since $a_0$ is always the depot $x_0$, the real decoding will start at $t=1$; 2) the node embeddings $\{h_0^{(N)}, h_1^{(N)}, \dots, h_{2n}^{(N)}\}$ from the encoder, which serves as the key and value for the attention computation in decoder; 3) a ($2n + 1$)-dimension mask vector, which ensures no duplicate nodes in a complete route and no violation of the precedence constraint of the pickup and delivery node pairs. As the context embedding and mask vector vary at each construction step, the partially constructed route is captured by masking the node from the context embedding at different steps.

Two attention layers are adopted in the decoder. The first one is an MHA layer with $M$ heads. At each time step, the current $h_c^{(N)}$ serves as the query, and node embeddings $\{h_0^{(N)}, h_1^{(N)}, \dots, h_{2n}^{(N)}\}$ serve as the key and value. After the MHA layer, $h_c^{(N+1)}$ is obtained, which is also termed *glimpse vector*. The second one is an MHA layer with a single head without the entity of *value*. It only computes the compatibility of the glimpse vector with all node embeddings as follows,
$$
u_i =
\begin{cases}
C \cdot \tanh(\frac{q k_i^T}{\sqrt{d}}), & \text{if } i \text{ is not masked} \\
-\infty, & \text{otherwise}
\end{cases}
\quad (27)
$$
where $q = h_c^{(N+1)} W^Q, k_i = h_i^{(N)} W^K$; $W^Q, W^K \in \mathbb{R}^{d \times d}$ are trainable parameter matrices; $C$ is used to clip the result within $[-C, C]$ and we set $C = 10$. Then the final probability for action selection is calculated using the Softmax function as follows,
$$ \pi_{\theta'}(a_\tau = i | s_\tau) = \text{Softmax}(\{u_i\}_{i=0}^{2n+1}). \quad (28) $$
According to the yielded probability vector, we sample the next node to visit. If the route has not yet been completely constructed, the sampled node will be used to update the context embedding $h_c^{(N)}$ and mask vector, based on which a new node to be visited next will be determined successively. This process is repeated until a complete route is constructed.

---
**Algorithm 3 n-step PPO (shared-critic) with CL and IL strategy**
**Input:** improvement policy $\pi_\theta$, construction policy $\pi_{\theta'}$, critic $v_\phi$, shared-critic parameter $\zeta$, PPO clipping threshold $\varepsilon$, learning rate $\eta_\theta, \eta_{\theta'}, \eta_\phi, \eta_\zeta$, learning rate decay $\beta$, epochs $E$, batches $B$, mini-batch $\kappa$, training steps $T_{\text{train}}$
1: **for** $e = 1$ to $E$ **do**
2:     **for** $b = 1$ to $B$ **do**
3:         Generate training data $\mathcal{D}_b$ on the fly;
4:         $\delta_0 \leftarrow \textbf{CurriculumLearning}(\mathcal{D}_b, \pi_\theta, \pi_{\theta'}, e)$;
5:         Set initial state $s_0$ based on $\delta_0$ and Eq. (4); $t \leftarrow 0$;
6:         **while** $t < T_{\text{train}}$ **do**
7:             $\mathcal{D}'_b \leftarrow \textbf{InvariantTransform}(\mathcal{D}_b)$;
8:             Get $(\delta', f(\delta'))$ where $\delta' \sim \pi_{\theta'}(\cdot | s'_0)$ on $\mathcal{D}'_b$;
9:             Get $\{(s_\tau, a_\tau, r_\tau), f(\delta_\tau)\}_{\tau=t}^{t+n-1}$ where $a_\tau \sim \pi_\theta(a_\tau | s_\tau)$;
10:            $t \leftarrow t + n$;
11:            $\pi_{\text{old}} \leftarrow \pi_\theta$, $v_{\text{old}} \leftarrow v_\phi$;
12:            $\pi_{\text{old}'} \leftarrow \pi_{\theta'}$, $\zeta_{\text{old}} \leftarrow \zeta$;
13:            **for** $k = 1$ to $\kappa$ **do**
14:                $\hat{R}_t = v_\phi(s_t)$;
15:                **for** $\tau \in \{t - 1, ..., t - n\}$ **do**
16:                    $\hat{R}_\tau \leftarrow r_\tau + \gamma \hat{R}_{\tau+1}$;
17:                    $\hat{A}_\tau \leftarrow \hat{R}_\tau - v_\phi(s_\tau)$;
18:                **end for**
19:                $V'(\zeta) \leftarrow \text{mean}(\{ f(\delta_\tau) - \zeta v_\phi(s_\tau) \}_{\tau=t-n}^{t-1})$;
20:                $A' \leftarrow f(\delta') - V'(\zeta)$;
21:                Compute RL loss $J_{RL}(\theta), J'_{RL}(\theta')$ using Eq. (19, 29) and clipped critic loss $L_{BL}(\phi), L'_{BL}(\zeta)$ using Eq. (20, 30);
22:                $\theta \leftarrow \theta + \eta_\theta \nabla J_{RL}(\theta)$;
23:                $\theta' \leftarrow \theta' - \eta_{\theta'} \nabla J'_{RL}(\theta')$;
24:                $\phi \leftarrow \phi - \eta_\phi \nabla L_{BL}(\phi)$;
25:                $\zeta \leftarrow \zeta - \eta_\zeta \nabla L'_{BL}(\zeta)$;
26:            **end for**
27:        **end while**
28:        $\theta' \leftarrow \textbf{ImitationLearning}(\pi_{\theta'}, \delta_{t-1}^*, \delta', \mathcal{D}_b, e)$
29:    **end for**
30:    $\eta_\theta \leftarrow \beta \eta_\theta$, $\eta_\phi \leftarrow \beta \eta_\phi$;
31: **end for**
---

### 5.3 Training Algorithm
The overall architecture of our NCS is illustrated in Fig. 5, while the training flow is summarized in Algorithm 3, which is essentially a collaborative proximal policy optimization (PPO) using the proposed shared-critic mechanism with curriculum learning and imitation learning. It jointly learns an improvement policy $\pi_\theta$ (the one in N2S) and a construction policy $\pi_{\theta'}$, where the critic of the construction model is attained based on the one for the improvement model.

We train NCS for $E$ epochs and $B$ batches per epoch, where the training dataset $\mathcal{D}_b$ is randomly generated on the fly with the uniform distribution (line 3). While the improvement policy performs the gradient update for multiple times on the same PDP instance (line 22), the construction policy only needs to perform the gradient update once. Therefore, each time the improvement policy samples an action on the same instance, the construction policy will sample a solution on an invariant transformation (line 7) of the same instance to strengthen its exploration capability. The procedure of the invariant transformation is summarized in Algorithm 5 of Appendix A.

The backbone of the whole algorithm basically follows the original PPO, where we present the reinforcement learning loss function and the baseline loss function in Eq. (29) and Eq. (30) for the construction policy, respectively. Here, we clip the estimated value $V'(\zeta)$ to make it around the previous one as $V'_{\text{clip}}(\zeta) = \text{clip}\left[ V'(\zeta), V'(\zeta_{\text{old}}) - \varepsilon, V'(\zeta_{\text{old}}) + \varepsilon \right]$. And $V'(\zeta)$ is the critic value (i.e., will be used for the actor in the construction model) calculated using the shared-critic mechanism, which will be introduced next.
$$
J'_{RL}(\theta') = \frac{1}{|\mathcal{D}_b|} \sum_{\mathcal{D}_b} \max \left( \frac{\pi_{\theta'}(\delta'|s'_0)}{\pi_{\text{old}'}(\delta'|s'_0)} A', \right. \nonumber \\
\left. \text{clip}\left[ \frac{\pi_{\theta'}(\delta'|s'_0)}{\pi_{\text{old}'}(\delta'|s'_0)}, 1 - \varepsilon, 1 + \varepsilon \right] A' \right), \quad (29)
$$
$$
L'_{BL}(\zeta) = \frac{1}{|\mathcal{D}_b|} \sum_{\mathcal{D}_b} \max \left( \left| V'(\zeta) - f(\delta') \right|, \right. \nonumber \\
\left. \left| V'_{\text{clip}}(\zeta) - f(\delta') \right| \right)^2. \quad (30)
$$

**Shared-Critic Mechanism.** The critic network for the improvement model (from N2S) takes its state $s_t$ as the input, and fits the loss function in Eq. (20). We deduce that this critic network is trained to output a good estimation of the cumulative expected future rewards from the given state $s_t$, which is also the gap of objective value (i.e., route length) between solution $\delta_t$ and the optimal one. It means that the optimal objective for instance $\{l(x)\}_{x \in V}$ approximately equals $f(\delta_t) - v_\phi(s_t)$. In practice, we may not need a rigorously precise estimation during training, so we introduce a trainable parameter $\zeta$, and thus the estimated optimal objective of the instance could become $f(\delta_t) - \zeta v_\phi(s_t)$. This can be potentially used as the critic of the construction model, termed as the shared-critic mechanism, since it is derived directly from the improvement model. In practice, we use the n-step $f(\delta_t)$ and $v_\phi(s_t)$ to estimate this value, also known as the baseline of the construction model, i.e., $V'(\zeta) \!\leftarrow\! \text{mean}(\{ f(\delta_\tau) \!-\! \zeta v_\phi(s_\tau) \}_{\tau=t}^{t+n-1})$, thus the construction advantage could be calculated as $A' \leftarrow f(\delta') - V'(\zeta)$ (line 19, 20). Naturally, this shared-critic mechanism acts as a bridge to enable collaborative training with one critic network for two models. Compared with the traditional critic (i.e., a separate concrete network) [29] and rollout baseline (i.e., a backup of policy network) [9], our shared-critic has only one trainable parameter for the construction model, so the computation cost is greatly reduced.

**Upgraded Curriculum Learning.** Although a CL strategy already exists in the original N2S (the improvement model), with the introduced construction model, an upgraded CL (line 4) could make the whole model perform better. Recall that the CL in original N2S can only use the iteratively improved results yielded by the N2S policy itself as the initial solutions, and once the policy gets stuck, e.g., falling into a local optimum, it is difficult for the N2S policy to jump out of it. But with the construction model, we could construct more suitable initial solutions for the improvement model. This is the main reason why construction model could boost the performance of the improvement model from the original N2S. Another advantage of using a construction model to guide the curriculum learning is that it saves computation time, since the route construction process with a construction network is much faster than a multi-step neighborhood search process by the improvement model. Particularly, in the upgraded CL, a solution $\delta_0$ is yielded from the current construction policy $\pi_{\theta'}$ and then the improvement policy $\pi_\theta$ uses it to set its initial state $s_0$. The quality of solution $\delta_0$ will be improved as the epoch $e$ increases. In such a way, the hardness level of neighborhood search is gradually increased per epoch following the rationale of curriculum learning. The detailed procedure of the upgraded curriculum learning is shown in Algorithm 6 of Appendix A.

**Imitation learning.** Inspired by [30], an imitation learning (IL) strategy is applied at the end of each batch. Curriculum learning enables the construction model to help the improvement model during collaborative training, while imitation learning enables the other way around. After the complete training with a small batch, we could obtain the solution $\delta_{t-1}^*$ which is first yielded using curriculum learning and then improved for $T_{\text{train}}$ steps by the improvement model. So it is reasonable to consider it as a solution of good quality, which could be used as a positive imitation sample. Thus, the construction policy could further update itself by imitating this sample (line 28). In particular, the loss function for imitating learning is expressed as follows,
$$ J'_{IL} = - \frac{1}{|\mathcal{D}_b|} \sum_{\mathcal{D}_b} (\xi \cdot \pi_{\theta'}(\delta^* | s'_0)), \quad (31) $$
where $\xi$ is a tunable parameter to regulate the importance of imitation for the construction model. The procedure of imitation learning is summarized in Algorithm 7 of Apendix A.

---
**Algorithm 4 NCS(-A) Inference**
**Input:** Instance $\mathcal{I}$ with size $|V|$, policy $\pi_\theta$ and $\pi_{\theta'}$, sample times $S$, maximum step $T$
1: **if** augment enabled **then**
2:     **for** $i = 1, ..., \lfloor \frac{1}{2}|V| \rfloor$ **do**
3:         $\mathcal{I}_i \leftarrow \textbf{InvariantTransform}(\mathcal{I})$;
4:     **end for**
5: **end if**
6: Sample $S$ solutions for all instances $\mathcal{I}_i$ in parallel with $\pi_{\theta'}$;
7: Keep the best solution among these $S$ solutions as the initial solution $\delta_0$;
8: Solve all instances $\mathcal{I}_i$ starting at $\delta_0$ in parallel with $\pi_\theta$ for $T$ steps;
9: **return** the best solution found among all $\mathcal{I}_i$;
---

### 5.4 Diversity Enhancement during Inference
Regarding the inference, NCS first yields the initial solution by sampling with the construction policy, and then passes it to the improvement policy as the starting solution for iterative updates. In the end, the best solution yielded by the improvement policy is retrieved as the final output. Similar to N2S, NCS is also equipped with the augmentation scheme during inference (i.e., NCS-A), which is summarized in Algorithm 4.

## 6 EVALUATION
We design experiments to answer the following questions:
1) How good are the proposed N2S and NCS against the baselines, including the state-of-the-art neural methods and the strong LKH3 solver? (see Table 3 and Table 4)
2) Can N2S Synth-Att reduce computation costs while achieving competitive performance to DAC-Att? (see Table 5)
3) How crucial are the proposed learnable node-pair removal and node-pair reinsertion decoders for achieving an efficient search in N2S? (see Table 6)
4) How does the upgraded curriculum learning in NCS help the improvement model compared to the one trained individually? (see Table 7 and Fig. 6)
5) How does the imitation learning in NCS help the construction model? (see Table 7 and Fig. 7)
6) Can our N2S and NCS generalize well to benchmark instances that are different from training ones? (see Table 8)

**TABLE 2: Training details of the adopted neural baselines.**

| Method | Code | Training Time | Hyper-parameters |
| :--- | :--- | :--- | :--- |
| Heter-AM | online | $\sim 20$ days | train 800 epochs as per the original setting. |
| Heter-POMO | online | $\sim 14$ days | train 2,000 epochs as per the original setting. |
| DACT | online | $\sim 10$ days | $\xi^{CL} = 0.25, 1, 4$ for sizes $|v|=21, 51, 101$, respectively;<br>$n=5, T_{\text{train}}=250$ (same as ours);<br>train 200 epochs (same as ours). |

### 6.1 Setup
We evaluate N2S and NCS on PDTSP and PDTSP-LIFO with three sizes $|V| = 21, 51, 101$ following the conventions in [2], [3], where the node coordinates of instances are randomly and uniformly generated in the unit square $[0, 1] \times [0, 1]$. For N2S, the initial solution $\delta_0$ is sequentially constructed in a random fashion. Our experiments were conducted on a server equipped with 8 RTX 2080 Ti GPU cards and Intel E5-2680 CPU @ 2.4GHz. The training time of N2S varies with problem sizes, i.e., around 1 day for $|V|=21$, 3 days for $|V|=51$, and 7 days for $|V|=101$. As for NCS, it is 1 day, 7 days and 10 days, respectively, both of which are shorter than all the neural baselines in Table 2. Our code is publicly available online.

**Hyper-parameters.** Our N2S and NCS are trained with $E = 200$ epochs and $B = 20$ batches per epoch using batch size 600. We set $n = 5, T_{\text{train}} = 250$ for the $n$-step PPO with $\kappa = 3$ mini-batch updates and a clip threshold $\epsilon = 0.1$. Adam optimizer is used with learning rate $\eta_\theta = 8 \times 10^{-5}$ for $\pi_\theta$, $\eta_\phi = 2 \times 10^{-5}$ for $v_\phi$ (decayed $\beta = 0.985$ per epoch), and $\eta'_{\theta'} = 10^{-4}$ for $\pi'_{\theta'}$, $\eta_\zeta = 0.01$, both without decay. The reward discount factor $\gamma$ is set to 0.999 for both PDPs. We clip the gradient norm of N2S network to be within $0.05, 0.15, 0.3$, and set the curriculum learning $\rho$ to $2, 1.5, 1$ for the three problem sizes, respectively. As for the construction model, its gradient norm is clipped to 1. Although $\xi$ in Eq. (31) can regulate the strength of imitation learning, we actually use gradient clip to control this. We set $\xi = 1$ and clip the gradient norm of imitation learning to $0.1, 0.1, 0.01$ for the three problem sizes. Particularly, the used hyperparameters regarding curriculum learning and imitation learning in NCS are shown in Table 9 of Appendix A.

### 6.2 Comparison Evaluation
We compare our N2S and NCS with the state-of-the-art (SOTA) neural methods and the highly-optimized LKH3 solver.

Regarding the former baseline, we consider the SOTA *improvement* method DACT [2] and the SOTA *construction* method Heter-AM [4] (specially designed for PDPs). To make a fair comparison with our N2S-A and NCS-A (with the diversity enhancement), we upgrade Heter-AM to Heter-POMO, also given the known superiority of POMO to AM. In specific, we reserve the policy network in Heter-AM as the backbone while adopting the diverse rollouts and the data augmentation techniques in POMO [3] to leverage the advantages of them for the best performance. Each neural baseline is trained using the respective implementation code that is publicly available. For the upgraded Heter-POMO, we adapt and combine the model architecture from the original Heter-AM and the original POMO. The links to their original implementations, approximate training time for the size $|V| = 101$, and the used hyper-parameters are presented in Table 2. For other hyper-parameters, we follow the recommendation in their papers. Regarding the latter baseline, LKH3 is a strong heuristic (as reviewed in Section 2) which is widely used as a baseline to benchmark neural methods in recent studies (e.g., [2], [14], [15], [31]). We report its results with two settings of iterations, i.e., LKH (5k) and LKH (10k). Note that we did not use the sole construction model in NCS as a baseline for comparison, since it is used as a complement to the improvement model. Heter-AM trains for 800 epochs and samples 1280000 instances in each epoch, while the construction model in NCS only trains for 200 epochs and samples only 12000 instances in each epoch.

All baselines are evaluated on a test dataset with 2,000 instances, and we report the metrics of averaged objective values, averaged gaps to LKH3 (10K) and the total solving time. Note that it is hard to perform an absolutely fair time comparison between running Python codes on GPUs (neural methods) and running ANSI C codes on CPUs (LKH solver). Thus we follow the guidelines in [32] to perform the *facilitate comparison* that lets each method make full use of the best settings on our machine. In particular, we report the time of LKH3 when running in parallel with 16 CPU cores and the time of each neural method when all 8 GPU cards are available (but do not need to be fully used).

**Results on PDTSP.** Table 3 shows the results on PDTSP.In the first group, we compare N2S and NCS with Heter-AM (*greedy* and *sampling*), and DACT. Compared to Heter-AM (5k), our N2S with only 1k steps attains lower gaps with less time for all sizes. Although DACT offers the best gap on PDTSP-21, its performance drops significantly as the problem size increases, partly because its decoder is less efficient than our node-pair *removal* and *reinsertion* ones when tackling larger-scale problems. Instead, our N2S achieves higher performance and consistently dominates DACT in terms of both the gaps and the time on PDTSP-51 and PDTSP-101. As for NCS, the performance is further boosted compared to N2S, with better results for all three different total steps. Even NCS with 1k steps has surpassed N2S with 3k steps, without incurring much extra consumption time. In the second group, our augmented N2S-A and NCS-A is compared to the upgraded Heter-POMO method with three variants. It is shown that even with only 1k steps, our N2S-A attains a significantly smaller gap than all three Heter-POMO variants, by almost an order of magnitude. Although Heter-POMO-A (gr.) tends to be competitive with fast speed, the gap is hard to be further reduced by increasing inference time if we refer to Heter-POMO-A (3k). Moreover, our N2S-A (2k) keeps abreast of, or even slightly exceeds the strong LKH3 solver, achieving gaps of -0.03% and -0.01% with less time on PDTSP-51 and PDTSP-101, respectively. Those gaps are further reduced to -0.04% and -0.20% with more steps, i.e., 3k. And NCS-A can further extend the advantage to -0.06% and -0.22%. However, we observe that there is no significant difference between N2S-A and NCS-A, which implies that the results of the two approaches may be fairly close to the real optimum, with the help of diversity enhancement.

**TABLE 3: Results for PDTSP with sizes $|V| = 21, 51, 101$. The "+" in "Total Time" means construction time plus improvement time.**

| Methods | PDTSP-21 | | | PDTSP-51 | | | PDTSP-101 | | |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| | Obj. Value | Gap to LKH | Total Time | Obj. Value | Gap to LKH | Total Time | Obj. Value | Gap to LKH | Total Time |
| LKH (5k) | 4.563 | 0.00% | 3m | 6.866 | 0.06% | 10m | 9.443 | 0.16% | 49m |
| LKH (10k) | 4.563 | 0.00% | 5m | 6.862 | 0.00% | 19m | 9.428 | 0.00% | 98m |
| Heter-AM (gr.) | 4.655 | 2.02% | (0s) | 7.333 | 6.86% | (1s) | 10.348 | 9.76% | (2s) |
| Heter-AM (5k) | 4.578 | 0.33% | (33s) | 7.108 | 3.58% | (1.5m) | 10.051 | 6.61% | (5m) |
| DACT (1k) | 4.572 | 0.20% | (18s) | 7.245 | 5.57% | (29s) | 10.551 | 11.91% | (51s) |
| DACT (2k) | 4.566 | 0.07% | (37s) | 7.118 | 3.72% | (1m) | 10.312 | 9.38% | (1.5m) |
| DACT (3k) | 4.564 | 0.03% | (1m) | 7.057 | 2.83% | (1.5m) | 10.195 | 8.13% | (2.5m) |
| N2S (1k) | 4.573 | 0.21% | (21s) | 7.103 | 3.51% | (31s) | 10.030 | 6.38% | (1m) |
| N2S (2k) | 4.567 | 0.09% | (42s) | 7.053 | 2.77% | (1m) | 9.905 | 5.06% | (2m) |
| N2S (3k) | 4.565 | 0.05% | (1m) | 7.027 | 2.40% | (1.5m) | 9.846 | 4.44% | (3m) |
| NCS (1k) | 4.570 | 0.15% | (1s+21s) | 6.974 | 1.63% | (7s+31s) | 9.808 | 4.03% | (46s+1m) |
| NCS (2k) | 4.567 | 0.09% | (1s+42s) | 6.957 | 1.38% | (7s+1m) | 9.757 | 3.49% | (46s+2m) |
| NCS (3k) | 4.565 | 0.05% | (1s+1m) | 6.948 | 1.25% | (7s+1.5m) | 9.730 | 3.20% | (46s+3m) |
| Heter-POMO (gr.) | 4.634 | 1.56% | (0s) | 7.168 | 4.45% | (1s) | 10.060 | 6.70% | (2s) |
| Heter-POMO-A (gr.)| 4.584 | 0.46% | (1s) | 6.995 | 1.93% | (5s) | 9.681 | 2.68% | (11s) |
| Heter-POMO-A (3k) | 4.564 | 0.03% | (7m) | 6.916 | 0.77% | (32m) | 9.567 | 1.47% | (135m) |
| N2S-A (1k) | 4.563 | 0.01% | (1m) | 6.865 | 0.03% | (8m) | 9.475 | 0.50% | (40m) |
| N2S-A (2k) | 4.563 | **0.00%**| (2m) | 6.860 | **-0.03%**| (16m) | 9.427 | **-0.01%**| (80m) |
| N2S-A (3k) | 4.563 | **0.00%**| (3m) | 6.860 | **-0.04%**| (24m) | 9.409 | **-0.20%**| (121m) |
| NCS-A (1k) | 4.563 | **0.00%**| (6s+1m) | 6.864 | 0.03% | (35s+8m) | 9.471 | 0.46% | (9m+40m)|
| NCS-A (2k) | 4.563 | **0.00%**| (6s+2m) | 6.860 | **-0.03%**| (35s+16m)| 9.424 | **-0.04%**| (9m+80m)|
| NCS-A (3k) | 4.563 | **0.00%**| (6s+3m) | 6.858 | **-0.06%**| (35s+24m)| 9.407 | **-0.22%**| (9m+121m)|

**TABLE 4: Results for PDTSP-LIFO with sizes $|V| = 21, 51, 101$. The "+" in "Total Time" means construction time plus improvement time.**

| Methods | PDTSP-LIFO-21 | | | PDTSP-LIFO-51 | | | PDTSP-LIFO-101 | | |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| | Obj. Value | Gap to LKH | Total Time | Obj. Value | Gap to LKH | Total Time | Obj. Value | Gap to LKH | Total Time |
| LKH (5k) | 5.539 | 0.00% | 1m | 10.218 | 0.17% | 8m | 17.115 | 0.34% | 33m |
| LKH (10k) | 5.539 | 0.00% | 3m | 10.200 | 0.00% | 16m | 17.057 | 0.00% | 67m |
| N2S (1k) | 5.541 | 0.04% | (2m) | 10.365 | 1.61% | (2.5m) | 17.660 | 3.54% | (3.5m) |
| N2S (2k) | 5.540 | 0.02% | (4m) | 10.301 | 0.99% | (5.5m) | 17.471 | 2.43% | (6.5m) |
| N2S (3k) | 5.539 | 0.01% | (6m) | 10.265 | 0.64% | (8m) | 17.373 | 1.85% | (10m) |
| NCS (1k) | 5.540 | 0.02% | (1s+2m) | 10.299 | 0.97% | (7s+2.5m) | 17.466 | 2.40% | (46s+3.5m) |
| NCS (2k) | 5.539 | 0.01% | (1s+4m) | 10.260 | 0.59% | (7s+5.5m) | 17.343 | 1.68% | (46s+6.5m) |
| NCS (3k) | 5.539 | 0.01% | (1s+6m) | 10.242 | 0.41% | (7s+8m) | 17.276 | 1.29% | (46s+10m) |
| Heter-POMO (gr.) | 5.636 | 1.75% | (0s) | 10.540 | 3.33% | (1s) | 17.583 | 3.08% | (2s) |
| Heter-POMO-A (gr.)| 5.567 | 0.51% | (1s) | 10.353 | 1.50% | (5s) | 17.276 | 1.29% | (10s) |
| Heter-POMO-A (4k) | 5.545 | 0.12% | (10m) | 10.209 | 0.09% | (48m) | 16.890 | -0.98% | (180m) |
| N2S-A (1k) | 5.539 | **0.00%**| (3m) | 10.144 | **-0.55%**| (13m) | 16.976 | -0.47% | (54m) |
| N2S-A (2k) | 5.539 | **0.00%**| (6m) | 10.137 | **-0.62%**| (27m) | 16.859 | **-1.16%**| (107m) |
| N2S-A (3k) | 5.539 | **0.00%**| (9m) | 10.135 | **-0.64%**| (40m) | 16.806 | **-1.47%**| (161m) |
| NCS-A (1k) | 5.539 | **0.00%**| (6s+3m) | 10.142 | **-0.57%**| (35s+13m)| 16.894 | -0.96% | (9m+54m)|
| NCS-A (2k) | 5.539 | **0.00%**| (6s+6m) | 10.136 | **-0.63%**| (35s+27m)| 16.815 | **-1.42%**| (9m+107m)|
| NCS-A (3k) | 5.539 | **0.00%**| (6s+9m) | 10.134 | **-0.65%**| (35s+40m)| 16.771 | **-1.68%**| (9m+161m)|

**Results on PDTSP-LIFO.** In Table 4, we report the results on PDTSP-LIFO. Due to the more constrained search space, DACT failed to work well (see Table 6). Therefore, we mainly compare our approaches with Heter-POMO (the best neural baseline in Table 3) and the LKH3 solver. As exhibited, the advantages of neural methods over LKH3 have been further enhanced on this harder problem, where our approach consistently outperforms Heter-POMO for all sizes. Compared to LKH3, our N2S-A with 3k steps presents superior performance again, and attains gaps of -0.64% and -1.47% on PDTSP-LIFO-51 and PDTSP-LIFO-101, respectively. And our NCS-A achieves even better results than that of N2S-A, with gaps of -0.65% and -1.68%, respectively.

### 6.3 Ablation Evaluation

**TABLE 5: Effects of different encoding methods.**

| Att. in Encoders | Dim. | # Param.(M) | Time(s) | Gap(%) |
| :--- | :--- | :--- | :--- | :--- |
| Vanilla-Att | 64 | 0.18 (1.00×) | 239 (1.00×) | 4.66 |
| DAC-Att | 64 | 0.31 (1.72×) | 285 (1.19×) | 2.89 |
| **Synth-Att** | **64** | **0.19 (1.06×)** | **255 (1.07×)** | **2.88** |
| Vanilla-Att | 128 | 0.72 (1.00×) | 322 (1.00×) | 3.92 |
| DAC-Att | 128 | 1.25 (1.73×) | 400 (1.24×) | 2.43 |
| **Synth-Att** | **128** | **0.76 (1.06×)** | **340 (1.06×)** | **2.40** |

**Effects of Different Encoding Methods in N2S.** We replace the proposed Synth-Att in our N2S encoder with vanilla-Att and DAC-Att, respectively. Using only one GPU card, we report the number of model parameters, time, and gaps for solving 2,000 PDTSP-51 instances with 3k steps in Table 5. As exhibited, Synth-Att attains slightly smaller gaps than DAC-Att with much fewer computation costs, and considerably lower gaps than vanilla-Att with slight extra computation costs.

**TABLE 6: Effects of different decoding methods.**

| Removal Decoder | Reinsertion Decoder | Gap(%) on PDTSP | Gap(%) on PDTSP-LIFO |
| :---: | :---: | :---: | :---: |
| ✗(random) | ✗(random) | 210.82 | 112.37 |
| ✗(random) | ✗($\epsilon$-greedy) | 18.03 | 17.87 |
| ✗($\epsilon$-greedy)| ✗(random) | 86.31 | 47.57 |
| ✗($\epsilon$-greedy)| ✗($\epsilon$-greedy) | 15.62 | 12.64 |
| ✓ | ✗(random) | 43.12 | 17.75 |
| ✓ | ✗($\epsilon$-greedy) | 3.54 | 3.88 |
| ✗(random) | ✓ | 6.38 | 4.38 |
| ✗($\epsilon$-greedy)| ✓ | 8.42 | 6.28 |
| ✗(DACT) | ✗(DACT) | 2.83 | 11.73 |
| **✓** | **✓** | **2.40** | **0.64** |

**TABLE 7: Effects of upgraded curriculum learning and imitation learning.**

| Method | Obj. Value | | | Gap to LKH | | |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| | 1k | 2k | 3k | 1k | 2k | 3k |
| NCS | 6.974 | 6.957 | **6.948** | 1.63% | 1.38% | **1.25%** |
| N2S-w-construction | 6.982 | 6.963 | 6.953 | 1.75% | 1.47% | 1.33% |
| NCS-w/o-imitation | 7.043 | 7.012 | 6.996 | 2.64% | 2.19% | 1.95% |
| NCS-A | 6.864 | 6.860 | **6.858** | 0.03% | -0.03% | **-0.06%**|
| N2S-A-w-construction| 6.865 | 6.861 | 6.860 | 0.04% | -0.01% | -0.03% |
| NCS-A-w/o-imitation | 6.866 | 6.861 | 6.859 | 0.06% | -0.01% | -0.04% |

**Effects of Different Decoding Methods in N2S.** To highlight the desirability of our two decoders, we replace the trainable decoders with hand-crafted ones (i.e., random and the $\epsilon$-greedy with $\epsilon = 0.1$) while ensuring feasibility. We gather the results on PDTSP-51 and PDTSP-LIFO-51 with 3k steps in Table 6 where mark ‘✓’ means our proposed decoder is used (the one without augments) and mark ‘✗’ means the decoder in the parentheses is used instead. As revealed, the methods of retaining at least one trainable decoder always attain much lower gaps than the ones equipped with only hand-crafted decoders. The method with both trainable decoders (i.e., N2S) achieves the best performance. We also notice that DACT fails to solve PDTSP-LIFO well. This might be because its decoder considers removing and reinserting only one node instead of a pickup-delivery node pair in each action, which is a serious limitation for more constrained PDPs. For example, on PDTSP-LIFO, over 92% of the action space need to be masked for DACT, which leads to extremely low efficiency.

**Effects of Upgraded Curriculum Learning in NCS.** To assess the effects of the upgraded CL in NCS, we compare the improvement policy trained collaboratively with the construction policy using NCS (with upgraded CL) and the one trained independently using N2S (with original CL). Meanwhile, we use the construction policy trained in NCS to also generate initial solutions to boost the latter improvement model during the inference for a fair comparison, named N2S-w-construction. The comparison results on PDTSP-51 instances, with and without the diversity enhancement, are gathered in Table 7. As can be seen, our NCS consistently outperforms N2S-w-construction regardless of the diversity enhancement. In Fig. 6, we further depict the averaged search curves of the best-so-far and current solutions during inference for both methods, where we observe that although both improvement policies start with identical initial solutions, their iterative trajectories differ a lot. Our NCS explores regions of lower objective values earlier and converges to better solutions faster. This may be attributed to the distinct CL strategies that result in different scopes of the initial solutions during training for the improvement policies, where the initial solutions yielded by the construction model with the upgraded CL strategy often lead to more desirable areas that are closer to the optimal solutions.

**Effects of Imitation Learning in NCS.** As shown in Table 7, the absence of imitation learning in NCS would greatly affect the performance of the construction model and thus the eventual results. The construction model in NCS is lightweight and uses considerably fewer computational resources compared to individually learned models (e.g., Heter-AM samples 426.67 times more instances during training than our construction model). Consequently, relying solely on reinforcement learning (i.e., without imitation learning) leads to underfitting for the construction model, resulting in poor initial solutions as illustrated in Fig. 7. This hinders the effectiveness of collaborative training, as the scope of initial solutions provided by the construction model is not adequately desirable, thus undermining the overall performance.

### 6.4 Generalization Evaluation
We further evaluate our N2S and NCS on benchmark instances, including all the ones from [19] for PDTSP and the ones with size $|V| \le 201$ from [21] for PDTSP-LIFO, which are largely different from our training ones, e.g., different sizes (i.e., 200 nodes) as shown in Fig. 8 and different node distributions as shown in Fig. 9. In Table 8, we report the best and the average gaps (with 10 independent runs) achieved by N2S-A, NCS-A and neural baseline Heter-POMO-A w.r.t. optimal solutions for PDTSP, or heuristic baseline B1 [21] and B2 [22] for PDTSP-LIFO. It can be seen that our N2S and NCS significantly outstrip Heter-POMO in all cases, with NCS further strengthening the advantage over N2S. Without re-training or leveraging other techniques, this is fairly hard to achieve because machine learning often suffers from a mediocre out-of-distribution zero-shot generalization [33], [34]. The results also imply slight inferiority of our N2S and NCS to the LKH3 solver, given that LHK3 reports similar performance to the B2 baseline on those instances [6]. Accordingly, we will focus on further improving the out-of-distribution generalization performance for our N2S and NCS in the future.

**TABLE 8: Generalization performance on benchmark instances.**

| Problem | $|V|$ | Gaps to | Heter-POMO-A | | N2S-A | | NCS-A | |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| | | | Avg. | Best | Avg. | Best | Avg. | Best |
| PDTSP | 101<br>201 | Opt. | 1.64%<br>9.71% | 1.46%<br>8.66% | 0.08%<br>2.82% | **0.00%**<br>2.19% | **0.03%**<br>**2.24%** | **0.00%**<br>**1.93%** |
| PDTSP-LIFO| $\le 101$ | B1<br>B2 | 9.63%<br>10.98% | 8.80%<br>10.14% | 0.81%<br>2.02% | **-0.17%**<br>**1.02%** | **0.42%**<br>**1.62%** | -0.04%<br>1.15% |

## 7 CONCLUSION
We first present an efficient neural neighborhood search (N2S) approach for PDPs. It utilizes a novel Synth-Att to synthesize node relationships from various types of solution features and exploits the node-pair removal and reinsertion decoders to tackle the precedence constraint. Then, on top of N2S, we present a neural collaborative search (NCS) approach, which introduces a construction model to the improvement model in the N2S. The two models are connected with a shared-critic in NCS, and they also further enhance each other through an upgraded curriculum learning and an imitation learning, respectively. Extensive experiments on PDTSP and PDTSP-LIFO verified our design, where N2S and NCS achieve state-of-the-art performance among existing neural methods. Further equipped with a diversity enhancement scheme, they even become the first neural methods to surpass LKH3 on synthesized PDP instances. In future, we will 1) deploy N2S or NCS as a low-level agent in the hierarchical framework of [35] for dynamic PDPs, 2) combine N2S or NCS with a similar divide-and-conquer strategy in [31] for much larger-scale instances, 3) enhance the out-of-distribution generalization for N2S and NCS so as to exceed LKH3 on instances of arbitrary distributions, 4) investigate different types of improvement model and construction model in NCS and solve more other PDP variants.

## ACKNOWLEDGMENTS
This work was supported by the National Natural Science Foundation of China (Grant No. 62072258, 61772290).

## REFERENCES
[1] S. N. Parragh, K. F. Doerner, and R. F. Hartl, “A survey on pickup and delivery problems,” *Journal fur Betriebswirtschaft*, vol. 58, no. 2, pp. 81–117, 2008.
[2] Y. Ma, J. Li, Z. Cao, W. Song, L. Zhang, Z. Chen, and J. Tang, “Learning to iteratively solve routing problems with dual-aspect collaborative Transformer,” in *Advances in Neural Information Processing Systems*, vol. 34, 2021, pp. 11 096–11 107.
[3] Y.-D. Kwon, J. Choo, B. Kim, I. Yoon, Y. Gwon, and S. Min, “POMO: Policy optimization with multiple optima for reinforcement learning,” in *Advances in Neural Information Processing Systems*, vol. 33, 2020, pp. 21 188–21 198.
[4] J. Li, L. Xin, Z. Cao, A. Lim, W. Song, and J. Zhang, “Heterogeneous attentions for solving pickup and delivery problem via deep reinforcement learning,” *IEEE Transactions on Intelligent Transportation Systems*, 2021.
[5] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” in *Advances in Neural Information Processing Systems*, vol. 30, 2017, pp. 6000–6010.
[6] K. Helsgaun, “An extension of the lin-kernighan-helsgaun tsp solver for constrained traveling salesman and vehicle routing problems,” *Roskilde: Roskilde University*, 2017.
[7] M. Nazari, A. Oroojlooy, M. Takáč, and L. V. Snyder, “Reinforcement learning for solving the vehicle routing problem,” in *Advances in Neural Information Processing Systems*, 2018, pp. 9861–9871.
[8] C. K. Joshi, T. Laurent, and X. Bresson, “An efficient graph convolutional network technique for the travelling salesman problem,” arXiv preprint arXiv: 1906.01227, 2019.
[9] W. Kool, H. van Hoof, and M. Welling, “Attention, learn to solve routing problems!” in *International Conference on Learning Representations*, 2018.
[10] M. Kim, J. Park, and J. Kim, “Learning collaborative policies to solve NP-hard routing problems,” in *Advances in Neural Information Processing Systems*, vol. 34, 2021, pp. 10 418–10 430.
[11] Z.-H. Fu, K.-B. Qiu, and H. Zha, “Generalize a small pre-trained model to arbitrarily large TSP instances,” in *AAAI Conference on Artificial Intelligence*, 2021.
[12] X. Chen and Y. Tian, “Learning to perform local rewriting for combinatorial optimization,” in *Advances in Neural Information Processing Systems*, vol. 32, 2019, pp. 6281–6292.
[13] A. Hottung and K. Tierney, “Neural large neighborhood search for the capacitated vehicle routing problem,” in *European Conference on Artificial Intelligence*, 2020.
[14] Y. Wu, W. Song, Z. Cao, J. Zhang, and A. Lim, “Learning improvement heuristics for solving routing problems,” *IEEE Transactions on Neural Networks and Learning Systems*, 2021.
[15] A. Hottung, B. Bhandari, and K. Tierney, “Learning a latent search space for routing problems using variational autoencoders,” in *International Conference on Learning Representations*, 2021.
[16] W. Kool, H. van Hoof, J. Gromicho, and M. Welling, “Deep policy dynamic programming for vehicle routing problems,” arXiv preprint arXiv:2102.11756, 2021.
[17] M. W. Savelsbergh, “An efficient implementation of local search algorithms for constrained routing problems,” *European Journal of Operational Research*, vol. 47, no. 1, pp. 75–85, 1990.
[18] J. Renaud, F. F. Boctor, and J. Ouenniche, “A heuristic for the pickup and delivery traveling salesman problem,” *Computers & Operations Research*, vol. 27, no. 9, pp. 905–916, 2000.
[19] J. Renaud, F. F. Boctor, and G. Laporte, “Perturbation heuristics for the pickup and delivery traveling salesman problem,” *Computers & Operations Research*, vol. 29, no. 9, pp. 1129–1141, 2002.
[20] M. Veenstra, K. J. Roodbergen, I. F. Vis, and L. C. Coelho, “The pickup and delivery traveling salesman problem with handling costs,” *European Journal of Operational Research*, vol. 257, no. 1, pp. 118–132, 2017.
[21] F. Carrabs, J.-F. Cordeau, and G. Laporte, “Variable neighborhood search for the pickup and delivery traveling salesman problem with LIFO loading,” *INFORMS Journal on Computing*, vol. 19, no. 4, pp. 618–632, 2007.
[22] Y. Li, A. Lim, W.-C. Oon, H. Qin, and D. Tu, “The tree representation for the pickup and delivery traveling salesman problem with LIFO loading,” *European Journal of Operational Research*, vol. 212, no. 3, pp. 482–496, 2011.
[23] J. Zhao, M. Mao, X. Zhao, and J. Zou, “A hybrid of deep reinforcement learning and local search for the vehicle routing problems,” *IEEE Transactions on Intelligent Transportation Systems*, vol. 22, no. 11, pp. 7208–7218, 2020.
[24] R. García-Torres, A. A. Macias-Infante, S. E. Conant-Pablos, J. C. Ortiz-Bayliss, and H. Terashima-Marín, “Combining constructive and perturbative deep learning algorithms for the capacitated vehicle routing problem,” arXiv preprint arXiv:2211.13922, 2022.
[25] X. Wang, Y. Chen, and W. Zhu, “A survey on curriculum learning,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 44, no. 9, pp. 4555–4576, 2021.
[26] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2016, pp. 770–778.
[27] J. L. Ba, J. R. Kiros, and G. E. Hinton, “Layer normalization,” arXiv preprint arXiv:1607.06450, 2016.
[28] S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep network training by reducing internal covariate shift,” in *International conference on machine learning*, pmlr, 2015, pp. 448–456.
[29] I. Bello, H. Pham, Q. V. Le, M. Norouzi, and S. Bengio, “Neural combinatorial optimization with reinforcement learning,” in *International Conference on Machine Learning (Workshop)*, 2017.
[30] A. Hottung, Y.-D. Kwon, and K. Tierney, “Efficient active search for combinatorial optimization problems,” in *International Conference on Learning Representations*, 2022.
[31] S. Li, Z. Yan, and C. Wu, “Learning to delegate for large-scale vehicle routing,” in *Advances in Neural Information Processing Systems*, vol. 34, 2021, pp. 26 198–26 211.
[32] L. Accorsi, A. Lodi, and D. Vigo, “Guidelines for the computational testing of machine learning approaches to vehicle routing problems,” *Operations Research Letters*, vol. 50, no. 2, pp. 229–234, 2022.
[33] K. Zhou, Z. Liu, Y. Qiao, T. Xiang, and C. C. Loy, “Domain generalization: A survey,” arXiv preprint arXiv:2103.02503, 2021.
[34] J. Li, Y. Ma, R. Gao, Z. Cao, A. Lim, W. Song, and J. Zhang, “Deep reinforcement learning for solving the heterogeneous capacitated vehicle routing problem,” *IEEE Transactions on Cybernetics*, 2021.
[35] Y. Ma, X. Hao, J. Hao, J. Lu, X. Liu, T. Xialiang, M. Yuan, Z. Li, J. Tang, and Z. Meng, “A hierarchical reinforcement learning based optimization framework for large-scale dynamic pickup and delivery problems,” in *Advances in Neural Information Processing Systems*, vol. 34, 2021, pp. 23 609–23 620.

---

**Detian Kong** received the M.Sc. degree in computer science and technology from the China University of Geosciences Beijing in 2021. He is currently pursuing the Ph.D. degree with the Research Center of Logistics, Nankai University, Tianjin, China. His research interests include deep reinforcement learning, evolutionary computation, combinatorial optimization and logistics system optimization.

**Yining Ma** received the B.E. degree in computer science from the South China University of Technology, Guangzhou, China, in 2019. He is currently pursuing the Ph.D. degree with the Department of Industrial Systems Engineering and Management, National University of Singapore, Singapore. His research interests include learning to optimize, deep reinforcement learning, evolutionary computation, and combinatorial optimization.

**Zhiguang Cao** received the Ph.D. degree from Interdisciplinary Graduate School, Nanyang Technological University. He received the B.Eng. degree in Automation from Guangdong University of Technology, Guangzhou, China, and the M.Sc. in Signal Processing from Nanyang Technological University, Singapore, respectively. He was a Research Fellow with the Energy Research Institute @ NTU (ERI@N), a Research Assistant Professor with the Department of Industrial Systems Engineering and Management, National University of Singapore, and a Scientist with the Agency for Science Technology and Research (A*STAR), Singapore. He joins the School of Computing and Information Systems, Singapore Management University, as an Assistant Professor. His research interests focus on learning to optimize (L2Opt).

**Tianshu Yu** is now an assistant professor at the School of Data Science, The Chinese University of Hong Kong, Shenzhen, leading Learning Of Graph & Optimization (LOGO) lab. He is also an associate researcher at Shenzhen Institute of Artificial Intelligence and Robotics for Society. He obtained his Ph.D. degree at Arizona State University. His research interests are in machine learning for combinatorial and discrete problems.

**Jianhua Xiao** received the Ph.D. degree in system engineering from the Huazhong University of Science and Technology, China, in 2008. He is currently a Professor with the Research Centre of Logistics, Nankai University, Tianjin, China. His research interests include combinatorial optimization, bio-inspired computation, and logistics system optimization.

---

## APPENDIX A
## DETAILS OF ALGORITHMS AND HYPERPARAMETERS IN NCS

The details of *ImitationLearning*, *CurriculumLearning* and *InvariantTransform* used as functions in Algorithm 3 (main paper) are presented in Algorithm 5, 6 and 7, respectively. In Algorithm 6, $\Gamma_i, \Gamma_d, \Theta_m, \Theta_i, \Upsilon_m, \Upsilon_i$ are all hyperparameters which are used to regulate the curriculum learning phase. In Algorithm 7, $\Psi_m, \Psi_w, \Psi_b$ are hyperparameters which are used to regulate the number of iterations. The setup of hyperparameters in curriculum learning and imitation learning of NCS are summarized in Table 9.

**TABLE 9: Setup of hyperparameters in NCS**

| NCS stage | Hyper parameter | Problem size $|V|$ | | |
| :--- | :--- | :--- | :--- | :--- |
| | | 21 | 51 | 101 |
| High-temperature CL | $\Gamma_i$ | 10 | 10 | 10 |
| ($e < e_{ts}$) | $\Gamma_d$ | 0.94 | 0.93 | 0.92 |
| Multi-sample CL | $\Theta_m$ | 128 | 256 | 512 |
| ($e \ge e_{ts}$) | $\Theta_i$ | 1.1 | 1.2 | 1.3 |
| Improvement CL | $\Upsilon_m$ | / | / | 25 |
| ($|V| > 100 \text{ and } e \ge e_{ps}$) | $\Upsilon_i$ | / | / | 2 |
| Imitation learning | $\Psi_m$ | 10 | 25 | 25 |
| | $\Psi_w$ | 1 | 1 | 1 |
| | $\Psi_b$ | 0 | 0 | -2 |

---
**Algorithm 5 Invariant Transform**
**Input:** Instance $\mathcal{I}$
1: $\mathcal{A}_i \leftarrow \textbf{RandomShuffle}([\text{flip-x-y, 1-x, 1-y, rotate}])$;
2: **for** each augment method $j \in \mathcal{A}_i$ **do**
3:     $\varrho_j \leftarrow \textbf{RandomConfig}(j)$;
4:     $\mathcal{I}' \leftarrow \text{perform augment } j \text{ on } \mathcal{I}_i \text{ with config } \varrho_j$;
5: **end for**
6: **return** $\mathcal{I}'$;
---

---
**Algorithm 6 Curriculum Learning**
**Input:** Instance batch $\mathcal{D}_b$, improvement policy $\pi_\theta$, construction policy $\pi_{\theta'}$, epoch $e$
1: $e_{ts} = \lceil \log_{\Gamma_d}(1/\Gamma_i) \rceil$, $e_{ps} = e_{ts} + \lceil \log_{\Theta_i} \Theta_m \rceil$;
2: **if** $e < e_{ts}$ **then**
3:     $\Gamma = \Gamma_i * (\Gamma_d)^e$;
4:     Sample one solution $\delta_0$ on each instance of $\mathcal{D}_b$ via $\pi_{\theta'}$ with modified Eq. (28):
       $\pi_{\theta'}(a_\tau = i|s_\tau) = \text{softmax}(\{u_i/\Gamma\}_{i=0}^{2n+1})$;
5: **else**
6:     $\Theta = \min(\Theta_m, (\Theta_i)^{e-e_{ts}})$;
7:     Sample $\Theta$ solutions and pick the shortest one as $\delta_0$ on each instance of $\mathcal{D}_b$ via $\pi_{\theta'}$;
8:     **if** problem size $> 100$ and $e \ge e_{ps}$ **then**
9:         $\Upsilon = \min(\Upsilon_m, (e - e_{ps})/\Upsilon_i)$;
10:        Improve $\delta_0$ for $\Upsilon$ steps via $\pi_\theta$;
11:    **end if**
12: **end if**
13: **return** $\delta_0$;
---

---
**Algorithm 7 Imitation Learning**
**Input:** construction policy $\pi_{\theta'}$, imitation solution $\delta^*$, constructed solution $\delta'$, Instance batch $\mathcal{D}_b$, epoch $e$
1: $\Psi = \max(0, \min(\Psi_m, \lfloor \Psi_w \cdot e + \Psi_b \rfloor))$;
2: $\xi = 1$ $if$ $f(\delta^*) < f(\delta')$ $else$ $0$;
3: **for** $i = 1, ..., \Psi$ **do**
4:     $\mathcal{D}'_b \leftarrow \textbf{InvariantTransform}(\mathcal{D}_b)$;
5:     Get $\pi_{\theta'}(\delta^* | s'_0)$ on $\mathcal{D}'_b$;
6:     Compute imitation loss $J'_{IL}$ using Eq. (31);
7:     $\theta' \leftarrow \theta' - \eta_{\theta'} \nabla J'_{IL}$;
8: **end for**
9: **return** updated policy $\pi_{\theta'}$
---

## APPENDIX B
## FULL RESULTS OF GENERALIZATION

We present more details for our generalization evaluation on benchmark datasets. For Heter-POMO-A, N2S-A and NCS-A, the coordinates of instances are normalized to $[0, 1] \times [0, 1]$ as per training, and we adopted the “closest” model learned in Section 6 to infer the instances, e.g., models trained on PDP-21 are used to infer PDP-25 instances, and models trained on PDP-101 are used to infer instances with $|V| \ge 101$. We increase $C$ to 10 in our N2S decoders during generalization since it boosts the performance. Full results of Table 8 are gathered in Table 10 (PDTSP) and Table 11 (PDTSP-LIFO). The gaps in Table 10 are computed w.r.t the optimal solutions and the gaps in Table 11 are computed w.r.t. heuristic baselines B1 [21] and B2 [22]. In all cases, our N2S-A and NCS-A significantly outperform Heter-POMO-A. In specific, our NCS-A achieves the best gaps of 0.00% ($|V| = 101$), and 1.93% ($|V| = 201$) on PDTSP while the gaps of Heter-POMO are 1.46% ($|V| = 101$) and 8.66% ($|V| = 201$); our NCS-A achieves best average gaps of 0.42% (w.r.t. B1) and 1.62% (w.r.t. B2) on PDTSP-LIFO while the gaps of Heter-POMO are 9.63% (w.r.t. B1) and 10.98% (w.r.t. B2). This further reveals the favorable generalization of our N2S and NCS over others.

## APPENDIX C
## DEALING WITH CAPACITY CONSTRAINT

Our N2S is generic to the capacity constraint, similar to DACT for handling capacity in CVRP [2]. Specifically, we can, 1) make copies of depots (i.e., dummy depots) so that N2S can search solutions with different numbers of vehicles automatically; 2) add capacity/demand features to NFEs; 3) mask out infeasible choices in the Reinsertion decoder; and 4) use diversity enhancement as usual since it only affects the node coordinates. Those procedures also hold for the improvement model in our NCS. Nevertheless, the capacity in PDP might not be so crucial as the vehicle may always alternatively load or unload the goods.

---

**TABLE 10: Generalization performance on benchmark instances from [19] for PDTSP using the trained model in Section 6.**

| Instances | $|V|$ | Optimal | Heter-POMO-A (3k) | | N2S-A (3k) | | NCS-A (3k) | | Heter-POMO-A (3k) | | N2S-A (3k) | | NCS-A (3k) | |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| | | | Avg. Cost | Best Cost | Avg. Cost | Best Cost | Avg. Cost | Best Cost | Avg. Gap(%) | Best Gap(%) | Avg. Gap(%) | Best Gap(%) | Avg. Gap(%) | Best Gap(%) |
| N101P1 | 101 | 799 | 824 | 820 | **800** | **799** | **800** | **799** | 3.13 | 2.63 | **0.13** | **0.00** | **0.13** | **0.00** |
| N101P2 | 101 | 729 | 736 | 735 | **730** | **729** | **730** | **729** | 0.96 | 0.82 | **0.14** | **0.00** | **0.14** | **0.00** |
| N101P3 | 101 | 748 | 751 | 751 | **748** | **748** | **748** | **748** | 0.40 | 0.40 | **0.00** | **0.00** | **0.00** | **0.00** |
| N101P4 | 101 | 807 | 815 | 814 | **808** | **807** | **807** | **807** | 0.99 | 0.87 | **0.12** | **0.00** | **0.00** | **0.00** |
| N101P5 | 101 | 783 | 794 | 791 | **783** | **783** | **783** | **783** | 1.40 | 1.02 | **0.00** | **0.00** | **0.00** | **0.00** |
| N101P6 | 101 | 755 | 766 | 763 | **755** | **755** | **755** | **755** | 1.46 | 1.06 | **0.00** | **0.00** | **0.00** | **0.00** |
| N101P7 | 101 | 767 | 787 | 787 | 768 | **767** | **767** | **767** | 2.61 | 2.61 | 0.13 | **0.00** | **0.00** | **0.00** |
| N101P8 | 101 | 762 | 783 | 782 | 764 | **762** | **762** | **762** | 2.76 | 2.62 | 0.26 | **0.00** | **0.00** | **0.00** |
| N101P9 | 101 | 766 | **766** | **766** | **766** | **766** | **766** | **766** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** |
| N101P10| 101 | 754 | 774 | 773 | **754** | **754** | **754** | **754** | 2.65 | 2.52 | **0.00** | **0.00** | **0.00** | **0.00** |
| Average| | 767.0* | 779.6 | 778.2 | 767.6 | **767.0*** | **767.2** | **767.0*** | 1.64 | 1.46 | 0.08 | **0.00** | **0.03** | **0.00** |
| N201P1 | 201 | 1039 | 1109 | 1095 | 1068 | 1059 | **1054** | **1049** | 6.74 | 5.39 | 2.79 | 1.92 | **1.44** | **0.96** |
| N201P2 | 201 | 1086 | 1117 | 1114 | **1102** | **1099** | 1103 | 1100 | 2.85 | 2.58 | **1.47** | **1.20** | 1.57 | 1.29 |
| N201P3 | 201 | 1070 | 1198 | 1177 | 1112 | 1107 | **1110** | **1106** | 11.96 | 10.00 | 3.93 | 3.46 | **3.74** | **3.36** |
| N201P4 | 201 | 1050 | 1114 | 1108 | 1073 | **1066** | **1069** | 1067 | 6.10 | 5.52 | 2.19 | **1.52** | **1.81** | 1.62 |
| N201P5 | 201 | 1052 | 1187 | 1169 | 1094 | 1078 | **1066** | **1063** | 12.83 | 11.12 | 3.99 | 2.47 | **1.33** | **1.04** |
| N201P6 | 201 | 1059 | 1120 | 1111 | **1077** | **1072** | 1085 | 1080 | 5.76 | 4.91 | **1.70** | **1.23** | 2.46 | 1.98 |**1.38** | **0.15** |
| d18512 | 25 | 4683.4 | 4672.0 | 4707 | 4705 | 4680 | **4672** | **4672** | **4672** | 0.50 | 0.46 | -0.07 | **-0.24** | **-0.24** | **-0.24** | 0.75 | 0.71 | 0.17 | **0.00** | **0.00** | **0.00** |
| | 51 | 7565.6 | 7502.0 | 8215 | 8126 | 7696 | 7519 | **7627** | 7532 | 8.58 | 7.41 | 1.72 | **-0.62** | **0.81** | -0.44 | 9.50 | 8.32 | 2.59 | 0.23 | **1.67** | 0.40 |
| | 75 | 8781.5 | 8629.0 | 10282| 10215| **8884** | **8802** | 8989 | 8946 | 17.09 | 16.32 | **1.17** | **0.23** | 2.36 | 1.87 | 19.16 | 18.38 | **2.96** | **2.00** | 4.17 | 3.67 |
| | 101| 10332.4| 10256.4| 13499| 13218| **10729**| **10555**| 10928| 10776| 30.65 | 27.93 | **3.84** | **2.15** | 5.76 | 4.29 | 31.62 | 28.88 | **4.61** | **2.91** | 6.55 | 5.07 |
| d15112 | 25 | 93981.0 | 93981.0 | **93981** | **93981** | **93981** | **93981** | **93981** | **93981** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** |
| | 51 | 143575.2| 142113.0| 144120| 143716| **142113**| **142113**| **142113**| **142113**| 0.38 | 0.10 | **-1.02** | **-1.02** | **-1.02** | **-1.02** | 1.41 | 1.13 | **0.00** | **0.00** | **0.00** | **0.00** |
| | 75 | 201385.4| 199047.8| 206442| 204944| **200004**| **199076**| 200084| **199076**| 2.51 | 1.77 | **-0.69** | **-1.15** | -0.65 | **-1.15** | 3.71 | 2.96 | **0.48** | **0.01** | 0.52 | **0.01** |
| | 101| 276876.8| 266925.3| 272784| 273693| 269160| 267305| **269135**| **267001**| -1.48 | -1.15 | -2.79 | -3.46 | **-2.80** | **-3.57** | 2.19 | 2.54 | 0.84 | 0.14 | **0.83** | **0.03** |
| nrw1379| 25 | 3194.8 | 3192.0 | **3192** | **3192** | **3192** | **3192** | **3192** | **3192** | **-0.09** | **-0.09** | **-0.09** | **-0.09** | **-0.09** | **-0.09** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** |
| | 51 | 5095.0 | 5055.0 | 6369 | 6517 | 5086 | 5056 | **5059** | **5055** | 25.00 | 27.91 | -0.18 | -0.77 | **-0.71** | **-0.79** | 25.99 | 28.92 | 0.61 | 0.02 | **0.08** | **0.00** |
| | 75 | 6865.1 | 6831.0 | 9647 | 9272 | 7081 | 6960 | **7005** | **6928** | 40.52 | 35.06 | 3.14 | 1.38 | **2.04** | **0.92** | 41.22 | 35.73 | 3.66 | 1.89 | **2.55** | **1.42** |
| | 101| 10197.5| 9889.4 | 13898| 13592| 10330| 9996 | **10147**| **10124**| 36.29 | 33.29 | 1.30 | **-1.98** | **-0.50** | -0.72 | 40.53 | 37.44 | 4.46 | **1.08** | **2.60** | 2.37 |
| **Average**| | 40880.4 | 40134.4 | 41713.8| 41593.6| 40501.0| 40280.7| **40450.2**| **40265.0**| 9.63 | 8.80 | 0.81 | -0.17 | **0.42** | -0.04 | 10.98 | 10.14 | 2.02 | 1.02 | **1.62** | 1.15 | 201 | 1079 | 1215 | 1196 | 1112 | 1108 | **1100** | **1096** | 12.60 | 10.84 | 3.06 | 2.69 | **1.95** | **1.58** |
| N201P9 | 201 | 1050 | 1208 | 1198 | **1076** | **1073** | 1094 | 1093 | 15.05 | 14.10 | **2.48** | **2.19** | 4.19 | 4.10 |
| N201P10| 201 | 1085 | 1198 | 1192 | 1116 | 1112 | **1109** | **1107** | 10.41 | 9.86 | 2.86 | 2.49 | **2.21** | **2.03** |
| Average| | 1060.7*| 1163.6 | 1152.4 | 1090.6 | 1083.9 | **1084.5** | **1081.2** | 9.71 | 8.66 | 2.82 | 2.19 | **2.24** | **1.93** |

**TABLE 11: Generalization performance on benchmark instances from [22] for PDTSP-LIFO using the trained model in Section 6.**

| Instances | $|V|$ | B1(2007) | B2(2011) | Heter-POMO-A (3k) | | N2S-A (3k) | | NCS-A (3k) | | Heter-POMO-A (3k) | | N2S-A (3k) | | NCS-A (3k) | | Heter-POMO-A (3k) | | N2S-A (3k) | | NCS-A (3k) | |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| | | | | Avg. Cost | Best Cost | Avg. Cost | Best Cost | Avg. Cost | Best Cost | Avg. Gap to B1(%) | Best Gap to B1(%) | Avg. Gap to B1(%) | Best Gap to B1(%) | Avg. Gap to B1(%) | Best Gap to B1(%) | Avg. Gap to B2(%) | Best Gap to B2(%) | Avg. Gap to B2(%) | Best Gap to B2(%) | Avg. Gap to B2(%) | Best Gap to B2(%) |
| brd14051 | 25 | 4682.2 | 4672.0 | 4705 | 4705 | 4680 | **4672** | **4672** | **4672** | 0.49 | 0.49 | -0.05 | **-0.22** | **-0.22** | **-0.22** | 0.71 | 0.71 | 0.17 | **0.00** | **0.00** | **0.00** |
| | 51 | 7763.2 | 7740.0 | 8276 | 8201 | 7948 | 7828 | **7845** | **7764** | 6.61 | 5.64 | 2.38 | 0.83 | **1.05** | **0.01** | 6.93 | 5.96 | 2.69 | 1.14 | **1.36** | **0.31** |
| | 75 | 7309.1 | 7232.4 | 9059 | 8938 | 7775 | **7554** | **7738** | 7639 | 23.94 | 22.29 | 6.37 | **3.35** | **5.87** | 4.51 | 25.26 | 23.58 | 7.50 | **4.45** | **6.99** | 5.62 |
| | 101| 10005.2| 9735.0 | 13315| 13000| 10539| **10370**| **10458**| 10404| 33.08 | 29.93 | 5.34 | **3.65** | **4.53** | 3.99 | 36.77 | 33.54 | 8.26 | **6.52** | **7.43** | 6.87 |
| pr1002 | 25 | 16221.0| 16221.0| **16221**| **16221**| **16221**| **16221**| **16221**| **16221**| **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** |
| | 51 | 31187.7| 30936.0| **30936**| **30936**| **30936**| **30936**| **30936**| **30936**| **-0.80** | **-0.80** | **-0.80** | **-0.80** | **-0.80** | **-0.80** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** |
| | 75 | 46911.0| 46673.0| 47404| 47202| 47284| 46923| **47067**| **46700**| 1.05 | 0.62 | 0.80 | 0.03 | **0.33** | **-0.45** | 1.57 | 1.13 | 1.31 | 0.54 | **0.84** | **0.06** |
| | 101| 63611.1| 61433.0| 62569| 62565| 62787| 62353| **62292**| **62096**| -1.64 | -1.64 | -1.30 | -1.98 | **-2.07** | **-2.38** | 1.85 | 1.84 | 2.20 | 1.50 | **1.40** | **1.08** |
| fnl4461 | 25 | 2168.0 | 2168.0 | **2168** | **2168** | **2168** | **2168** | **2168** | **2168** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** |
| | 51 | 4020.0 | 4020.0 | 4038 | 4037 | **4020** | **4020** | **4020** | **4020** | 0.45 | 0.42 | **0.00** | **0.00** | **0.00** | **0.00** | 0.45 | 0.42 | **0.00** | **0.00** | **0.00** | **0.00** |
| | 75 | 5865.0 | 5739.0 | 6118 | 6038 | 5905 | 5763 | **5775** | 5768 | 4.31 | 2.95 | 0.68 | **-1.74** | **-1.53** | -1.65 | 6.60 | 5.21 | 2.8
