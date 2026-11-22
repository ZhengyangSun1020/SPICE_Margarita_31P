# World Models in Machine Learning: First Principles and State of the Field (as of November 2026)

> **Verification note for the report writer.** In this session `arxiv.org` and `en.wikipedia.org` could not be fetched (blocked by the session's network allowlist), so no paper was opened in full. Each finding is tagged:
> - **[S]** = confirmed by a web-search result this session (snippet of the primary page or abstract; full text not read).
> - **[M]** = from the researcher's background knowledge of a well-known paper; the citation (authors, year, venue, arXiv ID) was not re-checked this session. These are standard, widely cited results, but arXiv IDs should be spot-checked before publication.
> - **[S2]** = from secondary/blog/news coverage only; treat as lower confidence.

---

## 1. What is the formal definition of a world model, and how does it relate to classical ideas?

### Takeaway
A world model is a learned (or designed) internal model of an environment consisting of (i) a state estimator/encoder that infers a compact state from a history of observations, (ii) a transition (dynamics) model p(s_{t+1} | s_t, a_t), and optionally (iii) an observation/decoder model p(o_t | s_t) and reward/cost model. This is exactly the structure of a (controlled) state-space model / POMDP; Kalman filtering is the linear-Gaussian special case, and model-based RL, predictive coding and the free-energy principle are the same idea approached from control, neuroscience and variational-inference directions. There is no single agreed definition in 2025-2026 surveys; the field splits into "understanding" (representation) and "predicting" (simulation) readings.

### Cited Findings

**Formal structure**
- The canonical latent-dynamics formulation (PlaNet's Recurrent State-Space Model, RSSM) has four learned components: a representation/posterior model q(s_t | s_{t-1}, a_{t-1}, o_t), a transition/prior model p(s_t | s_{t-1}, a_{t-1}), an observation model p(o_t | s_t), and a reward model p(r_t | s_t). It is trained by maximising a variational bound (ELBO) on the observation likelihood: reconstruction term minus a KL term between posterior and prior. The state is split into a deterministic recurrent part and a stochastic part. [M] — [Hafner et al., "Learning Latent Dynamics for Planning from Pixels", ICML 2019, arXiv:1811.04551](https://arxiv.org/abs/1811.04551)
- The underlying decision-theoretic object is the POMDP: the true state is hidden, and the agent maintains a belief state (a sufficient statistic of the action-observation history) updated by a Bayes filter; the world model is the pair (transition model, observation model) that makes this update possible. [M] — [Kaelbling, Littman & Cassandra, "Planning and acting in partially observable stochastic domains", Artificial Intelligence 101, 1998](https://doi.org/10.1016/S0004-3702(98)00023-X)
- LeCun's formulation: the world model is one module in a six-module architecture (configurator, perception, world model, cost, actor, short-term memory). Its roles are "(1) estimate missing information about the state of the world not provided by perception, (2) predict plausible future states of the world", including the outcomes of action sequences proposed by the actor; it must represent uncertainty via latent variables. [M] — [LeCun, "A Path Towards Autonomous Machine Intelligence", OpenReview, v0.9.2, June 2022](https://openreview.net/forum?id=BZ5a1r-kVsf)
- Ha & Schmidhuber decompose the agent into Vision (V: VAE compressing each frame to latent z), Memory (M: MDN-RNN modelling p(z_{t+1} | a_t, z_t, h_t)), and Controller (C: small linear policy). The world model is V+M; the controller can be trained entirely inside the model's "dream". [M] — [Ha & Schmidhuber, "World Models", arXiv:1803.10122 (NeurIPS 2018 as "Recurrent World Models Facilitate Policy Evolution")](https://arxiv.org/abs/1803.10122)
- 2025-2026 surveys agree there is no consensus definition. Ding et al. divide world models by two functions: "constructing internal representations to understand the mechanisms of the world" and "predicting future dynamics". [S] — [Ding et al., "Understanding World or Predicting Future? A Comprehensive Survey of World Models", arXiv:2411.14499 (ACM Computing Surveys)](https://arxiv.org/abs/2411.14499)
- A Nov 2025 survey states that "the definition, scope, architectures, and essential capabilities of world models remain ambiguous". [S] — [arXiv:2511.02097](https://arxiv.org/abs/2511.02097)
- The embodied-AI survey restricts the term to models that "yield actionable predictions" for agents, explicitly separating them from purely generative visual models, with a three-axis taxonomy (functionality; temporal modelling; spatial representation). [S] — [Li et al., "A Comprehensive Survey on World Models for Embodied AI", arXiv:2510.16732, Oct 2025](https://arxiv.org/abs/2510.16732)
- A 2026 survey frames world models as "internal simulators of environment structure and dynamics" with a category for "state-space and recurrent latent world models". [S] — [Zidan et al., "World Models: A Comprehensive Survey", arXiv:2606.00133](https://arxiv.org/abs/2606.00133)

**Kalman filters / state-space models**
- The Kalman filter is the exact Bayes filter for a linear-Gaussian state-space model x_{t+1} = A x_t + B u_t + w_t, y_t = C x_t + v_t; it alternates a predict step (dynamics model) and an update step (observation model). [M] — [Kalman, "A New Approach to Linear Filtering and Prediction Problems", J. Basic Engineering 82, 1960](https://doi.org/10.1115/1.3662552)
- Deep state-space models replace A, C with neural networks and exact inference with amortised variational inference: Deep Kalman Filters [M] — [Krishnan, Shalit & Sontag, arXiv:1511.05121, 2015](https://arxiv.org/abs/1511.05121); Embed to Control learns a latent space in which dynamics are locally linear so that classical optimal control applies [M] — [Watter et al., NeurIPS 2015, arXiv:1506.07365](https://arxiv.org/abs/1506.07365). PlaNet's RSSM is the direct descendant.

**Model-based RL**
- Dyna: learn a model from real experience, then use simulated ("imagined") experience from the model to update the value function/policy; this is the template for Dreamer's "learning in imagination". [M] — [Sutton, "Dyna, an integrated architecture for learning, planning, and reacting", SIGART Bulletin 2(4), 1991](https://doi.org/10.1145/122344.122377)
- Theoretical necessity claim: Richens, Abel, Bellot & Everitt prove that any agent able to generalise to a sufficiently broad set of multi-step goal-directed tasks must have learned a predictive model of its environment, that this model can be extracted from the agent's policy, and that "the error in this approximation decreases as we increase the agent's performance or the complexity of the goals it can achieve". [S] — [Richens et al., "General agents need world models", ICML 2025, PMLR 267 (arXiv:2506.01622, titled "General agents contain world models")](https://proceedings.mlr.press/v267/richens25a.html)
- This is a modern, quantitative version of the cybernetics "good regulator" theorem: every good regulator of a system must be a model of that system. [M] — [Conant & Ashby, Int. J. Systems Science 1(2), 1970](https://doi.org/10.1080/00207727008920220)
- A 2026 follow-up argues real agents may hold only partial world models and that the global-validity assumption is too strong. [S] — ["World Models in Pieces: Structural Certification for General Agents", arXiv:2606.24842](https://arxiv.org/abs/2606.24842)

**Craik's mental models**
- Craik (1943) proposed that an organism carries "a small-scale model of external reality and of its own possible actions within its head", allowing it to try out alternatives, react to future situations before they arise, and use knowledge of past events; this is the standard historical origin cited by Ha & Schmidhuber and LeCun. [M] — Craik, *The Nature of Explanation*, Cambridge University Press, 1943 (quoted in [Ha & Schmidhuber 2018](https://arxiv.org/abs/1803.10122))

**Predictive coding and the free-energy principle**
- Predictive coding: higher cortical areas send predictions of lower-level activity downward; only residual prediction errors are sent upward. A hierarchical network trained this way on natural images reproduces extra-classical receptive-field effects. [M] — [Rao & Ballard, "Predictive coding in the visual cortex", Nature Neuroscience 2, 1999](https://doi.org/10.1038/4580)
- Free-energy principle: an agent maintains a generative model of its sensory inputs and minimises variational free energy (an upper bound on surprise, i.e. negative log model evidence) through both perception (updating beliefs) and action (active inference). The variational free energy is the negative of the ELBO used to train VAE/RSSM world models. [M] — [Friston, "The free-energy principle: a unified brain theory?", Nature Reviews Neuroscience 11, 2010](https://doi.org/10.1038/nrn2787)
- LeCun's architecture is explicitly energy-based (not probabilistic): compatibility between observed x and predicted y is scored by an energy function with no requirement of normalisation, and he argues against forcing a normalised likelihood. [M] — [LeCun 2022](https://openreview.net/forum?id=BZ5a1r-kVsf)

### Inferences
- "World model" is best treated as a role, not an architecture: anything that supplies a state estimate plus an action-conditioned transition function qualifies. A Kalman filter with hand-specified A, B, C is a world model with a physics prior and zero learned parameters; an RSSM is the same graphical model with learned nonlinear maps; JEPA drops the observation model entirely.
- The decisive design axis is what the model is required to predict: observations (generative), latents (JEPA), or only task-relevant quantities such as reward/value/policy (MuZero). Section 3 covers this.
- For a signal-processing domain (e.g. spectroscopy), the classical state-space view is the natural bridge: hidden physical state (e.g. spin/metabolite state), known or partially known dynamics, and a measurement operator. "Action" becomes the acquisition/control input (pulse sequence, experimental condition), or is absent (pure sequence model).

### Gaps
- Could not open the survey PDFs, so exact formal definitions as written in the 2025-2026 surveys (notation, taxonomy trees) are not quoted verbatim beyond search snippets.
- Whether the Richens et al. theorem (stated for goal-conditioned agents in fully observed Markov environments) extends to partially observed settings was not verified.

---

## 2. Key milestones (1990 to late 2026)

### Takeaway
The lineage runs: Schmidhuber's controller+model RNNs (1990) and Dyna (1991) -> Ha & Schmidhuber (2018) -> RSSM-based PlaNet/Dreamer V1-V4 (2019-2025) and value-equivalent MuZero (2020) -> LeCun's JEPA programme (2022; I-JEPA, V-JEPA, V-JEPA 2/2.1, LeJEPA, LeWorldModel) -> large generative interactive video models (Genie 1-3, Sora, Cosmos, Marble). In 2025-2026 the field bifurcated commercially and technically between pixel-generating "world simulators" and latent-predictive JEPA models, with LeCun leaving Meta to found AMI Labs ($1.03B seed, March 2026).

### Cited Findings

**Early work**
- Schmidhuber (1990) trained a recurrent "model network" to predict the environment's next inputs and reward and a recurrent controller that learned by backpropagating through the differentiable model; he also introduced curiosity as reward for model prediction-error improvement. [M] — [Schmidhuber, "Making the World Differentiable", TR FKI-126-90, TU Munich, 1990](https://people.idsia.ch/~juergen/FKI-126-90_(revised)bw_ocr.pdf)
- Schmidhuber (2015) formalised an RNN controller querying an RNN world model in arbitrary learned ways ("learning to think"). [M] — [Schmidhuber, "On Learning to Think", arXiv:1511.09249](https://arxiv.org/abs/1511.09249)
- Ha & Schmidhuber (2018): VAE + MDN-RNN + tiny evolved controller; solved CarRacing-v0 and trained a VizDoom policy entirely inside the learned dream, using a temperature parameter to stop the controller from exploiting model errors. [M] — [arXiv:1803.10122](https://arxiv.org/abs/1803.10122)

**PlaNet / Dreamer**
- PlaNet (2019): RSSM + online planning by cross-entropy method in latent space; introduced latent overshooting. [M] — [Hafner et al., ICML 2019, arXiv:1811.04551](https://arxiv.org/abs/1811.04551)
- Dreamer V1 (2020): replaces online planning with an actor-critic trained by backpropagating analytic value gradients through imagined latent rollouts. [M] — [Hafner et al., "Dream to Control", ICLR 2020, arXiv:1912.01603](https://arxiv.org/abs/1912.01603)
- DreamerV2 (2021): discrete (categorical) latents and KL balancing; first world-model agent at human level on the Atari-200M benchmark. [M] — [Hafner et al., ICLR 2021, arXiv:2010.02193](https://arxiv.org/abs/2010.02193)
- DreamerV3: a single fixed hyperparameter configuration across >150 tasks (symlog predictions, two-hot returns, free bits, etc.); "to our knowledge, the first algorithm to collect diamonds in Minecraft from scratch without human data or curricula". [S] — [Hafner et al., "Mastering Diverse Domains through World Models", arXiv:2301.04104; published in Nature 2025 as "Mastering diverse control tasks through world models"](https://arxiv.org/abs/2301.04104)
- Dreamer 4 (Sept 2025): Hafner, Yan & Lillicrap. A scalable world model in which behaviours are learned by RL inside the model ("imagination training"); "the first agent to obtain diamonds in Minecraft purely from offline data, without environment interaction"; outperforms OpenAI's VPT offline agent "while using 100 times less data". [S] — [Hafner, Yan & Lillicrap, "Training Agents Inside of Scalable World Models", arXiv:2509.24527](https://arxiv.org/abs/2509.24527)

**MuZero and decision-aware models**
- MuZero learns a representation function, a dynamics function and a prediction function, trained only to predict reward, value and policy (no observation reconstruction), and plans with MCTS in that latent space; matched AlphaZero in Go/chess/shogi without being given the rules and set a state of the art on Atari. [M] — [Schrittwieser et al., "Mastering Atari, Go, chess and shogi by planning with a learned model", Nature 588, 2020, arXiv:1911.08265](https://arxiv.org/abs/1911.08265)
- TD-MPC2: decoder-free latent world model with joint-embedding consistency + reward + value losses, one hyperparameter set across 104 continuous-control tasks. [M] — [Hansen, Su & Wang, ICLR 2024, arXiv:2310.16828](https://arxiv.org/abs/2310.16828)

**LeCun's programme (JEPA)**
- Position paper (June 2022) proposes the Joint-Embedding Predictive Architecture (JEPA) and hierarchical H-JEPA, trained with non-contrastive regularised methods, as the route to world models; argues generative and probabilistic models should be abandoned for this purpose in favour of energy-based latent prediction. [M] — [LeCun 2022, OpenReview](https://openreview.net/forum?id=BZ5a1r-kVsf)
- I-JEPA (images): predict representations of masked target blocks from a context block; no pixel decoder, no hand-crafted augmentations. [M] — [Assran et al., CVPR 2023, arXiv:2301.08243](https://arxiv.org/abs/2301.08243)
- V-JEPA (video): feature prediction as the sole objective for video pretraining. [M] — [Bardes et al., TMLR 2024, arXiv:2404.08471](https://arxiv.org/abs/2404.08471)
- V-JEPA 2 (June 2025): two-stage recipe: action-free pretraining on internet video, then an action-conditioned predictor (V-JEPA 2-AC) post-trained on "less than 62 hours of unlabeled robot video" from DROID; deployed zero-shot on Franka arms in two labs for pick-and-place from image goals. [S] — [Assran et al., arXiv:2506.09985](https://arxiv.org/abs/2506.09985); Meta reports 65-80% success on pick-and-place of new objects in unseen environments [S] — [Meta AI blog](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/)
- V-JEPA 2.1 (March 2026): dense predictive loss on all tokens (visible and masked); authors report +20% real-robot grasping success over V-JEPA 2-AC and 10x faster planning for navigation (self-reported, not independently replicated). [S] — [arXiv:2603.14482](https://arxiv.org/abs/2603.14482)
- DINO-WM: world model built on frozen DINOv2 patch features, planning by latent goal-reaching without reconstruction. [M] — [Zhou et al., arXiv:2411.04983](https://arxiv.org/abs/2411.04983)
- LeJEPA (Nov 2025): Balestriero & LeCun derive that the optimal embedding distribution for JEPA is an isotropic Gaussian and enforce it with SIGReg (Sketched Isotropic Gaussian Regularisation), removing stop-gradient/EMA-teacher heuristics. [M] — [Balestriero & LeCun, arXiv:2511.08544](https://arxiv.org/abs/2511.08544)
- LeWorldModel (LeWM, March 2026; NeurIPS 2026 poster): end-to-end JEPA world model trained from pixels with only two loss terms: next-embedding prediction and the SIGReg Gaussian regulariser; authors report its latent space encodes meaningful physical structure. [S] — [arXiv:2603.19312](https://arxiv.org/abs/2603.19312); [NeurIPS 2026 poster](https://neurips.cc/virtual/2026/poster/150076)
- Theory (2026): "LeJEPA learns a linearly identifiable representation if and only if the latent variables are Gaussian". [S] — ["When Does LeJEPA Learn a World Model?", arXiv:2605.26379](https://arxiv.org/abs/2605.26379). Also a generalisation theory for JEPA world models [S, title only] — [arXiv:2606.27014](https://arxiv.org/abs/2606.27014)
- LeCun left Meta and co-founded AMI Labs (Advanced Machine Intelligence), headquartered in Paris with hubs in New York, Montreal and Singapore; it announced a $1.03B seed round on 10 March 2026 at a $3.5B pre-money valuation, with investors including Cathay Innovation, Greycroft, Hiro Capital, HV Capital, Bezos Expeditions, Nvidia and Samsung. [S2] — [SiliconANGLE, 10 Mar 2026](https://siliconangle.com/2026/03/10/yann-lecuns-new-startup-ami-labs-raises-1-03b-train-world-models/); [The Next Web](https://thenextweb.com/news/yann-lecun-ami-labs-world-models-billion). Some blogs give conflicting figures (e.g. EUR 500M at EUR 3B) [S2] — [Taskade blog](https://www.taskade.com/blog/ai-world-models)

**Generative video / interactive world models**
- Genie 1 (Feb 2024): 11B-parameter model trained on unlabeled internet gameplay video; learns a discrete latent action space without action labels (video tokenizer + latent action model + autoregressive dynamics model), yielding playable 2D environments from an image prompt. [M] — [Bruce et al., "Genie: Generative Interactive Environments", ICML 2024, arXiv:2402.15391](https://arxiv.org/abs/2402.15391)
- Sora (Feb 2024): OpenAI's technical report is titled "Video generation models as world simulators" and claims scaling video diffusion transformers is "a promising path towards building general purpose simulators of the physical world", while acknowledging failures such as incorrect physics for glass shattering. [M] — [OpenAI, 2024](https://openai.com/index/video-generation-models-as-world-simulators/)
- Genie 2 (Dec 2024): action-controllable 3D environments from a single image, consistent for up to roughly a minute. [M] — [Google DeepMind blog](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)
- NVIDIA Cosmos (Jan 2025): "world foundation model" platform (diffusion and autoregressive video models, tokenizers, curation pipeline) for physical AI: robotics and driving synthetic data. [M] — [NVIDIA, arXiv:2501.03575](https://arxiv.org/abs/2501.03575)
- Genie 3 (5 Aug 2025): text-prompted interactive worlds navigable in real time at 24 fps, 720p, consistent for a few minutes; "promptable world events" let users change the world state by text on the fly. Stated limitations: only a few minutes of interaction, no geographically accurate real locations. [S] — [Google DeepMind blog](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/); limitations per [TechRadar](https://www.techradar.com/ai-platforms-assistants/watch-this-google-genie-3-can-create-a-3d-world-let-you-explore-it-and-interact-with-it-in-real-time) and [Engadget](https://www.engadget.com/ai/google-deepminds-genie-3-can-dynamically-alter-the-state-of-its-simulated-worlds-140052124.html)
- 2026 (secondary reports only): Genie 3 exposed as "Project Genie" to Google AI Ultra subscribers; Waymo adopting Genie 3 for a driving simulation model (Feb 2026); World Labs (Fei-Fei Li) "Marble" generating exportable 3D environments from text/image/video/panorama; >$3B invested in world-model startups in H1 2026. [S2] — [tech-insider.org comparison](https://tech-insider.org/genie-3-vs-marble-vs-nvidia-cosmos-world-models-2026/); [introl.com](https://introl.com/blog/world-models-race-agi-2026)
- Other notable generative world models: GAIA-1 for driving [M] — [Hu et al., arXiv:2309.17080](https://arxiv.org/abs/2309.17080); UniSim [M] — [Yang et al., ICLR 2024, arXiv:2310.06114](https://arxiv.org/abs/2310.06114); DIAMOND, diffusion world model for Atari [M] — [Alonso et al., NeurIPS 2024, arXiv:2405.12399](https://arxiv.org/abs/2405.12399); GameNGen, diffusion model running DOOM in real time [M] — [Valevski et al., arXiv:2408.14837](https://arxiv.org/abs/2408.14837)

### Inferences
- Three distinct "ends" for a world model now coexist: (a) policy learning in imagination (Dreamer line), (b) planning by search/MPC in latent space (MuZero, TD-MPC2, V-JEPA 2-AC, DINO-WM, LeWM), (c) the model itself as product: interactive media, synthetic training data, simulation (Genie 3, Cosmos, Marble, Sora).
- Dreamer 4 blurs the generative/latent dichotomy: it is a large generative video-style model, yet is used for the classical agent purpose (imagination training). The dichotomy is thus about training objective, not use.
- 2026 is the first year the JEPA line has (i) a principled anti-collapse objective (LeJEPA/SIGReg), (ii) identifiability/generalisation theory, and (iii) dedicated billion-dollar funding. It remains less demonstrated at scale than generative video.

### Gaps
- No primary source was opened for Genie 3's public availability, Waymo's use, Marble's general-availability date, or AMI Labs' first model releases; all are secondary. I found no reliable source on any AMI Labs model released by November 2026.
- Sora 2 (2025) and Veo 3 "zero-shot reasoner" claims were not searched this session and are omitted from cited findings.
- Parameter counts and planning-speed numbers for LeWorldModel could not be confirmed (arXiv unreachable).

---

## 3. Generative (pixel-reconstructing) vs latent-predictive world models: the theoretical argument for latent prediction

### Takeaway
The argument for latent prediction is that observations contain a large amount of information that is unpredictable and/or irrelevant to any downstream decision; a likelihood or reconstruction loss in observation space forces capacity onto that information and cannot represent multi-modal futures without blurring or heavy generative machinery, whereas predicting in a learned representation space lets the encoder discard the unpredictable. The cost is that the trivial solution (constant representation) minimises the loss, so latent prediction needs an anti-collapse mechanism, and the model produces no human-inspectable output. Empirically neither side has won outright.

### Cited Findings

**The case for latent prediction**
- LeCun's argument: the world is only partially predictable; a generative model must predict every detail of y including irrelevant ones, whereas a JEPA predicts s_y = Enc(y) from s_x, and the encoder can learn invariances that eliminate unpredictable/irrelevant details. A JEPA is trained by (1) maximising information content of s_x about x, (2) maximising information content of s_y about y, (3) making s_y predictable from s_x, (4) minimising the information content of the latent variable z used for prediction. [M] — [LeCun 2022](https://openreview.net/forum?id=BZ5a1r-kVsf)
- Reconstruction is misaligned with perception: for natural images, the subspace explaining most pixel variance is nearly the least useful for classification, and features useful for perception are learned last under a reconstruction objective. [M] — [Balestriero & LeCun, "Learning by Reconstruction Produces Uninformative Features For Perception", ICML 2024, arXiv:2402.11337](https://arxiv.org/abs/2402.11337)
- In deep linear models, JEPA objectives preferentially learn high-influence (predictive, low-noise) features, while MAE-style reconstruction prioritises high-variance features even if noisy. [M] — [Littwin et al., "How JEPA Avoids Noisy Features: The Implicit Bias of Deep Linear Self Distillation Networks", NeurIPS 2024, arXiv:2407.03475](https://arxiv.org/abs/2407.03475)
- Value equivalence: for planning, a model need not match the true dynamics; it only needs to yield the same Bellman updates for the relevant policies/value functions. The set of value-equivalent models is much larger than the set of accurate ones, justifying MuZero-style models that never reconstruct observations. [M] — [Grimm, Barreto, Singh & Silver, "The Value Equivalence Principle for Model-Based Reinforcement Learning", NeurIPS 2020, arXiv:2011.03506](https://arxiv.org/abs/2011.03506)
- Bisimulation: an ideal state abstraction groups observations with identical reward and transition behaviour and ignores task-irrelevant details (e.g. background video distractors); reconstruction-based latents fail under such distractors. [M] — [Zhang et al., "Learning Invariant Representations for Reinforcement Learning without Reconstruction", ICLR 2021, arXiv:2006.10742](https://arxiv.org/abs/2006.10742)
- A 2026 survey reports a unified probing study showing "it is the prediction (rather than reconstruction) objective that drives the action-relevance of video-world-model latents". [S] — ["From Generation to Simulation: How Far Are World Models from Being True Simulators?", arXiv:2608.23070](https://arxiv.org/abs/2608.23070)
- Empirical point in a signal domain: Laya (LeJEPA applied to EEG) reports latent prediction beating reconstruction-based pretraining on clinical tasks using only 10% of the pretraining data. [S] — [arXiv:2603.16281](https://arxiv.org/abs/2603.16281)

**The collapse problem and its fixes**
- Joint-embedding predictive objectives admit a trivial solution: the encoder maps all inputs to (nearly) the same vector, making prediction error zero. Existing JEPA methods are described as "highly prone to collapse". [S] — [LeWorldModel, arXiv:2603.19312](https://arxiv.org/abs/2603.19312)
- Fixes, in historical order: contrastive negatives; asymmetric architectures with stop-gradient and an EMA target encoder (BYOL) [M] — [Grill et al., NeurIPS 2020, arXiv:2006.07733](https://arxiv.org/abs/2006.07733); explicit variance/covariance regularisation (VICReg) [M] — [Bardes, Ponce & LeCun, ICLR 2022, arXiv:2105.04906](https://arxiv.org/abs/2105.04906); distribution matching to an isotropic Gaussian (SIGReg in LeJEPA / LeWM) [S] — [arXiv:2603.19312](https://arxiv.org/abs/2603.19312); or freezing a pretrained encoder so collapse is impossible (DINO-WM, V-JEPA 2-AC) [S] — [arXiv:2506.09985](https://arxiv.org/abs/2506.09985)

**The case for generative / reconstruction models**
- Reconstruction provides a dense, collapse-free learning signal; the Dreamer family uses it and holds the broadest single-configuration results (DreamerV3; Dreamer 4). [S] — [arXiv:2301.04104](https://arxiv.org/abs/2301.04104); [arXiv:2509.24527](https://arxiv.org/abs/2509.24527)
- Interpretability/usability: "the JEPA route produces no readable pixel output, and its latent-vector feedback is opaque to non-learning downstream consumers". [S] — [arXiv:2608.23070](https://arxiv.org/abs/2608.23070)
- Critique of LeCun's position from the generative camp: Xing et al. argue against discarding generative/probabilistic modelling and propose a hierarchical, mixed continuous-discrete generative architecture (PAN) instead. [M] — [Xing et al., "Critiques of World Models", arXiv:2507.05169, 2025](https://arxiv.org/abs/2507.05169)
- Objective depends on the task: a 2026 comparison of masked-reconstruction and JEPA pretraining on 3D brain MRI concludes that the benefit of each objective depends on the structure of the downstream task. [S] — [arXiv:2606.13315](https://arxiv.org/abs/2606.13315)

### Inferences
- The latent-prediction argument is strongest when observations are dominated by nuisance variability (noise, texture, background, stochastic detail) relative to the state of interest, and weakest when nearly all observation content is state-relevant and a faithful decoder is itself a deliverable.
- "Latent vs pixel" is a continuum: Dreamer predicts in latent space but grounds latents by reconstruction; latent-diffusion video models predict in a reconstruction-trained VAE latent; JEPA predicts in a latent trained only by predictability plus an information-preserving regulariser; MuZero predicts in a latent trained only by task signals.
- For a measurement-physics domain where a known forward model maps state to signal, a hybrid is available that neither camp emphasises: predict in latent space but constrain/ground latents with the known observation operator (a physics decoder) rather than a learned pixel decoder. This is an inference, not a cited result.
- The 2026 identifiability result (linear identifiability iff Gaussian latents) implies SIGReg's Gaussian prior is a modelling assumption with consequences: if true generative factors are strongly non-Gaussian (e.g. sparse, bounded, discrete), guarantees do not directly apply.

### Gaps
- Did not read the proofs/assumptions of arXiv:2605.26379 or arXiv:2606.27014; statements are from abstracts via search snippets.
- No controlled, same-scale, same-data comparison of JEPA vs generative world models on downstream planning was found; the cited "unified probing study" was not located as a primary source.

---

## 4. Open debates and criticisms

### Takeaway
Three live debates: (1) whether sequence models (LLMs) trained on next-token prediction acquire coherent world models (probing says partly yes; stricter consistency tests say the recovered models are incoherent and fragile); (2) whether photorealistic video generators are world models (benchmarks show visual realism does not imply physical law-following, and they generalise by case-matching rather than rule induction); (3) how to evaluate a world model at all (probing vs behavioural consistency vs downstream control vs physics benchmarks), with benchmark validity itself now contested.

### Cited Findings

**Do LLMs / sequence models have world models?**
- Othello-GPT: a GPT trained only on legal Othello move sequences develops internal representations of the board state recoverable by (nonlinear) probes, and intervening on them changes predicted moves. [S] — [Li et al., "Emergent World Representations", ICLR 2023, arXiv:2210.13382](https://arxiv.org/abs/2210.13382)
- A 2025 re-examination across seven models of varying size reports the models "not only learn to play Othello, but also induce the Othello board layout". [S] — ["Revisiting the Othello World Model Hypothesis", arXiv:2503.04421](https://arxiv.org/abs/2503.04421)
- Chess/Othello follow-up notes probing techniques gave poor results on a model trained on human-played games, raising the question of whether robust world models require large synthetic datasets. [S] — [Karvonen, "Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models", arXiv:2403.15498](https://arxiv.org/abs/2403.15498)
- Counter-evidence: Vafa et al. propose metrics from the Myhill-Nerode theorem (sequence compression and sequence distinction) for tasks governed by a DFA. Models "do well on existing diagnostics for assessing world models, but our evaluation metrics reveal their world models to be far less coherent than they appear"; e.g. a transformer giving near-perfect shortest-path directions in Manhattan has an incoherent implicit street map and breaks under detours. [S] — [Vafa, Chen, Rambachan, Kleinberg & Mullainathan, "Evaluating the World Model Implicit in a Generative Model", NeurIPS 2024, arXiv:2406.03689](https://arxiv.org/abs/2406.03689)
- Inductive-bias probe: a transformer trained on orbital trajectories predicts orbits accurately yet, when fine-tuned to output forces, does not recover Newtonian gravitation but task-specific heuristics. [M] — [Vafa, Chang, Rambachan & Mullainathan, "What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models", ICML 2025, arXiv:2507.06952](https://arxiv.org/abs/2507.06952)
- LeCun's position: autoregressive LLMs lack grounded world models and planning; text is too low-bandwidth, and token-level generative prediction does not extend to high-dimensional continuous signals. [M] — [LeCun 2022](https://openreview.net/forum?id=BZ5a1r-kVsf)
- Richens et al. cut the other way: any sufficiently general goal-directed agent necessarily encodes a recoverable world model, whatever its training objective. [S] — [ICML 2025](https://proceedings.mlr.press/v267/richens25a.html)

**Is video generation a world model?**
- Kang et al. train video diffusion models on a 2D physics simulator: perfect in-distribution generalisation, reasonable combinatorial generalisation, failure out-of-distribution; models do not abstract general physical rules but exhibit "case-based" generalisation, mimicking the closest training example, and scaling alone does not fix this. [S] — [Kang et al., "How Far Is Video Generation from World Model: A Physical Law Perspective", ICML 2025, arXiv:2411.02385](https://icml.cc/virtual/2025/poster/46015)
- Physics-IQ: 396 eight-second real videos across 66 physical scenarios; across Sora, Runway Gen-3, Pika, Lumiere, Stable Video Diffusion and VideoPoet, "visual realism does not imply physical understanding". [S] — [Motamed et al., "Do generative video models understand physical principles?", arXiv:2501.09038 (WACV 2026)](https://arxiv.org/abs/2501.09038)
- Benchmark validity is itself contested: a 2026 audit "Physics-IQ Verified" refines 57.6% of samples and improves 34.8% of prompts. [S] — [arXiv:2606.18943](https://arxiv.org/abs/2606.18943). Newer diagnostic suites include WorldBench [S] — [arXiv:2601.21282](https://arxiv.org/abs/2601.21282)
- A physical-AI survey's summary: current video world models "capture benchmark-specific aspects of dynamics, but still lack uniformly strong physical consistency across evaluation settings". [S] — [arXiv:2510.04978](https://arxiv.org/abs/2510.04978)
- The embodied survey's definitional criticism: a world model must yield actionable (action-conditioned) predictions; pure text-to-video without action conditioning is a generative visual model, not a world model. [S] — [arXiv:2510.16732](https://arxiv.org/abs/2510.16732)
- A 2026 survey is devoted to the gap between generation and true simulation. [S, title/snippet only] — [arXiv:2608.23070](https://arxiv.org/abs/2608.23070)
- Known practical limits of interactive generative models: Genie 3 sustains only minutes of consistency; DeepMind states hours are needed. [S2] — [Engadget](https://www.engadget.com/ai/google-deepminds-genie-3-can-dynamically-alter-the-state-of-its-simulated-worlds-140052124.html)

**Evaluation**
- Four non-equivalent evaluation families are in use: (i) representation probing (Othello-GPT), (ii) behavioural/logical consistency (Vafa et al.), (iii) downstream control success (Dreamer, V-JEPA 2-AC), (iv) physics-plausibility benchmarks for generated video (Physics-IQ, WorldBench). Models routinely pass one family and fail another. [S] — [arXiv:2406.03689](https://arxiv.org/abs/2406.03689); [arXiv:2501.09038](https://arxiv.org/abs/2501.09038)
- Model exploitation: policies optimised inside a learned model find and exploit its errors; noted as early as Ha & Schmidhuber (mitigated with stochasticity/temperature). [M] — [arXiv:1803.10122](https://arxiv.org/abs/1803.10122)

### Inferences
- Much of the "do X have world models" disagreement is definitional: "has decodable state information" (weak), "has a state-transition structure consistent under all histories" (strong), and "supports counterfactual, action-conditioned planning" (functional) are different properties.
- Good next-step prediction is necessary but not sufficient evidence for a correct world model: the Vafa orbital example and Kang's case-based generalisation show accurate in-distribution prediction can coexist with wrong underlying laws. For scientific applications this is the central risk: a surrogate can interpolate measured regimes yet violate the governing physics off-distribution.
- Compounding error over long rollouts and uncalibrated uncertainty remain the main technical weaknesses common to both generative and latent families.

### Gaps
- No standard, community-accepted benchmark specifically for latent (non-generative) world models was found; they are evaluated mostly by downstream planning success and probing.
- Did not retrieve Morpheus benchmark findings or 2026 results for Sora 2/Veo 3-class models on physics benchmarks.

---

## 5. Are world models used in scientific/physical domains (physics surrogates, medical imaging, signal processing)?

### Takeaway
Yes, though usually under other names. Physics has "neural operators", "emulators/surrogates" and, since 2023, "PDE/physics foundation models": learned autoregressive transition models of physical state, i.e. action-free world models. Biomedicine is adopting the term explicitly in 2025-2026, mostly as perspectives plus JEPA-style representation learners for MRI, X-ray and EEG. I found no world-model or JEPA work on magnetic resonance spectroscopy (MRS) or NMR specifically.

### Cited Findings

**Physics simulation surrogates**
- Fourier Neural Operator: learns mappings between function spaces (solution operators of PDE families), resolution-independent, orders of magnitude faster than numerical solvers. [M] — [Li et al., ICLR 2021, arXiv:2010.08895](https://arxiv.org/abs/2010.08895)
- GraphCast: autoregressive learned transition model of global atmospheric state at 0.25 degrees, outperforming ECMWF HRES on about 90% of verification targets for 10-day forecasts. [M] — [Lam et al., Science 382, 2023, arXiv:2212.12794](https://arxiv.org/abs/2212.12794)
- Multiple Physics Pretraining: one transformer pretrained autoregressively across heterogeneous fluid systems transfers to unseen physics. [M] — [McCabe et al., NeurIPS 2024, arXiv:2310.02994](https://arxiv.org/abs/2310.02994)
- The Well: 15 TB collection of 16 datasets of diverse physics simulations for training such models. [M] — [Ohana et al., NeurIPS 2024 D&B, arXiv:2412.00568](https://arxiv.org/abs/2412.00568)
- PhysiX: 4.5B-parameter autoregressive generative model for physics simulation trained over 8 tasks of The Well, borrowing the tokenizer + autoregressive transformer recipe of video world models. [S] — [Nguyen et al., arXiv:2506.17774](https://arxiv.org/abs/2506.17774)
- "Towards a Physics Foundation Model" (Sept 2025) [S, title only] — [arXiv:2509.13805](https://arxiv.org/abs/2509.13805); MORPH, "a PDE foundation model designed to accommodate heterogeneous data across diverse physical phenomena", defined as a universal PDE surrogate/operator pretrained on multi-physics data [S] — [arXiv:2509.21670](https://arxiv.org/abs/2509.21670); PDE-FM reports a 46% mean VRMSE reduction vs prior operator-learning baselines across six domains (self-reported) [S] — [arXiv:2511.21861](https://arxiv.org/abs/2511.21861)
- JEPA-style world models with physics-structured latents exist in 2026 preprints. [S, titles only; content not verified] — ["Spectral-Target Physical Latent Structuring for JEPA-Style World Models", arXiv:2609.04264](https://arxiv.org/abs/2609.04264); ["World Modeling through Spectral Alignment", arXiv:2610.15429](https://arxiv.org/abs/2610.15429)
- JEPA for time series connected to dynamical systems theory (discovering underlying dynamical regimes). [S] — [arXiv:2511.09783](https://arxiv.org/abs/2511.09783)

**Biomedicine and medical imaging**
- A June 2026 perspective argues biomedical world models could serve as "data engines, environment simulators and scientific planning substrates across applications including virtual cells, organoids, virtual patients and surgical simulation", learning latent representations from "heterogeneous and partially observed biomedical data, including omics, imaging, physiological signals and clinical records"; it says application of these ideas to biomedicine "remains largely unexplored". [S] — [arXiv:2606.05925](https://arxiv.org/abs/2606.05925)
- The Zidan et al. survey lists scientific modelling and medical imaging among world-model application domains. [S] — [arXiv:2606.00133](https://arxiv.org/abs/2606.00133)
- JEPA-style (latent-predictive) medical/biosignal models:
  - Neuro-JEPA: latent-predictive foundation model with a Mixture-of-Experts ViT, pretrained on about 1.55M T1w/T2w/FLAIR brain MRI scans; reports outperforming existing neuroimaging foundation models. [S] — [arXiv:2606.14957](https://huggingface.co/papers/2606.14957)
  - COJEPA: JEPA + contrastive loss for volumetric T1w brain MRI, extending I-JEPA to 3D. [S] — [arXiv:2607.11962](https://arxiv.org/abs/2607.11962)
  - Masked vs JEPA SSL on 3D brain MRI: benefit depends on downstream task structure. [S] — [arXiv:2606.13315](https://arxiv.org/abs/2606.13315)
  - RadJEPA: chest X-ray encoder predicting latents of masked regions. [S] — [arXiv:2601.15891](https://arxiv.org/abs/2601.15891)
  - EEG-VJEPA: V-JEPA adapted to multichannel EEG treated as video-like sequences. [S] — [arXiv:2507.03633](https://arxiv.org/abs/2507.03633)
  - Laya: LeJEPA for EEG; latent prediction outperforms reconstruction pretraining on clinical tasks with 10% of pretraining data. [S] — [arXiv:2603.16281](https://arxiv.org/abs/2603.16281)
  - Clin-JEPA: JEPA over electronic-health-record trajectories. [S] — [arXiv:2605.10840](https://arxiv.org/abs/2605.10840)
- Chemistry: Mol-JEPA, "a scalable framework for learning molecular world models" (NeurIPS 2026) [S] — [NeurIPS 2026](https://neurips.cc/virtual/2026/177443); LeJEPA pretraining for molecular graph encoders [S] — [arXiv:2609.04261](https://arxiv.org/abs/2609.04261)
- Generative medical world models from memory (not re-verified): Medical World Model (tumour-evolution simulation for treatment planning) [M] — [arXiv:2506.02327](https://arxiv.org/abs/2506.02327); CheXWorld (image world modelling for radiograph representation learning) [M] — [CVPR 2025, arXiv:2504.13820](https://arxiv.org/abs/2504.13820)

**Signal processing**
- Classical signal processing already contains the world-model template: the Kalman filter is a recursive state estimator with an explicit dynamics and measurement model. [M] — [Kalman 1960](https://doi.org/10.1115/1.3662552). Deep variants learn those models from data. [M] — [Krishnan et al., arXiv:1511.05121](https://arxiv.org/abs/1511.05121)

### Inferences
- Most medical "JEPA" papers are representation learners (masked latent prediction within a static sample), not world models in the strict sense: there is no action/control variable and often no temporal transition. They inherit the latent-prediction argument (ignore noise/nuisance) but not the planning use. A true world model in imaging/spectroscopy would need a temporal or intervention axis (disease progression, treatment, acquisition parameters, dynamic experiments).
- Physics surrogates are the opposite: genuine transition models, but almost always predicting in observation (field) space with reconstruction losses; latent-predictive (JEPA) physics models are only starting to appear in 2026 preprints.
- MRS/NMR is an open niche: signals are low-dimensional (1D time-domain FIDs/spectra), noise-dominated, and governed by a well-understood parametric forward model (sums of damped complex exponentials, Bloch/Bloch-McConnell dynamics for exchange and saturation-transfer experiments). These are the conditions under which (a) latent prediction's noise-rejection argument applies and (b) a physics-informed state-space model is a strong competitor that any learned world model must beat.
- The Vafa-style caveat is especially relevant to scientific use: predictive accuracy does not certify that the latent state corresponds to physical quantities; identifiability must be tested, e.g. by probing latents for known physical parameters or checking consistency under known interventions.

### Gaps
- **No reliable source found applying world models or JEPA to MRS, 31P spectroscopy, or NMR**; one targeted search returned nothing. This is absence of evidence from a single query, not proof of absence.
- No ECG-specific JEPA result was returned.
- The content of the two "spectral" JEPA world-model papers (arXiv:2609.04264, 2610.15429) is unknown: "spectral" there may refer to eigen-spectra/frequency-domain structure of latents rather than measured spectra.
- Self-reported gains (PDE-FM 46%, Neuro-JEPA, Laya) are from preprints without independent replication.
- Session limitation: arxiv.org and en.wikipedia.org need to be added to the session's allowed network domains for full-text verification of any [S]/[M] item.
