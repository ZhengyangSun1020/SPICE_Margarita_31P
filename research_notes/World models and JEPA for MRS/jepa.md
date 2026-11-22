# JEPA (Joint-Embedding Predictive Architecture): first principles, mathematics, variants, theory, limits (as of Nov 2026)

> **Provenance note for the report writer (read first).** In this session `arxiv.org`, `ar5iv.labs.arxiv.org` and `machinelearning.apple.com` were not reachable by direct fetch (blocked by the session network allowlist); only web search worked. Every finding below is therefore tagged with one of:
> - **[S]** = confirmed this session from search-result text of the primary source (abstract-level in most cases).
> - **[M]** = stated from the researcher's prior knowledge of the cited primary paper; the link is the primary source, but the page was NOT re-read this session. Numbers tagged [M] should be spot-checked before being quoted as exact.
> - **[2nd]** = secondary source (blog/news/summary) only.
>
> Epistemic tags: **PROVEN** (theorem under stated assumptions), **EMPIRICAL** (measured), **ARGUMENT/SPECULATION** (position, intuition, or the note-writer's inference).

---

## 1. What exactly is JEPA? (EBM formulation, components, contrast with generative and contrastive methods)

### Takeaway
JEPA is an energy-based, non-generative architecture: encode x and y separately, and score compatibility by how well a predictor maps the embedding of x (plus optional latent/conditioning z) onto the embedding of y; the loss lives in representation space, so the target encoder is free to drop whatever it cannot or need not predict. It differs from generative models (loss in signal space) and from invariance-based joint-embedding methods (which force two augmented views to the *same* embedding with no conditioned predictor), and its central design problem is avoiding representation collapse.

### Cited Findings

**Position-paper formulation (LeCun 2022)**
- [M] ARGUMENT. LeCun's "A Path Towards Autonomous Machine Intelligence" (OpenReview, v0.9.2, 27 Jun 2022) defines JEPA as an energy-based model: s_x = Enc_x(x), s_y = Enc_y(y), energy E_w(x, y, z) = D(s_y, Pred(s_x, z)), and free energy F_w(x, y) = min_z E_w(x, y, z). Low energy = y is a compatible continuation/completion of x. It is explicitly *not* generative: it cannot directly produce y from x. — [LeCun 2022, OpenReview](https://openreview.net/forum?id=BZ5a1r-kVsf)
- [M] ARGUMENT. The stated advantage: the y-encoder can learn invariances that eliminate irrelevant/unpredictable details of y from s_y, so the model need not predict every detail of the signal; multi-modality of y given x is handled by (a) encoder invariance and (b) the latent variable z. — [LeCun 2022](https://openreview.net/forum?id=BZ5a1r-kVsf)
- [M] ARGUMENT. Because both encoders are trained, a trivial solution exists (constant s_x, s_y gives zero energy everywhere = collapse). LeCun proposes non-contrastive ("regularized") training with four criteria: maximize information content of s_x about x; maximize information content of s_y about y; make s_y predictable from s_x; minimize the information content of the latent z used for prediction. VICReg is named as a way to implement the information-maximization terms. Stacking JEPAs gives H-JEPA (hierarchical; higher levels predict more abstract states over longer horizons); the JEPA is the "world model" module of a larger cognitive architecture (configurator, perception, world model, cost, actor, short-term memory). — [LeCun 2022](https://openreview.net/forum?id=BZ5a1r-kVsf)

**Canonical instantiation (I-JEPA, masked-latent-prediction recipe)**
- [M] I-JEPA (Assran et al., CVPR 2023, arXiv:2301.08243) components: context encoder f_θ (ViT, sees only visible context patches), target encoder f_θ̄ (same architecture; weights are an exponential moving average of θ; receives the *full* image; no gradient), predictor g_φ (narrow ViT that takes context-encoder outputs + positional mask tokens for each target block). Loss: average squared L2 distance between predicted and target patch-level representations over M target blocks: L = (1/M) Σ_i Σ_{j∈B_i} ‖ŝ_yj − s_yj‖². Masking: 4 target blocks (scale 0.15–0.2, aspect 0.75–1.5) and one large context block (scale 0.85–1.0) with target regions removed. No hand-crafted view augmentations. — [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)
- [S] The I-JEPA paper states its predictions are made in representation space and that this is the key difference from MAE (which shares the encoder–predictor layout); computing the loss in representation space lets the target encoder discard irrelevant pixel detail; an ablation moving the loss to pixel space degraded linear-probe performance. — [I-JEPA, CVPR 2023 open-access PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.pdf)
- [M] EMPIRICAL. That ablation (ViT-L/16, 500 epochs, 1% ImageNet linear eval): target-encoder-output targets 66.9% vs pixel targets 40.7%. — [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)

**How it differs from other families**
- [M] *Generative / reconstruction* (MAE arXiv:2111.06377; diffusion; autoregressive): loss in input space, so the model must spend capacity on every detail including unpredictable noise/texture. JEPA: loss in latent space, with learned targets. — [MAE](https://arxiv.org/abs/2111.06377); [LeCun 2022](https://openreview.net/forum?id=BZ5a1r-kVsf)
- [M] *Joint-embedding / invariance-based* (SimCLR arXiv:2002.05709 contrastive with negatives; BYOL arXiv:2006.07733 and DINO arXiv:2104.14294 self-distillation with EMA teacher; VICReg arXiv:2105.04906 with variance/covariance regularizers): Siamese encoders are trained so two hand-augmented views of the same input map to the same/similar embedding. JEPA generalizes this: the predictor is *conditioned* on information about the transformation (mask positions, action, time offset, latent z), so the embedding need not be invariant—the predictor accounts for the difference. LeCun's taxonomy: JEA (joint-embedding), generative, JEPA. — [I-JEPA](https://arxiv.org/abs/2301.08243); [VICReg](https://arxiv.org/abs/2105.04906); [BYOL](https://arxiv.org/abs/2006.07733)
- [S] A 2026 survey-style paper frames the field as a move "from alignment to prediction" (SSL → predictive representation learning). (Abstract-level only.) — [arXiv:2604.13518](https://arxiv.org/pdf/2604.13518)
- [M] EMPIRICAL/ARGUMENT. Image World Models (Garrido et al., arXiv:2403.00504) show the JEPA predictor can be trained as a latent "world model" of photometric augmentations (conditioned on the transform parameters); the strength of the predictor controls where the representation sits on an invariance–equivariance spectrum. — [arXiv:2403.00504](https://arxiv.org/abs/2403.00504)

### Inferences
- JEPA is best thought of as an *architecture + energy definition*, not a specific loss: I-JEPA/V-JEPA (EMA + stop-grad), PLDM/MC-JEPA (VICReg), LeJEPA/LeWorldModel (SIGReg) and DINO-WM (frozen pretrained encoder) all satisfy the same "predict s_y from s_x in latent space" template but differ in the anti-collapse mechanism.
- Practical "JEPA" models to date are almost all *deterministic* predictors (z is only positional/action conditioning). The latent-variable z of the 2022 paper is mostly unrealized; a Nov-2026 preprint explicitly proposes "a stochastic latent variable generalization" and notes the deterministic I-JEPA predictor "produces no uncertainty" ([arXiv:2611.09302](https://arxiv.org/pdf/2611.09302), [S], abstract/snippet-level).

### Gaps
- The exact notation of LeCun 2022 (e.g., precise wording of the four criteria) was written from memory; OpenReview was not re-fetched.

---

## 2. Representation collapse: why it happens, how it is prevented, and what theory exists

### Takeaway
Collapse is the trivial minimizer of any latent-prediction loss when targets are learned (constant or low-rank embeddings give zero loss). Three families of fixes exist—asymmetric self-distillation heuristics (EMA teacher + stop-gradient + predictor), explicit second-moment regularization (VICReg variance/covariance), and full distribution matching (SIGReg in LeJEPA, pushing embeddings toward an isotropic Gaussian)—plus "sidestepping" by freezing a pretrained encoder (DINO-WM). Rigorous theory exists only for linear/idealized settings (Tian et al. 2021; Littwin et al. 2024) and for the *optimal target distribution* (LeJEPA); there is no proof that EMA+stop-grad avoids collapse in realistic nonlinear networks.

### Cited Findings

**Why collapse happens**
- [M] ARGUMENT. In a JEPA both s_x and s_y are learned, so mapping all inputs to a constant minimizes D(s_y, Pred(s_x)) trivially (complete collapse); "dimensional collapse" (embeddings span a low-dimensional subspace) is the partial version. — [LeCun 2022](https://openreview.net/forum?id=BZ5a1r-kVsf); [VICReg](https://arxiv.org/abs/2105.04906)

**Mechanism A: EMA target + stop-gradient + predictor (BYOL/SimSiam lineage; used by I-JEPA, V-JEPA, V-JEPA 2)**
- [M] EMPIRICAL. BYOL (arXiv:2006.07733) and SimSiam (arXiv:2011.10566) showed non-contrastive Siamese training avoids collapse with a predictor and stop-gradient (SimSiam: stop-gradient is essential; EMA is not strictly required). — [BYOL](https://arxiv.org/abs/2006.07733); [SimSiam](https://arxiv.org/abs/2011.10566)
- [M] PROVEN (linear setting only). Tian, Chen & Ganguli (ICML 2021, arXiv:2102.06810) analyze the learning dynamics of BYOL/SimSiam in linear networks and show how predictor, stop-gradient, EMA and weight decay jointly yield non-collapsed fixed points (and derive DirectPred, setting the predictor from the correlation eigenstructure). — [arXiv:2102.06810](https://arxiv.org/abs/2102.06810)
- [M] EMPIRICAL. V-JEPA uses an L1 latent regression loss with EMA target/stop-grad and gives only a heuristic argument for non-collapse (optimal predictor under L1 predicts a conditional median; encoder gradient pushes toward increasing information). — [V-JEPA, arXiv:2404.08471](https://arxiv.org/abs/2404.08471)
- [M] EMPIRICAL/CRITIQUE. C-JEPA (Mo & Tong, NeurIPS 2024, arXiv:2410.19560) argues I-JEPA's EMA is insufficient to fully prevent collapse and that I-JEPA's prediction poorly learns the mean of patch representations; adding VICReg terms improved stability and quality. — [arXiv:2410.19560](https://arxiv.org/abs/2410.19560)
- [M] EMPIRICAL. "Rethinking JEPA" / SALT (arXiv:2509.24317) reports that a *frozen* teacher (trained first with pixel reconstruction) can replace the EMA teacher for video JEPA, decoupling the anti-collapse mechanism from the student. — [arXiv:2509.24317](https://arxiv.org/abs/2509.24317)

**Mechanism B: VICReg-style variance/covariance regularization**
- [M] VICReg (Bardes, Ponce & LeCun, ICLR 2022, arXiv:2105.04906): loss = λ·invariance (MSE between embeddings) + μ·variance (hinge keeping the per-dimension std above a threshold γ=1 over the batch) + ν·covariance (penalize squared off-diagonal covariance entries). Needs no negatives, EMA, stop-grad, or weight sharing. — [arXiv:2105.04906](https://arxiv.org/abs/2105.04906)
- [M] Used in JEPA settings by MC-JEPA (arXiv:2307.12698) and JEPA world models such as PLDM (arXiv:2502.14819), which adds inverse-dynamics and temporal-smoothness terms—giving many loss hyperparameters. — [MC-JEPA](https://arxiv.org/abs/2307.12698); [PLDM](https://arxiv.org/abs/2502.14819)
- [S] LeWorldModel says it "reduces tunable loss hyperparameters from six to one compared to the only existing end-to-end alternative" (i.e., the VICReg-style multi-term recipe). — [arXiv:2603.19312](https://arxiv.org/abs/2603.19312v1)

**Mechanism C: SIGReg / LeJEPA (Balestriero & LeCun, arXiv:2511.08544, Nov 2025)**
- [S] The paper (v1 11 Nov 2025; v3 14 Nov 2025) "identif[ies] the isotropic Gaussian as the optimal distribution that JEPAs' embeddings should follow to minimize downstream prediction risk", introduces Sketched Isotropic Gaussian Regularization (SIGReg) to constrain embeddings to that distribution, and combines the JEPA predictive loss with SIGReg. Claimed properties: single trade-off hyperparameter, linear time and memory complexity, stability across architectures/domains, heuristics-free (no stop-gradient, no teacher–student, no schedulers). — [arXiv:2511.08544](https://arxiv.org/abs/2511.08544)
- [S, via secondary summaries of the paper] PROVEN under the paper's assumptions: for linear probes (OLS/ridge) and common nonlinear probes (k-NN, kernel regression), isotropic embeddings minimize estimator bias and variance; anisotropy inflates them. SIGReg applies the Epps–Pulley (characteristic-function) normality test statistic to random 1-D projections of the embeddings; Epps–Pulley was chosen because it gives bounded loss, gradients and curvature. Single parameter λ balances prediction and SIGReg. — [alphaXiv overview of 2511.08544](https://www.alphaxiv.org/abs/2511.08544); [arxiviq summary](https://arxiviq.substack.com/p/lejepa-provable-and-scalable-self)
- [M] The reduction from a d-dimensional distribution-matching problem to 1-D tests on random directions is justified via the Cramér–Wold theorem (a distribution is determined by all its 1-D projections); loss form L = (1−λ)·L_pred + λ·SIGReg, with L_pred pulling each view's embedding toward the mean embedding of the global views; core implementation ≈50 lines; validated on 60+ architectures and 10+ datasets. (This session's search could *not* confirm the Cramér–Wold wording; verify in the PDF.) — [arXiv:2511.08544](https://arxiv.org/abs/2511.08544)
- [S] EMPIRICAL. ImageNet-1k pretraining + frozen-backbone linear evaluation: 79% with ViT-H/14. In-domain LeJEPA pretraining on small domain datasets (e.g., Galaxy10, ~11k images) is reported to beat transfer from DINOv2/DINOv3 (secondary summary—check paper figures). — [arXiv:2511.08544](https://arxiv.org/abs/2511.08544); [alphaXiv](https://www.alphaxiv.org/abs/2511.08544)
- [S] Official code: `pip install lejepa`. — [github.com/rbalestr-lab/lejepa](https://github.com/rbalestr-lab/lejepa)
- [S] LeWorldModel (Maes, Le Lidec, Scieur, LeCun, Balestriero; arXiv:2603.19312, 13 Mar 2026; NeurIPS 2026 poster) transfers SIGReg to action-conditioned world models: "the first JEPA that trains stably end-to-end from raw pixels using only two loss terms: a next-embedding prediction loss and a regularizer enforcing Gaussian-distributed latent embeddings"; ~15M parameters, single GPU, a few hours; plans up to 48× faster than foundation-model-based world models while remaining competitive on 2D/3D control tasks. — [arXiv:2603.19312](https://arxiv.org/abs/2603.19312v1); [NeurIPS 2026 poster](https://neurips.cc/virtual/2026/poster/150076)
- [M] Related: "Gaussian Embeddings: How JEPAs Secretly Learn Your Data Density" (Balestriero, Ballas, Rabbat, LeCun; arXiv:2510.05949) argues the anti-collapse term makes a trained JEPA implicitly estimate the data density (recoverable from the encoder Jacobian). — [arXiv:2510.05949](https://arxiv.org/abs/2510.05949)

**Mechanism D: sidestep with a frozen encoder**
- [M] DINO-WM (Zhou, Pan, LeCun, Pinto; arXiv:2411.04983; ICML 2025) trains only a latent dynamics predictor on frozen DINOv2 patch features; collapse is impossible because targets are fixed. — [arXiv:2411.04983](https://arxiv.org/abs/2411.04983)

**2026 work showing collapse is not a solved problem (all [S], abstract/snippet-level only, preprints, not verified in full)**
- "Slow-feature"/temporal collapse: MotionJEPA argues standard JEPA training favors slow features and suppresses fast ones, adding static + dynamic terms. — [arXiv:2609.23881](https://arxiv.org/abs/2609.23881)
- H-JEPA (2026) formally defines "slow-feature collapse" (share of latent variance that is within-episode) and uses an inverse-dynamics loss as the main lever. — [arXiv:2610.06805](https://arxiv.org/html/2610.06805)
- AVL-JEPA argues existing anti-collapse terms guard only global statistics (variance, rank) and miss selective loss of causal-dynamics information. — [arXiv:2610.03587](https://arxiv.org/pdf/2610.03587)
- "The JEPA Paradox in Language": deterministic latent prediction can lower error by mapping inputs into a lower-dimensional space. — [arXiv:2607.23531](https://arxiv.org/pdf/2607.23531)
- Further regularizer variants: λ-JEPA (spectral anti-collapse) [arXiv:2609.35288](https://arxiv.org/pdf/2609.35288); SPHERE-JEPA (hyperspherical/homogeneous embeddings) [arXiv:2605.26900](https://arxiv.org/pdf/2605.26900).

### Inferences
- Hierarchy of guarantees: EMA+stop-grad = empirical heuristic (theory only in linear models); VICReg = constrains 1st/2nd moments only (prevents complete and dimensional collapse but not higher-order degeneracy); SIGReg = matches the whole distribution, with a proof about *which* distribution is optimal for downstream probes—but that proof concerns downstream risk given the embedding distribution, not a guarantee that the learned features are the semantically right ones.
- "Variance/rank is fine" ≠ "information is preserved": the 2026 collapse papers (AVL-JEPA, MotionJEPA, H-JEPA) all describe *informational* collapse where global statistics look healthy while specific (fast, causal, small) factors are dropped. This matters for any application where the signal of interest is a small component.
- For small scientific datasets, LeJEPA's reported in-domain result (Galaxy10) and its single hyperparameter make it the most plausible starting recipe; LeWorldModel shows the same regularizer working at 15M parameters.

### Gaps
- Exact LeJEPA theorem statements (assumptions, constants), number of slices, default λ, and Galaxy10 numbers were not re-read from the PDF.
- No rigorous non-collapse proof for EMA/stop-grad JEPA in nonlinear networks was found.

---

## 3. Variants and their results

### Takeaway
The JEPA template has been ported to essentially every modality (image, video, audio, point clouds, graphs, EEG, fMRI, time series, language, vision-language, robot world models, medical imaging). The strongest, best-validated results are in video (V-JEPA 2 / 2.1) and latent-space planning (V-JEPA 2-AC, DINO-WM, PLDM, LeWorldModel); many domain ports are single-paper claims, and in 3D brain MRI a 2026 head-to-head found MAE generally ahead of baseline JEPA.

### Cited Findings

**Images**
- [M] I-JEPA (Assran et al., CVPR 2023): ViT-H/14 trained on ImageNet with 16 A100 GPUs in <72 h; ImageNet linear probe 79.3% (ViT-H/14, 300 ep) and 81.1% (ViT-H/16 at 448px), vs MAE ViT-H/14 at 77.2% (1600 ep); strong low-shot results; no hand-crafted augmentations. — [arXiv:2301.08243](https://arxiv.org/abs/2301.08243). (A search result flagged a "79.3 vs 81.1" discrepancy between secondary sites; from memory both are in the paper for different model/resolution—verify.)
- [S] LeJEPA: 79% ImageNet-1k linear probe with ViT-H/14. — [arXiv:2511.08544](https://arxiv.org/abs/2511.08544)
- [S] DMT-JEPA reports I-JEPA trailing it by 0.9 mIoU on ADE20K and 1.0/1.1 AP on COCO detection/instance segmentation (follow-up paper's own comparison). — [arXiv:2405.17995](https://arxiv.org/pdf/2405.17995)

**Video**
- [M] V-JEPA (Bardes et al., TMLR 2024, arXiv:2404.08471): feature prediction as the sole objective on 2M videos (VideoMix2M), no pretrained image encoder, text, negatives, or pixel reconstruction; frozen ViT-H/16 with attentive probe: 81.9% Kinetics-400, 72.2% SSv2, 77.9% ImageNet-1k. — [arXiv:2404.08471](https://arxiv.org/abs/2404.08471)
- [S] V-JEPA 2 (Assran et al., arXiv:2506.09985, 11 Jun 2025): 77.3 top-1 on Something-Something v2 (attentive probe). V-JEPA 2-AC: an action-conditioned latent world model post-trained on <62 hours of unlabeled Droid robot video, deployed zero-shot on Franka arms in two labs for pick-and-place by planning with image goals. — [arXiv:2506.09985](https://arxiv.org/abs/2506.09985)
- [S][2nd for the number] Meta's blog gives 65–80% success for pick-and-place of new objects in unseen environments. — [Meta AI blog](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/)
- [M] V-JEPA 2 details: pretraining on >1M hours of internet video with a ~1B-param ViT-g; Epic-Kitchens-100 action anticipation 39.7 recall@5; after LLM alignment, video-QA results such as 84.0 PerceptionTest and 76.9 TempCompass; V-JEPA 2-AC uses a ~300M-param predictor on the frozen encoder, trained with teacher-forcing + rollout losses, and plans with the cross-entropy method by minimizing L1 distance between predicted and goal-image embeddings (~16 s/action vs ~4 min for a video-generation baseline). — [arXiv:2506.09985](https://arxiv.org/abs/2506.09985)
- [S] V-JEPA 2.1 (Mur-Labadia et al., Meta FAIR + Univ. Zaragoza; arXiv:2603.14482, 15 Mar 2026): argues dense spatial/temporal features do not emerge reliably when the loss covers only masked tokens; introduces a Dense Predictive Loss where visible *and* masked tokens contribute. Reported: 7.71 mAP Ego4D short-term object-interaction anticipation, 40.8 Recall@5 EPIC-KITCHENS, 0.307 RMSE NYUv2 linear-probe depth; releases ViT-g/G (1B/2B) and distilled ViT-B/L. — [arXiv:2603.14482](https://arxiv.org/abs/2603.14482)
- [S] MC-JEPA (Bardes, Ponce, LeCun; arXiv:2307.12698): shared encoder jointly learns optical flow and content features; DAVIS-2017 video segmentation 70.5 vs 69.9 for DINO; optical flow on par with some unsupervised methods (UFlow) but below SMURF on all flow benchmarks. — [arXiv:2307.12698](https://arxiv.org/pdf/2307.12698)

**World models / planning**
- [M] DINO-WM (arXiv:2411.04983, ICML 2025): predicts future DINOv2 patch features from offline trajectories; zero-shot goal-reaching by test-time MPC/CEM in latent space; no pixel reconstruction. — [arXiv:2411.04983](https://arxiv.org/abs/2411.04983)
- [M] PLDM (Sobal et al., arXiv:2502.14819): JEPA latent dynamics trained on reward-free offline data with VICReg-style regularization + planning; reported to generalize better to unseen layouts/suboptimal data than goal-conditioned RL baselines. — [arXiv:2502.14819](https://arxiv.org/abs/2502.14819)
- [S] LeWorldModel (arXiv:2603.19312, Mar 2026; NeurIPS 2026): see §2. — [arXiv:2603.19312](https://arxiv.org/abs/2603.19312v1)
- [S] 2026 follow-ups (abstract-level only): hierarchical H-JEPA world models [arXiv:2610.06805](https://arxiv.org/html/2610.06805); RoboJEPA "scaling robotic latent world models" [arXiv:2610.10515](https://arxiv.org/pdf/2610.10515); "Keeping JEPA World Models Plannable When Little of the Frame Moves" [arXiv:2610.03137](https://arxiv.org/pdf/2610.03137); "I Act Therefore I Am: When Is JEPA's Action-Conditioning Enough to Learn Causal Mechanisms?" [arXiv:2609.31161](https://arxiv.org/pdf/2609.31161).

**Other modalities**
- [M] A-JEPA (Fei, Fan, Huang; arXiv:2311.15830): I-JEPA on audio spectrograms with time–frequency-aware curriculum masking. — [arXiv:2311.15830](https://arxiv.org/abs/2311.15830)
- [S] Point-JEPA (Saito et al.; arXiv:2404.16432; 2024 preprint, cited as 2025—WACV 2025 per [M]): a lightweight sequencer orders point-patch embeddings; claims state-of-the-art and much faster pretraining convergence than comparable methods. — [arXiv:2404.16432](https://arxiv.org/abs/2404.16432)
- [S] Graph-JEPA (Skenderi et al., 2023; arXiv:2309.16014): predicts embeddings of masked subgraphs, optionally with a hyperbolic-coordinate target to encode hierarchy. — [awesome-jepa list](https://github.com/AbdelStark/awesome-jepa); [arXiv:2309.16014](https://arxiv.org/abs/2309.16014)
- [S] Name clash: "S-JEPA" refers both to Signal-JEPA for EEG/BCI (Guetschel et al., 2024; arXiv:2403.11772 per [M]) and to a skeleton-action S-JEPA (Abdelfattah & Alahi, 2024). — [awesome-jepa list](https://github.com/AbdelStark/awesome-jepa); [arXiv:2403.11772](https://arxiv.org/abs/2403.11772)
- [S] Brain-JEPA (NeurIPS 2024; arXiv:2409.19407): fMRI "brain dynamics foundation model" with JEPA, brain gradient positioning and spatiotemporal masking. — [arXiv:2409.19407](https://arxiv.org/pdf/2409.19407)
- [M] TS-JEPA: "Joint Embeddings Go Temporal" (arXiv:2509.25449; originally a NeurIPS 2024 time-series workshop paper)—JEPA adapted to time-series patches, reported competitive with masked-reconstruction/contrastive baselines on classification and forecasting. This session's search returned no direct hit; treat details as unverified. — [arXiv:2509.25449](https://arxiv.org/abs/2509.25449)
- [S] Other time-series JEPAs: MTS-JEPA (multi-resolution, anomaly prediction; arXiv:2602.04643); a 2026 systems paper ("Mask-Aware Execution for Efficient JEPA Training") benchmarks V-JEPA, EEG-JEPA, T-JEPA, TS-JEPA and Brain-JEPA and reports up to 1.7× training speedup; Distributed JEPA for energy forecasting. — [arXiv:2609.22674](https://arxiv.org/pdf/2609.22674); [arXiv:2609.17029](https://arxiv.org/pdf/2609.17029)
- [S] LLM-JEPA (Huang, LeCun, Balestriero; arXiv:2509.14252): adds a JEPA objective (predict the embedding of one "view", e.g., code, from another, e.g., text) to next-token training; e.g., Llama-3.2-1B-Instruct fine-tuning 71.46 ± 1.34% vs 57.29 ± 5.32% for next-token-only; more robust to overfitting; pretraining evidence preliminary. — [arXiv:2509.14252](https://arxiv.org/pdf/2509.14252)
- [M] VL-JEPA (arXiv:2512.10942, Dec 2025): vision-language JEPA that predicts continuous text embeddings rather than tokens. Not re-verified this session. — [arXiv:2512.10942](https://arxiv.org/abs/2512.10942)

**Medical imaging (2026; all [S], abstract-level)**
- "Masked and Predictive Self-Supervised Pretraining for Brain MRI: Effects of Objective Design on Disease Detection": on 3D structural brain MRI, "MAE-based pretraining generally outperforms baseline JEPA, with SL-MAE [spectral-loss MAE] providing the strongest gains in several disease-detection settings"; the authors also add variance–covariance regularization to JEPA; benefit is "conditioned on the task's structure". — [arXiv:2606.13315](https://arxiv.org/abs/2606.13315)
- A related preprocessing cost–utility study reports a similar pattern but JEPA ahead on some IDH-prediction results. — [arXiv:2606.08164](https://arxiv.org/html/2606.08164v1)
- COJEPA (contrastive + JEPA for volumetric structural MRI) states that no prior brain-MRI foundation model used joint-embedding predictive pretraining. — [arXiv:2607.11962](https://arxiv.org/pdf/2607.11962)
- MR-JEPA: cardiac-MRI video foundation model extending *LeJEPA* to 3D spatiotemporal inputs (tubelets, spatiotemporal masking), pretrained on cine/LGE/mapping data from 10,505 patients at two centers. — [arXiv:2608.30975](https://arxiv.org/abs/2608.30975v1)
- Rad-JEPA 3D for CT. — [arXiv:2607.26196](https://arxiv.org/pdf/2607.26196)

**AMI Labs / LeCun (2026)**
- [2nd] LeCun's Paris-based startup AMI Labs reportedly raised a ~$1.03B seed at ~$3.5B pre-money in March 2026 (one source says January 2026—sources conflict); one source says it had not yet shipped a model. LeWorldModel lists LeCun/AMI-affiliated authors. A news piece says LeCun's group posted further preprints in May–June 2026 on when JEPA world models learn faithful environment representations. — [DataCamp explainer](https://www.datacamp.com/blog/jepa-architecture-explained); [Taskade blog](https://www.taskade.com/blog/ai-world-models); [startuphub.ai](https://www.startuphub.ai/ai-news/ai-figures/2026/figure-yann-lecun-jepa-release-cadence-2026-08-20); critical commentary: [Medium/AIGuys](https://medium.com/aiguys/lecuns-world-model-is-real-useful-and-nowhere-near-as-new-as-the-funding-round-implies-ecc3b05e2204)

### Inferences
- Trend 2025→2026: (i) from EMA heuristics to SIGReg-style principled regularizers (LeJEPA → LeWorldModel → MR-JEPA); (ii) from global/semantic features toward *dense* features (V-JEPA 2.1 had to change the loss so visible tokens are supervised—an implicit admission that mask-only latent prediction under-trains local detail); (iii) from representation learning toward planning.
- Evidence quality varies sharply: V-JEPA 2 has real-robot evaluation in two labs; most modality ports (audio, graph, EEG, time series) are single-paper benchmarks without independent replication found here.

### Gaps
- No reliable primary source on AMI Labs' technical roadmap or products; funding figures are from secondary media and dates conflict.
- **No JEPA work on MR spectroscopy (MRS/MRSI, 1H or 31P) or NMR spectra was found** in any search this session.
- Quantitative results for A-JEPA, Graph-JEPA, Signal-JEPA, Brain-JEPA, TS-JEPA and DINO-WM/PLDM success rates were not retrieved.

---

## 4. Key theoretical property: predicting abstract representations and discarding unpredictable detail — formal statements, evidence, and what it implies for low-variance-but-important information

### Takeaway
The claim "JEPA discards unpredictable detail" has one rigorous backing—Littwin et al. (NeurIPS 2024), in deep *linear* self-distillation networks: JEPA learns features in order of their regression coefficient ρ (predictability/"influence"), whereas MAE learns in order of covariance λ (variance explained), and the JEPA bias only appears with deep encoders. Everything beyond that is empirical or argumentative, and there is a documented flip side: JEPA favors *slow/predictable* features even when they are nuisances (static distractors), and can drop fast, unpredictable-but-relevant factors.

### Cited Findings

**Formal result (PROVEN, deep linear setting)**
- [S] Littwin, Saremi, Advani, Thilak, Nakkiran, Huang, Susskind (Apple), "How JEPA Avoids Noisy Features: The Implicit Bias of Deep Linear Self Distillation Networks", arXiv:2407.03475, NeurIPS 2024. Motivation in the abstract: a common reason to prefer JEPA over MAE is that it "prioritizes abstract features over fine-grained pixel information (which can be unpredictable and uninformative)". — [arXiv:2407.03475](https://arxiv.org/abs/2407.03475); [NeurIPS 2024 poster](https://neurips.cc/virtual/2024/poster/94233)
- [S] Result: both objectives show greedy (stepwise) learning dynamics with respect to λ (feature covariance), but **only JEPA is greedy with respect to ρ** (the regression coefficient of the target feature on the input feature); "the order of feature learning is dictated by ρ for the JEPA objective, and λ for the MAE objective"; JEPA can *reverse* the learning order relative to MAE when ρ and λ trend oppositely. JEPA is biased toward "influential features" (large regression coefficient); MAE prioritizes highly covarying features "even if they are noisy and thus less informative". — [NeurIPS 2024 proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/a600f0f740605205133553cb74a1c107-Paper-Conference.pdf); [arXiv:2407.03475](https://arxiv.org/html/2407.03475v1)
- [S] Depth dependence: "this distinct bias takes hold only when the encoder is deep and diminishes substantially when the encoder is shallow." The paper characterizes a critical time t*(λ, ρ) at which each feature is learned. — [arXiv:2407.03475](https://arxiv.org/abs/2407.03475)
- [M] Setup as recalled: deep linear encoder W = W_L···W_1 with linear predictor; JEPA loss ‖V W x − StopGrad(W y)‖² vs MAE loss ‖V W x − y‖²; simultaneously diagonalizable Σ_xx and Σ_xy; small balanced initialization; gradient flow; per-feature λ_i = cov(x_i, y_i), ρ_i = λ_i / var(x_i). In a simple additive-noise model, per-feature noise lowers ρ but not λ—hence "JEPA avoids noisy features". Exact t* formulas were not retrievable this session. — [arXiv:2407.03475](https://arxiv.org/abs/2407.03475)

**Supporting evidence (EMPIRICAL / related theory)**
- [M] Balestriero & LeCun, "Learning by Reconstruction Produces Uninformative Features For Perception" (ICML 2024, arXiv:2402.11337): the directions explaining most pixel variance are the least useful for perception; e.g., TinyImageNet images projected on the top subspace explaining 90% of pixel variance give 45% supervised test accuracy, while the bottom subspace holding only 20% of variance gives 55%. This is the reconstruction-side complement to Littwin. (Search did not return this paper this session.) — [arXiv:2402.11337](https://arxiv.org/abs/2402.11337)
- [M] I-JEPA pixel-target ablation (66.9 → 40.7, §1) is the direct empirical counterpart. — [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)

**The flip side: slow-feature bias (PROVEN-in-toy + EMPIRICAL)**
- [S] Sobal, Jyothir S V, Jalagam, Carion, Cho, LeCun, "Joint Embedding Predictive Architectures Focus on Slow Features" (arXiv:2211.10831; NeurIPS 2022 SSL workshop, 4 pages): in a moving-dot environment with background distractors, JEPA (VICReg/SimCLR-trained) performs on par or better than reconstruction when distractor noise changes every time step, **but fails when the noise is fixed**; the paper gives a theoretical explanation—the objective favors slowly changing signals, so static background noise (perfectly predictable) can dominate the representation. — [arXiv:2211.10831](https://arxiv.org/abs/2211.10831)
- [S] 2026 preprints (abstract-level): MotionJEPA (slow features favored, fast suppressed) [arXiv:2609.23881](https://arxiv.org/abs/2609.23881); H-JEPA "slow-feature collapse" [arXiv:2610.06805](https://arxiv.org/html/2610.06805); AVL-JEPA "causal dynamics information collapse" [arXiv:2610.03587](https://arxiv.org/pdf/2610.03587).
- [S] "Predictive Closure: A Set-Level Characterization of JEPA's Feature Selection" (arXiv:2610.27364, Oct 2026): recasts the JEPA objective as selecting the *set* of features whose present values carry the most information about that set's own future, within the latent's regularization budget. Agrees with earlier per-feature views when features evolve independently, but predicts that an individually unpredictable feature can be retained if it helps predict others (e.g., a traffic light); reinterprets observations such as the Pong score being kept while the ball is suppressed; suggests auxiliary objectives or side information to the predictor can strengthen a targeted feature. The abstract itself notes benchmark evidence is limited. — [arXiv:2610.27364](https://arxiv.org/pdf/2610.27364)

### Inferences (note-writer's reasoning; SPECULATION unless stated)
- **What is selected is predictability, not variance.** Under Littwin et al., JEPA's ordering key is ρ (signal-to-total variance of a feature / how reliably it is predicted), MAE's is λ (absolute covariance). So a **low-variance but highly predictable** feature (small amplitude, but deterministically related between context and target) is *favored* by JEPA relative to MAE, which would learn it late or never. This is potentially good news for small-but-structured signals (e.g., a low-amplitude spectral peak whose presence is predictable from the rest of the spectrum/neighboring voxels).
- **Low-variance and weakly predictable from context = at risk.** A feature that is important but *not* predictable from the context view under the chosen masking/view design (e.g., an isolated, voxel-specific or time-specific small peak with no redundancy elsewhere in the input) has low ρ and will be learned last or dropped; anti-collapse regularizers (VICReg/SIGReg) fill the embedding dimensions with *some* information but do not guarantee it is that feature (cf. AVL-JEPA's point about global statistics).
- **Anti-collapse whitening decouples embedding variance from input variance.** VICReg/SIGReg normalize every embedding direction to unit variance, so a small-amplitude input factor can occupy a full embedding dimension if it is predictable—amplitude in signal space is not what limits it.
- **Structured, static nuisances are the main hazard, not i.i.d. noise.** Thermal (i.i.d.) noise is unpredictable from context and is exactly what JEPA is biased to ignore; but slowly varying or fixed nuisances (baseline, static artifacts, coil/site signatures, lipid/water residuals that persist across time or space) are *maximally* predictable and, per Sobal et al., can dominate the representation. Design of what is masked/predicted (so the nuisance is not the easiest predictable content) and/or conditioning the predictor on nuisance variables is the lever (consistent with Predictive Closure's side-information suggestion).
- **Depth caveat.** The Littwin bias requires a deep encoder; shallow/linear JEPA encoders behave much more like MAE.
- **Scope caveat.** The only theorem is linear, with simultaneously diagonalizable covariances and small init; it describes *learning order under gradient flow*, i.e., an implicit bias in time—not a statement that noisy features are never learned at convergence, nor a statement about nonlinear ViTs.

### Gaps
- Exact closed-form critical-time expressions and the L-dependence in Littwin et al. could not be retrieved (arXiv fetch blocked).
- No formal, nonlinear, information-theoretic theorem of the form "JEPA embeddings retain exactly the predictable information" was found; Predictive Closure (2026) is the closest but was read at abstract level only and is a preprint.
- No study was found that directly measures JEPA retention of *low-amplitude* signal components in spectroscopy-like 1-D data.

---

## 5. Limitations and criticisms: decodability, post-hoc decoders, dense/pixel-level performance, inverse problems

### Takeaway
JEPA has no native path back to signal space: every published visualization uses a decoder trained *afterwards* on frozen features, and those decodings show the embedding keeps shared/semantic content while losing exact low-level detail. On dense and low-level tasks JEPA has historically been at or below reconstruction-based models, prompting loss changes (V-JEPA 2.1). No use of JEPA as a prior/solver for MRI reconstruction, denoising, or other inverse problems was found; diffusion/consistency models dominate that niche.

### Cited Findings

**Not decodable by construction; decoders are post hoc**
- [S] I-JEPA: after pretraining, context-encoder and predictor are frozen and a decoder is trained following the RCDM (representation-conditional diffusion) framework to map average-pooled predictor outputs to pixels; "qualities that are common across samples represent information contained in the average-pooled predictor representation" (i.e., what varies across samples is *not* in the representation). — [arXiv:2301.08243](https://arxiv.org/pdf/2301.08243)
- [S] V-JEPA: a conditional diffusion model is trained to decode feature-space predictions to pixels with encoder and predictor frozen; the decoder sees only the predicted representations for missing regions; the authors state V-JEPA is not a generative model and samples are not expected to exactly match the input. — [arXiv:2404.08471](https://arxiv.org/pdf/2404.08471)
- [S] V-JEPA 2: trains a simple feed-forward frame decoder on Droid to map representations to human-interpretable pixels (for visualization only). — [arXiv:2506.09985](https://arxiv.org/pdf/2506.09985)
- [S] A Nov-2026 stochastic-JEPA preprint also decodes frozen predictor representations with a diffusion decoder and remarks that the deterministic I-JEPA predictor yields no uncertainty information about the underlying image. — [arXiv:2611.09302](https://arxiv.org/pdf/2611.09302)
- [2nd] Commentary: because JEPA doesn't produce pixels, visualizing/inspecting a prediction needs a separate decoder, complicating interpretation. — [Medium explainer](https://medium.com/@izzetcansorna/joint-embedding-predictive-architecture-jepa-predicting-in-latent-space-instead-of-pixel-space-9682c64be866)

**Dense / low-level / pixel-level comparisons**
- [M] EMPIRICAL. I-JEPA's own low-level probes (linear, frozen): Clevr/Count 86.7 and Clevr/Dist 72.4 for I-JEPA ViT-H/14 vs MAE ViT-H/14 at 90.5 and 72.4—i.e., I-JEPA beats view-invariance methods (DINO ViT-B/8: 86.6 / 53.4) on low-level tasks but does *not* beat MAE on counting. — [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)
- [S] I-JEPA trails later variants on ADE20K/COCO (DMT-JEPA's comparison). — [arXiv:2405.17995](https://arxiv.org/pdf/2405.17995)
- [S] V-JEPA 2.1's premise is that dense features "don't emerge reliably" with mask-only latent prediction; needed a dense loss on visible tokens. — [arXiv:2603.14482](https://arxiv.org/abs/2603.14482)
- [S] 3D brain MRI: MAE-based pretraining generally outperformed baseline JEPA for disease detection (2026). — [arXiv:2606.13315](https://arxiv.org/abs/2606.13315)
- [M] Fine-tuning regime: MAE-style models remain very strong under full fine-tuning; JEPA's advantage is mainly in frozen/linear/low-shot evaluation and compute efficiency (I-JEPA paper's framing). — [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)

**Other criticisms**
- [M] Heuristic anti-collapse and many hyperparameters for EMA/VICReg recipes (motivation of LeJEPA; C-JEPA critique). — [arXiv:2511.08544](https://arxiv.org/abs/2511.08544); [arXiv:2410.19560](https://arxiv.org/abs/2410.19560)
- [S] Slow-feature / static-distractor failure (Sobal et al. 2022) and 2026 "informational collapse" papers (§2, §4). — [arXiv:2211.10831](https://arxiv.org/abs/2211.10831)
- [S] Deterministic predictors: no uncertainty representation in mainstream JEPAs. — [arXiv:2611.09302](https://arxiv.org/pdf/2611.09302)
- [S] V-JEPA 2-AC evidence is limited to pick-and-place with image goals on Franka arms in two labs. — [arXiv:2506.09985](https://arxiv.org/abs/2506.09985)
- [2nd] Critical commentary that the "world model" framing is less novel than the 2026 funding implies. — [Medium/AIGuys](https://medium.com/aiguys/lecuns-world-model-is-real-useful-and-nowhere-near-as-new-as-the-funding-round-implies-ecc3b05e2204)

**JEPA for inverse problems / denoising / reconstruction**
- [S] A targeted extended search ("JEPA" + MRI reconstruction / denoising / inverse problem) returned **no** JEPA-based work; all hits were diffusion posterior sampling, regularization-by-denoising with consistency models, unrolled networks, etc. — e.g., [arXiv:2411.12919](https://arxiv.org/html/2411.12919v1); [arXiv:2608.20561](https://arxiv.org/html/2608.20561v1)
- [M] The nearest related items known to the researcher: "D-JEPA: Denoising with a Joint-Embedding Predictive Architecture" (arXiv:2410.03755), which is a *generative* model (JEPA-style masked modeling combined with a diffusion/flow loss per token), not a denoiser for measured signals; and hybrid uses of a JEPA/frozen-SSL encoder as a perceptual or alignment target for generative models. Not verified this session. — [arXiv:2410.03755](https://arxiv.org/abs/2410.03755)

### Inferences (SPECULATION)
- For reconstruction/denoising of measured signals (e.g., low-SNR 31P MRSI), a pure JEPA is structurally mismatched: the objective deliberately throws away signal-space detail and gives no data-consistency mechanism. Plausible roles for JEPA in such a pipeline are indirect: (a) a learned latent/feature-space *regularizer or perceptual loss* for a separate reconstruction network; (b) a pretraining stage for an encoder later attached to a trained decoder/quantification head (the post-hoc-decoder pattern of I-JEPA/V-JEPA/V-JEPA 2); (c) direct latent-space *quantification* (probe metabolite parameters from the embedding) which bypasses reconstruction entirely; (d) a latent dynamics/world model (LeWorldModel-style) over time- or acquisition-indexed data.
- Any post-hoc decoder on frozen JEPA features will hallucinate whatever detail the embedding dropped (the RCDM visualizations show exactly this: sample-to-sample variation = information absent from the embedding). For quantitative spectroscopy that makes a generative decoder on JEPA features risky unless combined with a physics/data-consistency term.
- Given the brain-MRI head-to-head (MAE ≥ baseline JEPA) and V-JEPA 2.1's dense-loss fix, a hybrid (latent prediction + dense/visible-token or reconstruction term, or SIGReg-regularized JEPA with a jointly trained physics-model decoder) is better supported by 2026 evidence than a vanilla I-JEPA for tasks needing fine spectral detail.

### Gaps
- The absence of JEPA-for-inverse-problems work is based on one extended search plus prior knowledge; it is "none found", not "none exists".
- No quantitative study of how much pixel/signal-level information a linear or nonlinear decoder can recover from frozen JEPA features (e.g., PSNR vs MAE features) was retrieved.
- Session limitation: arxiv.org must be added to the network allowlist to verify the [M]-tagged numbers and theorem statements against the PDFs.
