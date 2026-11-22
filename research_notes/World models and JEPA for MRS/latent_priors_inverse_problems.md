# Latent priors (JEPA / world models) and inverse problems: theoretical bridge for MRSI reconstruction

**Verification note for the report writer.** In this session `arxiv.org` could not be fetched (host not in the allowed network domains); only web search was available. Citations are therefore tagged:

- **[S]** = surfaced and summarised by a search in this session (abstract/snippet level; theorem statements NOT read in full text).
- **[B]** = cited from the researcher's background knowledge of well-known literature; the arXiv ID/venue was not re-verified in this session. These are standard references and very likely correct, but check IDs before publication.

Epistemic labels used in the text: **ESTABLISHED** (published result), **INFERENCE** (my derivation/plausible consequence, not published as such to my knowledge), **SPECULATION**.

Running example: an FID is x[n] = Σ_{k=1..K} a_k z_k^n + noise, with z_k = exp((−d_k + i2πf_k)Δt). MRSI adds a spatial index and a k-space sampling operator A.

---

## 1. How are learned priors used in inverse problems, and what does each require of the prior?

### Takeaway
Every mainstream route needs one of four objects: a **generator/decoder** G(z), a **denoiser/score** D_σ(x) ≈ x + σ²∇log p_σ(x), an **explicit scalar energy** R(x) with usable gradient, or a **projection** onto a set/subspace/manifold. A plain JEPA supplies none of these directly: it supplies an encoder f(x) and a latent predictor g. Any use of JEPA as a reconstruction prior must construct one of the four objects from (f, g).

### Cited Findings

**Requirement table (ESTABLISHED, compiled from the cited sources)**

| Family | Formulation | Object required from the prior | Guarantees available |
|---|---|---|---|
| Plug-and-Play (PnP) | Replace prox of R in ADMM/PGD/HQS by a denoiser | Gaussian denoiser D_σ (no density, no decoder) | Fixed-point convergence if denoiser is non-expansive / Lipschitz-constrained; explicit objective only for special denoisers (gradient-step) |
| RED | ∇R(x) = x − D(x) | Denoiser; needs symmetric Jacobian + local homogeneity for the claimed energy to exist | Often violated by deep denoisers; reinterpretable as score matching |
| Generative-model CS (CSGM) | min_z ‖A G(z) − y‖² | Decoder/generator G with low-dim latent | Recovery with O(k·d·log n) Gaussian measurements for k-dim latent, d-layer ReLU G (S-REC condition) |
| Diffusion posterior sampling | Reverse SDE with ∇log p(y\|x_t) approximation | Score/denoiser at all noise levels | Approximate posterior sampling; DPS approximation is biased; no data-consistency guarantee unless projected |
| Deep image prior / deep decoder | min_θ ‖A G_θ(z0) − y‖² | Untrained architecture only (no data) | Implicit bias toward smooth/structured signals; needs early stopping at low SNR |
| Learned regularizers (AR, ACR, ICNN, ridge regularizers, EBMs) | min_x ‖Ax−y‖² + λR_θ(x) | Scalar energy R_θ with gradient | Convex/weakly-convex R gives uniqueness, stability, convergent regularization |
| Subspace / manifold | x = Φc, or x ∈ range of AE, or ‖x − P(x)‖² penalty | Basis Φ, or autoencoder (E,D), or learned projector P | Linear subspace: classical least-squares theory; nonlinear: local only |
| Unrolled networks | Fixed iterations of prox-grad with learned prox | End-to-end training pairs (or self-supervised loss, Sec. 6) | Data consistency per layer; no general stability guarantee |

- PnP priors (Venkatakrishnan, Bouman, Wohlberg, GlobalSIP 2013) replace the proximal operator with an off-the-shelf denoiser; a 2025 overview describes the approach as one that "integrates powerful image denoisers directly into iterative optimization algorithms" — [S] [From Image Denoisers to Regularizing Imaging Inverse Problems: An Overview, arXiv:2509.03475](https://arxiv.org/abs/2509.03475)
- PnP convergence requires constraints on the denoiser (e.g. Lipschitz condition on D − I): Ryu et al., ICML 2019 — [B] [arXiv:1905.05406](https://arxiv.org/abs/1905.05406). Gradient-step denoisers make PnP minimise an explicit (non-convex) functional: Hurault, Leclaire, Papadakis, ICLR 2022 — [B] [arXiv:2110.03220](https://arxiv.org/abs/2110.03220)
- RED (Romano, Elad, Milanfar, SIAM J. Imaging Sci. 2017) defines R(x)=½xᵀ(x−D(x)) — [B] [arXiv:1611.02862](https://arxiv.org/abs/1611.02862); Reehorst & Schniter (IEEE TCI 2019) show the gradient formula fails for denoisers without symmetric Jacobians and reinterpret RED via score matching — [B] [arXiv:1806.02296](https://arxiv.org/abs/1806.02296)
- Bora, Jalal, Price, Dimakis (ICML 2017): for a d-layer ReLU generator with k-dim input, O(k d log n) Gaussian measurements suffice for recovery to within the representation error of the generator's range — [B] [arXiv:1703.03208](https://arxiv.org/abs/1703.03208)
- Diffusion Posterior Sampling (Chung et al., ICLR 2023) — [B] [arXiv:2209.14687](https://arxiv.org/abs/2209.14687); score-based MRI (Chung & Ye, MedIA 2022) — [B] [arXiv:2110.05243](https://arxiv.org/abs/2110.05243); posterior sampling via Langevin for MRI with robustness to sampling pattern shifts (Jalal et al., NeurIPS 2021) — [B] [arXiv:2108.01368](https://arxiv.org/abs/2108.01368)
- The Daras et al. survey (2024) taxonomises diffusion solvers into families including explicit measurement-matching approximations (DPS), variational inference (RED-Diff) and CSGM-type methods that optimise the initial noise — [S] [A Survey on Diffusion Models for Inverse Problems, arXiv:2410.00083](https://arxiv.org/abs/2410.00083)
- Deep image prior (Ulyanov et al., CVPR 2018) — [B] [arXiv:1711.10925](https://arxiv.org/abs/1711.10925); deep decoder (Heckel & Hand, ICLR 2019) — [B] [arXiv:1810.03982](https://arxiv.org/abs/1810.03982)
- Learned regularisers: adversarial regularizers (Lunz, Öktem, Schönlieb, NeurIPS 2018) — [B] [arXiv:1805.11572](https://arxiv.org/abs/1805.11572); convergence-guaranteed methods and the role of input-convex networks are surveyed by Mukherjee, Hauptmann, Öktem, Pereyra, Schönlieb (IEEE SPM) — [S] [arXiv:2206.05431](https://arxiv.org/abs/2206.05431)
- A 2025 comparative study of learned regularisation functionals (convex ridge, weakly convex ridge, ICNN, etc.) notes that for PnP-type denoisers "the regularizer is only given implicitly", restricting one to proximal algorithms — [S] [Learning Regularization Functionals for Inverse Problems: A Comparative Study, arXiv:2510.01755](https://arxiv.org/abs/2510.01755)
- Most data-driven approaches do not provide a convergent regularisation strategy; PnP with a learned *linear* denoiser plus spectral filtering does — [S] [Found. Comput. Math. 2024, doi:10.1007/s10208-024-09654-x](https://link.springer.com/article/10.1007/s10208-024-09654-x)
- Standard explicit deep energy models are not convex and so lack uniqueness/convergence guarantees; LC-MUSE (Chand & Jacob) builds a multi-scale energy that is strongly convex near the data manifold and claims uniqueness, convergence and robustness — [S] [arXiv:2502.03302](https://arxiv.org/abs/2502.03302)

**MRSI-specific precedent for learned low-dimensional priors (all decoder/projector/denoiser based)**

- SPICE (Lam & Liang, MRM 2014) uses a linear subspace (partially separable) model whose basis is estimated from training data; later work replaces/augments it with learned nonlinear manifolds via deep autoencoders, reported to give "higher accuracy than the linear subspace approximation" at equal dimension — [S] [ISMRM 2019 abstract 0947](https://archive.ismrm.org/2019/0947.html)
- RAIISE: joint learning of a nonlinear representation and a **projector** for fast constrained MRSI reconstruction (encoding model + learned-projector low-dimensional constraint + spatial constraint) — [S] [NSF PAR record](https://par.nsf.gov/servlets/purl/10394735)
- Combining a predetermined linear subspace (SPICE) with a learned nonlinear manifold projector: the manifold constraint "effectively regularizes the subspace fitting, especially at higher orders" — [S] [ISMRM 2023 abstract 0870](https://cds.ismrm.org/protected/23MProceedings/PDFfiles/0870_OaUvAfRLS.html)
- Subspace model + self-supervised spatiotemporal denoiser in PnP-ADMM for highly noisy MRSI, no clean reference needed (IEEE TMI 44(7):3002–3011, 2025) — [S] [Illinois Experts record](https://experts.illinois.edu/en/publications/mr-spatiospectral-reconstruction-integrating-subspace-modeling-an/); precursor abstracts [ISMRM 2023 0955](https://archive.ismrm.org/2023/0955.html), [ISMRM 2024 0810](https://archive.ismrm.org/2024/0810.html)
- Subspace + adaptive generative (GAN-latent) models for high-dimensional MR reconstruction — [S] [arXiv:2306.08630](https://arxiv.org/abs/2306.08630)
- Latent signal models combining Bloch models, linear subspaces and learned latent representations for time-resolved multi-contrast MRI — [S] [PMC12554331](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12554331/)
- A review by Lam and colleagues notes that deep networks can learn a low-dimensional representation of spectra used as a prior in constrained reconstruction rather than as a direct inverse mapping — [S] [PMC8049099](https://pmc.ncbi.nlm.nih.gov/articles/PMC8049099)

### Inferences
- INFERENCE: The existing MRSI learned-prior literature already occupies the "decoder", "projector" and "denoiser" slots. A JEPA would be new only if it (a) produces a better encoder that is then given a decoder/projector, or (b) is used as an energy. Route (a) is architecturally an autoencoder-constrained reconstruction with a different pretraining objective.
- INFERENCE: Because MRS signals have a K-parameter physical generator (K damped exponentials, known chemical-shift structure), the CSGM measurement bound (measurements ∝ intrinsic dimension) is the relevant theoretical lens; a JEPA encoder gives no such bound because it has no range to project onto.

### Gaps
- Could not read full texts (arXiv fetch blocked), so exact conditions of each theorem (e.g. S-REC constants, the Lipschitz constants in PnP convergence) are from background knowledge.
- No head-to-head comparison found of autoencoder-pretrained versus JEPA-pretrained encoders as manifold constraints in any reconstruction problem.

---

## 2. JEPA has no decoder and discards unpredictable information: can non-generative embeddings regularise an inverse problem?

### Takeaway
As of late 2026 I found **no paper that reconstructs MRI/MRSI/CT/seismic/radio data using a JEPA as the prior**. The published bridges are indirect: (i) SSL features as an auxiliary *alignment regulariser on top of a generative (diffusion) prior* (Align & Invert); (ii) frozen SSL encoders given a *trained decoder* (Representation Autoencoders), which turns them into generative latents; (iii) a theoretical claim that JEPA anti-collapse training implicitly learns the data density (JEPA-SCORE), which is the closest thing to "JEPA as an energy E(x)". Feature-space penalties alone cannot be a complete prior because the encoder is, by design, non-injective.

### Cited Findings

**JEPA basics**
- I-JEPA predicts target-block representations from a context block in latent space with an EMA target encoder, with no pixel decoder (Assran et al., CVPR 2023) — [B] [arXiv:2301.08243](https://arxiv.org/abs/2301.08243). LeCun's position paper frames JEPA as an energy-based model whose energy is the latent prediction error between compatible (x, y) pairs — [B] [A Path Towards Autonomous Machine Intelligence, OpenReview](https://openreview.net/forum?id=BZ5a1r-kVsf)
- An open-source library for energy-based JEPAs exists (Feb 2026) — [S] [arXiv:2602.03604](https://arxiv.org/abs/2602.03604)

**JEPA as a density / energy (closest to argmin ‖Ax−y‖² + λE(x))**
- Balestriero, Ballas, Rabbat, LeCun (Oct 2025), "Gaussian Embeddings: How JEPAs Secretly Learn Your Data Density": the anti-collapse term "provably estimates the data density"; the extracted quantity is called JEPA-SCORE; any successfully trained JEPA can give sample probabilities for curation, outlier detection or density estimation; experiments on synthetic, controlled and ImageNet data with I-JEPA, DINOv2 and MetaCLIP; samples can be drawn via Langevin dynamics — [S] [arXiv:2510.05949](https://arxiv.org/abs/2510.05949). (From background knowledge, not re-verified: the score is computed from the encoder Jacobian — a change-of-variables argument when embeddings are Gaussian — so evaluating it needs Jacobian singular values, which is expensive.)
- VJEPA (Jan 2026) gives a variational/probabilistic formulation of JEPA as a world model with uncertainty via sampling, and reports filtering of high-variance distractors — [S] [arXiv:2601.14354](https://arxiv.org/abs/2601.14354)

**SSL features as regularisers for inverse problems**
- Align & Invert (Sfountouris, Daras, Giampouras; arXiv Nov 2025; listed at ICLR 2026 and NeurIPS 2026 — venue status ambiguous): applies REPA-style alignment between a pretrained diffusion/flow model's internal representations and a DINOv2 encoder as an *inference-time regulariser*; reports consistent gains on super-resolution, box inpainting, Gaussian and motion deblurring and fewer steps; theory frames the regulariser as minimising a divergence in DINOv2 embedding space; works with features of an *approximate* target (no ground truth) — [S] [arXiv:2511.16870](https://arxiv.org/abs/2511.16870); [ICLR 2026 listing](https://iclr.cc/virtual/2026/10020380); [NeurIPS 2026 listing](https://neurips.cc/virtual/2026/poster/148851)
- REPA (Yu et al., ICLR 2025): aligning diffusion-transformer hidden states to DINOv2 features accelerates training and improves generation — [B] [arXiv:2410.06940](https://arxiv.org/abs/2410.06940)
- Representation Autoencoders (Zheng et al., Oct 2025, ICLR 2026): a frozen pretrained encoder (default DINOv2-B) plus a trained ViT decoder replaces the VAE in latent diffusion; reported reconstruction FID 0.49 — [S] [arXiv:2510.11690](https://arxiv.org/abs/2510.11690). Follow-ups show fidelity is limited: DecQ reports PSNR rising from 19.13 dB to 22.76 dB with detail-condensing queries on frozen DINOv2 RAEs — [S] [arXiv:2605.22777](https://arxiv.org/abs/2605.22777); dense DINOv2 patch grids are highly redundant — [S] [arXiv:2602.04873](https://arxiv.org/abs/2602.04873); RAE and REPA are reported complementary — [S] [arXiv:2605.18324](https://arxiv.org/abs/2605.18324)
- D-JEPA ("Denoising with a Joint-Embedding Predictive Architecture", Chen et al., ICLR 2025) combines JEPA masked latent prediction with a diffusion/flow loss head for *generation*; it is a generative model, not an inverse-problem solver — [B] [arXiv:2410.03755](https://arxiv.org/abs/2410.03755). My search for D-JEPA in this session returned nothing, so this entry is from memory.
- JEDI: a latent diffusion world model whose latent space is trained with a JEPA framework via the denoising loss; aimed at model-based RL, not reconstruction — [S] [arXiv:2503.08915](https://arxiv.org/abs/2503.08915) (attribution from search snippet; verify)
- Perceptual (feature-space) losses (LPIPS, Zhang et al., CVPR 2018) — [B] [arXiv:1801.03924](https://arxiv.org/abs/1801.03924); perception–distortion trade-off: improving perceptual quality beyond a point necessarily increases distortion (Blau & Michaeli, CVPR 2018) — [B] [arXiv:1711.06077](https://arxiv.org/abs/1711.06077)

**JEPA in medical/scientific imaging is representation learning, not reconstruction**
- 2026 JEPA papers on MRI are all downstream-classification oriented: Neuro-JEPA (~1.55 M scans) — [S] [arXiv:2606.14957](https://huggingface.co/papers/2606.14957.md); COJEPA for 3D structural MRI — [S] [arXiv:2607.11962](https://arxiv.org/abs/2607.11962); RadJEPA — [S] [arXiv:2601.15891](https://arxiv.org/abs/2601.15891); EchoJEPA (latent prediction beats pixel reconstruction and degrades less under perturbation in ultrasound) — [S] [arXiv:2602.02603](https://arxiv.org/abs/2602.02603)
- Counter-evidence: for 3D brain MRI, an MAE with a spectral-domain loss beat a JEPA variant on downstream disease detection — [S] [arXiv:2606.13315](https://arxiv.org/abs/2606.13315)
- A general search "JEPA inverse problems reconstruction denoising MRI" returned no JEPA-based reconstruction method; hits were conventional self-supervised denoising / reconstruction (e.g. Noisier2Inverse) — [S] [arXiv:2503.19468](https://arxiv.org/abs/2503.19468), [arXiv:2411.12919](https://arxiv.org/abs/2411.12919)

### Inferences
- INFERENCE (first principles): A feature-consistency penalty λ‖f(x) − t‖² regularises only the directions the encoder is sensitive to. Because JEPA encoders are trained to be invariant to unpredictable content, the penalty has a large null space (in the Jacobian sense). Combined with A's null space, the reconstruction is determined only if null(A) ∩ null(J_f) = {0} locally. For MRSI, noise-like and weak-peak directions are precisely those the encoder is likely to ignore, so a pure feature penalty will not constrain them: data consistency alone fixes them (good for avoiding hallucination, useless for denoising them).
- INFERENCE: The three workable constructions of an inverse-problem prior from a JEPA are: (1) **energy** E(x) = ‖g(f(x_ctx)) − f(x_tgt)‖² summed over maskings (prediction-error energy: low when parts of the signal are mutually predictable); (2) **density** via JEPA-SCORE (needs Jacobians; unproven for reconstruction); (3) **decoder bolt-on** (RAE-style), after which the standard CSGM / latent-diffusion machinery applies. Only (3) has empirical precedent, and the DecQ numbers (≈19–23 dB PSNR for natural images) indicate that frozen semantic encoders lose fine detail — a warning for low-amplitude spectral peaks.
- INFERENCE: The prediction-error energy (1) is a learned generalisation of classical "self-consistency" priors: in the linear case it reduces to a linear-prediction / annihilating-filter residual (see Sec. 3), i.e. the same structure exploited by LORAKS/ALOHA/SPIRiT-type methods. This is the most principled bridge for MRS.
- SPECULATION: Align & Invert suggests a hybrid — a physics/subspace or diffusion prior supplying the generator, with a JEPA feature-alignment term as an auxiliary regulariser. Whether this helps at MRSI SNR levels is untested.

### Gaps
- No published use of JEPA-SCORE, or a JEPA prediction-error energy, inside a variational reconstruction. No convergence/stability theory exists for such an energy (it is non-convex and unconstrained; cf. Sec. 1 on convex energies).
- Could not verify the exact JEPA-SCORE formula or its assumptions (isotropic Gaussian embeddings; how closely real JEPAs satisfy them) from the full text.
- D-JEPA details are from memory; not confirmed this session.

---

## 3. Linear JEPA / predictive latent models versus subspace methods (SFA, CCA, Koopman, DMD, Prony/HSVD)

### Takeaway
There is a rigorous classical chain — Prony ⇔ linear prediction ⇔ Hankel low rank ⇔ shift-invariant subspace (HSVD/matrix pencil) ⇔ Hankel-DMD ⇔ Koopman eigen-decomposition of a linear system — and a noiseless FID is exactly a finite-dimensional Koopman-invariant signal. Published JEPA theory (deep linear analyses, a Koopman-invariant-subspace argument, BYOL/self-predictive spectral results) supports the statement that *linear latent prediction learns invariant subspaces of the transition operator*, but I found **no paper proving that JEPA on FIDs recovers the Prony/HSVD solution**; the link below is an inference, and it comes with a non-uniqueness caveat that matters for weak peaks.

### Cited Findings

**Classical signal model (ESTABLISHED)**
- A sum of K damped complex exponentials satisfies an order-K linear recurrence; its Hankel matrix has rank K; HSVD (Barkhuijsen, de Beer, van Ormondt, J. Magn. Reson. 1987) and LPSVD (Kumaresan & Tufts, IEEE TASSP 1982) exploit this for MRS quantification — [B] [Barkhuijsen et al. 1987, doi:10.1016/0022-2364(87)90023-0](https://doi.org/10.1016/0022-2364(87)90023-0); [Kumaresan & Tufts 1982, doi:10.1109/TASSP.1982.1163974](https://doi.org/10.1109/TASSP.1982.1163974)
- Hankel-DMD (Arbabi & Mezić, SIADS 2017): DMD applied to Hankel matrices of delay-embedded observables converges to true Koopman eigenvalues/eigenfunctions, and is described as closely related to the Prony approximation of the Koopman mode decomposition — [S] [arXiv:1611.06664](https://arxiv.org/abs/1611.06664)
- Susuki & Mezić defined a Prony-type approximation of the Koopman mode decomposition using Hankel matrices; plain linear delay coordinates cannot exactly span a Koopman-invariant subspace for *nonlinear* systems — [S] [Subspace DMD, arXiv:1705.04908](https://arxiv.org/abs/1705.04908)
- Time-delay observables and HAVOK are tied to Prony; Takens gives a 2n+1 delay-dimension bound — [S] [arXiv:1810.01479](https://arxiv.org/abs/1810.01479). Spectral equivalence between original and delay-reconstructed Koopman operators (measure-preserving case) — [S] [arXiv:1706.01006](https://arxiv.org/abs/1706.01006). Hankel methods link to ERA and singular spectrum analysis — [S] [The Multiverse of DMD Algorithms, arXiv:2312.00137](https://arxiv.org/abs/2312.00137)

**JEPA / self-predictive theory (ESTABLISHED within stated linear settings)**
- Littwin et al. (NeurIPS 2024), deep linear self-distillation: JEPA has an implicit bias toward "influential features", i.e. features with a large **regression coefficient**, whereas MAE is mostly agnostic to this and prioritises high-covariance features; the bias "takes hold only when the encoder is deep and diminishes substantially when the encoder is shallow" — [S] [arXiv:2407.03475](https://arxiv.org/abs/2407.03475); [Apple ML Research page](https://machinelearning.apple.com/research/implicit-bias)
- Sobal et al. (2022), "JEPAs Focus on Slow Features": in a moving-dot testbed JEPA methods match or beat reconstruction when distractor noise changes every step, but **fail when the noise is fixed in time**, with a theoretical explanation: a static (perfectly predictable) distractor is the easiest feature to latch onto — [S] [arXiv:2211.10831](https://arxiv.org/abs/2211.10831)
- "Koopman Invariants as Drivers of Emergent Time-Series Clustering in JEPAs" (arXiv Nov 2025, v2 Jan 2026; AAAI): argues an idealised JEPA loss is minimised when the encoder spans Koopman-invariant regime-indicator functions (eigenvalue-1 eigenfunctions); constraining the linear predictor to near-identity is identified as the key inductive bias; validated on synthetic data — [S] [arXiv:2511.09783](https://arxiv.org/abs/2511.09783); [AAAI proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/39708)
- Self-predictive RL theory (Tang et al., ICML 2023): under idealised two-timescale linear dynamics the columns of the optimal encoder span the same subspace as top-k eigenvectors of the transition matrix (symmetric case; (T^π)² appears in one statement) — [S] [arXiv:2212.03319](https://arxiv.org/abs/2212.03319); action-conditional extension — [S] [arXiv:2406.02035](https://arxiv.org/abs/2406.02035)
- BYOL-type linear theory: the linear predictor aligns with the correlation matrix of representations (Tian, Chen, Ganguli, ICML 2021) — [S] [arXiv:2102.06810](https://arxiv.org/abs/2102.06810); the predictor acts as an approximate orthogonal projection / spectral expansion operator for the latent covariance (Richemond et al.) — [S] [arXiv:2302.04817](https://arxiv.org/abs/2302.04817)
- A 2026 survey on Koopman operator theory covers its connection to ML and world-model architectures — [S] [arXiv:2607.01819](https://arxiv.org/abs/2607.01819)
- Slow feature analysis (Wiskott & Sejnowski, Neural Comput. 2002) extracts unit-variance, decorrelated features minimising temporal derivative — [B] [doi:10.1162/089976602317318938](https://doi.org/10.1162/089976602317318938). My search found no paper explicitly proving JEPA ≡ SFA or ≡ CCA; Sobal et al. is the closest (title-level claim).

### Inferences

**Derivation: linear JEPA on a noiseless FID (INFERENCE; elementary linear algebra, not found published in this form)**
1. Delay-embed: h_n = (x[n],…,x[n+L−1])ᵀ = V Zⁿ a, with V the L×K Vandermonde matrix of the z_k, Z = diag(z_k). All h_n lie in the K-dim subspace range(V), and h_{n+1} = (V Z V⁺) h_n.
2. Linear JEPA: minimise Σ_n ‖P E h_n − E h_{n+1}‖² over encoder E (r×L) and predictor P (r×r). If E is injective on range(V) (r = K), then P = (EV) Z (EV)⁻¹ gives zero loss and the eigenvalues of P are exactly the z_k. This is the HSVD / matrix-pencil shift-invariance equation and the Hankel-DMD operator restricted to the signal subspace. So **the predictor's spectrum = the Prony poles (frequencies and dampings)**, invariant to the arbitrary invertible reparametrisation of the latent.
3. Non-uniqueness: any E whose restriction to range(V) retains only a *subset* S of modes (an invariant subspace of the shift) also gives zero loss with P having eigenvalues {z_k, k∈S}. Total collapse (E = 0) is the extreme case. Hence the prediction loss alone does **not** force all K resonances to be kept; which modes survive is decided by the anti-collapse mechanism (variance/covariance terms, EMA+stop-gradient dynamics, SIGReg) and by the learning dynamics.
4. What is identifiable: poles z_k (from eig(P)) are identifiable; amplitudes/phases a_k are not available from (E, P) without a decoder or a linear probe, because E is defined only up to an invertible latent transform.

**Consequences**
- INFERENCE: In the idealised linear/noiseless case, "latent prediction recovers the linear-prediction/Hankel low-rank structure" is true in the sense of point 2, but only up to the choice of invariant subspace (point 3). A rigorous statement would be: *the global minimisers of the linear JEPA loss with rank-r non-collapse constraint are exactly the encoders whose row space, restricted to the signal subspace, is an r-dim shift-invariant subspace.* This is consistent with the Koopman-invariant-subspace argument of arXiv:2511.09783 and the transition-eigenvector result of arXiv:2212.03319, but those papers treat eigenvalue-1 regime indicators and symmetric Markov transitions respectively, not decaying/oscillating complex modes.
- INFERENCE (noise): with additive noise of variance σ², regressing the mode-k coordinate at n+1 on that at n gives an attenuated coefficient ≈ z_k · SNR_k/(1+SNR_k) (errors-in-variables shrinkage; the same mechanism behind the known low-SNR bias of linear prediction, which is why LPSVD/HSVD use SVD truncation and why total-least-squares variants exist). Under the Littwin et al. bias toward large regression coefficients, modes are therefore prioritised by |z_k|·SNR_k/(1+SNR_k): **strong, narrow (slowly decaying) resonances first; weak or broad resonances last or never**. This is the Sobal et al. "slow features" result expressed in spectroscopy terms.
- INFERENCE: The slow/predictable bias is partly *desirable* (white noise has zero regression coefficient and is rejected; this is what SVD truncation does too) and partly *dangerous*: nuisance signals that are highly predictable — residual water/lipid, baseline, B0-drift patterns, coil-specific structure, fixed spurious signals — are exactly the "static distractor" of Sobal et al. and will be favoured over low-concentration metabolites.
- INFERENCE: SFA picks eigenvalues nearest 1 (slowest); DMD/Prony keeps all eigenvalues of the signal subspace regardless of modulus. For FIDs after demodulation, "slow" = near on-resonance and long T2*. A near-identity predictor constraint (as advocated in arXiv:2511.09783 for regime clustering) would be the wrong inductive bias for MRS; an unconstrained (complex/rotational) linear predictor is needed so that oscillating, decaying modes are representable.
- SPECULATION: A JEPA with nonlinear encoder and *linear* predictor is a Koopman autoencoder without the decoder. Its potential gain over HSVD is amortisation across voxels/subjects (a shared lifted space in which lineshape distortions — B0 inhomogeneity, non-Lorentzian shapes, J-coupling patterns in ¹H, or multiplets in ³¹P — become linear), not better per-FID estimation, where Cramér–Rao bounds already govern model-based fitting.

### Gaps
- No published theorem connecting JEPA to Prony/HSVD/Hankel low rank for damped exponentials; no experiment of JEPA on FIDs or other NMR time series found.
- The Littwin et al. theorem details (assumption of simultaneously-diagonalisable covariances, exact depth dependence) and the proof status of the Koopman–JEPA claim (theorem vs. idealised argument) could not be checked against full text.
- No explicit JEPA–CCA equivalence paper located (the known linear SSL–CCA links are for VICReg/Barlow-type objectives; not searched in depth).

---

## 4. World models (latent dynamics) for physical time series and dynamic imaging

### Takeaway
Latent-dynamics models with a decoder (Koopman autoencoders, latent ODEs, deep state-space / learned Kalman models) are established for physical time series and are beginning to appear in MRI (k-space interpolation, DCE motion suppression, latent signal-evolution models), but always **with a decoder and a reconstruction loss**. Decoder-free JEPA world models for physics appeared in 2026 (PDE-JEPA, PI-JEPA, Phys-JEPA) and target representation/parameter estimation and surrogate forecasting, not measurement inversion.

### Cited Findings
- Deep Koopman autoencoders: encoder lifts, linear layer propagates, decoder maps back (Lusch, Kutz, Brunton, Nat. Commun. 2018) — [B] [arXiv:1712.09707](https://arxiv.org/abs/1712.09707)
- Latent ODEs for irregularly sampled time series (Rubanova, Chen, Duvenaud, NeurIPS 2019) — [B] [arXiv:1907.03907](https://arxiv.org/abs/1907.03907); KalmanNet: learned Kalman gain with partially known dynamics (Revach et al., IEEE TSP 2022) — [B] [arXiv:2107.10043](https://arxiv.org/abs/2107.10043)
- Deep Koopman autoencoder for parallel-imaging k-space interpolation (nonlinear encoder/decoder, linear latent interpolation; compared with GRAPPA and RAKI on two brain scans) — [S] [ISMRM 2025 abstract 5150](https://archive.ismrm.org/2025/5150.html); complex-valued extension — [S] [ISMRM 2026 abstract](https://echo.ismrm.org/abstracts/view/d4929681-be93-4db9-bf55-4025133e7390)
- Koopman-operator spectral decomposition for nonlinear motion suppression in head-and-neck DCE-MRI, with reconstruction, next-state prediction and latent-consistency losses — [S] [arXiv:2607.19401](https://arxiv.org/abs/2607.19401)
- Latent signal models for time-resolved multi-contrast MRI (Bloch models + linear subspaces + latent representations) — [S] [PMC12554331](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12554331/)
- Dynamic MRI with deep manifold self-learning — [S] [arXiv:1911.02492](https://arxiv.org/abs/1911.02492); dynamic-aware spatio-temporal representation (INR-type) for dynamic MRI — [S] [arXiv:2501.09049](https://arxiv.org/abs/2501.09049); classical Kalman filtering is the "straightforward" route to online dynamic MR reconstruction — [S] [RODEO, arXiv:1912.07519](https://arxiv.org/abs/1912.07519)
- JEPA world models for physics (2026): latent-prediction embeddings beat pixel-based ones for physical parameter estimation on a latent-dynamics testbed (ICLR 2026 AI&PDE workshop) — [S] [arXiv:2603.13227](https://arxiv.org/abs/2603.13227); PDE-JEPA (frozen JEPA representation, predictor split into parameter-independent/dependent parts integrated by an ODE solver) — [S] [arXiv:2609.34715](https://arxiv.org/abs/2609.34715); PI-JEPA (masked latent prediction with per-sub-operator PDE residual regularisation) — [S] [arXiv:2604.01349](https://arxiv.org/abs/2604.01349); Phys-JEPA (latent state split into physical + residual parts) — [S] [arXiv:2606.16076](https://arxiv.org/abs/2606.16076)
- V-JEPA 2 (Meta, 2025): video JEPA pretraining plus an action-conditioned latent predictor used for planning — [B] [arXiv:2506.09985](https://arxiv.org/abs/2506.09985)

### Inferences
- INFERENCE: For MRS(I) the "world" is trivially simple compared with video: the dynamics along the FID axis are linear time-invariant with known structure (Bloch/Liouville–von Neumann), so a physics-based state-space model (a Kalman filter with K complex states) is already the exact world model. Learned latent dynamics would add value only for what physics does not capture compactly: lineshape distortions, macromolecular baselines, spatial priors, and inter-voxel/inter-subject regularities.
- INFERENCE: The Phys-JEPA / PI-JEPA pattern (latent = physical part + residual part; physics residual enforced in latent space) is the natural template if one wanted an MRS-specific JEPA: physical latent = (frequencies, dampings, amplitudes) evolving by known linear dynamics; residual latent absorbs nuisance signals.
- INFERENCE: World models applied to *dynamic* MRSI (e.g. ³¹P exercise–recovery, deuterium/hyperpolarised ¹³C kinetics) would act along the slow time axis, where the generative model is a low-order kinetic ODE; latent ODE / KalmanNet-type models with a decoder are better matched to that than decoder-free JEPA, since quantified concentrations (decoded amplitudes) are the output of interest.

### Gaps
- No learned Kalman / neural state-space model trained end-to-end for dynamic MRI reconstruction was found; none for MRSI.
- The 2026 physics-JEPA papers were seen at snippet level; their noise robustness and whether any handles partial/indirect measurements is unknown.

---

## 5. Noise, weak peaks, data consistency, hallucination and identifiability

### Takeaway
Theory and evidence point the same way: JEPA's objective selects features by predictability/regression strength, not by scientific importance, so low-amplitude or broad metabolite resonances are at structural risk of being discarded by the encoder; and any learned prior acting in the null space (or the noise-dominated subspace) of the acquisition can hallucinate. For quantitative spectroscopy the prior must be subordinate to explicit data consistency and a physics model, and validated at the level of small peaks, not global error.

### Cited Findings
- JEPA prioritises features with large regression coefficients (deep linear setting) — [S] [arXiv:2407.03475](https://arxiv.org/abs/2407.03475); JEPA latches onto slow/static distractors and fails when noise is temporally fixed — [S] [arXiv:2211.10831](https://arxiv.org/abs/2211.10831)
- "The Troublesome Kernel" (Gottschling, Antun, Hansen, Adcock; published in SIAM/ASA JUQ 2025 per search; the search summary listed the authors differently, so verify): hallucinations, instabilities and unpredictable generalisation are tied to the forward operator's kernel (and extend to ill-conditioned operators); no-free-lunch theorems; "optimizing the accuracy-stability trade-off is generally difficult"; hallucinations and instabilities "are not rare and may be encouraged by standard training" — [S] [arXiv:2001.01258](https://arxiv.org/abs/2001.01258)
- Hallucinations in tomographic reconstruction formalised via "hallucination maps" (Bhadra et al., IEEE TMI 2021) — [S] [arXiv:2012.00646](https://arxiv.org/abs/2012.00646); regularising instabilities arising from learned denoisers — [S] [arXiv:2108.13551](https://arxiv.org/abs/2108.13551); review of hallucinations in medical devices (noise augmentation improves stability at some performance cost) — [S] [arXiv:2508.14118](https://arxiv.org/abs/2508.14118); a 2026 stability benchmark of generative regularisers — [S] [arXiv:2605.10076](https://arxiv.org/abs/2605.10076)
- Perception–distortion trade-off (Blau & Michaeli) — [B] [arXiv:1711.06077](https://arxiv.org/abs/1711.06077): methods optimised to look plausible (matching a learned distribution) must give up some fidelity.
- Evidence that semantic SSL latents lose fine detail: frozen-DINOv2 RAE reconstructions at ≈19 dB PSNR before decoder improvements — [S] [arXiv:2605.22777](https://arxiv.org/abs/2605.22777)
- In MRSI practice, learned manifold constraints are combined with a physics-based subspace and explicit encoding model precisely to keep the solution anchored — [S] [ISMRM 2023 abstract 0870](https://cds.ismrm.org/protected/23MProceedings/PDFfiles/0870_OaUvAfRLS.html); [TMI 2025 PnP-subspace paper](https://experts.illinois.edu/en/publications/mr-spatiospectral-reconstruction-integrating-subspace-modeling-an/)

### Inferences
- INFERENCE: Decompose the unknown into (i) components determined by data (row space of A, above noise), (ii) components in null(A) or below the noise floor. A prior only matters in (ii). Low-amplitude metabolite peaks at low SNR live in (ii) per voxel; they become estimable only by pooling (spatial smoothness, subspace sharing across voxels). A prior that does not *represent* weak peaks (JEPA encoder discarding low-ρ features, Sec. 3) cannot pool them; a prior that represents them too confidently (generative prior trained on healthy spectra) will *insert* them — the classic hallucination mode (e.g. normal-appearing metabolite ratios imposed on a lesion).
- INFERENCE: The MAE-vs-JEPA contrast of Littwin et al. suggests a concrete design consequence: reconstruction-based objectives weight features by variance (large peaks dominate), JEPA by regression coefficient (predictable peaks dominate). Neither weights by scientific relevance. For ³¹P (few, well-separated, known resonances such as PCr, ATP multiplets, Pi, PME/PDE) a parametric physics model already encodes relevance; the learned component should model residuals/nuisance and spatial structure rather than the spectral basis itself.
- INFERENCE (identifiability): With a decoder-free prior, quantities of interest (amplitudes → concentrations) are not functions of the latent in any guaranteed way (Sec. 3, point 4). A reconstruction pipeline must end in a physics-model fit of data-consistent signals; the JEPA can at most supply initialisation, a regularising energy, or a learned similarity for pooling voxels.
- INFERENCE (safeguards): use hard data consistency or a noise-level-matched discrepancy constraint; restrict the learned prior to the complement of the measured subspace where possible (null-space/range-space decomposition); validate with inserted synthetic low-amplitude peaks and with pathology-like out-of-distribution spectra; report bias and variance against Cramér–Rao bounds rather than PSNR.
- SPECULATION: JEPA-SCORE-style density could serve as an out-of-distribution detector flagging voxels where the prior should be down-weighted, which is a lower-risk use than as the regulariser itself.

### Gaps
- No empirical study of JEPA feature retention as a function of component SNR on any spectroscopic or 1-D physical signal; the "weak peaks are discarded" conclusion is an inference from linear theory plus natural-image evidence.
- No hallucination analysis specific to learned-prior MRSI reconstructions was found in this session.

---

## 6. Self-supervised learning for inverse problems without ground truth (relevant because MRSI has none)

### Takeaway
A mature theory exists for training reconstruction networks from noisy and/or incomplete measurements alone (Noise2Noise family, SURE/GSURE, measurement splitting such as SSDU, equivariant imaging), with explicit identifiability conditions. This "SSL for inverse problems" is a different research strand from SSL representation learning (DINO/JEPA), although both are masked-prediction ideas: SSDU predicts held-out *measurements* in data space under a known forward model, JEPA predicts held-out *embeddings*.

### Cited Findings
- Tachella & Davies monograph (Jan 2026), "Self-Supervised Learning from Noisy and Incomplete Data": chapters on learning from noisy measurements, from incomplete measurements (multiple operators; invariance to transformations; model identification theory), and finite-dataset effects — [S] [arXiv:2601.03244](https://arxiv.org/abs/2601.03244)
- The monograph explicitly separates representation-learning SSL (e.g. DINO, pretext tasks) from inverse-problem SSL, which relies on noisy/incomplete data alone — [S] [arXiv:2601.03244](https://arxiv.org/abs/2601.03244)
- SURE is an unbiased estimator of the supervised MSE loss under known Gaussian noise (extensions: PURE for Poisson), linked to Tweedie's formula — [S] [arXiv:2601.03244](https://arxiv.org/abs/2601.03244)
- Equivariant Imaging (Chen, Tachella, Davies, ICCV 2021) combines measurement consistency with an equivariance loss; identifiability requires the forward operator **not** to be equivariant to the group and the group to be large enough — [S] [arXiv:2601.03244](https://arxiv.org/abs/2601.03244); original paper [B] [arXiv:2103.14756](https://arxiv.org/abs/2103.14756)
- Robust Equivariant Imaging combines SURE and equivariance for fully unsupervised learning from noisy partial measurements — [S] [arXiv:2111.12855](https://arxiv.org/abs/2111.12855); Equivariant Splitting (2025) — [S] [arXiv:2510.00929](https://arxiv.org/abs/2510.00929)
- SSDU (Yaman et al., MRM 2020) splits acquired k-space into a data-consistency set and a loss set — [B] [arXiv:1910.09116](https://arxiv.org/abs/1910.09116); Noise2Noise (Lehtinen et al., ICML 2018) — [B] [arXiv:1803.04189](https://arxiv.org/abs/1803.04189); Noisier2Noise — [B] [arXiv:1910.11908](https://arxiv.org/abs/1910.11908); ENSURE (Aggarwal, Pramanik, Jacob; GSURE-based, for varying sampling operators) — [B] [arXiv:2010.10631](https://arxiv.org/abs/2010.10631)
- Noisier2Inverse: self-supervised joint reconstruction/denoising with correlated noise (MRI, CT) — [S] [arXiv:2503.19468](https://arxiv.org/abs/2503.19468); self-supervised denoising as preprocessing improves DL multi-coil MRI reconstruction — [S] [arXiv:2411.12919](https://arxiv.org/abs/2411.12919)
- Scan-specific self-supervised Bayesian deep nonlinear inversion (DNLINV) learns from a single undersampled scan — [S] [arXiv:2203.00361](https://arxiv.org/abs/2203.00361)
- MRSI already uses this strand: a self-supervised spatiotemporal denoiser trained without noisy/clean pairs, plugged into subspace-constrained PnP-ADMM (IEEE TMI 2025) — [S] [Illinois Experts record](https://experts.illinois.edu/en/publications/mr-spatiospectral-reconstruction-integrating-subspace-modeling-an/); [ISMRM 2024 0810](https://archive.ismrm.org/2024/0810.html)
- Align & Invert shows representation-alignment regularisation can use features of an approximate (non-ground-truth) target — [S] [arXiv:2511.16870](https://arxiv.org/abs/2511.16870)

### Inferences
- INFERENCE: A JEPA can be trained on raw noisy MRSI data without ground truth (it is self-supervised by construction), but nothing in its loss makes the embedding of noisy data equal the embedding of the clean signal. With independent noise in context and target, the latent regression target is conditionally unbiased in a Noise2Noise-like sense *for the predictor output*, yet the encoder input is still noisy (errors-in-variables), so the learned features are noise-attenuated (Sec. 3). No published identifiability theorem covers this; the Tachella–Davies conditions concern data-space estimators with known A.
- INFERENCE: MRSI offers natural "multiple views with independent noise" for either strand: separate averages/transients, separate coils, odd/even FID samples, repeated TRs. These supply Noise2Noise pairs for denoiser training and JEPA context/target pairs alike. Independent-noise view construction also avoids the Sobal "static distractor" failure for the noise itself (but not for static nuisance signals).
- INFERENCE: The lowest-risk synthesis consistent with the literature: (1) physics subspace/parametric model + explicit forward model for data consistency; (2) a denoiser or projector trained by SURE/Noise2Noise/SSDU-type losses as the main learned prior (has theory, has MRSI precedent); (3) JEPA-type pretraining, if used at all, as an encoder initialisation or auxiliary alignment regulariser, with weak-peak retention tested explicitly.

### Gaps
- How the monograph treats SSDU-type splitting in detail was not retrieved. No identifiability results found for self-supervised learning with *subspace-structured spatiospectral* operators as in SPICE-type MRSI.
- No work found that trains a JEPA with a measurement-domain (k-space/FID) masking strategy tied to a known forward operator, which would be the formal meeting point of the two SSL strands.
