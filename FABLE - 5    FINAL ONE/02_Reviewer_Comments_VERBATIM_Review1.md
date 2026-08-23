# Reviewers' Comments to Author — VERBATIM (Review Round 1)

**Manuscript ID:** Access-2026-28453
**Title:** A Reproducible Feasibility Baseline for Segmentation-Free nWBV Regression from 64mT Ultra-Low-Field MRI Using Physics-Constrained Deep Learning
**Journal:** IEEE Access
**Decision date:** 19-Jul-2026
**Decision:** Reject — resubmission permitted (ONE opportunity only)
**Associate Editor:** Prof. Sandra Costanzo

> This file is a verbatim transcription of the four reviewer reports as received.
> No summarization, grouping, or interpretation has been added.

---

## Reviewer: 1

Comments:

1. The main contribution should be reconsidered because the proposed model does not outperform the constant-mean baseline (MAE: 0.0134 vs. 0.0126). Please avoid overstating the model's effectiveness.
2. The study uses only 23 real 64mT MRI subjects, which is insufficient for strong conclusions. Please clearly discuss the impact of the small sample size.
3. External validation on an independent dataset is required to demonstrate model generalization.
4. The physics-based simulation should be quantitatively validated against real 64mT MRI data using image similarity or distribution analysis metrics.
5. More ablation studies are needed to evaluate the contribution of each component:
   physics simulation,
   self-supervised pretraining,
   ViT architecture,
   LayerNorm + head adaptation.
6. The dementia evaluation is limited because it relies on simulated data and very few pathological subjects. Please avoid strong clinical interpretations.
7. The uncertainty estimation results show severe miscalibration (4.3% coverage for 95% intervals). Please improve this analysis or clearly state its limitations.
8. Statistical comparisons between models and experimental settings should be strengthened with appropriate significance tests.
9. The robustness of the model should be evaluated using multiple random seeds or repeated experiments.
10. The simulation parameters (noise, B0 distortion, relaxation values) require clearer justification and sensitivity analysis.
11. The manuscript contains repeated discussions of MAE, ICC, and limitations. Please reduce redundancy and improve readability.

Additional Questions:

Please confirm that you have reviewed all relevant files, including supplementary files and any author response files, which can be found in the "View Author's Response" link above (author responses will only appear for resubmissions): **Yes, all files have been reviewed**

1) Does the paper contribute to the body of knowledge?: **Partially**

2) Is the paper technically sound?: **Partially**

3) Is the subject matter presented in a comprehensive manner?: **Yes**

4) Are the references provided applicable and sufficient?: **yes**

5) Are there references that are not appropriate for the topic being discussed?: **No**

5a) If yes, then please indicate which references should be removed.: *(blank)*

---

## Reviewer: 2

Comments:

Issues:

- One of the most concerning findings is that the proposed ViT performs worse than the CNN baseline on the OASIS dataset. CNN3D achieves Pearson correlation of 0.877 and MAE of 0.024, whereas ViT3D achieves only 0.722 correlation and MAE of 0.058. The authors attribute the improvement on the 64 mT dataset to the narrow value range rather than to the transformer architecture itself. Therefore, the experimental results do not convincingly demonstrate that Vision Transformers provide better feature learning than CNNs. Instead, the reported gains appear to result primarily from the adaptation procedure rather than from the transformer architecture.

- The proposed ViT3D architecture is a very basic implementation consisting of only four transformer encoder layers with 4.23M parameters, yet the paper does not explain why this architecture was selected over more advanced medical Vision Transformer models such as Swin UNETR, UNETR, ViT-V-Net, or hierarchical transformers.

- The Stage-1 pretraining uses a denoising autoencoder to reconstruct high-field MRI from simulated low-field MRI. However, the paper provides no evidence that this reconstruction objective actually learns representations useful for nWBV regression. Modern Vision Transformer pretraining methods, such as Masked Autoencoders (MAE), DINO, SimMIM, or contrastive learning, have shown superior feature learning for downstream medical imaging tasks. The paper never compares its denoising objective against these stronger self-supervised approaches, making the claimed advantage of the proposed pretraining strategy unconvincing.

- The proposed adaptation updates only the final LayerNorm and regression head (769 parameters), but this design choice is based on intuition rather than experimental evidence. The paper states that comparisons with head-only adaptation, full fine-tuning, LoRA, adapters, prompt tuning, and other parameter-efficient transformer adaptation methods are left for future work. Since Vision Transformers are sensitive to adaptation strategy under domain shift, it is difficult to conclude that the proposed LayerNorm+head adaptation is the most effective solution.

Additional Questions:

Please confirm that you have reviewed all relevant files, including supplementary files and any author response files, which can be found in the "View Author's Response" link above (author responses will only appear for resubmissions): **Yes, all files have been reviewed**

1) Does the paper contribute to the body of knowledge?: **No**

2) Is the paper technically sound?: **No**

3) Is the subject matter presented in a comprehensive manner?: **No**

4) Are the references provided applicable and sufficient?: **NA**

5) Are there references that are not appropriate for the topic being discussed?: **No**

5a) If yes, then please indicate which references should be removed.: *(blank)*

---

## Reviewer: 3

Comments:

1. The manuscript addresses an important and timely problem in point-of-care neuroimaging. Direct nWBV regression from 64mT ultra-low-field MRI without segmentation or super-resolution is a practically relevant research direction, especially for low-resource or bedside environments where conventional high-field MRI morphometry pipelines are difficult to deploy.

2. The feasibility-oriented framing is appropriate and should be maintained consistently. The study is strongest when presented as a reproducible feasibility baseline rather than as a clinically deployable nWBV estimation system. The title, abstract, discussion, and conclusion should consistently preserve this distinction and avoid implying clinical readiness.

3. The leakage-free subject-independent cross-session LOOCV protocol is a methodological strength. Training the lightweight adapter on one session from non-held-out subjects and testing on the second session of the held-out subject is an appropriate design for reducing subject/session leakage in a very small real-hardware cohort. This protocol is one of the strongest aspects of the manuscript.

4. The main technical concern is that the primary real-hardware MAE is comparable to a constant-mean LOOCV baseline. Although the adapted ViT3D achieves MAE below the predefined reference threshold, the result does not clearly demonstrate superiority over a trivial mean predictor. The authors should better explain what evidence supports anatomy-dependent learning rather than range-compressed mean regression within a narrow healthy cohort.

5. Additional evidence of input-dependent learning would strengthen the manuscript. The reported inter-session ICC suggests some model-dependent behavior, but the confidence interval is wide due to the small sample size. Analyses such as adapter ablation, head-only versus LayerNorm+head comparison, permutation testing, or validation on a wider real-hardware nWBV range would make the technical claim more convincing.

6. The clinical generalizability should be stated more conservatively. The real 64mT cohort includes only healthy adults, while nWBV is clinically motivated as a neurodegeneration biomarker. The simulated dementia/atrophy analysis shows large errors and regression toward the healthy-adult mean; therefore, the current model should not be interpreted as suitable for dementia screening, pathological morphometry, or longitudinal clinical monitoring.

7. The physics simulation, uncertainty estimation, and deployment claims should be carefully bounded. The physics-constrained simulation pipeline is useful as a reproducible pre-training strategy, but its advantage over Gaussian-blur degradation is small and statistically non-significant. Similarly, the MC Dropout intervals are severely miscalibrated under domain shift and should not be used for individual-level clinical decisions. The reported 47 ms inference time supports computational feasibility, but edge-device deployment and prospective clinical readiness have not yet been demonstrated.

Additional Questions:

Please confirm that you have reviewed all relevant files, including supplementary files and any author response files, which can be found in the "View Author's Response" link above (author responses will only appear for resubmissions): **Yes, all files have been reviewed**

1) Does the paper contribute to the body of knowledge?: **Yes.** The paper contributes a useful reproducible feasibility baseline for direct nWBV regression from 64mT ultra-low-field MRI without segmentation or super-resolution. Its main contribution lies in the physics-constrained simulation recipe, leakage-free cross-session LOOCV protocol, and transparent failure characterization.

2) Is the paper technically sound?: **Partially yes.** The methodology is generally coherent, and the subject-independent cross-session LOOCV design is appropriate. However, the primary real-hardware MAE is comparable to a constant-mean baseline, the real 64mT cohort is very small, and the model does not clearly capture age-related biological variation. Therefore, the technical claims should be bounded carefully.

3) Is the subject matter presented in a comprehensive manner?: **Yes, in general.** The manuscript presents the motivation, simulation pipeline, ViT3D architecture, domain adaptation protocol, baseline comparisons, reliability analysis, uncertainty evaluation, and failure analysis in a comprehensive manner. The paper is strongest when framed as a feasibility study rather than a clinically deployable system.

4) Are the references provided applicable and sufficient?: **Yes.** The references are generally applicable and sufficient, covering brain morphometry, ultra-low-field MRI, segmentation-based pipelines, Vision Transformers, MRI simulation, uncertainty estimation, and related neuroimaging methods. I do not suggest additional references at this stage.

5) Are there references that are not appropriate for the topic being discussed?: **No**

5a) If yes, then please indicate which references should be removed.: *(blank)*

---

## Reviewer: 4

Comments:

1. The manuscript is written in sufficiently clear and understandable English, making the presented work generally easy to follow.
2. The proposed methodology is described with an adequate level of clarity.
3. However, the blurred figures included in the manuscript limit the effective presentation and interpretation of the experimental results.
4. Furthermore, the performance of the proposed method should be compared with existing state-of-the-art approaches through qualitative image-based analysis.
5. To provide a more comprehensive evaluation of the model's performance, the comparative study should include a broader range of medical image modalities.
6. In conclusion, the proposed method is not recommended for acceptance in its current form.
   A more comprehensive evaluation is required, including both quantitative and qualitative comparisons with existing state-of-the-art methods across a broader range of image types, with particular emphasis on medical imaging applications. In addition, image quality comparisons with established approaches should be provided to more convincingly demonstrate the effectiveness, robustness, and generalizability of the proposed method.

Additional Questions:

Please confirm that you have reviewed all relevant files, including supplementary files and any author response files, which can be found in the "View Author's Response" link above (author responses will only appear for resubmissions): **Yes, all files have been reviewed**

1) Does the paper contribute to the body of knowledge?: **Yes**

2) Is the paper technically sound?: **Yes**

3) Is the subject matter presented in a comprehensive manner?: **No**

4) Are the references provided applicable and sufficient?: **Yes**

5) Are there references that are not appropriate for the topic being discussed?: **No**

5a) If yes, then please indicate which references should be removed.: *(blank)*

---

*Article administrator: Mr. Shri Krishna Mishra — k.mishra@ieee.org*
