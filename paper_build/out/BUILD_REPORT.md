# Paper build report

- Built: 2026-10-18T17:27:07+00:00
- Git commit: `59d6c78b2bcb66f6a3981303695680b40b95cae1`
- Uncommitted changes in paper_build/experiments/scripts: False
- Lint: passed (no RNG outside the seeded bootstrap, no typed-in results)
- Cross-checks: 48/48 passed
- Inputs: 45 files, 0 not tracked by git
- Outputs: 20 tables, 32 figures, 475 numbers

## Cross-checks (paper_build recomputation vs value saved by the experiment script)

| Check | paper_build | experiment file | OK |
|---|---|---|---|
| MAE ridge_swin_gap vs E2 | 0.0075 | 0.0075 | yes |
| MSE skill ridge_swin_gap vs E2 | 0.67339 | 0.6734 | yes |
| MAE ridge_image_stats vs E2 | 0.01216 | 0.01216 | yes |
| MSE skill ridge_image_stats vs E2 | 0.11854 | 0.1185 | yes |
| MAE ridge_vit_meantokens vs E2 | 0.01085 | 0.01085 | yes |
| MSE skill ridge_vit_meantokens vs E2 | 0.31983 | 0.3198 | yes |
| MAE vit_frozen_offset vs E2 | 0.01274 | 0.01274 | yes |
| MSE skill vit_frozen_offset vs E2 | 0.04879 | 0.0488 | yes |
| MAE constant_loo_mean vs E2 | 0.01281 | 0.01281 | yes |
| MSE skill constant_loo_mean vs E2 | 0.0 | 0.0 | yes |
| MAE headline vs E2 reference | 0.01338 | 0.01338 | yes |
| MAE ridge_cnn_globalpool256 vs E2c | 0.00901 | 0.00901 | yes |
| MAE cnn_frozen_affine vs E2c | 0.01001 | 0.01001 | yes |
| MAE age-only vs E2b | 0.00808 | 0.00808 | yes |
| MAE Swin+age vs E2b | 0.00654 | 0.00654 | yes |
| ICC frozen vs E1 | 0.50828 | 0.5083 | yes |
| ICC headline vs E1 | 0.61462 | 0.6146 | yes |
| ICC cross-seed mean vs E1 | 0.28897 | 0.289 | yes |
| conformal coverage 90 vs E7 | 0.91304 | 0.913 | yes |
| conformal width 90 vs E7 | 0.05706 | 0.0571 | yes |
| external conformal width 90 vs E7 | 0.0576 | 0.0576 | yes |
| conformal coverage 95 vs E7 | 0.95652 | 0.957 | yes |
| conformal width 95 vs E7 | 0.0664 | 0.0664 | yes |
| external conformal width 95 vs E7 | 0.0668 | 0.0668 | yes |
| external adapter MAE vs E3 | 0.07597 | 0.076 | yes |
| external Swin MAE vs E3 | 0.07362 | 0.0736 | yes |
| external old adapter = submitted 0.0731 | 0.07311 | 0.0731 | yes |
| MSE skill headline vs r2_2 | -0.06089 | -0.0609 | yes |
| calibration slope headline vs r2_2 | -0.44102 | -0.441 | yes |
| headline MAE vs submitted LOOCV file | 0.01338 | 0.0134 | yes |
| internal 95% conformal coverage identical for every readout (fixed by construction) | [0.9565] | — | yes |
| ICC frozen recomputed twice | 0.50828 | 0.50828 | yes |
| permutation replay p (ICC) vs permutation_test | 0.0024 | 0.0024 | yes |
| permutation replay p (|r|) vs permutation_test | 0.59007 | 0.59007 | yes |
| OASIS ViT3D MAE vs oasis_cnn_comparison | 0.05839 | 0.0584 | yes |
| OASIS ViT3D r (rows) vs finetune_results | 0.722 | 0.722 | yes |
| CDR-stratified MAE CDR_None vs simulated_dementia | 0.01556 | 0.0156 | yes |
| CDR-stratified MAE CDR_0.0 vs simulated_dementia | 0.08653 | 0.0865 | yes |
| CDR-stratified MAE CDR_0.5 vs simulated_dementia | 0.09871 | 0.0987 | yes |
| CDR-stratified MAE CDR_1.0 vs simulated_dementia | 0.15665 | 0.1567 | yes |
| adapter head_only MAE vs ablation_adapter | 0.01333 | 0.0133 | yes |
| adapter ln_head MAE vs ablation_adapter | 0.01365 | 0.0137 | yes |
| adapter full_ft MAE vs ablation_adapter | 0.01229 | 0.0123 | yes |
| adapter LoRA MAE vs ablation_lora | 0.01278 | 0.0128 | yes |
| unadapted ViT3D MAE vs ablation_vit_vs_cnn | 0.0403 | 0.0403 | yes |
| unadapted CNN3D MAE vs ablation_vit_vs_cnn | 0.07628 | 0.0763 | yes |
| MC Dropout 95% coverage vs mc_dropout_ci | 4.34783 | 4.3 | yes |
| unpaired physics-vs-blur Mann-Whitney p (manuscript 0.81) | 0.80989 | 0.81 | yes |

## Input files

- `experiments/ablation_adapter/results.json`
- `experiments/ablation_gaussblur/results.json`
- `experiments/ablation_lora/results.json`
- `experiments/ablation_vit_vs_cnn/results.json`
- `experiments/arch_comparator_swin/results.json`
- `experiments/arch_comparator_unetr/results.json`
- `experiments/dino_headline_loocv/results.json`
- `experiments/fastsurfer_output/nwbv_ground_truth.csv`
- `experiments/fastsurfer_zenodo/nwbv_ground_truth_zenodo.csv`
- `experiments/inference_latency/results.json`
- `experiments/loocv_cross_session/results.json`
- `experiments/model_param_counts/results.json`
- `experiments/multiseed_loocv/results.json`
- `experiments/oasis_bootstrap/results.json`
- `experiments/oasis_cnn_comparison/results.json`
- `experiments/oasis_finetune/finetune_results.json`
- `experiments/oasis_mc_dropout/results.json`
- `experiments/paper_statistics/failure_analysis.json`
- `experiments/permutation_test/results.json`
- `experiments/pseudolabel_ablation/results.json`
- `experiments/r3_e1_reproducibility/results.json`
- `experiments/r3_e2_nonneural_baselines/results.json`
- `experiments/r3_e2b_swin_verification/results.json`
- `experiments/r3_e2c_cnn3d_target/results.json`
- `experiments/r3_e3_external/results.json`
- `experiments/r3_e4_physics_vs_blur/blur_seed1.json`
- `experiments/r3_e4_physics_vs_blur/blur_seed42.json`
- `experiments/r3_e4_physics_vs_blur/blur_seed7.json`
- `experiments/r3_e4_physics_vs_blur/physics_seed1.json`
- `experiments/r3_e4_physics_vs_blur/physics_seed42.json`
- `experiments/r3_e4_physics_vs_blur/physics_seed7.json`
- `experiments/r3_e4_physics_vs_blur/summary.json`
- `experiments/r3_e6_time_budget/results.json`
- `experiments/r3_e7_conformal/results.json`
- `experiments/r3_e8_oasis_fixed_inputs/results.json`
- `experiments/r3_existing_data_analyses/results.json`
- `experiments/r3_frozen_predictions/predictions.json`
- `experiments/real64mt_eval/mc_dropout_ci.json`
- `experiments/sim_validation/results.json`
- `experiments/simulated_dementia/results.json`
- `experiments/simulation_sensitivity/results.json`
- `experiments/ssl_comparator_contrastive/results.json`
- `experiments/ssl_comparator_dino/results.json`
- `experiments/ssl_comparator_mae/results.json`
- `experiments/ssl_comparator_simmim/results.json`

## Tables

- `tables/readouts_real64mt.md` — Real 64 mT, cross-session LOOCV (fit on HFC of 22, predict HFE of held-out subject), n = 23
- `tables/multiseed.md` — ViT3D LN+head adapter, five seeds, cross-session LOOCV (n = 23)
- `tables/reproducibility.md` — Session reproducibility of predictions (n = 23)
- `tables/nuisance.md` — Stable nuisance signals in the 64 mT scans (n = 23)
- `tables/swin_partial.md` — Swin-UNETR ridge readout: correlation with true nWBV after removing nuisance variables (n = 23)
- `tables/oasis.md` — OASIS-1 test set (n = 38), physics-simulated 64 mT inputs
- `tables/physics_vs_blur.md` — Matched physics-simulation vs Gaussian-blur pretraining (identical pipeline, splits, budgets and seeds)
- `tables/conformal.md` — Conformal prediction intervals by cohort
- `tables/external_per_subject.md` — External cohort (van den Broek et al.), per subject
- `tables/external_summary.md` — External cohort (n = 10): every readout fails
- `tables/pseudolabel.md` — SynthSeg+ pseudo-label test: internal same-session feasibility test (HFC → HFC, fixed 15/8 split)
- `tables/label_noise.md` — Propagation of reference-label noise (Monte Carlo, ViT3D headline)
- `tables/run_reconciliation.md` — Three runs of the identical LN+head recipe (not pooled)
- `tables/time_budget.md` — Time budget per scan and per site (Intel i7-8750H CPU, 6 threads)
- `tables/labels.md` — Reference labels by cohort
- `tables/kept_table4_cdr.md` — Table IV. CDR-stratified MAE on physics-simulated 64 mT inputs (OASIS-1 test set, n = 38; no adaptation)
- `tables/kept_table8_worst5.md` — Table VIII. Five largest LOOCV errors (real 64 mT, n = 23)
- `tables/kept_table6_pretraining.md` — Table VI. Pretraining comparators on OASIS-1 (n = 38)
- `tables/kept_table9_stats.md` — Table IX. Summary of statistical comparisons
- `tables/kept_table2_summary.md` — Table II. Summary of results by evaluation context

## Figures

- `figures/fig_pipeline.png` / `.svg`
- `figures/fig_simulation_sensitivity.png` / `.svg`
- `figures/fig_oasis_scatter.png` / `.svg`
- `figures/fig_physics_vs_blur.png` / `.svg`
- `figures/fig_readouts_real64mt.png` / `.svg`
- `figures/fig_readout_scatter.png` / `.svg`
- `figures/fig_reproducibility.png` / `.svg`
- `figures/fig_multiseed.png` / `.svg`
- `figures/fig_conformal.png` / `.svg`
- `figures/fig_external.png` / `.svg`
- `figures/fig24_external_validation.png` / `.svg`
- `figures/manuscript/Figure_02_simulation_parameter_sensitivity.png` / `.svg`
- `figures/manuscript/Figure_03_three_stage_training_pipeline.png` / `.svg`
- `figures/manuscript/Figure_04_predicted_vs_true_oasis.png` / `.svg`
- `figures/manuscript/Figure_05_true_nwbv_by_cdr_stage.png` / `.svg`
- `figures/manuscript/Figure_06_physics_vs_gaussian_blur.png` / `.svg`
- `figures/manuscript/Figure_07_method_comparison_real64mt.png` / `.svg`
- `figures/manuscript/Figure_08_fourway_comparison_with_latency.png` / `.svg`
- `figures/manuscript/Figure_09_age_vs_nwbv.png` / `.svg`
- `figures/manuscript/Figure_10_cross_session_loocv_scatter.png` / `.svg`
- `figures/manuscript/Figure_11_bootstrap_mae_bland_altman.png` / `.svg`
- `figures/manuscript/Figure_12_permutation_tests.png` / `.svg`
- `figures/manuscript/Figure_13_multiseed_loocv_stability.png` / `.svg`
- `figures/manuscript/Figure_14_intersession_reliability_icc.png` / `.svg`
- `figures/manuscript/Figure_15_cdr_stratified_mae_simulated.png` / `.svg`
- `figures/manuscript/Figure_16_pseudolabel_ablation.png` / `.svg`
- `figures/manuscript/Figure_17_architecture_pretraining_comparators.png` / `.svg`
- `figures/manuscript/Figure_18_adapter_strategy_ablation.png` / `.svg`
- `figures/manuscript/Figure_20_mcdropout_intervals_real64mt.png` / `.svg`
- `figures/manuscript/Figure_21_conformal_calibration.png` / `.svg`
- `figures/manuscript/Figure_22_mcdropout_calibration_curve.png` / `.svg`
- `figures/manuscript/Figure_23_failure_analysis.png` / `.svg`
