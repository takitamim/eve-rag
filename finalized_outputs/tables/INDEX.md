# Finalized tables — INDEX

This folder is the **single source of truth** for every table that goes into the paper, slides, or appendix.

| # | File | Purpose | Used in |
|---|---|---|---|
| 01 | `01_headline_cross_dataset.csv` | The hero table: GSAF + ablations + streams on all 3 test sets, AUROC + F1 | Main results §5 |
| 02 | `02_ablation_delta_auroc.csv`   | Δ-AUROC vs full GSAF for each ablation, per dataset                       | Ablation §5.x |
| 03 | `03_dataset_summary.csv`        | NewsCLIPpings / MMFakeBench / VERITE row counts, splits, label balance     | Datasets §3 |
| 04 | `04_per_stream_results.csv`     | Each stream's standalone metrics on every test split                       | Method §4 |
| 05 | `05_per_class_newsclippings.csv`| Per-class precision/recall/F1 of GSAF on NewsCLIPpings test                | Main results §5 |
| 06 | `06_ablations_full.csv`         | Full ablation grid (model × dataset × all metrics)                         | Appendix |
| 07 | `07_ablation_legacy.csv`        | Original ablation table from `fusion_training.ipynb` (sanity)              | Appendix |
| 08 | `08_fusion_results_full.csv`    | All splits, both default + tuned thresholds                                | Appendix |
| 09 | `09_newsclippings_eval_all_models.csv` | NewsCLIPpings eval of every variant + streams                       | Appendix |
| 10 | `10_cross_dataset_mmfakebench.csv` | MMFakeBench-only OOD table                                              | Appendix |
| 11 | `11_cross_dataset_verite.csv`     | VERITE-only OOD table                                                  | Appendix |
| 12 | `12_qualitative_examples.csv`     | The 24 illustrative pairs picked for the qualitative figures           | Appendix |
| 13 | `13_qualitative_gate_attribution.csv` | Average gate weights per (dataset, bucket)                          | Discussion §6 |
| 14 | `14_baseline_results.csv`         | Naïve text-only and image-only baselines                                | Main results §5 |
| 15 | `15_clip_baseline_results.csv`    | Vanilla CLIP-similarity baseline                                        | Main results §5 |

LaTeX (booktabs) sources for the headline tables are in `../latex/`.
