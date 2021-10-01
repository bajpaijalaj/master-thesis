# Evaluation

The main scripts to be executed here are (no specific order required & to be run on need basis):

| Scripts                                                            | Description                                                                                                                                     |
| ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| run_constrained_evaluation_with_percent.py                         | For running the constrained feature selection **evaluation** with the featurre subset percentage as an argument                                 |
| run_unconstrained_feature_selection_1_evaluation.py                | For running the unconstrained feature selection **evaluation** for combinations of Kendall's Tau with {NDCG@10, DCG@10, ERR@10, Pfound@10}      |
| run_unconstrained_feature_selection_2_evaluation.py                | For running the unconstrained feature selection **evaluation** for combinations of Spearman's Rho with {NDCG@10, DCG@10, ERR@10, Pfound@10}     |
| run_unconstrained_feature_selection_3_evaluation.py                | For running the unconstrained feature selection **evaluation** for combinations of {Kendall's Tau, Spearman's Rho} with {MAP@10, MRR@10, F1@10} |
| run_unconstrained_feature_selection_evaluation_with_metric_args.py | For running the unconstrained feature selection **evaluation** with an option to provide the metric combination as arguments                    |
| evaluate_full_dataset.py                                           | For full dataset evaluation (No Feature Selection case)                                                                                         |
| evaluate_competitors.py                                            | For generating the evaluation results for other methods ([FS results obtained using the repository](https://github.com/andrgig/FSA))                                                                                                                                            |

> The following scripts are used in the above files: *aa_feature_subset_extractor.py*, *bb_grouping_and_evaluation.py*, *bb_grouping_and_evaluation_for_full_eval.py*, *cc_run_evaluation_fs.py*.



