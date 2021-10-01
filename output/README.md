# Output

\*_full_evaluation.txt files contains ranking evaluation scores for full datasets (without feature selection).


### Folder structure:

| Folder                                 | Description                                                                                                                                                          |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FSA_results                            | Contains feature selection and evaluation results of other methods                                                                                                   |
| DATASET_NAME/fs_constrained            | Contains the binary encoded pareto-optimal solutions for each fold and ranking type (for Spearman's Rho & NDCG@10) - Constrained Feature Selection                   |
| DATASET_NAME/fs_constrained_evaluation | Contains fold-wise evaluation results for each ranking type and feature subset percentages (for Spearman's Rho & NDCG@10) - Constrained Feature Selection Evaluation |
| DATASET_NAME/fs_unconstrained          | Contains the binary encoded pareto-optimal solutions for each fold, ranking type and metric combination - Unconstrained Feature Selection                            |
| DATASET_NAME/fs_constrained_evaluation | Contains evaluation results for all folds, for each ranking type and metric combinations - Unconstrained Feature Selection Evaluation                                |
| DATASET_NAME/scores                    | Contains the feature importance and similarity scores for each fold and ranking type                                                                                 |