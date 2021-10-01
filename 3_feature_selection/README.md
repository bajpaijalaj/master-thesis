# Feature Selection

The main scripts to be executed here are:

| Scripts                                  | Description                                                                                                                      |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| run_constrained_feature_selection.py     | For running the constrained feature selection                                                                                    |
| run_unconstrained_feature_selection_1.py | For running the unconstrained feature selection for combinations of Kendall's Tau with {NDCG@10, DCG@10, ERR@10, Pfound@10}      |
| run_unconstrained_feature_selection_2.py | For running the unconstrained feature selection for combinations of Spearman's Rho with {NDCG@10, DCG@10, ERR@10, Pfound@10}     |
| run_unconstrained_feature_selection_3.py | For running the unconstrained feature selection for combinations of {Kendall's Tau, Spearman's Rho} with {MAP@10, MRR@10, F1@10} |



>The two scripts used in the above files are *NSGA2_constrained.py* and *NSGA2_unconstrained.py* which contains the use of NSGA 2 algorithm.