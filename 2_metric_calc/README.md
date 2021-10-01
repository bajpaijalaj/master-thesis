# Preprocessing

The main script to be executed here is **scores_calculation.py**. The scripts used within **scores_calculation.py** are named in an alphabetical order based on their order of execution for easy viewing:

| Scripts                                  | Description                                                                 |
| ---------------------------------------- | --------------------------------------------------------------------------- |
| aa_individual_feature_sets_extraction.py | For individual feature set extraction (required for XGBoost)                |
| bb_individual_feature_sets_grouping.py   | For individual feature set grouping (required for XGBoost)                  |
| cc_individual_feature_sets_ranking.py    | For individual feature set ranking                                          |
| dd_importance_scores_generation.py       | For calculating the importance scores using evaluation metrics and measures |
| ee_similarity_scores_generation.py       | For calculating the similarity scores                                       |