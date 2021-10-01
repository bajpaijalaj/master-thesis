# Preprocessing

The main scripts and the order in which they should be executed are:

| Scripts                                     | Description                                                                                     |
|---------------------------------------------|-------------------------------------------------------------------------------------------------|
| main_preprocessing.py                       | For removing unneeded columns and preparing correct input file names (uses prepare_datasets.py) |
| hyper_parameter_optimization.py             | For hyper parameter optimization                                                                |
| find_cutoff_for_binary_relevance.py         | For finding the cutoff relevance label for binary dataset generation                            |
| generate_binary_relevance_labels_dataset.py | For generating the datasets with binary relevance labels (uses generate_datasets.py). The cutoff value should be updated in the config file before running this.            |