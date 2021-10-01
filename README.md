# Feature Selection for Ranking
This repository contains the implementation for the Master's thesis titled 'Feature Selection for Ranking'.

## Abstract
This work reports the importance of feature selection for ranking and how it is often formalized as a multi-objective optimization task. It solves the multi-objective optimization task for feature selection by maximizing the total importance scores and minimizing the total similarity scores of features. This is achieved by utilizing a binary encoded scheme with an unconstrained elitist multi-objective genetic algorithm (NSGA-II), which is able to reduce the number of features and at the same time improve ranking performance. Additionally, combinations of a variety of ranking metrics and measures are employed as objective values in the approach and found to be equally effective. A constrained variant of the approach is also used to select defined feature subsets and found to perform better than other evaluated methods, in some cases even selecting fewer features. This empirical study is conducted on a variety of learning to rank benchmark datasets and the implementation is made available for further research.
   
   
## Repository structure
| Folder/Files        | Description                                                                               |
| ------------------- | ----------------------------------------------------------------------------------------- |
| 0_dataset           | Folder to keep the datasets                                                               |
| 1_preprocessing     | Folder contains the scripts needed for preprocessing the data                             |
| 2_metric_calc       | Folder contains the scripts needed for feature importance & similarity scores calculation |
| 3_feature_selection | Folder contains the scripts for feature selection                                         |
| 4_evaluation        | Folder contains the scripts for ranking evaluations                                       |
| output              | Folder contains the output results of the experiments                                     |
| sample_job_scripts  | Folder contains some sample job scripts that were used for running the experiments on HPC |
| working_folder      | Empty folder that is used during the executions                                           |
| config.ini          | File containing the configurations used for all datasets                                  |
| config_*.ini        | Individual dataset configurations for running for a single dataset                        |

> Folder contains respective *<span>README.md</span>* files for description.

## How to run
Traverse through each of the folders in the following order and execute the desired scripts/tasks as per the readme instructions:  
  
0_dataset -> 1_preprocessing -> 2_metric_calc -> 3_feature_selection -> 4_evaluation
