import bb_NSGA2_constrained as bb
import cc_run_evaluation_fs as ee
import configparser
import os
import ast
from datetime import datetime
import ast
import concurrent.futures
import itertools
import shutil
import sys

config = configparser.ConfigParser()
config.read('../config_'+sys.argv[1]+'.ini')
if_folds = True
sim_evaluated = "spearman"
imp_evaluated = "NDCG@10"

percent_list = []#[5, 10, 20, 30, 40, 50, 75]
percent_list.append(int(sys.argv[2]))

for section in config.sections():
    dataset_name = config[section]['dataset_name']
    num_features = int(config[section]['num_features'])
    num_folds = int(config[section]['num_folds'])
    trees = config[section]['trees'].split(",")
    leaves = config[section]['leaves'].split(",")
    learning_rate = config[section]['learning_rate'].split(",")
    max_depth = config[section]['max_depth'].split(",")
    num_boost_round = config[section]['num_boost_round'].split(",")
    constraint_num_features = [round((elem/100)*num_features) for elem in percent_list]
    ranking_models = config[section]['ranking_models'].split(",")
    fold_str = ["Fold"+str(i) for i in range(1,num_folds+1)]
    fold_ints = [i for i in range(1,num_folds+1)]
    parameters = [{} for i in range(num_folds)]
    
    for i in range(1,num_folds+1):
        parameters[i-1]["max_depth"] = max_depth[i-1]
        parameters[i-1]["num_boost_round"] = num_boost_round[i-1]
        parameters[i-1]["trees"] = trees[i-1]
        parameters[i-1]["leaves"] = leaves[i-1]
        parameters[i-1]["learning_rate"] = learning_rate[i-1]
        try:
            shutil.rmtree("../working_folder/"+dataset_name+"/evaluation/Fold"+str(i))
        except:
            ...
    
    for rank in ranking_models:
        for percent in percent_list:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                fold_results = executor.map(ee.evaluation_constrained, itertools.repeat(rank), itertools.repeat(dataset_name), itertools.repeat(sim_evaluated), itertools.repeat(imp_evaluated), itertools.repeat("constrained"), fold_ints, parameters, itertools.repeat(percent))
                
                temp = str(list(fold_results))
                f = open("../output/"+dataset_name+"/fs_constrained_evaluation/"+rank+"_"+dataset_name+"_"+sim_evaluated+"_"+imp_evaluated+"_"+str(percent)+"_evaluation.txt", "w")
                f.write(temp)
                f.close()
                
    