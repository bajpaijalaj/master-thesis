import NSGA2_unconstrained as aa
import configparser
import os
import ast
from datetime import datetime
import concurrent.futures
import itertools
import pandas as pd
import shutil
import sys

similarity_metrics = ["spearman"]
importance_metrics = ["NDCG@10","DCG@10","ERR@10","Pfound@10"]
pop_size = 100

config = configparser.ConfigParser()
config.read('../config_'+sys.argv[1]+'.ini')

def make_single_dir(path):
    try:
        os.mkdir(path)
    except OSError as error:
        print(path+" already exists.")
        


for section in config.sections():
    dataset_name = config[section]['dataset_name']
    input_path = config[section]['input_path']
    output_path = config[section]['output_path']
    num_folds = int(config[section]['num_folds'])
    num_features = int(config[section]['num_features'])
    
    trees = config[section]['trees'].split(",")
    leaves = config[section]['leaves'].split(",")
    learning_rate = config[section]['learning_rate'].split(",")
    max_depth = config[section]['max_depth'].split(",")
    num_boost_round = config[section]['num_boost_round'].split(",")
    binary_output_path = config[section]['binary_output_path']
    output_filenames = config[section]['output_filenames'].split(",")
    ranking_models = config[section]['ranking_models'].split(",")
    make_single_dir("../output/"+dataset_name+"/fs_unconstrained")
    make_single_dir("../output/"+dataset_name+"/fs_unconstrained_evaluation")
    make_single_dir("../output/"+dataset_name+"/fs_constrained")
    make_single_dir("../output/"+dataset_name+"/fs_constrained_evaluation")
    parameters = [{} for i in range(num_folds)]
    fold_ints = [i for i in range(1,num_folds+1)]
    fold_str = ["Fold"+str(i) for i in range(1,num_folds+1)]
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
        for simi in similarity_metrics:
            for imp in importance_metrics:
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    executor.map(aa.runNSGA2_unconstrained, itertools.repeat(dataset_name), itertools.repeat(num_features), itertools.repeat(pop_size), itertools.repeat(rank), itertools.repeat(imp), itertools.repeat(simi), fold_str)