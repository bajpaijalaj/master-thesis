import bb_NSGA2_constrained as bb
import ee_run_evaluation_fs as ee
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
pop_size = 100
sim_evaluated = "spearman"
imp_evaluated = "NDCG@10"

percent_list = [5, 10, 20, 30, 40, 50, 75]

for section in config.sections():
    dataset_name = config[section]['dataset_name']
    num_features = int(config[section]['num_features'])
    num_folds = int(config[section]['num_folds'])
    
    constraint_num_features = [round((elem/100)*num_features) for elem in percent_list]
    ranking_models = config[section]['ranking_models'].split(",")
    #best_accuracy, best_imp, best_sim = 0, "", ""
    fold_str = ["Fold"+str(i) for i in range(1,num_folds+1)]
    
    for i in range(1,num_folds+1):
        try:
            shutil.rmtree("../working_folder/"+dataset_name+"/evaluation/Fold"+str(i))
        except:
            ...
    for ranking in ranking_models:
        for constraint_zip in zip(constraint_num_features, percent_list):
            with concurrent.futures.ProcessPoolExecutor() as executor:
                executor.map(bb.runNSGA2_constrained, itertools.repeat(dataset_name), itertools.repeat(constraint_zip), itertools.repeat(num_features), itertools.repeat(pop_size), itertools.repeat(ranking), itertools.repeat(imp_evaluated), itertools.repeat(sim_evaluated), fold_str)