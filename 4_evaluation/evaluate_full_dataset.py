import sys
import os
import aa_feature_subset_extractor as cc
import bb_grouping_and_evaluation_for_full_eval as dd
import pandas as pd
from bitarray import bitarray
import shutil
import configparser
import lightgbm
from shutil import copyfile

config = configparser.ConfigParser()
config.read('../config_'+sys.argv[1]+'.ini')

output_path = "../output/"
working_folder = "../working_folder/"
filenamess = ["train.txt","vali.txt","test.txt"]
fold_name = "Fold"
files = ["train","vali","test"]

def make_dir(path):
    try:
        os.mkdir(path)
    except OSError as error:
        print(path+" - already exists.")

for section in config.sections():
    dataset_name = config[section]['dataset_name']
    num_features = int(config[section]['num_features'])
    num_folds = int(config[section]['num_folds'])
    trees = config[section]['trees'].split(",")
    leaves = config[section]['leaves'].split(",")
    learning_rate = config[section]['learning_rate'].split(",")
    max_depth = config[section]['max_depth'].split(",")
    num_boost_round = config[section]['num_boost_round'].split(",")
    is_binary = int(config[dataset_name]['binary'])
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
        
    make_dir(working_folder+dataset_name+"/evaluation")
    evaluation = []
    for ranking in ranking_models:
        total = 0
        obj = { "ranking": ranking, "dataset": dataset_name, "accuracy": 0, "features": num_features, "Fold1": 0, "Fold2": 0, "Fold3": 0, "Fold4": 0, "Fold5": 0}
        for i in range(1,num_folds+1):
            make_dir(working_folder+dataset_name+"/evaluation/"+fold_name+str(i))
            copyfile("../0_dataset/"+dataset_name+"/"+fold_name+str(i)+'/test.txt', working_folder+dataset_name+"/evaluation/"+fold_name+str(i)+'/test.txt')
            value = 0
            
            if ranking == "XGBOOST":
                dd.feature_set_grouping(True, fold_str[i-1], files, "../0_dataset/"+dataset_name+"/"+fold_name+str(i)+'/', working_folder+dataset_name+"/evaluation/"+fold_name+str(i)+'/')
                if is_binary == 0:
                    value = dd.predict(files, working_folder+dataset_name+"/evaluation/"+fold_name+str(i)+'/', parameters[i-1], num_features)
                else:
                    value = dd.predict_binary(files, working_folder+dataset_name+"/evaluation/"+fold_name+str(i)+'/', parameters[i-1])
            elif ranking == "LIGHTGBM":
                can = [1]*num_features
                feature_subset = list(map(bool,can))
                if is_binary == 0:
                    value = dd.predict_lightgbm(feature_subset, files, "../0_dataset/"+dataset_name+"/"+fold_name+str(i)+'/', parameters[i-1])
                else:
                    value = dd.predict_lightgbm_binary(feature_subset, files, "../0_dataset/"+dataset_name+"/"+fold_name+str(i)+'/', parameters[i-1])
            total = total + value
            obj[fold_name+str(i)] = value
            shutil.rmtree(working_folder+dataset_name+"/evaluation/"+fold_name+str(i))
        total = total/num_folds
        obj["accuracy"] = total
        evaluation.append(obj)
    print(evaluation)
    temp = str(evaluation)
    f = open("../output/"+dataset_name+"_full_evaluation.txt", "w")
    f.write(temp)