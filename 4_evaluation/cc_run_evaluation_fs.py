import os
import aa_feature_subset_extractor as cc
import bb_grouping_and_evaluation as dd
import pandas as pd
from bitarray import bitarray
import shutil
import configparser

fold_name = "Fold"
output_path = "../output/"
working_folder = "../working_folder/"
filenamess = ["train.txt","vali.txt","test.txt"]
files = ["train","vali","test"]

def make_dir(path):
    try:
        os.mkdir(path)
    except OSError as error:
        print(path+" - already exists.")


def evaluation(ranking, dataset_name, sim_metric, imp_metric, constrainedOR, i, parameters):
    if_folds = True
    dataset_path = "../0_dataset/"+dataset_name+"/"
    if if_folds == True:
        make_dir(working_folder+dataset_name+"/evaluation")
        config = configparser.ConfigParser()
        config.read('../config_'+dataset_name+'.ini')
        is_binary = int(config[dataset_name]['binary'])
        f = pd.read_csv(output_path+dataset_name+"/fs_"+constrainedOR+"/"+ranking+"_"+dataset_name+"_"+"Fold"+str(i)+"_"+sim_metric+"_"+imp_metric+".txt", delimiter = "\n", header=None)
        evaluation = []
        fold_eval = {}
        fold_eval = {"fold": "Fold"+str(i), "accuracy": 0, "count": 0, "subset": []}
        for candidate in f[0]:
            elem = bitarray(candidate)
            listOfOnes = elem.search(bitarray('1'))
            #To set actual feature columns, query id column & relevance labels column
            make_dir(working_folder+dataset_name+"/evaluation/"+fold_name+str(i))
            value = 0
            if ranking == "XGBOOST":
                feature_subset = [element + 2 for element in listOfOnes]
                feature_subset.insert(0, 1)
                feature_subset.insert(0, 0)
                cc.extract_individual_features(feature_subset, if_folds, fold_name+str(i), dataset_name, dataset_path+fold_name+str(i)+"/", filenamess)
                dd.feature_set_grouping(if_folds, fold_name, files, working_folder+dataset_name+"/evaluation/"+fold_name+str(i)+'/', working_folder+dataset_name+"/evaluation/"+fold_name+str(i)+'/',dataset_name=dataset_name)
                if is_binary == 0:
                    value = dd.predict(files, working_folder+dataset_name+"/evaluation/"+fold_name+str(i)+'/', parameters)
                else:
                    value = dd.predict_binary(files, working_folder+dataset_name+"/evaluation/"+fold_name+str(i)+'/', parameters)
            elif ranking == "LIGHTGBM":
                can = bitarray(candidate).tolist()
                feature_subset = list(map(bool,can))
                if is_binary == 0:
                    value = dd.predict_lightgbm(feature_subset, files, "../0_dataset/"+dataset_name+"/"+fold_name+str(i)+'/', parameters)
                else:
                    value = dd.predict_lightgbm_binary(feature_subset, files, "../0_dataset/"+dataset_name+"/"+fold_name+str(i)+'/', parameters)
            shutil.rmtree(working_folder+dataset_name+"/evaluation/"+fold_name+str(i))
            if value > fold_eval["accuracy"]:
                fold_eval["count"] = len(listOfOnes)
                fold_eval["subset"] = listOfOnes
                fold_eval["accuracy"] = value
        return fold_eval

def evaluation_constrained(ranking, dataset_name, sim_metric, imp_metric, constrainedOR, i, parameters, percent):
    if_folds = True
    dataset_path = "../0_dataset/"+dataset_name+"/"
    if if_folds == True:
        make_dir(working_folder+dataset_name+"/evaluation")
        config = configparser.ConfigParser()
        config.read('../config_'+dataset_name+'.ini')
        is_binary = int(config[dataset_name]['binary'])
        f = pd.read_csv(output_path+dataset_name+"/fs_"+constrainedOR+"/"+ranking+"_"+dataset_name+"_"+"Fold"+str(i)+"_"+sim_metric+"_"+imp_metric+"_"+str(percent)+".txt", delimiter = "\n", header=None)
        evaluation = []
        fold_eval = {}
        fold_eval = {"fold": "Fold"+str(i), "accuracy": 0, "count": 0, "subset": []}
        for candidate in f[0]:
            elem = bitarray(candidate)
            listOfOnes = elem.search(bitarray('1'))
            #To set actual feature columns, query id column & relevance labels column
            make_dir(working_folder+dataset_name+"/evaluation/"+fold_name+str(i))
            value = 0
            if ranking == "XGBOOST":
                feature_subset = [element + 2 for element in listOfOnes]
                feature_subset.insert(0, 1)
                feature_subset.insert(0, 0)
                cc.extract_individual_features(feature_subset, if_folds, fold_name+str(i), dataset_name, dataset_path+fold_name+str(i)+"/", filenamess)
                dd.feature_set_grouping(if_folds, fold_name, files, working_folder+dataset_name+"/evaluation/"+fold_name+str(i)+'/', working_folder+dataset_name+"/evaluation/"+fold_name+str(i)+'/',dataset_name=dataset_name)
                if is_binary == 0:
                    value = dd.predict(files, working_folder+dataset_name+"/evaluation/"+fold_name+str(i)+'/', parameters)
                else:
                    value = dd.predict_binary(files, working_folder+dataset_name+"/evaluation/"+fold_name+str(i)+'/', parameters)
            elif ranking == "LIGHTGBM":
                can = bitarray(candidate).tolist()
                feature_subset = list(map(bool,can))
                if is_binary == 0:
                    value = dd.predict_lightgbm(feature_subset, files, "../0_dataset/"+dataset_name+"/"+fold_name+str(i)+'/', parameters)
                else:
                    value = dd.predict_lightgbm_binary(feature_subset, files, "../0_dataset/"+dataset_name+"/"+fold_name+str(i)+'/', parameters)
            shutil.rmtree(working_folder+dataset_name+"/evaluation/"+fold_name+str(i))
            if value > fold_eval["accuracy"]:
                fold_eval["count"] = len(listOfOnes)
                fold_eval["subset"] = listOfOnes
                fold_eval["accuracy"] = value
        return fold_eval