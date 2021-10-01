#References:
#https://github.com/hpclab/rankeval/commit/0b5090325228afe197f0708cb158ada50b8f7b7a
#https://github.com/dmlc/xgboost/blob/73b1bd27899e35ff57f5aca33f1685ea6db10f31/demo/rank/rank.py
#https://github.com/dmlc/xgboost/blob/73b1bd27899e35ff57f5aca33f1685ea6db10f31/demo/rank/trans_data.py
import xgboost as xgb
from xgboost import DMatrix
from sklearn.datasets import load_svmlight_file
import pandas as pd
from rankeval.dataset import Dataset
from rankeval.metrics import NDCG
import numpy as np
import lightgbm
import os
from rankeval.model import RTEnsemble
from datetime import datetime

def save_data(group_data,output_feature,output_group,dataset_name):
    if len(group_data) == 0:
        return

    output_group.write(str(len(group_data))+"\n")
    if dataset_name == "MQ2007":
        for data in group_data:
            # Fix for MQ2007 issue. Include zero values as well. Only size will increase.
            feats = [ p for p in data[2:] ]
            output_feature.write(data[0] + " " + " ".join(feats) + "\n")
    else:
        for data in group_data:
            # only include nonzero features
            feats = [ p for p in data[2:] if float(p.split(':')[1]) != 0.0 ]
            output_feature.write(data[0] + " " + " ".join(feats) + "\n")

def train_test_vali(arg1, arg2, arg3,dataset_name):
    fi = open(arg1)
    output_feature = open(arg2,"w")
    output_group = open(arg3,"w")

    group_data = []
    group = ""
    for line in fi:
        if not line:
            break
        if "#" in line:
            line = line[:line.index("#")]
        splits = line.strip().split(" ")
        if splits[1] != group:
            save_data(group_data,output_feature,output_group,dataset_name)
            group_data = []
        group = splits[1]
        group_data.append(splits)

    save_data(group_data,output_feature,output_group,dataset_name)

    fi.close()
    output_feature.close()
    output_group.close()
    
def feature_set_grouping(folds, fold, files = ["train","vali","test"], input_dir = "../working_folder/MSLR-WEB10K/evaluation/Fold1/", output_dir = "../working_folder/MSLR-WEB10K/evaluation/Fold1/",dataset_name=""):
    if folds == True:
        for file in files:
            train_test_vali(input_dir+file+".txt", output_dir+file+"."+file, output_dir+file+"."+file+'.group',dataset_name)
    else:
        print("Implement")
        

def predict(files, input_dir, parameters, num_features):
    x_train, y_train = load_svmlight_file(input_dir+files[0]+".train", n_features=num_features)
    x_valid, y_valid = load_svmlight_file(input_dir+files[1]+".vali", n_features=num_features)
    x_test, y_test = load_svmlight_file(input_dir+files[2]+".test", n_features=num_features)
        
    group_train = []
    with open(input_dir+files[0]+".train.group", "r") as f:
        data = f.readlines()
        for line in data:
            group_train.append(int(line.split("\n")[0]))
	
    group_valid = []
    with open(input_dir+files[1]+".vali.group", "r") as f:
        data = f.readlines()
        for line in data:
            group_valid.append(int(line.split("\n")[0]))
	
    group_test = []
    with open(input_dir+files[2]+".test.group", "r") as f:
        data = f.readlines()
        for line in data:
            group_test.append(int(line.split("\n")[0]))
	
    train_dmatrix = DMatrix(x_train, y_train)
    valid_dmatrix = DMatrix(x_valid, y_valid)
    test_dmatrix = DMatrix(x_test)
	
    train_dmatrix.set_group(group_train)
    valid_dmatrix.set_group(group_valid)
	
    params = {'objective': 'rank:ndcg', 'eta': 0.1, 'gamma': 1.0, 'min_child_weight': 0.1, 'max_depth': int(parameters["max_depth"])}
    xgb_model = xgb.train(params, train_dmatrix, num_boost_round=int(parameters["num_boost_round"]), evals=[(valid_dmatrix, 'validation')])
    pred = xgb_model.predict(test_dmatrix)
    
    test_data = Dataset.load(input_dir+files[2]+".txt")
    ndcg_10 = NDCG(cutoff=10, no_relevant_results=0.5, implementation='exp')
    ndcg_10_mean_score = ndcg_10.eval(test_data, pred)[0]
    
    return ndcg_10_mean_score
    
def predict_binary(files, input_dir, parameters):
    x_train, y_train = load_svmlight_file(input_dir+files[0]+".train")
    x_valid, y_valid = load_svmlight_file(input_dir+files[1]+".vali")
    x_test, y_test = load_svmlight_file(input_dir+files[2]+".test")
        
    group_train = []
    with open(input_dir+files[0]+".train.group", "r") as f:
        data = f.readlines()
        for line in data:
            group_train.append(int(line.split("\n")[0]))
	
    group_valid = []
    with open(input_dir+files[1]+".vali.group", "r") as f:
        data = f.readlines()
        for line in data:
            group_valid.append(int(line.split("\n")[0]))
	
    group_test = []
    with open(input_dir+files[2]+".test.group", "r") as f:
        data = f.readlines()
        for line in data:
            group_test.append(int(line.split("\n")[0]))
	
    train_dmatrix = DMatrix(x_train, y_train)
    valid_dmatrix = DMatrix(x_valid, y_valid)
    test_dmatrix = DMatrix(x_test)
	
    train_dmatrix.set_group(group_train)
    valid_dmatrix.set_group(group_valid)
	
    params = {'objective': 'rank:map', 'eta': 0.1, 'gamma': 1.0, 'min_child_weight': 0.1, 'max_depth': int(parameters["max_depth"])}
    xgb_model = xgb.train(params, train_dmatrix, num_boost_round=int(parameters["num_boost_round"]), evals=[(valid_dmatrix, 'validation')])
    pred = xgb_model.predict(test_dmatrix)
    
    test_data = Dataset.load(input_dir+files[2]+".txt")
    ndcg_10 = NDCG(cutoff=10, no_relevant_results=0.5, implementation='exp')
    ndcg_10_mean_score = ndcg_10.eval(test_data, pred)[0]
    
    return ndcg_10_mean_score
    
def predict_lightgbm(feature_subset, files, input_dir, parameters):
    msn_train = Dataset.load(input_dir+files[0]+".txt")
    msn_vali = Dataset.load(input_dir+files[1]+".txt")
    msn_test = Dataset.load(input_dir+files[2]+".txt")
    
    msn_train = msn_train.subset_features(np.array(feature_subset))
    msn_vali = msn_vali.subset_features(np.array(feature_subset))#[False,False,False]
    msn_test = msn_test.subset_features(np.array(feature_subset))
        
    lgbm_train_dataset = lightgbm.Dataset(data=msn_train.X, label=msn_train.y, group=msn_train.get_query_sizes())
    lgbm_vali_dataset = lightgbm.Dataset(data=msn_vali.X, label=msn_vali.y, group=msn_vali.get_query_sizes())
    
    msn_train = None
    msn_vali = None
    
    params = {
        'boosting_type': 'gbdt',
        'objective': 'lambdarank',
        'max_position': 10,
        'metric': ['ndcg'],
        'ndcg_at': [10],
        'num_leaves': int(parameters["leaves"]),
        'learning_rate': float(parameters["learning_rate"]),
        'verbose': -1,
        'use_missing': False
    }
    lgbm_model = lightgbm.train(
        params, 
        lgbm_train_dataset, 
        num_boost_round=int(parameters["trees"]),
        valid_sets=[lgbm_train_dataset, lgbm_vali_dataset],
        valid_names=['train', 'vali'],
        early_stopping_rounds=100,
        verbose_eval=10
    )
    

    filename = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:-3]+str(parameters["leaves"])+'lgbm.model'
    rankeval_model = None
    try:
        lgbm_model.save_model(filename=filename)
        rankeval_model = RTEnsemble(filename, 
                                    name="LightGBM model", 
                                    format="LightGBM")
    finally:
        os.remove(filename)
    
    ndcg_10 = NDCG(cutoff=10, no_relevant_results=0.5, implementation='exp')
    y_pred_test = rankeval_model.score(msn_test)
    ndcg_10_mean_score = ndcg_10.eval(msn_test, y_pred_test)[0]
    return ndcg_10_mean_score
    
def predict_lightgbm_binary(feature_subset, files, input_dir, parameters):
    msn_train = Dataset.load(input_dir+files[0]+".txt")
    msn_vali = Dataset.load(input_dir+files[1]+".txt")
    msn_test = Dataset.load(input_dir+files[2]+".txt")
    
    msn_train = msn_train.subset_features(np.array(feature_subset))
    msn_vali = msn_vali.subset_features(np.array(feature_subset))#[False,False,False]
    msn_test = msn_test.subset_features(np.array(feature_subset))
        
    lgbm_train_dataset = lightgbm.Dataset(data=msn_train.X, label=msn_train.y, group=msn_train.get_query_sizes())
    lgbm_vali_dataset = lightgbm.Dataset(data=msn_vali.X, label=msn_vali.y, group=msn_vali.get_query_sizes())
    
    msn_train = None
    msn_vali = None
    
    params = {
        'boosting_type': 'gbdt',
        'objective': 'lambdarank',
        'max_position': 10,
        'metric': ['map'],
        'ndcg_at': [10],
        'num_leaves': int(parameters["leaves"]),
        'learning_rate': float(parameters["learning_rate"]),
        'verbose': -1,
        'use_missing': False
    }
    lgbm_model = lightgbm.train(
        params, 
        lgbm_train_dataset, 
        num_boost_round=int(parameters["trees"]),
        valid_sets=[lgbm_train_dataset, lgbm_vali_dataset],
        valid_names=['train', 'vali'],
        early_stopping_rounds=100,
        verbose_eval=10
    )
    

    filename = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:-3]+str(parameters["leaves"])+'lgbm.model'
    rankeval_model = None
    try:
        lgbm_model.save_model(filename=filename)
        rankeval_model = RTEnsemble(filename, 
                                    name="LightGBM model", 
                                    format="LightGBM")
    finally:
        os.remove(filename)
    
    ndcg_10 = NDCG(cutoff=10, no_relevant_results=0.5, implementation='exp')
    y_pred_test = rankeval_model.score(msn_test)
    ndcg_10_mean_score = ndcg_10.eval(msn_test, y_pred_test)[0]
    return ndcg_10_mean_score