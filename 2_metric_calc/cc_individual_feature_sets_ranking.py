#References:
#https://github.com/hpclab/rankeval/commit/0b5090325228afe197f0708cb158ada50b8f7b7a
#https://github.com/dmlc/xgboost/blob/73b1bd27899e35ff57f5aca33f1685ea6db10f31/demo/rank/rank.py
import xgboost as xgb
from xgboost import DMatrix
from sklearn.datasets import load_svmlight_file
#from sklearn.metrics import ndcg_score
import pandas as pd
from rankeval.dataset import Dataset
import lightgbm
import os
from rankeval.model import RTEnsemble
from datetime import datetime

def predict(folder_names, input_dir, filename, parameters):
    x_train, y_train = load_svmlight_file(input_dir+folder_names[0]+"/"+filename+".train")
    x_valid, y_valid = load_svmlight_file(input_dir+folder_names[1]+"/"+filename+".vali")
    x_test, y_test = load_svmlight_file(input_dir+folder_names[2]+"/"+filename+".test")

    group_train = []
    with open(input_dir+folder_names[0]+"/"+filename+".train.group", "r") as f:
        data = f.readlines()
        for line in data:
            group_train.append(int(line.split("\n")[0]))
	
    group_valid = []
    with open(input_dir+folder_names[1]+"/"+filename+".vali.group", "r") as f:
        data = f.readlines()
        for line in data:
            group_valid.append(int(line.split("\n")[0]))
	
    group_test = []
    with open(input_dir+folder_names[2]+"/"+filename+".test.group", "r") as f:
        data = f.readlines()
        for line in data:
            group_test.append(int(line.split("\n")[0]))
	
    train_dmatrix = DMatrix(x_train, y_train)
    valid_dmatrix = DMatrix(x_valid, y_valid)
    test_dmatrix = DMatrix(x_test)
	
    train_dmatrix.set_group(group_train)
    valid_dmatrix.set_group(group_valid)
	
    params = {'objective': 'rank:ndcg', 'eta': 0.1, 'gamma': 1.0, 'min_child_weight': 0.1, 'max_depth': int(parameters["max_depth"])}
    trials = []
    for i in range(int(parameters["trials"])):
        xgb_model = xgb.train(params, train_dmatrix, num_boost_round=int(parameters["num_boost_round"]), evals=[(valid_dmatrix, 'validation')])
        trials.append(xgb_model.predict(test_dmatrix))
        
    return trials

# Metric change for binary : MAP
def predict_for_binary(folder_names, input_dir, filename, parameters):
    x_train, y_train = load_svmlight_file(input_dir+folder_names[0]+"/"+filename+".train")
    x_valid, y_valid = load_svmlight_file(input_dir+folder_names[1]+"/"+filename+".vali")
    x_test, y_test = load_svmlight_file(input_dir+folder_names[2]+"/"+filename+".test")
        
    group_train = []
    with open(input_dir+folder_names[0]+"/"+filename+".train.group", "r") as f:
        data = f.readlines()
        for line in data:
            group_train.append(int(line.split("\n")[0]))
	
    group_valid = []
    with open(input_dir+folder_names[1]+"/"+filename+".vali.group", "r") as f:
        data = f.readlines()
        for line in data:
            group_valid.append(int(line.split("\n")[0]))
	
    group_test = []
    with open(input_dir+folder_names[2]+"/"+filename+".test.group", "r") as f:
        data = f.readlines()
        for line in data:
            group_test.append(int(line.split("\n")[0]))
	
    train_dmatrix = DMatrix(x_train, y_train)
    valid_dmatrix = DMatrix(x_valid, y_valid)
    test_dmatrix = DMatrix(x_test)
	
    train_dmatrix.set_group(group_train)
    valid_dmatrix.set_group(group_valid)
	
    params = {'objective': 'rank:map', 'eta': 0.1, 'gamma': 1.0, 'min_child_weight': 0.1, 'max_depth': int(parameters["max_depth"])}
    trials = []
    for i in range(int(parameters["trials"])):
        xgb_model = xgb.train(params, train_dmatrix, num_boost_round=int(parameters["num_boost_round"]), evals=[(valid_dmatrix, 'validation')])
        trials.append(xgb_model.predict(test_dmatrix))
        
    return trials
    
def predict_lightgbm(folder_names, input_dir, filename, parameters):
    msn_train = Dataset.load(input_dir+folder_names[0]+"/"+filename+".txt")
    msn_vali = Dataset.load(input_dir+folder_names[1]+"/"+filename+".txt")
    msn_test = Dataset.load(input_dir+folder_names[2]+"/"+filename+".txt")
    
    lgbm_train_dataset = lightgbm.Dataset(data=msn_train.X, label=msn_train.y, group=msn_train.get_query_sizes())
    lgbm_vali_dataset = lightgbm.Dataset(data=msn_vali.X, label=msn_vali.y, group=msn_vali.get_query_sizes())
    
    params = {
        'boosting_type': 'gbdt',
        'objective': 'lambdarank',
        'max_position': 10,
        'metric': ['ndcg'],
        'ndcg_at': [10],
        'num_leaves': int(parameters["leaves"]),
        'learning_rate': float(parameters["learning_rate"]),
        'verbose': 1,
        'use_missing': False
    }
    trials = []
    for i in range(int(parameters["trials"])):
        lgbm_model = lightgbm.train(
            params, 
            lgbm_train_dataset, 
            num_boost_round=int(parameters["trees"]),
            valid_sets=[lgbm_train_dataset, lgbm_vali_dataset],
            valid_names=['train', 'vali'],
            early_stopping_rounds=100,
            verbose_eval=10
        )
        
    
        filename = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:-3]+'lgbm.model'
        rankeval_model = None
        try:
            lgbm_model.save_model(filename=filename)
            rankeval_model = RTEnsemble(filename, 
                                        name="LightGBM model", 
                                        format="LightGBM")
        finally:
            os.remove(filename)
        trials.append(rankeval_model.score(msn_test))
        
    return trials

# Metric change for binary : MAP    
def predict_lightgbm_for_binary(folder_names, input_dir, filename, parameters):
    msn_train = Dataset.load(input_dir+folder_names[0]+"/"+filename+".txt")
    msn_vali = Dataset.load(input_dir+folder_names[1]+"/"+filename+".txt")
    msn_test = Dataset.load(input_dir+folder_names[2]+"/"+filename+".txt")
    
    lgbm_train_dataset = lightgbm.Dataset(data=msn_train.X, label=msn_train.y, group=msn_train.get_query_sizes())
    lgbm_vali_dataset = lightgbm.Dataset(data=msn_vali.X, label=msn_vali.y, group=msn_vali.get_query_sizes())
    
    params = {
        'boosting_type': 'gbdt',
        'objective': 'lambdarank',
        'max_position': 10,
        'metric': ['map'],
        'ndcg_at': [10],
        'num_leaves': int(parameters["leaves"]),
        'learning_rate': float(parameters["learning_rate"]),
        'verbose': 1,
        'use_missing': False
    }
    trials = []
    for i in range(int(parameters["trials"])):
        lgbm_model = lightgbm.train(
            params, 
            lgbm_train_dataset, 
            num_boost_round=int(parameters["trees"]),
            valid_sets=[lgbm_train_dataset, lgbm_vali_dataset],
            valid_names=['train', 'vali'],
            early_stopping_rounds=100,
            verbose_eval=10
        )
        
    
        filename = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:-3]+'lgbm.model'
        rankeval_model = None
        try:
            lgbm_model.save_model(filename=filename)
            rankeval_model = RTEnsemble(filename, 
                                        name="LightGBM model", 
                                        format="LightGBM")
        finally:
            os.remove(filename)
        trials.append(rankeval_model.score(msn_test))
        
    return trials