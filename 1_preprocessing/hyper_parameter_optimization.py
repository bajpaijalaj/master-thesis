#References:
#https://github.com/hpclab/rankeval/commit/0b5090325228afe197f0708cb158ada50b8f7b7a
#https://github.com/dmlc/xgboost/blob/73b1bd27899e35ff57f5aca33f1685ea6db10f31/demo/rank/rank.py
import os
import pandas as pd
from rankeval.metrics import NDCG,MAP
import xgboost as xgb
from xgboost import DMatrix
from sklearn.datasets import load_svmlight_file
from rankeval.dataset import Dataset
import lightgbm
import os
from rankeval.model import RTEnsemble
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.pyll.base import scope
import math
import concurrent.futures
import itertools
import configparser
from functools import partial
import time
import sys

def make_dir(path):
    try:
        os.mkdir(path)
    except OSError as error:
        print(path+" already exists.")

def save_data(group_data,output_feature,output_group):
    if len(group_data) == 0:
        return

    output_group.write(str(len(group_data))+"\n")
    for data in group_data:
        # only include nonzero features
        feats = [ p for p in data[2:] if float(p.split(':')[1]) != 0.0 ]
        output_feature.write(data[0] + " " + " ".join(feats) + "\n")

def train_test_vali(arg1, arg2, arg3):
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
            save_data(group_data,output_feature,output_group)
            group_data = []
        group = splits[1]
        group_data.append(splits)

    save_data(group_data,output_feature,output_group)

    fi.close()
    output_feature.close()
    output_group.close()

def indi_feature_sets_grouping(folds, files, input_dir, output_dir):
    if folds == True:
        for file in files:
            train_test_vali(input_dir+file+".txt", output_dir+file+"."+file, output_dir+file+'.'+file+'.group')
    else:
        print("Implement")


def predict(parameters, ii):    
    if binary == 0:
        params = {'objective': 'rank:ndcg', 'eta': 0.1, 'gamma': 1.0, 'min_child_weight': 0.1, 'max_depth': parameters["max_depth"], 'verbosity': 0}
    else:
        params = {'objective': 'rank:map', 'eta': 0.1, 'gamma': 1.0, 'min_child_weight': 0.1, 'max_depth': parameters["max_depth"], 'verbosity': 0}
    
    xgb_model = xgb.train(params, train_dmatrix[ii], num_boost_round=parameters["num_boost_round"], evals=[(valid_dmatrix[ii], 'validation')])
    
    try:
        preds = xgb_model.predict(test_dmatrix[ii])
        
        if binary == 0:
            ndcg_10 = NDCG(cutoff=10, no_relevant_results=0.5, implementation='exp')
            ndcg_10_mean_score = ndcg_10.eval(test_data[ii], preds)[0]
            return ndcg_10_mean_score
        else:
            map_10 = MAP(cutoff=10)
            map_avg_score = map_10.eval(test_data[ii], preds)[0]
            return map_avg_score
    except:
        print(input_dir[ii])
    

def predict_lightgbm(parameters, ii):
    if binary == 0:
        params = {
            'boosting_type': 'gbdt',
            'objective': 'lambdarank',
            'metric': ['ndcg'],
            'ndcg_at': [10],
            'num_leaves': parameters["leaves"],
            'learning_rate': parameters["learning_rate"],
            'verbose': -1,
            'use_missing': False
        }
    else:
        params = {
            'boosting_type': 'gbdt',
            'objective': 'lambdarank',
            'metric': ['map'],
            'map_at': [10],
            'num_leaves': parameters["leaves"],
            'learning_rate': parameters["learning_rate"],
            'verbose': -1,
            'use_missing': False
        }
        
    lgbm_model = lightgbm.train(
        params, 
        lgbm_train_dataset[ii], 
        num_boost_round=parameters["trees"],
        valid_sets=[lgbm_train_dataset[ii], lgbm_vali_dataset[ii]],
        valid_names=['train', 'vali'],
        early_stopping_rounds=100,
        verbose_eval=False #10
    )
    

    filename = dataset_name+str(ii)+'_lgbm.model'
    rankeval_model = None
    try:
        lgbm_model.save_model(filename=filename)
        rankeval_model = RTEnsemble(filename, name="LightGBM model", format="LightGBM")
    finally:
        os.remove(filename)
    y_pred_test = rankeval_model.score(msn_test[ii])
    if binary == 0:
        ndcg_10 = NDCG(cutoff=10, no_relevant_results=0.5, implementation='exp')
        ndcg_10_mean_score = ndcg_10.eval(msn_test[ii], y_pred_test)[0]
        return ndcg_10_mean_score
    else:
        map_10 = MAP(cutoff=10)
        map_avg_score = map_10.eval(msn_test[ii], y_pred_test)[0]
        return map_avg_score

def eval(space, ii):
    
    files = ["train","vali","test"]
    
    if ranking == "XGBOOST":
        parameters = {"max_depth": space["max_depth"], "num_boost_round": space["num_boost_round"]}
        return predict(parameters,ii)                   
    elif ranking == "LIGHTGBM":
        parameters = {"trees": space["trees"], "leaves": space["leaves"], "learning_rate": round(space["learning_rate"],2)}
        return predict_lightgbm(parameters,ii)

def optimization(ii):
    if ranking=="XGBOOST":
        space = {'max_depth': scope.int(hp.quniform('max_depth', 1, 500, 1)), 'num_boost_round': scope.int(hp.quniform('num_boost_round', 1, 500, 1))}
        df = pd.DataFrame(columns = ["NDCG10/MAP10","max_depth","num_boost_round"])
        
        x_train[ii], y_train[ii] = load_svmlight_file(input_dir[ii]+filenames[0]+".train", n_features=num_features)
        x_valid[ii], y_valid[ii] = load_svmlight_file(input_dir[ii]+filenames[1]+".vali", n_features=num_features)
        x_test[ii], y_test[ii] = load_svmlight_file(input_dir[ii]+filenames[2]+".test", n_features=num_features)
            
        group_train = []
        with open(input_dir[ii]+filenames[0]+".train.group", "r") as f:
            data = f.readlines()
            for line in data:
                group_train.append(int(line.split("\n")[0]))
        
        group_valid = []
        with open(input_dir[ii]+filenames[1]+".vali.group", "r") as f:
            data = f.readlines()
            for line in data:
                group_valid.append(int(line.split("\n")[0]))
        
        train_dmatrix[ii] = DMatrix(x_train[ii], y_train[ii])
        valid_dmatrix[ii] = DMatrix(x_valid[ii], y_valid[ii])
        test_dmatrix[ii] = DMatrix(x_test[ii])
        
        train_dmatrix[ii].set_group(group_train)
        valid_dmatrix[ii].set_group(group_valid)
        test_data[ii] = Dataset.load(test_file_path[ii]+filenames[2]+".txt")
    elif ranking=="LIGHTGBM":
        space = {'trees': scope.int(hp.quniform('trees', 1, 500, 1)), 'leaves': scope.int(hp.quniform('leaves', 2, 500, 1)), 'learning_rate': hp.quniform('learning_rate', 0.05, 0.15, 0.05)}
        df = pd.DataFrame(columns = ["NDCG10/MAP10","trees","leaves","learning_rate"])
        msn_train[ii] = Dataset.load(input_dir_lightgbm[ii]+filenames[0]+".txt")
        msn_vali[ii] = Dataset.load(input_dir_lightgbm[ii]+filenames[1]+".txt")
        msn_test[ii] = Dataset.load(input_dir_lightgbm[ii]+filenames[2]+".txt")
        
        lgbm_train_dataset[ii] = lightgbm.Dataset(data=msn_train[ii].X, label=msn_train[ii].y, group=msn_train[ii].get_query_sizes(), params={'verbose': -1}, free_raw_data=False)
        lgbm_vali_dataset[ii] = lightgbm.Dataset(data=msn_vali[ii].X, label=msn_vali[ii].y, group=msn_vali[ii].get_query_sizes(), params={'verbose': -1}, free_raw_data=False)
    
    trials = Trials()
    best = fmin(fn=partial(eval, ii=ii), space=space, algo= tpe.suggest, max_evals = 500, trials= trials, verbose=0)
    leng = len(input_dir[ii])        
    
    if ranking=="XGBOOST":
        for elem in trials.trials:
            #print(elem["result"]["loss"])
            df = df.append({'NDCG10/MAP10': elem["result"]["loss"], 'max_depth': elem["misc"]["vals"]["max_depth"][0], 'num_boost_round': elem["misc"]["vals"]["num_boost_round"][0]}, ignore_index=True)
    elif ranking=="LIGHTGBM":
        for elem in trials.trials:
            df = df.append({'NDCG10/MAP10': elem["result"]["loss"], 'trees': elem["misc"]["vals"]["trees"][0], 'leaves': elem["misc"]["vals"]["leaves"][0], 'learning_rate': round(elem["misc"]["vals"]["learning_rate"][0],2)}, ignore_index=True)
    df = df.sort_values(by=['NDCG10/MAP10'], ascending=False)
    df.to_csv("../output/hyperopt/trials_file_"+dataset_name+"_"+ranking+"_"+input_dir[ii][leng-6:leng-1]+".csv", index=False)
    

######################################################################
make_dir("../working_folder/parameter_tuning")
make_dir("../output/hyperopt")
config = configparser.ConfigParser()
config.read('../config_'+sys.argv[1]+'.ini')
train_dmatrix = [None, None, None, None, None]
valid_dmatrix = [None, None, None, None, None]
test_dmatrix = [None, None, None, None, None]
x_train = [None, None, None, None, None]
y_train = [None, None, None, None, None]
x_valid = [None, None, None, None, None]
y_valid = [None, None, None, None, None]
x_test = [None, None, None, None, None]
y_test = [None, None, None, None, None]
input_dir = [None, None, None, None, None]
input_dir_lightgbm = [None, None, None, None, None]
test_file_path = [None, None, None, None, None]
test_data = [None, None, None, None, None]
filenames = ["train","vali","test"]

msn_train = [None, None, None, None, None]
msn_vali = [None, None, None, None, None]
msn_test = [None, None, None, None, None]
lgbm_train_dataset = [None, None, None, None, None]
lgbm_vali_dataset = [None, None, None, None, None]

ii = [None, None, None, None, None]
ranking = None
dataset_name = None
num_features = None
binary = None
for section in config.sections():
    dataset_name = config[section]['dataset_name']
    binary = int(config[section]['binary'])
    num_folds = int(config[section]['num_folds'])
    num_features = int(config[section]['num_features'])
    ranking_models = config[section]['ranking_models'].split(",")
    files = ["train","vali","test"]
    
    make_dir("../working_folder/parameter_tuning/"+dataset_name)

    for i in range(1,num_folds+1):
        make_dir("../working_folder/parameter_tuning/"+dataset_name+"/Fold"+str(i))
        indi_feature_sets_grouping(True, files, "../0_dataset/"+dataset_name+"/Fold"+str(i)+"/", "../working_folder/parameter_tuning/"+dataset_name+"/Fold"+str(i)+"/")
    
    
    for j in range(0,num_folds):
        ii[j] = j
        input_dir[j] = "../working_folder/parameter_tuning/"+dataset_name+"/Fold"+str(j+1)+"/"
        input_dir_lightgbm[j] = "../0_dataset/"+dataset_name+"/Fold"+str(j+1)+"/"
        test_file_path[j] = "../0_dataset/"+dataset_name+"/Fold"+str(j+1)+"/"
        
    for rankingg in ranking_models:
        ranking = rankingg
        with concurrent.futures.ThreadPoolExecutor() as executor:
                    temp = executor.map(optimization, ii)

