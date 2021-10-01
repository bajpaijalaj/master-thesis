#References:
#https://github.com/hpclab/rankeval/commit/0b5090325228afe197f0708cb158ada50b8f7b7a
import pandas as pd
import os
from rankeval.dataset import Dataset
import lightgbm
from rankeval.metrics import MAP
from sklearn.datasets import load_svmlight_file
from rankeval.model import RTEnsemble
import numpy as np
import configparser
import sys

def make_dir(path):
    try:
        os.mkdir(path)
    except OSError as error:
        print(path+" - already exists.")

def predicttt(trees, leaves, learning_rate, msn_train, msn_vali, msn_test):
    lgbm_train_dataset = lightgbm.Dataset(data=msn_train.X, label=msn_train.y, group=msn_train.get_query_sizes(), params={'verbose': -1}, free_raw_data=False)
    lgbm_vali_dataset = lightgbm.Dataset(data=msn_vali.X, label=msn_vali.y, group=msn_vali.get_query_sizes(), params={'verbose': -1}, free_raw_data=False)
        
    params = {
        'boosting_type': 'gbdt',
        'objective': 'lambdarank',
        'metric': ['map'],
        'map_at': [10],
        'num_leaves': leaves,
        'learning_rate': learning_rate,
        'verbose': 1,
        'use_missing': False
    }
    lgbm_model = lightgbm.train(
        params, 
        lgbm_train_dataset, 
        num_boost_round=trees,
        valid_sets=[lgbm_train_dataset, lgbm_vali_dataset],
        valid_names=['train', 'vali'],
        early_stopping_rounds=100,
        verbose_eval=10
    )
    

    filename = 'lgbm.model'
    rankeval_model = None
    try:
        lgbm_model.save_model(filename=filename)
        rankeval_model = RTEnsemble(filename, 
                                    name="LightGBM model", 
                                    format="LightGBM")
    finally:
        os.remove(filename)
    return rankeval_model

########################################################
config = configparser.ConfigParser()
config.read('../config_'+sys.argv[1]+'.ini')

make_dir("../0_dataset/binary")

for section in config.sections():
    max_label = int(config[section]['max_label'])
    ones = max_label
    new_labels = []
    for i in range(1,max_label+1):
        temp = []
        for j in range(1,i+1):
            temp.append(0)
        for k in range(ones,0,-1):
            temp.append(1)
        ones -=1
        new_labels.append(temp)
    
    
    dataset_name = config[section]['dataset_name']
    make_dir("../0_dataset/binary/"+dataset_name+"")
    binary_output_path = config[section]['binary_output_path']
    
    binary = int(config[section]['binary'])
    num_folds = int(config[section]['num_folds'])
    num_features = int(config[section]['num_features'])
    trees = config[section]['trees'].split(",")
    leaves = config[section]['leaves'].split(",")
    learning_rate = config[section]['learning_rate'].split(",")
    
    output_path = binary_output_path
    filenamess = ["train.txt","vali.txt","test.txt"]
    
    input_path = "../0_dataset/"+dataset_name+"/"
    
    
    
    old_labels = list(range(0, max_label+1)) #[0,1,2]
    
    map_accuracy_array = [None]*max_label
    map_accuracy_array = [{"dataset_name": dataset_name, "relevance_>=": i+1, "MAP@10": 0, "Fold1": 0, "Fold2": 0, "Fold3": 0, "Fold4": 0, "Fold5": 0}  for i,value in enumerate(map_accuracy_array)]
        
    for i in range(0,len(new_labels)):
        #print(new_labels[i])
        make_dir("../0_dataset/binary/"+dataset_name+"/"+str(i+1))
        map_accuracy = 0
        for j in range(1,num_folds+1):
            make_dir(output_path+str(i+1)+"/Fold"+str(j))
            
            data = pd.read_csv(input_path+"Fold"+str(j)+'/'+filenamess[0], header=None, sep=" ")
            data[0] = data[0].replace(old_labels, new_labels[i])
            data.to_csv(output_path+str(i+1)+"/Fold"+str(j)+"/train.txt", sep=' ', header=False, index=False)
            
            data = pd.read_csv(input_path+"Fold"+str(j)+'/'+filenamess[1], header=None, sep=" ")
            data[0] = data[0].replace(old_labels, new_labels[i])
            data.to_csv(output_path+str(i+1)+"/Fold"+str(j)+"/vali.txt", sep=' ', header=False, index=False)
            
            data = pd.read_csv(input_path+"Fold"+str(j)+'/'+filenamess[2], header=None, sep=" ")
            data[0] = data[0].replace(old_labels, new_labels[i])
            data.to_csv(output_path+str(i+1)+"/Fold"+str(j)+"/test.txt", sep=' ', header=False, index=False)
        
            msn_train = Dataset.load(output_path+str(i+1)+"/Fold"+str(j)+"/train.txt")
            msn_vali = Dataset.load(output_path+str(i+1)+"/Fold"+str(j)+"/vali.txt")
            msn_test = Dataset.load(output_path+str(i+1)+"/Fold"+str(j)+"/test.txt")
            
            msn_lgbm_lmart_1Ktrees_model = predicttt(int(trees[j-1]), int(leaves[j-1]), float(learning_rate[j-1]), msn_train, msn_vali, msn_test)
            
            y_pred_test = msn_lgbm_lmart_1Ktrees_model.score(msn_test)
            map = MAP(cutoff=10)
            map_10_mean_score = map.eval(msn_test, y_pred_test)[0]
            
            map_accuracy_array[i]["Fold"+str(j)] = map_10_mean_score
            map_accuracy += map_10_mean_score
            
        map_accuracy = map_accuracy / num_folds
        map_accuracy_array[i]["MAP@10"] = map_accuracy
        print("DONE: "+str(new_labels[i]))
    print(str(map_accuracy_array))
    f = open("../output/"+dataset_name+"_binary_relevance_cutoff.txt", "w")
    f.write(str(map_accuracy_array)+"\n")
    f.close()



