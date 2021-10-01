import time
import aa_individual_feature_sets_extraction as aa
import bb_individual_feature_sets_grouping as bb
import cc_individual_feature_sets_ranking as cc
import dd_importance_scores_generation as dd
import ee_similarity_scores_generation as ee
import os
import pandas as pd
import numpy as np
from datetime import datetime
import configparser
import sys

start_time = time.time()

if_folds = True
fold_name = "Fold" ## Not Fold1
output = "../output/"
folders = ['train','vali','test']
working_folder = "../working_folder/"
rel_queryid_cols = [0,1]

config = configparser.ConfigParser()
config.read('../config_'+sys.argv[1]+'.ini')

if len(sys.argv) == 3:
    if_folds = False

def make_single_dir(path):
    try:
        os.mkdir(path)
    except OSError as error:
        print(path+" already exists.")
        
def make_dir(path):
    try:
        os.mkdir(path)
        os.mkdir(path+"/train")
        os.mkdir(path+"/vali")
        os.mkdir(path+"/test")
    except OSError as error:
        print(path+" or "+path+"/train /vali /test - already exists.")

for section in config.sections():
    dataset_name = config[section]['dataset_name']
    input_path = config[section]['input_path']
    output_path = config[section]['output_path']
    num_folds = int(config[section]['num_folds'])
    num_features = int(config[section]['num_features'])
    ranking_models = config[section]['ranking_models'].split(",")
    binary_output_path = config[section]['binary_output_path']
    output_filenames = config[section]['output_filenames'].split(",")
    is_binary = int(config[section]['binary'])
    trees = config[section]['trees'].split(",")
    leaves = config[section]['leaves'].split(",")
    learning_rate = config[section]['learning_rate'].split(",")
    max_depth = config[section]['max_depth'].split(",")
    num_boost_round = config[section]['num_boost_round'].split(",")
    trials = int(config[section]['trials'])
    

    preds = np.zeros((num_folds,num_features), dtype=np.ndarray)
    preds_binary = np.zeros((num_folds,num_features), dtype=np.ndarray)
    preds_lightgbm = np.zeros((num_folds,num_features), dtype=np.ndarray)
    preds_binary_lightgbm = np.zeros((num_folds,num_features), dtype=np.ndarray)
    
    make_single_dir("../working_folder/"+dataset_name)
    make_single_dir("../working_folder/"+dataset_name+"/binary")
    make_single_dir("../output/"+dataset_name)
    make_single_dir("../output/"+dataset_name+"/scores")
    timings_file = open("../output/timings_score_calculation.txt", "a")
    timings_file.write("Dataset: "+dataset_name+" - "+ str(datetime.now())+"\n")
    if if_folds == True:
        for i in range(1,num_folds+1):
            parameters = {}
            parameters["max_depth"] = max_depth[i-1]
            parameters["num_boost_round"] = num_boost_round[i-1]
            parameters["trees"] = trees[i-1]
            parameters["leaves"] = leaves[i-1]
            parameters["learning_rate"] = learning_rate[i-1]
            parameters["trials"] = trials
            
            make_dir(working_folder+dataset_name+"/"+fold_name+str(i))
            make_dir(working_folder+dataset_name+"/binary/"+fold_name+str(i))
            timings_file.write("Fold"+str(i)+" "+ str(datetime.now())+"\n")
            print("Individual features extraction starts .. "+str(datetime.now()))
            timings_file.write("Individual features extraction starts .. "+ str(datetime.now())+"\n")
            aa.extract_individual_features(dy_list = rel_queryid_cols, folds = if_folds, fold = fold_name+str(i), dataset = dataset_name, features = num_features, path=output_path+fold_name+str(i)+'/', filenames=output_filenames)
            aa.extract_individual_features_for_binary(dy_list = rel_queryid_cols, folds = if_folds, fold = fold_name+str(i), dataset = dataset_name, features = num_features, path=binary_output_path+fold_name+str(i)+'/', filenames=output_filenames)
            print("Individual feature set grouping starts .. "+str(datetime.now()))
            timings_file.write("Individual feature set grouping starts .. "+ str(datetime.now())+"\n")
            bb.indi_feature_sets_grouping(folds = if_folds, features = num_features, folder_names = folders, input_dir = working_folder+dataset_name+"/"+fold_name+str(i)+"/", output_dir = working_folder+dataset_name+"/"+fold_name+str(i)+"/",dataset_name=dataset_name)
            bb.indi_feature_sets_grouping(folds = if_folds, features = num_features, folder_names = folders, input_dir = working_folder+dataset_name+"/binary/"+fold_name+str(i)+"/", output_dir = working_folder+dataset_name+"/binary/"+fold_name+str(i)+"/",dataset_name=dataset_name)
            print("Prediction starts .."+str(datetime.now()))
            timings_file.write("Prediction starts .."+ str(datetime.now())+"\n")
            
            for j in range(1, num_features+1):
                if "XGBOOST" in ranking_models:
                    preds_binary[i-1][j-1] = cc.predict_for_binary(folders, working_folder+dataset_name+"/binary/"+fold_name+str(i)+"/", str(j), parameters)
                    if is_binary == 1:
                        preds[i-1][j-1] = preds_binary[i-1][j-1]
                    else:
                        preds[i-1][j-1] = cc.predict(folders, working_folder+dataset_name+"/"+fold_name+str(i)+"/", str(j), parameters)
                if "LIGHTGBM" in ranking_models:
                    preds_binary_lightgbm[i-1][j-1] = cc.predict_lightgbm_for_binary(folders, working_folder+dataset_name+"/binary/"+fold_name+str(i)+"/", str(j), parameters)
                    if is_binary == 1:
                        preds_lightgbm[i-1][j-1] = preds_binary_lightgbm[i-1][j-1]
                    else:
                        preds_lightgbm[i-1][j-1] = cc.predict_lightgbm(folders, working_folder+dataset_name+"/"+fold_name+str(i)+"/", str(j), parameters)
            
            timings_file.write("Similarity scoring starts .. "+"Fold"+str(i)+" "+str(datetime.now())+"\n")
            ee.similarity_scores_using_data_column_loop(num_features, output_path+"Fold"+str(i)+"/train.txt", output, "Fold"+str(i), dataset_name)
            timings_file.write("Similarity scoring ends .. "+"Fold"+str(i)+" "+str(datetime.now())+"\n")
        #print(str(preds))
        timings_file.write("Prediction ends .."+ str(datetime.now())+"\n")
        print("Importance scores calculation starts .. "+str(datetime.now()))
        timings_file.write("Importance scores calculation starts .. "+ str(datetime.now())+"\n")
        if "XGBOOST" in ranking_models:
            dd.generate_importance_scores(preds, preds_binary, num_folds, fold_name, num_features, output, dataset_name, working_folder, "XGBOOST", trials)
        if "LIGHTGBM" in ranking_models:
            dd.generate_importance_scores(preds_lightgbm, preds_binary_lightgbm, num_folds, fold_name, num_features, output, dataset_name, working_folder, "LIGHTGBM", trials)
        timings_file.write("Importance scores calculation ends .. "+ str(datetime.now())+"\n")
        
        print("End .. "+str(datetime.now()))
        timings_file.close()
    else:
        for f in sys.argv[2]:
            i = int(f)
            parameters = {}
            parameters["max_depth"] = max_depth[i-1]
            parameters["num_boost_round"] = num_boost_round[i-1]
            parameters["trees"] = trees[i-1]
            parameters["leaves"] = leaves[i-1]
            parameters["learning_rate"] = learning_rate[i-1]
            parameters["trials"] = trials
            
            make_dir(working_folder+dataset_name+"/"+fold_name+str(i))
            make_dir(working_folder+dataset_name+"/binary/"+fold_name+str(i))
            timings_file.write("Fold"+str(i)+" "+ str(datetime.now())+"\n")
            print("Individual features extraction starts .. "+str(datetime.now()))
            timings_file.write("Individual features extraction starts .. "+ str(datetime.now())+"\n")
            aa.extract_individual_features(dy_list = rel_queryid_cols, folds = True, fold = fold_name+str(i), dataset = dataset_name, features = num_features, path=output_path+fold_name+str(i)+'/', filenames=output_filenames)
            aa.extract_individual_features_for_binary(dy_list = rel_queryid_cols, folds = True, fold = fold_name+str(i), dataset = dataset_name, features = num_features, path=binary_output_path+fold_name+str(i)+'/', filenames=output_filenames)
            print("Individual feature set grouping starts .. "+str(datetime.now()))
            timings_file.write("Individual feature set grouping starts .. "+ str(datetime.now())+"\n")
            bb.indi_feature_sets_grouping(folds = True, features = num_features, folder_names = folders, input_dir = working_folder+dataset_name+"/"+fold_name+str(i)+"/", output_dir = working_folder+dataset_name+"/"+fold_name+str(i)+"/",dataset_name=dataset_name)
            bb.indi_feature_sets_grouping(folds = True, features = num_features, folder_names = folders, input_dir = working_folder+dataset_name+"/binary/"+fold_name+str(i)+"/", output_dir = working_folder+dataset_name+"/binary/"+fold_name+str(i)+"/",dataset_name=dataset_name)
            print("Prediction starts .."+str(datetime.now()))
            timings_file.write("Prediction starts .."+ str(datetime.now())+"\n")
            
            for j in range(1, num_features+1):
                if "XGBOOST" in ranking_models:
                    preds_binary[i-1][j-1] = cc.predict_for_binary(folders, working_folder+dataset_name+"/binary/"+fold_name+str(i)+"/", str(j), parameters)
                    if is_binary == 1:
                        preds[i-1][j-1] = preds_binary[i-1][j-1]
                    else:
                        preds[i-1][j-1] = cc.predict(folders, working_folder+dataset_name+"/"+fold_name+str(i)+"/", str(j), parameters)
                if "LIGHTGBM" in ranking_models:
                    preds_binary_lightgbm[i-1][j-1] = cc.predict_lightgbm_for_binary(folders, working_folder+dataset_name+"/binary/"+fold_name+str(i)+"/", str(j), parameters)
                    if is_binary == 1:
                        preds_lightgbm[i-1][j-1] = preds_binary_lightgbm[i-1][j-1]
                    else:
                        preds_lightgbm[i-1][j-1] = cc.predict_lightgbm(folders, working_folder+dataset_name+"/"+fold_name+str(i)+"/", str(j), parameters)
            
            timings_file.write("Similarity scoring starts .. "+"Fold"+str(i)+" "+str(datetime.now())+"\n")
            ee.similarity_scores_using_data_column_loop(num_features, output_path+"Fold"+str(i)+"/train.txt", output, "Fold"+str(i), dataset_name)
            timings_file.write("Similarity scoring ends .. "+"Fold"+str(i)+" "+str(datetime.now())+"\n")
        #print(str(preds))
        timings_file.write("Prediction ends .."+ str(datetime.now())+"\n")
        print("Importance scores calculation starts .. "+str(datetime.now()))
        timings_file.write("Importance scores calculation starts .. "+ str(datetime.now())+"\n")
        if "XGBOOST" in ranking_models:
            dd.generate_importance_scores_one_fold(i, preds, preds_binary, num_folds, fold_name, num_features, output, dataset_name, working_folder, "XGBOOST", trials)
        if "LIGHTGBM" in ranking_models:
            dd.generate_importance_scores_one_fold(i, preds_lightgbm, preds_binary_lightgbm, num_folds, fold_name, num_features, output, dataset_name, working_folder, "LIGHTGBM", trials)
        timings_file.write("Importance scores calculation ends .. "+ str(datetime.now())+"\n")
        
        print("End .. "+str(datetime.now()))
        timings_file.close()