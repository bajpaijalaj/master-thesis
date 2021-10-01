import os
import aa_feature_subset_extractor as cc
import bb_grouping_and_evaluation as dd
import pandas as pd
from bitarray import bitarray
import shutil
import pandas as pd
from bitarray import bitarray
import shutil
import configparser
import lightgbm
from shutil import copyfile
import sys

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

def eval(subset_string, ranking, dataset_name, i, parameters, is_binary):
    if_folds = True
    dataset_path = "../0_dataset/"+dataset_name+"/"
    if if_folds == True:
        make_dir(working_folder+dataset_name+"/evaluation")
        
        elem = bitarray(subset_string)
        listOfOnes = elem.search(bitarray('1'))
        #To set actual feature columns, query id column & relevance labels column
        make_dir(working_folder+dataset_name+"/evaluation/"+fold_name+str(i))
        value = 0
        if ranking == "XGBOOST":
            feature_subset = [element + 2 for element in listOfOnes]
            feature_subset.insert(0, 1)
            feature_subset.insert(0, 0)
            cc.extract_individual_features(feature_subset, if_folds, fold_name+str(i), dataset_name, dataset_path+fold_name+str(i)+"/", filenamess)
            dd.feature_set_grouping(if_folds, fold_name, files, working_folder+dataset_name+"/evaluation/"+fold_name+str(i)+'/', working_folder+dataset_name+"/evaluation/"+fold_name+str(i)+'/')
            if is_binary == 0:
                value = dd.predict(files, working_folder+dataset_name+"/evaluation/"+fold_name+str(i)+'/', parameters)
            else:
                value = dd.predict_binary(files, working_folder+dataset_name+"/evaluation/"+fold_name+str(i)+'/', parameters)
        elif ranking == "LIGHTGBM":
            can = bitarray(subset_string).tolist()
            feature_subset = list(map(bool,can))
            if is_binary == 0:
                value = dd.predict_lightgbm(feature_subset, files, "../0_dataset/"+dataset_name+"/"+fold_name+str(i)+'/', parameters)
            else:
                value = dd.predict_lightgbm_binary(feature_subset, files, "../0_dataset/"+dataset_name+"/"+fold_name+str(i)+'/', parameters)
        shutil.rmtree(working_folder+dataset_name+"/evaluation/"+fold_name+str(i))
        return value


gas_files = [ "gas_selection", "ngas_selection", "xgas_selection"]
p_to_filename = {0:"5",1:"10",2:"20",3:"30",4:"40",5:"50",6:"75"}
for ddd in ["OHSUMED-MIN","TD2003","TD2004","MQ2008","MQ2007","MSLR-WEB10K"]:
    config = configparser.ConfigParser()
    config.read('../config_'+ddd+'.ini')
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
        
        for gf in gas_files:
            gf_percent_5fold = []
            for x in range(1,num_folds+1):
                f = open("../output/FSA_results/"+dataset_name+"/Fold"+str(x)+"/"+gf+".txt", "r")
                ff = f.read().splitlines()
                gf_percent_5fold.append(ff)
            for ranking in ranking_models:
                method_eval = []
                for percentage in range(0,7):#5%,10%, etc.
                    print("Percentage: "+str(percentage)+", Ranking: "+ranking)
                    accuracy = 0
                    each_percent_accuracy = []
                    for i in range(0,num_folds):#starts with 0 for accessing the list
                        a_list = gf_percent_5fold[i][percentage].split(" ")
                        map_object = map(int, a_list)
                        list_of_integers = list(map_object)
                        subset_string = "0"*num_features
                        for li in list_of_integers:
                            subset_string = subset_string[:li-1] + '1' + subset_string[li:]
                        print(subset_string)
                        temp = eval(subset_string, ranking, dataset_name, i+1, parameters[i], is_binary)
                        each_percent_accuracy.append({'fold': 'Fold'+str(i+1), 'accuracy': temp})
                        accuracy += temp
                    accuracy = accuracy/num_folds
                    method_eval.append({"percent": percentage, gf: accuracy})
                    ddf = str(each_percent_accuracy)                    
                    f = open("../output/FSA_results/"+dataset_name+"/"+ranking+"_"+dataset_name+"_"+gf+"_"+p_to_filename[percentage]+"_evaluation.txt", "w")
                    f.write(ddf)
                df = pd.DataFrame.from_dict(method_eval)
                df.to_csv("../output/FSA_results/"+dataset_name+"/"+dataset_name+"_"+gf+"_"+ranking+".csv", sep=',', header=True, index=False)
