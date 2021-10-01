import time
import os
from rankeval.dataset import Dataset
from rankeval.metrics import NDCG,DCG,ERR,Pfound,MAP,MRR,Precision,Recall
import numpy as np

start_time = time.time()
if_folds = True

def make_dir(path):
    try:
        os.mkdir(path)
    except OSError as error:
        print(path+" - already exists.")

def generate_importance_scores(preds, preds_binary, num_folds, fold_name, num_features, output_path, dataset_name, working_folder, ranking, trials):
    for i in range(1,num_folds+1):
        feature_metrics_array = [None]*num_features
        feature_metrics_array = [{"feature": i+1, "NDCG@10": 0, "DCG@10": 0, "ERR@10": 0, "Pfound@10": 0, "MAP@10": 0, "MRR@10": 0, "F1@10": 0} for i,value in enumerate(feature_metrics_array)]
        for j in range(1, num_features+1):
            data = Dataset.load(working_folder+dataset_name+"/"+fold_name+str(i)+"/test/"+str(j)+".txt")
            data_binary = Dataset.load(working_folder+dataset_name+"/binary/"+fold_name+str(i)+"/test/"+str(j)+".txt")
            
            #NDCG@10
            ndcg_10 = NDCG(cutoff=10, no_relevant_results=0.5, implementation='exp')
            ndcg_10_mean_score = 0
            for elem in preds[i-1][j-1]:
                ndcg_10_mean_score += ndcg_10.eval(data, elem)[0]
            feature_metrics_array[j-1]["NDCG@10"] = ndcg_10_mean_score/trials
            
            #DCG
            dcg_10 = DCG(cutoff=10)
            dcg_avg_score = 0
            for elem in preds[i-1][j-1]:
                dcg_avg_score += dcg_10.eval(data, elem)[0]
            feature_metrics_array[j-1]["DCG@10"] = dcg_avg_score/trials
            
            #ERR
            err_10 = ERR(cutoff=10)
            err_avg_score = 0
            for elem in preds[i-1][j-1]:
                err_avg_score += err_10.eval(data, elem)[0]
            feature_metrics_array[j-1]["ERR@10"] = err_avg_score/trials
            
            #Pfound
            pfound_10 = Pfound(cutoff=10)
            pfound_avg_score = 0
            for elem in preds[i-1][j-1]:
                pfound_avg_score += pfound_10.eval(data, elem)[0]
            feature_metrics_array[j-1]["Pfound@10"] = pfound_avg_score/trials
            
            ## For binary relevance
            #MAP
            map_10 = MAP(cutoff=10)
            map_avg_score = 0
            for elem in preds_binary[i-1][j-1]:
                map_avg_score += map_10.eval(data_binary, elem)[0]
            feature_metrics_array[j-1]["MAP@10"] = map_avg_score/trials
            
            #MRR
            mrr_10 = MRR(cutoff=10)
            mrr_avg_score = 0
            for elem in preds_binary[i-1][j-1]:
                mrr_avg_score += mrr_10.eval(data_binary, elem)[0]
            feature_metrics_array[j-1]["MRR@10"] = mrr_avg_score/trials
            
            ##Precision & Recall = F1
            prec_10 = Precision(cutoff=10)
            rec_10 = Recall(cutoff=10)
            temp_f1 = 0
            for elem in preds_binary[i-1][j-1]:
                prec_avg_score = prec_10.eval(data_binary, elem)[0]
                rec_avg_score = rec_10.eval(data_binary, elem)[0]
                temp_f1 += 2*((prec_avg_score*rec_avg_score)/(prec_avg_score+rec_avg_score))
            feature_metrics_array[j-1]["F1@10"] = temp_f1/trials
            
        f = open(output_path+dataset_name+"/scores/"+ranking+"_"+dataset_name+"_Fold"+str(i)+"_importance_scores.txt", "w")
        f.write(str(feature_metrics_array))
        f.close()
    
def generate_importance_scores_one_fold(f, preds, preds_binary, num_folds, fold_name, num_features, output_path, dataset_name, working_folder, ranking, trials):
    i = f
    for x in range(1):
        feature_metrics_array = [None]*num_features
        feature_metrics_array = [{"feature": i+1, "NDCG@10": 0, "DCG@10": 0, "ERR@10": 0, "Pfound@10": 0, "MAP@10": 0, "MRR@10": 0, "F1@10": 0} for i,value in enumerate(feature_metrics_array)]
        for j in range(1, num_features+1):
            data = Dataset.load(working_folder+dataset_name+"/"+fold_name+str(i)+"/test/"+str(j)+".txt")
            data_binary = Dataset.load(working_folder+dataset_name+"/binary/"+fold_name+str(i)+"/test/"+str(j)+".txt")
            
            #NDCG@10
            ndcg_10 = NDCG(cutoff=10, no_relevant_results=0.5, implementation='exp')
            ndcg_10_mean_score = 0
            for elem in preds[i-1][j-1]:
                ndcg_10_mean_score += ndcg_10.eval(data, elem)[0]
            feature_metrics_array[j-1]["NDCG@10"] = ndcg_10_mean_score/trials
            
            #DCG
            dcg_10 = DCG(cutoff=10)
            dcg_avg_score = 0
            for elem in preds[i-1][j-1]:
                dcg_avg_score += dcg_10.eval(data, elem)[0]
            feature_metrics_array[j-1]["DCG@10"] = dcg_avg_score/trials
            
            #ERR
            err_10 = ERR(cutoff=10)
            err_avg_score = 0
            for elem in preds[i-1][j-1]:
                err_avg_score += err_10.eval(data, elem)[0]
            feature_metrics_array[j-1]["ERR@10"] = err_avg_score/trials
            
            #Pfound
            pfound_10 = Pfound(cutoff=10)
            pfound_avg_score = 0
            for elem in preds[i-1][j-1]:
                pfound_avg_score += pfound_10.eval(data, elem)[0]
            feature_metrics_array[j-1]["Pfound@10"] = pfound_avg_score/trials
            
            ## For binary relevance
            #MAP
            map_10 = MAP(cutoff=10)
            map_avg_score = 0
            for elem in preds_binary[i-1][j-1]:
                map_avg_score += map_10.eval(data_binary, elem)[0]
            feature_metrics_array[j-1]["MAP@10"] = map_avg_score/trials
            
            #MRR
            mrr_10 = MRR(cutoff=10)
            mrr_avg_score = 0
            for elem in preds_binary[i-1][j-1]:
                mrr_avg_score += mrr_10.eval(data_binary, elem)[0]
            feature_metrics_array[j-1]["MRR@10"] = mrr_avg_score/trials
            
            ##Precision & Recall = F1
            prec_10 = Precision(cutoff=10)
            rec_10 = Recall(cutoff=10)
            temp_f1 = 0
            for elem in preds_binary[i-1][j-1]:
                prec_avg_score = prec_10.eval(data_binary, elem)[0]
                rec_avg_score = rec_10.eval(data_binary, elem)[0]
                temp_f1 += 2*((prec_avg_score*rec_avg_score)/(prec_avg_score+rec_avg_score))
            feature_metrics_array[j-1]["F1@10"] = temp_f1/trials
            
        f = open(output_path+dataset_name+"/scores/"+ranking+"_"+dataset_name+"_Fold"+str(i)+"_importance_scores.txt", "w")
        f.write(str(feature_metrics_array))
        f.close()