import pandas as pd
import os
from rankeval.dataset import Dataset
import lightgbm
from rankeval.metrics import NDCG,MAP
from sklearn.datasets import load_svmlight_file
from rankeval.model import RTEnsemble
import numpy as np


def make_dir(path):
    try:
        os.mkdir(path)
    except OSError as error:
        print(path+" - already exists.")

def generate_dataset_binary_relevance_labels(max_label, cutoff_label, dataset_name, output_path, num_folds, filenamess, input_path):#cutoff label is greater than equal to
    make_dir("../0_dataset/binary")
    make_dir("../0_dataset/binary/"+dataset_name)
    if max_label != 1:    
        old_labels = list(range(0, max_label+1)) #[0,1,2]
        new_labels = []
        for j in range(1,cutoff_label+1):
            new_labels.append(0)
        for k in range(cutoff_label,max_label+1):
            new_labels.append(1)
        
        print(old_labels)
        print(new_labels)
        
        for j in range(1,num_folds+1):
            make_dir(output_path+"Fold"+str(j))
            
            data = pd.read_csv(input_path+"Fold"+str(j)+'/'+filenamess[0], header=None, sep=" ")
            data[0] = data[0].replace(old_labels, new_labels)
            data.to_csv(output_path+"Fold"+str(j)+"/train.txt", sep=' ', header=False, index=False)
            
            data = pd.read_csv(input_path+"Fold"+str(j)+'/'+filenamess[1], header=None, sep=" ")
            data[0] = data[0].replace(old_labels, new_labels)
            data.to_csv(output_path+"Fold"+str(j)+"/vali.txt", sep=' ', header=False, index=False)
            
            data = pd.read_csv(input_path+"Fold"+str(j)+'/'+filenamess[2], header=None, sep=" ")
            data[0] = data[0].replace(old_labels, new_labels)
            data.to_csv(output_path+"Fold"+str(j)+"/test.txt", sep=' ', header=False, index=False)
        
        print("DONE: "+str(new_labels))
    elif max_label == 1:
        for j in range(1,num_folds+1):
            make_dir(output_path+"Fold"+str(j))
            
            data = pd.read_csv(input_path+"Fold"+str(j)+'/'+filenamess[0], header=None, sep=" ")
            data.to_csv(output_path+"Fold"+str(j)+"/train.txt", sep=' ', header=False, index=False)
            
            data = pd.read_csv(input_path+"Fold"+str(j)+'/'+filenamess[1], header=None, sep=" ")
            data.to_csv(output_path+"Fold"+str(j)+"/vali.txt", sep=' ', header=False, index=False)
            
            data = pd.read_csv(input_path+"Fold"+str(j)+'/'+filenamess[2], header=None, sep=" ")
            data.to_csv(output_path+"Fold"+str(j)+"/test.txt", sep=' ', header=False, index=False)
        print("DONE: The dataset already had binary labels.")