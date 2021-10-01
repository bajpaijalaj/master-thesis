import pandas as pd
from bitarray import bitarray

def extract_individual_features(subset_list, folds, fold, dataset, input_path, filenames):
    if folds == True:
        data = pd.read_csv(input_path+filenames[0], header=None, sep=" ")
        d = data.iloc[:,subset_list]
        d.to_csv("../working_folder/"+dataset+"/evaluation/"+fold+"/train.txt", sep=' ', header=False, index=False)
        
        data = pd.read_csv(input_path+filenames[1], header=None, sep=" ")
        d = data.iloc[:,subset_list]
        d.to_csv("../working_folder/"+dataset+"/evaluation/"+fold+"/vali.txt", sep=' ', header=False, index=False)
        
        data = pd.read_csv(input_path+filenames[2], header=None, sep=" ")
        d = data.iloc[:,subset_list]
        d.to_csv("../working_folder/"+dataset+"/evaluation/"+fold+"/test.txt", sep=' ', header=False, index=False)