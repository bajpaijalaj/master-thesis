import pandas as pd
import os

def make_dir(path):
    try:
        os.mkdir(path)
    except OSError as error:
        print(path+" - already exists.")


## Code to remove additional columns at the end like docid, etc.
def removeLastColumnDocid(input_path, output_path, fold, data_list, filenamess):
    make_dir(output_path+fold)
    
    data = pd.read_csv(input_path+fold+'/'+filenamess[0], header=None, sep=" ")
    d = data.iloc[:,data_list]
    d.to_csv(output_path+fold+"/train.txt", sep=' ', header=False, index=False)
    
    data = pd.read_csv(input_path+fold+'/'+filenamess[1], header=None, sep=" ")
    d = data.iloc[:,data_list]
    d.to_csv(output_path+fold+"/vali.txt", sep=' ', header=False, index=False)
    
    data = pd.read_csv(input_path+fold+'/'+filenamess[2], header=None, sep=" ")
    d = data.iloc[:,data_list]
    d.to_csv(output_path+fold+"/test.txt", sep=' ', header=False, index=False)
    
## Code to remove additional columns at the end like docid, etc. in ALL file if available
def removeLastColumnDocidInALL(input_path, data_list, output_path, filenamess, dataset_name):
    # To remove in the single file in ALL folder 
    data = pd.read_csv(input_path+'ALL/'+dataset_name+'.txt', header=None, sep=" ")
    d = data.iloc[:,data_list]
    d.to_csv(output_path+"/"+dataset_name+".txt", sep=' ', header=False, index=False)

