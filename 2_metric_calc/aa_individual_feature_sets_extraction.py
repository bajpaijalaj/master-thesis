import pandas as pd

def extract_individual_features(dy_list, folds, fold, dataset, features, path, filenames):
    if folds == True:
        data = pd.read_csv(path+filenames[0], header=None, sep=" ")
        for i in range(0,features):
            dy_list.append(i+2)
            d = data.iloc[:,dy_list]
            d.to_csv("../working_folder/"+dataset+"/"+fold+"/train/"+str(i+1)+".txt", sep=' ', header=False, index=False)
            dy_list.pop()
    
        data = pd.read_csv(path+filenames[1], header=None, sep=" ")
        for i in range(0,features):
            dy_list.append(i+2)
            d = data.iloc[:,dy_list]
            d.to_csv("../working_folder/"+dataset+"/"+fold+"/vali/"+str(i+1)+".txt", sep=' ', header=False, index=False)
            dy_list.pop()
        
        data = pd.read_csv(path+filenames[2], header=None, sep=" ")
        for i in range(0,features):
            dy_list.append(i+2)
            d = data.iloc[:,dy_list]
            d.to_csv("../working_folder/"+dataset+"/"+fold+"/test/"+str(i+1)+".txt", sep=' ', header=False, index=False)
            dy_list.pop()

    else:
        print("Implement for the other case")


def extract_individual_features_for_binary(dy_list, folds, fold, dataset, features, path, filenames):
    if folds == True:
        data = pd.read_csv(path+filenames[0], header=None, sep=" ")
        for i in range(0,features):
            dy_list.append(i+2)
            d = data.iloc[:,dy_list]
            d.to_csv("../working_folder/"+dataset+"/binary/"+fold+"/train/"+str(i+1)+".txt", sep=' ', header=False, index=False)
            dy_list.pop()
    
        data = pd.read_csv(path+filenames[1], header=None, sep=" ")
        for i in range(0,features):
            dy_list.append(i+2)
            d = data.iloc[:,dy_list]
            d.to_csv("../working_folder/"+dataset+"/binary/"+fold+"/vali/"+str(i+1)+".txt", sep=' ', header=False, index=False)
            dy_list.pop()
        
        data = pd.read_csv(path+filenames[2], header=None, sep=" ")
        for i in range(0,features):
            dy_list.append(i+2)
            d = data.iloc[:,dy_list]
            d.to_csv("../working_folder/"+dataset+"/binary/"+fold+"/test/"+str(i+1)+".txt", sep=' ', header=False, index=False)
            dy_list.pop()

    else:
        print("Implement for the other case")
