import time
import pandas as pd
from scipy import stats

#print(data)
def extract_val(feat):
    return feat.split(':')[1]
    
def similarity_scores_using_data_column_loop(num_features, input_file_path, output_path, fold, dataset_name):
    data = pd.read_csv(input_file_path, header=None, sep=" ")    
    data[data.columns[2:num_features+2]] = data[data.columns[2:num_features+2]].applymap(extract_val)
    data = data.drop(0, axis = 1)
    data = data.drop(1, axis = 1)
    data = data.astype(float)
    
    scores = [pd.Series([None] * num_features) for elem in range(0,num_features)]
    #print(scores)
    print("Kendall starting ..")
    for i in range(1,num_features):
        scores[i-1][i-1] = 1
        for j in range(0,i):
            x = pd.Series(data.iloc[:,i])
            y = pd.Series(data.iloc[:,j])
            if x.equals(y):
                scores[i][j]= 1
                scores[j][i]=scores[i][j]
            else:
                scores[i][j]= stats.kendalltau(x, y, variant='c')[0] #x.corr(y, method='kendall')
                scores[j][i]=scores[i][j]
    scores[num_features-1][num_features-1] = 1
    df = pd.DataFrame(scores)
    df.to_csv(output_path+dataset_name+"/scores/"+"XGBOOST"+"_"+dataset_name+"_"+fold+"_kendall.csv", sep=',', header=False, index=False)
    #Duplicating for consistency of file names
    df.to_csv(output_path+dataset_name+"/scores/"+"LIGHTGBM"+"_"+dataset_name+"_"+fold+"_kendall.csv", sep=',', header=False, index=False)
    print("Kendall ends ..")

    print("Spearman starting ..")
    for i in range(1,num_features):
        scores[i-1][i-1] = 1
        for j in range(0,i):
            x = pd.Series(data.iloc[:,i])
            y = pd.Series(data.iloc[:,j])
            if x.equals(y):
                scores[i][j]= 1
                scores[j][i]=scores[i][j]
            else:
                scores[i][j]= x.corr(y, method='spearman')
                scores[j][i]=scores[i][j]
    df = pd.DataFrame(scores)
    df.to_csv(output_path+dataset_name+"/scores/"+"XGBOOST"+"_"+dataset_name+"_"+fold+"_spearman.csv", sep=',', header=False, index=False)
    #Duplicating for consistency of file names
    df.to_csv(output_path+dataset_name+"/scores/"+"LIGHTGBM"+"_"+dataset_name+"_"+fold+"_spearman.csv", sep=',', header=False, index=False)
