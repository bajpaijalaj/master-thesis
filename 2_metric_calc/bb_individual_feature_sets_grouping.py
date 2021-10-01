#References:
#https://github.com/dmlc/xgboost/blob/73b1bd27899e35ff57f5aca33f1685ea6db10f31/demo/rank/trans_data.py
def save_data(group_data,output_feature,output_group,dataset_name):
    if len(group_data) == 0:
        return

    output_group.write(str(len(group_data))+"\n")
    if dataset_name == "MQ2007":
        for data in group_data:
            # Fix for MQ2007 issue. Include zero values as well. Only size will increase.
            feats = [ p for p in data[2:] ]
            output_feature.write(data[0] + " " + " ".join(feats) + "\n")
    else:
        for data in group_data:
            # only include nonzero features
            feats = [ p for p in data[2:] if float(p.split(':')[1]) != 0.0 ]
            output_feature.write(data[0] + " " + " ".join(feats) + "\n")

def train_test_vali(arg1, arg2, arg3,dataset_name):
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
            save_data(group_data,output_feature,output_group,dataset_name)
            group_data = []
        group = splits[1]
        group_data.append(splits)

    save_data(group_data,output_feature,output_group,dataset_name)

    fi.close()
    output_feature.close()
    output_group.close()

	

def indi_feature_sets_grouping(folds = True, fold = "Fold1", features = 136, folder_names = ["train","vali","test"], input_dir = "../working_folder/MSLR-WEB10K/Fold1/", output_dir = "../working_folder/MSLR-WEB10K/Fold1/",dataset_name=""):
    if folds == True:
        for folder in folder_names:
            for k in range(1,features+1):
                train_test_vali(input_dir+folder+"/"+str(k)+".txt", output_dir+folder+"/"+str(k)+"."+folder, output_dir+folder+"/"+str(k)+"."+folder+'.group',dataset_name)