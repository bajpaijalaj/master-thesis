import time
import prepare_datasets as aa

from datetime import datetime
import configparser
import sys
import shutil
import os

def make_dir(path):
    try:
        os.mkdir(path)
    except OSError as error:
        print(path+" - already exists.")

config = configparser.ConfigParser()
config.read('../config_'+sys.argv[1]+'.ini')

for section in config.sections():
    dataset_name = config[section]['dataset_name']
    input_path = config[section]['input_path']
    output_path = config[section]['output_path']
    num_folds = int(config[section]['num_folds'])
    data_list = config[section]['data_list'].split(",")
    data_list = list(map(int, data_list))
    filenamess = config[section]['filenamess'].split(",")
    max_label = int(config[section]['max_label'])
    cutoff_label = int(config[section]['cutoff_label'])
    binary_output_path = config[section]['binary_output_path']
    output_filenames = config[section]['output_filenames'].split(",")

    if dataset_name=="OHSUMED-MIN" or dataset_name=="OHSUMED-QLN":
        for i in range(1,num_folds+1):
            aa.removeLastColumnDocid(input_path, output_path, "Fold"+str(i), data_list, filenamess)
    elif dataset_name=="MQ2007":
        for i in range(1,num_folds+1):
            aa.removeLastColumnDocid(input_path, output_path, "Fold"+str(i), data_list, filenamess)
    elif dataset_name=="MQ2008":
        for i in range(1,num_folds+1):
            aa.removeLastColumnDocid(input_path, output_path, "Fold"+str(i), data_list, filenamess)
    elif dataset_name=="TD2003":
        for i in range(1,num_folds+1):
            aa.removeLastColumnDocid(input_path, output_path, "Fold"+str(i), data_list, filenamess)
    elif dataset_name=="TD2004":
        for i in range(1,num_folds+1):
            aa.removeLastColumnDocid(input_path, output_path, "Fold"+str(i), data_list, filenamess)
    elif dataset_name=="MSLR-WEB10K":
        for i in range(1,num_folds+1):
            make_dir(output_path+"Fold"+str(i))
            for f_in,f_out in zip(filenamess,output_filenames):
                shutil.copyfile(input_path+"Fold"+str(i)+"/"+f_in, output_path+"Fold"+str(i)+"/"+f_out)