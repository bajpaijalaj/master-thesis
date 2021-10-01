import time
import generate_datasets as cc
from datetime import datetime
import configparser
import sys

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
    # NEED TO PASS THE HYPER PARAMETERS
    if dataset_name=="OHSUMED-MIN" or dataset_name=="OHSUMED-QLN":
        cc.generate_dataset_binary_relevance_labels(max_label,cutoff_label, dataset_name, binary_output_path, num_folds, output_filenames, output_path) #the output_path is the input_path for cc
    elif dataset_name=="MQ2007":
        cc.generate_dataset_binary_relevance_labels(max_label,cutoff_label, dataset_name, binary_output_path, num_folds, output_filenames, output_path) #the output_path is the input_path for cc
    elif dataset_name=="MQ2008":
        cc.generate_dataset_binary_relevance_labels(max_label,cutoff_label, dataset_name, binary_output_path, num_folds, output_filenames, output_path) #the output_path is the input_path for cc
    elif dataset_name=="TD2003":
        cc.generate_dataset_binary_relevance_labels(max_label,cutoff_label, dataset_name, binary_output_path, num_folds, output_filenames, output_path) #the output_path is the input_path for cc
    elif dataset_name=="TD2004":
        cc.generate_dataset_binary_relevance_labels(max_label,cutoff_label, dataset_name, binary_output_path, num_folds, output_filenames, output_path) #the output_path is the input_path for cc
    elif dataset_name=="MSLR-WEB10K" or dataset_name=="YAHOO-SET1" or dataset_name=="YAHOO-SET2" :
        cc.generate_dataset_binary_relevance_labels(max_label,cutoff_label, dataset_name, binary_output_path, num_folds, output_filenames, output_path) #the output_path is the input_path for cc