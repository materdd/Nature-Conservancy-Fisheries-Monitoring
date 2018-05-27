import os
import sys
from tqdm import tqdm
import numpy as np

fish_label ={"ALB":0, "BET":1, "DOL":2, "LAG":3,"SHARK":4, "YFT":5, "NoF":6, "OTHER":7}

def split_train_val():
    np.random.seed(100)

    ALL_PATH = "../data/all.txt"
    TRAIN_PATH = "../data/train.txt"
    VAL_PATH = "../data/val.txt"
    
    with open(ALL_PATH, "r") as f:
        all_data = f.readlines()
    
    f_train = open(TRAIN_PATH, "w")
    f_val = open(VAL_PATH, "w")

    n_data = len(all_data)
    print("Number of data: {}".format(n_data))    

    ratio_val = 0.15
    n_val = int(n_data * ratio_val)
    n_train = n_data - n_val
    
    assert n_train + n_val == n_data

    print("Number of train data: {}".format(n_train))
    print("NUmber of validation data: {}".format(n_val))

    np.random.shuffle(all_data)

    for i, line in enumerate(all_data):
        if i < n_train:
            f_train.write(line)
        else:
            f_val.write(line)
            
    f_train.close()
    f_val.close()   
        

def make_list():
    TRAIN_PATH = "/media/matterd/0C228E28228E173C/dataset/kaggle/fisheries_monitoring/train/train/"
    #VAL_PATH = "/media/matterd/0C228E28228E173C/dataset/kaggle/fisheries_monitoring/val/"
    OUT_PATH = "../data/all.txt"

    # for all data
    folders = os.listdir(TRAIN_PATH)
    print(folders)
    #folders = [x + TRAIN_PATH for x in temp_folders]
    
    f = open(OUT_PATH, "w")


    for folder in tqdm(folders):
        if os.path.isdir(TRAIN_PATH + folder) == False:
            continue
        
        files = os.listdir(TRAIN_PATH + folder)
        for file in files:
            f.write(folder + "/" + file)
            f.write(" ")
            f.write(str(fish_label[folder]))
            f.write("\n")

    f.close()
    
if __name__ == "__main__":
    make_list()
    split_train_val()