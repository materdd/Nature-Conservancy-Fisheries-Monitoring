import os
import sys


def make_list():
    TRAIN_PATH = "/media/matterd/0C228E28228E173C/dataset/kaggle/fisheries_monitoring/train/"
    #VAL_PATH = "/media/matterd/0C228E28228E173C/dataset/kaggle/fisheries_monitoring/val/"
    OUT_PATH = "../data/"

    # for train data
    folders = os.listdir(TRAIN_PATH)
    #folders = [x + TRAIN_PATH for x in temp_folders]
    
    f_train = open("../data/train.txt", "w")
    f_val = open("../data/val.txt", "w")

    ratio_val = 0.15

    for folder in folders:
        if os.path.isdir(TRAIN_PATH + folder) == False:
            continue
        
        files = os.listdir(TRAIN_PATH + folder)
        for file in files:
            
        
        
    

    # for valid data

    # for test data

    f_train.close()
    f_val.close()

if __name__ == "__main__":
    make_list()