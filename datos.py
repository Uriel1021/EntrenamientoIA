import csv
import os
import numpy as np

batch_size=64

label = {
    "0": [1,0,0,0,0,0,0,0,0,0],
    "1": [0,1,0,0,0,0,0,0,0,0],
    "2": [0,0,1,0,0,0,0,0,0,0],
    "3": [0,0,0,1,0,0,0,0,0,0],
    "4": [0,0,0,0,1,0,0,0,0,0],
    "5": [0,0,0,0,0,1,0,0,0,0],
    "6": [0,0,0,0,0,0,1,0,0,0],
    "7": [0,0,0,0,0,0,0,1,0,0],
    "8": [0,0,0,0,0,0,0,0,1,0],
    "9": [0,0,0,0,0,0,0,0,0,1]
}



def get_data(archivo):
    file = open(archivo,"r")
    data = csv.reader(file)
    next(data)
    label_train=[]
    img_train=[]
    for data in data:
        label_train.append(label[data[0]])
        img_vect = np.array(data[1:], dtype = "int64")
        img_train.append(img_vect)
    
    return label_train, img_train


def next_batch(a,pos,label_train,img_train):
    ini=batch_size*a
    fin=batch_size*a+batch_size
    l=pos[ini:fin]
    label_batch=[]
    img_batch=[]
    for i in l:
        label_batch.append(label_train[i])
        img_batch.append(img_train[i])
    return label_batch,img_batch


def next_batch_test(a,label_train,img_train):
    ini=batch_size*a
    fin=batch_size*a+batch_size
    return label_train[ini:fin],img_train[ini:fin]
