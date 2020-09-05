import numpy as np
import scipy as sp
import matplotlib.mlab as mlab
import keras.initializers as k_init
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn import metrics 
import tensorflow as tf
import keras
from keras.models import Model
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras import layers,regularizers,models,backend,utils,optimizers
import os
from keras.models import model_from_json,load_model
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def Highway(inputs):
    depth=256
    H=layers.Dense(units=depth,activation="relu")(inputs)
    T=layers.Dense(units=depth,activation="sigmoid",bias_constraint=k_init.Constant(-1.0))(inputs)
    return H*T+inputs*(1.0 - T)
def get_filename(dir):
    np_path=[]
    for filename in os.listdir(dir):
        np_path.append(filename)
    return np_path
def get_data(filename):
    data=[]
    for name in filename:
        data_x=np.load(X_dir+name)
        data.append(data_x)
    return data
model = load_model('./model_final_256_con/CBHGweights.best.h5',custom_objects={"Highway":Highway,"layers":layers,"k_init":k_init})

X_dir="/home/team06/week3_code/WT/testppgs/"
Y_dir="/home/team06/week3_code/WT/testmels/"

"""
X_dir="/home/linux/Desktop/summer/CBHG/testppgs/"
Y_dir="/home/linux/Desktop/summer/CBHG/testmels/"
"""
filesname=get_filename(X_dir)
X_data=get_data(filesname)
print(len(X_data))
i=-1
for name in filesname:
    i=i+1
    data_y=model.predict(X_data[i].reshape((1,X_data[i].shape[0],X_data[i].shape[1])))
    np.save(Y_dir+name,data_y)
print("ready")
