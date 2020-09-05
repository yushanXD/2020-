from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt 
import datetime
import hvplot.pandas

import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
print(tf.__version__)

tf.test.is_gpu_available()

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.callbacks import TensorBoard,ModelCheckpoint
   

# Load Dataset 
mfccData = pd.read_csv("Frogs_MFCCs.csv")

# Get labels for classification
Family_class = np.unique(mfccData.values[:, 22]).tolist()
Genus_class = np.unique(mfccData.values[:, 23]).tolist()
Species_class = np.unique(mfccData.values[:, 24]).tolist()

total_num = len(mfccData.values)
print(total_num)
Family_labels = np.zeros(total_num, dtype=np.int32)
Genus_labels = np.zeros(total_num, dtype=np.int32)
Species_labels = np.zeros(total_num, dtype=np.int32)

for i in range(total_num):
    Family_labels[i] = Family_class.index(mfccData.values[:, 22][i])
    Genus_labels[i] = Genus_class.index(mfccData.values[:, 23][i])
    Species_labels[i] = Species_class.index(mfccData.values[:, 24][i])

X = mfccData.values[:,0:22]
print(X.shape)
Y = Species_labels

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
Y_train = np_utils.to_categorical(Y_train, len(Species_class))
Y_test = np_utils.to_categorical(Y_test, len(Species_class))

def DNN_2layers():
    model_2 = Sequential()
    model_2.add(Dense(512,activation='relu',input_dim = 22,use_bias = True,kernel_initializer = 'normal'))
    model_2.add(Dense(512,activation='relu',use_bias = True, kernel_initializer='normal'))
    model_2.add(Dense(len(Species_class),kernel_initializer='normal',activation='softmax'))
    model_2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model_2


model_2 = DNN_2layers()
print('model.summary:')
model_2.summary()



log_dir = "./frog_DNN_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
filepath="DNNweights.best.hdf5"

tensorboard_cb = TensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(filepath,monitor='val_acc',verbose=1,save_best_only=True,mode='max',period=1)


model_2.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=100,batch_size=100,callbacks=[checkpoint,tensorboard_cb],verbose=2)

scores = model_2.evaluate(X_test,Y_test,verbose=0)

print("baseline error: %.2f%%"%(100-scores[1]*100))
