from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import scipy as sp

#import matplotlib.mlab as mlab
#import matplotlib.pyplot as plt 

import hvplot.pandas

from keras.models import Sequential
from keras.models import Model

from keras.layers import Dense
from keras.layers import Dropout

from keras.layers.recurrent import GRU

from keras.utils import np_utils

from keras.callbacks import TensorBoard,ModelCheckpoint

from keras import backend as K

import datetime

# Load Dataset 
mfccData = pd.read_csv("Frogs_MFCCs.csv")


# Get labels for classification
Family_class = np.unique(mfccData.values[:, 22]).tolist()
Genus_class = np.unique(mfccData.values[:, 23]).tolist()
Species_class = np.unique(mfccData.values[:, 24]).tolist()

print(len(Family_class))
print(len(Genus_class))
print(len(Species_class))

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


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

print(Y_train)


# X_train = X_train[:,:,None]
# X_test = X_test[:,:,None]

X_train = X_train.reshape(-1,22,1)
X_test = X_test.reshape(-1,22,1)
print(X_train.shape)
print(X_test.shape)



def GRU_():
    model = Sequential()
    model.add(GRU(32,input_shape = (22,1)))
    #model.add(Dense(128,activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation='softmax'))
    
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model


model2 = GRU_()
print('model.summary:')
model2.summary()
log_dir = "./frog_GRU_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
filepath="GRUweights.best.hdf5"

tensorboard_cb = TensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(filepath,monitor='val_acc',verbose=1,save_best_only=True,mode='max',period=1)


model2.fit(x=X_train,y=Y_train,epochs=100,validation_data=(X_test,Y_test),callbacks=[checkpoint,tensorboard_cb],verbose=2)


scores = model2.evaluate(X_test,Y_test,verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
