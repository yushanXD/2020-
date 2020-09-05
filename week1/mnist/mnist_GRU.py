import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
print(tf.__version__)


import numpy as np
print(np.__version__)

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.layers.recurrent import GRU
import datetime
from keras.callbacks import TensorBoard,ModelCheckpoint

K.set_image_data_format("channels_first") 


(x_train,y_train),(x_test,y_test) = mnist.load_data()
seed = 7
np.random.seed(seed)

x_train = x_train.reshape(-1,28,28)/255
x_test = x_test.reshape(-1,28,28)/255

y_train = np_utils.to_categorical(y_train,num_classes= 10)
y_test = np_utils.to_categorical(y_test,num_classes= 10)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


def GRU_():
    model  = Sequential()
    model.add(GRU(32,input_shape = (28,28)))
    model.add(Dense(10,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation='softmax'))
    
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model



model = GRU_()
print('model.summary:')
model.summary()



log_dir = "./mnist_GRU_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
filepath="GRUweights.best.hdf5"

tensorboard_cb = TensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(filepath,monitor='val_acc',verbose=1,save_best_only=True,mode='max',period=1)



import datetime
log_dir = "./logs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = TensorBoard(log_dir=log_dir,histogram_freq=1)

model.fit(x = x_train,y=y_train,epochs=50,validation_data=(x_test,y_test),callbacks=[checkpoint,tensorboard_cb])
scores = model.evaluate(x_test,y_test,verbose=0)
