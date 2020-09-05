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
import datetime
from keras.callbacks import TensorBoard,ModelCheckpoint


K.set_image_data_format("channels_first") 


(x_train,y_train),(x_test,y_test) = mnist.load_data()

seed = 7
np.random.seed(seed)

# reshape to be [samples][pixels][width][height]
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')
 
x_train = x_train / 255
x_test = x_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


def baseline_model():
    model = Sequential()
    model.add(Dense(num_pixels, input_dim = num_pixels, kernel_initializer = 'normal',activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal',activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

model = baseline_model()
print('model.summary:')
model.summary()


log_dir = "./mnist_DNN_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
filepath="DNNweights.best.hdf5"

tensorboard_cb = TensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(filepath,monitor='val_acc',verbose=1,save_best_only=True,mode='max',period=1)


history = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,batch_size=100,callbacks=[checkpoint,tensorboard_cb],verbose=2)

score = model.evaluate(x_test,y_test,verbose=0)


print("Baseline Error: %.2f%%" % (100-score[1]*100))
