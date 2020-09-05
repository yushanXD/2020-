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
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def get_filename(dir):
    np_path=[]
    for filename in os.listdir(dir):
        np_path.append(filename)
    return np_path
"""
# 最大化处理
def get_data(filename):
    np_x_data=np.zeros((10000,665,256),dtype=np.float32)
    np_y_data=np.zeros((10000,665,80),dtype=np.float32)
    i=-1
    for name in filename:
        i=i+1
        data_x=np.load(x_dir+name)
        data_y=(np.load(y_dir+name))[:data_x.shape[0]]
        np_x_data[i]=np.pad(data_x,((0,665-data_x.shape[0]),(0,0)),'constant')
        np_y_data[i]=np.pad(data_y,((0,665-data_x.shape[0]),(0,0)),'constant')
    return np_x_data,np_y_data
"""
def get_data(filename):
    np_x_data=np.zeros((10000,340,256),dtype=np.float32)
    np_y_data=np.zeros((10000,340,80),dtype=np.float32)
    i=-1
    for name in filename:
        i=i+1
        data_x=np.load(x_dir+name)
        data_y=(np.load(y_dir+name))[:data_x.shape[0]]
        if(data_x.shape[0]<340):
            np_x_data[i]=np.pad(data_x,((0,340-data_x.shape[0]),(0,0)),'constant')
            np_y_data[i]=np.pad(data_y,((0,340-data_x.shape[0]),(0,0)),'constant')
        else:
            np_x_data[i]=data_x[0:340,:]
            np_y_data[i]=data_y[0:340,:]
    return np_x_data,np_y_data
def Highway(inputs):
    depth=128
    H=layers.Dense(units=depth,activation="relu")(inputs)
    T=layers.Dense(units=depth,activation="sigmoid",bias_constraint=k_init.Constant(-1.0))(inputs)
    return H*T+inputs*(1.0 - T)
def CBHG(K):
    input_data=layers.Input(shape=(None,256),dtype=np.float32)
    inputs=layers.Dense(units=128)(input_data)
    conv1dbank=layers.Conv1D(filters=128,kernel_size=1,padding='same',activation='relu')(inputs)
    for i in range (2,1+K):
        conv=layers.Conv1D(filters=128,kernel_size=i,padding='same',activation='relu')(inputs)
        conv1dbank=layers.Concatenate()([conv1dbank, conv])
    conv1dpro=layers.MaxPooling1D(pool_size=2,strides=1,padding='same')(conv1dbank)
    conv1dpro=layers.Conv1D(filters=128,kernel_size=3,padding='same',activation='relu')(conv1dpro)
    conv1dpro=layers.Conv1D(filters=128,kernel_size=3,padding='same',activation='linear')(conv1dpro)
    residual=layers.Add()([inputs, conv1dpro])
    highway_net=layers.Lambda(Highway)(residual)
    highway_net=layers.Lambda(Highway)(highway_net)
    highway_net=layers.Lambda(Highway)(highway_net)
    highway_net=layers.Lambda(Highway)(highway_net)
    """
    highway_net=Highway(residual)
    highway_net=Highway(highway_net)
    highway_net=Highway(highway_net)
    highway_net=Highway(highway_net)
    """
    gru_output=layers.Bidirectional(layers.GRU(128, return_sequences=True))(highway_net)
    output=layers.Dense(units=80)(gru_output)
    model = Model(inputs=input_data, outputs=[output])
    return model
"""
loss_object = keras.losses.SparseCategoricalCrossentropy(
    reduction='none'
)
def loss_func(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))  # 将y_true 中所有为0的找出来，标记为False
    loss_ = loss_object(y_true, y_pred)
    mask = tf.cast(mask, dtype=loss_.dtype)  # 将前面统计的是否零转换成1，0的矩阵
    loss_ *= mask     # 将正常计算的loss加上mask的权重，就剔除了padding 0的影响
    return tf.reduce_mean(loss_)    # 最后将loss求平均
"""

x_dir="/home/team06/week3_code/db1/ppgs/"
y_dir="/home/team06/week3_code/db1/mels/"
filename=get_filename(x_dir)
#读取数据
X_data,Y_data=get_data(filename)
print(X_data.shape,Y_data.shape)
#分配数据
X_train, X_test, Y_train, Y_test = train_test_split(X_data,Y_data,test_size=0.01)
print(X_train.shape,X_test.shape)
print(Y_train.shape,Y_test.shape)
print("data ready")
LR=0.001
# checkpoint_dir = '/home/team06/week3_code/WT/model/CBHGweights.best.h5'
checkpoint_dir = '/home/team06/week3_code/WT/model_average/CBHGweights.best.h5'
if os.path.exists(checkpoint_dir):
    print('checkpoint exists, Load weights from %s\n'%checkpoint_dir)
    #model = load_model('./model/CBHGweights.best.h5',custom_objects={"Highway":Highway,"layers":layers,"k_init":k_init})
    model = load_model('./model_average/CBHGweights.best.h5',custom_objects={"Highway":Highway,"layers":layers,"k_init":k_init})
else:
    print('No checkpoint found')
    model = CBHG(16)
model.compile(loss='mse',optimizer=optimizers.Adam(lr=LR,decay=1e-7), metrics=['accuracy'])
model.summary()
#保存模型
#filepath="./model/cbhgmodel_{epoch:02d}-{loss:.2f}.h5"
filepath="./model_average/cbhgmodel_{epoch:02d}-{loss:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1,save_best_only=False,mode='min',period=5)
# filebestmodel="./model/CBHGweights.best.h5"
filebestmodel="./model_average/CBHGweights.best.h5"
checkpoint2= ModelCheckpoint(filebestmodel, monitor='loss', verbose=1,save_best_only=True,mode='min',period=1)
#tensorboard
callbacks_list = [checkpoint,checkpoint2,TensorBoard(log_dir="./logs/CBHG_average")]
#训练
history=model.fit(X_train, Y_train, batch_size=32, epochs=1000, shuffle=True, verbose=1,validation_data=(X_test,Y_test),callbacks=callbacks_list)
print("Training finished \n")
