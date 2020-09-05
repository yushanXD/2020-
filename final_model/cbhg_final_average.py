import numpy as np
import scipy as sp
import matplotlib.mlab as mlab
import keras.initializers as k_init
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn import metrics 
import tensorflow as tf
import keras
from keras.models import Model,model_from_json,load_model
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras import layers,regularizers,models,backend,utils,optimizers
import keras.backend.tensorflow_backend as KTF
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#### 显存按需占用
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)

##参数
LR=0.0003
Batch_size=32
savadirname='model_final_average'
startepoch=180
def load_test_generator(X_test,Y_test):
    max_len=0
    for i in range(len(X_test)):
        if(X_test[i].shape[0]>max_len):
            max_len= X_test[i].shape[0]
    np_x_data=np.zeros((len(X_test),max_len,256),dtype=np.float32)
    np_y_data=np.zeros((len(X_test),max_len,80),dtype=np.float32)
    for i in range(len(X_test)):
        np_x_data[i]=np.pad(X_test[i],((0,max_len-X_test[i].shape[0]),(0,0)),'constant')
        np_y_data[i]=np.pad(Y_test[i],((0,max_len-X_test[i].shape[0]),(0,0)),'constant')
    return np_x_data,np_y_data


def load_input_generator(batch_size,X_train,Y_train):
    train_length = len(X_train)
    step = train_length//batch_size + 1
    while True:
        for i in range(step):
            begin = i * batch_size
            end = (i+1)*batch_size
            if end > train_length:
                end = train_length
            total_len=0
            num=0
            for j in range (begin,end):
                num=num+1
                total_len=total_len+X_train[j].shape[0]
            batch_len=total_len//num
            np_x_data=np.zeros((batch_size,batch_len,256),dtype=np.float32)
            np_y_data=np.zeros((batch_size,batch_len,80),dtype=np.float32)
            k=-1
            for j in range (begin,end):
                k=k+1
                if(X_train[j].shape[0]<batch_len):
                    np_x_data[k]=np.pad(X_train[j],((0,batch_len-X_train[j].shape[0]),(0,0)),'constant')
                    np_y_data[k]=np.pad(Y_train[j],((0,batch_len-X_train[j].shape[0]),(0,0)),'constant')
                else:
                    np_x_data[k]=X_train[j][0:batch_len,:]
                    np_y_data[k]=Y_train[j][0:batch_len,:]
            yield ({'input_1':np_x_data},{'output':np_y_data})
def get_filename(dir):
    np_path=[]
    for filename in os.listdir(dir):
        np_path.append(filename)
    return np_path
def get_data(filename):
    np_x_data=[]
    np_y_data=[]
    for name in filename:
        data_x=np.load(x_dir+name)
        data_y=(np.load(y_dir+name))[:data_x.shape[0]]
        np_x_data.append(data_x)
        np_y_data.append(data_y)
    return np_x_data,np_y_data
def Highway(inputs):
    depth=256
    H=layers.Dense(units=depth,activation="relu")(inputs)
    T=layers.Dense(units=depth,activation="sigmoid",bias_constraint=k_init.Constant(-1.0))(inputs)
    return H*T+inputs*(1.0 - T)
def CBHG(K):
    input_data=layers.Input(shape=(None,256),dtype=np.float32)
#     inputs=layers.Dense(units=128)(input_data)
    
    conv1dbank=layers.Conv1D(filters=128,kernel_size=1,padding='same',activation='relu')(input_data)
    conv1dbank=layers.normalization.BatchNormalization()(conv1dbank)
    for i in range (2,1+K):
        conv=layers.Conv1D(filters=128,kernel_size=i,padding='same',activation='relu')(input_data)
        conv=layers.normalization.BatchNormalization()(conv)
        conv1dbank=layers.Concatenate()([conv1dbank, conv])
    conv1dpro=layers.MaxPooling1D(pool_size=2,strides=1,padding='same')(conv1dbank)
    conv1dpro=layers.Conv1D(filters=256,kernel_size=3,padding='same',activation='relu')(conv1dpro)
    conv1dpro=layers.normalization.BatchNormalization()(conv1dpro)
    conv1dpro=layers.Conv1D(filters=256,kernel_size=3,padding='same',activation='linear')(conv1dpro)
    conv1dpro=layers.normalization.BatchNormalization()(conv1dpro)
    residual=layers.Add()([input_data, conv1dpro])
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
    output=layers.Dense(units=80,name="output")(gru_output)
    model = Model(inputs=input_data, outputs=[output])
    return model

x_dir="/home/team06/week3_code/db1/ppgs/"
y_dir="/home/team06/week3_code/db1/mels/"
filename=get_filename(x_dir)
#读取数据
X_data,Y_data=get_data(filename)
#分配数据
X_train, X_test_t, Y_train, Y_test_t = train_test_split(X_data,Y_data,test_size=0.01)
X_test,Y_test=load_test_generator(X_test_t,Y_test_t)
print("data ready")
# checkpoint_dir = '/home/team06/week3_code/WT/model/CBHGweights.best.h5'
checkpoint_dir = '/home/team06/week3_code/WT/'+savadirname+'/CBHGweights.best.h5'
if os.path.exists(checkpoint_dir):
    print('checkpoint exists, Load weights from %s\n'%checkpoint_dir)
    #model = load_model('./model/CBHGweights.best.h5',custom_objects={"Highway":Highway,"layers":layers,"k_init":k_init})
    model = load_model('./'+savadirname+'/CBHGweights.best.h5',custom_objects={"Highway":Highway,"layers":layers,"k_init":k_init})
else:
    print('No checkpoint found')
    model = CBHG(16)
model.compile(loss='mse',optimizer=optimizers.Adam(lr=LR,decay=1e-7), metrics=['accuracy'])
model.summary()
#保存模型
#filepath="./model/cbhgmodel_{epoch:02d}-{loss:.2f}.h5"
filepath="./"+savadirname+"/cbhgmodel_{epoch:02d}-{loss:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1,save_best_only=False,mode='min',period=5)
# filebestmodel="./model/CBHGweights.best.h5"
filebestmodel="./"+savadirname+"/CBHGweights.best.h5"
checkpoint2= ModelCheckpoint(filebestmodel, monitor='loss', verbose=1,save_best_only=True,mode='min',period=1)
#tensorboard
callbacks_list = [checkpoint,checkpoint2,TensorBoard(log_dir="./logs/"+savadirname)]
#训练
history=model.fit_generator(load_input_generator(Batch_size,X_train,Y_train),steps_per_epoch=len(X_train)//Batch_size,epochs=1000,verbose=1,callbacks=callbacks_list,validation_data=(X_test,Y_test),initial_epoch=startepoch)
print("Training finished \n")
