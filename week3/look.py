import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import pandas as pd

# def get_batch(input_data,output_data,batch_size):
#    # print(len(input_data))
#     batch_num = len(input_data)//batch_size
#     print(batch_num)
#     for k in range(batch_num):
#         begin = k*batch_size
#         end = begin + batch_size
#         input_batch = input_data[begin:end]
#         target_batch = output_data[begin:end]
#         yield input_batch,target_batch

# batch = get_batch(X_train,Y_train,64)
# input_batch, target_batch = next(batch)
# print(input_batch.shape)
# print(target_batch.shape)

def Conv1d(inputs,filters,size,padding='same',use_bias=False,activation = None,scope='conv1d'):
    # inputs: A 3D tensor with shape of[batch, time, depth].
    with tf.variable_scope(scope):
#         conv1d_output = tf.layers.conv1d(
#       inputs,
#       filters=channels,
#       kernel_size=kernel_size,
#       activation=activation,
#       padding='same')
#         params = {"inputs":inputs,"filters":filters,"kernel_size":size,"padding":padding,"activation":activation,"use_bias": use_bias}

        conv1d_output = tf.layers.conv1d(inputs = inputs,filters=filters,kernel_size=size,activation=activation,padding='same')

    return conv1d_output

def conv1d_banks(inputs, num_units = None,K=16,is_training=True,scope="conv1d_banks"):
    with tf.variable_scope(scope):
#         outputs = conv1d(inputs = inputs,filters = num_units,size = 1,activation = tf.nn.relu)
#         for k in range(2,K+1):
#             with tf.variable_scope("num_{}".format(k)):
#                 output = conv1d(inputs = inputs,filters = num_units,size = k,activation = tf.nn.relu, is_training = is_training)
#                 outputs = tf.concat((outputs,output),-1)
                
        conv_outputs = tf.concat([Conv1d(inputs=inputs,filters = 128,size = k, activation = tf.nn.relu,scope = 'conv1d_%d' % k) for k in range(1, K+1)],axis=-1)
#         outputs = normalize(outputs,is_training=is_training,activation_fn=tf.nn.relu)
        outputs = tf.layers.batch_normalization(conv_outputs,training=is_training)

    return outputs

def gru_bi(inputs,num_units=128,seqlen=None,scope='gru'):
    with tf.variable_scope(scope):
        cell = tf.contrib.rnn.GRUCell(num_units//2)
        cell_bw = tf.contrib.rnn.GRUCell(num_units//2)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell,cell_bw,inputs,sequence_length=seqlen,dtype=tf.float32)

        return tf.concat(outputs,2)

def highway(inputs,num_units,scope='highway'):
    with tf.variable_scope(scope):
        H = tf.layers.dense(inputs,units=num_units,activation=tf.nn.relu,name='H')
        T = tf.layers.dense(inputs,units=num_units,
                            activation=tf.nn.sigmoid,name='T',bias_initializer=tf.constant_initializer(-1.0))
        
        return H*T + inputs*(1.0)  

class cbhg():
    #inputs,is_training,scope
    def __init__(self,arg):
        tf.reset_default_graph()
        self.is_training = arg.is_training
        self.lr = arg.lr
        self.num_highway = arg.num_highway
        
        self.x = tf.placeholder(tf.float32,shape=(None,665, 256))
        self.y = tf.placeholder(tf.float32,shape=(None,665, 80))
        print(self.x.shape)
        #inputs shape:[Batch,T,256]
        
        inputs = tf.layers.dense(self.x,units=128,activation=None,use_bias=False,name='trans')

        convbanks_outputs = conv1d_banks(inputs,K=16,num_units=128,is_training=self.is_training)
        #output shape:[Batch,T,128*16]

        maxpool_outputs = tf.layers.max_pooling1d(convbanks_outputs,pool_size = 2,strides = 1,padding='same')
        #output shape:[Batch,T,128*16]

        c1 = Conv1d(inputs = maxpool_outputs,filters = 128,size = 3,activation=tf.nn.relu, scope='conv1d_1')
        #output shape:[Batch,T,128]

    #         x = normalize(x,is_training=self.is_training,activation_fn=None, scope="norm1")
        n1 = tf.layers.batch_normalization(c1,training = self.is_training)
        #output shape:[Batch,T,128]

        c2 = Conv1d(inputs = n1, filters = 128, size = 3, activation=None, scope='conv1d_2')
        #output shape:[Batch,T,128]

    #         x = normalize(x, is_training=self.is_training, activation_fn=None, scope="norm2")
        n2 = tf.layers.batch_normalization(c2,training = self.is_training)

        #output shape:[Batch,T,128]

        highway_inputs  = n2 + inputs
        #output shape:[Batch,T,128]

        for i in range(self.num_highway):
            highway_inputs = highway(highway_inputs,num_units=128,scope='highwaynet_{}'.format(i))
        #output shape:[Batch,T,128]

        bigru_outputs = gru_bi(inputs = highway_inputs,num_units = 128,scope='gru1')
        #output shape:[Batch,T,128]

        outputs = tf.layers.dense(bigru_outputs,units=80,activation=None,use_bias=False,name='O')
        #output shape:[Batch,T,80]
        
        if self.is_training:
            #self.loss = tf.square(tf.subtract(outputs, self.y, name=None), name=None)
#             print(self.loss.shape)
            
            self.mean_loss = tf.losses.mean_squared_error(self.y,outputs)
#            print(self.mean_loss.shape)
           # self.istarget = tf.to_float(tf.not_equal(self.y, tf.zeros_like(self.y))) # masking
#             print(self.istarget)
            
           # self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / 64.0
#             print(self.mean_loss.shape)
           

            self.global_step = tf.Variable(0,name='global_step',trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
            
            tf.summary.scalar('loss',self.mean_loss)
            self.merged = tf.summary.merge_all()

def create_hparams():
    params = tf.contrib.training.HParams(
        lr = 0.001,
        num_highway = 4,
        is_training = True)
    
    return params

import warnings
warnings.filterwarnings('ignore')


ppgs_file = os.listdir('../db1/ppgs')
mels_file = os.listdir('../db1/mels')
total_num = len(ppgs_file)

ppgs = np.zeros([total_num,665,256],dtype=np.float32)
mels = np.zeros([total_num,665,80],dtype=np.float32)


arg = create_hparams()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
epochs = 100
batch_size = 50
batch_num = 200
checkpoint_steps = 10

for i in range(total_num):
    p = np.load('../db1/ppgs/'+ppgs_file[i])
    m = (np.load('../db1/mels/'+mels_file[i]))[:p.shape[0]]
    ppgs[i] = np.pad(p,((0,665-p.shape[0]),(0,0)),'constant')
    mels[i] = np.pad(m,((0,665-p.shape[0]),(0,0)),'constant')

#from sklearn.model_selection import train_test_split
#X_train, X_test, Y_train, Y_test = train_test_split(ppgs, mels, test_size=0.1)
#X_train = ppgs
#Y_train = mels

#print(X_train.shape,Y_train.shape)
#print(X_test.shape,Y_test.shape)
# with tf.Graph().as_default():  
g = cbhg(arg)
ppgs_placeholder = tf.placeholder(ppgs.dtype,ppgs.shape)
mels_placeholder = tf.placeholder(mels.dtype,mels.shape)
dataset = tf.data.Dataset.from_tensor_slices((ppgs_placeholder,mels_placeholder))
dataset = dataset.shuffle(9000).batch(batch_size).repeat(epochs)
iterator = dataset.make_initializable_iterator()
data_element = iterator.get_next()

init = tf.global_variables_initializer()
# saver = tf.train.Saver()

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    
    sess.run(init)
    sess.run(iterator.initializer, feed_dict={ppgs_placeholder: ppgs,mels_placeholder: mels})
#     if os.path.exists('logtest/model.meta'):
#         saver.restore(sess, './logtest/model')
        
    writer = tf.summary.FileWriter('tensorboard/test', tf.get_default_graph())
    saver = tf.train.Saver()
    min_loss = 500
    cnt = 0
    for k in range(epochs):
        #print("epochs ",k)
        total_loss =  0
        
        for i in range(batch_num):
            #print("batch ",i)
            input_batch,target_batch = sess.run(data_element)
            feed = {g.x:input_batch,g.y:target_batch}
            cost,_ = sess.run([g.mean_loss,g.train_op],feed_dict=feed)
            print(g.mean_loss)
            total_loss += cost
            if (k * batch_num + i) % 10 == 0:
                rs = sess.run(merged, feed_dict=feed)
                writer.add_summary(rs, k * batch_num + i)
            #print("batch ",i,"loss:",cost)
        av_loss = total_loss/batch_num
        if (cnt + 1) % checkpoint_steps == 0:
            if av_loss < min_loss:
                min_loss = av_loss 
                saver.save(sess, './loging/model', global_step=cnt+1)
        print('epochs', k+1, ':average loss = ', av_loss)
        cnt+=1
    writer.close()
