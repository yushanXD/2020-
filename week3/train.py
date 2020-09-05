import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import pandas as pd


def Conv1d(inputs,filters,size,padding='same',use_bias=False,activation = None,scope='conv1d'):
    # inputs: A 3D tensor with shape of[batch, time, depth].
    with tf.variable_scope(scope):
        conv1d_output = tf.layers.conv1d(inputs = inputs,filters=filters,kernel_size=size,activation=activation,padding='same')

    return tf.layers.batch_normalization(conv1d_output)

def conv1d_banks(inputs, num_units = None,K=16,is_training=True,scope="conv1d_banks"):
    with tf.variable_scope(scope):              
        conv_outputs = tf.concat([Conv1d(inputs=inputs,filters = 128,size = k, activation = tf.nn.relu,scope = 'conv1d_%d' % k) for k in range(1, K+1)],axis=-1)
#         outputs = normalize(outputs,is_training=is_training,activation_fn=tf.nn.relu)
#         outputs = tf.layers.batch_normalization(conv_outputs,training=is_training)

    return conv_outputs

def gru_bi(inputs,num_units=128,seqlen=None,scope='gru'):
    with tf.variable_scope(scope):
#         seqlen = inputs.shape[1]
        cell = tf.contrib.rnn.GRUCell(num_units)
        cell_bw = tf.contrib.rnn.GRUCell(num_units)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell,cell_bw,inputs,sequence_length=seqlen,dtype=tf.float32)

        return tf.concat(outputs,2)

def highway(inputs,num_units,scope='highway'):
    with tf.variable_scope(scope):
        H = tf.layers.dense(inputs,units=num_units,activation=tf.nn.relu,name='H')
        T = tf.layers.dense(inputs,units=num_units,
                            activation=tf.nn.sigmoid,name='T',bias_initializer=tf.constant_initializer(-1.0))
        
        return H*T + inputs*(1.0)  

def CBHG(arg,inputs,scope):
#     tf.reset_default_graph()
    with tf.variable_scope(scope):
        
        is_training = arg.is_training
        lr = arg.lr
        num_highway = arg.num_highway

#         x = tf.placeholder(tf.float32,shape=(None,None, 256))

        #inputs shape:[Batch,T,256]

#         inputs = tf.layers.dense(inputs_x,units=128,activation=None,use_bias=False,name='trans')

        convbanks_outputs = conv1d_banks(inputs,K=16,num_units=128,is_training=is_training)
        #output shape:[Batch,T,128*16]

        maxpool_outputs = tf.layers.max_pooling1d(convbanks_outputs,pool_size = 2,strides = 1,padding='same')
        #output shape:[Batch,T,128*16]

        c1 = Conv1d(inputs = maxpool_outputs,filters = 256,size = 3,activation=tf.nn.relu, scope='conv1d_18')
        #output shape:[Batch,T,128]

#         n1 = tf.layers.batch_normalization(c1,training = is_training)
        #output shape:[Batch,T,128]

        c2 = Conv1d(inputs = c1, filters = 256, size = 3, activation=None, scope='conv1d_19')
        #output shape:[Batch,T,128]


#         n2 = tf.layers.batch_normalization(c2,training = is_training)
        #output shape:[Batch,T,128]

        highway_inputs  = c2 + inputs
        #output shape:[Batch,T,128]

        for i in range(num_highway):
            highway_inputs = highway(highway_inputs,num_units=256,scope='highwaynet_{}'.format(i))
        #output shape:[Batch,T,128]
        
        bigru_outputs = gru_bi(inputs = highway_inputs,num_units = 128,scope='gru1')
        #output shape:[Batch,T,128]
        
        outputs = Conv1d(inputs = bigru_outputs,filters = 80,size = 3,activation=None, scope='conv1d_999')
#         outputs = tf.layers.dense(bigru_outputs,units=80,activation=None,use_bias=False,name='O')
        #output shape:[Batch,T,80]

        return outputs

def create_hparams():
    params = tf.contrib.training.HParams(
        lr = 0.001,
        num_highway = 4,
        is_training = True)
    
    return params

if __name__ =="__main__":

    import warnings
    warnings.filterwarnings('ignore')

    ppgs_file = os.listdir('../db1/ppgs')
    mels_file = os.listdir('../db1/mels')
    total_num = len(ppgs_file)
    print(total_num)

    # ppgs = np.zeros([total_num,665,256],dtype=np.float32)
    # mels = np.zeros([total_num,665,80],dtype=np.float32)

    # for i in range(total_num):
    #     p = np.load('../db1/ppgs/'+ppgs_file[i])
    #     m = (np.load('../db1/mels/'+mels_file[i]))[:p.shape[0]]
    #     ppgs[i] = np.pad(p,((0,665-p.shape[0]),(0,0)),'constant')
    #     mels[i] = np.pad(m,((0,665-p.shape[0]),(0,0)),'constant')


    arg = create_hparams()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    epochs = 100
    batch_size = 50
    batch_num = 200
    checkpoint_steps = 1

    # ppgs_placeholder = tf.placeholder(ppgs.dtype,ppgs.shape)
    # mels_placeholder = tf.placeholder(mels.dtype,mels.shape)

    # dataset = tf.data.Dataset.from_tensor_slices((ppgs_placeholder,mels_placeholder))
    # dataset = dataset.shuffle(10000).batch(batch_size).repeat(epochs)
    # iterator = dataset.make_initializable_iterator()
    # data_element = iterator.get_next()

    PPGs = tf.placeholder(tf.float32, [None, None, 256])
    print(PPGs.shape)
    Mels = tf.placeholder(tf.float32, [None, None, 80])
    global_step = tf.Variable(0)

    pre_mel = CBHG(arg,PPGs,"cbhg")
    #istarget = tf.to_float(tf.not_equal(Mels,tf.zeros_like(Mels)))
    loss = tf.losses.mean_squared_error(pre_mel,Mels)
    tf.summary.scalar('loss',loss)
    opt = tf.train.AdamOptimizer(arg.lr).minimize(loss, global_step=global_step)    #lr=0.02 loss=0.32 800 lr=0.01 loss=0.26 
    init = tf.global_variables_initializer()
    # saver = tf.train.Saver()

    with tf.Session() as sess:
        merged = tf.summary.merge_all()

        sess.run(init)
        writer = tf.summary.FileWriter('tensorboard/lm', tf.get_default_graph())
        saver = tf.train.Saver()
        min_loss = 500000
        #cnt = 0
        data_ppg = [[[]] for i in range(batch_size)]
        data_mel = [[[]] for i in range(batch_size)]
        ppg = None
        mel = None
        for k in range(epochs):
            #print("epochs ",k)
            total_loss =  0
            index = np.random.permutation(np.arange(10000))
            for i in range(batch_num):
                max_len = 0
                for q in range(batch_size):
                    ind = i*batch_size + q
    #                 print(ind)
                    p = np.load('../db1/ppgs/' + ppgs_file[index[ind]])
                    m = np.load('../db1/mels/' + mels_file[index[ind]])
                    max_len = p.shape[0] > max_len and p.shape[0] or max_len
                    data_ppg[q] = p
                    data_mel[q] = m

                ppg = np.array([np.pad(data_ppg[i][:max_len],((0,max_len-data_ppg[i][:max_len].shape[0]),(0,0)),'constant') for i in range(batch_size)], dtype=object)
                mel = np.array([np.pad(data_mel[i][:max_len],((0,max_len-data_mel[i][:max_len].shape[0]),(0,0)),'constant') for i in range(batch_size)], dtype=object)
#                 print(ppg.shape)
#                 sess.run(opt,{PPGs: ppg, Mels: mel})
                _, cost = sess.run([opt, loss],feed_dict={PPGs: ppg, Mels: mel})

                total_loss+=cost

                if (k * batch_num + i) % 10 == 0:
                    rs = sess.run(merged, feed_dict={PPGs: ppg, Mels: mel})
                    writer.add_summary(rs, k * batch_num + i)

                print("batch ",i,"loss:",cost)
            av_loss = total_loss/batch_num
            if (k + 1) % checkpoint_steps == 0:
                if av_loss < min_loss:
                    min_loss = av_loss 
                    saver.save(sess, './change/model.ckpt', global_step=k)
            print('epochs', k+1, ':average loss = ', av_loss)

        writer.close()
