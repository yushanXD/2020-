import tensorflow as tf
import numpy as np
from train import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

PPGs = tf.placeholder(tf.float32, [None, None, 256])

ppg_file = 'xinwenzhibojian.npy'
arg = create_hparams()

ppg = np.load(ppg_file)
ppg = np.reshape(ppg,[1,ppg.shape[0],256])

pre_mel = CBHG(arg,PPGs,"cbhg")


#saver = tf.train.Saver()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    new_saver = tf.train.import_meta_graph('model.ckpt-0.meta')
    new_saver.restore(sess,tf.train.latest_checkpoint("./"))
    
    graph = tf.get_default_graph()
    c_mel = sess.run(pre_mel,feed_dict = {PPGs: ppg})
    res = np.array((c_mel.data),dtype=np.float32)
    res = np.reshape(res, (ppg.shape[1],80))
    np.save("trans_"+ppg_file, res)
    
