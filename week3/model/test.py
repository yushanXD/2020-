import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('model.ckpt-1.meta')
    new_saver.restore(sess,tf.train.latest_checkpoint("./"))

    graph = tf.get_default_graph()
#     X = graph.get_tensor_by_name("inputs/Placeholder:0")
#     Y = graph.get_tensor_by_name("inputs/Placeholder:1")
    X = graph.get_operation_by_name('Placeholder_2').outputs[0]
    print(type(X))
    Y = graph.get_operation_by_name('Placeholder_3').outputs[0]
    pre = graph.get_operation_by_name('mul').inputs[0]

    ppgs_file = os.listdir('/home/team06/week3_code/db1/ppgs')
    mels_file = os.listdir('/home/team06/week3_code/db1/mels')
    total_num = len(ppgs_file)

    mels = np.zeros([total_num,665,80],dtype=np.float32)
    ppgs = np.zeros([total_num,665,256],dtype=np.float32)

    for i in range(total_num):
        p = np.load('/home/team06/week3_code/db1/ppgs/'+ppgs_file[i])
        m = (np.load('/home/team06/week3_code/db1/mels/'+mels_file[i]))[:p.shape[0]]
        if p.shape[0] > 665:
            ppgs[i] = p[:665]
            mels[i] = m[:665]
        else:
            ppgs[i] = np.pad(p,((0,665-p.shape[0]),(0,0)),'constant')
            mels[i] = np.pad(m,((0,665-p.shape[0]),(0,0)),'constant')
#     print(type(ppgs))
#     ppgs = tf.convert_to_tensor(ppgs, tf.float32)
#     mels = tf.convert_to_tensor(mels, tf.float32)
    pre_ = sess.run(pre, feed_dict={X:ppgs,Y:mels})
    path = "/home/team06/week3_code/ZYH/test/output_mel/"
    for e in range(len(pre_)):
        t = path + ppgs_file[k]
        np.save(t, pre_[k])

    
  