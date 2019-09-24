import tensorflow as tf
import numpy as np
import itertools
from operator import itemgetter
from datetime import datetime
import sys
from matplotlib import pyplot as plt

from img_utils import load_bin_raw, load_labels_onehot
from mobilenet import MobileNet


imgWidth = 224
imgHeight = 224
numChannels = 3
numClasses = 1000

data_eval_slice = 20 

bin_dir = '/tank/local/ruiliu/dataset/imagenet1k.bin'
#bin_dir = '/home/ruiliu/Development/mtml-tf/dataset/imagenet1k.bin'
label_path = '/tank/local/ruiliu/dataset/imagenet1k-label.txt'
#label_path = '/home/ruiliu/Development/mtml-tf/dataset/imagenet1k-label.txt'

def get_params(n_conf):
    batch_size = np.arange(10,61,5)
    opt_conf = ['Adam','SGD','Adagrad','Momentum']
    #data_conf = ['Same','Diff']
    #preprocess_list = ['Include','Not Include']
    #all_conf = [batch_size,opt_conf,data_conf,preprocess_list]
    all_conf = [batch_size,opt_conf]

    hp_conf = list(itertools.product(*all_conf))
    idx_list = np.random.choice(np.arange(0, len(hp_conf)), n_conf, replace=False)
    rand_conf = itemgetter(*idx_list)(hp_conf)

    return rand_conf

def run_params_pack(batch_size, opt, iterations, conn):
    features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
    labels = tf.placeholder(tf.int64, [None, numClasses])

    X_data = load_bin_raw(bin_dir, numChannels, imgWidth, imgHeight)
    Y_data = load_labels_onehot(label_path, numClasses)

    X_data_eval = X_data[0:data_eval_slice,:,:,:]
    Y_data_eval = Y_data[0:data_eval_slice,:]

    if len(opt) == 1:
        dt = datetime.now()
        np.random.seed(dt.microsecond)
        net_instnace = np.random.randint(sys.maxsize)
        modelEntity = MobileNet("mobilenet_"+str(net_instnace), 1, imgHeight, imgWidth, batch_size, numClasses, opt[0])
        modelLogit = modelEntity.build(features)
        trainOps = modelEntity.train(modelLogit, labels)
        evalOps = modelEntity.evaluate(modelLogit, labels)
        acc_pack = []
        config = tf.ConfigProto()
        config.allow_soft_placement = True    
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            num_batch = Y_data.shape[0] // batch_size
            for e in range(iterations):
                for i in range(num_batch):
                    print('epoch %d / %d, step %d / %d' %(e+1, iterations, i+1, num_batch))
                    batch_offset = i * batch_size
                    batch_end = (i+1) * batch_size
                    X_mini_batch_feed = X_data[batch_offset:batch_end,:,:,:]
                    Y_mini_batch_feed = Y_data[batch_offset:batch_end,:]
                    sess.run(trainOps, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
            
            acc_arg = evalOps.eval({features: X_data_eval, labels: Y_data_eval})
            acc_pack.append(acc_arg)
            conn.send(acc_pack)
            conn.close()
            print("Accuracy:", acc_pack)
    else:
        dt = datetime.now()
        np.random.seed(dt.microsecond)
        net_instnace = np.random.randint(sys.maxsize, size=len(opt))
        train_pack = []
        eval_pack = [] 
        acc_pack = []

        for idx, o in enumerate(opt):
            modelEntity = MobileNet("mobilenet_"+str(net_instnace[idx]), 1, imgHeight, imgWidth, batch_size, numClasses, o)
            modelLogit = modelEntity.build(features)
            trainOps = modelEntity.train(modelLogit, labels)
            evalOps = modelEntity.evaluate(modelLogit, labels)
            train_pack.append(trainOps)
            eval_pack.append(evalOps)
        
        config = tf.ConfigProto()
        config.allow_soft_placement = True   
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            num_batch = Y_data.shape[0] // batch_size
            for e in range(iterations):
                for i in range(num_batch):
                    print('epoch %d / %d, step %d / %d' %(e+1, iterations, i+1, num_batch))
                    batch_offset = i * batch_size
                    batch_end = (i+1) * batch_size
                    X_mini_batch_feed = X_data[batch_offset:batch_end,:,:,:]
                    Y_mini_batch_feed = Y_data[batch_offset:batch_end,:]
                    sess.run(train_pack, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
            
            for evalOps in eval_pack:
                acc_arg = evalOps.eval({features: X_data_eval, labels: Y_data_eval})
                acc_pack.append(acc_arg)
            
            conn.send(acc_pack)
            conn.close()
            print("Accuracy:", acc_pack)
    


def run_params(hyper_params, iterations, conn):
    features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
    labels = tf.placeholder(tf.int64, [None, numClasses])

    X_data = load_bin_raw(bin_dir, numChannels, imgWidth, imgHeight)
    Y_data = load_labels_onehot(label_path, numClasses)
    
    X_data_eval = X_data[0:data_eval_slice,:,:,:]
    Y_data_eval = Y_data[0:data_eval_slice,:]

    dt = datetime.now()
    np.random.seed(dt.microsecond)
    net_instnace = np.random.randint(sys.maxsize)
    batch_size = hyper_params[0]
    opt = hyper_params[1]
    print("\n*** batch size: {} | Opt: {} ***".format(batch_size, opt))
    #input_data = hyper_params[2]
    #prep = hyper_params[3]

    modelEntity = MobileNet("mobilenet_"+str(net_instnace), 1, imgHeight, imgWidth, batch_size, numClasses, opt)
    modelLogit = modelEntity.build(features)
    trainOps = modelEntity.train(modelLogit, labels)
    evalOps = modelEntity.evaluate(modelLogit, labels)

    config = tf.ConfigProto()
    config.allow_soft_placement = True    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_data.shape[0] // batch_size
        for e in range(iterations):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' %(e+1, iterations, i+1, num_batch))
                batch_offset = i * batch_size
                batch_end = (i+1) * batch_size
                X_mini_batch_feed = X_data[batch_offset:batch_end,:,:,:]
                Y_mini_batch_feed = Y_data[batch_offset:batch_end,:]
                sess.run(trainOps, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
        
        acc_arg = evalOps.eval({features: X_data_eval, labels: Y_data_eval})
        conn.send(acc_arg)
        conn.close()
        print("Accuracy:", acc_arg)
        #print("Accuracy:", evalOps.eval({features: X_data, labels: Y_data}))

def evaluate_model():
    features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
    labels = tf.placeholder(tf.int64, [None, numClasses])

    X_data = load_bin_raw(bin_dir, numChannels, imgWidth, imgHeight)
    Y_data = load_labels_onehot(label_path, numClasses)
    X_data_eval = X_data[0:data_eval_slice,:,:,:]
    Y_data_eval = Y_data[0:data_eval_slice,:]

    dt = datetime.now()
    np.random.seed(dt.microsecond)
    net_instnace = np.random.randint(sys.maxsize)

    batch_size = 32
    opt = 'Adam'

    modelEntity = MobileNet("mobilenet_"+str(net_instnace), 1, imgHeight, imgWidth, batch_size, numClasses, opt)
    modelLogit = modelEntity.build(features)
    trainOps = modelEntity.train(modelLogit, labels)
    evalOps = modelEntity.evaluate(modelLogit, labels)
    
    iterations = 20

    config = tf.ConfigProto()
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_data.shape[0] // batch_size
        for e in range(iterations):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' %(e+1, iterations, i+1, num_batch))
                batch_offset = i * batch_size
                batch_end = (i+1) * batch_size
                X_mini_batch_feed = X_data[batch_offset:batch_end,:,:,:]
                Y_mini_batch_feed = Y_data[batch_offset:batch_end,:]
                sess.run(trainOps, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
        
        #acc_arg = sess.run(evalOps, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
        acc_arg = evalOps.eval({features: X_data_eval, labels: Y_data_eval})
        print('accuracy:',acc_arg)
        
        
