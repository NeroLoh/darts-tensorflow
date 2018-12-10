# coding=utf-8

import tensorflow as tf
import time
import os
from data_utils import read_data
import tensorflow.contrib.slim as slim
import tflearn
BATCH_SIZE = 64
DECAY_STEP = 10
INITIAL_LEARNING_RATE = 0.01
LEARNING_RATE_DECAY_FACTOR = 0.9997

MOMENTUM = 0.9
EPOCH_NUM = 10000001

DROPOUT = 0.9


def conv2d(_x, _w, _b):
    return tf.nn.bias_add(tf.nn.conv2d(_x, _w, [1, 1, 1, 1], padding='SAME'), _b)


def max_pool(_x, f):
    return tf.nn.max_pool(_x, [1, f, f, 1], [1, 1, 1, 1], padding='SAME')


def lrn(_x):
    return tf.nn.lrn(_x, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)


def init_w(namespace, shape, wd, stddev, reuse=False):
    with tf.variable_scope(namespace, reuse=reuse):
        initializer = tf.truncated_normal_initializer(dtype=tf.float32, stddev=stddev)
        w = tf.get_variable("w", shape=shape, initializer=initializer)
        
        if wd:
            weight_decay = tf.multiply(tf.nn.l2_loss(w), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
    return w


def init_b(namespace, shape, reuse=False):
    with tf.variable_scope(namespace, reuse=reuse):
        initializer = tf.constant_initializer(0.0)
        b = tf.get_variable("b", shape=shape, initializer=initializer)
    return b


def batch_normal(xs, out_size):
    axis = list(range(len(xs.get_shape()) - 1))
    n_mean, n_var = tf.nn.moments(xs, axes=axis)
    scale = tf.Variable(tf.ones([out_size]))
    shift = tf.Variable(tf.zeros([out_size]))
    epsilon = 0.001
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    
    def mean_var_with_update():
        ema_apply_op = ema.apply([n_mean, n_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(n_mean), tf.identity(n_var)
        
    mean, var = mean_var_with_update()
        
    bn = tf.nn.batch_normalization(xs, mean, var, shift, scale, epsilon)
    return bn
    

def inference(images, reuse=False):
    '''Build the network model and return logits'''
    
    # conv1
    w1 = init_w("conv1", [3, 3, 3, 24], None, 0.01, reuse)
    bw1 = init_b("conv1", [24], reuse)
    conv1 = conv2d(images, w1, bw1)
    # bn1 = batch_normal(conv1, 24)
    bn1=lrn(conv1)
    c_output1 = tf.nn.relu(bn1)
    pool1 = max_pool(c_output1, 2)
    
    # conv2
    w2 = init_w("conv2", [3, 3, 24, 96], None, 0.01, reuse)
    bw2 = init_b("conv2", [96], reuse)
    conv2 = conv2d(pool1, w2, bw2)
    # bn2 = batch_normal(conv2, 96)
    bn2=lrn(conv2)
    c_output2 = tf.nn.relu(bn2)
    pool2 = max_pool(c_output2, 2)
    
    # conv3
    w3 = init_w("conv3", [3, 3, 96, 192], None, 0.01, reuse)
    bw3 = init_b("conv3", [192], reuse)
    conv3 = conv2d(pool2, w3, bw3)
    # bn3 = batch_normal(conv3, 192)
    c_output3 = tf.nn.relu(conv3)
    
    # conv4
    w4 = init_w("conv4", [3, 3, 192, 192], None, 0.01, reuse)
    bw4 = init_b("conv4", [192], reuse)
    conv4 = conv2d(conv3, w4, bw4)
    # bn4 = batch_normal(conv4, 192)
    c_output4 = tf.nn.relu(conv4)
    
    # conv5
    w5 = init_w("conv5", [3, 3, 192, 96], None, 0.01, reuse)
    bw5 = init_b("conv5", [96], reuse)
    conv5 = conv2d(conv4, w5, bw5)
    # bn5 = batch_normal(conv5, 96)
    c_output5 = tf.nn.relu(conv5)
    pool5 = max_pool(c_output5, 2)
                
    # FC1
    wfc1 = init_w("fc1", [96*32*32, 1024], None, 1e-2, reuse)
    bfc1 = init_b("fc1", [1024], reuse)
    shape = pool5.get_shape()
    reshape = tf.reshape(pool5, [-1, shape[1].value*shape[2].value*shape[3].value])
    w_x1 = tf.matmul(reshape, wfc1) + bfc1
    # bn6 = batch_normal(w_x1, 1024)
    fc1 = tf.nn.relu(w_x1)
    
    # FC2
    wfc2 = init_w("fc2", [1024, 1024], None, 1e-2, reuse)
    bfc2 = init_b("fc2", [1024], reuse)
    w_x2 = tf.matmul(fc1, wfc2) + bfc2
    # bn7 = batch_normal(w_x2, 1024)
    fc2 = tf.nn.relu(w_x2)
    
    # FC3
    wfc3 = init_w("fc3", [1024, 10], None, 1e-2, reuse)
    bfc3 = init_b("fc3", [10], reuse)
    softmax_linear = tf.add(tf.matmul(fc2, wfc3), bfc3)
    
    return softmax_linear

def model_test(x,is_training,class_num=10):
    with tf.variable_scope('lw',reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.conv2d], normalizer_fn=None,activation_fn=tflearn.relu,padding='SAME'):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=0.01),
                                weights_regularizer=None,biases_initializer=tf.constant_initializer(0.0)):
                with slim.arg_scope([slim.batch_norm],
                                    decay=0.9,
                                    scale=False, epsilon=1e-3, is_training=is_training,
                                    zero_debias_moving_mean=True):

                    x = slim.conv2d(x, 24, [3,3],normalizer_fn=tf.nn.lrn)
                    x = slim.max_pool2d(x, [2, 2])
                    x = slim.conv2d(x, 96, [3,3],normalizer_fn=tf.nn.lrn)
                    x = slim.max_pool2d(x, [2, 2])                    
                    x = slim.conv2d(x, 192, [3,3])
                    x = slim.conv2d(x, 192, [3,3])
                    x = slim.conv2d(x, 96, [3,3])
                    x = slim.max_pool2d(x, [2, 2])

                    # x = slim.dropout(x,is_training=is_training, keep_prob=0.9)
                    
                    x = slim.flatten(x)                        
                    x = slim.fully_connected(x, 1024, activation_fn=None)
                    x = tflearn.relu(x)
                    x = slim.fully_connected(x, 1024, activation_fn=None)
                    x = tflearn.relu(x)


                    logits = slim.fully_connected(x, class_num,activation_fn=None)
                    return logits

def loss_function(logits, labels):
    '''return loss'''
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train_step(loss, global_step):
    
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  DECAY_STEP,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
    tf.add_to_collection('learning_rate', lr)
    
    train_op = tf.train.MomentumOptimizer(lr, MOMENTUM).minimize(loss)
    return train_op
    

def train():
    with tf.Graph().as_default():
        
        global_step = tf.train.get_or_create_global_step()
        
        images, labels = read_data("./data/cifar10")
        
        train_dataset = tf.data.Dataset.from_tensor_slices((images["train"],labels["train"]))
        train_dataset=train_dataset.shuffle(10000).batch(BATCH_SIZE)
        train_iter=train_dataset.make_initializable_iterator()
        x_train,y_train=train_iter.get_next()

        test_dataset = tf.data.Dataset.from_tensor_slices((images["test"],labels["test"]))
        test_dataset=test_dataset.shuffle(10000).batch(BATCH_SIZE)
        test_iter=test_dataset.make_initializable_iterator()
        x_test,y_test=test_iter.get_next()

        # train step
        logits = model_test(x_train, True)
        loss = loss_function(logits, y_train)
        train_op = train_step(loss, global_step)
   
        
        ##### Test step

        test_logits = model_test(x_test, False)
        test_labels = tf.one_hot(y_test, depth=10)
        
        test_correct_pred = tf.equal(tf.argmax(test_logits, 1), tf.argmax(test_labels, 1))
        test_accuracy = tf.reduce_mean(tf.cast(test_correct_pred, tf.float32))
        
        add_global = global_step.assign_add(1)
        

        config = tf.ConfigProto()
        os.environ["CUDA_VISIBLE_DEVICES"] =  str(1)
        config.gpu_options.allow_growth = True

        sess=tf.Session()
        sess.run(tf.global_variables_initializer())
        for e in range(60):
            
            f = open("result.txt", 'a+')
            sess.run(train_iter.initializer)
            while True:
                try:
                    sess.run(train_op)
                    step = sess.run(add_global)
                    
                    if step % 1000 == 0:
                        lo =  sess.run(loss)
                        lr = sess.run(tf.get_collection('learning_rate'))
                        
                        print("%d  losses: %f" % (step, lo))
                        print("%d  learning rate: %f" % (step, lr[0]))
                        f.write("%.5f\n" % lo)
                        f.write("%.5f\n" % lr[0])
                        
                except tf.errors.OutOfRangeError:
                    print('-'*80)
                    print("end of an train epoch")
                    break
            sess.run(test_iter.initializer)                               
            test_acc = 0.0
            for i in range(156):
                test_acc += sess.run(test_accuracy)
                
            test_acc /= 156
                    
            print("%d  Test acc: %f" % (step, test_acc))
            f.write("%.5f\n" % test_acc)
            f.flush()
            print("Train over")
                
def main():
    train()
  

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print("Total time: %f" % (end_time-start_time))
