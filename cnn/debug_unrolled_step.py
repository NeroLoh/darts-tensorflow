import glob
import numpy as np
import utils
import logging
import argparse
import tensorflow as tf
from model_search import *
from data_utils import read_data
from datetime import datetime

CLASS_NUM=10
def main():
	images, labels = read_data('./data/cifar10',0.5)
	train_dataset = tf.data.Dataset.from_tensor_slices((images["train"],labels["train"]))
	train_dataset=train_dataset.shuffle(100).batch(16)
	train_iter=train_dataset.make_initializable_iterator()
	x_train,y_train=train_iter.get_next()

	logits,train_loss=Model_test(x_train,y_train,True)
	w_var=utils.get_var(tf.trainable_variables(), 'lw')[1]
	arch_var=utils.get_var(tf.trainable_variables(), 'arch_params')[1]

	R=0.01
	valid_grads=tf.gradients(train_loss,w_var)

	arch_grad_before=tf.gradients(train_loss,arch_var)
	with tf.control_dependencies([v+R*g for v,g in zip(w_var,valid_grads)]):
		arch_grad_after=tf.gradients(train_loss,arch_var)

	config = tf.ConfigProto()
	os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
	config.gpu_options.allow_growth = True
	sess=tf.Session(config=config)

	sess.run(tf.global_variables_initializer())
	sess.run([train_iter.initializer])
	print(sess.run(arch_grad_before)[0])
	print(sess.run(arch_grad_after)[0])

if __name__ == '__main__':
	main() 