import glob
import numpy as np
import utils
import logging
import argparse
import tensorflow as tf
import time
from model_search import *
from data_utils import read_data
from datetime import datetime
def get_genotype(sess,cells_num=4,multiplier=4):

	def _parse(stride,sess):
		offset=0
		genotype=[]


		for i in range(cells_num):
			edges=[]
			edges_confident=[]
			for j in range(i+2):
				with tf.variable_scope("",reuse=tf.AUTO_REUSE):

					value=[4,1,2,3,1]
				value_sorted=np.argsort(value)
				max_index=value_sorted[-2] if value_sorted[-1]==PRIMITIVES.index('none') else value_sorted[-1]
					
				edges.append((PRIMITIVES[max_index],j))
				edges_confident.append(value[max_index])

			edges_confident=np.array(edges_confident)
			max_edges=[edges[np.argsort(edges_confident)[-1]],edges[np.argsort(edges_confident)[-2]]]
			genotype.extend(max_edges)
			offset+=i+2
		return genotype
	concat = list(range(2+cells_num-multiplier, cells_num+2))
	gene_normal=_parse(1,sess)
	gene_reduce=_parse(2,sess)
	genotype = Genotype(
	normal=gene_normal, normal_concat=concat,
	reduce=gene_reduce, reduce_concat=concat
	)
	return genotype


CLASS_NUM=10
def main():
	tf.set_random_seed(1234)
	images, labels = read_data('./data/cifar10',0.5)
	train_dataset = tf.data.Dataset.from_tensor_slices((images["train"],labels["train"]))
	train_dataset=train_dataset.shuffle(100).batch(16)
	train_iter=train_dataset.make_initializable_iterator()
	x_train,y_train=train_iter.get_next()
	# x_train=images["train"][:16]
	# y_train=labels["train"][:16]

	logits,train_loss=Model_test(x_train,y_train,True)
	w_var=utils.get_var(tf.trainable_variables(), 'weight_var')[1]
	arch_var=utils.get_var(tf.trainable_variables(), 'arch_var')[1]

	_,unrolled_train_loss=Model_test(x_train,y_train,True, "unrolled_weight_var")  
	unrolled_w_var=utils.get_var(tf.trainable_variables(), 'unrolled_weight_var')[1]
	cpoy_weight_opts=[v_.assign(v) for v_,v in zip(unrolled_w_var,w_var)]

	with tf.control_dependencies(cpoy_weight_opts):
		unrolled_optimizer=tf.train.GradientDescentOptimizer(0.001)
		unrolled_optimizer=unrolled_optimizer.minimize(unrolled_train_loss,var_list=unrolled_w_var)
	#w'
	with tf.control_dependencies([unrolled_optimizer]):
		valid_grads=tf.gradients(unrolled_train_loss,unrolled_w_var)

	R=0.01/tf.global_norm(valid_grads)
	
	#Original Implementation
	# opts=[v.assign(v+R*g) for v,g in zip(w_var,valid_grads)]
	# with tf.control_dependencies(opts):
	# 	arch_grad_after=tf.gradients(train_loss,arch_var)

	optimizer1=tf.train.GradientDescentOptimizer(R)
	optimizer1=optimizer1.apply_gradients(zip(valid_grads,w_var))
	with tf.control_dependencies([optimizer1]):
		arch_grad_after=tf.gradients(train_loss,arch_var)


	config = tf.ConfigProto()
	os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
	config.gpu_options.allow_growth = True
	sess=tf.Session(config=config)

	sess.run(tf.global_variables_initializer())
	sess.run([train_iter.initializer])
	print(sess.run(valid_grads)[0])
	start = time.time()	     
	print(sess.run(arch_grad_after)[0])
	print("time_is {}".format(time.time()-start))
	print(sess.run(valid_grads)[0])
	# start = time.time()	
	# print(sess.run(arch_grad_after)[0])
	# print("time_is {}".format(time.time()-start))
	# start = time.time()	
	# print(sess.run(arch_grad_2)[0])
	# print("time_is {}".format(time.time()-start))


if __name__ == '__main__':
	main() 
	# print(get_genotype(None))