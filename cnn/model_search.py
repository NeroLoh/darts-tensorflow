import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflearn
from genotypes import PRIMITIVES
from genotypes import Genotype
from operations import *
import utils
null_scope=tf.VariableScope("")



def MixedOp(x,C_out,stride,index,reduction):
	ops=[]

	with tf.variable_scope(null_scope):
		with tf.variable_scope("arch_params",reuse=tf.AUTO_REUSE):
			weight=tf.get_variable("weight{}_{}".format(2 if reduction else 1,index),[len(PRIMITIVES)],initializer=tf.random_normal_initializer(0,1e-3),regularizer=slim.l2_regularizer(0.0001))
	weight=tf.nn.softmax(weight)
	weight=tf.reshape(weight,[-1,1,1,1])
	index=0
	for primitive in PRIMITIVES:

		op = OPS[primitive](x, C_out, stride)
		if 'pool' in primitive:
			op = slim.batch_norm(op)

		mask=[i==index for i in range(len(PRIMITIVES))]
		w_mask  = tf.constant(mask, tf.bool)
		w = tf.boolean_mask(weight, w_mask)
		ops.append(op*w)
		index+=1
	return tf.add_n(ops)  


def Cell(s0,s1,cells_num, multiplier, C_out, reduction, reduction_prev):
	if reduction_prev:
		s0 = FactorizedReduce(s0,C_out)
	else:
		s0 = ReLUConvBN(s0,C_out)
	s1=ReLUConvBN(s1,C_out)

	state=[s0,s1]
	offset=0
	for i in range(cells_num):
		temp=[]
		for j in range(2+i):
			stride = [2,2] if reduction and j < 2 else [1,1]
			temp.append(MixedOp(state[j],C_out, stride,offset+j,reduction))  
		offset+=len(state)
		state.append(tf.add_n(temp))
	out=tf.concat(state[-multiplier:],axis=-1)
	return out

def Model(x,y,is_training,first_C,class_num,layer_num,cells_num=4,multiplier=4,stem_multiplier=3):
	with tf.variable_scope('lw',reuse=tf.AUTO_REUSE):
		with slim.arg_scope([slim.conv2d,slim.separable_conv2d],activation_fn=None,padding='SAME',biases_initializer=None,weights_regularizer=slim.l2_regularizer(0.0001)):
			with slim.arg_scope([slim.batch_norm],is_training=is_training):
				C_curr = stem_multiplier*first_C
				s0 =slim.conv2d(x,C_curr,[3,3],activation_fn=tflearn.relu)
				s0=slim.batch_norm(s0)
				s1 =slim.conv2d(x,C_curr,[3,3],activation_fn=tflearn.relu)
				s1=slim.batch_norm(s1)
				reduction_prev = False
				for i in range(layer_num):
					if i in [layer_num//3, 2*layer_num//3]:
						C_curr *= 2
						reduction = True
					else:
						reduction = False
					s0,s1 =s1,Cell(s0,s1,cells_num, multiplier, C_curr, reduction, reduction_prev)
					reduction_prev = reduction
				out=tf.reduce_mean(s1, [1, 2], keep_dims=True, name='global_pool')
				logits = slim.conv2d(out, class_num, [1, 1], activation_fn=None,normalizer_fn=None,weights_regularizer=slim.l2_regularizer(0.0001))
				logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
	train_loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
	return logits,train_loss
def Model_test(x,y,is_training):
	weight_decay=3e-4			
	with tf.variable_scope('lw',reuse=tf.AUTO_REUSE):
		with slim.arg_scope([slim.conv2d,slim.separable_conv2d],activation_fn=None,padding='SAME',biases_initializer=None,weights_regularizer=slim.l2_regularizer(0.0001)):
			with slim.arg_scope([slim.batch_norm],is_training=is_training):
				x=slim.max_pool2d(x,[3,3],stride=2)
				out=Cell(x,x,2, 4, 32, False, False)
				out=tf.reduce_mean(out, [1, 2], keep_dims=True, name='global_pool')
				logits = slim.conv2d(out, 10, [1, 1], activation_fn=None,normalizer_fn=None,weights_regularizer=slim.l2_regularizer(0.0001))
				logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
	train_loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

	return logits,train_loss
def get_genotype(sess,cells_num=4,multiplier=4):

	def _parse(stride,sess):
		offset=0
		genotype=[]
		arch_var_name,arch_var=utils.get_var(tf.trainable_variables(), 'arch_params')

		for i in range(cells_num):
			edges=[]
			edges_confident=[]
			for j in range(i+2):
				with tf.variable_scope("",reuse=tf.AUTO_REUSE):
					weight=arch_var[arch_var_name.index("arch_params/weight{}_{}:0".format(stride,offset+j))]
					value=sess.run(weight)
				value_sorted=value.argsort()
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