import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflearn
from genotypes import PRIMITIVES
from genotypes import Genotype
from operations import *
arch_param_scope=tf.VariableScope("arch_params")



def MixedOp(x,C_out,stride,index,reduction):
	ops=[]

	with tf.variable_scope(arch_param_scope):
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
	# mix=tf.concat([ops],axis=0)
	return tf.add_n(ops)   #????


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
			temp.append(MixedOp(state[j],C_out, stride,offset+j,reduction))   #???
		offset+=len(state)
		state.append(tf.add_n(temp))
	out=tf.concat(state[-multiplier:],axis=-1)
	return out

class Model():
	def __init__(self,x,y,is_training,first_C,class_num,layer_num,cells_num=4,multiplier=4,stem_multiplier=3):
		self.cells_num=cells_num
		self.multiplier=multiplier
		self.x=x
		self.y=y
		with tf.variable_scope('lw',reuse=tf.AUTO_REUSE):
			with slim.arg_scope([slim.conv2d,slim.separable_conv2d],activation_fn=None,padding='SAME',biases_initializer=None,weights_regularizer=slim.l2_regularizer(0.0001)):
				with slim.arg_scope([slim.batch_norm],is_training=is_training):
					C_curr = stem_multiplier*first_C
					s0 =slim.conv2d(x,C_curr,[3,3],activation_fn=tflearn.relu,padding='SAME',biases_initializer=None,weights_regularizer=slim.l2_regularizer(0.0001))
					s0=slim.batch_norm(s0)
					s1 =slim.conv2d(x,C_curr,[3,3],activation_fn=tflearn.relu,padding='SAME',biases_initializer=None,weights_regularizer=slim.l2_regularizer(0.0001))
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
					self.logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
			
			
	def outputs(self):
		return self.logits,tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))

	def get_genotype(self,sess):

		def _parse(stride,sess):
			offset=0
			genotype=[]
			for i in range(self.cells_num):
				edges=[]
				edges_confident=[]
				for j in range(i+2):
					with tf.variable_scope("arch_params", reuse=tf.AUTO_REUSE):
						weight=tf.get_variable("weight{}_{}".format(stride,offset+j))
					max_value=tf.nn.softmax(weight)
					if sess.run(tf.argmax(max_value))!=PRIMITIVES.index('none'):
						edges.append((PRIMITIVES[sess.run(tf.argmax(max_value))],j))
						edges_confident.append(sess.run(tf.reduce_max(max_value)))
				edges=np.array(edges)
				edges_confident=np.array(edges_confident)
				max_edges=edges[np.argsort(edges_confident)][-2:]
				genotype.extend(max_edges.tolist())
				offset+=i+2
			return genotype
		concat = list(range(2+self.cells_num-self.multiplier, self.cells_num+2))
		gene_normal=_parse(1,sess)
		gene_reduce=_parse(2,sess)
		genotype = Genotype(
		normal=gene_normal, normal_concat=concat,
		reduce=gene_reduce, reduce_concat=concat
		)
		return genotype