import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflearn

OPS = {
  'none' : lambda x, C, stride: Zero(x,stride),
  'avg_pool_3x3' : lambda x, C, stride: slim.avg_pool2d(x,[3,3], stride=stride,padding='SAME'),
  'max_pool_3x3' : lambda x, C, stride: slim.max_pool2d(x,[3,3], stride=stride,padding='SAME'),
  'skip_connect' : lambda x, C, stride: tf.identity(x) if stride == [1,1] else FactorizedReduce(x, C),
  'sep_conv_3x3' : lambda x, C, stride: SepConv(x, C, [3,3], stride),
  'sep_conv_5x5' : lambda x, C, stride: SepConv(x, C, [5,5], stride),
  'sep_conv_7x7' : lambda x, C, stride: SepConv(x, C, [7,7], stride),
  'dil_conv_3x3' : lambda x, C, stride: DilConv(x, C, [3,3], stride,2),
  'dil_conv_5x5' : lambda x, C, stride: DilConv(x, C, [5,5], stride,2),
  # 'conv_7x1_1x7' : lambda x, C, stride: nn.Sequential(
  #   nn.ReLU(inplace=False),
  #   nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
  #   nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
  #   nn.BatchNorm2d(C, affine=affine)
  #   ),
}
def Zero(x,stride):
	return tf.zeros_like(x)[:,::stride[0],::stride[1],:]

def DilConv(x,C_out,kernel_size,stride,rate):
	x=tflearn.relu(x)
	x=slim.separable_convolution2d(x,C_out,kernel_size,depth_multiplier=1,stride=stride,rate=rate)
	x=slim.batch_norm(x)
	return x
def SepConv(x,C_out,kernel_size,stride):
	x=tflearn.relu(x)
	C_in=x.get_shape()[-1].value

	x=slim.separable_convolution2d(x,C_in,kernel_size,depth_multiplier=1,stride=stride)
	x=slim.batch_norm(x)

	x=slim.separable_convolution2d(x,C_out,kernel_size,depth_multiplier=1)
	x=slim.batch_norm(x)
	return x

def FactorizedReduce(x,c_out):
	x=tflearn.relu(x)
	conv1=slim.conv2d(x,c_out//2,[1,1],stride=[2,2])
	conv2=slim.conv2d(x[:,1:,1:,:],c_out//2,[1,1],stride=[2,2])
	x=tf.concat([conv1,conv2],-1)
	x=slim.batch_norm(x)
	return x
def ReLUConvBN(x,C_out):
	x=tflearn.relu(x)
	x=slim.conv2d(x,C_out,[1,1])
	x=slim.batch_norm(x)
	return x