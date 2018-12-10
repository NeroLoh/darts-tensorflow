import os
import sys
import time
import glob
import numpy as np
import utils
import logging
import argparse
import tensorflow as tf
from model import *
from data_utils import read_data
from collections import namedtuple
import genotypes 
from datetime import datetime


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='./data/cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=48, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()
output_dir='./outputs/test_model/'
if not os.path.isdir(output_dir):
	print("Path {} does not exist. Creating.".format(output_dir))
	os.makedirs(output_dir)
tf.set_random_seed(args.seed)
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
CLASS_NUM=10
def main():
	global_step = tf.train.get_or_create_global_step()

	images, labels = read_data(args.data)
	train_dataset = tf.data.Dataset.from_tensor_slices((images["train"],labels["train"]))
	train_dataset=train_dataset.map(_pre_process).shuffle(5000).batch(args.batch_size)
	train_iter=train_dataset.make_initializable_iterator()
	x_train,y_train=train_iter.get_next()

	test_dataset = tf.data.Dataset.from_tensor_slices((images["test"],labels["test"]))
	test_dataset=test_dataset.shuffle(5000).batch(args.batch_size)
	test_iter=test_dataset.make_initializable_iterator()
	x_test,y_test=test_iter.get_next()

	genotype = eval("genotypes.%s" % args.arch)
	train_logits,aux_logits=Model(x_train,y_train,True,args.init_channels,CLASS_NUM,args.layers,args.auxiliary,genotype)
	train_loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_train, logits=train_logits))

	w_regularization_loss = tf.add_n(utils.get_var(tf.losses.get_regularization_losses(), 'lw')[1])
	train_loss+=1e4*args.weight_decay*w_regularization_loss
	# tf.summary.scalar('train_loss', train_loss)

	if args.auxiliary:
		loss_aux = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_train, logits=aux_logits))
		train_loss += args.auxiliary_weight*loss_aux

	lr=tf.train.cosine_decay(args.learning_rate,global_step,50000/args.batch_size*args.epochs)
	accuracy=tf.reduce_mean(tf.cast(tf.nn.in_top_k(train_logits, y_train, 1), tf.float32))	

	test_logits,_=Model(x_test,y_test,False,args.init_channels,CLASS_NUM,args.layers,args.auxiliary,genotype)
	test_accuracy=tf.reduce_mean(tf.cast(tf.nn.in_top_k(test_logits, y_test, 1), tf.float32))
	test_accuracy_top5=tf.reduce_mean(tf.cast(tf.nn.in_top_k(test_logits, y_test, 5), tf.float32))
	tf.summary.scalar('test_accuracy_top1', test_accuracy)


	with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
		opt=tf.train.MomentumOptimizer(lr,args.momentum)
		opt=opt.minimize(train_loss,global_step)

	merged = tf.summary.merge_all()


	config = tf.ConfigProto()
	os.environ["CUDA_VISIBLE_DEVICES"] =  str(args.gpu)
	config.gpu_options.allow_growth = True
	sess=tf.Session(config=config)

	writer = tf.summary.FileWriter(output_dir+TIMESTAMP,sess.graph)
	saver = tf.train.Saver(max_to_keep=1)
	sess.run(tf.global_variables_initializer())
	test_batch=0
	for e in range(args.epochs):
		objs = utils.AvgrageMeter()
		top1 = utils.AvgrageMeter()
		sess.run(train_iter.initializer)
		while True:
			try:
				_,loss, acc,crrunt_lr,gs=sess.run([opt,train_loss,accuracy,lr,global_step])
				objs.update(loss, args.batch_size)
				top1.update(acc, args.batch_size)
				if gs % args.report_freq==0:
					print("epochs {} steps {} currnt lr is {:.3f}  loss is {}  train_acc is {}".format(e,gs,crrunt_lr,objs.avg,top1.avg))
			except tf.errors.OutOfRangeError:
				print('-'*80)
				print("end of an train epoch")
				break
		if e % 5 ==0:
			test_top1 = utils.AvgrageMeter()
			sess.run(test_iter.initializer)
			while True:
				try:
					test_batch+=1
					summary,test_acc=sess.run([merged,test_accuracy])
					test_top1.update(test_acc, args.batch_size)
					if test_batch % 100:
						writer.add_summary(summary, test_batch)
				except tf.errors.OutOfRangeError:
					print("******************* epochs {}   test_acc is {}".format(e,test_top1.avg))
					saver.save(sess, output_dir+"model",test_batch)
					print('-'*80)
					print("end of an test epoch")
					break

def _pre_process(x,label):
	cutout_length=args.cutout_length
	x = tf.pad(x, [[4, 4], [4, 4], [0, 0]])
	x = tf.random_crop(x, [32, 32, 3])
	x = tf.image.random_flip_left_right(x)
	if cutout_length is not None:
		mask = tf.ones([cutout_length, cutout_length], dtype=tf.int32)
		start = tf.random_uniform([2], minval=0, maxval=32, dtype=tf.int32)
		mask = tf.pad(mask, [[cutout_length + start[0], 32 - start[0]],
		                     [cutout_length + start[1], 32 - start[1]]])
		mask = mask[cutout_length: cutout_length + 32,
		            cutout_length: cutout_length + 32]
		mask = tf.reshape(mask, [32, 32, 1])
		mask = tf.tile(mask, [1, 1, 3])
		x = tf.where(tf.equal(mask, 0), x=x, y=tf.zeros_like(x))
	return x,label
if __name__ == '__main__':
	main() 

