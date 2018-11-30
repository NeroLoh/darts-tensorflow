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
import genotype 


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

def main():
	class_num=10

	global_step = tf.train.get_or_create_global_step()

	images, labels = read_data(args.data,args.train_portion)
	train_dataset = tf.data.Dataset.from_tensor_slices((images["train"],labels["train"]))
	train_dataset=train_dataset.shuffle(100).batch(args.batch_size)
	train_iter=train_dataset.make_initializable_iterator()
	x_train,y_train=train_iter.get_next()

	valid_dataset = tf.data.Dataset.from_tensor_slices((images["test"],labels["test"]))
	valid_dataset=valid_dataset.shuffle(100).batch(args.batch_size*5)
	valid_iter=valid_dataset.make_initializable_iterator()
	x_valid,y_valid=valid_iter.get_next()

	train_logits,aux_logits=Model(x_train,y_train,True,args.init_channels,class_num,args.layers)
	train_loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_train, logits=train_logits))

	w_regularization_loss = tf.add_n(utils.get_var(tf.losses.get_regularization_losses(), 'lw')[1])
	train_loss+=1e4*args.weight_decay*w_regularization_loss

	if args.auxiliary:
		loss_aux = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_train, logits=aux_logits))
		train_loss += args.auxiliary_weight*loss_aux

	lr=tf.train.cosine_decay(args.learning_rate,global_step,50000/args.batch_size*50)
	accuracy=tf.reduce_mean(tf.cast(tf.nn.in_top_k(train_logits, y_train, 1), tf.float32))	

	valid_logits,_=Model(x_valid,y_valid,False,args.init_channels,class_num,args.layers)
	test_accuracy=tf.reduce_mean(tf.cast(tf.nn.in_top_k(valid_logits, y_valid, 1), tf.float32))


	objs = utils.AvgrageMeter()
	top1 = utils.AvgrageMeter()
	test_top1 = utils.AvgrageMeter()


	config = tf.ConfigProto()
	os.environ["CUDA_VISIBLE_DEVICES"] = '1'
	config.gpu_options.allow_growth = True
	sess=tf.Session(config=config)
	saver = tf.train.Saver(max_to_keep=1)
	sess.run(tf.global_variables_initializer())
	for e in range(args.epochs):
		sess.run([train_iter.initializer,valid_iter.initializer])
		batch=0
		while True:
			try:
				batch+=1
				_,loss, acc,crrunt_lr,gs=sess.run([leader_opt,train_loss,accuracy,lr,global_step])
				objs.update(loss, args.batch_size)
				top1.update(acc, args.batch_size)
				if batch % args.report_freq ==0:
					print("epochs {} steps {} currnt lr is {:.3f}  loss is {}  train_acc is {}".format(e,gs,crrunt_lr,objs.avg,top1.avg))
			except tf.errors.OutOfRangeError:
				print('-'*80)
				print("end of an epoch")
				break
		genotype=model_train.get_genotype(sess)
		print("genotype is {}".format(genotype))
		sess.run(valid_iter.initializer)
		test_acc=sess.run([test_accuracy])
		test_top1.update(acc, args.batch_size)
		print("epochs {} *******************  test_acc is {}".format(e,test_top1.avg))
		saver.save(self.sess, output_dir,gs)
if __name__ == '__main__':
  main() 