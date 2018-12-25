import os
import sys
import time
import glob
import numpy as np
import utils
import logging
import argparse
import tensorflow as tf
from model_search import *
from data_utils import read_data
from datetime import datetime
parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='./data/cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=200, help='report frequency')
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

tf.set_random_seed(args.seed)
output_dir='./outputs/train_model/'
if not os.path.isdir(output_dir):
	print("Path {} does not exist. Creating.".format(output_dir))
	os.makedirs(output_dir)

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
CLASS_NUM=10
def main():
	global_step = tf.train.get_or_create_global_step()

	images, labels = read_data(args.data,args.train_portion)
	train_dataset = tf.data.Dataset.from_tensor_slices((images["train"],labels["train"]))
	train_dataset=train_dataset.map(_pre_process).shuffle(100).batch(args.batch_size)
	train_iter=train_dataset.make_initializable_iterator()
	x_train,y_train=train_iter.get_next()

	valid_dataset = tf.data.Dataset.from_tensor_slices((images["valid"],labels["valid"]))
	valid_dataset=valid_dataset.shuffle(100).batch(args.batch_size)
	valid_iter=valid_dataset.make_initializable_iterator()
	x_valid,y_valid=valid_iter.get_next()

	logits,train_loss=Model(x_train,y_train,True,args.init_channels,CLASS_NUM,args.layers)

	lr=tf.train.cosine_decay(args.learning_rate,global_step,50000/args.batch_size*args.epochs,args.learning_rate_min)

	accuracy=tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y_train, 1), tf.float32))	
	w_regularization_loss = tf.add_n(utils.get_var(tf.losses.get_regularization_losses(), 'lw')[1])
	train_loss+=1e4*args.weight_decay*w_regularization_loss
	tf.summary.scalar('train_loss', train_loss)

	w_var=utils.get_var(tf.trainable_variables(), 'lw')[1]
	
	with tf.control_dependencies([tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))]):
		follower_opt=tf.train.MomentumOptimizer(lr,args.momentum)
		follower_grads=tf.gradients(train_loss, w_var)
		clipped_gradients, norm =tf.clip_by_global_norm(follower_grads,args.grad_clip)
		follower_opt=follower_opt.apply_gradients(zip(clipped_gradients,w_var),global_step)

	infer_logits,infer_loss=Model(x_valid,y_valid,False,args.init_channels,CLASS_NUM,args.layers)
	test_accuracy=tf.reduce_mean(tf.cast(tf.nn.in_top_k(infer_logits, y_valid, 1), tf.float32))
	
	leader_opt=compute_unrolled_step(x_valid,y_valid,w_var,train_loss,follower_opt)

	merged = tf.summary.merge_all()

	config = tf.ConfigProto()
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
	config.gpu_options.allow_growth = True
	sess=tf.Session(config=config)

	writer = tf.summary.FileWriter(output_dir+TIMESTAMP,sess.graph)
	saver = tf.train.Saver(max_to_keep=1)

	sess.run(tf.global_variables_initializer())
	# saver.restore(sess, tf.train.latest_checkpoint(output_dir))

	genotype_record_file=open(output_dir+"genotype_record_file.txt",'w')
	for e in range(args.epochs):
		objs = utils.AvgrageMeter()
		top1 = utils.AvgrageMeter()
		test_top1 = utils.AvgrageMeter()
		sess.run([train_iter.initializer,valid_iter.initializer])
		while True:
			try:
				_,loss, acc,crrunt_lr,gs=sess.run([leader_opt,train_loss,accuracy,lr,global_step])
				objs.update(loss, args.batch_size)
				top1.update(acc, args.batch_size)
				if gs % args.report_freq==0:
					print("epochs {} steps {} currnt lr is {:.3f}  loss is {}  train_acc is {}".format(e,gs,crrunt_lr,objs.avg,top1.avg))
					summary=sess.run(merged)
					writer.add_summary(summary, gs)
			except tf.errors.OutOfRangeError:
				print('-'*80)
				print("end of an epoch")
				break
		genotype=get_genotype(sess)
		print("genotype is {}".format(genotype))
		genotype_record_file.write("{}".format(genotype)+'\n')
		sess.run([valid_iter.initializer])
		test_acc=sess.run(test_accuracy)
		test_top1.update(test_acc, args.batch_size)
		print(" ******************* epochs {} test_acc is {}".format(e,test_top1.avg))
		saver.save(sess, output_dir+"model",gs)	


def compute_unrolled_step(x_valid,y_valid,w_var,train_loss,follower_opt):
	arch_var=utils.get_var(tf.trainable_variables(), 'arch_params')[1]
	with tf.control_dependencies([follower_opt]):
		leader_opt= tf.train.AdamOptimizer(args.arch_learning_rate, 0.5, 0.999)
		leader_grads=leader_opt.compute_gradients(train_loss,var_list =arch_var)

	_,valid_loss=Model(x_valid,y_valid,True,args.init_channels,CLASS_NUM,args.layers)
	tf.summary.scalar('valid_loss', valid_loss)

	valid_grads=tf.gradients(valid_loss,w_var)
	r=1e-2

	sum_grads=tf.get_variable(name='sum_grads',shape=[],initializer=tf.constant_initializer(0.0))
	opt=sum_grads.assign(0)
	with tf.control_dependencies([opt]):
		for v in valid_grads:
			sum_grads=sum_grads+tf.nn.l2_loss(v)

	R = r / tf.sqrt(sum_grads)

	with tf.control_dependencies([v+R*g for v,g in zip(w_var,valid_grads)]):
		train_grads_pos=tf.gradients(train_loss,arch_var)

	with tf.control_dependencies([v-2*R*g for v,g in zip(w_var,valid_grads)]):
		train_grads_neg=tf.gradients(train_loss,arch_var)

	with tf.control_dependencies([v+R*g for v,g in zip(w_var,valid_grads)]):
		implicit_grads=[tf.divide(gp-gn,2*R) for gp,gn in zip(train_grads_pos,train_grads_neg)]
	for i,(g,v) in enumerate(leader_grads):
		leader_grads[i]=(g-args.learning_rate*implicit_grads[i],v)
	leader_opt=leader_opt.apply_gradients(leader_grads)
	return leader_opt

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