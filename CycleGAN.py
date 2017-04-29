#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: CycleGAN.py
# Author: tmquan

import cv2
import numpy as np
import skimage.io
import glob
import pickle
import os
import sys
import argparse

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops

from tensorpack import (FeedfreeTrainerBase, QueueInput, ModelDesc, DataFlow)
from tensorpack import *

from tensorpack.utils.viz import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
import tensorpack.tfutils.symbolic_functions as symbolic_functions
#from GAN import GANTrainer, GANModelDesc

# Global variables
BATCH_SIZE = 1
EPOCH_SIZE = 30
CHANNEL = 1
LAMBDA = 1e+2
BETA   = 1e+2
NF	   = 64 # number of filter in generator F and G
SHAPE  = 512

# Declare operators from https://github.com/XHUJOY/CycleGAN-tensorflow/blob/master/ops.py
def batch_norm(x, name="batch_norm"):
	return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)

def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
	with tf.variable_scope(name):
		return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
							weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
							biases_initializer=None)

def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
	with tf.variable_scope(name):
		return slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None,
									weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
									biases_initializer=None)

def lrelu(x, leak=0.2, name="lrelu"):
	return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
	with tf.variable_scope(scope or "Linear"):
		matrix = tf.get_variable("Matrix", [input_.get_shape()[-1], output_size], tf.float32,
								 tf.random_normal_initializer(stddev=stddev))
		bias = tf.get_variable("bias", [output_size],
			initializer=tf.constant_initializer(bias_start))
		if with_w:
			return tf.matmul(input_, matrix) + bias, matrix, bias
		else:
			return tf.matmul(input_, matrix) + bias

# 
# https://github.com/XHUJOY/CycleGAN-tensorflow/blob/master/module.py
def discriminator(image, name="discriminator"):
	with tf.variable_scope(name):
		# image is 256 x 256 x input_c_dim
		h0 = lrelu(conv2d(image, NF, name='d_h0_conv'))
		# h0 is (128 x 128 x self.df_dim)
		h1 = lrelu(batch_norm(conv2d(h0, NF*2, name='d_h1_conv'), 'd_bn1'))
		# h1 is (64 x 64 x self.df_dim*2)
		h2 = lrelu(batch_norm(conv2d(h1, NF*4, name='d_h2_conv'), 'd_bn2'))
		# h2 is (32x 32 x self.df_dim*4)
		h3 = lrelu(batch_norm(conv2d(h2, NF*8, s=1, name='d_h3_conv'), 'd_bn3'))
		# h3 is (32 x 32 x self.df_dim*8)
		h4 = conv2d(h3, 1, s=1, name='d_h3_pred')
		# h4 is (32 x 32 x 1)
		return h4

def generator_unet(image, name="generator"):
	with tf.variable_scope(name):
		# image is 256 x 256 x input_c_dim

		# image is (256 x 256 x input_c_dim)
		e1 = conv2d(image, NF, name='g_e1_conv')
		# e1 is (128 x 128 x self.gf_dim)
		e2 = batch_norm(conv2d(lrelu(e1), NF*2, name='g_e2_conv'), 'g_bn_e2')
		# e2 is (64 x 64 x self.gf_dim*2)
		e3 = batch_norm(conv2d(lrelu(e2), NF*4, name='g_e3_conv'), 'g_bn_e3')
		# e3 is (32 x 32 x self.gf_dim*4)
		e4 = batch_norm(conv2d(lrelu(e3), NF*8, name='g_e4_conv'), 'g_bn_e4')
		# e4 is (16 x 16 x self.gf_dim*8)
		e5 = batch_norm(conv2d(lrelu(e4), NF*8, name='g_e5_conv'), 'g_bn_e5')
		# e5 is (8 x 8 x self.gf_dim*8)
		e6 = batch_norm(conv2d(lrelu(e5), NF*8, name='g_e6_conv'), 'g_bn_e6')
		# e6 is (4 x 4 x self.gf_dim*8)
		e7 = batch_norm(conv2d(lrelu(e6), NF*8, name='g_e7_conv'), 'g_bn_e7')
		# e7 is (2 x 2 x self.gf_dim*8)
		e8 = batch_norm(conv2d(lrelu(e7), NF*8, name='g_e8_conv'), 'g_bn_e8')
		# e8 is (1 x 1 x self.gf_dim*8)

		d1 = deconv2d(tf.nn.relu(e8), NF*8, name='g_d1')
		d1 = tf.concat([tf.nn.dropout(batch_norm(d1, 'g_bn_d1'), 0.5), e7], 3)
		# d1 is (2 x 2 x self.gf_dim*8*2)

		d2 = deconv2d(tf.nn.relu(d1), NF*8, name='g_d2')
		d2 = tf.concat([tf.nn.dropout(batch_norm(d2, 'g_bn_d2'), 0.5), e6], 3)
		# d2 is (4 x 4 x self.gf_dim*8*2)

		d3 = deconv2d(tf.nn.relu(d2), NF*8, name='g_d3')
		d3 = tf.concat([tf.nn.dropout(batch_norm(d3, 'g_bn_d3'), 0.5), e5], 3)
		# d3 is (8 x 8 x self.gf_dim*8*2)

		d4 = deconv2d(tf.nn.relu(d3), NF*8, name='g_d4')
		d4 = tf.concat([batch_norm(d4, 'g_bn_d4'), e4], 3)
		# d4 is (16 x 16 x self.gf_dim*8*2)

		d5 = deconv2d(tf.nn.relu(d4), NF*4, name='g_d5')
		d5 = tf.concat([batch_norm(d5, 'g_bn_d5'), e3], 3)
		# d5 is (32 x 32 x self.gf_dim*4*2)

		d6 = deconv2d(tf.nn.relu(d5), NF*2, name='g_d6')
		d6 = tf.concat([batch_norm(d6, 'g_bn_d6'), e2], 3)
		# d6 is (64 x 64 x self.gf_dim*2*2)

		d7 = deconv2d(tf.nn.relu(d6), NF, name='g_d7')
		d7 = tf.concat([batch_norm(d7, 'g_bn_d7'), e1], 3)
		# d7 is (128 x 128 x self.gf_dim*1*2)

		d8 = deconv2d(tf.nn.relu(d7), CHANNEL, name='g_d8')
		# d8 is (256 x 256 x output_c_dim)

		return tf.nn.tanh(d8)

def generator_resnet(image, name="generator"):
	with tf.variable_scope(name):
		# image is 256 x 256 x input_c_dim

		def residual_block(x, dim, ks=3, s=1, name='res'):
			p = int((ks - 1) / 2)
			y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
			y = batch_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
			y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
			y = batch_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
			return y + x

		# s = options.image_size
		# Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
		# The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
		# R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
		c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
		c1 = tf.nn.relu(batch_norm(conv2d(c0, NF, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
		c2 = tf.nn.relu(batch_norm(conv2d(c1, NF*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
		c3 = tf.nn.relu(batch_norm(conv2d(c2, NF*4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
		# define G network with 9 resnet blocks
		r1 = residual_block(c3, NF*4, name='g_r1')
		r2 = residual_block(r1, NF*4, name='g_r2')
		r3 = residual_block(r2, NF*4, name='g_r3')
		r4 = residual_block(r3, NF*4, name='g_r4')
		r5 = residual_block(r4, NF*4, name='g_r5')
		r6 = residual_block(r5, NF*4, name='g_r6')
		r7 = residual_block(r6, NF*4, name='g_r7')
		r8 = residual_block(r7, NF*4, name='g_r8')
		r9 = residual_block(r8, NF*4, name='g_r9')

		d1 = deconv2d(r9, NF*2, 3, 2, name='g_d1_dc')
		d1 = tf.nn.relu(batch_norm(d1, 'g_d1_bn'))
		d2 = deconv2d(d1, NF, 3, 2, name='g_d2_dc')
		d2 = tf.nn.relu(batch_norm(d2, 'g_d2_bn'))
		d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
		pred = conv2d(d2, CHANNEL, 7, 1, padding='VALID', name='g_pred_c')
		pred = tf.nn.tanh(batch_norm(pred, 'g_pred_bn'))

		return pred


# Declare models
class CycleGANModel(ModelDesc):

	def _get_inputs(self):
		return [InputDesc(tf.float32, [None, SHAPE, SHAPE, CHANNEL], 'image'),
				InputDesc(tf.float32, [None, SHAPE, SHAPE, CHANNEL], 'label')] # if 1 AtoB, if 0 BtoA
		# pass	
	def collect_variables(self, gG_scope='gG', dX_scope='dX',
								gF_scope='gF', dY_scope='dY'):
		self.gG_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, gG_scope)
		self.gF_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, gF_scope)
		self.dX_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, dX_scope)
		self.dY_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, dY_scope)

	# @auto_reuse_variable_scope
	def generator_G(self, imgs):
		return generator_unet(imgs, name='gG')
		# pass

	# @auto_reuse_variable_scope
	def generator_F(self, imgs):
		return generator_unet(imgs, name='gF')
		# pass

	# @auto_reuse_variable_scope
	def discriminator_X(self, imgs):
		return discriminator(imgs, name='dX')
		# pass

	# @auto_reuse_variable_scope
	def discriminator_Y(self, imgs):
		return discriminator(imgs, name='dY')
		# pass

	def _get_optimizer(self):
		lr = symbolic_functions.get_scalar_var('learning_rate', 2e-4, summary=True)
		return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)
		# pass


	def _build_graph(self, inputs):
		image, label = inputs
		image = (image / 255.0 - 0.5)*2.0
		label = (label / 255.0 - 0.5)*2.0
		
		X = image
		Y = label


		###############################################################################################
		#with tf.variable_scope('AtoB'):
		
		
		with tf.name_scope('genAtoB'):
			with tf.name_scope('gG'):
				Y_ = self.generator_G(X)
			with tf.name_scope('gF'):			
				X__ = self.generator_F(Y_)
			with tf.name_scope('dY'):
				pred_real_Y = self.discriminator_Y(Y)
				pred_fake_Y = self.discriminator_Y(Y_)
		with tf.name_scope('genBtoA'):
			with tf.name_scope('gF'):
				X_ = self.generator_F(Y)	
			with tf.name_scope('gG'):			
				Y__ = self.generator_G(X_)
			with tf.name_scope('dX'):
				pred_real_X = self.discriminator_X(X)
				pred_fake_X = self.discriminator_X(X_)

		with tf.name_scope("GAN_loss"):
			###############################################################################################
			with tf.name_scope('discriminator'):
				
				# d_loss_pos_Y = tf.reduce_mean(tf.square(tf.subtract(pred_real_Y, tf.ones_like(pred_real_Y))), name='loss_real_Y')
				# d_loss_neg_Y = tf.reduce_mean(tf.square(           (pred_fake_Y                           )), name='loss_fake_Y')
				
				d_loss_pos_Y = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
						logits=pred_real_Y, 
						labels=tf.ones_like(pred_real_Y)), name='loss_real_Y')
				d_loss_neg_Y = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
						logits=pred_fake_Y, 
						labels=tf.zeros_like(pred_fake_Y)), name='loss_fake_Y')

				self.dY_loss  = tf.add(.5 * d_loss_pos_Y, .5 * d_loss_neg_Y, name='dY_loss')

				score_real_Y = tf.sigmoid(pred_real_Y)
				score_fake_Y = tf.sigmoid(pred_fake_Y)

				tf.summary.histogram('score_real_Y', score_real_Y)
				tf.summary.histogram('score_fake_Y', score_fake_Y)

				d_pos_acc_Y = tf.reduce_mean(tf.cast(score_real_Y > 0.5, tf.float32), name='accuracy_real_Y')
				d_neg_acc_Y = tf.reduce_mean(tf.cast(score_fake_Y < 0.5, tf.float32), name='accuracy_fake_Y')
				
				dY_accuracy   = tf.add(.5 * d_pos_acc_Y,  .5 * d_neg_acc_Y,  name='dY_accuracy')

				###############################################################################################
				# d_loss_pos_X = tf.reduce_mean(tf.square(tf.subtract(pred_real_X, tf.ones_like(pred_real_X))), name='loss_real_X')
				# d_loss_neg_X = tf.reduce_mean(tf.square(           (pred_fake_X                           )), name='loss_fake_X')
				
				d_loss_pos_X = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
						logits=pred_real_X, 
						labels=tf.ones_like(pred_real_X)), name='loss_real_X')
				d_loss_neg_X = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
						logits=pred_fake_X, 
						labels=tf.zeros_like(pred_fake_X)), name='loss_fake_X')
				self.dX_loss  = tf.add(.5 * d_loss_pos_X, .5 * d_loss_neg_X, name='dX_loss')
				
				score_real_X = tf.sigmoid(pred_real_X)
				score_fake_X = tf.sigmoid(pred_fake_X)

				tf.summary.histogram('score_real_X', score_real_X)
				tf.summary.histogram('score_fake_X', score_fake_X)
				
				d_pos_acc_X = tf.reduce_mean(tf.cast(score_real_X > 0.5, tf.float32), name='accuracy_real_X')
				d_neg_acc_X = tf.reduce_mean(tf.cast(score_fake_X < 0.5, tf.float32), name='accuracy_fake_X')
				
				dX_accuracy   = tf.add(.5 * d_pos_acc_X,  .5 * d_neg_acc_X,  name='dX_accuracy')
		
			
			###############################################################################################
			with tf.name_scope('generator'):
				###############################################################################################
				self.gG_loss   = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
											   logits=pred_fake_Y, 
											   labels=tf.ones_like(pred_fake_Y)), name='gG_loss')
						
				# Consistency loss
				self.cX_loss   = tf.reduce_mean(tf.abs(X__ - X), name='cX_loss')
				gG_accuracy    = tf.reduce_mean(tf.cast(score_fake_Y > 0.5, tf.float32), name='gG_accuracy')
				###############################################################################################
				self.gF_loss   = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
											   logits=pred_fake_X, 
											   labels=tf.ones_like(pred_fake_X)), name='gF_loss')
						
			
				self.cY_loss   = tf.reduce_mean(tf.abs(Y__ - Y), name='cY_loss')
				gF_accuracy    = tf.reduce_mean(tf.cast(score_fake_X > 0.5, tf.float32), name='gF_accuracy')
				###############################################################################################
				self.gG_loss   = tf.add(self.gG_loss, LAMBDA*self.cY_loss, name='gG_total')  # Add the l1 loss
				self.gF_loss   = tf.add(self.gF_loss, LAMBDA*self.cX_loss, name='gF_total')  # Add the l1 loss
		###############################################################################################

		###############################################################################################
		# tensorboard visualization            
		#viz_total = (tf.concat([image,Y_,X__,label,X_,Y__], 2) + 1.0) * 128.0
		#viz_total = tf.cast(tf.clip_by_value(viz_total, 0, 255), tf.uint8, name='viz_total') 
		# Take one slice
		#tf.summary.image('image,Yh,Xhh,label,Xh,Yhh', viz_total[:,:,:,0:1], max_outputs=2)
		viz_cycleG = tf.concat([image,Y_,X__], 2, name='viz_cycleG')
		viz_cycleF = tf.concat([label,X_,Y__], 2, name='viz_cycleF')
		viz_concat = tf.concat([viz_cycleG, viz_cycleF], 1, name='viz_concat')
		tf.summary.image('concatenation', viz_concat[:,:,:,0:1], max_outputs=30)

		viz_genG = (Y_ / 2.0 + 0.5) * 255.0
		viz_genG = tf.cast(tf.clip_by_value(viz_genG, 0, 255), tf.uint8, name='viz_genG') 
		tf.summary.image('genG', viz_genG[:,:,:,0:1], max_outputs=30)

		viz_genF = (X_ / 2.0 + 0.5) * 255.0
		viz_genF = tf.cast(tf.clip_by_value(viz_genF, 0, 255), tf.uint8, name='viz_genF') 
		tf.summary.image('genF', viz_genF[:,:,:,0:1], max_outputs=30)


		# Collect all the variable loss 
		add_moving_summary(self.gG_loss, self.dY_loss, 
						   self.gF_loss, self.dX_loss, 
						   gG_accuracy, dY_accuracy, 
						   gF_accuracy, dX_accuracy, 
						   self.cX_loss, self.cY_loss,
						   )
		self.collect_variables()
		
		

class CycleGANTrainer(FeedfreeTrainerBase):
	def __init__(self, config):
		self._input_method = QueueInput(config.dataflow)
		super(CycleGANTrainer, self).__init__(config)

	def _setup(self):
		super(CycleGANTrainer, self)._setup()
		self.build_train_tower()
		opt = self.model.get_optimizer()

		# by default, run one d_min after one g_min
		self.gG_min = opt.minimize(self.model.gG_loss, var_list=self.model.gG_vars, name='gG_op')
		with tf.control_dependencies([self.gG_min]):
			self.dY_min = opt.minimize(self.model.dY_loss, var_list=self.model.dY_vars, name='dY_op')
		# self.trainY_op = self.dY_min

		self.gF_min = opt.minimize(self.model.gF_loss, var_list=self.model.gF_vars, name='gF_op')
		with tf.control_dependencies([self.gF_min]):
			self.dX_min = opt.minimize(self.model.dX_loss, var_list=self.model.dX_vars, name='dX_op')
		# self.trainX_op = self.dX_min
		# self.train_op = [self.trainX_op, self.trainY_op]
		self.train_op = [self.gG_min, self.dY_min, self.gF_min, self.dX_min]


class ImagePairData(RNGDataFlow):
	def __init__(self, imageDir, labelDir, size, dtype='float32'):
		"""
		Args:
			shapes (list): a list of lists/tuples. Shapes of each component.
			size (int): size of this DataFlow.
			random (bool): whether to randomly generate data every iteration.
				Note that merely generating the data could sometimes be time-consuming!
			dtype (str): data type.
		"""
		# super(FakeData, self).__init__()

		self.dtype  = dtype
		self.imageDir = imageDir
		self.labelDir = labelDir
		self._size  = size

	def size(self):
		return self._size

	def reset_state(self):
		self.rng = get_rng(self)   
		pass

	def get_data(self):
		self.reset_state()
		images = glob.glob(self.imageDir + '/*.png')
		labels = glob.glob(self.labelDir + '/*.png')
		# print images
		# print labels
		# EPOCH_SIZE = len(images)
		# EPOCH_SIZE = 30
		for k in range(self._size):
			from random import randrange
			rand_index_image = randrange(0, len(images))
			rand_index_label = randrange(0, len(labels))
			image = skimage.io.imread(images[rand_index_image])
			label = skimage.io.imread(labels[rand_index_label])
			#TODO: augmentation here

			image = np.expand_dims(image, axis=0)
			label = np.expand_dims(label, axis=0)
			image = np.expand_dims(image, axis=-1)
			label = np.expand_dims(label, axis=-1)
			yield [image.astype(np.uint8), label.astype(np.uint8)]

def get_data():
	ds_train = ImagePairData(args.imageDir, args.labelDir, 60)
	ds_valid = ImagePairData(args.imageDir, args.labelDir, 60)
	PrintData(ds_train, num=3)
	PrintData(ds_valid, num=3)
	return ds_train, ds_valid

def get_config():
    logger.auto_set_dir()
    # dataset = get_data()
    ds_train, ds_valid = get_data()
    ds_train.reset_state()
    ds_valid.reset_state() 
    #print ds_train.size()
    return TrainConfig(
        dataflow=ds_train,
        callbacks=[
            PeriodicTrigger(ModelSaver(), every_k_epochs=100),
            ScheduledHyperParamSetter('learning_rate', [(200, 1e-4)]),
            ],
        model=CycleGANModel(),
        steps_per_epoch=ds_train.size(),
        max_epoch=4000,
    )

if __name__ == '__main__':
	#https://docs.python.org/3/library/argparse.html
	parser = argparse.ArgumentParser()
	#
	parser.add_argument('--gpu', 		help='comma separated list of GPU(s) to use.')
	parser.add_argument('--load', 		help='load models for continue train or predict')
	parser.add_argument('--sample',		help='run sampling one instance', 
										action='store_true')
	parser.add_argument('--imageDir',   help='Image directory', required=True)
	parser.add_argument('--labelDir',   help='Label directory', required=True)
	parser.add_argument('-db', '--debug', type=int, default=0) # Debug one particular function in main flow
	global args
	args = parser.parse_args() # Create an object of parser
	if args.gpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	if args.sample:
		#sample(args.data, args.load)
		sample(args.load)
	else:
		config = get_config()
		if args.load:
			config.session_init = SaverRestore(args.load)
		CycleGANTrainer(config).train()

	if args.debug:
		get_data()
		pass 