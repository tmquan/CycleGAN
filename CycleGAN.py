#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: CycleGAN.py
# Author: tmquan

import cv2
import numpy as numpy
import glob
import pickle
import os
import sys
import argparse

from tensorpack import *
from tensorpack.utils.viz import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
import tensorpack.tfutils.symbolic_functions as symbolic_functions
from GAN import GANTrainer, GANModelDesc

# Global variables
BATCH_SIZE = 1
SRC_CH = 3
DST_CH = 3
LAMBDA = 1e+2
BETA   = 1e-2
NF	   = 64 # number of filter in generator F and G
SHAPE  = 1024

class Model(GANModelDesc):
	def generator_F(self, imgs):
		pass

	def generator_G(self, imgs):
		pass

	@auto_reuse_variable_scope
	def discriminator_Dx(self, inputs, outputs):
		pass

	@auto_reuse_variable_scope
	def discriminator_Dy(self, inputs, outputs):
		pass

	def _get_input(self):
		pass
	def _get_optimizer(self):
		pass
	def _build_graph(self, inputs):
		pass

if __name__ = '__main__':
	#https://docs.python.org/3/library/argparse.html
	parser = argparse.ArgumentParser()
	#
	parser.add_argument('--gpu', 		help='comma separated list of GPU(s) to use.')
	parser.add_argument('--load', 		help='load models for continue train or predict')
	parser.add_argument('--sample',		help='run sampling one instance', 
										action='store_true')
	 parser.add_argument('-db', '--debug', type=int, default=0) # Debug one particular function in main flow
	global args
	args = parser.args() # Create an object of parser
	if args.gpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
		
    if args.sample:
        #sample(args.data, args.load)
        sample(args.load)
    else:
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        GANTrainer(config).train()

	if args.debug:
        pass 