from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from six.moves import xrange
from ops import *
from utils import *

def conv_out_size_same(size, stride):
	return int(math.ceil(float(size) / float(stride)))

def fdepth(ii):
	return 2 ** (10 - ii)

def lerp_clip(a, b, t):
	return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)

def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
	if fan_in is None: fan_in = np.prod(shape[:-1])
	std = gain / np.sqrt(fan_in) # He init
	if use_wscale:
		wscale = tf.constant(np.float32(std), name='wscale')
		return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
	else:
		return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))

def nvconv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
	assert kernel >= 1 and kernel % 2 == 1
	w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
	w = tf.cast(w, x.dtype)
	return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW')

#----------------------------------------------------------------------------
# Apply bias to the given activation tensor.

def apply_bias(x):
	b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros())
	b = tf.cast(b, x.dtype)
	if len(x.shape) == 2:
		return x + b
	else:
		return tf.nn.bias_add(x, b, data_format='NCHW')


# def torgb(x, res): # res = 2..resolution_log2
def torgb(x): # res = 2..resolution_log2
	# lod = resolution_log2 - res
	# with tf.variable_scope('ToRGB_lod%d' % lod):
	return apply_bias(nvconv2d(x, fmaps=1, kernel=1, gain=1, use_wscale=True))

# def torgb(x, res): # res = 2..resolution_log2
#         lod = resolution_log2 - res
#         with tf.variable_scope('ToRGB_lod%d' % lod):
# 			return apply_bias(conv2d(x, fmaps=num_channels, kernel=1, gain=1, use_wscale=use_wscale))

class DCGAN(object):
	def __init__(self, sess, input_height=108, input_width=108, crop=True,
				 batch_size=64, sample_num = 64, output_height=64, output_width=64,
				 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
				 gfc_dim=1024, dfc_dim=1024, c_dim=1, dataset_name='default',
				 input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None,
				 grow=8):
		"""

		Args:
			sess: TensorFlow session
			batch_size: The size of batch. Should be specified before training.
			y_dim: (optional) Dimension of dim for y. [None]
			z_dim: (optional) Dimension of dim for Z. [100]
			gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
			df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
			gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
			dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
			c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
		"""
		self.sess = sess
		self.crop = crop

		self.batch_size = batch_size
		self.sample_num = sample_num
		self.grow = grow

		self.imsize = input_height
		self.stacks = int(np.log2(self.imsize)) - 1 # lowest supported imsize=4

		self.input_height = input_height
		self.input_width = input_width
		self.output_height = output_height
		self.output_width = output_width

		self.y_dim = y_dim
		self.z_dim = z_dim

		self.gfc_dim = gfc_dim
		self.dfc_dim = dfc_dim

		self.d_bn = []
		for ii in range(self.stacks + 1):
			self.d_bn.append(batch_norm(name='d_bn%d'%ii))

		self.g_bn = []
		for ii in range(self.stacks + 1):
			self.g_bn.append(batch_norm(name='g_bn%d'%ii))


		self.dataset_name = dataset_name
		self.input_fname_pattern = input_fname_pattern
		self.checkpoint_dir = checkpoint_dir

		if self.dataset_name == 'mnist':
			self.data_X, self.data_y = self.load_mnist()
			self.c_dim = self.data_X[0].shape[-1]
		else:
			imgpath = os.path.join("../data", self.dataset_name, self.input_fname_pattern)
			self.data = glob(imgpath)
			imreadImg = imread(self.data[0])
			self.c_dim = 1

		self.grayscale = (self.c_dim == 1)

		print('SAVE DIR:', self.model_dir)
		print('LOAD DIR:', self.model_growcond_dir)
		try:
			input()
		except:
			pass
		self.build_model()

	def build_model(self):
		if self.y_dim:
			self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
		else:
			self.y = None

		image_dims = [self.imsize, self.imsize, 1]

		# input params
		self.inputs = tf.placeholder(
			tf.float32, [self.batch_size] + image_dims, name='real_images')

		self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
		self.z_sum = histogram_summary("z", self.z)

		# Gen and discrim
		self.G                  = self.generator(self.z, self.y)
		self.D, self.D_logits   = self.discriminator(self.inputs, self.y, reuse=False)
		self.sampler            = self.sampler(self.z, self.y)
		self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)

		# Loss calculations
		self.d_sum = histogram_summary("d", self.D)
		self.d__sum = histogram_summary("d_", self.D_)
		self.G_sum = image_summary("G", self.G)

		def sigmoid_cross_entropy_with_logits(x, y):
			try:
				return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
			except:
				return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

		self.d_loss_real = tf.reduce_mean(
			sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
		self.d_loss_fake = tf.reduce_mean(
			sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
		self.g_loss = tf.reduce_mean(
			sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

		self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
		self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

		self.d_loss = self.d_loss_real + self.d_loss_fake

		self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
		self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

		t_vars = tf.trainable_variables()

		self.d_vars = [var for var in t_vars if 'd_' in var.name]
		self.g_vars = [var for var in t_vars if 'g_' in var.name]


		if self.grow is not None:
			current_vars = slim.get_variables_to_restore()
			prev_vars = []
			ignored_vars = []
			for var in current_vars:
				if '_h' in str(var):
					parts = str(var).split('/')[1]
					parts = parts.split('_')[1]
					hvar = int(parts.replace('h', ''))
					if 2**(hvar + 1) >= self.grow:
						ignored_vars.append(var)
						continue
				elif '_b' in str(var):
					parts = str(var).split('/')[1]
					parts = parts.split('_')[1]
					hvar = int(parts.replace('bn', ''))
					if 2**(hvar + 1) >= self.grow:
						ignored_vars.append(var)
						continue
				prev_vars.append(var)
			self.load_vars = prev_vars

			print('These nodes will be ignored:')
			for var in ignored_vars:
				print('    ', var.name)
			print('These nodes will be loaded:')
			for var in prev_vars:
				print('    ', var.name)
			# try:
			# 	input()
			# except:
			# 	pass
		# 	self.saver = tf.train.Saver(prev_vars)
		# else:
		# 	self.saver = tf.train.Saver()

	def train(self, config):
		d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
							.minimize(self.d_loss, var_list=self.d_vars)
		g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
							.minimize(self.g_loss, var_list=self.g_vars)
		try:
			tf.global_variables_initializer().run()
		except:
			tf.initialize_all_variables().run()

		self.g_sum = merge_summary([self.z_sum, self.d__sum,
			self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
		self.d_sum = merge_summary(
				[self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
		self.writer = SummaryWriter("./logs/tissue%d" % self.imsize, self.sess.graph)

		# Sample from the random Z distribuion
		sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))

		if config.dataset == 'mnist':
			sample_inputs = self.data_X[0:self.sample_num]
			sample_labels = self.data_y[0:self.sample_num]
		else:
			sample_files = self.data[0:self.sample_num]
			sample = [
					get_image(sample_file,
										input_height=self.input_height,
										input_width=self.input_width,
										resize_height=self.output_height,
										resize_width=self.output_width,
										crop=self.crop,
										grayscale=self.grayscale) for sample_file in sample_files]
			if (self.grayscale):
				sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
			else:
				sample_inputs = np.array(sample).astype(np.float32)

		counter = 1
		start_time = time.time()
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			counter = checkpoint_counter
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")

		for epoch in xrange(config.epoch):
			if config.dataset == 'mnist':
				batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size
			else:
				self.data = glob(os.path.join(
					"../data", config.dataset, self.input_fname_pattern))
				batch_idxs = min(len(self.data), config.train_size) // config.batch_size

			for idx in xrange(0, batch_idxs):
				if config.dataset == 'mnist':
					batch_images = self.data_X[idx*config.batch_size:(idx+1)*config.batch_size]
					batch_labels = self.data_y[idx*config.batch_size:(idx+1)*config.batch_size]
				else:
					batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
					batch = [
							get_image(batch_file,
												input_height=self.input_height,
												input_width=self.input_width,
												resize_height=self.output_height,
												resize_width=self.output_width,
												crop=self.crop,
												grayscale=self.grayscale) for batch_file in batch_files]
					if self.grayscale:
						# FIXME: network only works on grayscale...
						batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
					else:
						batch_images = np.array(batch).astype(np.float32)

				batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
							.astype(np.float32)

				if config.dataset == 'mnist':
					# Update D network
					_, summary_str = self.sess.run([d_optim, self.d_sum],
						feed_dict={
							self.inputs: batch_images,
							self.z: batch_z,
							self.y:batch_labels,
						})
					self.writer.add_summary(summary_str, counter)

					# Update G network
					_, summary_str = self.sess.run([g_optim, self.g_sum],
						feed_dict={
							self.z: batch_z,
							self.y:batch_labels,
						})
					self.writer.add_summary(summary_str, counter)

					# Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
					_, summary_str = self.sess.run([g_optim, self.g_sum],
						feed_dict={ self.z: batch_z, self.y:batch_labels })
					self.writer.add_summary(summary_str, counter)

					errD_fake = self.d_loss_fake.eval({
							self.z: batch_z,
							self.y:batch_labels
					})
					errD_real = self.d_loss_real.eval({
							self.inputs: batch_images,
							self.y:batch_labels
					})
					errG = self.g_loss.eval({
							self.z: batch_z,
							self.y: batch_labels
					})
				else:
					# Update D network
					_, summary_str = self.sess.run([d_optim, self.d_sum],
						feed_dict={ self.inputs: batch_images, self.z: batch_z })
					self.writer.add_summary(summary_str, counter)

					# Update G network
					_, summary_str = self.sess.run([g_optim, self.g_sum],
						feed_dict={ self.z: batch_z })
					self.writer.add_summary(summary_str, counter)

					# Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
					_, summary_str = self.sess.run([g_optim, self.g_sum],
						feed_dict={ self.z: batch_z })
					self.writer.add_summary(summary_str, counter)

					errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
					errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
					errG = self.g_loss.eval({self.z: batch_z})

				counter += 1
				print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
					% (epoch, idx, batch_idxs,
						time.time() - start_time, errD_fake+errD_real, errG))

				if np.mod(counter, 100) == 1:
					if config.dataset == 'mnist':
						samples, d_loss, g_loss = self.sess.run(
							[self.sampler, self.d_loss, self.g_loss],
							feed_dict={
									self.z: sample_z,
									self.inputs: sample_inputs,
									self.y:sample_labels,
							}
						)
						save_images(samples, image_manifold_size(samples.shape[0]),
									'./{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
						print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
					else:
						samples, d_loss, g_loss = self.sess.run(
							[self.sampler, self.d_loss, self.g_loss],
							feed_dict={
									self.z: sample_z,
									self.inputs: sample_inputs,
							},
						)
						save_images(samples, image_manifold_size(samples.shape[0]),
									'./results/{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
						print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

				if np.mod(counter, 500) == 2:
					self.save(config.checkpoint_dir, counter)



	def discriminator(self, image, y=None, reuse=False):
		print('DISCRIM', image.get_shape())
		with tf.variable_scope("discriminator") as scope:
			if reuse: scope.reuse_variables()

			prevlayer = conv2d(image, fdepth(0), name='d_h%d_conv' % 0) # first layer
			for ii in range(1, self.stacks): # Other (self.stacks - 1) layers
				convdim = fdepth(ii)
				convop = conv2d(prevlayer, convdim, name='d_h%d_conv' % ii)
				convop = self.d_bn[ii](convop)
				prevlayer = lrelu(convop)
			hf = linear(tf.reshape(prevlayer, [self.batch_size, -1]), 1, 'd_h%d_lin' % self.stacks)

			return tf.nn.sigmoid(hf), hf

	def generator(self, z, y=None):
		with tf.variable_scope("generator") as scope:
			imsize = self.imsize

			minsize = 4
			self.z_, self.h0_w, self.h0_b = linear(z, minsize * minsize * fdepth(0), 'g_h0_lin', with_w=True)
			self.h0 = tf.reshape(self.z_, [-1, minsize, minsize, fdepth(0)])
			h0 = tf.nn.relu(self.g_bn[0](self.h0))

			prevlayer = h0
			for ii in range(1, self.stacks):
				outres = 2 ** (ii + 2) # r(1) = 2 ** (1 + 2) = 8
				hi, _, _ = deconv2d(
					prevlayer,
					[self.batch_size, outres, outres, fdepth(ii)],
					name='g_h%d'%ii,
					with_w=True)
				# if ii != self.stacks - 1: # not last, batch norm
				hi = tf.nn.relu(self.g_bn[ii](hi))
				prevlayer = hi


			prevlayer = deconv2d(
				prevlayer,
				[self.batch_size, imsize, imsize, 1],
				k_h=1, k_w=1, d_h=1, d_w=1,
				name='g_h%d'%self.stacks)
			# assert prevlayer.dtype == tf.as_dtype('float32')
			return tf.nn.tanh(prevlayer)
			# return torgb(prevlayer)

	def sampler(self, z, y=None):
		with tf.variable_scope("generator") as scope:
			scope.reuse_variables()


			imsize = self.imsize

			minsize = 4
			self.z_, self.h0_w, self.h0_b = linear(z, minsize * minsize * fdepth(0), 'g_h0_lin', with_w=True)
			self.h0 = tf.reshape(self.z_, [-1, minsize, minsize, fdepth(0)])
			h0 = tf.nn.relu(self.g_bn[0](self.h0, train=False))

			prevlayer = h0
			for ii in range(1, self.stacks):
				outres = 2 ** (ii + 2) # r(1) = 2 ** (1 + 2) = 8
				hi, _, _ = deconv2d(
					prevlayer,
					[self.batch_size, outres, outres, fdepth(ii)],
					name='g_h%d'%ii,
					with_w=True)
				# if ii != self.stacks - 1: # not last, batch norm
				hi = tf.nn.relu(self.g_bn[ii](hi, train=False))
				prevlayer = hi

			prevlayer = deconv2d(
				prevlayer,
				[self.batch_size, imsize, imsize, 1],
				k_h=1, k_w=1, d_h=1, d_w=1,
				name='g_h%d'%self.stacks)
			return tf.nn.tanh(prevlayer)
			# return torgb(prevlayer)

	def load_mnist(self):
		data_dir = os.path.join("../data", self.dataset_name)

		fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
		loaded = np.fromfile(file=fd,dtype=np.uint8)
		trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

		fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
		loaded = np.fromfile(file=fd,dtype=np.uint8)
		trY = loaded[8:].reshape((60000)).astype(np.float)

		fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
		loaded = np.fromfile(file=fd,dtype=np.uint8)
		teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

		fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
		loaded = np.fromfile(file=fd,dtype=np.uint8)
		teY = loaded[8:].reshape((10000)).astype(np.float)

		trY = np.asarray(trY)
		teY = np.asarray(teY)

		X = np.concatenate((trX, teX), axis=0)
		y = np.concatenate((trY, teY), axis=0).astype(np.int)

		seed = 547
		np.random.seed(seed)
		np.random.shuffle(X)
		np.random.seed(seed)
		np.random.shuffle(y)

		y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
		for i, label in enumerate(y):
			y_vec[i,y[i]] = 1.0

		return X/255.,y_vec

	@property
	def model_growcond_dir(self):
		if self.grow is not None:
			path = "{}_{}_{}_{}".format(
				'tissue%d/default'%self.grow, self.batch_size,
				self.grow, self.grow)
		else:
			path = "{}_{}_{}_{}".format(
					self.dataset_name, self.batch_size,
					self.output_height, self.output_width)
		return path

	@property
	def model_dir(self):
		path = "{}_{}_{}_{}".format(
				self.dataset_name, self.batch_size,
				self.output_height, self.output_width)
		return path

	def save(self, checkpoint_dir, step):
		print(' [!] Saving model: %d' % step)
		model_name = "DCGAN.model"
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		tf.train.Saver().save(self.sess,
						os.path.join(checkpoint_dir, model_name),
						global_step=step)

	def load(self, checkpoint_dir):
		import re
		print(" [*] Reading checkpoints...")
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_growcond_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

			restore_path = os.path.join(checkpoint_dir, ckpt_name)
			if self.grow is not None:
				tf.train.Saver(self.load_vars).restore(self.sess, restore_path)
			else:
				tf.train.Saver().restore(self.sess, restore_path)
			# self.saver
			counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
			print(" [*] Success to read {}".format(ckpt_name))
			return True, counter
		else:
			print(" [*] Failed to find a checkpoint")
			return False, 0
