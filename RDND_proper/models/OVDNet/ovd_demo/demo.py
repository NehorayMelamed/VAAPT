#!/usr/bin/python
__author__ = 'thkim'
import tensorflow as tf
import models.OVDNet.ovd_demo.model

import models.OVDNet.ovd_demo.io_data
import time
import numpy as np
import sys, getopt
import os
import math
from scipy import misc
# from skimage.measure import structural_similarity as ssim
#~ import config


num_input_imgs = 3
batch_size = 1



test_dir_info = ['./in/', 1, 10, '.png']
img_width = 640#960
img_height = 480#540



save_dir = './out/'
# print'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'
# print(save_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# print'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'

def count_params():
    "print number of trainable variables"
    size = lambda v: reduce(lambda x, y: x*y, v.get_shape().as_list())
    n = sum(size(v) for v in tf.trainable_variables())
    print "@@@@@@@@@@@@@@@@@@@@@@Model size: %dK" % (n/1000,)


def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    
    

def preprocess(img):
	img = tf.cast(img, tf.float32)
	img = img/255

	return img


net_channel = 64

B_cur = tf.placeholder(tf.float32, [1, img_height, img_width, 3*num_input_imgs])

F = tf.placeholder(tf.float32, [batch_size, img_height//2, img_width//2, net_channel//2])
H = tf.placeholder(tf.float32, [1, batch_size, img_height//2, img_width//2, net_channel])

with tf.variable_scope("model") as scope:		
	(L, F_o, H_o, A) = model.ovd(B_cur, F, H, net_channel)
	
	
	count_params()
		

model_path = './model.ckpt'
print model_path		

saver = tf.train.Saver(max_to_keep = None)
init = tf.global_variables_initializer()

with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8)) as sess:
	sess.run(init)		
	
	
	saver.restore(sess, model_path)
	
	
	psnr_sum = 0

	(B_seq, S_seq) = io_data.load_gopro_set(img_height, img_width, test_dir_info)
	sequence_length = len(B_seq)
	
	
	
	for i in range(sequence_length - num_input_imgs + 1):
		if i == 0:
			L_prediction = B_seq[num_input_imgs//2]
			
			F_init = np.zeros([batch_size, img_height//2, img_width//2, net_channel//2], dtype=np.float32)				
			H_init = np.zeros([1, batch_size, img_height//2, img_width//2, net_channel], dtype=np.float32)
			feed_dict={B_cur: np.concatenate(B_seq[i:i + num_input_imgs], 3)/255., F:F_init, H:H_init}
		else:
			feed_dict={B_cur: np.concatenate(B_seq[i:i + num_input_imgs], 3)/255., F:f, H:h}
			
		b = B_seq[i+num_input_imgs//2]/255.
		(l, f, h) = sess.run([L, F_o, H_o], feed_dict = feed_dict)
	
		
		if i > 2:#skip first few results		
			io_data.save_img(l[0], save_dir + 'out_%04d.png'%(i+1))
			io_data.save_img(b[0], save_dir + 'in_%04d.png'%(i+1))
			
			#remove below if gt is not available
			s = S_seq[i+num_input_imgs//2]				
			psnr_val = psnr(255*l, s)
			psnr_sum += psnr_val
			print psnr_val, psnr_sum
		
			
			
