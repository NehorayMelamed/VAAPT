import tensorflow as tf
import threading
import numpy as np
from scipy import misc
from scipy import ndimage
import os
import time




def img_crop(img, img_height, img_width, upper_left_x = None, upper_left_y = None):  # 2D img from misc

    if (upper_left_x is not None) and (upper_left_y is not None):
        out_img = img[upper_left_y:upper_left_y + img_height, upper_left_x:upper_left_x + img_width]
    else:        
        pos_y = (img.shape[0] - img_height) // 2;
        pos_x = (img.shape[1] - img_width) // 2;
        out_img = img[pos_y:pos_y + img_height, pos_x:pos_x + img_width]
    # print(img.shape, out_img.shape, pos_x , pos_y)
    return out_img


	
	
def load_gopro_set(img_height, img_width, test_dir_info):
	
	seq = test_dir_info

	start_frame = seq[1]
	end_frame = seq[2]
	ext  = seq[3]
	
	sequence_length = end_frame - start_frame + 1
	#~ print 'sequence_length:%d'%sequence_length
	
	seq_B = []
	seq_S = []
	for i in range(sequence_length):
		
		sharp_img_name = seq[0] + 'S_%04d' % (start_frame + i) + ext
		blurry_img_name = seq[0] + 'B_%04d' % (start_frame + i) + ext
			
		tmp_B = misc.imread(blurry_img_name)
		tmp_B = img_crop(tmp_B, img_height, img_width)
		tmp_S = misc.imread(sharp_img_name)
		tmp_S = img_crop(tmp_S, img_height, img_width)
		
		#~ print tmp_B.shape
		#~ print tmp_S.shape
		
		tmp_B = np.reshape(tmp_B, [1, img_height, img_width, 3])
		tmp_S = np.reshape(tmp_S, [1, img_height, img_width, 3])
		
		seq_B.append(tmp_B)
		seq_S.append(tmp_S)
		
	return seq_B, seq_S
	

	
    
    
def save_img(img, file_name):
	
	#~ img = 255*np.clip(img, 0, 1)
	
	img = 255*img
	img = np.clip(np.round_(img), 0, 255)
	[H, W, C] = img.shape
	#~ print img.shape
		
	#~ if C == 1:
		#~ img = np.reshape(img, [H, W])
	#~ else:
		#~ img = np.reshape(img, [H, W, C])
	
	img = misc.toimage(img, cmin=0, cmax=255)
	misc.imsave(file_name, img)

	return
	
	
