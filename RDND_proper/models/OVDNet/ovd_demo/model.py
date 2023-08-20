import tensorflow as tf


def get_shape(x, i):
    return x.get_shape().as_list()[i]
    
    
def weight_variable(shape, stddev=0.02, name = 'weight'):#stddev=0.1, name this explicitly
#~ def weight_variable(shape, stddev=0.1, name = 'weight'):#stddev=0.1, name this explicitly
        
    w = tf.get_variable(name, shape, initializer = tf.random_normal_initializer(stddev=stddev))
    return w


def bias_variable(shape, name):#explicit naming    
	b = tf.get_variable(name, initializer = tf.zeros(shape))
	return b



def conv2d(x, W, stride = 1):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')



def conv2d_transpose(x, w, output_shape, stride = 2):
    return tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=[1, stride, stride, 1], padding='SAME')
        

def bn(x):
	net = x
	out_channels = get_shape(net, 3)
	mean, var = tf.nn.moments(net, axes=[0,1,2])
	beta = bias_variable([out_channels], name="beta") 
	gamma = weight_variable([out_channels], name="gamma")

	net = tf.nn.batch_normalization(net, mean, var, beta, gamma, 0.001)#batch
	
	return net


def conv_bn(x, filter_shape):#cvpr16_variant
	net = x;
	net = tf.nn.conv2d(net, weight_variable(filter_shape, name = "weight"), strides=[1, 1, 1, 1], padding="SAME")#conv

	out_channels = filter_shape[3]
	mean, var = tf.nn.moments(net, axes=[0,1,2])
	beta = bias_variable([out_channels], name="beta") 
	gamma = weight_variable([out_channels], name="gamma")


	net = tf.nn.batch_normalization(net, mean, var, beta, gamma, 0.001)#batch
	return net
	
def resnet_block(x, out_channel, filter_size = 3):
	x_channel = x.get_shape().as_list()[3]
	
	with tf.variable_scope("conv_bn_relu"):
		net = conv_bn(x, filter_shape=[filter_size, filter_size, out_channel, out_channel])
		net = tf.nn.relu(net)

	with tf.variable_scope("conv_bn"):
		net = conv_bn(net, filter_shape=[filter_size, filter_size, out_channel, out_channel])

	net = net + x
	tf.nn.relu(net)

	return net



def dynamic_fusion(x, h, filter_size = 5):

	n_channel = get_shape(x, 3)
	
	t = tf.concat([x, h], 3)
	
	similarity = tf.nn.conv2d(t, weight_variable([filter_size, filter_size, n_channel*2, n_channel], name = "wt"), strides=[1, 1, 1, 1], padding='VALID')
	epsilon = bias_variable([1], name = 'bias_epsilon')
	
	
	
	alpha = 2*tf.abs(tf.sigmoid(similarity) - 0.5) + epsilon#simulate tanh, slightly fast than using tanh directly
	alpha = tf.clip_by_value(alpha, 0, 1)
	
	hflt_filter_size = filter_size//2
	alpha = tf.pad(alpha-1, [[0, 0], [hflt_filter_size, hflt_filter_size], [hflt_filter_size, hflt_filter_size], [0, 0]], "CONSTANT") + 1	
	
	y = alpha*x + (1-alpha)*h
	
	return y, alpha
	
	
	
def ovd(X, F, H, net_channel = 64):
	#~ net = X
		
	H_curr = []
	A_curr = []
	with tf.variable_scope("encoding"):		
		
		with tf.variable_scope("conv1"):
			filter_size = 5
			net_X = conv2d(X, weight_variable([filter_size, filter_size, get_shape(X, 3), net_channel]))
			net_X = tf.nn.relu(net_X)
			
		with tf.variable_scope("conv2"):
			filter_size = 3
			net_X = conv2d(net_X, weight_variable([filter_size, filter_size, get_shape(net_X, 3), net_channel//2]), stride = 2)
			net_X = tf.nn.relu(net_X)
			
			
		
		net = tf.concat([net_X, F], 3)
		f0 = net
		
		filter_size = 3
		num_resnet_layers = 8
		for i in range (num_resnet_layers):
			with tf.variable_scope('resnet_block%d' % (i+1)):
				net = resnet_block(net, net_channel)
				
				if i == 3:
					(net, alpha) = dynamic_fusion(net, H[0])
					h = tf.expand_dims(net, axis=0)
					H_curr = h
					
	with tf.variable_scope("feat_out"):
		F = conv2d(net, weight_variable([filter_size, filter_size, get_shape(net, 3), net_channel//2], name = 'conv_F'))
		F = tf.nn.relu(F)

	with tf.variable_scope("img_out"):
		filter_size = 4
		shape = [get_shape(X, 0), get_shape(X, 1), get_shape(X, 2), net_channel]
		Y = conv2d_transpose(net, weight_variable([filter_size, filter_size, net_channel, net_channel], name = "deconv"), shape, stride = 2)
		Y = tf.nn.relu(Y)
		
		filter_size = 3
		Y = conv2d(Y, weight_variable([filter_size, filter_size, get_shape(Y, 3), 3], name = 'conv'))
		
		
	return Y, F, H_curr, A_curr
