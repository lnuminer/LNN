# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 07:53:40 2017

@author: lenov
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python import debug as tf_debug # it didn't work if import tensorflow.python.debug as tf_debug
import numpy as np
import time
import gzip
from PIL import Image
from pylab import *

IMAGE_SIZE = 28
RECEPTION_FIELD = 4
COMPETITION_FIELD=3
BATCH_SIZE = 100
MAX_LOOP = 100
LEARNING_RATE = 1
PIXEL_DEPTH = 255
NUM_IMAGES = 1000
NUM_EPOCH = 100

layer_neuron_num_axis = IMAGE_SIZE-RECEPTION_FIELD+1
layer_neuron_num = layer_neuron_num_axis*layer_neuron_num_axis
input_data_dims = IMAGE_SIZE*IMAGE_SIZE

def test():
    '''
    test 1: under the same context and input, the node in the graph is evaluated only once
    if the target nodes are computed at same time. In the following codes, the node of n4 run once only
    '''
    n1 = tf.Variable([1.0,2,0], name='n1')
    n2 = n1*2
    n4 = tf.assign(n1, n2)
    n3 = n4*2
    n5 = n4*3
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    r1, r2 = sess.run([n3, n5])
    print(r1, r2)
   
def extract_minst_data(filename, num_images):
  """Extract the minst images into a 3D tensor [image index, y, x].

  Values are rescaled from [0, 255] down to [0, 1].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data / PIXEL_DEPTH #image.show will be fallen if pixel is between 0~1
    data = data.reshape(num_images, input_data_dims,1)
  print('Extracting finish!')
  return data

def sharp_minst_data(data):
    print('Sharping Data...')
    shape = data.shape
    image_nums = shape[0]
    data = data.reshape(image_nums, IMAGE_SIZE, IMAGE_SIZE)
    for img_idx in range(image_nums):
        im = data[img_idx]
        for i in range(IMAGE_SIZE-1): #except for the last line
            for j in range(IMAGE_SIZE-1):
                x = abs(im[i,j+1]-im[i,j])
                y = abs(im[i+1,j]-im[i,j])
                im[i,j] = max(x,y)
        for i in range(IMAGE_SIZE):
            im[IMAGE_SIZE-1, i] = 0.0
            im[i, IMAGE_SIZE-1] = 0.0
    data  = data.reshape(image_nums, input_data_dims, 1)
    print('Sharping finish!')
    return data
                
def build_fake_data():
    #fake data
    fake_data = np.ndarray(
          shape=(input_data_dims, 1),
          dtype=np.float32)
    fake_data[:]=np.random.random_sample((input_data_dims,1))
    return fake_data

def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting Label', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  return labels

def dbg_show_weights(weights):
    #weights : neurons * input data dimension
    shape = weights.shape;
    neuron_nums = shape[0]
    neuron_axis_size = int(np.sqrt(shape[0]))
    data_axis_size = int(np.sqrt(shape[1]))
    data = np.array(weights)
    data = data.reshape([neuron_nums, data_axis_size, data_axis_size])
    img_data = np.zeros([neuron_axis_size*data_axis_size, neuron_axis_size*data_axis_size])
    #print('neuron_axis_size:{} data_axis_size:{}'.format(neuron_axis_size, data_axis_size))
    for i in range(neuron_axis_size):
        for j in range(neuron_axis_size):
            x_start, x_end = i*data_axis_size, (i+1)*data_axis_size
            y_start, y_end = j*data_axis_size, (j+1)*data_axis_size
#            print('x_start:{} x_end:{} y_start:{} y_end:{} idx:{}'.format(x_start, x_end,  y_start, y_end,i*neuron_axis_size+j))
#            print(data[0],data[0].shape)
#            print('img shape: ', img_data.shape)
#            print(img_data[x_start:x_end, y_start:y_end])
            img_data[x_start:x_end, y_start:y_end] = data[i*neuron_axis_size+j]
    imshow(img_data)

class FeatureLayer:
    
    def __init__(self, reception_field, competition_field, input_dim_r, input_dim_c, input_data, learn_rate):
        self.reception_field = reception_field
        self.competition_field = competition_field
        self.input_dim_r = input_dim_r
        self.input_dim_c = input_dim_c
        self.layer_neuron_dim_r = input_dim_r-reception_field+1
        self.layer_neuron_dim_c = input_dim_c-reception_field+1
        self.layer_neuron_num = self.layer_neuron_dim_r*self.layer_neuron_dim_c
        self.input_data_dims = input_dim_r*input_dim_c
        self.input_data = input_data
        self.learn_rate = learn_rate
    
    def buildLayer(self):    
        #create a mask matrix for weight
        self.layer_weight_mask = np.zeros([self.layer_neuron_num, self.input_data_dims])
        for i in range(self.layer_neuron_dim_r):
            for j in range(self.layer_neuron_dim_c):
                row = i*self.layer_neuron_dim_c+j
                for k in range(self.reception_field):
                    start = self.input_dim_c*(i+k)+j 
                    end = start+self.reception_field
                    self.layer_weight_mask[row,start:end]=1
        #initialize weights
        self.layer_weights = tf.Variable(tf.random_uniform([self.layer_neuron_num, self.input_data_dims], minval=0.0, maxval=1.0))
        self.layer_weights_mask = tf.multiply(self.layer_weights, self.layer_weight_mask)
        self.layer_weights_norm = tf.nn.l2_normalize(self.layer_weights_mask,1) #confirm ok
        #initial output
        self.layer_previous_output = tf.Variable(tf.zeros(shape=(self.layer_neuron_num,1)))
        #compute output
        self.layer_output = tf.matmul(self.layer_weights_norm, self.input_data)
        self.layer_output_2d = tf.reshape(self.layer_output, shape=(self.layer_neuron_dim_r, self.layer_neuron_dim_c))
        padding = int(self.competition_field/2)
        self.layer_output_padding = tf.pad(self.layer_output_2d, [[padding, padding],[padding,padding]])
        #max competition
        self.layer_output_padding_4d = tf.reshape(self.layer_output_padding, 
                                              shape=(1,self.layer_neuron_dim_r+2*padding,self.layer_neuron_dim_c+2*padding,1))
        self.layer_output_max_4d = tf.nn.max_pool(self.layer_output_padding_4d, 
                                           [1, self.competition_field, self.competition_field, 1],
                                           [1,1,1,1],
                                           padding='VALID')
        self.layer_output_max_2d=tf.reshape(self.layer_output_max_4d, 
                                        shape=(self.layer_neuron_dim_r, self.layer_neuron_dim_c))
        self.zero_matrix = np.ndarray(
                shape=(self.layer_neuron_dim_r, self.layer_neuron_dim_c),
                dtype=np.float32)
        self.zero_matrix[:]=1e-10
        self.layer_output_max_mask_2d = tf.less_equal(
                tf.subtract(self.layer_output_max_2d, self.layer_output_2d),
                self.zero_matrix)
        self.layer_output_max_mask = tf.reshape(
                tf.to_float(self.layer_output_max_mask_2d),
                shape=(self.layer_neuron_num,1)
                )
        self.layer_output_competition = tf.multiply(self.layer_output, self.layer_output_max_mask)
        #weights update
        self.input_data1=tf.tile(self.input_data, [1,self.layer_neuron_num])
        self.input_data2 = tf.transpose(self.input_data1)
        self.layer_delta = self.input_data2*self.layer_output_competition*self.learn_rate
        self.layer_weights_update = tf.add(self.layer_weights_norm, self.layer_delta)
        self.layer_weights_update_mask = tf.multiply(self.layer_weights_update, self.layer_weight_mask)
        self.layer_weights_update_mask_norm = tf.nn.l2_normalize(self.layer_weights_update_mask, 1)
        self.layer_update = tf.assign(self.layer_weights, self.layer_weights_update_mask_norm)
        
        #loss evaluation
        self.layer_output_loss = tf.nn.l2_loss(tf.subtract(self.layer_previous_output, self.layer_output_competition)) 
        self.layer_previous_output_updata = tf.assign(self.layer_previous_output, self.layer_output_competition)
        self.layer_weights_loss = tf.nn.l2_loss(tf.subtract(self.layer_weights_update_mask_norm, self.layer_weights_norm)) 
    
    def trainLayer(self, sess, feed_dict):
        new_weights, weight_loss, output_loss, _ = sess.run([self.layer_update, self.layer_weights_loss, self.layer_output_loss, self.layer_previous_output_updata], 
                                             feed_dict=feed_dict)
        return new_weights, weight_loss, output_loss

    
        
    
def main(_):
    #minst data
    minst_data = extract_minst_data('data/train-images-idx3-ubyte.gz', NUM_IMAGES)
    #img_data = minst_data[0, 0:input_data_dims, 0]
    #img_data = img_data.reshape(IMAGE_SIZE,IMAGE_SIZE)
    #imshow(img_data)
    
    minst_data = sharp_minst_data(minst_data)
    #img_data1 = minst_data[0, 0:input_data_dims, 0]
    #img_data1 = img_data1.reshape(IMAGE_SIZE,IMAGE_SIZE)
    #imshow(img_data1)
    
    #create a mask matrix for weight
    layer_weight_mask = np.zeros([layer_neuron_num, input_data_dims])
    for i in range(layer_neuron_num_axis):
        for j in range(layer_neuron_num_axis):
            row = i*layer_neuron_num_axis+j
            for k in range(RECEPTION_FIELD):
                start = IMAGE_SIZE*(i+k)+j 
                end = start+RECEPTION_FIELD
                layer_weight_mask[row,start:end]=1
    
    #initialize weights
    layer_weights = tf.Variable(tf.random_uniform([layer_neuron_num, input_data_dims], minval=0.0, maxval=1.0))
    layer_weights_mask = tf.multiply(layer_weights, layer_weight_mask)
    layer_weights_norm = tf.nn.l2_normalize(layer_weights_mask,1) #confirm ok
    #initial output
    layer_previous_output = tf.Variable(tf.zeros(shape=(layer_neuron_num,1)))
    
    #input data
    input_data = tf.placeholder(
          tf.float32,
          shape=(input_data_dims,1))
    
    #compute output
    layer_output = tf.matmul(layer_weights_norm, input_data)
    layer_output_2d = tf.reshape(layer_output, shape=(layer_neuron_num_axis, layer_neuron_num_axis))
    padding = int(COMPETITION_FIELD/2)
    layer_output_padding = tf.pad(layer_output_2d, [[padding, padding],[padding,padding]])
    
    #max competition
    layer_output_padding_4d = tf.reshape(layer_output_padding, 
                                          shape=(1,layer_neuron_num_axis+2*padding,layer_neuron_num_axis+2*padding,1))
    layer_output_max_4d = tf.nn.max_pool(layer_output_padding_4d, 
                                       [1, COMPETITION_FIELD, COMPETITION_FIELD, 1],
                                       [1,1,1,1],
                                       padding='VALID')
    layer_output_max_2d=tf.reshape(layer_output_max_4d, 
                                    shape=(layer_neuron_num_axis, layer_neuron_num_axis))
    zero_matrix = np.ndarray(
            shape=(layer_neuron_num_axis,layer_neuron_num_axis),
            dtype=np.float32)
    zero_matrix[:]=1e-10
    layer_output_max_mask_2d = tf.less_equal(
            tf.subtract(layer_output_max_2d, layer_output_2d),
            zero_matrix)
    layer_output_max_mask = tf.reshape(
            tf.to_float(layer_output_max_mask_2d),
            shape=(layer_neuron_num,1)
            )
    layer_output_competition = tf.multiply(layer_output, layer_output_max_mask)
    
    #weights update
    input_data1=tf.tile(input_data, [1,layer_neuron_num])
    input_data2 = tf.transpose(input_data1)
    layer_delta = input_data2*layer_output_competition*LEARNING_RATE
    layer_weights_update = tf.add(layer_weights_norm, layer_delta)
    layer_weights_update_mask = tf.multiply(layer_weights_update, layer_weight_mask)
    layer_weights_update_mask_norm = tf.nn.l2_normalize(layer_weights_update_mask, 1)
    layer_update = tf.assign(layer_weights, layer_weights_update_mask_norm)
    
    #loss evaluation
    layer_output_loss = tf.nn.l2_loss(tf.subtract(layer_previous_output, layer_output_competition)) 
    layer_previous_output_updata = tf.assign(layer_previous_output, layer_output_competition)
    layer_weights_loss = tf.nn.l2_loss(tf.subtract(layer_weights_update_mask_norm, layer_weights_norm)) 
    
    #layer_train = tf.group(layer_update, layer_weights_loss)                                  
    #implement model
    init_data = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_data)
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        start_time = time.time()
        for epoch in range(NUM_EPOCH):
            for image_idx in range(NUM_IMAGES):
                feed_dict = {input_data: minst_data[image_idx,]}
                new_weights, weight_loss, output_loss, _ = sess.run([layer_update, layer_weights_loss, layer_output_loss, layer_previous_output_updata], 
                                             feed_dict=feed_dict)

            if epoch%10 == 0:
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print('epoch: {} elapse time:{:.2f} sec weight loss: {} out loss:{}'.format(epoch, elapsed_time, weight_loss, output_loss))
                dbg_show_weights(new_weights)
            
if __name__ == '__main__':
	#main(_)
    test()
	