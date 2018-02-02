# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 07:53:40 2017

@author: lenov
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import FC as fc
import tensorflow as tf
#from tensorflow.python import debug as tf_debug # it didn't work if import tensorflow.python.debug as tf_debug
import numpy as np
import time
import gzip
from PIL import Image
#from pylab import *
#import matplotlib
import matplotlib.pyplot as plt
from DataSet import DataSet
from tensorflow.examples.tutorials.mnist import input_data

# NOTE: -1 is undefined, don't use it as the parameter value
#神经网络参数
PIXEL_DEPTH = 255.0 # 图片像素深度
IMAGE_SIZE = 28 #图片大小
RECEPTION_FIELD = 8 #感受野
COMPETITION_FIELD=3 #竞争域
LEARNING_RATE = 0.01 # 学习率
NUM_IMAGES = 1000 # 参与每轮训练的图片数
NUM_EPOCH = 3 #训练轮数
AMP_COF = 100 #竞争结果放大系数
EVAL_DATA_NUM = 10 # 测试数据数量
WEIGHT_COF = 0.1 #调整权值的系数，w=w*WEIGHT_COF
CONVERGE_THRESHOLD = 0.0001 #判断矩阵是否收敛
ASSOCIATION_FIELD = 7 # MUTSTBE ODD
ASSOCIATE_MODE_HOPFIELD = 1
ASSOCIATE_MODE_BSB = 2
#学习方法控制
MAX_COMPETITION = 0
AVG_COMPETITION = 1
UPDATE_THRESHOLD = 1
NO_UPDATE_THRESHOLD = 0
WEIGHT_RENORM = 1 #标准化方式更新权重
WEIGHT_LEAK = 0 #以leak方式更新权重
#距离定义
DIST_VECTOR_DOT = 1
DIST_L2 = 2
#其他常量
MAX_FLOAT = 1.7976931348623157e+308

competition_mode_dict = {MAX_COMPETITION:"max competition", AVG_COMPETITION: "avg competition"}
threshold_mode_dict = {UPDATE_THRESHOLD:"update threshold by avg", NO_UPDATE_THRESHOLD:"no threshold"}
weights_mode_dict = {WEIGHT_RENORM:"weights update by normalization", WEIGHT_LEAK:"weights update by leak"}
association_mode_dict = {ASSOCIATE_MODE_HOPFIELD:"association by hopfield classic", ASSOCIATE_MODE_BSB:"association by BSB"}

def test():
    print('test code running...')
    '''
    test 1: under the same context and input, the node in the graph is evaluated only once
    if the target nodes are computed at same time. In the following codes, the node of n4 run once only
    '''
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
    '''
    
    '''
    test 2: while-loop
    '''
    '''
    m = tf.Variable(0)

    def cond(c):
        return c<10
    
    def body(c):
        assign_op = m.assign_add(1)
        for i in range(10): #the for block - only runing once
            print('runing loop') 
        with tf.control_dependencies([assign_op]): #很重要，因为并行执行，否则可能导致body返回，而部分tf语句来不及执行
            print('here it is')#only once
            return c+1
    
    n = tf.while_loop(cond, body, [m])#[m]可以为tensor
    
    n1 = n+1
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        print(sess.run(m))
        print(sess.run(n1))
        print(sess.run(m))
    '''
    
    '''
    # Define a single queue with two components to store the input data.                                                                                           
    q_data = tf.FIFOQueue(100000, [tf.float32, tf.float32])
    
    # We will use these placeholders to enqueue input data.                                                                                                        
    placeholder_x = tf.placeholder(tf.float32, shape=[None])
    placeholder_y = tf.placeholder(tf.float32, shape=[None])
    enqueue_data_op = q_data.enqueue_many([placeholder_x, placeholder_y])
    
    gs = tf.Variable(0)
    w = tf.Variable(0.)
    b = tf.Variable(0.)
    
    optimizer = tf.train.GradientDescentOptimizer(0.05)
    
    # Construct the while loop.                                                                                                                                    
    def cond(i):
      return i < 10
    
    def body(i):
      # Dequeue a single new example each iteration.                                                                                                                      
      x, y = q_data.dequeue()
    
      # Compute the loss and gradient update based on the current example.                                                                                         
      loss = (tf.add(tf.multiply(x, w), b) - y)**2
      train_op = optimizer.minimize(loss, global_step=gs)
    
      # Ensure that the update is applied before continuing.                                                                                                       
      with tf.control_dependencies([train_op]):
        return i + 1
    
    loop = tf.while_loop(cond, body, [tf.constant(0)])
    
    data = [k*1. for k in range(10)]
    
    with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())
    
      for _ in range(1):
        # NOTE: Constructing the enqueue op ahead of time avoids adding                                                                                            
        # (potentially many) copies of `data` to the graph.                                                                                                        
        sess.run(enqueue_data_op, feed_dict={placeholder_x: data,
                                             placeholder_y: data})
    
      print(sess.run([gs, w, b]))  # Prints before-loop values.                                                                                                     
      sess.run(loop)
      print(sess.run([gs, w, b]) ) # Prints after-loop values.

    '''
    
    '''
    a = tf.Variable(1)
    b = tf.Variable(2)
    
    f = tf.constant(6)
    
    # Definition of condition and body
    def cond(a, b, f):
        return a < 3
    
    def body(a, b, f):
        # do some stuff with a, b
        a = a + 1
        b = b+1
        return a, b, f
    # Loop, 返回的tensor while 循环后的 a，b，f
    a, b, f = tf.while_loop(cond, body, [a, b, f])
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        r1, r2, r3 = sess.run([a, b, f])
        print(r1, r2, r3)
    '''    
    '''
    test tensor ops
    
    foo = tf.Variable([[1,2,3], [4,5,6], [7,8,9]])
    foo[0,:] = [7,7,7]
    #f2 = tf.reshape(f1, (1,3))
    #f3 = tf.concat([f1,[[9,9,9]]], 0)

    with tf.Session() as sess:
        print(sess.run(foo[0,:]))
    '''
    
    '''
    v = tf.Variable([1.0,2.0,3.0], dtype=tf.float32)
    d = tf.Variable([2.0,0.0,4.0],dtype=tf.float32)
    r = tf.divide(v, d)
    r1 =  tf.where(tf.is_inf(r), tf.zeros_like(r), r)
    r2 = tf.zeros_like(r)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        r, r1, r2 = sess.run([r, r1, r2])
        print(r, r1, r2)
    '''
    
    layer_weights = tf.Variable(tf.random_uniform([10, 10], minval=0.0, maxval=1.0))
    init_data = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_data)
        l = sess.run(layer_weights)
        print_matrix(l, 1)

def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]

        
def extract_minst_data(filename, num_images):
  """Extract the minst images into a 3D tensor [image index, y, x].
  Values are rescaled from [0, 255] down to [0, 1].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    #bytestream.read(16)
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, filename))
    max_num_images = _read32(bytestream)
    rows = _read32(bytestream) # rows and cols should be IMAGE_SIZE
    cols = _read32(bytestream)
    if(num_images == -1):
        num_images = max_num_images
    elif(num_images > max_num_images):
        raise ValueError('Invalid num_images - max number is %d'%max_num_images)
    
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data / PIXEL_DEPTH #image.show will be fallen if psixel is between 0~1
    data = data.reshape(num_images, IMAGE_SIZE * IMAGE_SIZE,1)
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
    data  = data.reshape(image_nums, IMAGE_SIZE * IMAGE_SIZE, 1)
    print('Sharping finish!')
    return data
                
def build_fake_data():
    #fake data
    fake_data = np.ndarray(
          shape=(IMAGE_SIZE * IMAGE_SIZE, 1),
          dtype=np.float32)
    fake_data[:]=np.random.random_sample((IMAGE_SIZE * IMAGE_SIZE,1))
    return fake_data

def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting Label', filename)
  with gzip.open(filename) as bytestream:
    #bytestream.read(8)
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, filename))
    num_items = _read32(bytestream)
    if(num_images > num_items):
        raise ValueError('Invalid num_images - max number is %d'%num_items)

    buf = bytestream.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  return labels

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def dbg_show_weights(weights, name):
    #weights : neurons * input data dimension
    shape = weights.shape;
    neuron_nums = shape[0]
    neuron_axis_size = int(np.sqrt(shape[0]))
    data_axis_size = int(np.sqrt(shape[1]))
    data = np.array(weights)
    data = data.reshape([neuron_nums, data_axis_size, data_axis_size])
    

    #show all the weights including zeros
    #img_data = np.zeros([neuron_axis_size*data_axis_size, neuron_axis_size*data_axis_size])
    im = Image.new('L',(neuron_axis_size*data_axis_size, neuron_axis_size*data_axis_size))
    img_data = im.load()
    for i in range(neuron_axis_size): #row
        for j in range(neuron_axis_size):#col
            x_start, x_end = i*data_axis_size, (i+1)*data_axis_size
            y_start, y_end = j*data_axis_size, (j+1)*data_axis_size
            for x in range(x_start, x_end):
                for y in range(y_start, y_end):
                    img_data[x,y]=int(data[i*neuron_axis_size+j][x-x_start][y-y_start]*255)
                    #print(img_data[x,y])
            #img_data[x_start:x_end, y_start:y_end] = data2[i*neuron_axis_size+j]
    
    im.save('.\\experiment\\'+name+'.png')#保存没问题
#    plt.imshow(Image.open('test.png'))#显示有错误，已确定。应该是数据准备的不对
    
#    plt.figure(200)
#    plt.imshow(im, cmap='gray')
#    plt.show()
    
def print_matrix(m, ignorezero):
    size = m.shape
    for i in range(size[0]):
        zero_out = False
        for j in range(size[1]):
            if(ignorezero and m[i][j] == 0.0):
                if(not zero_out ):
                    print(':', end=' ')
                    zero_out = True
                continue
            print(m[i][j], end=' ')
            zero_out = False
        print('\n')   
    return

def print_training_setting(competition_mode, weight_update_mode, threshold_update_mode, reception_field,  
                           learning_rate, association_mode, association_loop, association_field):
    
    print('competition_mode: {}\n'\
          'weight_update_mode: {}\n'\
          'threshold_update_mode: {}\n'\
          'reception_field: {}\n'\
          'learning_rate: {}\n' \
          'association_mode: {}\n'\
          'association_loop: {}\n'\
          'association_field: {}\n'
          .format(competition_mode_dict.get(competition_mode, "none"),
                  weights_mode_dict.get(weight_update_mode,"none"),
                  threshold_mode_dict.get(threshold_update_mode, "none"),
                  reception_field,
                  learning_rate,
                  association_mode_dict.get(association_mode, "none"),
                  association_loop,
                  association_field
                  )
          )
    return

class FeatureLayer:
    
    def __init__(self, reception_field, competition_field, input_dim_r, input_dim_c, input_data, learn_rate, amp_cof):
        self.reception_field = reception_field
        self.competition_field = competition_field
        self.input_dim_r = input_dim_r
        self.input_dim_c = input_dim_c
        self.layer_neuron_dim_r = input_dim_r-reception_field+1#output neuron rows
        self.layer_neuron_dim_c = input_dim_c-reception_field+1#output neuron cols
        self.layer_neuron_num = self.layer_neuron_dim_r*self.layer_neuron_dim_c
        self.input_data_dims = input_dim_r*input_dim_c
        self.input_data = input_data
        self.learn_rate = learn_rate
        self.zero_matrix = np.ndarray(
                shape=(self.layer_neuron_dim_r, self.layer_neuron_dim_c),
                dtype=np.float32)
        self.zero_matrix[:]=1e-10
        self.amp_cof = amp_cof
        
    
    def getOutPutDims(self):
        return self.layer_neuron_num
    
    def getInputDataDims(self):
        return self.input_data_dims
    
    def getOutPut(self):
        return self.layer_previous_output
    
    def createLayerWeightsMask(self):
        #create a mask matrix for weight
        self.layer_weight_mask = np.zeros([self.layer_neuron_num, self.input_data_dims])
        for i in range(self.layer_neuron_dim_r):
            for j in range(self.layer_neuron_dim_c):
                row = i*self.layer_neuron_dim_c+j
                for k in range(self.reception_field):
                    start = self.input_dim_c*(i+k)+j 
                    end = start+self.reception_field
                    self.layer_weight_mask[row,start:end]=1
        return

    def computeLayerOutput(self, competition, data):
        if competition == MAX_COMPETITION :
            layer_output_competition = self.max_competition(data)
        elif competition == AVG_COMPETITION :
            layer_output_competition = self.avg_competition(data)
        return layer_output_competition
    
    def buildLayer(self, competition, update_threshold, update_weight):
        self.train_loops = tf.Variable(0.0)#save the train times
        self.layer_weights_pre = tf.Variable(tf.zeros([self.layer_neuron_num, self.input_data_dims]))# save the previous weights
        
        self.createLayerWeightsMask()
        
        #initialize weights
        self.layer_weights = tf.Variable(tf.random_uniform([self.layer_neuron_num, self.input_data_dims], minval=0.0, maxval=1.0))
        layer_mask_weights = tf.multiply(self.layer_weights, self.layer_weight_mask)
        if(update_weight == WEIGHT_RENORM):
            print('weight update in WEIGHT_RENORM is true')
            self.layer_weights_norm = tf.assign(self.layer_weights,
                                           tf.nn.l2_normalize(layer_mask_weights,1)) #会不会重复标准化.NO
        elif(update_weight == WEIGHT_LEAK):
            print('weight update in WEIGHT_LEAK is true')
            self.layer_weights_norm = tf.assign(self.layer_weights,
                                           tf.nn.l2_normalize(layer_mask_weights,1)*WEIGHT_COF)
        #init threshold
        self.layer_threshold = tf.Variable(tf.zeros(shape=(self.layer_neuron_num,1)))
        #initial output
        self.layer_previous_output = tf.Variable(tf.zeros(shape=(self.layer_neuron_num,1)))#to save the previous output
        #compute output
        self.layer_output_competition = self.computeLayerOutput(competition, self.input_data)

        #weights update
        #re-normalize way
        if(update_weight == WEIGHT_RENORM):
            input_data1=tf.tile(self.input_data, [1,self.layer_neuron_num])
            input_data2 = tf.transpose(input_data1)
            layer_delta = input_data2*self.layer_output_competition*self.learn_rate
            layer_weights_update = tf.add(self.layer_weights, layer_delta)
            layer_weights_update_mask = tf.nn.l2_normalize(tf.multiply(layer_weights_update, 
                                                                       self.layer_weight_mask),
                                                           1)
        elif(update_weight == WEIGHT_LEAK):
        #leak way
            input_data1=tf.tile(self.input_data, [1,self.layer_neuron_num])
            input_data2 = tf.transpose(input_data1)
            weight_leak = self.layer_weights*tf.square(self.layer_output_competition)
            weight_leak = tf.where(tf.is_nan(weight_leak), tf.zeros_like(weight_leak), weight_leak)#maybe nan
            layer_delta = (input_data2*self.layer_output_competition-weight_leak)*self.learn_rate
            layer_delta = tf.where(tf.less(layer_delta, 0.0), tf.zeros_like(layer_delta), layer_delta)
            layer_weights_update = tf.add(self.layer_weights, layer_delta)
            layer_weights_update_mask = tf.multiply(layer_weights_update, self.layer_weight_mask)
            
        save_weights = tf.assign(self.layer_weights_pre, self.layer_weights)
        with tf.control_dependencies([save_weights]):
            self.layer_update_weight = tf.assign(self.layer_weights, layer_weights_update_mask)
        
        
        #threshold update
        if update_threshold == UPDATE_THRESHOLD:
            print('update threshold is selected')
            self.layer_update_threshold = tf.assign(self.layer_threshold, 
                                                    tf.divide(tf.add(self.layer_threshold*self.train_loops, self.layer_output_competition), 
                                                              self.train_loops.assign_add(1)
                                                              )
                                                    )
        else:
            print('update threshold isnt selected')
            self.layer_update_threshold = self.layer_threshold
        
        #output and evaluate
        #self.layer_output_loss = tf.nn.l2_loss(tf.subtract(layer_previous_output, self.layer_output_competition)) 
        self.layer_output_energy = tf.reduce_sum(self.layer_output_competition)
        self.layer_output_save = tf.assign(self.layer_previous_output, self.layer_output_competition)
        #weight evaluating
        self.layer_weights_loss = tf.nn.l2_loss(tf.subtract(self.layer_update_weight, save_weights))
        self.layer_weights_energy =tf.reduce_sum(tf.square(self.layer_update_weight))/self.getOutPutDims()
        #threshold evaluating
        self.layer_threshold_energy = tf.reduce_sum(self.layer_threshold)
        return
    
        
    def max_competition(self, input_data):
        '''
        对于单峰分布问题（只有峰顶元素（全局最大）能够激活），无法解决。或者是说
        完全实现的代价太高，目前采用avg competition方法。
        '''
        layer_out_original = tf.matmul(self.layer_weights, input_data)
        layer_out_original = tf.multiply(layer_out_original, 
                                         tf.to_float(tf.less_equal(self.layer_threshold, 
                                                                   layer_out_original)))#如果超过阈值，则激活，且激活值不应减掉阈值
        layer_out_original_2d = tf.reshape(layer_out_original, shape=(self.layer_neuron_dim_r, self.layer_neuron_dim_c))
        
        padding = int(self.competition_field/2)
        layer_output_padding = tf.pad(layer_out_original_2d, [[padding, padding],[padding,padding]])
        layer_output_padding_4d = tf.reshape(layer_output_padding, 
                                              shape=(1,self.layer_neuron_dim_r+2*padding,self.layer_neuron_dim_c+2*padding,1))
        layer_output_max_4d = tf.nn.max_pool(layer_output_padding_4d, 
                                           [1, self.competition_field, self.competition_field, 1],
                                           [1,1,1,1],
                                           padding='VALID')
        layer_output_max_2d=tf.reshape(layer_output_max_4d, 
                                        shape=(self.layer_neuron_dim_r, self.layer_neuron_dim_c))
        layer_output_max_mask_2d = tf.to_float(tf.less_equal(layer_output_max_2d, layer_out_original_2d))
        
        layer_output_max_competitive = tf.reshape(tf.multiply(layer_out_original_2d, layer_output_max_mask_2d),
                                                  shape=(self.layer_neuron_num,1))
        
        return layer_output_max_competitive

    def avg_competition(self, input_data):
        '''
        add the code to amplif the larger value while depress the small value in order to select the maximal value locally
        对于单峰分布问题，即使非全局最大，在局部也可部分激活，且激活程度与局部有关。可避免max competition只能峰顶神经元激活的问题。
        '''
        layer_out_original = tf.matmul(self.layer_weights, input_data)
        layer_out_original = tf.multiply(layer_out_original, 
                                         tf.to_float(tf.less_equal(self.layer_threshold, 
                                                                   layer_out_original)))#如果超过阈值，则激活，且激活值不应减掉阈值
        layer_out_original_2d = tf.reshape(layer_out_original, shape=(self.layer_neuron_dim_r, self.layer_neuron_dim_c))
        
        #amplify the difference among the outputs
        layer_out_amplify_2d = tf.square(layer_out_original_2d)*self.amp_cof#按平方放大
        padding = int(self.competition_field/2)
        layer_output_padding = tf.pad(layer_out_amplify_2d, [[padding, padding],[padding,padding]])
        layer_output_padding_4d = tf.reshape(layer_output_padding, 
                                              shape=(1,self.layer_neuron_dim_r+2*padding,
                                                     self.layer_neuron_dim_c+2*padding,1))
        layer_output_avg_4d = tf.nn.avg_pool(layer_output_padding_4d, 
                                           [1, self.competition_field, self.competition_field, 1],
                                           [1,1,1,1],
                                           padding='VALID')
        layer_output_avg_2d=tf.reshape(layer_output_avg_4d, 
                                        shape=(self.layer_neuron_dim_r, self.layer_neuron_dim_c))
        
        layer_output_neuron_diff_amplify_2d = tf.exp(layer_out_amplify_2d - layer_output_avg_2d)#进一步按指数放大
        layer_output_neuron_diff_amplify_4d = tf.reshape(tf.pad(tf.square(layer_output_neuron_diff_amplify_2d), #平方
                                                                [[padding, padding],[padding,padding]]
                                                                ),
                                                         shape=(1,self.layer_neuron_dim_r+2*padding,self.layer_neuron_dim_c+2*padding,1))
        layer_output_neuron_amplify_cof_4d = tf.nn.avg_pool(layer_output_neuron_diff_amplify_4d,
                                           [1, self.competition_field, self.competition_field, 1],
                                           [1,1,1,1],
                                           padding='VALID')
        layer_output_neuron_amplify_cof_2d = tf.reshape(layer_output_neuron_amplify_cof_4d,
                                                        shape=(self.layer_neuron_dim_r, self.layer_neuron_dim_c))
        layer_output_neuron_amplify_cof_2d = tf.divide(layer_output_neuron_diff_amplify_2d,
                                                         tf.sqrt(layer_output_neuron_amplify_cof_2d*(self.competition_field*self.competition_field))#竞争域内的平方和的根。可能为nan
                                                         )
        layer_output_neuron_amplify_cof_2d = tf.where(tf.is_nan(layer_output_neuron_amplify_cof_2d),
                                                      tf.zeros_like(layer_output_neuron_amplify_cof_2d),
                                                      layer_output_neuron_amplify_cof_2d) #kill the nan
        
        layer_out = tf.reshape(tf.multiply(layer_out_original_2d, layer_output_neuron_amplify_cof_2d),
                               shape=(self.layer_neuron_num,1))
        return layer_out
        
    def initWeight(self, sess):
        return sess.run(self.layer_weights_norm)
    
    def trainLayer(self, sess, feed_dict):
        layer_output, new_weights, new_threshold, out_energy, threshold_energy, weights_energy, weights_loss, _ = sess.run(
                [self.layer_output_competition,
                 self.layer_update_weight, 
                 self.layer_update_threshold,
                 self.layer_output_energy,
                 self.layer_threshold_energy,
                 self.layer_weights_energy,
                 self.layer_weights_loss,
                 self.layer_output_save
                 ], 
                feed_dict=feed_dict)
        return layer_output, new_weights, new_threshold, out_energy, threshold_energy, weights_energy, weights_loss
    
    def trainHiddenLayer(self, sess):
        layer_output, new_weights, new_threshold, out_energy, threshold_energy, weights_energy, weights_loss, _ = sess.run(
                [self.layer_output_competition,
                 self.layer_update_weight, 
                 self.layer_update_threshold,
                 self.layer_output_energy,
                 self.layer_threshold_energy,
                 self.layer_weights_energy,
                 self.layer_weights_loss,
                 self.layer_output_save
                 ]
                 )
        return layer_output, new_weights, new_threshold, out_energy, threshold_energy, weights_energy, weights_loss
       
    
    def build_singlelayer_eval_model(self, competition, input_data, distance):
        '''
        for singleLayer test: comupte the distance among all the data
        '''
        #compute output ops
        data_num = input_data.shape[1]
        self.eval_result = np.zeros((data_num, self.getOutPutDims()))#data_num*neurons_num

        self.eval_input_data = tf.placeholder(tf.float32, shape=(self.getInputDataDims(), 1))
        out = self.computeLayerOutput(competition, self.eval_input_data)
        self.eval_out = tf.reshape(out,
                              shape=[1, self.getOutPutDims()])
        
        self.eval_dist_data = tf.placeholder(tf.float32, shape = self.eval_result.shape)
        #compute the vector dot distances
        if distance == DIST_VECTOR_DOT :
            d = tf.matmul(self.eval_dist_data, tf.transpose(self.eval_dist_data)) #data_num*data_num
            d = tf.where(tf.is_nan(d), tf.zeros_like(d), d)
            norm = tf.reduce_sum(tf.multiply(self.eval_dist_data, self.eval_dist_data), 1)
            d_num = data_num*(data_num-1)/2
            self.eval_avg = (tf.reduce_sum(d)-tf.reduce_sum(norm))/(2*d_num)
            self.eval_var = (tf.reduce_sum(tf.square(d))-tf.reduce_sum(tf.square(norm)))/d_num-tf.square(self.eval_avg)
        elif distance == DIST_L2 :
            #compute L2 distance
            dot = tf.matmul(self.eval_dist_data, tf.transpose(self.eval_dist_data)) #data_num*data_num
            dot = tf.where(tf.is_nan(dot), tf.zeros_like(dot), dot)
            sum_square = tf.reshape(tf.reduce_sum(tf.square(self.eval_dist_data), 1),
                                    shape=(data_num,1))
            sum_square_mat = tf.tile(sum_square, [1, data_num])
            sum_square_mat1 = tf.add(sum_square_mat, tf.transpose(sum_square_mat))
            l2_dist = sum_square_mat1-2*dot
            d_num = data_num*(data_num-1)
            self.eval_avg = tf.reduce_sum(l2_dist)/d_num #the diag is zero
            self.eval_var = tf.reduce_sum((tf.square(l2_dist)))/d_num - tf.square(self.eval_avg)
        
        return
    
    def eval_singlelayer_model(self, sess, input_data):
        
        data_num = input_data.shape[1]
        for i in range(data_num):
            data = np.reshape(input_data[:, i:i+1], newshape=(self.getInputDataDims(), 1))
            self.eval_result[i] = sess.run(self.eval_out, feed_dict={self.eval_input_data:data})
                 
        avg, var = sess.run([self.eval_avg, self.eval_var], feed_dict = {self.eval_dist_data:self.eval_result})
        return [avg, var]

class AssociationLayer:
    
    def __init__(self, association_field, layer_dim_r, layer_dim_c, learn_rate):
        '''
        association_field: the field where a neuron has the links 
        layer_dim_r, layer_dim_c: the association layer pane size
        '''
        self.association_field = association_field
        self.layer_dim_r = layer_dim_r
        self.layer_dim_c = layer_dim_c
        self.layer_neuron_num = self.layer_dim_r * self.layer_dim_c
        self.learn_rate = learn_rate
        
        return

    def getOutPutDims(self):
        return self.layer_neuron_num
    
    def createLayerWeightsMask(self):
        #create a mask matrix for weight :for i and j, the association field is the rectangle of (i-r/2, j-r/2), (i+r/2, j+r/2)
        association_radius = int(np.floor(self.association_field/2))
        self.layer_weight_mask = np.zeros([self.layer_neuron_num, self.layer_neuron_num])
        for i in range(self.layer_dim_r):
            for j in range(self.layer_dim_c):
                row = i*self.layer_dim_c+j
                if i<association_radius :
                    top = 0
                else:
                    top = i-association_radius
                    
                if j<association_radius :
                    left = 0
                else:
                    left = j-association_radius
                    
                if i>(self.layer_neuron_num-association_radius) :
                    bottom = self.layer_neuron_num
                else:
                    bottom = i+association_radius
                
                if j>(self.layer_neuron_num-association_radius) :
                    right = self.layer_neuron_num
                else:
                    right = j+association_radius
                
                for k in range(bottom-top):
                    start = self.layer_dim_c*(top+k)+left 
                    end = start+right-left+1
                    self.layer_weight_mask[row,start:end]=1
        #remove the self-link
        for i in range(self.layer_neuron_num):
            self.layer_weight_mask[i,i]=0
            
        return

    def buildLayer(self, update_threshold, update_weight):
        #save the train times up to now
        self.train_loops = tf.Variable(0.0)#save the train times
        #input data
        self.input_data = tf.placeholder(tf.float32, shape=(self.layer_neuron_num, 1))
        #initialize weights. to run layer_weights_norm to initialize the weights
        self.createLayerWeightsMask()
        self.layer_weights = tf.Variable(tf.random_uniform([self.layer_neuron_num, self.layer_neuron_num], minval=0.0, maxval=1.0))
        layer_mask_weights = tf.multiply(self.layer_weights, self.layer_weight_mask)
        self.layer_weights_pre = tf.Variable(tf.zeros([self.layer_neuron_num, self.layer_neuron_num]))# save the previous weights
        
        if(update_weight == WEIGHT_RENORM):
            self.layer_weights_norm = tf.assign(self.layer_weights,
                                           tf.nn.l2_normalize(layer_mask_weights,1))
        elif(update_weight == WEIGHT_LEAK):
            self.layer_weights_norm = tf.assign(self.layer_weights,
                                           tf.nn.l2_normalize(layer_mask_weights,1)*WEIGHT_COF)
        #init threshold
        self.layer_threshold = tf.Variable(tf.zeros(shape=(self.layer_neuron_num,1)))
        #initial output
        self.layer_previous_output = tf.Variable(tf.zeros(shape=(self.layer_neuron_num,1)))#to save the previous output
        self.layer_output = tf.Variable(tf.zeros(shape=(self.layer_neuron_num,1)))#to save the current output
        #compute output - to run layer_output_ops to get the output for the current input data
        self.layer_output_ops = tf.assign(self.layer_output, self.computeLayerOutput(self.input_data))

        #weights update. to run layer_update_weight to update the weights
        if(update_weight == WEIGHT_RENORM):
        #re-normalize way
            layer_delta = tf.matmul(self.layer_output, tf.transpose(self.layer_output))*self.learn_rate
            layer_weights_update = tf.add(self.layer_weights, layer_delta)
            layer_weights_update_mask = tf.nn.l2_normalize(tf.multiply(layer_weights_update, 
                                                                       self.layer_weight_mask),
                                                           1)
        elif(update_weight == WEIGHT_LEAK):
        #leak way
            weight_leak = self.layer_weights*tf.square(self.layer_output)
            weight_leak = tf.where(tf.is_nan(weight_leak), tf.zeros_like(weight_leak), weight_leak)#maybe nan
            layer_delta = (tf.matmul(self.layer_output, tf.transpose(self.layer_output))-weight_leak)*self.learn_rate
            layer_delta = tf.where(tf.less(layer_delta, 0.0), tf.zeros_like(layer_delta), layer_delta)
            layer_weights_update = tf.add(self.layer_weights, layer_delta)
            layer_weights_update_mask = tf.multiply(layer_weights_update, self.layer_weight_mask)
            
        save_weights = tf.assign(self.layer_weights_pre, self.layer_weights)
        with tf.control_dependencies([save_weights]):
            self.layer_update_weight = tf.assign(self.layer_weights, layer_weights_update_mask)
        
        
        #threshold update. to run layer_update_threshold to update the threshold
        if update_threshold == UPDATE_THRESHOLD:
            self.layer_update_threshold = tf.assign(self.layer_threshold, 
                                                    tf.divide(tf.add(self.layer_threshold*self.train_loops, self.layer_output), 
                                                              self.train_loops.assign_add(1)
                                                              )
                                                    )
        else:
            self.layer_update_threshold = self.layer_threshold
        
        #output and evaluate
        self.layer_output_energy = tf.reduce_sum(self.layer_output)
        self.layer_output_loss = tf.nn.l2_loss(tf.subtract(self.layer_previous_output, self.layer_output))
        with tf.control_dependencies([self.layer_output_loss]):
            self.layer_output_save = tf.assign(self.layer_previous_output, self.layer_output)
        
        #weight evaluating
        self.layer_weights_loss = tf.nn.l2_loss(tf.subtract(self.layer_update_weight, save_weights))
        self.layer_weights_energy =tf.reduce_sum(tf.square(self.layer_update_weight))/self.getOutPutDims()
        
        #threshold evaluating
        self.layer_threshold_energy = tf.reduce_sum(self.layer_threshold)
        return

    def computeLayerOutput(self, input_data):
        '''
        compute the layer out once
        '''
        layer_out_original = tf.matmul(self.layer_weights, input_data)
        layer_out_original = tf.multiply(layer_out_original, 
                                         tf.to_float(tf.less_equal(self.layer_threshold, 
                                                                   layer_out_original)))#如果超过阈值，则激活，且激活值不应减掉阈值
        layer_out = tf.reshape(layer_out_original, shape=(self.layer_neuron_num,1))
        return layer_out

    def initWeight(self, sess):
        return sess.run(self.layer_weights_norm)

    def trainLayer(self, sess, input_data, loop, associate_mode):
        '''
        loop: >0 : the train times. -1: unlimit train until the convergence
        input_data: the train data
        '''
        self.associateLayerOutput(sess, input_data, loop, associate_mode)
        
        return sess.run([self.layer_update_weight, self.layer_update_threshold, 
                  self.layer_output_energy, self.layer_output_loss,  self.layer_output_save,
                  self.layer_weights_loss, self.layer_weights_energy, self.layer_threshold_energy])
        
    def associateLayerOutput(self, sess, input_data, loop, associate_mode):
        '''
        loop: >0 : the train times. -1: unlimit train until the convergence
        input_data: the data
        '''
        data = input_data
        if loop>0 :
            for i in range(loop):
                output = sess.run(self.layer_output_ops, feed_dict = {self.input_data: data})
                if associate_mode == ASSOCIATE_MODE_HOPFIELD :
                    data = output
                elif associate_mode == ASSOCIATE_MODE_BSB : #here \belta = 1 for BSB
                    data = data + output
                else:
                    raise Exception("Invalid associate_mode!")
        else:
            delta = MAX_FLOAT
            while(delta > CONVERGE_THRESHOLD):
                output, delta, _ = sess.run([self.layer_output_ops, self.layer_output_loss, self.layer_output_save],
                                  feed_dict = {self.input_data: data})
                if associate_mode == ASSOCIATE_MODE_HOPFIELD :
                    data = output
                elif associate_mode == ASSOCIATE_MODE_BSB :
                    data = data + output
                else:
                    raise Exception("Invalid associate_mode!")

        return output
    
# test the model of multiple feature layers    
def build_multilayer_eval_model(competition, input_data, dist, layers):
    layers_num = len(layers)
    data_num = input_data.shape[1]
    eval_result = []
    for i in range(layers_num):
        layer_ret = np.zeros((data_num, layers[i].getOutPutDims()))#data_num*neurons_num
        eval_result.append(layer_ret)

    eval_input_data = tf.placeholder(tf.float32, shape=(layers[0].getInputDataDims(), 1))
    layers_out = []
    for i in range(layers_num):
        if i==0 :
            out = layers[i].computeLayerOutput(competition, eval_input_data)
            layers_out.append(out)
        else :
            out = layers[i].computeLayerOutput(competition, layers_out[i-1])
            layers_out.append(out)
    
    distances_matrix = []
    for i in range(layers_num):
        d = tf.reshape(layers_out[i], shape=(1, layers[i].getOutPutDims()))
        distances_matrix.append(d)
    
    eval_dist_data = []
    for i in range(layers_num):
        dist_data = tf.placeholder(tf.float32, shape = (data_num, distances_matrix[i].get_shape()[1]))
        eval_dist_data.append(dist_data)
    
    def compute_distance_statistic(distance, dist_data):
        if distance == DIST_VECTOR_DOT :
            #compute the vector dot distances
            d = tf.matmul(dist_data, tf.transpose(dist_data)) #data_num*data_num
            d = tf.where(tf.is_nan(d), tf.zeros_like(d), d)
            norm = tf.reduce_sum(tf.multiply(dist_data, dist_data), 1)
            d_num = data_num*(data_num-1)/2
            eval_avg = (tf.reduce_sum(d)-tf.reduce_sum(norm))/(2*d_num)
            eval_var = (tf.reduce_sum(tf.square(d))-tf.reduce_sum(tf.square(norm)))/d_num-tf.square(eval_avg)
        elif distance == DIST_L2 :
            #compute L2 distance
            dot = tf.matmul(dist_data, tf.transpose(dist_data)) #data_num*data_num
            dot = tf.where(tf.is_nan(dot), tf.zeros_like(dot), dot)
            sum_square = tf.reshape(tf.reduce_sum(tf.square(dist_data), 1),
                                    shape=(data_num,1))
            sum_square_mat = tf.tile(sum_square, [1, data_num])
            sum_square_mat1 = tf.add(sum_square_mat, tf.transpose(sum_square_mat))
            l2_dist = sum_square_mat1-2*dot
            d_num = data_num*(data_num-1)
            eval_avg = tf.reduce_sum(l2_dist)/d_num #the diag is zero
            eval_var = tf.reduce_sum((tf.square(l2_dist)))/d_num - tf.square(eval_avg)
        return eval_avg, eval_var

    eval_avgs = []
    eval_vars = []
    for i in range(layers_num):
        avg, var = compute_distance_statistic(dist, eval_dist_data[i])
        eval_avgs.append(avg)
        eval_vars.append(var)
        
    return eval_input_data, distances_matrix, eval_dist_data, eval_avgs, eval_vars

def eval_multilayer_eval_model(sess, layers_num, input_data, eval_input_data, distances_matrix, eval_dist_data, eval_avgs, eval_vars):
    data_num = input_data.shape[1]
    data_dims = input_data.shape[0]
    
    dist_result = []
    for i in range(layers_num):
        r = []
        dist_result.append(r)
        
    for i in range(data_num):
        data = np.reshape(input_data[:, i:i+1], newshape=(data_dims, 1))
        distances = sess.run(distances_matrix, feed_dict={eval_input_data:data})
        for j in range(layers_num):
            dist_result[j].append(distances[j])
            
    eval_dist_dicts = {}
    for i in range(layers_num):
        ary = np.array(dist_result[i])
        newshape = (ary.shape[0], ary.shape[2])
        eval_dist_dicts[eval_dist_data[i]]= ary.reshape(newshape)
        
    avg_result, var_result = sess.run([eval_avgs, eval_vars], feed_dict=eval_dist_dicts)
    return avg_result, var_result
    
    

def SingleFWLayerExperiment():
    print("RECEPTION_FIELD:{} COMPETITION_FIELD:{} IMAGE_SIZE:{} LEARNING_RATE:{} AMP_COF:{} NUM_EPOCH:{} NUM_IMAGES:{}".format(
            RECEPTION_FIELD, COMPETITION_FIELD, IMAGE_SIZE, LEARNING_RATE, AMP_COF, NUM_EPOCH, NUM_IMAGES))
    #minst data
    minst_data = extract_minst_data('data/train-images-idx3-ubyte.gz', NUM_IMAGES)
    minst_data = sharp_minst_data(minst_data)

    #input data
    input_data = tf.placeholder(
          tf.float32,
          shape=(IMAGE_SIZE*IMAGE_SIZE,1))
    #build model
    neuron_competition = AVG_COMPETITION# MAX_COMPETITION 
    threshold_update = UPDATE_THRESHOLD #NO_UPDATE_THRESHOLD
    weights_update = WEIGHT_LEAK#WEIGHT_RENORM
    distance = DIST_L2
    
    layer1 = FeatureLayer(RECEPTION_FIELD, COMPETITION_FIELD, IMAGE_SIZE, IMAGE_SIZE, input_data, LEARNING_RATE, AMP_COF)
    layer1.buildLayer(neuron_competition, threshold_update, weights_update)
    
    #prepare eval data and result saver
    eval_summary = np.zeros(shape=(NUM_EPOCH, 2))
    eval_data = np.zeros(shape=(EVAL_DATA_NUM, layer1.getInputDataDims()))
    for i in range(EVAL_DATA_NUM):
        idx = np.random.randint(0, NUM_IMAGES)
        eval_data[i] = minst_data[idx].reshape((layer1.getInputDataDims()))
    layer1.build_singlelayer_eval_model(neuron_competition, eval_data.transpose(), distance)
    
    #training and evaluating model
    init_data = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_data)
        init_weights = layer1.initWeight(sess)
        dbg_show_weights(init_weights, 'init_weights')
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        start_time = time.time()
        for epoch in range(NUM_EPOCH):
            for image_idx in range(NUM_IMAGES):
                feed_dict = {input_data: minst_data[image_idx,]}
                layer_output, new_weights, new_threshold, out_energy, threshold_energy, weights_energy, weights_loss = layer1.trainLayer(sess, feed_dict)

       
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print('epoch: {} elapse time:{:.2f} sec weight loss: {} weights_energy:{}'.format(epoch, elapsed_time, weights_loss, weights_energy))
            
        
            '''
            evaluate model
            '''
            print("evalating test data...")
            eval_summary[epoch] =  layer1.eval_singlelayer_model(sess, eval_data.transpose())
            print('mean:{} var:{}'.format(eval_summary[epoch][0], eval_summary[epoch][1]))
        #show the avg and var
        plt.figure(1)
        plt.title('Test Results')
        x_axis = [i for i in range(NUM_EPOCH)]
        plt.plot(x_axis, eval_summary[:,0], label="mean")
        plt.plot(x_axis, eval_summary[:,1], label="var")
        #show the weights
        dbg_show_weights(new_weights, name='weights')

def MultiFWLayerExperiment():
    print("MultiFWLayerExperiment(3 layers): \n RECEPTION_FIELD:{} COMPETITION_FIELD:{} IMAGE_SIZE:{} LEARNING_RATE:{} AMP_COF:{} NUM_EPOCH:{} NUM_IMAGES:{}".format(
            RECEPTION_FIELD, COMPETITION_FIELD, IMAGE_SIZE, LEARNING_RATE, AMP_COF, NUM_EPOCH, NUM_IMAGES))
    #minst data
    minst_data = extract_minst_data('data/train-images-idx3-ubyte.gz', NUM_IMAGES)
    minst_data = sharp_minst_data(minst_data)

    #input data
    input_data = tf.placeholder(
          tf.float32,
          shape=(IMAGE_SIZE*IMAGE_SIZE,1))
    
    #build model
    neuron_competition = AVG_COMPETITION# MAX_COMPETITION 
    threshold_update = UPDATE_THRESHOLD #NO_UPDATE_THRESHOLD
    weights_update = WEIGHT_LEAK#WEIGHT_RENORM
    distance = DIST_L2

    layer1 = FeatureLayer(RECEPTION_FIELD, COMPETITION_FIELD, IMAGE_SIZE, IMAGE_SIZE, input_data, LEARNING_RATE, AMP_COF)
    layer1.buildLayer(neuron_competition, threshold_update, weights_update)
    layer1_output = layer1.getOutPut()
    layer2 = FeatureLayer(RECEPTION_FIELD, COMPETITION_FIELD, int(np.sqrt(layer1.getOutPutDims())), int(np.sqrt(layer1.getOutPutDims())), layer1_output, LEARNING_RATE, AMP_COF)
    layer2.buildLayer(neuron_competition, threshold_update, weights_update)
    layer2_output = layer2.getOutPut()
    layer3 = FeatureLayer(RECEPTION_FIELD, COMPETITION_FIELD, int(np.sqrt(layer2.getOutPutDims())), int(np.sqrt(layer2.getOutPutDims())), layer2_output, LEARNING_RATE, AMP_COF)
    layer3.buildLayer(neuron_competition, threshold_update, weights_update)
    
    #prepare eval data and result saver
    eval_summary = np.zeros(shape=(NUM_EPOCH, 3, 2))#epoch * 3 layers  *  2 (avg var)
    eval_data = np.zeros(shape=(EVAL_DATA_NUM, layer1.getInputDataDims()))
    for i in range(EVAL_DATA_NUM):
        idx = np.random.randint(0, NUM_IMAGES)
        eval_data[i] = minst_data[idx].reshape((layer1.getInputDataDims()))
    eval_input_data, distances_matrix, eval_dist_data, eval_avgs, eval_vars = build_multilayer_eval_model(neuron_competition, eval_data.transpose(), distance, [layer1, layer2, layer3])
    #training and evaluating model
    init_data = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_data)
        layer1.initWeight(sess)
        layer2.initWeight(sess)
        layer3.initWeight(sess)
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        start_time = time.time()
        for epoch in range(NUM_EPOCH):
            for image_idx in range(NUM_IMAGES):
                feed_dict = {input_data: minst_data[image_idx,]}
                layer_output1, new_weights1, new_threshold1, out_energy1, threshold_energy1, weights_energy1, weights_loss1 = layer1.trainLayer(sess, feed_dict)
                layer_output2, new_weights2, new_threshold2, out_energy2, threshold_energy2, weights_energy2, weights_loss2 = layer2.trainHiddenLayer(sess)
                layer_output3, new_weights3, new_threshold3, out_energy3, threshold_energy3, weights_energy3, weights_loss3 = layer3.trainHiddenLayer(sess)
       
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print('epoch: {} elapse time:{:.2f} sec '.format(epoch, elapsed_time))
            print('layer1 -- weight loss: {} weights_energy:{}'.format(weights_loss1, weights_energy1))
            print('layer2 -- weight loss: {} weights_energy:{}'.format(weights_loss2, weights_energy2))
            print('layer3 -- weight loss: {} weights_energy:{}'.format(weights_loss3, weights_energy3))
            
            '''
            evaluate model
            '''
            print("evalating test data...")
            avg_result, var_result = eval_multilayer_eval_model(sess, 3, eval_data.transpose(), eval_input_data, distances_matrix, eval_dist_data, eval_avgs, eval_vars)
            eval_summary[epoch, :, 0] = avg_result
            eval_summary[epoch, :, 1] = var_result
            for i in range(3):
                print("layer : {} mean : {} var: {}".format(i, avg_result[i], var_result[i]))
    
        #show the avg and var
        plt.figure(1)
        plt.title('Test Results')
        x_axis = [i for i in range(NUM_EPOCH)]
        plt.plot(x_axis, eval_summary[:,0,0], label="mean-1")
        plt.plot(x_axis, eval_summary[:,0,1], label="var-1")
        plt.plot(x_axis, eval_summary[:,1,0], label="mean-2")
        plt.plot(x_axis, eval_summary[:,1,1], label="var-2")
        plt.plot(x_axis, eval_summary[:,2,0], label="mean-3")
        plt.plot(x_axis, eval_summary[:,2,1], label="var-3")
        #show the weights
        dbg_show_weights(new_weights1, name='weights-layer1')
        dbg_show_weights(new_weights2, name='weights-layer2')
        dbg_show_weights(new_weights3, name='weights-layer3')
    return

def AssociateLayerExperiment():
    '''
    check the program code of associatelayer
    '''
    #minst data
    minst_data = extract_minst_data('data/train-images-idx3-ubyte.gz', NUM_IMAGES)
    minst_data = sharp_minst_data(minst_data)
   
    #build model
    threshold_update = UPDATE_THRESHOLD #NO_UPDATE_THRESHOLD
    weights_update = WEIGHT_LEAK#WEIGHT_RENORM
    loop = 2 #if <0 loop until convergence
    associate_mode = ASSOCIATE_MODE_BSB#ASSOCIATE_MODE_HOPFIELD
    
    alayer = AssociationLayer(ASSOCIATION_FIELD, IMAGE_SIZE, IMAGE_SIZE, LEARNING_RATE)
    alayer.buildLayer(threshold_update, weights_update)
    
    #training and evaluating model
    init_data = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_data)
        init_weights = alayer.initWeight(sess)
        
        start_time = time.time()
        for epoch in range(NUM_EPOCH):
            for image_idx in range(NUM_IMAGES):
                _, _, layer_output_energy, layer_output_loss,  _, \
                layer_weights_loss, layer_weights_energy, layer_threshold_energy = \
                alayer.trainLayer(sess, minst_data[image_idx,], loop, associate_mode)


       
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print('epoch: {} elapse time:{:.2f} sec ' \
                  'layer_output_energy: {} ' \
                  'layer_output_loss:{} ' \
                  'layer_weights_loss:{} '\
                  'layer_weights_energy:{} ' \
                  'layer_threshold_energy:{} '
                  .format(epoch, 
                          elapsed_time, 
                          layer_output_energy, 
                          layer_output_loss, 
                          layer_weights_loss, 
                          layer_weights_energy, 
                          layer_threshold_energy))

    return

def AssociateLayerExperiment_1():
    '''
    check the convergence of weights and outputs as whole. only one pic is used in a epoch.
    结果：
    1、如果有阈值更新则2轮loss为0，因为阈值导致out为零。如果将阈值设为均值的0.618，效果如何？特别是在有多个输入模式时阈值还是有必要的
    2、weight leak 方式会导致loss出现下降然后上升然后再下降的反复波动现象，关于该现象的解释尚未处理：bug还是规律？
    3、weight renorm 方式会导致loss持续下降，效果不错
    4、association_field越大loss下降的越快，但可能导致虚假模式。需要进一步实验验证。
    5、associate_mode影响不明显。
    6、assocation_loop 似乎1轮就收敛了，bug？检查一下
    '''
    print('AssociateLayerExperiment_1...')
    #minst data
    minst_data = extract_minst_data('data/train-images-idx3-ubyte.gz', NUM_IMAGES)
    minst_data = sharp_minst_data(minst_data)
   
    #build model
    threshold_update =  NO_UPDATE_THRESHOLD#NO_UPDATE_THRESHOLD, UPDATE_THRESHOLD
    weights_update = WEIGHT_RENORM#WEIGHT_RENORM, WEIGHT_LEAK
    assocation_loop = 1 #if <=0 loop until convergence
    associate_mode = ASSOCIATE_MODE_BSB#ASSOCIATE_MODE_HOPFIELD, ASSOCIATE_MODE_BSB
    learning_rate = LEARNING_RATE
    association_field = 11
    total_epoch = 100
    
    print_training_setting(-1, weights_update, threshold_update, -1,  
                           learning_rate, associate_mode, assocation_loop, association_field)
        
    alayer = AssociationLayer(association_field, IMAGE_SIZE, IMAGE_SIZE, learning_rate)
    alayer.buildLayer(threshold_update, weights_update)
    
    #training and evaluating model
    init_data = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_data)
        init_weights = alayer.initWeight(sess)
        
        start_time = time.time()
        for epoch in range(total_epoch):
            _, _, layer_output_energy, layer_output_loss,  _, \
            layer_weights_loss, layer_weights_energy, layer_threshold_energy = \
            alayer.trainLayer(sess, minst_data[0,], assocation_loop, associate_mode)
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print('epoch: {} ' \
                  #'elapse time:{:.2f} sec ' \
                  'layer_output_energy: {} ' \
                  'layer_output_loss:{} ' \
                  'layer_weights_loss:{} '\
                  #'layer_weights_energy:{} ' \
                  #'layer_threshold_energy:{} '
                  .format(epoch, 
                  #        elapsed_time, 
                          layer_output_energy, 
                          layer_output_loss, 
                          layer_weights_loss 
                   #       layer_weights_energy, 
                   #       layer_threshold_energy
                   ))

    return

def AssociateLayerExperiment_2():
    print('AssociateLayerExperiment_2...')
    association_field = 3
    layer_dim_r = 28
    layer_dim_c = 28
    layer_neuron_num = layer_dim_r*layer_dim_c
    association_radius = int(np.floor(association_field/2))
    layer_weight_mask = np.zeros([layer_neuron_num, layer_neuron_num])
    for i in range(layer_dim_r):
        for j in range(layer_dim_c):
            row = i*layer_dim_c+j
            if i<association_radius :
                top = 0
            else:
                top = i-association_radius
                
            if j<association_radius :
                left = 0
            else:
                left = j-association_radius
                
            if i>(layer_neuron_num-association_radius) :
                bottom = layer_neuron_num
            else:
                bottom = i+association_radius
            
            if j>(layer_neuron_num-association_radius) :
                right = layer_neuron_num
            else:
                right = j+association_radius
            
            for k in range(bottom-top):
                start = layer_dim_c*(top+k)+left 
                end = start+right-left+1
                layer_weight_mask[row,start:end]=1
    #remove the self-link
    for i in range(layer_neuron_num):
        layer_weight_mask[i,i]=0
    
    weights = tf.Variable(tf.random_uniform([layer_neuron_num, layer_neuron_num], minval=0.0, maxval=1))
    weights_ret = tf.Variable(tf.random_uniform([layer_neuron_num, layer_neuron_num], minval=0.0, maxval=1))
    weights_mask = weights*layer_weight_mask
    weights_norm = tf.assign(weights, tf.nn.l2_normalize(weights_mask,1))
    with tf.control_dependencies([weights_norm]):
        tf.assign(weights_ret, weights_norm)
        weights_out = tf.matmul(weights, weights_ret)
    
    with tf.control_dependencies([weights_out]):
        weights_out1 = tf.assign(weights_ret, weights_out)
        weights_enery_2 = tf.reduce_sum(tf.square(weights_out1))
        weights_enery_1 = tf.reduce_sum(weights_out1)
    
    #training and evaluating model
    init_data = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_data) 
        for i in range(10):
            e2, e1 = sess.run([weights_enery_2, weights_enery_1])
            print('energy square: {} energe:{}'.format(e2, e1))
        
    return

class EntropyLayer:
    
    def __init__(self, reception_field, input_dim_r, input_dim_c, learn_rate):
        self.reception_field = reception_field
        self.input_dim_r = input_dim_r
        self.input_dim_c = input_dim_c
        self.input_data_dims = input_dim_r*input_dim_c
        self.layer_neuron_dim_r = input_dim_r-reception_field+1#output neuron rows
        self.layer_neuron_dim_c = input_dim_c-reception_field+1#output neuron cols
        self.layer_neuron_num = self.layer_neuron_dim_r*self.layer_neuron_dim_c
        self.learn_rate = learn_rate
        return
    
    def createLayerWeightsMask(self):
        #create a mask matrix for weight
        self.layer_weight_mask = np.zeros([self.layer_neuron_num, self.input_data_dims])
        for i in range(self.layer_neuron_dim_r):
            for j in range(self.layer_neuron_dim_c):
                row = i*self.layer_neuron_dim_c+j
                for k in range(self.reception_field):
                    start = self.input_dim_c*(i+k)+j 
                    end = start+self.reception_field
                    self.layer_weight_mask[row,start:end]=1
        
        return

    
    def buildMinstTrainingData(self, samples_count):
        minst_data = extract_minst_data('data/train-images-idx3-ubyte.gz', NUM_IMAGES)
        minst_data = sharp_minst_data(minst_data)
        minst_data_label = extract_labels('data/train-labels-idx1-ubyte.gz', NUM_IMAGES)
        self.ObervedSamples = np.ndarray(shape=(10, samples_count, IMAGE_SIZE * IMAGE_SIZE, 1), #10 categories * samples * sample_size 
                                         dtype=np.float32)
        
        samples_pos = [0 for i in range(10)]
        for i in range(NUM_IMAGES):
            digit = minst_data_label[i]
            pos = samples_pos[digit]
            if(pos < samples_count):
                self.ObervedSamples[digit, pos, :, :] = minst_data[i]
                samples_pos[digit] = samples_pos[digit]+1
        
        for i in range(10):
            if(samples_pos[i]<samples_count):
                print('Warning: samples for class: {} isnot full '.format(i))
        self.TrainingData = minst_data
        self.TrainingLabel = minst_data_label
        return

    def buildLayer(self):
        
        self.train_loops = tf.Variable(0.0)#save the train times
        
        self.input_data = tf.placeholder(tf.float32, shape=(self.input_data_dims, 1))#训练数据
        self.observed_data = tf.placeholder(tf.float32, shape=(self.ObervedSamples.shape))#已知类别数据
        oldshape = self.ObervedSamples.shape
        newshape = (oldshape[0]*oldshape[1], oldshape[2]*oldshape[3])#(10*samples_count, IMAGE_SIZE * IMAGE_SIZE*1)
        
        total_samples_count = oldshape[0]*oldshape[1]
        observed_data_2d = tf.transpose(tf.reshape(self.observed_data, shape=newshape))#(input_data_dims, total_samples_count)

        self.layer_weights = tf.Variable(tf.random_uniform([self.layer_neuron_num, self.input_data_dims], minval=0.0, maxval=1.0))
        self.createLayerWeightsMask()
        layer_mask_weights = tf.multiply(self.layer_weights, self.layer_weight_mask)#(layer_neuron_num, input_data_dims)
        layer_weights_norm = tf.nn.l2_normalize(layer_mask_weights,1)
        
        #init threshold
        self.layer_threshold = tf.Variable(tf.zeros(shape=(self.layer_neuron_num,1)))#(layer_neuron_num,1)
        
        #计算x与xs的输出,暂时未考虑控制系数c
        #(layer_neuron_num, input_data_dims)*(input_data_dims, 1) = (layer_neuron_num, 1)
        y_x0  = tf.matmul(layer_weights_norm, self.input_data)
        #y_x = tf.sigmoid(tf.multiply(y_x0, tf.to_float(tf.less_equal(self.layer_threshold, y_x0))))#如果超过阈值，则激活，且激活值不应减掉阈值
        y_x = tf.nn.relu(y_x0-self.layer_threshold)
        #(layer_neuron_num, input_data_dims)*(input_data_dims, total_samples_count) = (layer_neuron_num, total_samples_count)
        y_xs0 = tf.matmul(layer_weights_norm, observed_data_2d)
        #y_xs = tf.sigmoid(tf.multiply(y_xs0, tf.to_float(tf.less_equal(tf.tile(self.layer_threshold, [1, total_samples_count]), y_xs0))))#如果超过阈值，则激活，且激活值不应减掉阈值
        y_xs = tf.nn.relu(y_xs0-tf.tile(self.layer_threshold, [1, total_samples_count]))
        #计算概率
        delta_y = tf.subtract(tf.tile(y_x, [1,total_samples_count]), y_xs)#(layer_neuron_num, total_samples_count)
        dist = tf.reduce_mean(tf.square(delta_y), 0)#(1, total_samples_count)
        #计算距离的均值和方差-试验用
        self.dist_mean, self.dist_var = tf.nn.moments(dist, 0)
        #gauss influence - 窗口宽度为1
        influence = tf.exp((-0.5)*dist)#(1, total_samples_count)
        self.prob_y = tf.reduce_mean(influence)
        #计算delt_w
        #TODO:sigmoid
        delta_x = tf.subtract(tf.tile(self.input_data, [1,total_samples_count]), observed_data_2d)#(input_data_dims, total_samples_count)
        delta_y_inference = tf.multiply(delta_y, influence)
        delta_y_3ds = tf.reshape(delta_y_inference, shape=(1, self.layer_neuron_num, total_samples_count))
        delta_x_3ds = tf.reshape(delta_x, shape=(1, self.input_data_dims, total_samples_count))
        delta_y_3d = tf.tile(delta_y_3ds,[self.input_data_dims, 1, 1])
        delta_x_3d = tf.transpose(tf.tile(delta_x_3ds,[self.layer_neuron_num, 1, 1]),
                                  perm=[1,0,2])
        delta_w = tf.transpose(tf.reduce_sum(tf.multiply(delta_y_3d, delta_x_3d), 2)*tf.log(2*self.prob_y)*(-1.0)*(self.learn_rate)) #(layer_neuron_num, input_data_dims)
        #更新权重
        self.update_weight_ops = tf.assign(self.layer_weights, tf.add(layer_weights_norm, delta_w))
        #threshold update
        with tf.control_dependencies([self.prob_y]):
            y_avg = tf.divide(tf.add(y_x0, tf.reshape(tf.reduce_sum(y_xs0, 1), tf.shape(y_x0))), (total_samples_count+1))
            self.layer_update_threshold = tf.assign(self.layer_threshold, 
                                                        tf.divide(tf.add(self.layer_threshold*self.train_loops, y_avg), 
                                                                  self.train_loops.assign_add(1)
                                                                  )
                                                        )

        #feedforward -  to pre-training the output for FC
        self.batch_data = tf.placeholder(tf.float32, shape=[self.input_data_dims, None]) #[input_data_dims, batch_size]
        batch_out0 = tf.matmul(layer_weights_norm, self.batch_data)#[layer_neuron_num, batch_size]
        #self.batch_out = tf.nn.relu(batch_out0-tf.tile(self.layer_threshold, [1, tf.shape(batch_out0)[1]]))
        self.batch_out = tf.nn.relu(batch_out0-self.layer_threshold)
        return
    
    def trainLayer(self, sess):
        '''
        training the layer with the train data of self.TrainingData
        '''
        for i in range(NUM_IMAGES):
            p, nw, nt, dist_mean, dist_var = sess.run([self.prob_y, self.update_weight_ops, self.layer_update_threshold, self.dist_mean, self.dist_var], 
                     feed_dict={self.input_data:self.TrainingData[i, :, :], self.observed_data:self.ObervedSamples})
            #print_matrix(nw, 1)
            #print('prob:{}'.format(p))
            #print('dist mean:{} var:{}'.format(dist_mean, dist_var))
            #print('entropy:{}'.format(p*np.log2(p)))
        return
    
    def evaluateEntropy(self, sess, evaluate_data):
        '''
        NOTE: buildLayer MUST be called in advance.
        sess: tensorflow running context
        evaluate_data: test data with the format of [test_data_count, input_data_dims, 1]
        '''
        test_data_count = evaluate_data.shape[1]
        entropy = 0.0
        for i in range(test_data_count):
            h = sess.run([self.prob_y], 
                         feed_dict={self.input_data:evaluate_data[i,:,:], self.observed_data:self.ObervedSamples})
            entropy = entropy+h*np.log2(h)
            #print('evaluate prob:{}'.format(h))
        return (-1)*entropy

    def evaluateFeedForward(self, sess, batch_data):
        out = sess.run([self.batch_out], feed_dict = {self.batch_data:batch_data})
        return out
        

def EntropyLayerExperiment():
    '''
    2017-11-29:
        实验结果与learning rate有关。当rate减小，H也降低。总体来说小的rate的结果曲线收敛的不错。原因可能在于H曲线
        大部分曲率大，导致结果震荡。如加上sigmoid，则压缩了y，所以结果要好一些。
        最大问题是：距离函数对概率的估计问题，由于维度高，距离不具有区分能力（从统计来看方差很小）。是不是真正与x接近的样本
        被绝大多数其他样本给覆盖了？如何改进？
    2018-1-14:
        从最近看才材料来看，如果以流形学习角度解释的话，我的方式实际上是沿着流形的切面学习，而不是垂直切面。按照流形学习解释，沿着切面学习可以更好的
        学习分类平面，而垂直方向，不会改变分类平面方向，不太好。例如，由人脸构成的流形，切面方向移动时，切换到不同的人脸，而垂直，只是同一张脸的明暗变化。
    2018-1-31:
        增加阈值。希望能够降低输出的维度，进而使得Parzen窗中的距离函数有效。进一步：可以引入注意力或其他机制，只关注部分区域。
        修改了部分计算公式错误：之前的influence没有分别调整各个神经元。
        目前，从实际效果来看，较没加之前波动比较大，重新设定rate更小后效果还可以。
        为什么H值与rate有关？有bug吗？查了相关资料，好像过大的rate会导致学习不稳定或效果差
    '''
    print('EntropyLayerExperiment...')
    reception_field = 7
    input_dim_r = IMAGE_SIZE
    input_dim_c = IMAGE_SIZE
    learn_rate = 0.001
    total_epoch = 10
    samples_count = 10
    print('(reception_field:{} \n input_dim_r:{} \n input_dim_c:{} \n learn_rate:{} \n total_epoch:{} \n samples_count:{})'.format(
            reception_field, input_dim_r, input_dim_c, learn_rate, total_epoch, samples_count))
    
    eval_summary = np.zeros(shape=(total_epoch))
    
    alayer = EntropyLayer(reception_field, input_dim_r, input_dim_c, learn_rate)
    alayer.buildMinstTrainingData(samples_count)
    alayer.buildLayer()
    
    print('training ... ')
    init_data = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_data)
        
        start_time = time.time()
        for epoch in range(total_epoch):
            alayer.trainLayer(sess)
            #print_matrix(sess.run(alayer.layer_weights), 1)
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print('training epoch: {} elapse time:{:.2f} sec'.format(epoch, elapsed_time))

            entropy = alayer.evaluateEntropy(sess, alayer.TrainingData)
            eval_summary[epoch] =  entropy
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print('evaluating epoch: {} elapse time:{:.2f} sec entropy:{}'.format(epoch, elapsed_time, entropy))            
        #show the avg and var
        plt.figure(1)
        plt.title('Test Results')
        x_axis = [i for i in range(total_epoch)]
        plt.plot(x_axis, eval_summary, label="Entropy")
        dbg_show_weights(sess.run(tf.multiply(alayer.layer_weights, alayer.layer_weight_mask)), name='weights-entropy-2')
    return

def EntropyLayerExperiment_1():
    
    print('EntropyLayerExperiment_1...')
    #FC Params
    batch_size = 100
    hidden1 = 128
    hidden2 = 32
    learning_rate = 0.01
    log_dir = 'log'
    max_steps = 2000
    input_dims = IMAGE_SIZE*IMAGE_SIZE
    num_classes = 10
    #EntropyLayer Params
    reception_field = 7
    input_dim_r = IMAGE_SIZE
    input_dim_c = IMAGE_SIZE
    total_epoch = 2
    samples_count = 10
    
    train_num = 6000
    validation_num = 1000 #validation_num<train_num
    test_num = 1000
    
    print('(reception_field:{} \n input_dim_r:{} \n input_dim_c:{} \n learn_rate:{} \n total_epoch:{} \n samples_count:{})'.format(
            reception_field, input_dim_r, input_dim_c, learning_rate, total_epoch, samples_count))
    
    eval_summary = np.zeros(shape=(total_epoch))
        
    #build dataset
    minst_train_data = extract_minst_data('data/train-images-idx3-ubyte.gz', train_num)
    minst_train_data = sharp_minst_data(minst_train_data).reshape(train_num, IMAGE_SIZE, IMAGE_SIZE, 1)
    minst_train_labels = extract_labels('data/train-labels-idx1-ubyte.gz', train_num).astype(np.uint8)
    
    minst_test_data = extract_minst_data('data/t10k-images-idx3-ubyte.gz', test_num)
    minst_test_data = sharp_minst_data(minst_test_data).reshape(test_num, IMAGE_SIZE, IMAGE_SIZE, 1)
    minst_test_labels = extract_labels('data/t10k-labels-idx1-ubyte.gz', test_num).astype(np.uint8)

#    validation_images = minst_train_data[:validation_num]
#    validation_labels = minst_train_labels[:validation_num]
#    train_images = minst_train_data[validation_num:]
#    train_labels = minst_train_labels[validation_num:]
#    train = DataSet(train_images, train_labels)
#    validation = DataSet(validation_images, validation_labels)
#    test = DataSet(minst_test_data, minst_test_labels)
    
    # FC without Entropy pre-train
#    fc.run_training(train, validation, test, batch_size, hidden1, hidden2, 
#                    learning_rate, log_dir, max_steps, input_dims, num_classes)

    
    
    # FC with Entropy pre-train
    
    alayer = EntropyLayer(reception_field, input_dim_r, input_dim_c, learning_rate)
    alayer.buildMinstTrainingData(samples_count)
    alayer.buildLayer()
    
    print('Entropy pre-train ... ')
    init_data = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_data)
        
        start_time = time.time()
        for epoch in range(total_epoch):
            alayer.trainLayer(sess)
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print('training epoch: {} elapse time:{:.2f} sec'.format(epoch, elapsed_time))

            entropy = alayer.evaluateEntropy(sess, alayer.TrainingData)
            eval_summary[epoch] =  entropy
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print('evaluating epoch: {} elapse time:{:.2f} sec entropy:{}'.format(epoch, elapsed_time, entropy))            
        
        #create new data with pre-train
        minst_train_data =np.reshape(minst_train_data, newshape=(train_num, input_dims)).transpose()
        minst_train_data = alayer.evaluateFeedForward(sess, minst_train_data)
        oldshape = np.shape(minst_train_data)
        newshape = (oldshape[2], oldshape[1], oldshape[0])
        minst_train_data =np.transpose(minst_train_data).reshape(newshape)
        
        minst_test_data =np.reshape(minst_test_data, newshape=(test_num, input_dims)).transpose()
        minst_test_data = alayer.evaluateFeedForward(sess, minst_test_data)
        oldshape = np.shape(minst_test_data)
        newshape = (oldshape[2], oldshape[1], oldshape[0])
        minst_test_data =np.transpose(minst_test_data).reshape(newshape)

        validation_images = minst_train_data[:validation_num]
        validation_labels = minst_train_labels[:validation_num]
        train_images = minst_train_data[validation_num:]
        train_labels = minst_train_labels[validation_num:]
        
        train = DataSet(train_images, train_labels)
        validation = DataSet(validation_images, validation_labels)
        test = DataSet(minst_test_data, minst_test_labels)  

        fc.run_training(train, validation, test, batch_size, hidden1, hidden2, 
                        learning_rate, log_dir, max_steps, input_dims, num_classes)
        
    return
    
#################################################################
#EntropyLayer end
#################################################################
        
def PCA(_):
    """
    parameters
    """
    data_size = 100
    class_num = 4
    data_dims = 2
    rho = 0.5
    rho_delta = 0.5
    noisy_delta = 0.02
    learning_rate = 0.01#似乎对结果的质量影响不大，只对收敛速度有影响，再找些数据集测试一下。当rho比较时，rate应较大，否则maxcompetition差，avg competition 对于小rho来说效果不好。
    epoch = 10000
    AMP_COF = 100 #放大系数，当rho很小时，即使平方后，差值通常在百分位才有区别。可利用该系数放大差值。主要用于avg competitive
    
    print("rho_delta: {} noisy_delta: {} learning_rate: {} loops:{}".format(rho_delta, noisy_delta, learning_rate, epoch))
    
    """
    synatic data prepare
    """
    pca_data = np.zeros(shape=(class_num,data_size,data_dims))
    theta = np.zeros(class_num)
    for class_idx in range(class_num):
        theta[class_idx] = 1.57/(class_num+1)*(class_idx+1)
        rho_rand = np.abs(np.random.normal(rho, rho_delta, data_size))
        theta_rand = np.abs(np.random.normal(theta[class_idx], noisy_delta, data_size))
        pca_data[class_idx] = np.transpose([rho_rand*np.cos(theta_rand), rho_rand*np.sin(theta_rand)])
    """
    initial weights
    """
    w = np.random.uniform(0,1, data_dims*class_num)
    w = np.reshape(w,(class_num,data_dims))
    denomator = np.reshape(np.sqrt(np.sum(np.square(w), axis = 1)), (class_num,1))
    w = w/denomator
    init_w = w.copy()
    w_delta_plot = [];
    
    """
    PCA Benchmark
    """
    for loop in range(epoch):
        i = np.random.randint(0,data_size)
        i = loop%data_size
        x = np.zeros(shape=(class_num, data_dims))
        for j in range(class_num):
            x[j]=pca_data[j][i]
        
        y = np.reshape(np.sum(w*x, axis=1), (class_num,1))
        w0 = w.copy()
        w=w+learning_rate*x*y
        denomator = np.reshape(np.sqrt(np.sum(np.square(w), axis = 1)), (class_num,1))
        w = w/denomator
        
        w_delta = w-w0
        w_delta = np.sum(np.square(w_delta))/class_num
        w_delta_plot.append(w_delta) 
 
    plt.figure(1)
    plt.title('pca-benchmark: raw data v.s. pca vector')
    for c in range(class_num):
        plt.scatter(pca_data[c,:,0], pca_data[c,:,1], marker='x', color = 'm')
    plt.scatter(init_w[:,0], init_w[:,1], marker='s', color='b')
    plt.scatter(w[:,0], w[:,1], marker='o', color='r')
    plt.figure(2)
    plt.title('pca-benchmark: weight delta curve')
    x_axis = [i for i in range(epoch)]
    plt.plot(x_axis, w_delta_plot)
    
   
    """
    PCA competitive learning : winner-take-all
    对初始向量很敏感，如果比较相近则很难收敛到合适的结果。另外，如果初始向量正好处于两类中间，则结果很差
    """
    w1 = init_w.copy()
    w1_delta_plot = []
    for loop in range(epoch):
        w0_1 = w1.copy()
        
        c_i = np.random.randint(0, class_num);
        i = np.random.randint(0,data_size);
        x = pca_data[c_i][i];
        
        y = w1.dot(x)
        winner_idx = y.argmax()
        w1[winner_idx] = w1[winner_idx]+learning_rate*x*y[winner_idx]
        denomator = np.sqrt(np.sum(np.square(w1[winner_idx])))
        w1[winner_idx] = w1[winner_idx]/denomator    
        w1_delta = w1-w0_1
        w1_delta = np.sum(np.square(w1_delta))/class_num
        w1_delta_plot.append(w1_delta)
        

    plt.figure(3)
    plt.title('pca-competitve - wta: raw data v.s. pca vector')
    for c in range(class_num):
        plt.scatter(pca_data[c,:,0], pca_data[c,:,1], marker='x', color = 'm')
    plt.scatter(init_w[:,0], init_w[:,1], marker='s', color='b')
    plt.scatter(w1[:,0], w1[:,1], marker='o', color='r')
    plt.figure(4)
    plt.title('pca-competitve - wta: weight delta curve')
    x_axis = [i for i in range(epoch)]
    plt.plot(x_axis, w1_delta_plot)

    """
    PCA competitive learning : resource-competitive
    （1）趋向一个结果：因为y都差不多，目前的平方标准化方法对于y间差值放大不够
    经过修改采用当前放大方法，效果不错。具体为：平方后与均值的差值的指数形式
    （2）采用renormalize方法标准化每次更新后的权重，主要问题是pca间区别度太小。如果
    采用leak方法，pca的模不等于1，但从实验来看也没有无限变大。leak实验效果不错（学习率要小）。
    （3）threshold过滤。使得pca变化与benchmark基本一致，如果不加会一直波动
    """
    w2 = init_w.copy()
    w2_delta_plot = []
    threshold = np.zeros(class_num)
    threshold_plot = np.zeros((class_num, epoch))
    for loop in range(epoch):
        w0_2 = w2.copy()
        
        c_i = np.random.randint(0, class_num);
        i = np.random.randint(0,data_size);
        x = pca_data[c_i][i];
        
        
        y = w2.dot(x)
        '''
        增加了利用阈值过滤激活单元的方法，但从实际效果来看比较差，只能滤掉信号强度弱的输入，但无法通过阈值区分各个神经元，因为都差不多。
        delta_y=x(m1-m2),则m1和m2正交，且x与差向量同向时最大。当这两个条件很难满足
        '''
        
        threshold = (threshold*loop+y)/(loop+1)
        threshold_plot[:,loop] = threshold
        
        '''
        threshold过滤。平均值过滤。使得pca变化与benchmark基本一致，如果不加会一直波动
        '''
        y2 = np.less(threshold, y)
        y3 = y2.astype(float)
        y = y*y3
        
        #放大差值
        y1=np.square(y)*AMP_COF
        y_amp = np.exp(y1-np.average(y1))
        denomator_y = np.sqrt(np.sum(np.square(y_amp)))#有可能溢出，超精度
        y = y*y_amp/denomator_y
            
        x = np.reshape(x, (1,data_dims))
        y = np.reshape(y, (class_num,1))
        
        '''
        update weights: re-normalize the weights
        
        w2=w2+learning_rate*y.dot(x)
        denomator = np.reshape(np.sqrt(np.sum(np.square(w2), axis = 1)), (class_num,1))
        w2 = w2/denomator
        '''
        
        '''
        update weights:  in an manner of leaking 
        '''
        w2 = w2 + learning_rate*(y.dot(x)-np.square(y)*w2)
        '''
        compute the delta of w change
        '''
        w2_delta = w2-w0_2
        w2_delta = np.sum(np.square(w2_delta))/class_num
        w2_delta_plot.append(w2_delta)
        
        #print('x:{} \n y:{}'.format(x,y))
        #print('w2:{} \n w0_2:{}'.format(w2, w0_2))    
    
    plt.figure(5)
    plt.title('pca-competitve - RC: raw data v.s. pca vector')
    for c in range(class_num):
        plt.scatter(pca_data[c,:,0], pca_data[c,:,1], marker='x', color = 'm')
    plt.scatter(init_w[:,0]*rho, init_w[:,1]*rho, marker='s', color='b')
    plt.scatter(w2[:,0], w2[:,1], marker='o', color='r')
    plt.figure(6)
    plt.title('pca-competitve - RC: weight delta curve')
    x_axis = [i for i in range(int(epoch))]
    plt.plot(x_axis, w2_delta_plot)
    plt.figure(7)
    plt.title('pca-competitve - RC: threshold mean curve')
    for j in range(class_num):
        plt.plot(x_axis, threshold_plot[j])
        
    

if __name__ == '__main__':
    #SingleFWLayerExperiment()
    #MultiFWLayerExperiment()
    #AssociateLayerExperiment_1()
    #test()
    #PCA(_)
    #EntropyLayerExperiment()
    EntropyLayerExperiment_1()