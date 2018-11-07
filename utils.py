import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import pickle
from PIL import Image

os.chdir(os.path.dirname(__file__))
#train = pd.read_csv('./train.csv')

def create_dict():

    if os.path.exists('./label.dump') == False:
        ids = set(list(train['Id']))
        label_dict = {}
        for i, label in enumerate(ids):
            label_dict[label] = i

        with open('label.dump', 'wb') as f:
            pickle.dump(label_dict, f)

    label_dict = pickle.load(open('label.dump', 'rb'))

    return label_dict


def img_read(img_name, classifier=False):
    if classifier:
        PATH = './Resized'
    else:
        PATH = './Pictures'

    img = cv2.imread(os.path.join(PATH, img_name), cv2.IMREAD_COLOR)

    return img

def img_read_VOC(img_name, SegmentationObject=True, size=None, normalization=True):

    PATH = './VOC2012'
    if SegmentationObject:
        image_path = os.path.join(PATH, 'SegmentationObject')
        img = Image.open(os.path.join(image_path, img_name))

    else:
        image_path = os.path.join(PATH, 'JPEGImages')
        img = Image.open(os.path.join(image_path, img_name))

    if size is not None:
        resize_size = (size, size)
        img = img.resize(resize_size)

    img = np.asarray(img)

    if normalization:
        img = img / 255 #画像の正規化, 画像を(0.0~1.0に)正規化するなどの前処理が必要です

    return img

class VOC_dataset():

    def __init__(self):

        img_list, ori_imgs, seg_imgs = [], [], []

        PATH = './VOC2012/ImageSets/Segmentation/train.txt'
        with open(PATH, "r") as f:
            for line in f:
                img_list.append(line.rstrip('\n'))

        for img_name in img_list[:10]:

            ori_img_name = img_name + '.jpg'
            seg_img_name = img_name + '.png'

            ori_img = img_read_VOC(ori_img_name, SegmentationObject=False, size=512, normalization=True)
            seg_img = img_read_VOC(seg_img_name, SegmentationObject=True, size=512, normalization=False)
            seg_img = np.where(seg_img == 255, 21, seg_img) #NUM_CLASSES=21 21はVOID
            seg_img_one_hot = self.make_one_hot(seg_img) #for sparse_softmax_cross_entropy_with_logits

            ori_imgs.append(ori_img)
            seg_imgs.append(seg_img_one_hot)

        assert len(ori_imgs) == len(seg_imgs)

        self.ori_imgs_set = ori_imgs
        self.seg_imgs_set = seg_imgs

    def next_batch(self, batch_size):

        idx = np.arange(0 , len(self.ori_imgs_set))
        np.random.shuffle(idx)
        idx = idx[:batch_size]

        train_ori_imgs = [self.ori_imgs_set[i] for i in idx]
        train_seg_imgs = [self.seg_imgs_set[i] for i in idx]

        return np.asarray(train_ori_imgs,  dtype=np.float32), np.asarray(train_seg_imgs, dtype=np.int32)

    def make_one_hot(self, data):
        # The source code from: https://qiita.com/tktktks10/items/0f551aea27d2f62ef708
        # One hot encoding using identity matrix.
        #https://qiita.com/JeJeNeNo/items/8a7c1781f6a6ad522adf
        #target_vector = [0,2,1,3,4]               # クラス分類を整数値のベクトルで表現したもの
        #n_labels = len(np.unique(target_vector))  # 分類クラスの数 = 5
        #np.eye(n_labels)[target_vector]           # one hot表現に変換


        identity = np.identity(22, dtype=np.uint8)
        img_one_hot = identity[data]

        return img_one_hot

def next_batch(batch_size):

    imgs_name = list(train['Image'])
    labels = list(train['Id'])
    label_dict = create_dict()

    idx = np.arange(0 , len(imgs_name))
    np.random.shuffle(idx)
    idx = idx[:batch_size]

    img_name_shuffles = [imgs_name[i] for i in idx]
    label_shuffles = [label_dict[labels[i]] for i in idx]

    imgs_shuffles = []
    for img_name in img_name_shuffles:
        img = img_read(img_name, classifier=True)
        imgs_shuffles.append(img)

    return np.asarray(imgs_shuffles), np.asarray(label_shuffles)

def create_weight(shape, scope=None):
    weight = tf.truncated_normal(shape)
    return tf.Variable(weight)

def create_bias(shape, scope=None):
    bias = tf.constant(0.0, shape=shape, dtype=tf.float32)
    return tf.Variable(bias)

def batch_normal(inputs, is_training):
    return tf.layers.batch_normalization(inputs,
                                         training=is_training)

def conv_2d(inputs, kernel_size, num_output, step=1, activation=tf.nn.relu, padding='SAME', is_training=None, scope=None):
    with tf.variable_scope(scope):
        num_input = inputs.get_shape().as_list()[3]
        bias = create_bias([num_output])
        kernel = create_weight([kernel_size, kernel_size, num_input, num_output])
        stride = [1, step, step, 1]
        conv = tf.nn.conv2d(inputs, kernel, stride, padding=padding) + bias

        if is_training is not None:
            conv = batch_normal(conv, is_training)

        if activation is not None:
            conv_output = activation(conv)
        else:
            conv_output = conv

    return conv_output

def conv2_tp(inputs, kernel_size, num_output, output_shape=None, step=2, padding='SAME', scope=None):
    if output_shape is None:
        output_shape = inputs.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = num_output

    with tf.variable_scope(scope):

        inputs_shape = inputs.get_shape().as_list()
        num_input = inputs_shape[3]
        bias = create_bias([num_output])
        kernel = create_weight([kernel_size, kernel_size, num_output, num_input])

        #filter: A 4-D Tensor with the same type as value and shape [height, width, output_channels, in_channels].
        #filter's in_channels dimension must match that of value.

        deconv = tf.nn.conv2d_transpose(inputs,
                                        kernel,
                                        output_shape,
                                        strides = [1, step, step, 1],
                                        padding=padding)
    return tf.add(deconv, bias)

def max_pool(inputs, pool_size, step, padding='SAME',scope=None):
    with tf.variable_scope(scope):
        max_pool = tf.nn.max_pool(inputs,
                                  ksize=[1, pool_size, pool_size, 1], #Filter [2, 2]
                                  strides=[1, step, step, 1],
                                  padding=padding)

    return max_pool

def fully_connected(inputs, NUM_CLASSES, activation_fn=tf.nn.relu ,scope=None):
    with tf.variable_scope(scope):
        fc = tf.contrib.layers.fully_connected(inputs,
                                               NUM_CLASSES,
                                               activation_fn=activation_fn,
                                               weights_initializer = tf.contrib.layers.xavier_initializer(),
                                               biases_initializer = tf.zeros_initializer())

    return fc

def entropy_cost(logits, label):
    cross_entropy = -tf.reduce_sum(label*tf.log(tf.clip_by_value(logits,1e-10,1.0)))
    cost = tf.reduce_mean(cross_entropy)

    return cost

def training(cost):
    max_gradient_norm = 5
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    params = tf.trainable_variables()
    gradients = tf.gradients(cost, params)
    clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
    training_op = optimizer.apply_gradients(zip(clipped_gradients, params))

    return training_op

def accuracy(label, annotion_pred):
    correct_pred = tf.equal(annotion_pred,tf.argmax(label, axis=3))
    accuarcy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuarcy
