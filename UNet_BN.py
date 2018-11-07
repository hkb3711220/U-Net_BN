import tensorflow as tf
import time
import os
import utils
from utils import VOC_dataset
import matplotlib.pyplot as plt
import numpy as np

#########################################################################
#Model:https://qiita.com/tktktks10/items/0f551aea27d2f62ef708
#https://blog.csdn.net/qq_30239975/article/details/79454205
#########################################################################

cwd = os.getcwd()

class U_Net(object):

    def __init__(self):
        self.n_epoch = 1
        self.IMG_SIZE = 512
        self.batch_size = 1
        self.NUM_CLASSES = 22

    def Main(self, inputs, is_training):

        #DOWNSAMPLING

        conv1 = utils.conv_2d(inputs, kernel_size=3, num_output=64, step=1, padding='SAME', is_training=is_training, scope='conv1')
        conv2 = utils.conv_2d(conv1, kernel_size=3, num_output=64, step=1, padding='SAME',is_training=is_training, scope='conv2')
        max_pool3 = utils.max_pool(conv2, pool_size=2, step=2, padding='SAME', scope='pool1')

        conv4 = utils.conv_2d(max_pool3, kernel_size=3, num_output=128, step=1, padding='SAME',is_training=is_training, scope='conv4')
        conv5 = utils.conv_2d(conv4, kernel_size=3, num_output=128, step=1, padding='SAME',is_training=is_training, scope='conv5')
        max_pool6 = utils.max_pool(conv5, pool_size=2, step=2, padding='SAME', scope='pool2')

        conv7 = utils.conv_2d(max_pool6, kernel_size=3, num_output=256, step=1, padding='SAME',is_training=is_training, scope='conv7')
        conv8 = utils.conv_2d(conv7, kernel_size=3, num_output=256, step=1, padding='SAME',is_training=is_training, scope='conv8')
        max_pool9 = utils.max_pool(conv8, pool_size=2, step=2, padding='SAME', scope='pool3')

        conv10 = utils.conv_2d(max_pool9, kernel_size=3, num_output=512, step=1, padding='SAME',is_training=is_training, scope='conv10')
        conv11 = utils.conv_2d(conv10, kernel_size=3, num_output=512, step=1, padding='SAME',is_training=is_training, scope='conv11')
        max_pool12 = utils.max_pool(conv11, pool_size=2, step=2, padding='SAME', scope='pool4')

        conv13 = utils.conv_2d(max_pool12, kernel_size=3, num_output=1024, step=1, padding='SAME',is_training=is_training, scope='conv13')
        conv14 = utils.conv_2d(conv13, kernel_size=3, num_output=1024, step=1, padding='SAME',is_training=is_training, scope='conv14')

        #START UPSAMPLING
        deconv1 = utils.conv2_tp(conv14, kernel_size=2, num_output=512, step=2, padding='SAME', scope='deconv1')
        concated1 = tf.concat([deconv1, conv11], axis=3, name='concated1')
        conv15 = utils.conv_2d(concated1, kernel_size=3, num_output=512, step=1, padding='SAME', is_training=is_training, scope='conv15')
        conv16 = utils.conv_2d(conv15, kernel_size=3, num_output=512, step=1, padding='SAME', is_training=is_training, scope='conv16')

        deconv2 = utils.conv2_tp(conv16, kernel_size=2, num_output=256, step=2, padding='SAME', scope='deconv2')
        concated2 = tf.concat([deconv2, conv8], axis=3, name='concated2')
        conv17 = utils.conv_2d(concated2, kernel_size=3, num_output=256, step=1, padding='SAME', is_training=is_training, scope='conv17')
        conv18 = utils.conv_2d(conv17, kernel_size=3, num_output=256, step=1, padding='SAME', is_training=is_training, scope='conv18')

        deconv3 = utils.conv2_tp(conv18, kernel_size=2, num_output=128, step=2, padding='SAME', scope='deconv3')
        concated3 = tf.concat([deconv3, conv5], axis=3, name='concated3')
        conv19 = utils.conv_2d(concated3, kernel_size=3, num_output=128, step=1, padding='SAME', is_training=is_training, scope='conv19')
        conv20 = utils.conv_2d(conv19, kernel_size=3, num_output=128, step=1, padding='SAME', is_training=is_training, scope='conv20')

        deconv4 = utils.conv2_tp(conv20, kernel_size=2, num_output=64, step=2, padding='SAME', scope='deconv4')
        concated4 = tf.concat([deconv4, conv2], axis=3, name='concated4')
        conv21 = utils.conv_2d(concated4, kernel_size=3, num_output=64, step=1, padding='SAME', is_training=is_training, scope='conv21')
        conv22 = utils.conv_2d(conv21, kernel_size=3, num_output=64, step=1, padding='SAME', is_training=is_training, scope='conv22')

        outputs = utils.conv_2d(conv22, kernel_size=1, num_output=self.NUM_CLASSES, step=1,
                               activation=None, padding='SAME', scope='conv23')

        annotation_pred = tf.argmax(outputs, axis=3, name='prediction')

        return outputs, annotation_pred

    def Train(self):

        images = tf.placeholder(tf.float32, shape=[self.batch_size, self.IMG_SIZE, self.IMG_SIZE, 3], name="images")
        annotation = tf.placeholder(tf.float32, shape=[self.batch_size, self.IMG_SIZE, self.IMG_SIZE, 22], name="annotation")
        is_training = tf.placeholder_with_default(False, shape=(), name='is_training')

        #training
        output, annotation_pred = self.Main(images, is_training)
        cost = utils.entropy_cost(logits=output, label=annotation)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        training_op = utils.training(cost

        #Check
        accuracy = utils.accuracy(annotation, annotation_pred)
        saver = tf.train.Saver()

        min_loss = None
        with tf.Session() as sess:
            print("==============================start train:{}==============================".format(time.ctime()))
            init = tf.global_variables_initializer()
            sess.run(init)

            tf.summary.scalar("entropy", cost)
            tf.summary.scalar('accuracy', accuracy)
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter(cwd  + '/log', sess.graph)

            train_ori_imgs, train_seg_imgs = VOC_dataset().next_batch(self.batch_size)

            for epoch in range(self.n_epoch):
                train_ori_imgs, train_seg_imgs = VOC_dataset().next_batch(self.batch_size)
                sess.run([training_op, update_ops], feed_dict={is_training:True,
                                                               images:train_ori_imgs,
                                                               annotation:train_seg_imgs})

                if epoch % 10 == 0:
                    test_ori_imgs, test_seg_imgs = VOC_dataset().next_batch(self.batch_size)
                    summary, test_cost, test_acc = sess.run([merged, cost, accuracy], feed_dict={is_training:False,
                                                                                       images:test_ori_imgs,
                                                                                       annotation:test_seg_imgs)

                    writer.add_summary(summary, epoch)
                    if min_loss is None:
                        min_loss = test_cost
                        saver.save(sess, cwd + "//U_NET/my_model.ckpt")
                    elif test_cost < min_loss:
                        min_loss = test_cost
                        saver.save(sess, cwd + "//U_NET/my_model.ckpt")

                    print("step", epoch, "time", time.ctime(), "loss:", test_cost, 'accuracy:' test_acc)

            print("==============================end train:{}==============================".format(time.ctime()))


    def Test(self):

        model = tf.train.import_meta_graph('./U_NET/my_model.ckpt.meta')
        graph = tf.get_default_graph()
        images = graph.get_tensor_by_name('images:0')
        is_training = graph.get_tensor_by_name('is_training:0')
        annotation_pred = graph.get_tensor_by_name('prediction:0')

        with tf.Session() as sess:
            model.restore(sess, './U_NET/my_model.ckpt')
            test_ori_imgs, _ = VOC_dataset().next_batch(self.batch_size)
            preds = sess.run([annotation_pred], feed_dict={is_training:False,
                                                           images:test_ori_imgs})

            for pred in preds:
                plt.imshow(pred.reshape(512,512))
                plt.show()

if __name__ =='__main__':
    U_Net().Train()
    #U_Net().Test()
