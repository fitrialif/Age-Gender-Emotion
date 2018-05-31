import os
import time
import tensorflow as tf
import inception_resnet_v1
import utils

batch_size = 128
epoch = 10

tf.reset_default_graph()
sess = tf.InteractiveSession()
# this model only fine-tuned on emotion layers, check inception_resnet_v1.py
model = inception_resnet_v1.Model(learning_rate,keep_prob,weight_decay)
sess.run(tf.global_variables_initializer())
var_lists = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'InceptionResnetV1') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'logits')
utils.restore_from_source(sess,'models/',var_lists)
