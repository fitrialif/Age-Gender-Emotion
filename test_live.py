import cv2
import dlib
import tensorflow as tf
import inception_resnet_v1
from scipy.misc import imresize
import utils
import os

tf.reset_default_graph()
sess = tf.InteractiveSession()
model = inception_resnet_v1.Model(1e-3,0.8,1e-5,False)
sess.run(tf.global_variables_initializer())
var_lists = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'InceptionResnetV1') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'logits')
var_lists += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'emo')
utils.restore_from_source(sess,'models/',var_lists)

def get_logits(sess, images):
    for i in range(images): images[i] = imresize(images[i], (160,160))
    images = np.array(images)
    return sess.run([model.age_logits,model.gender_logits,model.emotion_logits],feed_dict={self.X:images})

detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detected = detector(input_img, 1)
    images = []
    for i, d in enumerate(detected):
        x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
        images.append(input_img[y1:y1+h,x:x1+w])
    if len(detected) > 0:
        logits = get_logits(images)
        
