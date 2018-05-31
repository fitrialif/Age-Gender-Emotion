import os
import tensorflow as tf
import shutil
from scipy.misc import imread, imsave
import cv2
import glob

#glob.glob('cohn-kanade-images/*/*/*')
IMAGE_DIR = 'cohn-kanade-images'
EMO_DIR = 'Emotion'
DATA_DIR = 'Data'
CROP_DIR = 'Crop'
min_size = (20,20)
haar_scale = 1.1
min_neighbors = 3
haar_flags = 0

def restore_from_source(sess,source_path,vars):
    s_saver = tf.train.Saver(var_list = vars)
    ckpt = tf.train.get_checkpoint_state(source_path)
    if ckpt and ckpt.model_checkpoint_path:
        s_saver.restore(sess, ckpt.model_checkpoint_path)
        print("restore and continue training!")
        return sess
    else:
        raise IOError("Not found source model")

def getemofromfile(f):
    fh = open(f)
    for l in fh.readlines():
        pass
    return int((float(l.strip())))

def detectFace(image, faceCascade):
    image = cv2.equalizeHist(image)
    faces = faceCascade.detectMultiScale(image, haar_scale, min_neighbors, haar_flags, min_size)
    return faces

def faceCrop(imagePattern, boxScale=1):
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    imgList = glob.glob(imagePattern)
    for img in imgList:
        print(img)
        read_img = imread(img)
        if len(read_img.shape) > 2:
            read_img = cv2.cvtColor(read_img, cv2.COLOR_RGB2GRAY)
        faces = detectFace(read_img, faceCascade)
        if faces.any():
            n = 1
            for (x,y,w,h) in faces:
                cropped_img = read_img[y:y+h,x:x+w]
                fname, ext = os.path.splitext(img)
                fname = fname.split('/')
                fname[-2] = CROP_DIR
                imsave('/'.join(fname) + '_crop' + str(n) + ext, cropped_img)
                n += 1
        else:
            print('no face found', img)

def processing():
    emopath = os.path.join(EMO_DIR, "*","*", "*")
    emofiles = glob.glob(emopath)
    emodict = {}
    for f in emofiles:
        todrop = len(EMO_DIR) +1
        newf = f[todrop:]
        pathnm= os.path.dirname(newf)
        fname =  os.path.basename(newf)
        emo = getemofromfile(f)
        emodict[pathnm] = emo
    print("There were %d emofiles"%(len(emodict.keys())))
    for key in emodict.keys():
        subpath = IMAGE_DIR + "/" + key
        files = glob.glob(os.path.join(subpath, "*"))
        for f in files:
            fname = os.path.basename(f)
            newfile = "E" + str(emodict[key]) + "--" + fname.split('.')[0] + "." + (fname.split('.')[1]).strip()
            shutil.copyfile(f, os.path.join(DATA_DIR, newfile))
    faceCrop(DATA_DIR+'/*')

#processing()
#faceCrop(DATA_DIR+'/*')
