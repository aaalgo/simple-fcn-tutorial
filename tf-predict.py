#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cv2
from skimage import measure
import tensorflow as tf
from tensorflow.python.framework import meta_graph


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('input', 'test.jpg', '')
flags.DEFINE_string('output', 'output.jpg', '')
flags.DEFINE_string('model', 'model', 'Directory to put the training data.')

def clip (v, stride=8):
    H, W, _ = v.shape
    lH = H // stride * stride
    oH = (H - lH)//2
    lW = W // stride * stride
    oW = (W - lW)//2
    return v[oH:(oH+lH), oW:(oW+lW),:]

def main (_):
    assert FLAGS.model and os.path.exists(FLAGS.model + '.meta')

    X = tf.placeholder(tf.float32, shape=(None, None, None, 3))

    mg = meta_graph.read_meta_graph_file(FLAGS.model + '.meta')
    Y, = tf.import_graph_def(mg.graph_def, name='wbc',
                        input_map={'images:0':X},
                        return_elements=['prob:0'])
    saver = tf.train.Saver(saver_def=mg.saver_def, name='wbc')

    image = cv2.imread(FLAGS.input)
    image = clip(image)

    batch = np.reshape(image, (1,) + image.shape)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.restore(sess, FLAGS.model)
        y, = sess.run([Y], feed_dict={X: batch})
        prob = y[0]
        pass

    contours = measure.find_contours(prob, 0.5)

    prob *= 255     # [0,1] -> [0,255]
    prob = cv2.cvtColor(prob, cv2.COLOR_GRAY2BGR)

    for contour in contours:
        tmp = np.copy(contour[:,0])
        contour[:, 0] = contour[:, 1]
        contour[:, 1] = tmp
        contour = contour.reshape((1, -1, 2)).astype(np.int32)
        # draw contours
        cv2.polylines(image, contour, True, 255)
        cv2.polylines(prob, contour, True, 255)
        pass

    output = np.concatenate((image, prob), axis=0)
    cv2.imwrite(FLAGS.output, output)

    pass

if __name__ == '__main__':
    tf.app.run()

