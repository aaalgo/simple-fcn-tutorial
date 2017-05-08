#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
from tqdm import tqdm
#from skimage import measure
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import picpac

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('db', 'db', '')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_string('model', 'model', 'Directory to put the training data.')
flags.DEFINE_integer('max_steps', 200000, '')
flags.DEFINE_integer('epoch_steps', 100, '')
flags.DEFINE_integer('ckpt_epochs', 20, '')
flags.DEFINE_integer('channels', 3, '')
flags.DEFINE_integer('max_to_keep', 1000, '')

def inference (X):
    net = X
    net = slim.conv2d(net, 64, 3, 2)
    net = slim.max_pool2d(net, 2, 2)
    net = slim.conv2d(net, 128, 3, 1)
    net = slim.conv2d(net, 128, 3, 1)
    net = slim.max_pool2d(net, 2, 2)
    net = slim.conv2d(net, 256, 3, 1)
    net = slim.conv2d(net, 256, 3, 1)
    net = slim.conv2d(net, 128, 1, 1)
    net = slim.conv2d(net, 32, 1, 1)
    net = slim.conv2d_transpose(net, 2, 17, 8, 
                                activation_fn=None,
                                normalizer_fn=None)
    logits = tf.identity(net, 'logits')

    shape = tf.shape(net)    # (?, ?, ?, 2)
    net = tf.reshape(net, (-1, 2))
    net = tf.nn.softmax(net)
    net = tf.reshape(net, shape)
    net = tf.slice(net, [0, 0, 0, 1], [-1, -1, -1, -1])
    net = tf.squeeze(net, axis=[3])

    prob = tf.identity(net, 'prob')
    return logits, prob, 8

def cross_entropy (logits, labels):
    logits = tf.reshape(logits, (-1, 2))
    labels = tf.reshape(labels, (-1,))
    xe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.to_int32(labels))
    xe = tf.reduce_mean(xe, name='xe')
    return xe

def main (_):
    try:
        os.mkdir(FLAGS.model)
    except:
        pass
    assert FLAGS.db and os.path.exists(FLAGS.db)

    X = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.channels), name="images")
    Y = tf.placeholder(tf.float32, shape=(None, None, None, 1), name="labels")

    logits, _, stride = inference(X)
    loss = cross_entropy(logits, Y)

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

    train_op = optimizer.minimize(loss)

    saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    picpac_config = dict(seed=2016,
                shuffle=True,
                reshuffle=True,
                batch=1,
                round_div=stride,
                annotate='json',
                channels=FLAGS.channels,
                pert_color1=20,
                pert_angle=20,
                pert_min_scale=0.8,
                pert_max_scale=1.2,
                pert_hflip=True,
                pert_vflip=True,
                channel_first=False # this is tensorflow specific
                                    # Caffe's dimension order is different.
                )
    stream = picpac.ImageStream(FLAGS.db, perturb=True, loop=True, **picpac_config)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        epoch = 0
        while step < FLAGS.max_steps:
            loss_sum = 0.0
            for _ in tqdm(range(FLAGS.epoch_steps), leave=False):
                images, labels, _ = stream.next()
                feed_dict = {X: images, Y: labels}
                ll, _, = sess.run([loss, train_op], feed_dict=feed_dict)
                loss_sum += ll
                step += 1
                pass
            print('step %d: loss=%.4f'
                    % (step, loss_sum / FLAGS.epoch_steps))
            epoch += 1
            if epoch and (epoch % FLAGS.ckpt_epochs == 0):
                ckpt_path = '%s/%d' % (FLAGS.model, step)
                saver.save(sess, ckpt_path)
                print('saving to %s' % ckpt_path)
            pass
        pass
    pass

if __name__ == '__main__':
    tf.app.run()

