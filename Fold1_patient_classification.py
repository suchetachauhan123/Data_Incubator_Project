from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
from datetime import datetime
import time
from decimal import *

import numpy as np
from numpy import swapaxes
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import scipy
from scipy.stats import pearsonr
import tensorflow as tf
#import input

NUM_of_SLICES = 61
IMAGE_SIZE_ROW = 61
IMAGE_SIZE_COL = 73
NUM_CHANNELS = 1			# RGB color channel
n_classes = 1
VALIDATION_SIZE = 22  # Size of the validation set.
SEED = None  # Set to None for random seed.
BATCH_SIZE = 120
EVAL_BATCH_SIZE = 2
EVAL_FREQUENCY = 10  # Number of steps between evaluations.

tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
FLAGS = tf.app.flags.FLAGS

def main(argv=None):  # pylint: disable=unused-argument

  ##################### load the data from numpy array.   # Get the data.
  path = os.getcwd()
  directory = 'patient_network'
  ffrom = 0
  h = 500
  lr = 0.001
  reg = 0.01
  std = 0.1
  f = 8
  NUM_EPOCHS = 50
  filter1 = 5
  to = ffrom + 2

  os.makedirs(directory)
  new_path= os.path.join(path, directory)

  inp = np.load('train.npz')['input']
  tar = np.load('train.npz')['labels']

  def sigmoid(x):
        return (1 / (1 + np.exp(-x)))

  inp = sigmoid(inp)

  test_data = inp[:to]
  test_labels = tar[:to]

  inp = inp[to:]
  tar = tar[to:]

  idx = np.random.permutation(inp.shape[0])
  inp = inp[idx]
  tar = tar[idx]

  val_data = inp[:VALIDATION_SIZE]
  val_labels = tar[:VALIDATION_SIZE]

  batch_xs = inp[VALIDATION_SIZE:]
  batch_ys = tar[VALIDATION_SIZE:]       

  # Extract it into np arrays.
  train_data = batch_xs
  train_labels = batch_ys

  num_epochs = NUM_EPOCHS
  train_size = train_labels.shape[0]

  print ('train_size, train_data.shape, len(train_data), len(val_data), val_data.shape') 
  print (train_size, train_data.shape, len(train_data), len(val_data), val_data.shape)

  train_data_node = tf.placeholder(tf.float32, shape=(None, NUM_of_SLICES, IMAGE_SIZE_ROW, IMAGE_SIZE_COL, NUM_CHANNELS))
  train_labels_node = tf.placeholder(tf.float32, shape=(None, n_classes))
  keep_probs = tf.placeholder(tf.float32, name='keep_probs')

  conv1_weights = tf.Variable(
      tf.truncated_normal([filter1, filter1, filter1, NUM_CHANNELS, f],  # 5x5 filter, depth 32.
                          stddev=std,
                          seed=SEED))
  conv1_biases = tf.Variable(tf.zeros([f]))

  fc1_weights = tf.Variable(tf.truncated_normal([16 * ((IMAGE_SIZE_ROW // 4)+1) * ((IMAGE_SIZE_COL // 4)+1) * f, h], stddev=std, seed=SEED))   

  print ('fc1 weights : NUM_of_SLICES * IMAGE_SIZE_ROW * IMAGE_SIZE_COL * 64')
  print (16, (IMAGE_SIZE_ROW //4)+1, (IMAGE_SIZE_COL // 4)+1, fc1_weights.get_shape())
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[h]))
  fc2_weights = tf.Variable(tf.truncated_normal([h, n_classes], stddev=std, seed=SEED))
  fc2_biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))

  def model(data):
    """The Model definition."""
    print ('conv-1 input')
    print (data)
    conv = tf.nn.conv3d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1, 1],
                        padding='SAME')

    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    pool = tf.nn.max_pool3d(relu,
                          ksize=[1, 4, 4, 4, 1],
                          strides=[1, 4, 4, 4, 1],
                          padding='SAME')
    print ('conv-1 output')
    print (pool)

    pool_shape = pool.get_shape().as_list()
    print (pool_shape)
    reshape = tf.reshape(pool, [-1, pool_shape[1] * pool_shape[2] * pool_shape[3] * pool_shape[4]])
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    return tf.nn.sigmoid(tf.matmul(hidden, fc2_weights) + fc2_biases)

  logits = model(train_data_node)

  loss = tf.reduce_mean(tf.pow(train_labels_node - logits, 2))
  regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc2_weights))
  loss += reg * regularizers  

  batch = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(
	lr,                # Base learning rate.
    	batch * BATCH_SIZE,  # Current index into the dataset.
    	train_size,          # Decay step.
    	0.95,                # Decay rate.
    	staircase=True)

  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step = batch) # Adam Optimizer

  ##############################################################

  def eval_in_batches(data, labels, sess):
    """Get all predictions for a dataset by running it in small batches."""
    #total_batches = data.shape[0] // EVAL_BATCH_SIZE
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = np.ndarray(shape=(size, n_classes), dtype=np.float32)
    eval_fp1 = np.ndarray(shape=(size, 2560), dtype=np.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(logits, feed_dict={train_data_node: data[begin:end, ...], 
                                                                train_labels_node: labels[begin:end, ...]})

      else:
        batch_predictions = sess.run(logits, feed_dict={train_data_node: data[-EVAL_BATCH_SIZE:, ...], 
                                                             train_labels_node: labels[-EVAL_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]

    return predictions

  def acc_loss(predictions, labels):
	a_loss = np.power(predictions - labels, 2)
      	return np.mean(a_loss)

  saver = tf.train.Saver()
  start_time = time.time()

  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print('Initialized!')
    step_end = 0
    flag = 0
    validation_loss_prev = 1
    no_of_batches = math.ceil(train_size / float(BATCH_SIZE))
    print('number of batches:',no_of_batches)
    i = 1
    count_loss = 0
    count_epoch = 0

    batch_data = train_data.reshape(len(train_data), NUM_of_SLICES, IMAGE_SIZE_ROW, IMAGE_SIZE_COL, NUM_CHANNELS)
    batch_labels = train_labels.reshape(len(train_data), n_classes)
    validation_data = val_data.reshape(len(val_data), NUM_of_SLICES, IMAGE_SIZE_ROW, IMAGE_SIZE_COL, NUM_CHANNELS)
    validation_labels = val_labels.reshape(len(val_data), n_classes)
    testing_data = test_data.reshape(len(test_data), NUM_of_SLICES, IMAGE_SIZE_ROW, IMAGE_SIZE_COL, NUM_CHANNELS)
    testing_labels = test_labels.reshape(len(test_data), n_classes)

    sys.stdout=open(os.path.join(new_path,"run_out.txt"),"w")
 
    f_acc=open(os.path.join(new_path,'loss.txt'), 'w')

    # Loop through training29_Rege-7_h50 steps.
    for step in xrange((int(num_epochs * train_size) // BATCH_SIZE)):
      offset = (step * BATCH_SIZE) % train_size
      print('offset:%d, offset + BATCH_SIZE:%d' %(offset, offset + BATCH_SIZE))
      inp_data = batch_data[offset:(offset + BATCH_SIZE),...]
      inp_labels = batch_labels[offset:(offset + BATCH_SIZE),...]
      feed_dict = {train_data_node: inp_data, train_labels_node:inp_labels}
      start_time = time.time()
      _, l, predictions, lr = sess.run([optimizer, loss, logits, learning_rate], feed_dict = feed_dict)
      duration = time.time() - start_time
      saver.save(sess, os.path.join(new_path,"convNet.ckpt"))

      if step == (i * no_of_batches) - 1:
	      idx = np.random.permutation(batch_data.shape[0])
	      #print(idx)
              batch_data = batch_data[idx]
	      batch_labels = batch_labels[idx]

              print('%s: Step %d (epoch %.2f), %.3f sec/epoch' % (datetime.now(), step, float(step) * BATCH_SIZE / train_size, duration))
    	      predictions = eval_in_batches(batch_data, batch_labels, sess)
              tr_err = acc_loss(predictions, batch_labels)

              preds = eval_in_batches(validation_data, validation_labels, sess)
              val_err = acc_loss(preds, validation_labels)

              print('Training error: %.3f, validation err: %.3f, learning rate: %.6f' % (l, tr_err, val_err, lr))
              print('------------------------------------------------------------------------------------')
	      f_acc.write('{:f} {:f}\n'.format(tr_err, val_err))

              if val_err <= validation_loss_prev:
                validation_loss_prev = val_err
                flag = 1
                count_epoch = 0
              elif val_err > validation_loss_prev and flag == 1:
                count_epoch = count_epoch + 1
                if count_epoch == 5:
                  break
              i = i + 1

      elif step % EVAL_FREQUENCY == 0:
        print('Step %d (epoch %.2f)' % (step, float(step+1) * BATCH_SIZE / train_size))
      sys.stdout.flush()

    #saver.restore(sess, os.path.join(new_path,"convNet.ckpt"))

    print('****************************************** training PREDICTIONS: ******************************************')
    predictions = eval_in_batches(batch_data, batch_labels, sess)
    tr_loss = acc_loss(predictions, batch_labels)

    print('****************************************** VALIDATION PREDICTIONS: ******************************************')
    preds = eval_in_batches(validation_data, validation_labels, sess)
    val_loss = acc_loss(preds, validation_labels)

    print('training loss: %.3f, validation loss: %.3f' % (tr_loss, val_loss))

    # Finally print the result!
    print('*********************************************** TESTING PREDICTIONS: ******************************************')
    test_preds = eval_in_batches(testing_data, testing_labels, sess)
    test_loss = acc_loss(test_preds, testing_labels)
    print('testing loss: %.3f' % (test_loss))

    np.savetxt(os.path.join(new_path,'testing_predictions.txt'), test_preds)
    np.savetxt(os.path.join(new_path,'testing_labels.txt'), testing_labels, fmt='%d')

    sys.stdout.close()

if __name__ == '__main__':
    tf.app.run()
