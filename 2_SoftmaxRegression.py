# run the following code to enter tensorflow virtual envrionment
# source ~/tensorflow/bin/activate

# the hello world program in ML is MNIST, a simple computer vision dataset
# it contains images of handwritten digits and labels for each about its digit

# About the MNIST data
# MNIST has three datasets of mnist.train, mnist.test, and mnist.validation
# which have 55,000, 10,000, and 5,000 points of data respectively
# Every data point has an image of a handwriting digit and a label, for example
# mnist.train.images has the training images and mnist.train.labels has labels
# each image is 28x28 pixels and interpreted as a big array of 784 numbers
# mnist.train.images is a tensor with a shape of [55000, 784], each entry in
# the tensor is a pixel intensity between 0 and 1.0. Each label is a one-hot
# vector with 0 in most dimension and 1 in a single dimension to represent the
# digit value of the corresponding image by 1 in the nth dimension. So a number
# 3 would be [0,0,0,1,0,0,0,0,0,0], and mnist.train.labels shapes [55000, 10]

# Softmax Regressions
# softmax assigns probabilities to the possible things of an object can be.
# it adds up the evidence of input being in certain classes then convert that
# evidence into probabilities. first, we sum the weights of pixel intensities
# of all images for a class, negative the weight if high intensity of the pixel
# is against the image being in that class, and positive if it is in favor.
# then add bias representing something independent of the input to evidence.
# so we have evidence(i)=Sigma(j)(W(i,j)*x(j))+b(i) in which i stands for a
# class, j is the index of pixels from input image, W(i,j) is the weight of j
# indexed pixel for class i, x(j) is the pixel intensity of input image, b(i)
# is the bias for class i. then calculate the probability softmax(evidence)(i)
# =exp(evidence(i))/Sigma(j)(exp(evidence(j))) which exponentiating inputs and
# then normalizing them so that they add up to one in probability distribution.

# Implementing
# Numpy can do resource consuming operations such as matrix multiplication
# outside Python, with some overhead for frequently switching back to Python.
# Tensorflow uses graph of interacting operations entirely outside Python.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import sys
import tensorflow as tf

FLAGS = None

def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  
  # Start TensorFlow InteractiveSession, which allows interleave operations
  # to build a computation graph with ones that run the graph, otherwise
  # the entire graph should be built before starting a session and launching
  sess = tf.InteractiveSession()

  # Create the model
  # placeholder for holding any number of MNIST images
  # None means any length dimension
  x = tf.placeholder(tf.float32, [None, 784])
  # placeholder for loss and optimizer, for the target output classes
  y_ = tf.placeholder(tf.float32, [None, 10])
  # variable for the weight for each pixel of each class
  W = tf.Variable(tf.zeros([784, 10]))
  # variable for the bias for each class
  b = tf.Variable(tf.zeros([10]))
  # initialize all variables before using them in a session
  sess.run(tf.global_variables_initializer())
  ## or use the following line of code to initialize variables
  ## tf.global_variables_initializer().run()
  # the regression model multiply vectorized input images x 
  # by the weight matrix, and add bias b
  y = tf.matmul(x, W) + b

  # following section defines the loss function which indicates how bad the
  # model's prediction was on a single sample. We try to minimize it while
  # training across all the examples.
  
  # Cross-Entropy is a function for determining the loss of a model
  # The raw formulation of cross-entropy, can be numerically unstable.
  #   cross_entropy = tf.reduce_mean(
  #     -tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)), reduction_indices=[1]))
  # But it reveals the true operation inside a cross-entropy function as above
  
  # apply the softmax on the model's unnormalized model predictioin and sum
  # across all classes, then take the average over these sums.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  # use automatic differentiation to find the gradients of the loss with
  # respect to each of the variables. the line of code below compute gradients, 
  # compute parameter update steps, and apply update steps to the parameters.
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  # stochastic training with different subset of small batches of random data
  # this is less expensive and has much of the same benefit as using all data
  # train the model by repeatedly run the train_step to apply the gradient
  # descent updates to the parameters.
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # use tf.argmax to find the highest entry in a tensor along some axis
  # use tf.equal to compare prediction with the truth
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  feed_dict = {x: mnist.test.images, y_: mnist.test.labels}
  print(sess.run(accuracy, feed_dict))
  ## or
  ## print(accuracy.eval(feed_dict))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

# Reference:
# http://neuralnetworksanddeeplearning.com/chap3.html#softmax
# https://colah.github.io/posts/2014-10-Visualizing-MNIST/
# http://colah.github.io/posts/2015-09-Visual-Information/
# http://colah.github.io/posts/2015-08-Backprop/ 
# https://en.wikipedia.org/wiki/Gradient_descent
# https://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results
  
  