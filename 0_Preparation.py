# get the Ubuntu environment prepared for TensorFlow
# OS is ubuntu 16.04 LTS, no NVIDIA GPU available, Anaconda installed
# following code will use ~/tensorflow as the DEST_DIR for installation

# TensorFlow uses recommended virtualenv installation method
# in terminal run following command in sequence:
# sudo apt-get install python3-pip python3-dev python-virtualenv
# virtualenv --system-site-packages -p python3 ~/tensorflow
# source ~/tensorflow/bin/activate
# pip3 install --upgrade tensorflow

# for each time to use TensorFlow, the virtualenv should be activated:
# source ~/tensorflow/bin/activate

# for quit the virtualenv of TensorFlow, run:
# deactivate

# for uninstall TensorFlow, run:
# rm -r ~/tensorflow

# for validating the installation, run following python program
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

# Reference:
# https://www.tensorflow.org/install/install_linux
