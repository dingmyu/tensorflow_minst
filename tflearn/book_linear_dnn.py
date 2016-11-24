import numpy as np
import pandas as pd
import tensorflow as tf
import skflow
from tensorflow.contrib import learn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pylab

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
y_train = train['label']
X_train = train.drop('label', 1)
X_test = test

def display(img):
    image_width = 28 
    image_height = 28
    # (784) => (28,28)
    one_image = img.reshape(image_width,image_height)
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)
    plt.show()
    #pylab.show()

# output image     
display(X_train[1:2].values)


mnist = learn.datasets.load_dataset('mnist')
feature_columns = learn.infer_real_valued_columns_from_input(mnist.train.images)

classifier = learn.LinearClassifier(feature_columns=feature_columns, n_classes=10, optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.01))
classifier.fit(X_train, y_train, steps=1000, batch_size=100)
linear_y_predict = classifier.predict(X_test)
accuracy_score = classifier.evaluate(X_train,
                                         y_train)["accuracy"]

print accuracy_score
print linear_y_predict[:100]
linear_submission = pd.DataFrame({'ImageId':range(1, 28001), 'Label': linear_y_predict})
linear_submission.to_csv('linear_submission.csv', index = False)
print 'linear done'


classifier = learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[200, 50], n_classes = 10, optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.01))  
# , learning_rate=0.01, steps=5000, batch_size=50
classifier.fit(X_train, y_train, steps=5000, batch_size=50)
accuracy_score = classifier.evaluate(X_train,
                                         y_train)["accuracy"]
dnn_y_predict = classifier.predict(X_test)
print accuracy_score
print dnn_y_predict[:100]
dnn_submission = pd.DataFrame({'ImageId':range(1, 28001), 'Label': dnn_y_predict})
dnn_submission.to_csv('dnn_submission.csv', index = False)
print 'dnn done!'


'''
tf.contrib.learn.DNNClassifier.__init__(self,
  hidden_units,
  feature_columns=None,
  model_dir=None,
  n_classes=2,
  weight_column_name=None,
  optimizer=None,
  activation_fn=relu,
  dropout=None,
  config=None)
def fit(self, x=None, y=None, input_fn=None, steps=None,batch_size=None,
          monitors=None):

# suoyi gaiban hou zhineng zai fit li shezhi steps=5000, batch_size=50
# skflow -> tensorflow.contrib.learn
'''

'''
# zheshi cnn huaile yongbuliao  laji 
def max_pool_2x2(tensor_in):
  return tf.nn.max_pool(
      tensor_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_model(X, y):
  # pylint: disable=invalid-name,missing-docstring
  # reshape X to 4d tensor with 2nd and 3rd dimensions being image width and
  # height final dimension being the number of color channels.
  X = tf.reshape(X, [-1, 28, 28, 1])
  # first conv layer will compute 32 features for each 5x5 patch
  with tf.variable_scope('conv_layer1'):
    h_conv1 = learn.ops.conv2d(X, n_filters=32, filter_shape=[5, 5],
                               bias=True, activation=tf.nn.relu)
    h_pool1 = max_pool_2x2(h_conv1)
  # second conv layer will compute 64 features for each 5x5 patch.
  with tf.variable_scope('conv_layer2'):
    h_conv2 = learn.ops.conv2d(h_pool1, n_filters=64, filter_shape=[5, 5],
                               bias=True, activation=tf.nn.relu)
    h_pool2 = max_pool_2x2(h_conv2)
    # reshape tensor into a batch of vectors
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
  # densely connected layer with 1024 neurons.
  h_fc1 = tf.contrib.layers.dropout(
      tf.contrib.layers.legacy_fully_connected(
          h_pool2_flat, 1024, weight_init=None, activation_fn=tf.nn.relu))
  return learn.models.logistic_regression(h_fc1, y)

# Training and predicting.
classifier = learn.TensorFlowEstimator(
    model_fn=conv_model, n_classes=10, batch_size=100, steps=20000,
    learning_rate=0.001)
classifier.fit(X_train, y_train)

print 'fit ok'

conv_y_predict = []
for i in np.arange(100, 28001, 100):
    conv_y_predict = np.append(conv_y_predict, classifier.predict(X_test[i - 100:i]))
conv_y_predict[:10]
conv_submission = pd.DataFrame({'ImageId':range(1, 28001), 'Label': np.int32(conv_y_predict)})
conv_submission.to_csv('conv_submission.csv', index = False)
'''
