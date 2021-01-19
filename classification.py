# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist # get dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() # load dataset
#print(train_images.shape) #prints 

""" plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show() """

# scales these values so range is between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

""" class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show() """

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), # transforms format of images from 2-dimensional array to 1-dimensional array
    tf.keras.layers.Dense(128, activation='relu'), # Each Dense object represents a layer, this one contains 128 nodes
    tf.keras.layers.Dense(10) # this layer contains 10 nodes because there are ten classes
])

model.compile(optimizer='adam', # 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #
              metrics=['accuracy'])