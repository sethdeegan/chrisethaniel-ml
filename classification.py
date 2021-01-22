# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist # get dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() # load dataset
#print(train_images.shape) #prints a tuple containing # of labels, size in px of each img in train_images

#generates pixels_of_training_image_123.png
"""
plt.figure()
plt.imshow(train_images[123])
plt.colorbar()
plt.grid(False)
plt.show()
"""

# scales the rgb values (originally between 0 and 255) so range is between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', # these are the names of our labels
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#generates 8x4_array_of_training_images_with_labels.png
"""
plt.figure(figsize=(10,10))
for i in range(32):
    plt.subplot(4,8,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
"""
# defining the structure of a model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), # transforms format of images from 2-dimensional array (28x28 px) to 1-dimensional array (784 numbers)
    tf.keras.layers.Dense(128, activation='relu'), # Each Dense object represents a layer, this one contains 128 nodes
    tf.keras.layers.Dense(10) # our final layer contains 10 nodes because there are ten labels in class_names
])

# compiling (adding a few more settings) to the model
model.compile(optimizer='adam', # name of optimizer algorithm
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # sets the loss function
              metrics=['accuracy']) # monitors the training and testing steps

model.fit(train_images, train_labels, epochs=5) # starts training

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2) # tests the model

print('\nTest accuracy:', test_acc) # 87% accuracy so far

# Makes predictions on test_images
"""
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[0]) # confidence value for each label for first image
# [1.7551858e-06 2.0831223e-08 9.5195647e-09 2.9056577e-07 7.7765840e-08, 6.0832584e-03 1.8298240e-06 5.3059533e-02 1.5631803e-06 9.4085169e-01]

print("Prediction made by model for test image 0:", np.argmax(predictions[0])) # outputs 9 (10th label)
print("Test label for test image 0:", test_labels[0]) #outputs 9, correct prediction
"""

# puts an image on the gui
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

# puts a bar graph on the gui
def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# Plots the first X test images, each with their predicted label, confidence, and the true label.
# Colors correct predictions in blue and incorrect predictions in red
# outputs 5x3_array_of_testing_images_with_predictions.png
"""
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
"""

# Uses the trained model to make a prediction on one image
"""
# Grabs an image (image 4) from the test dataset.
img = test_images[4]

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))
predictions_single = probability_model.predict(img)
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

print("Prediction of label name for test image 4:", class_names[np.argmax(predictions_single[0])]) # shirt
"""