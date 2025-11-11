import seaborn as sns
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print out versions of libraries
print(tf.__version__)
print(sns.__version__)

# create training and testing data frames
# (training data, training labels), (testing data, testing labels)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# show num of examples in each labeled set
sns.countplot(x = y_train)

# show plot
plt.show()

# check there are no vals that are not a number (NaN)
print("Any Nan Training:", np.isnan(x_train).any())
print("Any Nan Training:", np.isnan(x_test).any())

# tell model what shape to expect
input_shape = (32, 32, 3) #32x32 p0x, 3 color channels RGB

# reshape the training and testing data 
# training
#60k,28,28,0-255
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3)
x_train = x_train.astype('float32') / 255.0

# testing
#30k,28,28,0-255
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)
x_test = x_test.astype('float32') / 255.0

#convert labels to one hot, not sparse
y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
y_test = tf.one_hot(y_test.astype(np.int32), depth=10)

# show example from mnist
plt.imshow(x_train[100][:,:,0], cmap = 'gray')
plt.show()

# define batch size, go through 60 k imgs 128 at a time
batch_size = 128
# dictate num of classes; 10 numbers
num_classes = 10
# num epochs; go through data 5 times
epochs = 10

# build model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(64, (10,10), padding='same', activation='relu', input_shape = input_shape),
        # tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape = input_shape),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.5),
        # tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape = input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ]
)

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train,epochs=10,validation_data=(x_test, y_test))

# plot out training and validation accuracy and loss
fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color = 'b', label ='Training Loss')
ax[0].plot(history.history['val_loss'], color = 'r', label ='Validation Loss')
legend = ax[0].legend(loc='best', shadow=True)
ax[0].set_title('Loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')

ax[1].plot(history.history['acc'], color = 'b', label ='Training Accuracy')
ax[1].plot(history.history['val_acc'], color = 'r', label ='Validation Accuracy')
legend = ax[1].legend(loc='best', shadow=True)
ax[1].set_title('Accuracy')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')

plt.tight_layout()
plt.show()

