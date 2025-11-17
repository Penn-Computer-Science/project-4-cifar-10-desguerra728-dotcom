import seaborn as sns
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# print out versions of libraries
print(tf.__version__)
print(sns.__version__)

# create training and testing data frames
# (training data, training labels), (testing data, testing labels)
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# show plot
plt.show()

# check there are no vals that are not a number (NaN)
print("Any Nan Training:", np.isnan(x_train).any())
print("Any Nan Training:", np.isnan(x_test).any())

# tell model what shape to expect
input_shape = (32,32,3) #32x32 p0x, 3 color channels (RGB)

# reshape the training and testing data 
# training
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3)

# testing
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)

# Normalize imgs to [0,1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

#convert labels to categorical, not sparse
from keras.utils import to_categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# show example from cifar10
plt.imshow(x_train[100][:,:,0])
plt.show()

# define batch size, go through 60 k imgs 128 at a time
batch_size = 128
# dictate num of classes; 10 numbers
num_classes = 10
# num epochs; go through data 5 times
epochs = 5

# build model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(64, (10,10), padding='same', activation='relu', input_shape = input_shape),
        tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape = input_shape),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.60),

        tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu', input_shape = input_shape),
        tf.keras.layers.Dropout(0.60),
        
        tf.keras.layers.Conv2D(32, (10,10), padding='same', activation='relu', input_shape = input_shape),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ]
)

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['acc'])

history = model.fit(x_train, y_train,
                    batch_size = batch_size,
                    epochs=epochs,
                    validation_split=0.1)

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

#generate the confusion matrix

#generate the confusion matrix
# Predict the values from the testing dataset
Y_pred = model.predict(x_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred, axis=1) 
# Convert testing observations to one hot vectors
Y_true = np.argmax(y_test, axis=1)
# compute the confusion matrix
confusion_mtx = tf.math.confusion_matrix(Y_true, Y_pred_classes) 

# Define class labels
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt='g', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()