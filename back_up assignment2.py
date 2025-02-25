# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:13:20 2025

@author: 20212287
"""

# disable overly verbose tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf



# import required packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

# define settings
input_batch_size = 32
input_epochs = 30
input_verbose = 1
optimizer = SGD(learning_rate=0.001, momentum=0.0) #default optimizer is set to learning rate = 0.01 this was insufficient



# load the dataset using the builtin Keras method
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# derive a validation set from the training set
# the original training set is split into
# new training set (90%) and a validation set (10%)
X_train, X_val = train_test_split(X_train, test_size=0.10, random_state=101)
y_train, y_val = train_test_split(y_train, test_size=0.10, random_state=101)



# the shape of the data matrix is NxHxW, where
# N is the number of images,
# H and W are the height and width of the images
# keras expect the data to have shape NxHxWxC, where
# C is the channel dimension
X_train = np.reshape(X_train, (-1,28,28,1))
X_val = np.reshape(X_val, (-1,28,28,1))
X_test = np.reshape(X_test, (-1,28,28,1))


# convert the datatype to float32u
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')


# normalize our data values to the range [0,1]
X_train /= 255
X_val /= 255
X_test /= 255






#add the new classes to the y labels
def map_labels(y):
    
    
    mapping = {1: 0, 7: 0,  # Vertical digits (class 0)
               0: 1, 6: 1, 8: 1, 9: 1,  # Loopy digits (class 1)
               2: 2, 5: 2,  # Curly digits (class 2)
               3: 3, 4: 3}  # Other (class 3)
    
    
    return np.array([mapping[label] for label in y])


y_train, y_val, y_test = map_labels(y_train), map_labels(y_val), map_labels(y_test)


# convert 1D class arrays to 4D class matrices
y_train = to_categorical(y_train, 4)
y_val = to_categorical(y_val, 4)
y_test = to_categorical(y_test, 4)



model = Sequential()
# flatten the 28x28x1 pixel input images to a row of pixels (a 1D-array)
model.add(Flatten(input_shape=(28,28,1)))
# fully connected layer with 64 neurons and ReLU nonlinearity
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
# output layer with 4 nodes 1 for each class
model.add(Dense(4, activation='softmax'))


# compile the model
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# use this variable to name your model
model_name="assignment_2_3_4classes"

# create a way to monitor our model in Tensorboard
tensorboard = TensorBoard("logs/" + model_name)

# train the model
#model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_val, y_val), callbacks=[tensorboard])




# Train the model and store the training history
history = model.fit(X_train, y_train, batch_size=input_batch_size, epochs=input_epochs, verbose=input_verbose,
                    validation_data=(X_val, y_val), callbacks=[tensorboard])

# Evaluate the model on the test set to get test loss and accuracy
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

# Extract loss and accuracy from the training history
train_loss = history.history['loss']
train_accuracy = history.history['accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']

score = model.evaluate(X_test, y_test, verbose=0)

# Plot training and validation loss
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.axhline(y=test_loss, color='r', linestyle='--', label=f'Test Loss: {test_loss:.4f}')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.axhline(y=test_accuracy, color='r', linestyle='--', label=f'Test Accuracy: {test_accuracy:.4f}')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Show plots
plt.tight_layout()
plt.show()





#print initial set settings and final scores
print("Epochs: ",input_epochs)
print("Batch size: ",input_batch_size)
print("Verbose: ",input_verbose)
print("Optimizer: ",optimizer)
print("Test loss: ",test_loss)
print("Test accuracy: ",test_accuracy)

print("Loss: ",score[0])
print("Accuracy: ",score[1])

