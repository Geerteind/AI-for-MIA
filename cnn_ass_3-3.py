'''
TU/e BME Project Imaging 2021
Convolutional neural network for PCAM
Author: Suzanne Wetstein
'''

# disable overly verbose tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}   
import tensorflow as tf

# Get the current working directory
current_directory = os.getcwd()

print("Current Working Directory:", current_directory)


import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import layers

# Test GPU acceleration



# Check available devices
print("Available devices: ", tf.config.list_physical_devices())

# Check if GPU is being used
print("Is TensorFlow using the GPU? ", tf.test.is_gpu_available())





# For convolutional only CNN
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Activation

# unused for now, to be used for ROC analysis
from sklearn.metrics import roc_curve, auc


# the size of the images in the PCAM dataset
IMAGE_SIZE = 96

# predefine parameters
learning_rate_input = 0.01
train_batch_size_input = 8
val_batch_size_input = 8
epochs_input = 30
optimizer_input = SGD(learning_rate=learning_rate_input, momentum=0.90)




def get_pcam_generators(base_dir, train_batch_size=train_batch_size_input, val_batch_size=val_batch_size_input):

     # dataset parameters
     train_path = os.path.join(base_dir, 'train+val', 'train')
     valid_path = os.path.join(base_dir, 'train+val', 'valid')


     RESCALING_FACTOR = 1./255

     # instantiate data generators
     datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)

     train_gen = datagen.flow_from_directory(train_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=train_batch_size,
                                             class_mode='binary')

     val_gen = datagen.flow_from_directory(valid_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=val_batch_size,
                                             class_mode='binary')

     return train_gen, val_gen


def get_model(kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64):

     # build the model
     model = Sequential()

     model.add(Conv2D(first_filters, kernel_size, padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
     model.add(layers.BatchNormalization())
     model.add(layers.ReLU())
     model.add(MaxPool2D(pool_size = pool_size))


     model.add(Conv2D(second_filters, kernel_size, padding = 'same'))
     model.add(layers.BatchNormalization())
     model.add(layers.ReLU())
     model.add(MaxPool2D(pool_size = pool_size))

     model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
     model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
     model.add(layers.MaxPooling2D(pool_size=(2, 2)))



     # Instead a global averaging pooling layer is added to replace the flatten
     # and Dense(64) layer. On top of that an activation layer is added to classify probability
     # Furthermore an extre convolution layer is added to ensure output shape
     model.add(Conv2D(1, (1, 1), activation='sigmoid'))
     model.add(GlobalAveragePooling2D())
     #model.add(Activation('sigmoid'))
     model.add(tf.keras.layers.Activation('sigmoid'))

     # compile the model
     model.compile(optimizer_input, loss = 'binary_crossentropy', metrics=['accuracy'])

     return model


# get the model
model = get_model()


# get the data generators
train_gen, val_gen = get_pcam_generators('C:/Users/20212287/OneDrive - TU Eindhoven/Documents/COURSES/OGOs/AI/')



# save the model and weights
# Creating the model_name dynamically
model_name = (
    f"normalized_nr_epochs={epochs_input}_LR={learning_rate_input}_"
    f"batch_size_train={train_batch_size_input}_batch_size_val={val_batch_size_input}"
)

model_filepath = model_name + '.json'
weights_filepath = model_name + '_weights.keras' #.hdf5'

model_json = model.to_json() # serialize model to JSON
with open(model_filepath, 'w') as json_file:
    json_file.write(model_json)


# define the model checkpoint and Tensorboard callbacks
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(os.path.join('logs', model_name))
callbacks_list = [checkpoint, tensorboard]


# train the model
train_steps = train_gen.n//train_gen.batch_size
val_steps = val_gen.n//val_gen.batch_size

history = model.fit(train_gen, steps_per_epoch=train_steps,
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=epochs_input,
                    callbacks=callbacks_list)

# ROC analysis

# TODO Perform ROC analysis on the validation set
