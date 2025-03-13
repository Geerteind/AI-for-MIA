'''
TU/e BME Project Imaging 2021
Convolutional neural network for PCAM
Author: Suzanne Wetstein
'''

# disable overly verbose tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}   
import tensorflow as tf

import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt


# unused for now, to be used for ROC analysis
from sklearn.metrics import roc_curve, auc


# the size of the images in the PCAM dataset
IMAGE_SIZE = 96

from tensorflow.keras.callbacks import EarlyStopping

# Define EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Metric to monitor (use 'val_accuracy' if you prefer accuracy)
    patience=5,  # Number of epochs to wait for improvement
    verbose=1,  # Print messages when stopping the training
    restore_best_weights=True  # Restore model weights from the epoch with the best monitored value
)


learning_rate_input = 0.01
train_batch_size_input = 8
val_batch_size_input = 8
epochs_input = 15
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

     model.add(Conv2D(first_filters, kernel_size, activation = 'relu', padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
     model.add(MaxPool2D(pool_size = pool_size))

     model.add(Conv2D(second_filters, kernel_size, activation = 'relu', padding = 'same'))
     model.add(MaxPool2D(pool_size = pool_size))

     model.add(Flatten())
     model.add(Dropout(0.1))



     model.add(Dense(64, activation = 'relu'))
     model.add(Dense(1, activation = 'sigmoid'))


     # compile the model
     model.compile(optimizer_input, loss = 'binary_crossentropy', metrics=['accuracy'])

     return model


# get the model
model = get_model()


# get the data generators
train_gen, val_gen = get_pcam_generators(r"C:\Users\20212287\OneDrive - TU Eindhoven\Documents\COURSES\OGOs\AI")



# save the model and weights
model_name = (f"dropout_nr_epochs={epochs_input}_LR={learning_rate_input}_"
              f"batch_size_train={train_batch_size_input}_batch_size_val={val_batch_size_input}")
model_filepath = model_name + '.json'
weights_filepath = model_name + '_weights.keras' #.hdf5'

model_json = model.to_json() # serialize model to JSON
with open(model_filepath, 'w') as json_file:
    json_file.write(model_json)


# define the model checkpoint and Tensorboard callbacks
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(os.path.join('logs', model_name))
callbacks_list = [checkpoint, tensorboard, early_stopping]


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
def plot_roc_curve(y_true, y_pred, image_name, title="Receiver Operating Characteristic (ROC) Curve"):
    """
    Plots the ROC curve for given true labels and predicted probabilities.

    Parameters:
    - y_true: Array-like, true class labels.
    - y_pred: Array-like, predicted probabilities for the positive class.
    - title: (Optional) Title for the plot.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)  # Diagonal line (random classifier)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(image_name)
    plt.show()
    return

guesses = model.predict(val_gen).ravel()
y_true = val_gen.classes
plot_roc_curve(y_true, guesses, "ROC_curve.png")
import seaborn as sns
sns.histplot(guesses, bins=20, kde=True)
plt.show()

