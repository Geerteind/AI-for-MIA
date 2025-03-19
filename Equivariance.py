import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping


tf.keras.backend.clear_session()
# Define dataset path
base_dir = r"C:\Python\AI for MIA\train+val"

# Define image size
IMAGE_SIZE = 96
input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

# Load pretrained model (MobileNetV2 as feature extractor)
input_layer = Input(input_shape)
pretrained = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')

# Freeze layers if needed
# for layer in pretrained.layers:
#    layer.trainable = False

# Build classification model
output = pretrained(input_layer)
output = GlobalAveragePooling2D()(output)
output = Dropout(0.5)(output)
output = Dense(1, activation='sigmoid')(output)

model = Model(input_layer, output)

# Compile the model
model.compile(SGD(learning_rate=0.001, momentum=0.95), loss='binary_crossentropy', metrics=['accuracy','AUC'])

# Define data generators with augmentation for training
def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):
    train_path = base_dir + r"\train"
    valid_path = base_dir + r"\valid"

    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(train_path,
                                                  target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                  batch_size=train_batch_size,
                                                  class_mode='binary')

    val_gen = val_datagen.flow_from_directory(valid_path,
                                              target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                              batch_size=val_batch_size,
                                              class_mode='binary')

    return train_gen, val_gen

# Load data generators
train_gen, val_gen = get_pcam_generators(base_dir=base_dir)

# Define model checkpoint and TensorBoard callbacks
model_name = 'rotation_equivariance_test'
weights_filepath = model_name + '_weights.keras'

checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(os.path.join('logs', model_name))
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
callbacks_list = [checkpoint, tensorboard, early_stopping]

# Train model
history = model.fit(train_gen, epochs=30, validation_data=val_gen, callbacks=callbacks_list)

# # train the model, note that we define "mini-epochs"
# train_steps = train_gen.n//train_gen.batch_size//20
# val_steps = val_gen.n//val_gen.batch_size//20

# # since the model is trained for only 10 "mini-epochs", i.e. half of the data is
# # not used during training
# history = model.fit(train_gen, steps_per_epoch=train_steps,
#                     validation_data=val_gen,
#                     validation_steps=val_steps,
#                     epochs=1,
#                     callbacks=callbacks_list)

# enter evaluation mode
model.trainable = False

def get_samples(generator, num_samples):
    images, labels = [], []
    while len(images) < num_samples:
        batch_images, batch_labels = next(generator)
        images.extend(batch_images)
        labels.extend(batch_labels)
    return np.array(images[:num_samples]), np.array(labels[:num_samples])


def rotate_images_tf(images, angle_degrees):
    """Rotate images using TensorFlow Addons rotate function."""
    angle_radians = np.deg2rad(angle_degrees)  # Convert degrees to radians

    # Convert images to TensorFlow tensor if not already
    images = tf.convert_to_tensor(images, dtype=tf.float32)

    # Rotate images
    rotated_images = tfa.image.rotate(images, angle_radians, interpolation="BILINEAR")

    return rotated_images

def plot_rotation_equivariance(model, generator, num_samples=1000, num_display=3, angles=[0, 90, 180, 270]):
    """Tests rotation equivariance by computing mean absolute prediction differences."""

    # Fetch `num_samples` images
    batch = next(generator)
    images, labels = batch[0][:num_samples], batch[1][:num_samples]
    original_preds = model.predict(images, verbose=0)

    mean_diffs = []
    
    fig, axes = plt.subplots(num_display, len(angles) + 1, figsize=(3 * (len(angles) + 1), 3 * num_display))
    for i in range(num_display):
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(f"Original, Predict: {original_preds[i]}")
        axes[i, 0].axis("off")

    for j, angle  in enumerate(angles):
        rotated_images = rotate_images_tf(images, angle)

        # Get predictions for rotated images
        rotated_preds = model.predict(rotated_images, verbose=0)

        # Compute mean absolute prediction difference
        diff = np.abs(original_preds - rotated_preds).mean()
        mean_diffs.append(diff)
        
        for i in range(num_display):
            axes[i,j+1].imshow(rotated_images[i])
            axes[i, j+1].set_title(f"Angle: {angle}°, Predict: {rotated_preds[i]}")
            axes[i,j+1].axis("off")
    plt.tight_layout()
    plt.show()
    
    # Plot results
    plt.figure(figsize=(8, 5))
    plt.bar([f"{angle}°" for angle in angles], mean_diffs, color=['gray', 'blue', 'orange', 'green'])
    plt.xlabel("Rotation Angle")
    plt.ylabel("Mean Absolute Prediction Difference")
    plt.title("Rotation Equivariance Test")
    plt.ylim(0, max(mean_diffs) * 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, v in enumerate(mean_diffs):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center', fontsize=10)

    plt.show()

    
    
def plot_training_history(history):
    """Plots training & validation accuracy and loss over epochs."""
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, 'b', label='Training Accuracy')
    plt.plot(epochs_range, val_acc, 'r', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, 'b', label='Training Loss')
    plt.plot(epochs_range, val_loss, 'r', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()
    
def display_rotated_predictions(model, generator, num_samples=1, angles=[0, 90, 180, 270]):
    """Displays original and rotated images with their predictions."""
    
    # Fetch `num_samples` images
    batch = next(generator)
    images, labels = batch[0][:num_samples], batch[1][:num_samples]
    for angle in angles:
        rotated_images = rotate_images_tf(images, angle)
    original_preds = model.predict(original_image)
    rotated_pred = model.predict(rotated_image)

    plt.figure(figsize=(len(angles) * 3, num_samples * 3))

    for i in range(num_samples):
        original_image = images[i]
        original_pred = model.predict(original_image)

        for j, angle in enumerate(angles):
            rotated_image = rotate_images_tf(original_image)
            rotated_pred = model.predict(rotated_image)

            # Plot image
            plt.subplot(num_samples, len(angles), i * len(angles) + j + 1)
            plt.imshow(rotated_image.astype(np.uint8))
            plt.axis("off")
            plt.title(f"{angle}°\nPred: {rotated_pred:.3f}")

    plt.tight_layout()
    plt.show()
    
    fig, axes = plt.subplots(ncols=num_samples, nrows=len(angles+1))
    for i in range(num_samples):
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        for j, angle in enumerate(angles):
            rotated_images = rotate_images_tf(images, angle)
            rotated_images = (rotated_images.numpy() * 0.5 + 0.5).clip(0, 1)

            axes[i, j + 1].imshow(rotated_images[i])
            axes[i, j + 1].set_title(f"Rotated {angle}°")
            axes[i, j + 1].axis("off")

    plt.tight_layout()
    plt.show()

    

# Call the function after training
plot_training_history(history)
plot_rotation_equivariance(model, val_gen)
