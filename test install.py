import tensorflow as tf

# List all physical devices
devices = tf.config.list_physical_devices()
print("Available devices: ", devices)

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print("CUDA is configured properly. GPU is available: ", gpus)

else:
    print("CUDA is not working or no GPU detected.")
