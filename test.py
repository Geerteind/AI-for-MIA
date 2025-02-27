import os



import tensorflow as tf

# Enable verbose TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Show all messages (debug level)
tf.debugging.set_log_device_placement(True)  # Display device placement details

# List all physical devices
devices = tf.config.list_physical_devices()
print("Available devices: ", devices)

# Check for GPUs specifically
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("CUDA is configured properly. GPU is available: ", gpus)
    for gpu in gpus:
        # Check memory growth settings (optional)
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU detected or CUDA is not set up properly.")
