import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs:", gpus)
    for gpu in gpus:
        print("Details for GPU:", gpu)
        print(tf.config.experimental.get_device_details(gpu))
