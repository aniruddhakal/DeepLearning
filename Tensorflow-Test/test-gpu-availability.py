from keras import backend as K
import tensorflow as tf
# from tensorflow.python.client import device_lib
from tensorflow.python.client import device_lib

# gpu_list = K.tensorflow_backend._get_available_gpus()
devices_list = device_lib.list_local_devices()

print(">>>>>>>>>>>Devices List: " + str(devices_list))

print("GPU Availability: %s" % str(tf.test.is_gpu_available()))