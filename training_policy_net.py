import tensorflow as tf
from main import Caption_model_gen

print('TensorFlow Version', tf.__version__)
print(tf.config.experimental.list_physical_devices('GPU'))
actor_model = Caption_model_gen('policy')
#actor_model.summary()