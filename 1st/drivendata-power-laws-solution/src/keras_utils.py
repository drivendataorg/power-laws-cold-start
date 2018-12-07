import os
import numpy as np
import tensorflow as tf
import random as rn
import keras
import hashlib
from keras import backend as K


def keras_hash_of_model(model):
    w_str = b".".join([x.tostring() for x in model.get_weights()])
    h = hashlib.sha224(w_str).hexdigest()
    return h


def keras_set_random_state(random_state=0):
    np.random.seed(random_state)
    tf.set_random_seed(random_state)
    rn.seed(random_state)


def keras_initialize_random_state():
    # import numpy as np
    # import tensorflow as tf

    os.environ['PYTHONHASHSEED'] = '0'

    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.

    np.random.seed(42)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.

    rn.seed(12345)

    # Force TensorFlow to use single thread.
    # Multiple threads are a potential source of
    # non-reproducible results.
    # For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
        allow_soft_placement=True,
        device_count = {'CPU' : 1, 'GPU' : 0}
    )

    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
    # import tensorflow as tf
    tf.set_random_seed(1234)

    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


def generate_simple_model(input_columns, output_columns, layers_num=2, network_size=128):
    inputs = keras.layers.Input(shape=(input_columns, ), name='input')
    x = inputs
    for layer_num in range(layers_num):
        x = keras.layers.Dense(network_size, activation='relu')(x)
        x = keras.layers.Dropout(0.2, seed=layer_num)(x)
    outputs = keras.layers.Dense(output_columns, activation='linear')(x)
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model
