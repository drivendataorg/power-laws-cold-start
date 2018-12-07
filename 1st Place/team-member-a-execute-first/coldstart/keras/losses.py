"""
"""
import keras.backend as K

def nmae(y_true, y_pred):
    cost = K.sqrt((y_true - y_pred)**2)
    return K.mean(cost)
