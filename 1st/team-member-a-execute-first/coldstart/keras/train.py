"""
Train keras model
"""
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from coldstart.keras.losses import nmae
from coldstart.keras.callbacks import ModelCheckpointRAM

def fit(model, train_x, train_y, validation_data, conf):
    model.compile(optimizer=Adam(**conf['optimizer_kwargs']), loss=nmae)
    callbacks = _get_callbacks(conf['callbacks'])
    ret = model.fit(x=train_x, y=train_y, validation_data=validation_data,
                    callbacks=callbacks, **conf['train_kwargs'])
    if 'ModelCheckpointRAM' in conf['callbacks']:
        model.set_weights(callbacks[-1].weights)
    return model, ret

def _get_callbacks(kwargs):
    key_to_callback = {
        'EarlyStopping': EarlyStopping,
        'ReduceLROnPlateau': ReduceLROnPlateau,
        'ModelCheckpointRAM': ModelCheckpointRAM,
        'ModelCheckpoint': ModelCheckpoint,
    }
    callbacks = []
    for key in key_to_callback:
        if key in kwargs:
            callbacks.append(key_to_callback[key](**kwargs[key]))
    return callbacks
