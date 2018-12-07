"""
Functions used during the train
"""
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger
from coldstart.keras.train import _get_callbacks
from coldstart.keras.callbacks import ModelCheckpointRAM


def fit(model, train_x, train_y, validation_data, conf):
    model.compile(optimizer=Adam(**conf['optimizer_kwargs']), loss='mean_absolute_error')
    callbacks = _get_callbacks(conf['callbacks'])
    # callbacks.append(ModelCheckpoint(conf['model_path'], save_best_only=True))
    callbacks.append(CSVLogger(conf['model_path'].replace('.h5', '.csv')))
    callbacks.append(ModelCheckpointRAM())
    ret = model.fit(x=train_x, y=train_y, validation_data=validation_data,
                    callbacks=callbacks, **conf['train_kwargs'])
    model.set_weights(callbacks[-1].weights)
    model.save(conf['model_path'])
    return model, ret
