import keras
import re


def slice_layer(layer, layer_labels, pattern=None, sel_f=None):
    if sel_f is None:
        sel_f = filter(lambda x: re.match(pattern, x), layer_labels)
    idx = [layer_labels.index(f) for f in sel_f]
    idx1, idx2 = min(idx), max(idx) + 1
    assert idx2 - idx1 == len(idx)
    return keras.layers.Lambda(lambda x: x[:, idx1:idx2])(layer)


def original_gen_hourly_pred_model(features, cold_start_days=None):
    input_size = len(features)
    output_size = 24

    inputs = keras.layers.Input(shape=(input_size, ), name='input')
    x = inputs
    for layer_num in range(2):
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dropout(0.2, seed=layer_num)(x)

    out = []
    for i in range(cold_start_days):
        j = features.index(f'consumption_lag_h_{24*(i+1):03d}')
        assert len(features[j: j+24]) == 24
        inp_i = keras.layers.Lambda(lambda x: x[:, j:j+24])(inputs)
        x_i = keras.layers.Dense(1, activation='relu')(x)
        m_i = keras.layers.multiply([inp_i, x_i])
        out.append(m_i)
    avg = keras.layers.average(out)
    avg = keras.layers.Dense(output_size, activation='relu')(avg)
    outputs = avg

    model = keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model


def ensemble_models(models, features_len, targets_len):
    inputs = keras.layers.Input(shape=(features_len, ), name='e_input')
    out = [model(inputs) for model in models]
    avg = keras.layers.average(out)
    outputs = keras.layers.Dense(targets_len, activation='linear')(avg)
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model


def old_generate_model(network_size, input_columns, output_columns):
    inputs = keras.layers.Input(shape=(input_columns, ), name='input')
    x = inputs
    # buggy code, actually it uses only 1 layer!
    for layer_num in range(2):
        x = keras.layers.Dense(network_size, activation='relu')(inputs)
        x = keras.layers.Dropout(0.2, seed=layer_num)(x)
    outputs = keras.layers.Dense(output_columns, activation='linear')(x)
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model
