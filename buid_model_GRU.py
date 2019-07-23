import keras


def build_model(input_shape, nb_classes, pre_model_path=None, freezen=False, freezen_layers=None):
    input_layer = keras.layers.Input(input_shape)

    cell = keras.layers.GRUCell(units=200,activation="relu")
    gru = keras.layers.RNN(cell,input_shape=input_shape)
    output_layer = gru(input_layer)
    output_layer=keras.layers.Dense(nb_classes, activation='softmax')(output_layer)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])


    return model