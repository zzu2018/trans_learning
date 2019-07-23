import keras


def build_model(input_shape, nb_classes, pre_model_path=None, freezen=False, freezen_layers=None):
    input_layer = keras.layers.Input(input_shape)

    conv1_1 = keras.layers.Conv2D(filters=128, kernel_size=(3,1), padding='same')(input_layer)
    conv1 = keras.layers.Conv2D(filters=16, kernel_size=1, padding='same')(conv1_1)
    conv1 = keras.layers.normalization.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)

    conv2_1 = keras.layers.Conv2D(filters=256, kernel_size=(3,1), padding='same')(conv1)
    conv2 = keras.layers.Conv2D(filters=32, kernel_size=1, padding='same')(conv2_1)
    conv2 = keras.layers.normalization.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    conv3_1 = keras.layers.Conv2D(256, kernel_size=(3,1), padding='same')(conv2)
    conv3= keras.layers.Conv2D(filters=64, kernel_size=1, padding='same')(conv3_1)
    conv3 = keras.layers.normalization.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    conv4_1 = keras.layers.Conv2D(256, kernel_size=(3, 1), padding='same')(conv3)
    conv4 = keras.layers.Conv2D(filters=64, kernel_size=1, padding='same')(conv4_1)
    conv4 = keras.layers.normalization.BatchNormalization()(conv4)
    conv4 = keras.layers.Activation('relu')(conv4)

    conv1x_2 = keras.layers.Conv2D(filters=256, kernel_size=(1, 3), padding='same')(input_layer)
    conv1x = keras.layers.Conv2D(filters=16, kernel_size=1, padding='same')(conv1x_2)
    conv1x = keras.layers.normalization.BatchNormalization()(conv1x)
    conv1x = keras.layers.Activation(activation='relu')(conv1x)

    conv2x_1 = keras.layers.Conv2D(filters=512, kernel_size=(1, 3), padding='same')(conv1x)
    conv2x = keras.layers.Conv2D(filters=32, kernel_size=1, padding='same')(conv2x_1)
    conv2x = keras.layers.normalization.BatchNormalization()(conv2x)
    conv2x = keras.layers.Activation('relu')(conv2x)

    conv3x_1 = keras.layers.Conv2D(256, kernel_size=(1, 3), padding='same')(conv2x)
    conv3x= keras.layers.Conv2D(filters=64, kernel_size=1, padding='same')(conv3x_1)
    conv3x = keras.layers.normalization.BatchNormalization()(conv3x)
    conv3x = keras.layers.Activation('relu')(conv3x)

    conv4x_1 = keras.layers.Conv2D(256, kernel_size=(1, 3), padding='same')(conv3x)
    conv4x = keras.layers.Conv2D(filters=64, kernel_size=1, padding='same')(conv4x_1)
    conv4x = keras.layers.normalization.BatchNormalization()(conv4x)
    conv4x = keras.layers.Activation('relu')(conv4x)

    conv4= keras.layers.concatenate([conv4, conv4x], axis=3)
    conv4 =keras.layers.Conv2D(64,kernel_size=1,padding="same")(conv4)
    gap_layer = keras.layers.pooling.GlobalAveragePooling2D()(conv4)

    output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    if pre_model_path is not None:
        print('pre_model_path:', pre_model_path)
        print('loading model...')
        pre_model = keras.models.load_model(pre_model_path)
        print('loaded model !!!')
        for i in range(len(model.layers) - 1):
            model.layers[i].set_weights(pre_model.layers[i].get_weights())
        if freezen is True and freezen_layers is not None:
            for i in freezen_layers:
                model.layers[i].trainable = False

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model
