def get_a_layer(keras, sample):
    '''
        Small amount of layers
            Seems to under fit, but has a good evaluation result
            usually 80~90% of accuracy
            loss usualy at 40% and evaluation loss at 60%
    '''
    from loadDataSets import leitura1902 as sample
    leitura_mesa_layers = [
        keras.layers.MaxPooling2D(sample.shape[0] // 100, sample.shape[1] // 100,  # compress to aprox shape 100x100
                                  input_shape=(1025, 192, 1)),  # converts (shape) to (shape,1)
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(128, (16, 4), activation='relu'),
        keras.layers.Dropout(0.8),
        keras.layers.Flatten(),
        #keras.layers.Dense(9, activation="softmax"),
        keras.layers.Dense(9, activation="softmax")]

    return  leitura_mesa_layers
