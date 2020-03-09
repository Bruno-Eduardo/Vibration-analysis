def get_a_layer(keras, sample):
    '''
        Small amount of layers
            Seems to under fit, but has a good evaluation result
            usually 80~90% of accuracy
            loss usualy at 40% and evaluation loss at 60%
    '''
    #from loadDataSets import leituraMesa as sample
    leitura_mesa_layers = [
        keras.layers.MaxPooling2D(sample.shape[0] // 100, sample.shape[1] // 100,  # compress to aprox shape 100x100
                                  input_shape=(sample.shape + (1, None))[0:-1]),  # converts (shape) to (shape,1)
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (4, 4), activation='relu'),
        keras.layers.Dropout(0.015),
        keras.layers.Flatten(),
        keras.layers.Dense(9, activation="softmax"),
        keras.layers.Dense(sample.NofOutputs, activation="softmax")]
    # 1a tnetativa 100% em categorico no conjunto da mesa em 200 epocas
    # 2a tentantica 90% em 270 epocas, depois em 400 decaiu pra 80, mas loss ainda baixando
    # tanto no training quanto no validation, porem baixando cada vez menos. Em 1000 epocas ainda 80
    # 3a tentatica 400 epocas pra bater 80%. em 524 bateu 90%. Parou em 80%, mas loss continuou caindo
    # 4a tentativa 123 epocas 90%. em 138 epocas o loss comecou a aumentar. em 172 voltou a diminuir.
    # sempre ficoando em 90%
    # Em todos os casos o loss de treinamento nao tendeu a zero, entao existe um underfitting
    # porem a acuracia bateu 100 em todos os treinamentos

    #resultados
    #[89.99999761581421, 100.0, 69.9999988079071, 89.99999761581421, 89.99999761581421, 100.0, 80.0000011920929, 69.9999988079071, 89.99999761581421, 50.0]
    # 82.9999989271164
    # 15.670212085786929
    #
    # tirando o outlier
    # 86.66666547457378
    # 11.180339887499027

    # run 2
    # [60.00000238418579, 89.99999761581421, 80.0000011920929, 100.0, 100.0, 80.0000011920929, 80.0000011920929, 89.99999761581421, 80.0000011920929, 50.0]
    # 81.00000023841858
    # 15.951314137771385

    return  leitura_mesa_layers