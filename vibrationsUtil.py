
from os import listdir
import random
import pickle

import librosa.display
import tensorflow.compat.v1 as tf
from tensorflow import keras

print('importing utils....')
from generalUtil import *
print('Done!')

# Global vars
dataSetRawPath = r"F:\BrunoDeepLearning\ICvibracoesMesa\leitura0710"
dataSetRawPath = r"F:\BrunoDeepLearning\ICvibracoesMesa\vibracoesSimuladas"

scratchFilesListRAW = r"F:\BrunoDeepLearning\ICvibracoesMesa\VibrationsScratchFiles.txt"
scratchFilesListRAW = r"F:\BrunoDeepLearning\ICvibracoesMesa\SimulatedVibrationsScratchFiles.txt"

dataFileCSV = 'impactos.csv'
labelFileCSV = 'labels.csv'

distancesDict = {"impactos1":0, "impactos4":1, "impactos8":2}
distancesDict = {"impactos1":0, "impactos2":1, "impactos3":2}

files = listdir(dataSetRawPath)
DEBUG = False


def getBatch(set2process, dictOfOutputs, size=-1, reset=False):
    if size == -1:
        size = len(set2process)
    try:
        getBatch.lastProcessedFile
    except:
        getBatch.lastProcessedFile = 0
    if reset:
        getBatch.lastProcessedFile = 0
        return
    lastPrintedFlag = -1

    for i in range(size):
        try:
            pickledFile = open(dataSetRawPath + '\\scratch\\' + set2process[getBatch.lastProcessedFile], 'rb')
        except:
            getBatch.lastProcessedFile = (getBatch.lastProcessedFile) % len(set2process)
            pickledFile = open(dataSetRawPath + '\\scratch\\' + set2process[getBatch.lastProcessedFile], 'rb')

        D = (pickle.load(pickledFile))+80

        if DEBUG: print(set2process[getBatch.lastProcessedFile], end="")
        #plotD(D)                                                                   #----------------------------------- Remover para plotar as ffts

        D = np.resize(D, (D.shape[0],D.shape[1],1))
        D = D.astype(np.uint8)

        Y = np.zeros(len(dictOfOutputs), dtype = int)
        Y[dictOfOutputs[set2process[getBatch.lastProcessedFile].split('in',1)[0]]] = 1
        if DEBUG: print(", label:" + str(Y))

        try:
            batchX.append(D)
            batchY.append(Y)
        except:
            batchX = [D]
            batchY = [Y]

        printCandidate = int(10*getBatch.lastProcessedFile/size)
        if(lastPrintedFlag != printCandidate):
            lastPrintedFlag = printCandidate
            if DEBUG: print(str(10*printCandidate) + "%")
        getBatch.lastProcessedFile += 1

    print("stacking...")
    batchX = np.stack(batchX,axis=0)
    batchY = np.stack(batchY, axis=0)
    print("stacking DONE!")

    getBatch([], dictOfOutputs, reset=True)
    return (batchX[:,:,:,:], batchY)


def prepareBatches(dictOfOutputs):
    scratchFilesList = []

    with open(scratchFilesListRAW, 'r', encoding="utf8") as file:
        for line in file:
            line = line.replace('\n', '\r').replace('\r', '').replace('/', '\\').replace(dataSetRawPath + "\\scratch\\",
                                                                                         "")
            if 'txt' in line:
                scratchFilesList.append(line)
            else:
                raise Exception('scratchFilesListRAW may be corrupted. Check ' + scratchFilesListRAW)

    shuffledNice = False
    while not shuffledNice:
        random.shuffle(scratchFilesList)

        cut = int(len(scratchFilesList)*.8)

        training = scratchFilesList[0:cut]
        crssvald = scratchFilesList[int(len(scratchFilesList) * .6):int(len(scratchFilesList) * .8)]
        avaliati = scratchFilesList[cut:]

        # Nexts lines creates a list [a,b,...,z] where a is the amount of class 0 occurrences,
        #                                            * is the amount of class * occurrences,
        #                                            z is the amount of class N occurrences,
        outputs = list(dictOfOutputs.keys())
        founds = [sum(1 if y in x else 0 for x in avaliati) for y in outputs]

        shuffledNice = len(set(founds)) <= 1  # This expression returns true when all elements are the same
        shuffledNice = shuffledNice and len(founds) == len(outputs)  # This expression returns true when all outputs
        # have the same amount of samples
        if shuffledNice:
            trainingSet = getBatch(training, dictOfOutputs)
            getBatch([], {},reset=True)
            avaliatiSet = getBatch(avaliati, dictOfOutputs)

        else:
            print('not ok, reshufle' + '-' * 40)
            #TODO implemet a smart reshufle

    return (trainingSet, avaliatiSet, avaliati)


def generateScratch(parser=csv2array, file=dataFileCSV, labelsCsv=labelFileCSV):

    #TODO implementar a verificacao dos csv parseados
    # if False and (os.path.isfile(dataSetRawPath + '\\scratch\\' + file[0:-4] +'.txt')):
    #     pickledFile = open(dataSetRawPath + '\\scratch\\' + file[0:-4] +'.txt', 'rb')
    #     D = (pickle.load(pickledFile))#.reshape(1, -1) + 80

    data = parser(dataSetRawPath+"\\"+file, dataSetRawPath+"\\"+labelsCsv)

    signal = data[0]
    label = data[1]
    length = data[2]

    outFiles = []
    for i in range(length):
        # get the spectrogram of the signal
        D = librosa.amplitude_to_db(np.abs(librosa.stft(signal[i,:],hop_length=1)), ref=np.max)

        outFiles.append(dataSetRawPath + '\\scratch\\' + 'impactos' + str(int(label[i])) + 'inN' + str(i) +'.txt')

        if DEBUG: print("Saving    at: " + outFiles[-1]);
        pickle.dump(D, open(outFiles[-1], 'wb'))

    with open(scratchFilesListRAW, 'w') as file:
        file.write("\n".join(outFiles))


def setLayersAndCompile(shape=0, outputs=0, convProps=None, givenLayers = None):
    layers = []

    if convProps != None:
        # TODO: DEPRECATE THIS?
        # First layer needs input_shape
        layers.append(keras.layers.Conv2D(convProps[0]['nFilters'], convProps[0]['convSize'], activation='relu', input_shape=shape))
        if convProps[0]['Pooling'] != None: layers.append(keras.layers.MaxPooling2D(convProps[0]['Pooling'][0], convProps[0]['Pooling'][1]))
        layers.append(keras.layers.Dropout(convProps[0]['dropOut'], seed=42))

        for conv in convProps[1:]:
            layers.append(keras.layers.Conv2D(conv['nFilters'], conv['convSize'], activation='relu'))
            if conv['Pooling'] != None: layers.append(keras.layers.MaxPooling2D(conv['Pooling'][0], conv['Pooling'][1]))
            layers.append(keras.layers.Dropout(conv['dropOut'], seed=42))

        layers.append(keras.layers.Flatten())
        layers.append(keras.layers.Dense(128, activation='relu'))
        layers.append(keras.layers.Dense(outputs, activation=tf.nn.softmax))
    elif givenLayers != None:
        layers = givenLayers
    else:
        raise Exception ("No convProps or givenLayers")

    model = keras.Sequential(layers)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001) , loss='categorical_crossentropy', metrics=['accuracy', 'categorical_accuracy'])

    return model

def saveModel(epochs, convFilters, comments, convSizes,history, model, dropOut, Pooling):

    print("Saving at Results.txt...")
    modelName2save = 'savedModel' + 'Epoch' + str(epochs) + \
                     'filters' + str(convFilters) + \
                     'convSizes' + str(convSizes) + \
                     'drop' + str(dropOut) + \
                     'Pool' + str(Pooling) + \
                     comments

    with open("Results.txt", "a") as myfile:
        myfile.write(modelName2save + ": ")
        myfile.write(str(history.history))
        myfile.write('\r\n')

    print("Saving Model...")
    model.save(modelName2save)

    return str(history.history['val_acc'][-1] ) + modelName2save


def main(convProps, givenBatches=None, epochs=300, dictOfOutputs={}, batch_size=32, modelVerbose=1, comments="", save=True, givenLayers=None):

    # Generate batches if none given
    if givenBatches == None:
        generateScratch()
        (trainingSet, avaliatiSet, avaliati) = prepareBatches(dictOfOutputs)
    else:
        trainingSet = givenBatches[0]
        avaliatiSet = givenBatches[1]
        avaliati = None

    # Generate keras model and compile
    model = setLayersAndCompile(shape=trainingSet[0][0].shape, outputs=len(dictOfOutputs), convProps=convProps, givenLayers=givenLayers)

    #fit keras model
    print("Fitting...")
    history = model.fit(trainingSet[0], trainingSet[1],
                        epochs=epochs, validation_data=avaliatiSet,
                        batch_size=batch_size, verbose=modelVerbose)

    #print confusion matrix
    confusionMatrixPrint(avaliatiSet, model)

    #save keras model
    #TODO implement save without convProps
    if save and convProps:
        ret = saveModel(epochs,   convFilters=[d['nFilters'] for d in convProps],
                        convSizes  =[d['convSize'] for d in convProps],
                        dropOut  =[d['dropOut'] for d in convProps],
                        Pooling  =[d['Pooling'] for d in convProps],
                        comments="vibracoesMesa"+comments, history=history, model=model)
    else:
        ret = None

    return ret


if __name__ == '__main__':
    #Esse aqui parece bom
    # layers = [
    #             keras.layers.Flatten(input_shape=(1025,201,1)),
    #             keras.layers.Dense(128),
    #             keras.layers.Dense(128),
    #             keras.layers.Dense(128),
    #             keras.layers.Dense(3,  activation=tf.nn.softmax)]
    #main(convProps=None, dictOfOutputs=distancesDict, givenLayers=layers)

    layers = [keras.layers.Conv2D(4, (2, 2), activation='relu', input_shape=(1025, 201, 1)),
              keras.layers.MaxPooling2D(8, 8),
              keras.layers.Flatten(),
              keras.layers.Dense(128),
              keras.layers.Dense(3, activation=tf.nn.softmax)]

    main(convProps=None, dictOfOutputs=distancesDict, givenLayers=layers, epochs=50)
