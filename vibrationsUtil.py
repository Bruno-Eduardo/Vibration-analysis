from os import listdir
import os.path
from os import path
import random
import pickle
import statistics

import librosa.display
# import tensorflow.compat.v1 as tf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as Kbackend

from loadDataSets import leituraMesa, simulado3out, simulado10out, leitura1902
from generalUtil import np, csv2array, confusionMatrixPrint
# from generalUtil import plotD, debug, quit

import goodLayers

print('Done!')

sample = simulado10out

files = listdir(sample.dataSetRawPath)
DEBUG = True


def getBatch(set2process, dictOfOutputs, size=-1, reset=False):
    if size == -1:
        size = len(set2process)
    try:
        getBatch.lastProcessedFile  # TODO remove this counter, it beceme unecessary since keras
    except:
        getBatch.lastProcessedFile = 0
    if reset:
        getBatch.lastProcessedFile = 0
        return
    lastPrintedFlag = -1

    for i in range(size):
        try:
            pickledFile = open(sample.dataSetRawPath + '\\scratch\\' + set2process[getBatch.lastProcessedFile], 'rb')
        except:
            getBatch.lastProcessedFile = getBatch.lastProcessedFile % len(set2process)
            pickledFile = open(sample.dataSetRawPath + '\\scratch\\' + set2process[getBatch.lastProcessedFile], 'rb')

        D = (pickle.load(pickledFile)) + 80

        if DEBUG: print(set2process[getBatch.lastProcessedFile], end="")
        # plotD(D)                                       #----------------------------------- Remover para plotar as ffts

        D = np.resize(D, (D.shape[0], D.shape[1], 1))  # making D a "channel last" tensor
        D = D.astype(np.uint8)

        Y = np.zeros(len(dictOfOutputs), dtype=int)
        Y[dictOfOutputs[set2process[getBatch.lastProcessedFile].split('in', 1)[0]]] = 1
        if DEBUG: print(", label:" + str(Y))

        try:
            batchX.append(D)
            batchY.append(Y)
        except NameError:
            batchX = [D]
            batchY = [Y]

        printCandidate = int(10 * getBatch.lastProcessedFile / size)
        if lastPrintedFlag != printCandidate:
            lastPrintedFlag = printCandidate
            if DEBUG: print(str(10 * printCandidate) + "%")
        getBatch.lastProcessedFile += 1

    print("stacking...")
    batchX = np.stack(batchX, axis=0)
    batchY = np.stack(batchY, axis=0)
    print("stacking DONE!")

    getBatch([], dictOfOutputs, reset=True)  # TODO check if still needed
    return batchX[:, :, :, :], batchY


def prepareBatches(dictOfOutputs):
    scratchFilesList = []
    training = []
    crssvald = []
    avaliati = []

    with open(sample.scratchFilesListRAW, 'r', encoding="utf8") as file:
        for line in file:
            line = line.replace('\n', '\r').replace('\r', '').replace('/', '\\')  # TODO find a better parser
            line = line.replace(sample.dataSetRawPath + "\\scratch\\", "")
            if 'txt' in line:
                scratchFilesList.append(line)
            else:
                raise Exception('amostra.scratchFilesListRAW may be corrupted. Check ' + sample.scratchFilesListRAW)

    mappedSamples = {}

    # proportion or numbers of training
    cut = .8

    # cut = int(22/3)

    def getOutput(sampleFile, dictOfOutputs=dictOfOutputs):
        possibleOutput = [value for key, value in dictOfOutputs.items() if key in sampleFile]
        if not possibleOutput: raise Exception(sampleFile + "not found in dict of Outputs")
        return max(possibleOutput)

    for sampleFile in scratchFilesList:
        try:
            mappedSamples[getOutput(sampleFile)].append(sampleFile)
        except KeyError:
            mappedSamples[getOutput(sampleFile)] = [sampleFile]

    for key, value in mappedSamples.items():
        l = mappedSamples[key]
        random.shuffle(l)

        if cut < 1: cut = int(len(l) * cut)  # if cut represents a proportion, parse to absolute number

        training.extend(l[0:cut])
        # crssvald TODO implement cross validation cut
        avaliati.extend(l[cut:])

    random.shuffle(training)
    random.shuffle(crssvald)
    random.shuffle(avaliati)

    trainingSet = getBatch(training, dictOfOutputs)
    avaliatiSet = getBatch(avaliati, dictOfOutputs)

    return trainingSet, avaliatiSet, avaliati


def make_spectrogram_and_pickle(signal, out_name, hop_length=1, ref=np.max):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(signal, hop_length=hop_length)), ref=ref)
    pickle.dump(D, open(out_name, 'wb'))


def generateScratch(sample, forceNewPickle=False):

    signal, label, length = sample.parse()

    out_files = []
    for i in range(length):
        # create scratch directory if it does not exist
        if not path.exists(sample.dataSetRawPath + '\\scratch'):
            os.mkdir(sample.dataSetRawPath + '\\scratch')

        out_name = sample.get_out_name(str(int(label[i])), i)
        out_files.append(out_name)

        if path.exists(out_name) and not forceNewPickle:
            continue  # already parsed and saved

        # get the spectrogram of the signal and saves
        make_spectrogram_and_pickle(signal[i, :], out_name)
        if DEBUG: print("Saved at: " + out_name);

    # write all spectrograms paths at scratchFilesListRAW
    with open(sample.scratchFilesListRAW, 'w') as file:
        file.write("\n".join(out_files))


def saveModel(epochs, convFilters, comments, convSizes, history, model, dropOut, Pooling):
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

    return str(history.history['val_acc'][-1]) + modelName2save


def main(dictOfOutputs, givenBatches=None, epochs=300, batch_size=32, modelVerbose=1, comments="", save=True,
         layers=None, convProps=None):
    # Generate batches if none given
    if givenBatches is None:
        generateScratch(sample)
        (trainingSet, avaliatiSet, avaliati) = prepareBatches(dictOfOutputs)
    else:
        trainingSet = givenBatches[0]
        avaliatiSet = givenBatches[1]

    # Generate keras model and compile
    model = keras.Sequential(layers)
    model.compile(optimizer=keras.optimizers.Adam(decay=1e-6,
                                                  learning_rate=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'categorical_accuracy'])

    # fit keras model
    print("Fitting...")
    history = model.fit(trainingSet[0], trainingSet[1],
                        epochs=epochs, validation_data=avaliatiSet,
                        batch_size=batch_size, verbose=modelVerbose)

    # print confusion matrix
    confusionMatrixPrint(avaliatiSet, model)

    # save keras model
    # TODO implement save without convProps
    if save and convProps:
        ret = saveModel(epochs, convFilters=[d['nFilters'] for d in convProps],
                        convSizes=[d['convSize'] for d in convProps],
                        dropOut=[d['dropOut'] for d in convProps],
                        Pooling=[d['Pooling'] for d in convProps],
                        comments="vibracoesMesa" + comments, history=history, model=model)
    else:
        ret = (history.history['val_acc'][-1], history)

    return ret


if __name__ == '__main__':

    K = 10
    val_cat = []

    for _ in range(K):
        ret = main(dictOfOutputs=sample.distancesDict, batch_size=16, layers=goodLayers.get_a_layer(keras, sample),
                   epochs=200)
        Kbackend.clear_session()
        tf.keras.backend.clear_session()
        keras.backend.clear_session()
        val_cat.append(ret[0].item() * 100)

    print(type(val_cat[0]))
    print(val_cat)
    print(statistics.mean(val_cat))
    print(statistics.stdev(val_cat))
