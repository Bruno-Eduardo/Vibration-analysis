#!/usr/bin/python3
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import re
import random
import pickle
import statistics

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as Kbackend

from loadDataSets import leituraMesa, simulado3out, simulado10out, leitura1902, DatasetNdimentional
from generalUtil import np, csv2array, confusionMatrixPrint, make_spectrogram_and_pickle
from generalUtil import plotD, debug, quit

from generalUtil import make_spectrogram_N_dim_and_pickle # TODO: gambiarra aqui. Deveria ser pelo sample

import goodLayers

print('Done!')

sample = leitura1902

DEBUG = True


def getBatch(set2process, dictOfOutputs, size=-1, reset=False):
    if size == -1:
        size = len(set2process)
    lastPrintedFlag = -1

    batchX, batchY = [], []

    for file, counter in zip(set2process, range(size)):

        pickledFile = open(os.path.join(sample.dataSetRawPath,
                                        "scratch",
                                        file), 'rb')
        if DEBUG: print(file, end="")

        if "esp" in file:
            continue #print('stop')

        # Prepare D
        D = pickle.load(pickledFile)+80
        D = np.resize(D, (D.shape[0], D.shape[1], sample.channels))  # making D a "channel last" tensor
        D = D[:,:,0].astype(np.uint8)
        D = np.resize(D, (D.shape[0], D.shape[1], 1))
        # plotD(D[:,:,0])                                       #----------------------------------- Remover para plotar as ffts

        # Prepare Y (one hot array)
        Y = np.zeros(len(dictOfOutputs), dtype=int)
        Y[dictOfOutputs[re.sub(r'\d+N.*', '', file)]] = 1
        if DEBUG: print(", label:" + str(Y))


        #if re.sub(r'\d+N.*', '', file) not in ["z", "q"]:
        #    continue

        batchX.append(D)
        batchY.append(Y)

        # DEBUG print
        printCandidate = int(10 * counter / size)
        if lastPrintedFlag != printCandidate:
            lastPrintedFlag = printCandidate
            if DEBUG: print(str(10 * printCandidate) + "%")

    print("stacking...")
    batchX = np.stack(batchX, axis=0)
    batchY = np.stack(batchY, axis=0)
    print("stacking DONE!")

    return batchX[:, :, :, :], batchY


def prepareBatches(dictOfOutputs):
    scratchFilesList = []
    training = []
    crssvald = []
    avaliati = []

    with open(sample.scratchFilesListRAW, 'r', encoding="utf8") as file:
        for line in file:
            line = line.replace('\n', '\r').replace('\r', '') # TODO check @ windows
            line = line.replace(os.path.join(sample.dataSetRawPath, "scratch"), "").replace('/', '')
            if 'txt' in line:
                if 'esp' not in line:
                    scratchFilesList.append(line)
            else:
                raise Exception('amostra.scratchFilesListRAW may be corrupted. Check ' + sample.scratchFilesListRAW)

    mappedSamples = {}

    # proportion or numbers of training
    cut = .8
    # cut = int(22/3)

    def getOutput(sampleFile, dictOfOutputs=dictOfOutputs):
        # FIXME WRONG IF SAMPLEFILE IS ESP*
        #if 'p' in sampleFile:
        #    print('p found. Dbg here!')
        possibleOutput = [value for key, value in dictOfOutputs.items() if key == re.sub(r'\d+N.*', '', sampleFile)]
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


def generateScratch(sample, forceNewPickle=False):

    print('generating scratch')
    #TODO refacor this hell
    length = len(sample.csv_file_list)

    out_files = []

    # create scratch directory if it does not exist
    scratch_dir = os.path.join(sample.dataSetRawPath, "scratch")
    if not os.path.exists(scratch_dir):
        os.mkdir(scratch_dir)

    for file in sample.csv_file_list:
        if 'esp' in file:
            continue
        out_name = file.replace('.csv', '.txt')
        out_name_with_path = os.path.join(  sample.dataSetRawPath,
                                            "scratch",
                                            out_name)
        out_files.append(out_name_with_path)
        class_ = re.sub(r'\d+N.*', '', out_name)

        try:
            sample.dictOfOutputs
        except AttributeError:
            sample.dictOfOutputs = {class_: 0}
        if class_ not in sample.dictOfOutputs:
            sample.dictOfOutputs[class_] = 1 + sample.dictOfOutputs[
                                                    max(sample.dictOfOutputs,
                                                    key=sample.dictOfOutputs.get)]

        file_with_path = os.path.join( sample.dataSetRawPath, file)

        if os.path.exists(out_name_with_path) and (not forceNewPickle):
            continue

        success = make_spectrogram_N_dim_and_pickle(file_with_path, out_name_with_path)
        if success:
            if DEBUG: print("Saved at: " + out_name);
        else:
            out_files.pop()

    # write all spectrograms paths at scratchFilesListRAW
    with open(sample.scratchFilesListRAW, 'w') as file:
        file.write("\n".join(out_files))

    print("sample.dictOfOutputs:", sample.dictOfOutputs)

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
        generateScratch(sample, forceNewPickle=False)
        (trainingSet, avaliatiSet, avaliati) = prepareBatches(sample.dictOfOutputs)
    else:
        trainingSet = givenBatches[0]
        avaliatiSet = givenBatches[1]

    # Generate keras model and compile
    model = keras.Sequential(layers)
    model.compile(optimizer=keras.optimizers.Adam(decay=1e-4,
                                                  learning_rate=0.1),
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

    ret = main(dictOfOutputs=sample.distancesDict, batch_size=57, layers=goodLayers.get_a_layer(keras, sample),
               epochs=1000)
    #
    # for _ in range(K):
    #     ret = main(dictOfOutputs=sample.distancesDict, batch_size=16, layers=goodLayers.get_a_layer(keras, sample),
    #                epochs=200)
    #     Kbackend.clear_session()
    #     tf.keras.backend.clear_session()
    #     keras.backend.clear_session()
    #     val_cat.append(ret[0].item() * 100)
    #
    # print(type(val_cat[0]))
    # print(val_cat)
    # print(statistics.mean(val_cat))
    # print(statistics.stdev(val_cat))
