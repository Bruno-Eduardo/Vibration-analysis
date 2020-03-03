from telegram import *

import os
import time
import numpy as np
import re

import matplotlib.pyplot as plt
import librosa
import librosa.display

from sklearn.metrics import confusion_matrix


def csv2array(csvDataFileName, csvLabelsFileName=None):
    data = np.loadtxt(csvDataFileName, delimiter=',')
    if csvLabelsFileName is not None:
        labels = np.loadtxt(csvLabelsFileName, delimiter=',')
        length = labels.shape[0]

        return data, labels, length
    return data


def csv2array3D(classes=["a"], dimensions=("x", "y", "z"), path="./"):
    labels = []
    length = 0
    empty_list = [[] for dim in dimensions]
    dimensions_with_list = zip(dimensions, empty_list)

    files = os.listdir(path)
    mapped_files = {}

    # create data structure
    for class_ in classes:
        mapped_files[class_] = {k: v for k, v in dimensions_with_list}

    for file in files:
        file_class = re.match(r".*(?=\d+)", file).group(0)  # regex to find class #FIXME crashes with 2 digts value
        exec_number = re.search(r"\d+", "abc28x.txt").group(0)  # regex to find which execution
        data = csv2array(file)

        try:
            mapped_files[file_class][exec_number].append(data)
        except KeyError:
            mapped_files[file_class][exec_number] = [data]

        if len(mapped_files[file_class][exec_number]) == len(dimensions):
            SOME_DATA_STRUCT += np.stack(mapped_files[file_class][exec_number], axis=2)
            labels.append(int(exec_number))
            length += 1

    data = parse(SOME_DATA_STRUCT)

    return data, labels, length


def plotD(D):
    librosa.display.specshow(D, y_axis='linear', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram')
    plt.tight_layout();
    plt.show()


def debug(arg, parameter=None, stop=False):
    print("type: " + str(type(arg)))
    print("value: " + str(arg))
    try:
        print("attr: " + str(getattr(arg, parameter)))
    except:
        pass
    if stop:
        time.sleep(10 ** 4)


def confusionMatrixPrint(avaliatiSet, model):
    print("Confusion matrix")
    print(confusion_matrix(np.argmax(avaliatiSet[1], axis=1), model.predict_classes(avaliatiSet[0])))
    print("Labels: \t" + str(np.argmax(avaliatiSet[1], axis=1)))
    print("Predicted:\t" + str(model.predict_classes(avaliatiSet[0])))


def quit(reason=None):
    import sys
    print("QUIT called")
    if reason:
        print("Reason:", reason)
    sys.exit()
