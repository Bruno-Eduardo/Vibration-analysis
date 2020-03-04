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
    length = 0
    labels = []
    ready_data = []
    empty_list = [[] for _ in dimensions]
    dimensions_with_list = zip(dimensions, empty_list)

    files = os.listdir(path)
    files_with_path = [os.path.join(path, file) for file in files]
    mapped_files = {}

    # create data structure
    for class_ in classes:
        mapped_files[class_] = {k: v for k, v in dimensions_with_list}

    for file, file_with_path in zip(files, files_with_path):
        if not file.endswith('.csv'):
            continue

        file_class, exec_number = re.split(r"(\d+)",file)[0:2]
        data = csv2array(file_with_path)[:,1] # read csv and get just the data, ignore time

        try:
            mapped_files[file_class][exec_number].append(data)
        except KeyError:
            mapped_files[file_class][exec_number] = [data]

        if len(mapped_files[file_class][exec_number]) == len(dimensions):
            ready_data.append(np.stack(mapped_files[file_class][exec_number], axis=1))
            labels.append(int(exec_number))
            length += 1

    return np.stack(ready_data, axis=0), labels, length


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
