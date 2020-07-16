from telegram import *

import os
import time
import numpy as np
import re

import matplotlib.pyplot as plt
import librosa
import librosa.display
import pickle

from sklearn.metrics import confusion_matrix


def make_spectrogram_N_dim_and_pickle(csv_file, out_name):
    data = np.loadtxt(csv_file, delimiter=',')
    n_of_time_samples, n_of_dim = data.shape
    spectrograms_list = []

    for dim in range(n_of_dim):
        D = make_spectrogram(np.asfortranarray(data[:,dim]),
                             hop_length=n_of_time_samples//200+1)

        if D.shape[1] < 192:
            #print('sample too small'); print('Discarding');
            return False

        spectrograms_list.append(D[:,0:192])

    spectrograms = np.stack(spectrograms_list, axis=2)
    pickle.dump(spectrograms, open(out_name, 'wb'))

    return True

def make_spectrogram_and_pickle(signal, out_name, hop_length=1, ref=np.max):
    D = make_spectrogram(signal, hop_length=hop_length, ref=ref)
    pickle.dump(D, open(out_name, 'wb'))


def make_spectrogram(signal, hop_length=1, ref=np.max):
    try:
        dimensions = signal.shape[1]
        stacked_data = []

        for dim in range(dimensions):
             stacked_data.append(make_spectrogram(signal[:,dim], hop_length, ref))

        return np.stack(stacked_data)
    except IndexError:
        # Data is unidimensional so just make the spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(signal, hop_length=hop_length)), ref=ref)
        return D


def csv2array(csvDataFileName, csvLabelsFileName=None):
    data = np.loadtxt(csvDataFileName, delimiter=',')
    if csvLabelsFileName is not None:
        labels = np.loadtxt(csvLabelsFileName, delimiter=',')
        length = labels.shape[0]

        return data, labels, length
    return data


def get_meta_info_from_file_name(file_name):
    splited = re.split(r"(\d+|\.)", file_name)
    return splited[0:3]


def csv2array3D(files_list, classes=["a"], dimensions=("x", "y", "z"), path="./"):

    # FIXME O erro esta aqui
    #    A stack nao tem tamanho constante então np.stack nao funciona
    #   update:dimensions esta vindo {'N'}

    length = 0
    labels = []
    ready_data = []
    empty_list = [[] for _ in dimensions]
    dimensions_with_list = zip(dimensions, empty_list)

    files_with_path = [os.path.join(path, file) for file in files_list]
    mapped_files = {}

    # create data structure
    for class_ in classes:
        mapped_files[class_] = {k: v for k, v in dimensions_with_list}

    i=0
    for file, file_with_path in zip(files_list, files_with_path):
        print(100 * i/len(files_list)); i += 1

        file_class, exec_number, dim = get_meta_info_from_file_name(file) # TODO use dimensions axis
        data = csv2array(file_with_path)[:,1] # read csv and get just the data, ignore time

        try:
            mapped_files[file_class][exec_number].append(data)
        except KeyError:
            mapped_files[file_class][exec_number] = [data]

        if len(mapped_files[file_class][exec_number]) == len(dimensions):
            ready_data.append(np.stack(mapped_files[file_class][exec_number], axis=1))

    return np.stack(ready_data, axis=0)


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
