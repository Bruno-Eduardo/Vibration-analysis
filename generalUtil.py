
from telegram import *

import time
import numpy as np

import matplotlib.pyplot as plt
import librosa
import librosa.display

from sklearn.metrics import confusion_matrix

def csv2array(csvDataFileName, csvLabelsFileName):
    data = np.loadtxt(csvDataFileName, delimiter=',')
    labels = np.loadtxt(csvLabelsFileName, delimiter=',')
    length = labels.shape[0]

    return (data, labels, length)

def plotD(D):
    librosa.display.specshow(D, y_axis='linear', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram')
    plt.tight_layout();plt.show()

def debug(arg, parameter=None, stop = False):
    print("type: " + str(type(arg)))
    print("value: " + str(arg))
    try:
        print("attr: " + str(getattr(arg, parameter)))
    except:
        pass
    if stop:
        time.sleep(10**4)

def confusionMatrixPrint(avaliatiSet, model):
    print("Confusion matrix")
    print(confusion_matrix(np.argmax(avaliatiSet[1], axis=1), model.predict_classes(avaliatiSet[0])))
    print("Labels: \t" + str(np.argmax(avaliatiSet[1], axis=1)))
    print("Predicted:\t" + str(model.predict_classes(avaliatiSet[0])))