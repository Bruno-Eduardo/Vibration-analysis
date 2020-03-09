
import numpy as np
import librosa

from generalUtil import csv2array, csv2array3D

class Dataset():
    def __init__(   self,
                    dataSetRawPath,
                    scratchFilesListRAW,
                    distancesDict,
                    dataFileCSV = None,
                    labelFileCSV = None,
                    shapeIsRevelevant = True):
        self.dataSetRawPath         = dataSetRawPath
        self.scratchFilesListRAW    = scratchFilesListRAW
        self.distancesDict          = distancesDict
        self.NofOutputs             = len(self.distancesDict)
        self.parser                 = csv2array
        self.dataFileCSV            = dataFileCSV
        self.labelFileCSV           = labelFileCSV
        self.shape                  = self.getShapeFromFirstSample(shapeIsRevelevant)

    def getShapeFromFirstSample(self, shapeIsRevelevant):
        if not shapeIsRevelevant:
            return None

        data = self.parser(self.dataSetRawPath + "\\" + self.dataFileCSV, self.dataSetRawPath + "\\" + self.labelFileCSV)
        signal = data[0]    # TODO this method runs the whole list of files to get just the first data. \
                            #   Needs a better implementation
        return librosa.amplitude_to_db(np.abs(librosa.stft(signal[0,:],hop_length=1)), ref=np.max).shape

    def __str__(self):
        return str(self.__dict__)

class DatasetNdimentional(Dataset):
    def __init__(self, path, **kwargs):
        super(DatasetNdimentional, self).__init__(path,
                                                  None,
                                                  {},
                                                  **kwargs)
        self.parser = csv2array3D

    def parse(self):
        return self.parser(classes=self.classes,
                           dimensions=self.dimensions,
                           path=self.path)



leituraMesa     = Dataset(r"F:\BrunoDeepLearning\ICvibracoesMesa\leitura0710",
                          r"F:\BrunoDeepLearning\ICvibracoesMesa\VibrationsScratchFiles.txt",
                          {"impactos1":0, "impactos4":1, "impactos8":2},
                          dataFileCSV = 'impactos.csv',
                          labelFileCSV = 'labels.csv',
                          shapeIsRevelevant=True)

simulado3out    = Dataset(r"F:\BrunoDeepLearning\ICvibracoesMesa\vibracoesSimuladas",
                          r"F:\BrunoDeepLearning\ICvibracoesMesa\SimulatedVibrationsScratchFiles.txt",
                          {"impactos1":0, "impactos2":1, "impactos3":2},
                          dataFileCSV = 'impactos.csv',
                          labelFileCSV = 'labels.csv')

simulado10out   = Dataset(r"F:\BrunoDeepLearning\ICvibracoesMesa\vibracoesSimuladasMuitoDiscreta",
                          r"F:\BrunoDeepLearning\ICvibracoesMesa\SimulatedVibrationsTenCategoriesScratchFiles.txt",
                          {"impactos1": 0,
                           "impactos2": 1,
                           "impactos3": 2,
                           "impactos4": 3,
                           "impactos5": 4,
                           "impactos6": 5,
                           "impactos7": 6,
                           "impactos8": 7,
                           "impactos9": 8,
                           "impactos10": 9},
                          dataFileCSV = 'impactos.csv',
                          labelFileCSV = 'labels.csv')

leitura1902     = DatasetNdimentional(r"F:\BrunoDeepLearning\ICvibracoesMesa\amostras1902\parsed")

if __name__ == '__main__':
    print(leituraMesa)
    print(simulado3out)
    print(simulado10out)
