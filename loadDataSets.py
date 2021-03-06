import os
import numpy as np
import librosa

from generalUtil import csv2array, csv2array3D, make_spectrogram, get_meta_info_from_file_name

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
        self.spacer                 = 'inN'
        self.csv_file_list          = list(filter(lambda x: x.endswith('.csv'), os.listdir(self.dataSetRawPath)))
        self.length                 = len(self.csv_file_list)
        self.mainName               = 'impactos'
        self.channels               = 1
        self.shape                  = self.getShapeFromFirstSample(shapeIsRevelevant)

    def getShapeFromFirstSample(self, shapeIsRevelevant):
        if not shapeIsRevelevant or self.labelFileCSV is None:
            return None

        data = self.parser(os.path.join(self.dataSetRawPath, self.dataFileCSV),
		           os.path.join(self.dataSetRawPath, self.labelFileCSV))
        signal = data[0]    # TODO this method runs the whole list of files to get just the first data. \
                            #   Needs a better implementation
        return librosa.amplitude_to_db(np.abs(librosa.stft(signal[0,:],hop_length=1)), ref=np.max).shape

    def parse(self, file=None, labels_csv=None):
        file = self.dataFileCSV if file is None else file
        labels_csv = self.labelFileCSV if labels_csv is None else labels_csv

        return self.parser(os.path.join(self.dataSetRawPath, file),
                           os.path.join(self.dataSetRawPath, labels_csv))

    def get_out_name(self, class_, iteration, main_name=None, spacer=None, extension='.txt'):
        main_name = self.mainName if main_name is None else main_name
        spacer = self.spacer if spacer is None else spacer

        return os.path.join(self.dataSetRawPath, 'scratch', main_name, class_,spacer, str(iteration), extension)

    def __str__(self):
        return str(self.__dict__)

class DatasetNdimentional(Dataset):
    def __init__(self, path, **kwargs):
        super(DatasetNdimentional, self).__init__(path,
                                                  os.path.join(path, 'scratchFiles.txt'),
                                                  {},
                                                  **kwargs)
        print(self.channels)
        self.parser = csv2array3D
        self.classes, self.dimensions = self.get_meta_info()
        self.channels = 3 # TODO len(self.dimensions)

    def get_meta_info(self):
        classes = set()
        dimensions = set()

        for file in self.csv_file_list:
            class_, exec_number, dimension = get_meta_info_from_file_name(file)
            classes.add(class_)
            dimensions.add(dimension)

        return classes, dimensions

    def parse(self, **kargs):
        return self.parser(self.csv_file_list,
                           classes=self.classes,
                           dimensions=self.dimensions,
                           path=self.dataSetRawPath)

    def get_first_sample(self):
        return self.csv_file_list[0]

    def getShapeFromFirstSample(self, shapeIsRevelevant):
        first_sample = self.get_first_sample()
        first_data = csv2array(os.path.join(self.dataSetRawPath, first_sample))[:,0]
        try:
            signal = make_spectrogram(first_data)
        except librosa.util.exceptions.ParameterError:
            signal = make_spectrogram(np.asfortranarray(first_data.copy()))
        return signal.shape

    def get_label_list(self):
        list_of_labels = []
        for file in self.csv_file_list:
            _, label, _ = get_meta_info_from_file_name(file)
            list_of_labels.append(label)
        return list_of_labels

leitura1902     = DatasetNdimentional(os.path.join(r"../ICvibracoesMesa/amostras1902", "splited"))

if __name__ == '__main__':
    print(leituraMesa)
    print(simulado3out)
    print(simulado10out)
