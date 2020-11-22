#!/usr/bin/python3
#import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)

#import os
#import re
#import random
#import pickle
#import statistics
import matplotlib as plt

#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import backend as Kbackend

#from loadDataSets import leituraMesa, simulado3out, simulado10out, leitura1902, DatasetNdimentional
#from generalUtil import np, csv2array, confusionMatrixPrint, make_spectrogram_and_pickle
#from generalUtil import plotD, debug, quit

#from generalUtil import make_spectrogram_N_dim_and_pickle # TODO: gambiarra aqui. Deveria ser pelo sample

#import goodLayers

#print('Done!')

#sample = leitura1902

#DEBUG = False # True
#plot_enable = False # True

class InputParser():
    def __init__(self, debug=False):
        self.debug = debug
        self.batch_parsed = False
        self.last_parsed_batch = None
        self.chars_list = ['z', 'b', 'm', 'a', 'l', 'p'] # automate this

    def get_batch(self, force_new_batch=False):
        if self.batch_parsed:
            return self.last_parsed_batch
        self.password = self.get_a_password()
        self.validade_password()
        self.time_keystrokes = self.get_time_keystrokes()
        self.splitted_keystrokes = self.split_time_keystrokes()
        self.spectrograms = self.make_spectrograms()
        self.last_parsed_batch = self.make_a_batch()

    def get_a_password(self):
        self.password = input("Choose a password, with this chars" + self.chars_list.__str__() + '\n')
        return self.password

    def validade_password(self):
        char_in_list = [password_char in self.chars_list for password_char in self.password]
        if all(char_in_list):
           return True
        else:
           print("Invalid char, try again")
           self.get_a_password()
           self.validade_password()

    def get_time_keystrokes(self):
        self.data_set_eval_list = "some_file.txt" #FIXME
        with open(self.data_set_eval_list, 'r', encoding="utf8") as file:
            for line in file:
                if '.csv' not in line: #TODO better handler
                    continue
                line = line.split('.csv')[0]
                class = re.sub(r'\d+N.*', '', line)
                print(line)
                #for char in self.password:
            

        self.time_keystrokes = None

        if self.debug:
            self.plot_time_array(self.time_keystrokes)
        return self.time_keystrokes

    def plot_time_array(array):
        plt.plot(array)
        plt.show()

    def split_time_keystrokes(self):
        pass

    def make_spectrograms(self):
        pass

    def make_a_batch(self):
        pass

def main():
    input_parser = InputParser()
    input_parser.get_batch()


if __name__ == '__main__':
    main()
