#!/usr/bin/pdb3
from main import *

print("done importing main!")

#mock
K = 10
val_cat = []

DEBUG = True
plot_enable = True

ret = main(dictOfOutputs=sample.distancesDict,
           batch_size=57,
           layers=goodLayers.get_a_layer(keras, sample),
           epochs=1000)
