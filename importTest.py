#!/usr/bin/env python3
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import time
import tensorflow as tf
import statistics
import requests
import re
import random
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import librosa
import goodLayers
from tensorflow.keras import backend as Kbackend
from tensorflow import keras
from telegram import *
from sklearn.metrics import confusion_matrix
