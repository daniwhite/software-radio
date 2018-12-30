import pickle
import numpy
from wav_utils import *
from numpy.fft import fft, ifft

with open('signal.pkl', 'rb') as f:
    signal = pickle.load(f)
