import pickle
import numpy as np
from wav_utils import *
from numpy.fft import fft, ifft
import time

## Parameters determined by lab
TIMESTEP = 1e-7  # sec
TOTAL_TIME = 10  # sec
MAX_FREQ = 5e3  # Hz
WAV_SR = 44100  # Hz
CAR_FREQS = np.arange(560e3, 800e3, 10e3)  # all in hz

## Parameters that can be changed from run to run
# length of time analyzed, in seconds (TOTAL_TIME is whole song)
ANALYSIS_TIME = TOTAL_TIME
TIMES = np.arange(0, ANALYSIS_TIME, TIMESTEP)
ANALYSIS_INDX = len(TIMES)


def zoh(sig, old_period, new_period):
    """Periods are in seconds."""
    new_len = int(len(sig) * old_period / new_period)
    ret = np.zeros(new_len)

    for ret_indx in range(new_len):
      sig_indx = int(ret_indx * new_period / old_period)
      ret[ret_indx] = sig[sig_indx]
    return ret


def lo_pass(sig, cutoff, timestep):
    """
    sig    -- numpy array to be filtered
    cutoff -- frequency above which to zero (Hz)
    """

    freq_sig = fft(sig)

    # This should be correct because, as the numpy docs point out, indx = n/2
    # has the Nyquist frequency, so cutoff should be at 
    # cutoff * len(sig) / sampling_freq, and sampling_freq = 1 / timestep
    # Also the units work out
    cutoff_indx = int(cutoff * len(sig) * timestep)

    freq_sig[cutoff_indx:] = 0
    return np.abs(ifft(freq_sig))
    

def demod(sig, car_freq):
    """
    Synchronous demodulation (see lec slides 20-22)

    sig -- numpy array of d(t)
    """

    cosines = np.cos(2 * np.pi * car_freq * TIMES)
    sig = cosines * sig

    return lo_pass(sig, MAX_FREQ, TIMESTEP)


with open('signal.pkl', 'rb') as f:
    signal = pickle.load(open('signal.pkl', 'rb'))
    trunc_sig = signal[:ANALYSIS_INDX]

    for car_freq in CAR_FREQS:
        sig = trunc_sig[:]
        start_time = time.time()
        sig = demod(sig, car_freq)
        sig = zoh(sig, TIMESTEP, 1/WAV_SR)

        wav_write(sig, WAV_SR, "song_{}khz.wav".format(int(car_freq/1000)))
        end_time = time.time()
        print("Time for {} Hz: {}".format(car_freq, end_time - start_time))
