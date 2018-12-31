import pickle
import numpy as np
from wav_utils import *
from numpy.fft import fft, ifft

# Parameters determined by lab
TIMESTEP = 1e-7  # sec
TOTAL_TIME = 10  # sec
MAX_FREQ = 5e3  # Hz
WAV_SR = 44100  # Hz

# Parameters that can be changed from run to run
# length of time analyzed, in seconds (TOTAL_TIME is whole song)
ANALYSIS_TIME = 1
TIMES = np.arange(0, ANALYSIS_TIME, TIMESTEP)
ANALYSIS_INDX = len(TIMES)
CAR_FREQ = 560e3  # carrier frequency in Hz


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
    print("Signal len: {}, timestep: {}, index: {}".format(len(sig), timestep, cutoff_indx))

    freq_sig[cutoff_indx:] = 0
    sig = ifft(freq_sig)
    

def demod(sig):
    """
    Synchronous demodulation (see lec slides 20-22)

    sig -- numpy array of d(t)
    """

    cosines = np.cos(2 * np.pi * CAR_FREQ * TIMES)
    sig = cosines * sig

    # CAR_FREQ is a good a cutoff as any, given that it should scale with the
    # carrier frequency anyway
    lo_pass(sig, CAR_FREQ, TIMESTEP)


with open('signal.pkl', 'rb') as f:
    signal = pickle.load(open('signal.pkl', 'rb'))
    sig = signal[:ANALYSIS_INDX]
    
    demod(sig)

    wav_write(sig, WAV_SR, "out.wav")
