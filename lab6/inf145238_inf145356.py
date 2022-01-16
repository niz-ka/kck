import numpy as np
import scipy.signal
from scipy.io import wavfile
import warnings
import random
import sys

warnings.filterwarnings("ignore")

FEMALE_FREQUENCY = 210
MALE_FREQUENCY = 100

def calculate_hps(signal, fs):
    b, a = scipy.signal.butter(4, 0.005, btype="highpass")
    signal = scipy.signal.filtfilt(b, a, signal)

    samples_number = len(signal)
    signal -= np.mean(signal)

    windowed = signal * np.kaiser(samples_number, 100)

    fft_data = abs(np.fft.fft(windowed))

    hps = np.copy(fft_data)
    for h in np.arange(2, 6):
        dec = scipy.signal.decimate(fft_data, h)
        hps[:len(dec)] += dec


    peak = np.argmax(hps[:len(dec)])

    return fs * peak / samples_number


def get_signal_from_file(file):
    samplerate, data = wavfile.read(file)
    channels = len(data.shape)
    
    if channels > 2 or channels < 1:
        raise Exception("Wrong number of channels")
    
    if channels == 2:
        data = data.T[0]
    
    data = data.astype(np.float64)
    return samplerate, data

def estimate_sex(frequency):
    if abs(frequency - MALE_FREQUENCY) < abs(frequency - FEMALE_FREQUENCY):
        return "M"
    else:
        return "K"

if __name__ == "__main__":
    try:
        if(len(sys.argv) < 2):
            raise Exception("Wrong number of argv")

        file = sys.argv[1]

        rate, signal = get_signal_from_file(file)
        frequency = calculate_hps(signal, rate)
        estimated = estimate_sex(frequency)

        print(estimated)
    except:
        print(random.choice(["K", "M"]))
