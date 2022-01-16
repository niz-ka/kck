import numpy as np
import scipy.signal
from scipy.io import wavfile
import warnings
import os

# Suppress all stupid warnings
warnings.filterwarnings("ignore")

######
DIRECTORY = "data/"     # Remember about the slash!
FEMALE_FREQUENCY = 210
MALE_FREQUENCY = 100
######


def calculate_hps(signal, fs):
    """
    Calculate fundamental frequency using Harmonic Product Spectrum
    
    Args:
        signal (np.array): input signal
        fs (number): sampling frequency

    Returns:
        number: fundamental frequency
    """
    
    # High-pass filter
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


def get_signal_from_file(filename):
    """
    Read .wav file
    
    Args:
        filename (string): input filename (NOT PATH!)

    Returns:
        number: sampling frequency
        np.array: signal
    """
    samplerate, data = wavfile.read(DIRECTORY + filename)
    channels = len(data.shape)
    
    if channels > 2 or channels < 1:
        raise Exception("Wrong number of channels")
    
    if channels == 2:
        data = data.T[0]
    
    data = data.astype(np.float64)
    return samplerate, data


def get_sex_from_filename(filename):
    """
    Recognize correct sex through filename
    
    Args:
        filename (string): input filename (file doesn't have to exist)

    Returns:
        string: "K" when female or "M" when male
    """
    
    splitted = filename.split("_")
    sex = splitted[-1][0]
    if(sex != "K" and sex != "M"):
        raise Exception("Unknown sex!")
    return sex

def estimate_sex(frequency):
    """
    Estimates sex using frequency distance
    
    Args:
        frequency (number): voice frequency (Hz)

    Returns:
        string: "K" when female or "M" when male
    """

    if abs(frequency - MALE_FREQUENCY) < abs(frequency - FEMALE_FREQUENCY):
        return "M"
    else:
        return "K"

if __name__ == "__main__":
    audio_files = os.listdir(DIRECTORY)
    number_of_files = len(audio_files)
    correct = 0

    for file in audio_files:
        rate, signal = get_signal_from_file(file)
        frequency = calculate_hps(signal, rate)
        expected = get_sex_from_filename(file)
        estimated = estimate_sex(frequency)
        if expected == estimated:
            correct += 1
   
    correctness = (correct / number_of_files) * 100
    print(f'Próbek: {number_of_files} | Poprawnych: {correct} | Poprawność: {correctness:.2f}%')