import os
import pickle
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
from numpy.random import RandomState
import argparse
import tqdm


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def pySTFT(x, fft_length=1024, hop_length=256):

    x = np.pad(x, int(fft_length//2), mode='reflect')

    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)

    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T

    return np.abs(result)


def make_spec(datasetDir = "training_set"):
    mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
    min_level = np.exp(-100 / 20 * np.log(10))
    b, a = butter_highpass(30, 16000, order=5)


    # audio file directory
    rootDir = datasetDir + '/wavs'
    # spectrogram directory
    targetDir = datasetDir + '/spmel'

    if not os.path.exists(targetDir):
        os.mkdir(targetDir)

    dirs= os.listdir(rootDir)
    print('Processing speakers :')
    for speaker in tqdm.tqdm(dirs):
        rootDirName = f"{rootDir}/{speaker}/"
        targetDirName = f"{targetDir}/{speaker}/"
        if not os.path.exists(targetDirName):
            os.mkdir(targetDirName)
        for dirName, dirs, files in os.walk(rootDirName):
            subfolder = ''
            if len(dirName.split('/')):
                subfolder = dirName.split('/')[-1]
            for fileName in files:
                #prng = RandomState(int(subdir[1:]))
                prng = RandomState(1)
                # Read audio file
                x, fs = sf.read(os.path.join(dirName,fileName))
                # Remove drifting noise
                y = signal.filtfilt(b, a, x)
                # Ddd a little random noise for model roubstness
                wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
                # Compute spect
                D = pySTFT(wav).T
                # Convert to mel and normalize
                D_mel = np.dot(D, mel_basis)
                D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
                S = np.clip((D_db + 100) / 100, 0, 1)
                # save spect
                np.save(os.path.join(targetDirName, subfolder+fileName[:-4]),
                        S.astype(np.float32), allow_pickle=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset dir
    parser.add_argument('--dataset', type=str, default="training_set", help='dataset dir')
    config = parser.parse_args()
    make_spec(config.dataset)
