# DT2119, Lab 1 Feature Extraction

# Function given by the exercise ----------------------------------
import numpy as np
from scipy import signal
from scipy.fftpack import fft
from lab1_tools import *
from scipy.fftpack.realtransforms import dct
from scipy.spatial.distance import cdist

def mspec(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    """Computes Mel Filterbank features.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        samplingrate: sampling rate of the original signal

    Returns:
        N x nfilters array with mel filterbank features (see trfbank for nfilters)
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    mspecc = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspecc, nceps)
    return lifter(ceps, liftercoeff)

# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift, samplingrate = None):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
        samplingrate: is not None when winlen and winshift are given in ms
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
    if samplingrate is not None and samplingrate < 50: # <50 to capture cases when samplingrate is given eventhough the lenght are given in nr of samples
        srate=int(samplingrate/1000) # samples per ms
        winlen = srate * winlen # length of frame in nr of samples
        winshift = srate * winshift # shift of frames in nr of samples

    N = int((len(samples) - winlen) / winshift + 1) # nr of frames
    print("We are creating {} frames with {} samples each and {} samples overlapping.".format(N, winlen, winshift))
    frames = np.zeros((N, winlen))
    frames[0,:] = samples[:winlen]

    ix = winshift
    for i in range(1, N):
        frames[i,:] = samples[ix:ix+winlen]
        ix += winshift
    return frames
    
def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """
    N = input.shape[0]
    result = np.copy(input)

    # FIR Filter
    b = [1,-p]
    a = [1, 0]
    for i in range(N):
        result[i,:] = signal.lfilter(b,a, input[i,:])
    return result

def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """
    _,M = input.shape
    window = signal.hamming(M,sym=False)
    return input * window

def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    return np.abs(fft(input, nfft))**2

def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    filterbank = trfbank(samplingrate, 512)
    N,_ = input.shape
    M,_ = filterbank.shape
    result = np.zeros((N,M))
    for m in range(M):
        result[:,m] = [np.log(np.dot(filterbank[m,:],input[i,:])) for i in range(N) ]
    return result


def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """
    mfcc = dct(input)
    mfcc = mfcc[:,0:nceps]
    #return lifter(mfcc)
    return mfcc

def dtw(x, y, dist='euclidean',onlyd=False):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AccD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """

    LD = cdist(x,y,dist)
    #print(D)
    H,K = LD.shape
    AccD = np.zeros_like(LD)
    backpointers = np.zeros((2,H,K))
    a = [[1,0],[1,1],[0,1]]
    for h in range(H):
        for k in range(K):
            if h>0 and k>0:
                ind = np.argmin([AccD[h-1,k],AccD[h-1,k-1],AccD[h,k-1]])
                AccD[h,k] = LD[h,k] + np.min([AccD[h-1,k],AccD[h-1,k-1],AccD[h,k-1]])
                backpointers[:,h,k] = a[ind]

            elif k>0 and h==0:
                ind = 2
                AccD[h,k] = LD[h,k] + AccD[h,k-1]
                backpointers[:,h,k] = a[ind]
                
            elif h>0 and k==0:
                ind = 0
                AccD[h,k] = LD[h,k] + AccD[h-1,k]
                backpointers[:,h,k] = a[ind]

            elif h==0 and k==0:
#                 ind = 1
                AccD[h,k] = LD[h,k]
                backpointers[:,h,k] = [0,0]
            else:
                print("error")

                
    d = AccD[-1,-1]
    d = d/(H+K)
    if onlyd:
        return d


    path = []
    h = H-1
    k = K-1
    while True:
        path = path+[[k,h]]
        h,k = np.array([h,k])-np.array(backpointers[:,h,k],dtype=int)
        if h==0 and k==0:
            break
    return d,AccD,path,LD
