# -*- coding: utf-8 -*-
"""
A modified version of the python_speech_features library useful to compute different
feature vectors for the speech recognition task

@author: Simone Ceccato

"""

from __future__ import division
import numpy as np
from python_speech_features import sigproc
from scipy.fftpack import dct

def mfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=12,
        nfilt=40, nfft=400, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22):

    logfeat = logfbank(signal, samplerate, winlen, winstep, nfilt, nfft, lowfreq, highfreq, preemph)
    feat = dct(logfeat, type=2, axis=1, norm='ortho')[:,:numcep]
    feat = lifter(feat, ceplifter)
    return feat

def logfbank(signal, samplerate=16000, winlen=0.025, winstep=0.01,
             nfilt=40, nfft=400, lowfreq=0, highfreq=None, preemph=0.97):

    hamming_window = np.hamming
    highfreq = highfreq or samplerate/2
    signal = sigproc.preemphasis(signal, preemph)
    frames = sigproc.framesig(signal, winlen*samplerate, winstep*samplerate, winfunc=hamming_window)
    pspec = sigproc.powspec(frames, nfft)
    energy = np.sum(pspec,1) # this stores the total energy in each frame
    energy = np.where(energy == 0,np.finfo(float).eps,energy) # if energy is zero, we get problems with log

    fb = get_filterbanks(nfilt,nfft,samplerate,lowfreq,highfreq)
    feat = np.dot(pspec,fb.T) # compute the filterbank energies
    feat = np.where(feat == 0,np.finfo(float).eps,feat) # if feat is zero, we get problems with log
    logfeat = np.log(feat)
    return logfeat

def ssc(signal, samplerate=16000, winlen=0.025, winstep=0.01,
        nfilt=40, nfft=400, lowfreq=0, highfreq=None, preemph=0.97):

    hamming_window = np.hamming
    highfreq = highfreq or samplerate/2
    signal = sigproc.preemphasis(signal, preemph)
    frames = sigproc.framesig(signal, winlen*samplerate, winstep*samplerate, winfunc=hamming_window)
    pspec = sigproc.powspec(frames,nfft)
    pspec = np.where(pspec == 0,np.finfo(float).eps,pspec) # if things are all zeros we get problems

    fb = get_filterbanks(nfilt,nfft,samplerate,lowfreq,highfreq)
    feat = np.dot(pspec,fb.T) # compute the filterbank energies
    R = np.tile(np.linspace(1,samplerate/2,np.size(pspec,1)),(np.size(pspec,0),1))

    return np.dot(pspec*R,fb.T) / feat

def hz2mel(hz):
    """Convert a value in Hertz to Mels
    :param hz: a value in Hz. This can also be a np array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1+hz/700.)

def mel2hz(mel):
    """Convert a value in Mels to Hertz
    :param mel: a value in Mels. This can also be a np array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700*(10**(mel/2595.0)-1)

def get_filterbanks(nfilt=20,nfft=512,samplerate=16000,lowfreq=0,highfreq=None):
    highfreq = highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel,highmel,nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = np.floor((nfft+1)*mel2hz(melpoints)/samplerate)

    fbank = np.zeros([nfilt,nfft//2+1])
    for j in range(0,nfilt):
        for i in range(int(bin[j]), int(bin[j+1])):
            fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    return fbank

def lifter(cepstra, L=22):
    if L > 0:
        nframes,ncoeff = np.shape(cepstra)
        n = np.arange(ncoeff)
        lift = 1 + (L/2.)*np.sin(np.pi*n/L)
        return lift*cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra

def delta(feat, N):
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    NUMFRAMES = len(feat)
    denominator = 2 * sum([i**2 for i in range(1, N+1)])
    delta_feat = np.empty_like(feat)
    padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')   # padded version of feat
    for t in range(NUMFRAMES):
        delta_feat[t] = np.dot(np.arange(-N, N+1), padded[t : t+2*N+1]) / denominator   # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
    return delta_feat
