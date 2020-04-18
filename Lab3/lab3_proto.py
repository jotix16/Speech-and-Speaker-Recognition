import pickle
import numpy as np
from lab3_tools import *
from tqdm import tqdm

## Import scripts from lab2
import sys
sys.path.insert(1, '../Lab2/')
from prondict import*

## Import scripts from lab1
import sys
sys.path.insert(1, '../Lab1/')
from lab1_proto import*

## Import scripts from lab2
import sys
sys.path.insert(1, '../Lab2/')
from lab2_proto import*
from prondict import*


def words2phones(wordList, pronDict, addSilence=True, addShortPause=True):
    """ word2phones: converts word level to phone level transcription adding silence

    Args:
       wordList: list of word symbols
       pronDict: pronunciation dictionary. The keys correspond to words in wordList
       addSilence: if True, add initial and final silence
       addShortPause: if True, add short pause model "sp" at end of each word
    Output:
       list of phone symbols
    """
    res = []
    if addSilence: res.append('sil')
    for utt in wordList:
        res += pronDict[utt][:]
        if addShortPause: res.append('sp')

    if addSilence: res.append('sil')
    return res

def forcedAlignment(lmfcc, phoneHMMs, phoneTrans):
    """ forcedAlignmen: aligns a phonetic transcription at the state level

    Args:
       lmfcc: NxD array of MFCC feature vectors (N vectors of dimension D)
              computed the same way as for the training of phoneHMMs
       phoneHMMs: set of phonetic Gaussian HMM models
       phoneTrans: list of phonetic symbols to be aligned including initial and
                   final silence

    Returns:
       list of strings in the form phoneme_index specifying, for each time step
       the state from phoneHMMs corresponding to the viterbi path.
    """

def hmmLoop(hmmmodels, namelist=None):
    """ Combines HMM models in a loop

    Args:
       hmmmodels: list of dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to combine, if None,
                 all the models in hmmmodels are used

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models
       stateMap: map between states in combinedhmm and states in the
                 input models.

    Examples:
       phoneLoop = hmmLoop(phoneHMMs)
       wordLoop = hmmLoop(wordHMMs, ['o', 'z', '1', '2', '3'])
    """

    
def extract_features_and_targets(filename):
    """
    extracts lmfcc, mspecc and targets from a *.wav sound file
    """
    phoneHMMs_all = np.load('../Lab2/lab2_models_all.npz',allow_pickle=True)['phoneHMMs'].item()
    nstates, stateList, phones = pickle.load( open('saved_files/phoneHMM_states.pkl', 'rb'))
    samples, samplingrate = loadAudio(filename)
    wordTrans = list(path2info(filename)[2])
    phoneTrans = words2phones(wordTrans, prondict)
    stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans for stateid in range(nstates[phone])]
    utteranceHMM = concatHMMs(phoneHMMs_all, phoneTrans)

    lmfcc = mfcc(samples)
    mspecc = mspec(samples)

    obsloglik = log_multivariate_normal_density_diag(lmfcc, utteranceHMM['means'], utteranceHMM['covars'])
    viterbi_loglik, viterbi_path = viterbi(obsloglik, np.log(utteranceHMM['startprob']), np.log(utteranceHMM['transmat']), True)
    targets = [stateList.index(k) for k in list(np.array(stateTrans)[viterbi_path,])]
    return lmfcc, mspecc, targets



def create_dynamic_features(data):
    """
    data: list of dictionaries with keys: 'lmfcc', 'mspec' and 'targets'
    
    output: lmfcc_features: [NxD_lmfcc] where N is nr of all concatinated samples of all words in data
            mspec_features: [NxD_mspec] where N is nr of all concatinated samples of all words in data
            targets: [N,] index of state for each sample
    """
    
    D_lmfcc = data[0]['lmfcc'].shape[1]
    D_mspec = data[0]['mspec'].shape[1]
    # Features to be returned
    N = sum([len(x['targets']) for x in data])
    print(N)
    lmfcc_x = np.zeros((N,D_lmfcc*7))
    mspec_x = np.zeros((N,D_mspec*7))
    
    # Targets to be returned
    targets = []
    # through all data
    k = 0
    for x in tqdm(data): 
        times, dim = x['lmfcc'].shape

        ## for each timestep
        for i in range(times):
            if i<3 or i>=times-3:
                lmfcc_x[k,:]=np.hstack(np.pad(x['lmfcc'], pad_width=((3, 3), (0, 0)), mode='reflect')[i:i+7,:])
                mspec_x[k,:]=np.hstack(np.pad(x['mspec'], pad_width=((3, 3), (0, 0)), mode='reflect')[i:i+7,:])
            else:
                lmfcc_x[k,:]=np.hstack(x['lmfcc'][i-3:i+4,:])
                mspec_x[k,:]=np.hstack(x['mspec'][i-3:i+4,:])
            k +=1
        # lmfcc_x = np.vstack((lmfcc_x, lmfcc_features))
        #mspec_x = np.vstack((mspec_x, mspec_features))
        targets = targets + x['targets']
    return lmfcc_x, mspec_x, targets




def create_non_dynamic_features(data):
    """
    data: list of dictionaries with keys: 'lmfcc', 'mspec' and 'targets'
    
    output: lmfcc_features: [NxD_lmfcc] where N is nr of all concatinated samples of all words in data
            mspec_features: [NxD_mspec] where N is nr of all concatinated samples of all words in data
            targets: [N,] index of state for each sample
    """
    
    D_lmfcc = data[0]['lmfcc'].shape[1]
    D_mspec = data[0]['mspec'].shape[1]
    # Features to be returned
    N = sum([len(x['targets']) for x in data])
    print(N)
    lmfcc_x = np.zeros((N,D_lmfcc))
    mspec_x = np.zeros((N,D_mspec))
    
    # Targets to be returned
    targets = []
    # through all data
    k = 0
    for x in tqdm(data): 
        times, dim = x['lmfcc'].shape
        ## for each timestep
        for i in range(times):
            lmfcc_x[k,:]=x['lmfcc'][i,:]
            mspec_x[k,:]=x['mspec'][i,:]
            k +=1
        # lmfcc_x = np.vstack((lmfcc_x, lmfcc_features))
        #mspec_x = np.vstack((mspec_x, mspec_features))
        targets = targets + x['targets']
    return lmfcc_x, mspec_x, targets





