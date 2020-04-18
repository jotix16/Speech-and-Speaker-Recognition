import numpy as np
from lab2_tools import *


def concatTwoHMMs(hmm1, hmm2):
    """ Concatenates 2 HMM models

    Args:
       hmm1, hmm2: two dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be different for each)

    Output
       dictionary with the same keys as the input but concatenated models:
          startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models
   
    Example:
       twoHMMs = concatHMMs(phoneHMMs['sil'], phoneHMMs['ow'])

    See also: the concatenating_hmms.pdf document in the lab package
    """
    twoHMMs = dict()

    twoHMMs['name'] = hmm1['name'] + hmm2['name']
    twoHMMs['startprob'] = np.hstack((hmm1['startprob'][0:-1], hmm1['startprob'][-1]*hmm2['startprob']))
    
    # concatinate transmat
    upper = np.hstack((hmm1['transmat'][0:-1,0:-1], hmm1['transmat'][0:-1,-1][np.newaxis].T.dot(hmm2['startprob'][np.newaxis])))
    #print(np.zeros((hmm2['transmat'].shape[0],hmm1['transmat'].shape[1]-1)))
    lower = np.hstack((np.zeros((hmm2['transmat'].shape[0],hmm1['transmat'].shape[1]-1)), hmm2['transmat']))
    
    twoHMMs['transmat'] = np.vstack((upper, lower))
    twoHMMs['means'] = np.vstack((hmm1['means'],hmm2['means']))
    twoHMMs['covars'] = np.vstack((hmm1['covars'],hmm2['covars']))
    
    return twoHMMs

# this is already implemented, but based on concat2HMMs() above
def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: dictionary of models indexed by model name. 
       hmmmodels[name] is a dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models:
         startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    concat = hmmmodels[namelist[0]]
    for idx in range(1,len(namelist)):
        concat = concatTwoHMMs(concat, hmmmodels[namelist[idx]])
    return concat


def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """

def forward(log_emlik, log_startprob, log_transmat, obs_log_ll = False ):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states  log P(Oi|Xi)
        log_startprob: log probability to start in state i                    log P(Xi)
        log_transmat: log transition probability from state i to j            log P(Xj|Xi)
        obs_log_ll: True if we want the function to return both the alpha matrix and the log likelihood of seq.
    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """
    
    log_startprob = log_startprob[:-1]
    log_transmat = log_transmat[:-1,:-1]
    
    N, M = log_emlik.shape 
    forward_prob = np.zeros((N, M))                    # [N X M]      P(O1:)

    forward_prob[0,:] = log_startprob + log_emlik[0, :]

    for n in range(1,N):
        for i in range(M):
            forward_prob[n,i] = logsumexp(forward_prob[n-1,:]+log_transmat[:,i] ) + log_emlik[n,i]
    
    if obs_log_ll:
        return forward_prob,  logsumexp(forward_prob[-1,:])  
    
    return forward_prob

def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """
    log_startprob = log_startprob[:-1]
    log_transmat = log_transmat[:-1,:-1]

    N, M = log_emlik.shape
    backward_prob = np.zeros((N,M))
    #backward_prob[-1,:] = 1
    for n in range(N-2,-1,-1):
        for i in range(M):
            backward_prob[n,i] = logsumexp(backward_prob[n+1,:] + log_transmat[i,:] + log_emlik[n+1,:])
    return backward_prob

def viterbi(log_emlik, log_startprob, log_transmat, forceFinalState=True):
    """Viterbi path(decoding).


    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j
        forceFinalState: if True, start backtracking from the final state in
                  the model, instead of the best state at the last time step

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """
    log_startprob = log_startprob[:-1]
    log_transmat = log_transmat[:-1,:-1]

    N, M = log_emlik.shape
    V = np.ones((N,M)) # Viterbi loglikelihoods
    V_path = np.zeros((N,M),dtype=np.int64) # Viterbi indexes
    
    viterbi_path = np.zeros(N,dtype=np.int64)
    
    
    V[0,:] = log_startprob + log_emlik[0,:]
    for n in range(1,N):
        for i in range(M):
            # highest log likelihood to come to state i in time n
            V[n,i] = np.max(V[n-1,:]+log_transmat[:,i] + log_emlik[n,i])
            # best previous state in time n-1 that brought us to state i in time n
            V_path[n,i] = np.argmax(V[n-1,:]+log_transmat[:,i] + log_emlik[n,i])
    
    # path backtracking
    if forceFinalState:
        viterbi_path[-1] = M-1
    else:
        viterbi_path[-1] = np.argmax(V[-1,:])
	
    for i in range(V_path.shape[0]-2,0,-1):
        viterbi_path[i] = V_path[i+1,viterbi_path[i+1]]

    
    return np.max(V[-1,:]), viterbi_path

def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """
    return log_alpha + log_beta - logsumexp(log_alpha[-1,:])

def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
    """ Update Gaussian parameters with diagonal covariance

    Args:
         X: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
         varianceFloor: minimum allowed variance scalar
    were N is the lenght of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state
    """
    D = X.shape[1]
    N, M = log_gamma.shape
    means = np.zeros((M, D))
    covars = np.zeros((M, D))
    
    gamma = np.exp(log_gamma)

    means = gamma.T[:,:,np.newaxis]*X[np.newaxis,:,:] # NxMxD
    means = means .sum(1)
    means = means/ gamma.sum(0).T[:,np.newaxis]
    
    X_mu = X[:,np.newaxis,:]-means[np.newaxis,:,:]# NxMxD
    X_mu_2 = X_mu *X_mu
    covars = (X_mu_2*gamma[:,:,np.newaxis]).sum(0)/ gamma.sum(0).T[:,np.newaxis]
    covars[covars < varianceFloor] = varianceFloor
    return means,covars
