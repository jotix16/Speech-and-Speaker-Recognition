import numpy as np
from lab3_tools import *
from lab3_proto import *
from tqdm import tqdm
import os







def main():
   #  # Dynamic features for training
   #  X = np.load('datas/training_set.npz', allow_pickle=True)['X']
   #  lmfcc_x, mspec_x,targets = create_non_dynamic_features(X)
   #  print("lmfcc:",lmfcc_x.shape, "mspec_x:",mspec_x.shape, "targets:",len(targets),)
   #  np.savez('datas/lmfcc_train_x_non_dynamic.npz', lmfcc_x=lmfcc_x, targets=targets )
   #  np.savez('datas/mspec_train_x_non_dynamic.npz', mspec_x=mspec_x,  targets=targets )

   #  Dynamic features for testing
   #  X = np.load('datas/testdata.npz', allow_pickle=True)['testdata']
   #  lmfcc_x, mspec_x, targets = create_non_dynamic_features(X)
   #  np.savez('datas/lmfcc_test_x_non_dynamic.npz', lmfcc_x=lmfcc_x, targets=targets )
   #  np.savez('datas/mspec_test_x_non_dynamic.npz', mspec_x=mspec_x, targets=targets )

   #  Dynamic features for validation
   #  X = np.load('datas/validation_set.npz', allow_pickle=True)['X_val']
   #  lmfcc_x, mspec_x, targets = create_non_dynamic_features(X)
   #  np.savez('datas/lmfcc_val_x_non_dynamic.npz', lmfcc_x=lmfcc_x, targets=targets )
   #  np.savez('datas/mspec_val_x_non_dynamic.npz', mspec_x=mspec_x, targets=targets )


    #### Statistics for each set
    # #Training set
    #lmfcc
    with np.load('datas/lmfcc_train_x_non_dynamic.npz', allow_pickle=True) as data:
        lmfcc_x = data['lmfcc_x']
        targets = data['targets']
    print("Training set has",len(targets),"lmfcc_x samples", "with dimension ",lmfcc_x.shape[1],".")
    lmfcc_x = None
    targets = None
    #mspec
    with np.load('datas/mspec_train_x_non_dynamic.npz', allow_pickle=True) as data:
        lmfcc_x = data['mspec_x']
        targets = data['targets']
    print("Training set has",len(targets),"mspec_x samples", "with dimension ",lmfcc_x.shape[1],".")
    lmfcc_x = None
    targets = None

    #Test set
    #lmfcc
    with np.load('datas/lmfcc_test_x_non_dynamic.npz', allow_pickle=True) as data:
       lmfcc_x = data['lmfcc_x']
       targets = data['targets']
    print("Test set has",len(targets),"lmfcc_x samples", "with dimension ",lmfcc_x.shape[1],".")
    lmfcc_x = None
    targets = None
    #mspec
    with np.load('datas/mspec_test_x_non_dynamic.npz', allow_pickle=True) as data:
       lmfcc_x = data['mspec_x']
       targets = data['targets']
    print("Test set has",len(targets),"mspec_x samples", "with dimension ",lmfcc_x.shape[1],".")
    lmfcc_x = None
    targets = None


    #Validation set
    #lmfcc
    with np.load('datas/lmfcc_val_x_non_dynamic.npz', allow_pickle=True) as data:
       lmfcc_x = data['lmfcc_x']
       targets = data['targets']
    print("Validation set has",len(targets),"lmfcc_x samples", "with dimension ",lmfcc_x.shape[1],".")
    lmfcc_x = None
    targets = None
    #mspec
    with np.load('datas/mspec_val_x_non_dynamic.npz', allow_pickle=True) as data:
       lmfcc_x = data['mspec_x']
       targets = data['targets']
    print("Validation set has",len(targets),"mspec_x samples", "with dimension ",lmfcc_x.shape[1],".")

if __name__ == "__main__":
    main()
