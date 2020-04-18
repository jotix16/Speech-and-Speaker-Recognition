import numpy as np

## uncoment for proper run
# traindata = np.load('datas/traindata.npz', allow_pickle=True)['traindata']
# N = len(traindata)
# n_valid=int(N*0.1)
# # shuffle
# indexes = np.random.permutation(N)
# X = np.take(traindata,indexes)
# X_val = X[:n_valid]
# X = X[n_valid:]
# np.savez('datas/training_set.npz', X=X)
# np.savez('datas/validation_set.npz', X_val=X_val)




## Load again and count man and women
X = np.load('datas/training_set.npz', allow_pickle=True)['X']
X_val = np.load('datas/validation_set.npz', allow_pickle=True)['X_val']
N = len(X)+len(X_val)
print(len(X)/N,len(X_val)/N)
# count men and women
## Trainingset
N_X_women = sum(1 for data in X if data['filename'].split("/")[-3] == 'woman')
N_X_man = len(X)-N_X_women
## Validationset
N_X_val_women = sum(1 for data in X_val if data['filename'].split("/")[-3] == 'woman')
N_X_val_man = len(X_val)-N_X_val_women


print("Trainign set: Women->",N_X_women,", Men->",N_X_man)
print("Validation set: Women->",N_X_val_women,", Men->",N_X_val_man)
