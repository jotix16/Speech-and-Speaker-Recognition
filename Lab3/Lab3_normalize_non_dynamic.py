import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from keras.utils import np_utils


_, stateList, _ = pickle.load( open('saved_files/phoneHMM_states.pkl', 'rb'))
output_dim = len(stateList)
stateList = None


print("LMFCC")
print("Loading training set")
with np.load('datas/lmfcc_train_x_non_dynamic.npz', allow_pickle=True) as data:
    lmfcc_x = data['lmfcc_x']
    targets = data['targets']

scaler = StandardScaler()
scaler.fit(lmfcc_x)
lmfcc_x = scaler.transform(lmfcc_x).astype('float32')
targets = np_utils.to_categorical(targets, output_dim)

print("Saving normalized training set")
np.savez('datas/lmfcc_train_x_non_dynamic_reg.npz', lmfcc_x=lmfcc_x, targets=targets)


# test
lmfcc_x = None
targets = None
print("\nLoading test set")
with np.load('datas/lmfcc_test_x_non_dynamic.npz', allow_pickle=True) as data:
   lmfcc_x = data['lmfcc_x']
   targets = data['targets']
lmfcc_x = scaler.transform(lmfcc_x).astype('float32')
targets = np_utils.to_categorical(targets, output_dim)

print("Saving normalized test set")
np.savez('datas/lmfcc_test_x_non_dynamic_reg.npz', lmfcc_x=lmfcc_x, targets=targets)




#valid
lmfcc_x = None
targets = None
print("\nLoading validation set")
with np.load('datas/lmfcc_val_x_non_dynamic.npz', allow_pickle=True) as data:
   lmfcc_x = data['lmfcc_x']
   targets = data['targets']
lmfcc_x = scaler.transform(lmfcc_x).astype('float32')
targets = np_utils.to_categorical(targets, output_dim)

print("Saving normalized validation set")
np.savez('datas/lmfcc_val_x_non_dynamic_reg.npz', lmfcc_x=lmfcc_x, targets=targets)
      
      
scaler = None     
#####=================================================================================#####      
      
#mspec
lmfcc_x = None
targets = None
print("\n\nMSPEC")
print("Loading training set")
with np.load('datas/mspec_train_x_non_dynamic.npz', allow_pickle=True) as data:
    mspec_x = data['mspec_x']
    targets = data['targets']
mspec_x = mspec_x.astype('float32')
targets = np_utils.to_categorical(targets, output_dim)

mean = mspec_x.mean(0)[np.newaxis,:]
std = mspec_x.std(0)[np.newaxis,:]

print("Saving normalized training set")
np.savez('datas/mspec_train_x_non_dynamic_reg.npz', mspec_x=(mspec_x-mean)/std, targets=targets)


# test
mspec_x = None
targets = None
print("\nLoading test set")
with np.load('datas/mspec_test_x_non_dynamic.npz', allow_pickle=True) as data:
   mspec_x = data['mspec_x']
   targets = data['targets']
mspec_x =mspec_x.astype('float32')
targets = np_utils.to_categorical(targets, output_dim)

print("Saving normalized test set")
np.savez('datas/mspec_test_x_non_dynamic_reg.npz', mspec_x=(mspec_x-mean)/std, targets=targets)


#valid
mspec_x = None
targets = None
print("\nLoading validation set")
with np.load('datas/mspec_val_x_non_dynamic.npz', allow_pickle=True) as data:
   mspec_x = data['mspec_x']
   targets = data['targets']
mspec_x = mspec_x.astype('float32')
targets = np_utils.to_categorical(targets, output_dim)

print("Saving normalized validation set")
np.savez('datas/mspec_val_x_non_dynamic_reg.npz', mspec_x=(mspec_x-mean)/std, targets=targets)