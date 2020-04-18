import numpy as np
from lab3_tools import *
from lab3_proto import *
from tqdm import tqdm
import os



import warnings
warnings.filterwarnings('ignore')




def main():
	# # Create trainings data
	# print("CREATE TRAIN DATA")
	# traindata = []
	# for root, dirs, files in tqdm(os.walk('../datasets/tidigits/disc_4.1.1/tidigits/train')):
	# 	for file in files:
	# 		if file.endswith('.wav'):
	# 			filename = os.path.join(root, file)
	# 			samples, samplingrate = loadAudio(filename)
	# 			#...your code for feature extraction and forced alignment
	# 			lmfcc, mspecc, targets =  extract_features_and_targets(filename)
	# 			traindata.append({'filename': filename, 'lmfcc': lmfcc, 'mspec': mspecc, 'targets': targets})
	# np.savez('datas/traindata.npz', traindata=traindata)

	# # Create test data
	# print("CREATE TEST DATA")
	# testdata = []
	# i = 0
	# for root, dirs, files in tqdm(os.walk('../datasets/tidigits/disc_4.2.1/tidigits/test')):
	# 	for file in files:
	# 		if file.endswith('.wav'):
	# 			filename = os.path.join(root, file)
	# 			samples, samplingrate = loadAudio(filename)
	# 		#...your code for feature extraction and forced alignment
	# 			lmfcc, mspecc, targets =  extract_features_and_targets(filename)
	# 			testdata.append({'filename': filename, 'lmfcc': lmfcc, 'mspec': mspecc, 'targets': targets})
	# np.savez('datas/testdata.npz', testdata=testdata)	
	# print("Successfully saved")	

	## Get statistics
	traindata = np.load('datas/traindata.npz', allow_pickle=True)['traindata']
	testdata = np.load('datas/testdata.npz', allow_pickle=True)['testdata']	
	N = len(traindata)
	N_test = len(testdata)
	print("Nr of words in training set:",N)
	print("Nr of words in testing set:",N_test)	
	print("Nr of timesteps for train data:", sum([len(x['targets']) for x in traindata]))
	print("Nr of timesteps for test data:", sum([len(x['targets']) for x in testdata]))


if __name__ == "__main__":
    main()
