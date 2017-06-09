
import scipy.io as sp
import numpy as np
from matplotlib import pyplot as plt
# dataFile = './data/data.mat'
#
# data = sp.loadmat(dataFile)
from pdist.pdist import compute
import os
# print data['HksHistDesc'][:,:,70]
# ttt = np.array([[1,2,3,],[2,3,4]])
# mean = ttt.mean(axis=1).reshape([2,1])
# std = ttt.std(axis=1).reshape([2,1])
# print mean,'\n',std,'\n'
#
# new_ttt = (ttt-mean)/std
# print new_ttt
# final_data = np.zeros([51, 128, 500])
# for i in range(50):
#     new_data = data['HksHistDesc'][:,:,10*i]
#     for j in range(1,10):
#         new_data = np.hstack((new_data, data['HksHistDesc'][:,:,10*i+j]))
#     mean = new_data.mean(axis=1).reshape([51,1])
#     std = new_data.std(axis=1).reshape([51,1])
#     new_data = (new_data-mean)/(std)
#     for k in range(10):
#         final_data[:,:,i*10+k]=new_data[:,k*128:k*128+128]
# print final_data.shape


# compute(os.getcwd() +'/data/feature_hksl2mean5000.txt')








# print data['C']
# dict = {}
# for i in range(128):
#     tmp = np.zeros((51,500))
#     for j in range(500):
#         tmp[:,j] = data['HksHistDesc'][:,i,j]
#     dict[i] =tmp.T
# print dict[1].shape


import pandas as pd
for i in range(0,10):
     compute(os.getcwd() +'/data/sfeaturehksq'+str(i)+'.txt',201)
     compute(os.getcwd() +'/data/featurehksh'+str(i)+'.txt',201)
# path ="/data/sHKS100mean.csv"
# data = np.loadtxt(os.getcwd() + path,delimiter=',')
# print(data.shape)
# feature = np.loadtxt(os.getcwd() +'/data/featureallq0.csv',delimiter=',')
# print(feature.shape)
# print data[1,:131]
# plt.figure(1)
# plt.title('HKS/Time Curve')
# plt.xlabel('Time')
# plt.ylabel('HKS Vlaues')
# x = data[1,:131]
# plt.plot(x)
# plt.show()
# plt.savefig('HKS.png')

from sklearn.preprocessing import normalize
from sklearn import preprocessing
# # normalize the data attributes

# # standardize the data attributes
# standardized_X = preprocessing.scale(X)
# data = np.array([[1,1,1,1],[1,1,1,1]])
# normalized_X = preprocessing.normalize(data.T)
# print(normalized_X)