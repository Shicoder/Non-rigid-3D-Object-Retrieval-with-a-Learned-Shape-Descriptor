import numpy as np
from keras.optimizers import SGD, RMSprop
from keras.layers import Input, Lambda, Dense, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.cross_validation import train_test_split
import os
import createFaceData
import pandas as pd
from SiameseFunctions import create_base_network, eucl_dist_output_shape, euclidean_distance, contrastive_loss, compute_accuracy
from pdist.pdist import compute
import random
import scipy.io as sp
# from keras.regularizers import l2, activity_l2

from sklearn.preprocessing import normalize
from sklearn import preprocessing
# normalize the data attributes
# normalized_X = preprocessing.normalize(X)
# # standardize the data attributes
# standardized_X = preprocessing.scale(X)
# print os.getcwd()
# get the data


dataFile = './data/data.mat'

input_data = sp.loadmat(dataFile)

ori_data = np.zeros([51, 128, 500])
for i in range(50):
    new_data = input_data['HksHistDesc'][:,:,10*i]
    for j in range(1,10):
        new_data = np.hstack((new_data, input_data['HksHistDesc'][:,:,10*i+j]))
    mean = new_data.mean(axis=1).reshape([51,1])
    std = new_data.std(axis=1).reshape([51,1])
    new_data = (new_data-mean)/(std)
    for k in range(10):
        ori_data[:,:,i*10+k]=new_data[:,k*128:k*128+128]

##############test###################
# print ori_data['HksHistDesc'][:,:,1]
# for i in range(500):
#     np.savetxt('./data/data/'+str(i)+'.txt',ori_data['HksHistDesc'][:,:,i],delimiter=' ')




# print data['C']
for i in range(51):
    print ('iteration:',i)

    tmp = np.zeros((128,500))
    for j in range(500):
        tmp[:,j] = ori_data[i,:,j]
    tmp =tmp.T
    print('sample shape:\n',tmp.shape)
    total_to_samp1 = 4500
    total_to_samp2 = 1500
    sample_dim=128
    data = np.column_stack((tmp,input_data['C']))
    # path ="/data/hksArea130s2.csv"
    # x_train, y_train = createFaceData.gen_data_3D_new(total_to_samp1, sample_dim,path=path)
    x_train, y_train = createFaceData.gen_data_3D_new(total_to_samp1, sample_dim, data=data)
    index = [j for j in range(len(x_train))]
    random.shuffle(index)
    x_train = x_train[index]
    y_train = y_train[index]

    # x_train = max_min_normalization(x_train)
    # test = createFaceData.gen_data_3D_new_test2(total_to_samp2, sample_dim,path=path)
    # print 'tt',test.shape
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25)
    print("x_train shape:\n",x_train.shape)
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.30)


    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    input_dim = x_train.shape[2]
    print("input_dim:\n",input_dim)
    print("input_dim_type:\n",type(input_dim))
    print("x_train_type:\n",type(x_train))
    # print "x_train:",x_train
    input_a = Input(shape=(input_dim,))
    input_b = Input(shape=(input_dim,))
    output =128
    hidden_layer_sizes = [512,256,output]
    base_network = create_base_network(input_dim, hidden_layer_sizes)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model(input=[input_a, input_b], output=distance)

    # train
    #nb_epoch=100
    nb_epoch =1000

    rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms)
    early_stopping =EarlyStopping(monitor='val_loss', patience=5)
    model.fit([x_train[:, 0], x_train[:, 1]], y_train, validation_split=.25,
              batch_size=100, verbose=2, nb_epoch=nb_epoch)


    # compute final accuracy on training and test sets
    pred_tr = model.predict([x_train[:, 0], x_train[:, 1]])
    # print pred_tr


    ################################test
    import keras.backend as K
    data_te = data[:200,:]
    #########keyi youhua ,zhijie baocun la,yong dao xiamian de wangluo zhong
    # la = tmp[:,-1]
    # tmp = normalize(tmp[:,:-1], norm='l1',axis=1, copy=False)
    # tmp = np.column_stack((tmp,la))
    # print tmp.shape
    # tmp = test
    intermediate_layer_model = Model(input=model.input[0],
                                     output=processed_a)
    # intermediate_layer_model = Model(input=model.input[0], output=model.layers[2].get_output_at[0])
    intermediate_output = intermediate_layer_model.predict([data_te[:,:-1]])
    fea = np.column_stack((intermediate_output,data_te[:,-1]))
    feature = pd.DataFrame(fea)
    print(feature.head())
    feature.to_csv(os.getcwd() +'/data/feature_hksl2mean500'+str(i)+'.txt',header=False,index=False)
    #compute(os.getcwd() +'/data/feature_hksl2mean500'+str(i)+'.txt')
    # np.savetxt(os.getcwd() +'/data/feature.txt',intermediate_output)
    # print intermediate_output
    # get_feature = K.function([K.learning_phase(),model.layers[0].input],model.layers[2].output)
    # feature = get_feature([tmp[:,:-1]])
    # print 'qqq',intermediate_output
    # print 'www',intermediate_output.shape

    # model2 = Sequential()
    #
    # model2.add(Dense(200, input_shape=(x_train[:, 0],), activation='tanh', weights = model.layers[0].get_weights()))
    #
    # model2.compile(loss='categorical_crossentropy', optimizer='adam', class_mode = "categorical")
    # TT=model2.predict(x_test[:, 0], batch_size=22)
    # print TT
    #################################
    # auc and other things

features = np.zeros((200,51*output))
for i in range(51):
    feature = np.loadtxt(os.getcwd() +'/data/feature_hksl2mean500'+str(i)+'.txt',delimiter=',')
    feature =feature[:,0:output]
    print(feature.shape)
    for j in range(output):
        features[:,i*output+j] = feature[:,j]

print(features.shape)
features = np.column_stack((features,input_data['C'][0:200]))
features = pd.DataFrame(features)
print(features.head())
features.to_csv(os.getcwd() +'/data/features.txt',header=False,index=False)
compute(os.getcwd() +'/data/features.txt')