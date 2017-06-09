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
from define_trainingset import define_trainingset,define_trainingsetnotrain
import random

# from keras.regularizers import l2, activity_l2

from sklearn.preprocessing import normalize
from sklearn import preprocessing
# # normalize the data attributes
# normalized_X = preprocessing.normalize(X)
# # standardize the data attributes
# standardized_X = preprocessing.scale(X)
# print os.getcwd()
# get the data
import time
for train_num in range(10):
    start = time.clock()
    total_to_samp1 = 7200
    sample_dim=201
    path ="/data/sall200mean.csv"
    data = np.loadtxt(os.getcwd() + path,delimiter=',')
    # data =data[:400,:]
    ######## split datast test class in train class#########################
    # Lables = data[:,-1]
    # ratio =0.2
    # trainIdx = define_trainingset(Lables,ratio)
    # all_idx =[x for x in range(Lables.shape[0])]
    # # testIdx = trainIdx
    # Xtrain = data[trainIdx,:]
    # print(Xtrain.shape)
    # testIdx = list(set(all_idx).difference(set(trainIdx)))
    # Xtest = data[testIdx,:]
    # print(Xtest.shape)
    ##################################
    ######## split datast test class not in train class#########################
    Lables = data[:,-1]
    ratio =0.4
    trainIdx = define_trainingsetnotrain(Lables,ratio)
    all_idx =[x for x in range(Lables.shape[0])]
    # testIdx = trainIdx
    Xtrain = data[trainIdx,:]
    print('sa',Xtrain.shape)
    testIdx = list(set(all_idx).difference(set(trainIdx)))
    Xtest = data[testIdx,:]
    print(Xtest.shape)
    ##################################
    fea =  pd.DataFrame(Xtest)
    fea.to_csv(os.getcwd() +'/data/sfeatureallq'+str(train_num)+'.txt',header=False,index=False)
    # fea2 = pd.DataFrame(Xtrain)
    # fea2.to_csv(os.getcwd() +'/data/featuretr'+str(train_num)+'2'+'.txt',header=False,index=False)
    print ('finish')
    # compute(os.getcwd() +'/data/featureallq'+str(train_num)+'.txt',sample_dim)
    # if j==0:
    #     input_data = Xtrain
    #     tmp = Xtest
    # else:
    #     input_data = Xtest
    #     tmp = Xtrain
    x_train, y_train = createFaceData.gen_data_3D_new(total_to_samp1, sample_dim,data=Xtrain)
    index = [k for k in range(len(x_train))]
    random.shuffle(index)
    x_train = x_train[index]
    y_train = y_train[index]

    # x_train = max_min_normalization(x_train)
    # x_test, y_test = createFaceData.gen_data_3D_new(total_to_samp2, sample_dim,path=path)
    # test = createFaceData.gen_data_3D_new_test2(total_to_samp2, sample_dim,path=path)
    # print 'tt',test.shape
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25)
    print(x_train.shape)
    print(type(x_train))
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
    hidden_layer_sizes = [1000, 500, sample_dim]
    base_network = create_base_network(input_dim, hidden_layer_sizes)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model(input=[input_a, input_b], output=distance)

    # train
    #nb_epoch=100
    nb_epoch =600###########this test is good

    rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms)
    early_stopping =EarlyStopping(monitor='val_loss', patience=5)
    model.fit([x_train[:, 0], x_train[:, 1]], y_train, validation_split=.25,
        batch_size=100, verbose=2, nb_epoch=nb_epoch)


    # compute final accuracy on training and test sets
    pred_tr = model.predict([x_train[:, 0], x_train[:, 1]])
    # print pred_tr
    # pred_ts = model.predict([x_test[:, 0], x_test[:, 1]])


    ################################test
    import keras.backend as K
    # tmp = np.loadtxt(os.getcwd() + '/data/SynethicHksmean130.csv',delimiter=',')
    # tmp = tmp[:160,:]
    # tmp = Xtest
    #########keyi youhua ,zhijie baocun la,yong dao xiamian de wangluo zhong
    # la = tmp[:,-1]
    # tmp = normalize(tmp[:,:-1], norm='l1',axis=1, copy=False)
    # tmp = np.column_stack((tmp,la))
    # print tmp.shape
    tmp = Xtest
    # path ="/data/hkswks200mean.csv"
    # tmp = np.loadtxt(os.getcwd() + path,delimiter=',')
    # tmp = tmp[:400,:]
    intermediate_layer_model = Model(input=model.input[0],
                                 output=processed_a)
    # intermediate_layer_model = Model(input=model.input[0], output=model.layers[2].get_output_at[0])
    intermediate_output = intermediate_layer_model.predict([tmp[:,:-1]])
    fea = np.column_stack((intermediate_output,tmp[:,-1]))
    feature = pd.DataFrame(fea)
    print(feature.head())
    feature.to_csv(os.getcwd() +'/data/featureallh'+str(train_num)+'.txt',header=False,index=False)
    end = time.clock()
    print('Running time: %s Seconds'%(end-start))
    # compute(os.getcwd() +'/data/featurehksh'+str(train_num)+'.txt',sample_dim)
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
    # tpr, fpr, _ = roc_curve(y_test, pred_ts)
    # print(pred_ts)
    # roc_auc = auc(fpr, tpr)
    #
    # plt.figure(1)
    # plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    # # plt.hold(True)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # # plt.hold(False)
    # plt.savefig('roc_curve_face.png')
    # plt.show()
    # #
    thresh = .35
    tr_acc = accuracy_score(y_train, (pred_tr < thresh).astype('float32'))
    # te_acc = accuracy_score(y_test, (pred_ts < thresh).astype('float32'))
    print ('test',train_num)
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    # print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    # print('* Mean of error less than  thresh (match): %0.3f' % np.mean(pred_ts[pred_ts < thresh]))
    # print(pred_ts.shape)
    # print(pred_ts[pred_ts >thresh])
    # print('* Mean of error more than  thresh (no match): %0.3f' % np.mean(pred_ts[pred_ts >= thresh]))
    # print("* test case confusion matrix:")
    # print(confusion_matrix((pred_ts < thresh).astype('float32'), y_test))
    # plt.figure(2)
    # plt.plot(np.concatenate([pred_ts[y_test == 1], pred_ts[y_test == 0]]), 'bo')
    # # plt.hold(True)
    # plt.plot(np.ones(pred_ts.shape)*thresh, 'r')
    # # plt.hold(False)
    # plt.savefig('pair_errors_face.png')

# for train_num in range(10):
#     total_to_samp1 = 6000
#     total_to_samp2 = 1500
#     sample_dim=100
#     path ="/data/wks100mean.csv"
#     data = np.loadtxt(os.getcwd() + path,delimiter=',')
#     data =data[:400,:]
#     ######## split datast test class in train class#########################
#     # Lables = data[:,-1]
#     # ratio =0.2
#     # trainIdx = define_trainingset(Lables,ratio)
#     # all_idx =[x for x in range(Lables.shape[0])]
#     # # testIdx = trainIdx
#     # Xtrain = data[trainIdx,:]
#     # print(Xtrain.shape)
#     # testIdx = list(set(all_idx).difference(set(trainIdx)))
#     # Xtest = data[testIdx,:]
#     # print(Xtest.shape)
#     ##################################
#     ######## split datast test class not in train class#########################
#     Lables = data[:,-1]
#     ratio =0.5
#     trainIdx = define_trainingsetnotrain(Lables,ratio)
#     all_idx =[x for x in range(Lables.shape[0])]
#     # testIdx = trainIdx
#     Xtrain = data[trainIdx,:]
#     print('sa',Xtrain.shape)
#     testIdx = list(set(all_idx).difference(set(trainIdx)))
#     Xtest = data[testIdx,:]
#     print(Xtest.shape)
#     ##################################
#        fea =  pd.DataFrame(Xtest)
#     fea.to_csv(os.getcwd() +'/data/featurewksq'+str(train_num)+'.txt',header=False,index=False)
#        # compute(os.getcwd() +'/data/featureallq'+str(train_num)+'.txt',sample_dim)
#     x_train, y_train = createFaceData.gen_data_3D_new(total_to_samp1, sample_dim,data=Xtrain)
#     index = [j for j in range(len(x_train))]
#     random.shuffle(index)
#     x_train = x_train[index]
#     y_train = y_train[index]
#
#     # x_train = max_min_normalization(x_train)
#     x_test, y_test = createFaceData.gen_data_3D_new(total_to_samp2, sample_dim,path=path)
#     # test = createFaceData.gen_data_3D_new_test2(total_to_samp2, sample_dim,path=path)
#     # print 'tt',test.shape
#     # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25)
#     print(x_train.shape)
#     print(type(x_train))
#     # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.30)
#
#
#     # because we re-use the same instance `base_network`,
#     # the weights of the network
#     # will be shared across the two branches
#     input_dim = x_train.shape[2]
#     print("input_dim:\n",input_dim)
#     print("input_dim_type:\n",type(input_dim))
#     print("x_train_type:\n",type(x_train))
#     # print "x_train:",x_train
#     input_a = Input(shape=(input_dim,))
#     input_b = Input(shape=(input_dim,))
#     hidden_layer_sizes = [512, 256, sample_dim]
#     base_network = create_base_network(input_dim, hidden_layer_sizes)
#     processed_a = base_network(input_a)
#     processed_b = base_network(input_b)
#
#     distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
#
#     model = Model(input=[input_a, input_b], output=distance)
#
#     # train
#     #nb_epoch=100
#     nb_epoch =1000
#
#     rms = RMSprop()
#     model.compile(loss=contrastive_loss, optimizer=rms)
#     early_stopping =EarlyStopping(monitor='val_loss', patience=5)
#     model.fit([x_train[:, 0], x_train[:, 1]], y_train, validation_split=.25,
#           batch_size=100, verbose=2, nb_epoch=nb_epoch)
#
#
#     # compute final accuracy on training and test sets
#     pred_tr = model.predict([x_train[:, 0], x_train[:, 1]])
#     # print pred_tr
#     pred_ts = model.predict([x_test[:, 0], x_test[:, 1]])
#
#     thresh = .35
#     tr_acc = accuracy_score(y_train, (pred_tr < thresh).astype('float32'))
#     # te_acc = accuracy_score(y_test, (pred_ts < thresh).astype('float32'))
#     print ('test',train_num)
#     print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
#     ################################test
#     import keras.backend as K
#  tmp = np.loadtxt(os.getcwd() + '/data/SynethicHksmean130.csv',delimiter=',')
#     # tmp = tmp[:160,:]
#     tmp = Xtest
#     #########keyi youhua ,zhijie baocun la,yong dao xiamian de wangluo zhong
#     # la = tmp[:,-1]
#     # tmp = normalize(tmp[:,:-1], norm='l1',axis=1, copy=False)
#     # tmp = np.column_stack((tmp,la))
#     # print tmp.shape
#     # tmp = test
#     intermediate_layer_model = Model(input=model.input[0],
#                                  output=processed_a)
#     # intermediate_layer_model = Model(input=model.input[0], output=model.layers[2].get_output_at[0])
#     intermediate_output = intermediate_layer_model.predict([tmp[:,:-1]])
#     fea = np.column_stack((intermediate_output,tmp[:,-1]))
#     feature = pd.DataFrame(fea)
#     print(feature.head())
#     feature.to_csv(os.getcwd() +'/data/featurewksh'+str(train_num)+'.txt',header=False,index=False)
#
# for train_num in range(10):
#     total_to_samp1 = 6000
#     total_to_samp2 = 1500
#     sample_dim=201
#     path ="/data/hkswksk200mean.csv"
#     data = np.loadtxt(os.getcwd() + path,delimiter=',')
#     data =data[:400,:]
#     ######## split datast test class in train class#########################
#     # Lables = data[:,-1]
#     # ratio =0.2
#     # trainIdx = define_trainingset(Lables,ratio)
#     # all_idx =[x for x in range(Lables.shape[0])]
#     # # testIdx = trainIdx
#     # Xtrain = data[trainIdx,:]
#     # print(Xtrain.shape)
#     # testIdx = list(set(all_idx).difference(set(trainIdx)))
#     # Xtest = data[testIdx,:]
#     # print(Xtest.shape)
#     ##################################
#     ######## split datast test class not in train class#########################
#     Lables = data[:,-1]
#     ratio =0.5
#     trainIdx = define_trainingsetnotrain(Lables,ratio)
#     all_idx =[x for x in range(Lables.shape[0])]
#     # testIdx = trainIdx
#     Xtrain = data[trainIdx,:]
#     print('sa',Xtrain.shape)
#     testIdx = list(set(all_idx).difference(set(trainIdx)))
#     Xtest = data[testIdx,:]
#     print(Xtest.shape)
#     ##################################
#     fea =  pd.DataFrame(Xtest)
#     fea.to_csv(os.getcwd() +'/data/featureallq'+str(train_num)+'.txt',header=False,index=False)
#     # compute(os.getcwd() +'/data/featureallq'+str(train_num)+'.txt',sample_dim)
#     x_train, y_train = createFaceData.gen_data_3D_new(total_to_samp1, sample_dim,data=Xtrain)
#     index = [j for j in range(len(x_train))]
#     random.shuffle(index)
#     x_train = x_train[index]
#     y_train = y_train[index]
#
#     # x_train = max_min_normalization(x_train)
#     x_test, y_test = createFaceData.gen_data_3D_new(total_to_samp2, sample_dim,path=path)
#     # test = createFaceData.gen_data_3D_new_test2(total_to_samp2, sample_dim,path=path)
#     # print 'tt',test.shape
#     # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25)
#     print(x_train.shape)
#     print(type(x_train))
#     # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.30)
#
#
#     # because we re-use the same instance `base_network`,
#     # the weights of the network
#     # will be shared across the two branches
#     input_dim = x_train.shape[2]
#     print("input_dim:\n",input_dim)
#     print("input_dim_type:\n",type(input_dim))
#     print("x_train_type:\n",type(x_train))
#     # print "x_train:",x_train
#     input_a = Input(shape=(input_dim,))
#     input_b = Input(shape=(input_dim,))
#     hidden_layer_sizes = [512, 256, sample_dim]
#     base_network = create_base_network(input_dim, hidden_layer_sizes)
#     processed_a = base_network(input_a)
#     processed_b = base_network(input_b)
#
#     distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
#
#     model = Model(input=[input_a, input_b], output=distance)
#
#     # train
#     #nb_epoch=100
#     nb_epoch =1000
#
#     rms = RMSprop()
#     model.compile(loss=contrastive_loss, optimizer=rms)
#     early_stopping =EarlyStopping(monitor='val_loss', patience=5)
#     model.fit([x_train[:, 0], x_train[:, 1]], y_train, validation_split=.25,
#           batch_size=100, verbose=2, nb_epoch=nb_epoch)
#
#
#     # compute final accuracy on training and test sets
#     pred_tr = model.predict([x_train[:, 0], x_train[:, 1]])
#     # print pred_tr
#     pred_ts = model.predict([x_test[:, 0], x_test[:, 1]])
#     thresh = .35
#     tr_acc = accuracy_score(y_train, (pred_tr < thresh).astype('float32'))
#     # te_acc = accuracy_score(y_test, (pred_ts < thresh).astype('float32'))
#     print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
#
#     ################################test
#     import keras.backend as K
#     # tmp = np.loadtxt(os.getcwd() + '/data/SynethicHksmean130.csv',delimiter=',')
#     # tmp = tmp[:160,:]
#     tmp = Xtest
#     #########keyi youhua ,zhijie baocun la,yong dao xiamian de wangluo zhong
#     # la = tmp[:,-1]
#     # tmp = normalize(tmp[:,:-1], norm='l1',axis=1, copy=False)
#     # tmp = np.column_stack((tmp,la))
#     # print tmp.shape
#     # tmp = test
#     intermediate_layer_model = Model(input=model.input[0],
#                                  output=processed_a)
#     # intermediate_layer_model = Model(input=model.input[0], output=model.layers[2].get_output_at[0])
#     intermediate_output = intermediate_layer_model.predict([tmp[:,:-1]])
#     fea = np.column_stack((intermediate_output,tmp[:,-1]))
#     feature = pd.DataFrame(fea)
#     print(feature.head())
#     feature.to_csv(os.getcwd() +'/data/featureallh'+str(train_num)+'.txt',header=False,index=False)