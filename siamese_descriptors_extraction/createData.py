import numpy as np
import os
import re
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.preprocessing import normalize
def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))


def gen_train_data(samp_f, total_to_samp):

    # first run on 1 data to find array shape
    im1 = read_pgm(os.getcwd() + '/orl_faces/s' + str(1) + '/' + str(1) + '.pgm', 'rw+')
    im1 = im1[::samp_f, ::samp_f]
    sz_1 = im1.shape[0]
    sz_2 = im1.shape[1]

    count = 0
    x_tr_m = np.zeros([total_to_samp, sz_2*sz_1])
    y_tr_m = np.zeros([total_to_samp, 1])
    for i in range(40):
        for j in range(int(total_to_samp/(2*40))):
            ind1 = np.random.randint(10)
            ind2 = np.random.randint(10)

            im1 = read_pgm(os.getcwd() + '/orl_faces/s' + str(i+1) + '/' + str(ind1 + 1) + '.pgm', 'rw+')
            im2 = read_pgm(os.getcwd() + '/orl_faces/s' + str(i+1) + '/' + str(ind2 + 1) + '.pgm', 'rw+')

            im1 = im1[::samp_f, ::samp_f]
            im2 = im2[::samp_f, ::samp_f]

            x_tr_m[count, :] = im1.reshape(im1.shape[0]*im1.shape[1])
            y_tr_m[count] = 1
            count += 1
            x_tr_m[count, :] = im2.reshape(im2.shape[0] * im2.shape[1])
            y_tr_m[count] = 1
            count += 1

            # plt.figure(1)
            # plt.imshow(im1, cmap='Greys_r')
            # plt.figure(2)
            # plt.imshow(im2, cmap='Greys_r')
            # plt.show()

    count = 0
    x_tr_non = np.zeros([total_to_samp, sz_2*sz_1])
    y_tr_non = np.zeros([total_to_samp, 1])
    for i in range(int(total_to_samp/20)):
        for j in range(10):
            while True:
                ind1 = np.random.randint(40)
                ind2 = np.random.randint(40)
                if ind1 != ind2:
                    break

            im1 = read_pgm(os.getcwd() + '/orl_faces/s' + str(ind1+1) + '/' + str(j + 1) + '.pgm', 'rw+')
            im2 = read_pgm(os.getcwd() + '/orl_faces/s' + str(ind2+1) + '/' + str(j + 1) + '.pgm', 'rw+')

            im1 = im1[::samp_f, ::samp_f]
            im2 = im2[::samp_f, ::samp_f]

            x_tr_non[count, :] = im1.reshape(im1.shape[0]*im1.shape[1])
            y_tr_non[count] = 0
            count += 1
            x_tr_non[count, :] = im2.reshape(im2.shape[0] * im2.shape[1])
            y_tr_non[count] = 0
            count += 1

            # plt.figure(1)
            # plt.imshow(im1, cmap='Greys_r')
            # plt.figure(2)
            # plt.imshow(im2, cmap='Greys_r')
            # plt.show()

    x_train = np.concatenate([x_tr_m, x_tr_non], axis=0)/255
    y_train = np.concatenate([y_tr_m, y_tr_non], axis=0)

    return x_train, y_train


# this returns in unvectorized form, amenable for conv2d layers
def gen_train_data_for_conv(samp_f, total_to_samp):
    # first run on 1 data to find array shape
    im1 = read_pgm(os.getcwd() + '/orl_faces/s' + str(1) + '/' + str(1) + '.pgm', 'rw+')
    im1 = im1[::samp_f, ::samp_f]
    sz_1 = im1.shape[0]
    sz_2 = im1.shape[1]

    count = 0
    x_tr_m = np.zeros([total_to_samp, sz_1, sz_2])
    y_tr_m = np.zeros([total_to_samp, 1])
    for i in range(40):
        for j in range(int(total_to_samp/(2*40))):
            ind1 = np.random.randint(10)
            ind2 = np.random.randint(10)

            im1 = read_pgm(os.getcwd() + '/orl_faces/s' + str(i+1) + '/' + str(ind1 + 1) + '.pgm', 'rw+')
            im2 = read_pgm(os.getcwd() + '/orl_faces/s' + str(i+1) + '/' + str(ind2 + 1) + '.pgm', 'rw+')

            im1 = im1[::samp_f, ::samp_f]
            im2 = im2[::samp_f, ::samp_f]

            x_tr_m[count, :, :] = im1
            y_tr_m[count] = 1
            count += 1
            x_tr_m[count, :, :] = im2
            y_tr_m[count] = 1
            count += 1

            # plt.figure(1)
            # plt.imshow(im1, cmap='Greys_r')
            # plt.figure(2)
            # plt.imshow(im2, cmap='Greys_r')
            # plt.show()


    count = 0
    x_tr_non = np.zeros([total_to_samp, sz_1, sz_2])
    y_tr_non = np.zeros([total_to_samp, 1])
    for i in range(int(total_to_samp/20)):
        for j in range(10):
            while True:
                ind1 = np.random.randint(40)
                ind2 = np.random.randint(40)
                if ind1 != ind2:
                    break

            im1 = read_pgm(os.getcwd() + '/orl_faces/s' + str(ind1+1) + '/' + str(j + 1) + '.pgm', 'rw+')
            im2 = read_pgm(os.getcwd() + '/orl_faces/s' + str(ind2+1) + '/' + str(j + 1) + '.pgm', 'rw+')

            im1 = im1[::samp_f, ::samp_f]
            im2 = im2[::samp_f, ::samp_f]

            x_tr_non[count, :] = im1
            y_tr_non[count] = 0
            count += 1
            x_tr_non[count, :] = im2
            y_tr_non[count] = 0
            count += 1

            # plt.figure(1)
            # plt.imshow(im1, cmap='Greys_r')
            # plt.figure(2)
            # plt.imshow(im2, cmap='Greys_r')
            # plt.show()

    x_train = np.concatenate([x_tr_m, x_tr_non], axis=0)
    y_train = np.concatenate([y_tr_m, y_tr_non], axis=0)

    x_train = x_train.reshape([x_train.shape[0], 1, x_train.shape[1], x_train.shape[2]])/255

    return x_train, y_train


# this returns x_train and y_train for classification - y is a factor variable
def gen_data_for_classification(samp_f):
    # first run on 1 data to find array shape
    im1 = read_pgm(os.getcwd() + '/orl_faces/s' + str(1) + '/' + str(1) + '.pgm', 'rw+')
    im1 = im1[::samp_f, ::samp_f]
    sz_1 = im1.shape[0]
    sz_2 = im1.shape[1]

    count = 0
    x = np.zeros([400, sz_1, sz_2])
    y = np.zeros([400, 10])
    for i in range(40):
        for j in range(10):
            im1 = read_pgm(os.getcwd() + '/orl_faces/s' + str(i+1) + '/' + str(j+1) + '.pgm', 'rw+')
            im1 = im1[::samp_f, ::samp_f]

            x[count, :, :] = im1/255
            y[count, j] = 1
            count += 1


            # plt.figure(1)
            # plt.imshow(im1, cmap='Greys_r')
            # plt.figure(2)
            # plt.imshow(im2, cmap='Greys_r')
            # plt.show()

    x = x.reshape([x.shape[0], 1, x.shape[1], x.shape[2]])
    return x, y


def gen_data_new(samp_f, total_to_samp):
    # first run on 1 data to find array shape
    im1 = read_pgm(os.getcwd() + '/orl_faces/s' + str(1) + '/' + str(1) + '.pgm', 'rw+')
    print(im1.shape)
    print(im1)
    im1 = im1[::samp_f, ::samp_f]
    print(im1.shape)
    sz_1 = im1.shape[0]
    sz_2 = im1.shape[1]

    count = 0
    x_tr_m = np.zeros([total_to_samp, 2, sz_2*sz_1])
    y_tr_m = np.zeros([total_to_samp, 1])
    for i in range(40):
        for j in range(int(total_to_samp/40)):
            # let's make the pairs different, same one is adding no value
            ind1 = 0
            ind2 = 0
            while ind1 == ind2:
                ind1 = np.random.randint(10)
                ind2 = np.random.randint(10)

            im1 = read_pgm(os.getcwd() + '/orl_faces/s' + str(i+1) + '/' + str(ind1 + 1) + '.pgm', 'rw+')
            im2 = read_pgm(os.getcwd() + '/orl_faces/s' + str(i+1) + '/' + str(ind2 + 1) + '.pgm', 'rw+')

            im1 = im1[::samp_f, ::samp_f]
            im2 = im2[::samp_f, ::samp_f]

            x_tr_m[count, 0, :] = im1.reshape(im1.shape[0]*im1.shape[1])
            x_tr_m[count, 1, :] = im2.reshape(im1.shape[0]*im1.shape[1])
            y_tr_m[count] = 1
            count += 1

            # plt.figure(1)
            # plt.imshow(im1, cmap='Greys_r')
            # plt.figure(2)
            # plt.imshow(im2, cmap='Greys_r')
            # plt.show()

    count = 0
    x_tr_non = np.zeros([total_to_samp, 2, sz_2*sz_1])
    y_tr_non = np.zeros([total_to_samp, 1])
    for i in range(int(total_to_samp/10)):
        for j in range(10):
            while True:
                ind1 = np.random.randint(40)
                ind2 = np.random.randint(40)
                if ind1 != ind2:
                    break

            im1 = read_pgm(os.getcwd() + '/orl_faces/s' + str(ind1+1) + '/' + str(j + 1) + '.pgm', 'rw+')
            im2 = read_pgm(os.getcwd() + '/orl_faces/s' + str(ind2+1) + '/' + str(j + 1) + '.pgm', 'rw+')

            im1 = im1[::samp_f, ::samp_f]
            im2 = im2[::samp_f, ::samp_f]

            x_tr_non[count, 0, :] = im1.reshape(im1.shape[0]*im1.shape[1])
            x_tr_non[count, 1, :] = im2.reshape(im1.shape[0]*im1.shape[1])
            y_tr_non[count] = 0
            count += 1

            # plt.figure(1)
            # plt.imshow(im1, cmap='Greys_r')
            # plt.figure(2)
            # plt.imshow(im2, cmap='Greys_r')
            # plt.show()

    x_train = np.concatenate([x_tr_m, x_tr_non], axis=0)/255
    y_train = np.concatenate([y_tr_m, y_tr_non], axis=0)

    return x_train.astype('float32'), y_train.astype('float32')


# this returns in unvectorized form, amenable for conv2d layers
def gen_train_data_for_conv_new(samp_f, total_to_samp):
    # first run on 1 data to find array shape
    im1 = read_pgm(os.getcwd() + '/orl_faces/s' + str(1) + '/' + str(1) + '.pgm', 'rw+')
    im1 = im1[::samp_f, ::samp_f]
    sz_1 = im1.shape[0]
    sz_2 = im1.shape[1]

    count = 0
    x_tr_m = np.zeros([total_to_samp, 2, 1, sz_1, sz_2])  # 2 is for pairs
    y_tr_m = np.zeros([total_to_samp, 1])
    for i in range(40):
        for j in range(int(total_to_samp/40)):
            ind1 = 0
            ind2 = 0
            while ind1 == ind2:
                ind1 = np.random.randint(10)
                ind2 = np.random.randint(10)

            im1 = read_pgm(os.getcwd() + '/orl_faces/s' + str(i+1) + '/' + str(ind1 + 1) + '.pgm', 'rw+')
            im2 = read_pgm(os.getcwd() + '/orl_faces/s' + str(i+1) + '/' + str(ind2 + 1) + '.pgm', 'rw+')

            im1 = im1[::samp_f, ::samp_f]
            im2 = im2[::samp_f, ::samp_f]

            x_tr_m[count, 0, 0, :, :] = im1
            x_tr_m[count, 1, 0, :, :] = im2
            y_tr_m[count] = 1
            count += 1

    count = 0
    x_tr_non = np.zeros([total_to_samp, 2, 1, sz_1, sz_2])
    y_tr_non = np.zeros([total_to_samp, 1])
    for i in range(int(total_to_samp/10)):
        for j in range(10):
            while True:
                ind1 = np.random.randint(40)
                ind2 = np.random.randint(40)
                if ind1 != ind2:
                    break

            im1 = read_pgm(os.getcwd() + '/orl_faces/s' + str(ind1+1) + '/' + str(j + 1) + '.pgm', 'rw+')
            im2 = read_pgm(os.getcwd() + '/orl_faces/s' + str(ind2+1) + '/' + str(j + 1) + '.pgm', 'rw+')

            im1 = im1[::samp_f, ::samp_f]
            im2 = im2[::samp_f, ::samp_f]

            x_tr_non[count, 0, 0, :, :] = im1
            x_tr_non[count, 1, 0, :, :] = im2
            y_tr_non[count] = 0
            count += 1

    x_train = np.concatenate([x_tr_m, x_tr_non], axis=0)/255
    y_train = np.concatenate([y_tr_m, y_tr_non], axis=0)

    return x_train.astype('float32'), y_train.astype('float32')

def gen_data_3D_new(total_to_samp,sample_dim,path=None,data=None):

    #########################################################
    if path is not None:
        tmp = np.loadtxt(os.getcwd() + path,delimiter=',')
    if data is not None:
        tmp = data
    # print type(tmp)
    #tmp = tmp[:301,:]
    tmp =tmp[200:,:]
    # la=tmp[:,-1]
    # tmp = normalize(tmp[:,:-1], norm='l1',axis=1, copy=False)
    # tmp=np.column_stack((tmp,la))
    count = 0
    x_tr_m = np.zeros([total_to_samp, 2, sample_dim])
    y_tr_m = np.zeros([total_to_samp, 1])
    for i in range(30):
        for j in range(int(total_to_samp/30)):
            # let's make the pairs different, same one is adding no value
            ind1 = 0
            ind2 = 0
            while ind1 == ind2:
                ind1 = np.random.randint(10)
                ind2 = np.random.randint(10)

            im1 = tmp[10*i+ind1]
            # print "im1:",im1.shape
            im2 = tmp[10*i+ind2]

            #
            # im1 = im1[::samp_f, ::samp_f]
            # im2 = im2[::samp_f, ::samp_f]
            #
            if int(im1[-1])==int(im2[-1]):
                # print int(im1[-1])
                x_tr_m[count, 0, :] = im1[:-1]
                x_tr_m[count, 1, :] = im2[:-1]
                y_tr_m[count] = 1
                # print x_tr_m.shape
                count += 1

   #  #########################################################

    count = 0
    x_tr_non = np.zeros([total_to_samp, 2, sample_dim])
    y_tr_non = np.zeros([total_to_samp, 1])
    for i in range(int(total_to_samp/10)):
        for j in range(10):
            while True:
                ind1 = np.random.randint(30)
                ind2 = np.random.randint(30)
                if ind1 != ind2:
                    break

            im1 = tmp[10*ind1+j]
            im2 = tmp[10*ind2+j]

            x_tr_non[count, 0, :] = im1[:-1]
            x_tr_non[count, 1, :] = im2[:-1]
            y_tr_non[count] = 0
            count += 1

   #normation
    print("x_tr_m:",x_tr_m.shape)
    print("x_tr_non:",x_tr_non.shape)
    x_train = np.concatenate([x_tr_m, x_tr_non], axis=0)
    y_train = np.concatenate([y_tr_m, y_tr_non], axis=0)
    print(x_train)

    return x_train.astype('float32'), y_train.astype('float32')


def gen_data_3D_new(total_to_samp,sample_dim,path=None,data=None):

    #########################################################
    if path is not None:
        tmp = np.loadtxt(os.getcwd() + path,delimiter=',')
    if data is not None:
        tmp = data
    # print type(tmp)
    # tmp = tmp[:400,:]
    # tmp =tmp[400:,:]
    # la=tmp[:,-1]
    # tmp = normalize(tmp[:,:-1], norm='l1',axis=1, copy=False)
    # tmp=np.column_stack((tmp,la))
    count = 0
    x_tr_m = np.zeros([total_to_samp, 2, sample_dim])
    y_tr_m = np.zeros([total_to_samp, 1])
    class_size = 6
    class_in_size =20
    for i in range(class_size):
        for j in range(int(total_to_samp/class_size)):
            # let's make the pairs different, same one is adding no value
            ind1 = 0
            ind2 = 0
            while ind1 == ind2:
                ind1 = np.random.randint(class_in_size)
                ind2 = np.random.randint(class_in_size)

            im1 = tmp[class_in_size*i+ind1]
            # print "im1:",im1.shape
            im2 = tmp[class_in_size*i+ind2]

            #
            # im1 = im1[::samp_f, ::samp_f]
            # im2 = im2[::samp_f, ::samp_f]
            #
            if int(im1[-1])==int(im2[-1]):
                # print int(im1[-1])
                x_tr_m[count, 0, :] = im1[:-1]
                x_tr_m[count, 1, :] = im2[:-1]
                y_tr_m[count] = 1
                # print x_tr_m.shape
                count += 1

   #  #########################################################

    count = 0
    x_tr_non = np.zeros([total_to_samp, 2, sample_dim])
    y_tr_non = np.zeros([total_to_samp, 1])
    for i in range(int(total_to_samp/class_in_size)):
        for j in range(class_in_size):
            while True:
                ind1 = np.random.randint(class_size)
                ind2 = np.random.randint(class_size)
                if ind1 != ind2:
                    break

            im1 = tmp[class_in_size*ind1+j]
            im2 = tmp[class_in_size*ind2+j]

            x_tr_non[count, 0, :] = im1[:-1]
            x_tr_non[count, 1, :] = im2[:-1]
            y_tr_non[count] = 0
            count += 1

   #normation
    print("x_tr_m:",x_tr_m.shape)
    print("x_tr_non:",x_tr_non.shape)
    x_train = np.concatenate([x_tr_m, x_tr_non], axis=0)
    y_train = np.concatenate([y_tr_m, y_tr_non], axis=0)
    print(x_train)

    return x_train.astype('float32'), y_train.astype('float32')


def gen_data_3D_new_test(total_to_samp,sample_dim,path):

    #########################################################
    tmp = np.loadtxt(os.getcwd() + path,delimiter=',')
    print(tmp.shape)
    # print type(tmp)

    count = 0
    x_tr_m = np.zeros([total_to_samp, 2, sample_dim])
    y_tr_m = np.zeros([total_to_samp, 1])
    test = tmp[:300]
    for i in range(30,40):
        for j in range(int(total_to_samp/10)):
            # let's make the pairs different, same one is adding no value
            ind1 = 0
            ind2 = 0
            while ind1 == ind2:
                ind1 = np.random.randint(10)
                ind2 = np.random.randint(10)

            im1 = tmp[10*i+ind1]
            # print "im1:",im1.shape
            im2 = tmp[10*i+ind2]

            #
            # im1 = im1[::samp_f, ::samp_f]
            # im2 = im2[::samp_f, ::samp_f]
            #
            if int(im1[-1])==int(im2[-1]):
                # print int(im1[-1])
                x_tr_m[count, 0, :] = im1[:-1]
                x_tr_m[count, 1, :] = im2[:-1]
                y_tr_m[count] = 1
                # print x_tr_m.shape
                count += 1

   #  #########################################################

    count = 0
    x_tr_non = np.zeros([total_to_samp, 2, sample_dim])
    y_tr_non = np.zeros([total_to_samp, 1])
    for i in range(int(total_to_samp/10)):
        for j in range(10):
            while True:
                ind1 = np.random.randint(30,40)
                ind2 = np.random.randint(30,40)
                if ind1 != ind2:
                    break

            im1 = tmp[10*ind1+j]
            im2 = tmp[10*ind2+j]

            x_tr_non[count, 0, :] = im1[:-1]
            x_tr_non[count, 1, :] = im2[:-1]
            y_tr_non[count] = 0
            count += 1

   #normation
    print(x_tr_m.shape)
    print(x_tr_non.shape)
    x_train = np.concatenate([x_tr_m, x_tr_non], axis=0)
    y_train = np.concatenate([y_tr_m, y_tr_non], axis=0)
    print(x_train.shape)

    return x_train.astype('float32'), y_train.astype('float32')
def gen_data_3D_new_test2(total_to_samp,sample_dim):

    #########################################################
    tmp = np.loadtxt(os.getcwd() + "/data/tmp_3D.csv",delimiter=',')
    test = tmp[300:]
    print(test.shape)
    return test
    # return x_train.astype('float32'), y_train.astype('float32')
if __name__=="__main__":
    gen_data_3D_new(1000,150)