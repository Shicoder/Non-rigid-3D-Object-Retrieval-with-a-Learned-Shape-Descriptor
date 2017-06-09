__author__ = 'shi'

import numpy as np
import scipy.spatial.distance as disatnce
from  pdist.eval_on_shrec_modif import eval_on_shrec_modif
import os
def compute(path,dim):

    feature = np.loadtxt(path,delimiter=',')
    label = feature[:,dim]
    feature =feature[:,0:dim]
    # print label,feature
    # print feature.shape
    dist = disatnce.pdist(feature,metric='euclidean')
    d= disatnce.squareform(dist)
    shreceval = eval_on_shrec_modif(d,label,1)

if  __name__ == '__main__' :

        compute('../data/feature_hksl2mean500.txt')