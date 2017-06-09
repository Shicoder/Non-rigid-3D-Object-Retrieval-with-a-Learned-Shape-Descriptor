__author__ = 'shi'
import numpy as np
def define_trainingset(Lables,ratio):

    if ratio ==1:
        trainingidex = np.ones((Lables.shape))
        return trainingidex

    trainingidex = []
    uniqueLables = np.unique(Lables)
    print(uniqueLables.shape)
    for i in range(uniqueLables.shape[0]):
        iscurlable = (Lables == uniqueLables[i])
        numinstances = np.sum(iscurlable)
        a = np.where(iscurlable==1)
        sel = np.random.choice(np.array(a[0]),int(ratio*numinstances),replace=False)
        trainingidex.extend(sel)
    return trainingidex
def define_trainingsetnotrain(Lables,ratio):

    if ratio ==1:
        trainingidex = np.ones((Lables.shape))
        return trainingidex

    trainingidex = []
    uniqueLables = np.unique(Lables)
    print(uniqueLables.shape)
    # class_num = np.random.randint(uniqueLables.shape[0],size=20,)
    class_num = np.random.choice(uniqueLables,int(ratio*uniqueLables.shape[0]),replace=False)
    for i in class_num:
        tmp = int(i)-1
        idx = [x for x in range(tmp*20,(tmp+1)*20)]
        trainingidex.extend(idx)
    print("trainingidx",trainingidex)
    return trainingidex


