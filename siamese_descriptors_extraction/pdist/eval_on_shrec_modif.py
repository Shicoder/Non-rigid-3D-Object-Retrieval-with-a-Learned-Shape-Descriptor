__author__ = 'shi'
import numpy as np
from pdist.SHREC14Eval_modif import SHREC14Eval
import matplotlib as plb
import numpy as np
import matplotlib.pyplot as plt

def eval_on_shrec_modif(distance , label,plotPR):
    classSize = np.sum(label==8)
    print(classSize)
    # classSize=[]
    # [classSize.append(int(i)) for i in label if not i in classSize]
    # print classSize
    n = distance.shape[0]
    print(n ,classSize)
    simrankings = np.zeros([n,n-1])
    clarankings =np.zeros([n,classSize-1])

    sortedDistance = np.argsort(distance,axis=0)
    sortedDistance = sortedDistance.T

    for i in range(n):
        rankingsi = sortedDistance[i,:]
        rankingsi= rankingsi[rankingsi!=i]
        rankingsi = np.array(rankingsi)
        simrankings[i,:] = rankingsi
        clarankings[i,:] = rankingsi[:(classSize-1)]
    simtask = 1
    #nn,ft,st,em,dcg,fm,R,P
    nn,ft,st,em,dcg,R,P = SHREC14Eval(simrankings, label, simtask)
    print('nn:',nn,'\n','ft:',ft,'\n','st:',st,'\n','em:',em,'\n','dcg:',dcg,'\n','R:',R,'\n','P:',P,'\n')
    clatask = 2
    fm = SHREC14Eval(clarankings, label, clatask)
    print('fm:',fm)
    # shrecres = simres
    # shrecres.fm = clares.fm

    if plotPR:
        plt.figure(1)
        plt.title('Precision/Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        x=R
        y=P
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.plot(x,y)
        plt.show()
        plt.savefig('P-R.png')


