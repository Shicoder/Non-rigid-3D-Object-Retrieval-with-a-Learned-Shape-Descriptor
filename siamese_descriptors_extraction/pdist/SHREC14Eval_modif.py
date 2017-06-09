__author__ = 'shi'
import numpy as np


def SHREC14Eval(rankings, C, task):
    # The original version of SHREC14Eval is write by matlab
    # it's used for evaluated the result of a method submitted to thr SHREC 14 Non-Rigid
    #Human Models track
    # in order to adapt to my own code ,I just transform matlab code to python code.
    # all variables is same as original implementation,
    def __init__(self,rankings,C,task):
        self.rankings = rankings
        self.C = C
        self.task = task
    #################Function##############################
    def precision(rankings,C,n):
        nModels = rankings.shape[0]
        count = 0
        for i in range(nModels):
            sum = np.sum(C[rankings[i,:n].astype('int')] == C[i])
            count = count + (float(sum) / n)
        P = count / nModels
        return P

    def  recall(rankings,C,n):
        nModels = rankings.shape[0]
        count = 0
        for i in range(nModels):
            nClass = np.sum(C==C[i])-1
            idx = np.sum(C[rankings[i,:n].astype('int')] == C[i])
            count = count + (float(idx) / nClass)
        R = count / nModels
        return R

    def precisionSingle(rankings,C,n,m):
        idx = np.sum(C[rankings[m,:n+1].astype('int')] == C[m])
        P = (float(idx)) / (n+1)
        return P

    def recallSingle(rankings,C,n,m):
        nClass = np.sum(C==C[m])-1
        idx = np.sum(C[rankings[m,:n+1].astype('int')] == C[m])
        R = (float(idx) / nClass)
        return R

    def nearestNeighbour (rankings ,C):

        nModels = rankings.shape[0]
        C.tolist()
        count = 0
        for i in range(nModels):
            if (int(C[i]) == int(C[int(rankings[i,0])])):
                count = count + 1
        NN = float(count) / nModels
        return NN
    def firstTier(rankings,C):
        nModels = rankings.shape[0]
        count = 0
        for i in range(nModels):
            nClass = np.sum(C==C[i])-1
            sum = np.sum(C[rankings[i,:nClass].astype('int')] == C[i])
            count = count + (float(sum) / nClass)
        FT = count / nModels
        return FT
    def secondTier(rankings,C):
        nModels = rankings.shape[0]
        count = 0
        for i in range(nModels):
            nClass = np.sum(C==C[i])-1
            sum = np.sum(C[rankings[i,:nClass*2].astype('int')] == C[i])
            count = count + (float(sum)/ nClass)
        ST = count / nModels
        return ST
    def  eMeasure(rankings,C):
        P = precision(rankings,C,32)
        R = recall(rankings,C,32)
        EM = 2 / ((1/P) + (1/R))
        return  EM

    def  DCG(rankings,C):
        nModels = rankings.shape[0]
        count = 0
        for i in range(nModels):
            nClass = np.sum(C==C[i])-1
            ideal = 1+np.sum(1./np.log2(np.arange(2,nClass+1,1)))
            idx = np.where(C[rankings[i,:].astype('int')] == C[i])
            tmp = np.where(idx[0] == 0)
            if len(tmp[0])>0:
                idx = list(set(idx[0]).difference(set(tmp[0])))
                idx = np.array(idx)
                DCGi = 1 + np.sum(1./np.log2(idx+1))
            else:
                idx = np.array(idx[0])
                DCGi = np.sum(1./np.log2(idx+1))
            count = count + (DCGi / ideal)
        DCG = count / nModels
        return DCG
    def fMeasure(rankings,C):
        nModels = rankings.shape[0]
        count = 0
        for i in range(nModels):
            idx = np.where(rankings[i,:])
            if len(idx[0])==0:
                continue
            r = rankings[i,idx[0]]
            r= r[r!=i].astype('int')

            idx = np.where(C==C[i])
            nClass = len(idx[0])-1
            idx = np.where(C[r] == C[i])
            if len(idx[0])==0:
                continue
            P = float(len(idx[0])) / float(len(r))
            R = float(len(idx[0])) / float(nClass)
            count = count + (2 * ((P*R) / (P+R)))
        FM = count / float(nModels)
        return FM
    # Task 1
    if task == 1:

        #rankings = importdata(inRank);
        if rankings.min() == -1:
            rankings = rankings + 1
        if rankings.shape[0] == rankings.shape[1]:
            rankings = rankings[:,1:]
            print('Result matrix is square (result contains query).')

        nn = nearestNeighbour(rankings,C)
        ft = firstTier(rankings,C)
        st = secondTier(rankings,C)
        em = eMeasure(rankings,C)
        dcg = DCG(rankings,C)

        #fprintf(fid,'\t NN: %.4f\n', results.nn);
        #fprintf(fid,'\t 1-Tier: %.4f\n', results.ft);
        #fprintf(fid,'\t 2-Tier: %.4f\n', results.st);
        #fprintf(fid,'\t E-measure: %.4f\n', results.em);
        #fprintf(fid,'\t DCG: %.4f\n', results.dcg);

        #fidPR = fopen(outPR, 'w');
        Pi = np.zeros([rankings.shape[1],1])
        Ri = np.zeros([rankings.shape[1],1])
        P = []
        R = []
        for m in range(rankings.shape[0]):
            for i in  range(rankings.shape[1]):
                Pi[i] = precisionSingle(rankings,C,i,m)
                Ri[i] = recallSingle(rankings,C,i,m)
            ri_unique,ia = np.unique(Ri,return_index=True)
            ia=np.sort(ia)
            idx = np.where(Ri==0)
            ia = list(set(ia).difference(set(idx[0])))
            ia = np.array(ia)
            if m == 0:
                P = Pi[ia]
                R = Ri[ia]
            else:
                P = P + Pi[ia]
                R = R + Ri[ia]

        R = R/float(rankings.shape[0])
        P = P/float(rankings.shape[0])
        R=np.insert(R,0,0)
        P=np.insert(P,0,1)
        # for i = 1:numel(P)
        # fprintf(fidPR,'%f\t%f\n', R(i), P(i))
        # end
        # fclose(fidPR)
        return  nn,ft,st,em,dcg,R,P
    #Task 2
    if task == 2:
        #rankings = dlmread(inRank)
        if rankings.min() == 0:
            rankings = rankings

        fm = fMeasure(rankings,C)

        #fprintf(fid,'\t F-measure: %.4f\n', results.fm)
        #fclose(fid)


        return  fm


    #Functions



    #Uses code from Lian et al.'s SHREC'11 non-rigid track.
    # def readClassification(fname):
    #     fp = fopen(fname,'r')
    #
    #     #Check file header
    #     strTemp = fscanf(fp,'%s',1)
    #     if ~strcmp(strTemp,'PSB')
    #         display('The format of your classification file is incorrect!')
    #         return
    #     strTemp = fscanf(fp,'%s',1)
    #     if ~strcmp(strTemp,'1')
    #         display('The format of your classification file is incorrect!')
    #         return
    #
    #     numCategories = fscanf(fp,'%d',1)
    #     numTotalModels = fscanf(fp,'%d',1)
    #
    #     testCategoryList.numCategories = numCategories
    #     testCategoryList.numTotalModels = numTotalModels
    #
    #     currNumTotalModels = 0
    #
    #     C = []
    #
    #     for i=1:numCategories
    #         currNumCategories = i
    #         testCategoryList.categories(currNumCategories).name = fscanf(fp,'%s',1)
    #         fscanf(fp,'%d',1)
    #         numModels = fscanf(fp,'%d',1)
    #         testCategoryList.categories(currNumCategories).numModels = numModels
    #         for j=1:numModels
    #             currNumTotalModels = currNumTotalModels+1
    #             testCategoryList.modelsNo(currNumTotalModels) = fscanf(fp,'%d',1)+1
    #             testCategoryList.classNo(currNumTotalModels) = currNumCategories
    #             C(testCategoryList.modelsNo(currNumTotalModels)) = ...
    #                 testCategoryList.classNo(currNumTotalModels)
    #
    #     if (currNumTotalModels~=numTotalModels):
    #         display('The format of your classification file is incorrect!')
    #         return
    #     else:
    #         display('The format of your classification file is correct!')
    #     fclose(fp)
    #     return