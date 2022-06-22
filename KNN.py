import pandas as pd
import numpy as np
import random
import pylab as pl
from matplotlib.colors import ListedColormap

def splitData (data, percent): 
    #Разбивает данные на 2 выборки: обучающую и тестовую, percent - какой процент всех данных будет в обучающей выборке
    trainData = []
    testData  = []
    #i = 0
    #train, test = [], []
    for row in data.iloc:
        if random.random() < percent:
            trainData.append(row)
            #train.append(i)
        else:
            testData.append(row)
            #test.append(i)
        #i+=1
    '''print('train', train)
    print('test', test)'''
    return trainData, testData

def ErrorMatrix (WithLabels, WithoutLabels):
    TN, TP, FN, FP = 0, 0, 0, 0
    for i in range(len(WithLabels)):
        if WithoutLabels[i] == 0:
            if WithoutLabels[i] == WithLabels[i][4]:
                TN+=1
            else:
                FN+=1
        else:
            if WithoutLabels[i] == WithLabels[i][4]:
                TP+=1
            else:
                FP+=1
    return TN, TP, FN, FP
def Precision (TN, TP, FN, FP):
    return TP/(TP+FP)
def Recall(TN, TP, FN, FP):
    return TP/(TP+FN)
def  F_score(TN, TP, FN, FP):
    pre = Precision(TN, TP, FN, FP)
    rec = Recall(TN, TP, FN, FP)
    return 2*pre*rec/(pre+rec)

def classKNN (trainData , testData, k):
    result = []
    for test in testData:
        #Создаем массив из пар [Евклидовое расстояние между точкой тестовой выборки и точкой обучающей выборки, класс точки обуч.выборки]
        dist=[[np.linalg.norm(test[0:4]-trainData[i][0:4]), int(trainData[i][4])] for i in range(len(trainData))]
        #Матрица из пар [Кол-во соседей класса, класс]
        qual = [[0,0], [0,1]]
        for d in sorted(dist)[0:k]:
            qual[d[1]][0] +=1
        result.append(sorted(qual, reverse = True)[0][1])
    return result

data = pd.read_csv('data_banknote_authentication.csv', delimiter=',')
data = data.astype({"class": "Int64"})
Train, TestWithLables = splitData(data, 0.5)
TestWithLables = np.array(TestWithLables)
Train = np.array(Train)
TestWithoutLables = classKNN(Train, TestWithLables, 50)
TN, TP, FN, FP = ErrorMatrix(TestWithLables, TestWithoutLables)
print("TN : ",TN,"\nTP : ", TP,"\nFN : ",FN,"\nFP : ", FP)
print("Precision : ",Precision(TN, TP, FN, FP),"\nRecall : ", Recall(TN, TP, FN, FP),"\nF-score : ", F_score(TN, TP, FN, FP))