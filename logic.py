import pandas as pd
import numpy as np
import random
from scipy.optimize import fmin_tnc

def splitData (data, percent): 
    #Разбивает данные на 2 выборки: обучающую и тестовую, percent - какой процент всех данных будет в обучающей выборке
    trainData = []
    testData  = []
    for row in data.iloc:
        if random.random() < percent:
            trainData.append(row)
        else:
            testData.append(row)
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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def probability(theta, x):
    return sigmoid(np.dot(x, theta))
def cost_function(theta, x, y):
    n = x.shape[0]
    total_cost = -(1/n) * np.sum(y * np.log(probability(theta, x)) + (1 - y) * np.log(1 - probability(theta, x)))
    return total_cost
def gradient(theta, x, y):
    n = x.shape[0]
    return (1 / n) * np.dot(x.T, sigmoid(np.dot(x, theta)) - y)
def fit(theta, x, y):
    opt_costs = fmin_tnc(func=cost_function, x0=theta, fprime=gradient, args=(x, y))
    return opt_costs[0]

def classLogic (testData, theta):
    result = []
    for test in  testData:
        res = probability(test[:4], theta)
        if res >= 0.5:
            result.append(1)
        else:
            result.append(0)
    return result

data = pd.read_csv('data_banknote_authentication.csv', delimiter=',')
data = data.astype({"class": "Int64"})
Train, TestWithLables = splitData(data, 0.1)
TestWithLables = np.array(TestWithLables)
Train = np.array(Train)
theta = np.zeros(4)
theta = fit(theta,Train[:,:4].astype(float), Train[:,4].astype(float))
TestWithoutLables = classLogic(TestWithLables.astype(float), theta)
TN, TP, FN, FP = ErrorMatrix(TestWithLables, TestWithoutLables)
print("TN : ",TN,"\nTP : ", TP,"\nFN : ",FN,"\nFP : ", FP)
print("Precision : ",Precision(TN, TP, FN, FP),"\nRecall : ", Recall(TN, TP, FN, FP),"\nF-score : ", F_score(TN, TP, FN, FP))