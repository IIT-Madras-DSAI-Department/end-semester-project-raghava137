import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import math
import time
from scipy.stats import mode
from algorithms_ import KNNEnsemble
def read_data(path1, path2):
    dftrain = pd.read_csv(path1)
    dftest = pd.read_csv(path2)
    featurecols = list(dftrain.columns)
    targetcol = 'label'
    featurecols.remove('even')
    featurecols.remove(targetcol)
    print ('length of featurecolumns is', len(featurecols))
    Xtrain = np.array(dftrain[featurecols])
    ytrain = np.array(dftrain[targetcol])
    Xtest = np.array(dftest[featurecols])
    ytest = np.array(dftest[targetcol])
    return (Xtrain, ytrain, Xtest, ytest)
Xtrain, ytrain, Xval, yval = read_data('MNIST_train.csv', 'MNIST_validation.csv')
model=KNNEnsemble(val_ratio=0.4)
model.fit(Xtrain, ytrain)
ypred = model.predict(Xval)
ConfusionMatrixDisplay(confusion_matrix(yval, ypred)).plot()
accuracy = accuracy_score(yval, ypred)
precision = np.sum(yval == ypred) / len(yval)
recall = np.sum(yval == ypred) / len(yval)
f1_score = 2 * (precision * recall) / (precision + recall)
f1_score_per_class = f1_score
print('Accuracy:', accuracy)
print('F1 Score:', f1_score_per_class)
