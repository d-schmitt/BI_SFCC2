# -*- coding: utf-8 -*-

from pandas import DataFrame, Series
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics
from sklearn.metrics import confusion_matrix
import itertools

#Load and split the dataset
myData=pd.read_csv('BI_adjusted_2010_2014.csv', sep=',')
del myData['PdDistrict']
del myData['Dates']
del myData['Descript']
del myData['Address']
del myData['Minutes']
del myData['Seconds']
del myData['X']
del myData['Y']    
data_clean=myData.dropna() 
predictors=data_clean[['Hour','Year','Month','Day','x_grid', 'y_grid']]
targets=data_clean.Category
pred_train, pred_test, tar_train,tar_test=train_test_split(predictors, targets, test_size=.35,random_state=20)

def modell(pred_train, pred_test, tar_train,tar_test):
    #Build a model on training data  
    classifier= DecisionTreeClassifier(criterion='entropy', max_depth=10,min_samples_leaf=.0007)
    classifier= classifier.fit(pred_train,tar_train)
    return classifier

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalisierte confusion matrix")
    else:
        print('Confusion Matrix, ohne Normalisierung')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Wahrer Wert')
    plt.xlabel('Vorhersage')

def display(classifier):
    ##Display a decision tree
    from sklearn import tree
    from pydotplus import graph_from_dot_data
    myData=pd.read_csv('BI_adjusted_preprocessed.csv', sep=';')
    dot_data=tree.export_graphviz(classifier,out_file=None, feature_names=['Hour','Year','Month','Day','x_grid', 'y_grid'], class_names=myData.Category, filled=True, rounded=True,  
                         special_characters=True)
    graph=graph_from_dot_data(dot_data)
    graph.write_jpg('desTree.jpg')
    return 0

class_names=np.array([('Andere Delikte'),('B. Gegenstaende'),
                      ('Diebstahl'),('Drogen-/Waffendelikte'),
                      ('Einbruch/Raub'),('Kidnapping'),
                      ('Koerperverletzung'),('Sexualdelikte'),
                      ('Wirtschaftsdelikte')])

desTree=modell(pred_train, pred_test, tar_train,tar_test)

cnf_matrix = confusion_matrix(tar_test, desTree.predict(pred_test))
np.set_printoptions(precision=1)

plt.figure(figsize=(25,15))

plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalisierte Confusion Matrix')
print (sklearn.metrics.accuracy_score(tar_test,desTree.predict(pred_test)))
# vor dem einkommentieren Baumtiefe auf 6 ändern
#display(desTree)




