# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
import itertools
import matplotlib.pyplot as plt


data = pd.read_csv('/Users/Duong/Desktop/BI_adjusted_2010_2014.csv', sep=',', na_values=".")

del data['Unnamed: 0']
del data['Dates']
del data['Descript']
del data['Address']
del data['Minutes']
del data['Seconds']
del data['X']
del data['Y']

#Rausnehmen der Ausprägung Secondary Code

data = data[data.Category != 'SECONDARY CODES']


#Werte von DayOfWeek und PdDisrict und Category (für Confusion Matrix) in nummerische Werte umwandeln
data.DayOfWeek=pd.Categorical(data.DayOfWeek)
data['CodeForDayOfWeek'] = data.DayOfWeek.cat.codes
DayofWeekLegend=data[['DayOfWeek','CodeForDayOfWeek']].copy()

data.PdDistrict=pd.Categorical(data.PdDistrict)
data['CodeForPdDisctrict'] = data.PdDistrict.cat.codes
PdDistrictLegend=data[['PdDistrict','CodeForPdDisctrict']].copy()

data.Category=pd.Categorical(data.Category)
data['CodeForCategory'] = data.Category.cat.codes
CategoryLegend=data[['Category','CodeForCategory']].copy()

 
#in Trainungs- und Testsdaten splitten
train, test = train_test_split(data, test_size = 0.35, random_state=20)
train_Crime=train['CodeForCategory'].copy()
test_Crime=test['CodeForCategory'].copy()

class_names=np.array([('Andere Delikte'),('B. Gegenstaende'),
                      ('Diebstahl'),('Drogen-/Waffendelikte'),
                      ('Einbruch/Raub'),('Kidnapping'),
                      ('Koerperverletzung'),('Sexualdelikte'),
                      ('Wirtschaftsdelikte')])


train_IV2_für_Auswahl=train[['CodeForDayOfWeek','Hour','Year','Month','x_grid','y_grid','CodeForPdDisctrict']].copy()
mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(train_IV2_für_Auswahl, train_Crime)


#Recursive Feature Elimination
rfe = RFE(mul_lr, n_features_to_select=1)
rfe.fit(train_IV2_für_Auswahl, train_Crime)

print ("Features sorted by their rank:")
print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), namesVariable)))

#Modellerstellung mit Forward Selection
train_IV=train[['Year','Month','Hour','x_grid','y_grid']].copy()
test_IV=test[['Year','Month','Hour','x_grid','y_grid']].copy()
mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(train_IV, train_Crime)
print("Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(train_Crime, mul_lr.predict(train_IV)))
print("Multinomial Logistic regression Test Accuracy :: ", metrics.accuracy_score(test_Crime, mul_lr.predict(test_IV)))

#Legende für Confusion Matrix

namesVariable= np.array([('CodeForDayOfWeek'),('Hour'),('Year'),('Month'),
                      ('x_grid'),('y_grid'),('CodeForPdDisctrict')])

##Confusion Matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
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

#Plot Confusion Matrix mit Normalisierung
cnf_matrix = confusion_matrix(test_Crime, mul_lr.predict(test_IV))
np.set_printoptions(precision=1)

plt.figure(figsize=(25,15))
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalisierte Confusion Matrix')

#PLot Confusion Matrix ohne Normalisierung
#plt.figure(figsize=(20,10))
#plot_confusion_matrix(cnf_matrix, classes=class_names,
#                      title='Confusion matrix, ohne Normalisierung')
plt.show()