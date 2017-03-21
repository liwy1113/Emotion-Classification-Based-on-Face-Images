import glob2
from shutil import copyfile
import sys
import random
import numpy as np
import pickle
import re
import scipy as sci
from sklearn import linear_model, neighbors, preprocessing, cross_validation, svm, metrics
from sklearn.neighbors import KNeighborsClassifier
import operator
from numpy import arange
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
import pylab
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
import pylab
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from math import *
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix

sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2


emotions = ["neutral", "anger", "disgust", "happy", "sadness", "surprise"] 


"""
def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob2.glob("dataset//%s//*" %emotion)
    return files


training_data = []
training_labels = []
for emotion in emotions:
    training = get_files(emotion)
        #Append data to training list, and generate labels 0-7
    for item in training:
        image = cv2.imread(item) #open image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
        training_data.append(gray) #append image array to training data list
        training_labels.append(emotions.index(emotion))


pickle.dump(training_data, open('training_data.p','wb+'))
pickle.dump(training_labels, open('training_labels.p','wb+'))

#thefile = open('trainlabel.txt', 'w')
#for item in training_labels:
#    thefile.write("%s\n" % item)
"""

"""
##decrease the number of labels

indexes = range(ytr.index(2),ytr.index(2)+18)+range(ytr.index(4),ytr.index(4)+25)

for index in sorted(indexes, reverse=True):
    del Xtr[index]


for index in sorted(indexes, reverse=True):
    del ytr[index]

#emotions = ["neutral", "anger", "disgust", "happy", "sadness", "surprise"] #delete 2,4
#ytr = [2 if x==3 else x for x in ytr]
#ytr = [3 if x==5 else x for x in ytr]
#ytr = [4 if x==6 else x for x in ytr]
#ytr = [5 if x==7 else x for x in ytr]

#pickle.dump(Xtr, open('training_data_del.p','wb+'))
#pickle.dump(ytr, open('training_labels_del.p','wb+'))

"""


#Xtr =  pickle.load(open('training_data.p','rb'))
#ytr =  pickle.load(open('training_labels.p','rb')) 
"""

Xtr =  pickle.load(open('training_data_del.p','rb'))
y =  pickle.load(open('training_labels_del.p','rb')) 
del y[600] #delete the outlier
del Xtr[600] # delete the outlier




###############Dimensional Reduction##############################

from sklearn.decomposition import RandomizedPCA
imnbr = len(Xtr) #get the number of images
immatrix = np.array([np.array(Xtr[i]).flatten() for i in range(imnbr)],'f')  #(654, 122500)

#pca = RandomizedPCA(n_components = 100,whiten = True)
pca = RandomizedPCA()
pca.fit(immatrix)                 

#print(sum(pca.explained_variance_ratio_)) 


Xtrp = pca.transform(immatrix)
"""

"""
#stratified CV
Xtr_p,Xte_p,ytr,yte = cross_validation.train_test_split(Xtrp,y,train_size = 0.8, stratify = y)




pickle.dump(Xtr_p, open('training_data.p','wb+'))
pickle.dump(ytr, open('training_labels.p','wb+'))

pickle.dump(Xte_p, open('testing_data.p','wb+'))
pickle.dump(yte, open('testing_labels.p','wb+'))
"""


ytr =  pickle.load(open('training_labels.p','rb')) 

yte =  pickle.load(open('testing_labels.p','rb')) 
Xtr_p = pickle.load(open('training_data.p','rb'))
Xte_p = pickle.load(open('testing_data.p','rb'))



"""

import csv



with open('train_pca.csv', 'wb') as myfile:
    wr = csv.writer(myfile)
    for i in range(len(Xtr_p)):
        wr.writerow(Xtr_p[i])
        print(i)

with open('test_pca.csv', 'wb') as myfile:
    wr = csv.writer(myfile)
    for i in range(len(Xte_p)):
        wr.writerow(Xte_p[i])
        print(i)

thefile = open('trainlabel.txt', 'w')
for item in ytr:
    thefile.write("%s\n" % item)


thefile = open('testlabel.txt', 'w')
for item in yte:
    thefile.write("%s\n" % item)

"""



"""
pylab.figure()
pylab.gray()
pylab.imshow(Xtrp[200])

pylab.show()
"""




#Use 5 fold CV and tune C parameter in SVM linear
cverror = []
for c in (0.001,0.01,0.1,1, 10, 2**5):
    clf = svm.SVC(kernel='linear', C=c)
    scores = cross_validation.cross_val_score(clf, Xtr_p, ytr, cv=5, scoring = 'accuracy')
    cverror.append(np.mean(1-scores))  

print("Linear SVM:")  
print((0.001,0.01,0.1,1, 10, 2**5)[cverror.index(min(cverror,key = float))]) 
print(min(cverror,key = float))
##C = 0.1 is the best, error = 0.174598534259


#Use 5 fold CV and tune C parameter in SVM with rbf kernel
cverror = []
for c in (0.001,0.01,0.1,1, 10, 2**5):
    clf = svm.SVC(kernel='rbf', C=c)
    scores = cross_validation.cross_val_score(clf, Xtr_p, ytr, cv=5, scoring = 'accuracy')
    cverror.append(np.mean(1-scores))  
print("rbf kernel SVM:")  
print((0.001,0.01,0.1,1, 10, 2**5)[cverror.index(min(cverror,key = float))]) 
print(min(cverror,key = float))
##C = 10 is the best, error = 0.232424188232

#Use 5 fold CV and tune C parameter in SVM with polynomial kernel
cverror = []
for c in (0.001,0.01,0.1,1, 10, 2**5):
    clf = svm.SVC(kernel='poly', C=c)
    scores = cross_validation.cross_val_score(clf, Xtr_p, ytr, cv=5, scoring = 'accuracy')
    cverror.append(np.mean(1-scores))  
print("polynomial kernel SVM:")  
print((0.001,0.01,0.1,1, 10, 2**5)[cverror.index(min(cverror,key = float))]) 
print(min(cverror,key = float))
#C = 0.001 is the best, error = 0.463994389508


gammarange = np.arange(0.1,3,step = 0.1)
#Use 5 fold CV and tune gamma parameter in SVM with polynomial kernel
cverror = []
for r in gammarange:
    clf = svm.SVC(kernel='poly', gamma=r)
    scores = cross_validation.cross_val_score(clf, Xtr_p, ytr, cv=5, scoring = 'accuracy')
    cverror.append(np.mean(1-scores))  
print("polynomial kernel SVM when tuning gamma:")  
print(gammarange[cverror.index(min(cverror,key = float))]) 
print(min(cverror,key = float))
#r = 0.1 is the best, error = 0.484719510485

#Use 5 fold CV and tune gamma parameter in SVM with rbf kernel
cverror = []
for r in gammarange:
    clf = svm.SVC(kernel='rbf',C = 10, gamma=r)
    scores = cross_validation.cross_val_score(clf, Xtr_p, ytr, cv=5, scoring = 'accuracy')
    cverror.append(np.mean(1-scores))  
print("rbf kernel SVM when tuning gamma:")  
print(gammarange[cverror.index(min(cverror,key = float))]) 
print(min(cverror,key = float))
#r = 0.1 is the best, error = 0.463994389508

#Use 5 fold CV and tune C parameter in Logistic regression
cverror = []
for c in (0.001,0.01,0.1,1,10,2**5):
    model=LogisticRegression(C = c)
    scores = cross_validation.cross_val_score(model, Xtr_p, ytr, cv=5)
    cverror.append(np.mean(1-scores))    
print("Logistic regresion:")
print((0.001,0.01,0.1,1, 10, 2**5)[cverror.index(min(cverror,key = float))]) 
print(min(cverror,key = float))
#c = 10, error is 0.0473448480751


#Tuning the parameter in KNN, choosing k = 1 to 50.
error = []
for k in range(1,51):
    knn = neighbors.KNeighborsClassifier(k)
    scores = cross_validation.cross_val_score(knn, Xtr_p, ytr, cv=5)
    error.append(np.mean(1-scores)) 
print("KNN:")
print(error.index(min(error,key = float))+1) 
print(min(error,key = float))
#K = 27,error = 0.0392542053394



#Apply Naive Bayes
gnb = GaussianNB()
scores = cross_validation.cross_val_score(gnb, Xtr_p, ytr, cv=5)
error = np.mean(1-scores)
print("NB:")
print(error)

# Apply random forest
cverror = []
for e in (10,40,80,100):
    clf = RandomForestClassifier(n_estimators=e)
    scores = cross_validation.cross_val_score(clf, Xtr_p, ytr, cv=5, scoring = 'accuracy')
    cverror.append(np.mean(1-scores))  
print("Random Forest tree:")  
print((10,40,80,100)[cverror.index(min(cverror,key = float))]) 
print(min(cverror,key = float))

#Apply GradientBoosting 
clf = LinearDiscriminantAnalysis()
scores = cross_validation.cross_val_score(clf, Xtr_p, ytr, cv=5)
error = np.mean(1-scores)
print("LDA:")
print(error)



#Choose three best methods and then run onto the test dataset
model1 = svm.SVC(C = 0.01, kernel = 'linear',probability = True)
model2 = LogisticRegression(C = 0.1)
model3 = LinearDiscriminantAnalysis()
model1.fit(Xtr_p,ytr)
model2.fit(Xtr_p,ytr)
model3.fit(Xtr_p,ytr)

print("Three best model to fit the test:")
print("model1:")
pred1 = model1.predict(Xte_p)
print(1-metrics.accuracy_score(yte, pred1))
print(classification_report(yte, pred1, target_names=emotions))
print("model2:")
pred2 = model2.predict(Xte_p)
print(1-metrics.accuracy_score(yte, pred2))
print(classification_report(yte, pred2, target_names=emotions))
print("model3:")
pred3 = model3.predict(Xte_p)
print(1-metrics.accuracy_score(yte, pred3))
print(classification_report(yte, pred3, target_names=emotions))



#First fit the two chosen model using the whole data
model1 = svm.SVC(C = 0.01, kernel = 'linear',probability = True)
model2 = LogisticRegression(C = 0.1)


model1.fit(Xtr_p,ytr)
model2.fit(Xtr_p,ytr)




m1 = svm.SVC(C = 0.01, kernel = 'linear',probability = True)
m2 = LogisticRegression(C = 0.1)
#print(len(model2.predict_proba(Xte_p)))




a1 = log((1-0.165820332033)/0.165820332033)
a2 = log((1-0.143339333933)/0.143339333933)


alpha1 = a1/(a1+a2)
alpha2 = a2/(a1+a2)
print(alpha1)
print(alpha2)



eclf = VotingClassifier(estimators=[('svc', m1), ('lr', m2)], voting='soft',weights = [a1,a2])

predv = OneVsOneClassifier(eclf).fit(Xtr_p,ytr).predict(Xte_p)
print("Voting:")
print(metrics.accuracy_score(yte, predv))
print(classification_report(yte, predv, target_names=emotions))

print("Three best model to fit the test:")
print("model1:")
pred1 = model1.predict(Xte_p)
print(metrics.accuracy_score(yte, pred1))
print(classification_report(yte, pred1, target_names=emotions))
print("model2:")
pred2 = model2.predict(Xte_p)
print(metrics.accuracy_score(yte, pred2))
print(classification_report(yte, pred2, target_names=emotions))



def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(emotions))
    plt.xticks(tick_marks, emotions, rotation=45)
    plt.yticks(tick_marks, emotions)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
cm = confusion_matrix(yte, predv)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

plt.show()



