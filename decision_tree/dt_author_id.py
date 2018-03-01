#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier as DTF
clf2 = DTF(min_samples_split = 2) #the default DTF

#clf50 = DTF(criterion = "entropy", splitter = "best",max_depth = None,min_samples_leaf = 20, min_samples_split = 2) 
#Tuned DTF:
#criterion seems to have no impact on the data in this example
#splitter = 'random' seems to have a bad impact on the accuracy.
#max-depth limit the levels of branches that your tree will gave. Can be overwritten by the min_samples_split
#min_samples_leaf = imporve the accuracy by reducing the overfitting

clf2.fit(features_train,labels_train)

#clf50.fit(features_train,labels_train)

pred2 = clf2.predict(features_test)

#pred50 = clf50.predict(features_test)

acc_min_samples_split_2 = accuracy_score(labels_test, pred2)

#acc_min_samples_split_50 = accuracy_score(labels_test, pred50)

print('accuracy by default is {}'.format(acc_min_samples_split_2))
#print('accuracy tuned is {}'.format(acc_min_samples_split_50))

#########################################################
