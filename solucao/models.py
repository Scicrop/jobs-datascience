from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, cross_validate
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, mean_squared_error
from sklearn.pipeline import Pipeline

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

import seaborn as sns
import matplotlib.pyplot as plt


def acc_vs_parameter_knn(X_train, y_train, X_val, y_val):
    ## parameters:
    n_neighbors = list(np.linspace(1, 50, 5).astype(int))
    leaf_size = list(np.linspace(10, 40, 5))
    
    sns.set()
    
    acc = np.zeros([2, len(n_neighbors)])
    z = 0
    for i in n_neighbors:
        clf = KNeighborsClassifier(n_neighbors=i)
        clf.fit(X_train, y_train)
        
        pred_train = clf.predict(X_train)
        pred_val = clf.predict(X_val)
        
        acc[0, z] = accuracy_score(y_train, pred_train)
        acc[1, z] = accuracy_score(y_val, pred_val)
        
        z +=1
    
    plt.subplot(1, 2, 1)
    sns.lineplot(n_neighbors, acc[0, :], label = 'train')
    sns.lineplot(n_neighbors, acc[1, :], label = 'validation')
    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('K neighbors')
    
    acc = np.zeros([2, len(leaf_size)])
    z = 0
    for i in leaf_size:
        clf = KNeighborsClassifier(leaf_size=i)
        clf.fit(X_train, y_train)
        
        pred_train = clf.predict(X_train)
        pred_val = clf.predict(X_val)
        
        acc[0, z] = accuracy_score(y_train, pred_train)
        acc[1, z] = accuracy_score(y_val, pred_val)
        
        z +=1
    
    plt.subplot(1, 2, 2)
    sns.lineplot(leaf_size, acc[0, :], label = 'train')
    sns.lineplot(leaf_size, acc[1, :], label = 'validation')
    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('Leaf size')
    
    plt.show()
    
def acc_vs_parameter_svc(X_train, y_train, X_val, y_val):
    ## parameters:
    kernels = ['linear', 'rbf', 'poly']
    gamma = [0.1, 10, 100]
    C = [0.1, 10, 100]
    
    sns.set()
    
    acc = np.zeros([2, len(gamma)])
    z = 0
    for i in gamma:
        clf = SVC(gamma=i)
        clf.fit(X_train, y_train)
        
        pred_train = clf.predict(X_train)
        pred_val = clf.predict(X_val)
        
        acc[0, z] = accuracy_score(y_train, pred_train)
        acc[1, z] = accuracy_score(y_val, pred_val)
        
        z +=1
    
    plt.subplot(1, 2, 1)
    plot_comparacao(gamma, acc, 'gamma')
    
    acc = np.zeros([2, len(C)])
    z = 0
    for i in C:
        clf = SVC(C=i)
        clf.fit(X_train, y_train)
        
        pred_train = clf.predict(X_train)
        pred_val = clf.predict(X_val)
        
        acc[0, z] = accuracy_score(y_train, pred_train)
        acc[1, z] = accuracy_score(y_val, pred_val)
        
        z +=1
    
    plot_comparacao(C, acc, 'C')
    
    plt.show()
    
def plot_comparacao(x, acc, x_label):
    sns.lineplot(X, acc[0, :], label = 'train')
    sns.lineplot(X, acc[1, :], label = 'validation')
    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel(x_label)
    
    