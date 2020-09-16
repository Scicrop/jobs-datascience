from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, \
                            roc_auc_score

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

from sklearn.metrics import precision_recall_curve
from plotly.subplots import make_subplots
from plotly import graph_objects as go

def plot_confusion_matrix(cm, labels, suptitle = 'Confusion Matrix'):
  """
  _subplot_cm wapper - Plot normalized and not
  normilized confusion matrix

  :type cm: array
  :param cm: confusion matrix array

  :type labels: list
  :param labels: list containing label strings

  :type suptitle: string
  :param suptitle: plot title, defaults to Confusion Matrix

  """
  fig, ax = plt.subplots(1,2, sharey=True)
  fig.suptitle(suptitle)
  _subplot_cm(cm, labels ,fig, ax[0], normalize=False)
  _subplot_cm(cm, labels ,fig, ax[1], normalize=True)
  plt.show()

def _subplot_cm(cm,
        classes,
        fig, ax,
        normalize=False,
        title=None):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.

  :type cm: array
  :param cm: confusion matrix array

  :type classes: list
  :param classes: list containing label strings

  :type normalize: boolean
  :param normalize: normilize by rows

  :type title: string
  :param title: plot title, defaults to None
  """

  if not title:
    if normalize:
      title = 'Normalized'
    else:
      title = 'Without normalization'

  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

  ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

  ax.set(
    xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    xticklabels=classes,
    yticklabels=classes,
    title=title,
    ylabel='True label',
    xlabel='Predicted label')

  plt.setp(ax.get_xticklabels(),
       rotation=45,
       ha="right",
       rotation_mode="anchor")

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
      ax.text(j,
          i,
          format(cm[i, j], fmt),
          ha="center",
          va="center",
          color="white" if cm[i, j] > thresh else "black")
  fig.tight_layout()

def Evaluate(true_label, predicted_label, predicted_prob, labels):
  """
  Plot confusion Matrix and displays accuracy f1 and roc_auc scores

  :type true_label: array
  :param true_label: ground truth values

  :type predicted_label: array
  :param predicted_label: predicted values

  :type predicted_prob: array
  :param predicted_prob: probability for each predicted class

  :type labels: list
  :param labels: list containing label strings
  """
  cm = confusion_matrix(true_label, predicted_label)
  acc = accuracy_score(true_label, predicted_label)
  f1_weighted = f1_score(true_label, predicted_label,
               average = 'macro')
  roc_auc = roc_auc_score(true_label, predicted_prob,
              multi_class= 'ovr' )

  plot_confusion_matrix(cm,  labels )

  return  pd.DataFrame([[acc, f1_weighted, roc_auc]],
            columns = ['Accuracy', 'F1 Score Weighted', 'ROC AUC'])

def plot_precision_recall(y_true, preds_proba):
  """
  Plot precision recall curves

  :type y_true: array
  :param y_true: ground truth values

  :type preds_proba: array
  :param preds_proba: probability for positive predicted class
  """

  # calculate model precision-recall curve
  precision, recall, threshold = precision_recall_curve(y_true, preds_proba)
  # plot the model precision-recall curve

  fig = make_subplots(1,2, subplot_titles=("Recall x Precision", "Recall and Precision Curves"))

  fig.add_trace(go.Scatter(
      x=recall,
      y=precision,
      name = 'Recall x Precision',
                          ),
                row = 1,
                col = 1
              )

  fig.add_trace(go.Scatter(
      x=threshold,
      y=precision[:-1],
      name= 'Precision',
                          ),
                row = 1,
                col = 2
              )

  fig.add_trace(go.Scatter(
      x=threshold,
      y=recall[:-1],
      name = 'Recall',
                          ),
                row = 1,
                col = 2
              )

  for trace in fig['data']:
      if(trace['name'] == 'Precision x Recall'): trace['showlegend'] = False
  fig.update_yaxes(title_text="Precision", row=1, col=1)
  fig.update_xaxes(title_text="Recall", row=1, col=1)
  fig.update_xaxes(title_text="Threshold", row=1, col=2)

  return fig

def arg_nearest(array, value):
  """
  Find index of nearest value for a given number

  :type array: array
  :param array: numpy array

  :type value: float
  :param value: desired value

  :return: index
  :rtype: int
  """
  array = np.asarray(array)
  idx = (np.abs(array - value)).argmin()
  return idx
