# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:48:41 2024

@author: aneta.kartali
"""

from collections import defaultdict
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from data.params.data_params_Benfatto import data_params


manifest_path = data_params.manifest_path
filename = os.path.join(manifest_path, f'Segmented-{data_params.time_context}s-AttsAveragedEyes.csv')
data = pd.read_csv(filename)

columns = ['Subject', 'Task', 'Disease', 'active_read_time', 'fixation_intersection_coeff', 
                'saccade_variability', 'fixation_intersection_variability', 
                'fixation_fractal_dimension', 'fixation_count',
                'fixation_total_dur', 'fixation_freq', 'fixation_avg_dur', 
                'saccade_count', 'saccade_total_dur',
                'saccade_freq', 'saccade_avg_dur', 'total_read_time']
features = columns[3:]
filtered_data = data[columns]

cv_mode = "StratifiedGroupKFold"
# "LeaveOneSubjectOut", "StratifiedGroupKFold"

save_path = f'../results/{data_params.time_context}s_segments/{cv_mode}/'
path = Path(save_path)
if not path.exists():
    path.mkdir(parents=True, exist_ok=True)

X = filtered_data[features]
y = filtered_data['Disease'].replace({'bp': 0, 'dys': 1})

methods = {
    'dummy': Pipeline([
        ('dummy', DummyClassifier(strategy='prior')),
    ]),
    'lr': Pipeline([
        ('impute', SimpleImputer(missing_values=np.nan, strategy="mean", add_indicator=False)),
        ('scale', StandardScaler()),
        ('lr', LogisticRegressionCV()),
    ])
}

def tn_scorer(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    return cm[0, 0]

def fp_scorer(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    return cm[0, 1]

def fn_scorer(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    return cm[1, 0]

def tp_scorer(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    return cm[1, 1]

def confusion_matrix_scorer(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    return {'tn': cm[0, 0], 'fp': cm[0, 1],
            'fn': cm[1, 0], 'tp': cm[1, 1]}

scoring = {'acc': 'accuracy',
           'brier': 'neg_brier_score',
           'auc': 'roc_auc',
           'balanced_acc': 'balanced_accuracy',
           'f1': 'f1',
           'precision': 'precision',
           'recall': 'recall',
           'tp': tp_scorer,
           'tn': tn_scorer,
           'fp': fp_scorer,
           'fn': fn_scorer,
           }

if cv_mode == "StratifiedGroupKFold":
    rskf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=1234567)
    groups = filtered_data['Subject']
elif cv_mode == "LeaveOneSubjectOut":
    tmp_groups = []
    idx = 1
    for i in range(len(groups)):
        tmp_groups.append(idx)
        if i != len(groups)-1:
            if groups[i+1] != groups[i]:
                idx += 1
    groups = np.array(tmp_groups)
    rskf = LeaveOneGroupOut()
else:
    print(f"Invalid cross-validation mode: {cv_mode}")
    sys.exit()

results = defaultdict(list)
names = []
for name, method in methods.items():
    print(name)
    names.append(name)
    for i in range(10):
        scores = cross_validate(method, X, y.values.ravel(), groups=groups, scoring=scoring, 
                                cv=rskf, return_train_score=True, 
                                return_estimator=True, verbose=2)
        df_scores = pd.DataFrame(data=scores)
        if i == 0:
            df_scores.to_csv(save_path + name + "-reading.csv", index=False)
        else:
            df_scores.to_csv(save_path + name + "-reading.csv", mode='a', index=False, header=False)
        for s in scores:
            results[s].append(scores[s])

filepath = save_path

filename = 'dummy-reading.csv'
dummy = pd.read_csv(filepath+filename)
dummy_acc = np.mean(dummy['test_acc'])
dummy_brier = np.mean(dummy['test_brier'])
dummy_auc = np.mean(dummy['test_auc'])

filename = 'lr-reading.csv'
lr = pd.read_csv(filepath+filename)
lr_acc = np.mean(lr['test_acc'])
lr_brier = np.mean(lr['test_brier'])
lr_auc = np.mean(lr['test_auc'])