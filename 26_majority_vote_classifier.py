# %% SETUP

# Import the libraries

import os
import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.pipeline import Pipeline, _name_estimators
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc

# Image path

script_name = os.path.basename(__file__)
script_name = os.path.splitext(script_name)[0]

save_dir = os.path.join('images', script_name)
os.makedirs(save_dir, exist_ok=True)


# %% DATA

# Import the dataset

data = load_iris()

# Extract the class labels

y = data.target[50:]

# Extract the features

X = data.data[50:, [1, 2]]

# Encode the categorical class labels

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into train and test subsets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, stratify=y)


# %% MODEL

# Design the majority vote classifier

class MajorityVoteClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, classifiers, vote='classlabel', weights=None):

        self.classifiers = classifiers
        self.vote = vote
        self.weights = weights

        self.named_classifiers = {key: value for key, value in _name_estimators(classifiers)}

    def fit(self, X, y):

        if self.vote not in ('classlabel', 'probability'):

            raise ValueError("vote must be 'classlabel' or 'probability'")

        if self.weights and len(self.weights) != len(self.classifiers):

            raise ValueError("weights must have the same length as classifiers")

        self.labelenc = LabelEncoder()
        y_encoded = self.labelenc.fit_transform(y)

        self.classes_ = self.labelenc.classes_
        self.fitted_classifiers = []

        for clf in self.classifiers:

            fitted_clf = clone(clf).fit(X, y_encoded)
            self.fitted_classifiers.append(fitted_clf)

        return self

    def predict(self, X):

        if self.vote == 'classlabel':

            predictions = np.asarray([clf.predict(X) for clf in self.fitted_classifiers]).T         # n_samples x n_classifiers
            maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=self.weights)), axis=1, arr=predictions) # n_samples

        elif self.vote == 'probability':

            maj_vote = np.argmax(self.predict_proba(X), axis=1)                                     # n_samples

        maj_vote = self.labelenc.inverse_transform(maj_vote)

        return maj_vote

    def predict_proba(self, X):
        """
        Predict class probabilities for X
        
        Parameters:
        ----------
        X : array, shape = [n_samples, n_features]
        
        Returns:
        -------
        avg_proba : array, shape = [n_samples, n_classes]
        """

        probas = np.asarray([clf.predict_proba(X) for clf in self.fitted_classifiers])              # n_classifiers x n_samples x n_classes
        avg_proba = np.average(probas, axis=0, weights=self.weights)                                # n_samples x n_classes

        return avg_proba

    def get_params(self, deep=True):

        if not deep:

            return super().get_params(deep=False)

        else:

            out = self.named_classifiers.copy()

            for name, step in self.named_classifiers.items():

                for key, value in step.get_params(deep=True).items():

                    out[f'{name}__{key}'] = value

            return out

# Initialize the classifiers

clf1 = LogisticRegression(penalty='l2', C=0.001, random_state=1)
clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')

pipe1 = Pipeline([['sc', StandardScaler()], ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])
mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3], vote='classlabel', weights=None)


# %% IN-SAMPLE PERFORMANCE

clf_all = [pipe1, clf2, pipe3, mv_clf]
clf_labels = ['Logistic regression', 'Decision tree', 'KNN', 'Majority voting']

# Stratified K-fold cross-validation

for clf, label in zip(clf_all, clf_labels):

    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
    print(f'ROC AUC: {np.mean(scores):.2f} +/- {np.std(scores):.2f} | {label}')

# Plot the ROC curve

colors = ['black', 'orange', 'blue', 'green']

for clf, label, clr in zip(clf_all, clf_labels, colors):

    y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=clr, label=f'{label} (AUC = {roc_auc:.2f})')
    
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.savefig(os.path.join(save_dir, 'ROC_curves.png'))


# %% DECISION REGIONS

# Apply the standardization to scale the features

std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)

# Plot the decision boundary

X0_min, X0_max = X_train_std[:, 0].min() - 1, X_train_std[:, 0].max() + 1
X1_min, X1_max = X_train_std[:, 1].min() - 1, X_train_std[:, 1].max() + 1
X0_grid, X1_grid = np.meshgrid(np.arange(X0_min, X0_max, 0.1), np.arange(X1_min, X1_max, 0.1))
X0X1_combs = np.array([X0_grid.ravel(), X1_grid.ravel()]).T

fig, ax = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row', figsize=(7,5))

for i, clf, label in zip(product([0,1], [0,1]), clf_all, clf_labels):

    clf.fit(X_train_std, y_train)
    Z = clf.predict(X0X1_combs)
    Z = Z.reshape(X0_grid.shape)

    ax[i[0], i[1]].contourf(X0_grid, X1_grid, Z, alpha=0.3)
    ax[i[0], i[1]].scatter(X_train_std[y_train==0, 0], X_train_std[y_train==0, 1], c='blue', marker='+', s=50)
    ax[i[0], i[1]].scatter(X_train_std[y_train==1, 0], X_train_std[y_train==1, 1], c='green', marker='+', s=50)
    ax[i[0], i[1]].set_title(label)
    plt.text(-3.5, -5., s='Sepal width [standardized]', ha='center', va='center', fontsize=12)
    plt.text(-12.5, 4.5, s='Petal length [standardized]', ha='center', va='center', fontsize=12, rotation=90)

fig.savefig(os.path.join(save_dir, 'Decision_regions.png'))


# %% TRAINING

# Grid search cross-validation

param_grid = [{'decisiontreeclassifier__max_depth': [1, 2], 
              'pipeline-1__clf__C': [0.001, 0.1, 100.0]}]

gs = GridSearchCV(estimator=mv_clf, param_grid=param_grid, scoring='roc_auc', cv=10, refit=True, n_jobs=-1)

# Run cross-validation and retrain the model on the entire training dataset

gs.fit(X_train, y_train)

# Print the best score and the best parameters

for r, _ in enumerate(gs.cv_results_['mean_test_score']):
    mean_score = gs.cv_results_['mean_test_score'][r]
    std_dev = gs.cv_results_['std_test_score'][r]
    params = gs.cv_results_['params'][r]
    print(f'{mean_score:.3f} +/- {std_dev:.2f} {params}')

print(f'Best parameters: {gs.best_params_}')
print(f'ROC AUC : {gs.best_score_:.2f}')


# %% TESTING

# Predict the classes of the samples in the test set

y_pred = gs.predict(X_test)

# Evaluate the performance of the model

print(f'Prediction accuracy: {gs.score(X_test, y_test):.3f}')


# %% GENERAL

# Show plots

plt.show()
