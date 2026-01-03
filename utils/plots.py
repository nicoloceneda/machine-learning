# Import the libraries

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr

# Function to plot the decision boundary

def plot_decision_regions(X, y, classifier, resolution=0.02, test_idx=None):
    """
    Create a colormap from the list of colors.

    Generate a matrix with two columns, where rows are all possible combinations of all numbers from 
    min-1 to max+1 of the two series of features. A matrix with two columns is needed because the 
    perceptron was trained on a matrix with such shape.

    Use the step_activ method of the ppn to predict the class corresponding to all the possible 
    combinations of features generated in the above matrix. The step_activ method will use the 
    weights learnt during the training phase: since the number of misclassifications converged 
    during the training phase, we expect the perceptron to find a decision boundary that correctly 
    classifies all the samples in the training set.

    Reshape the vector of predictions as the X0_grid.

    Draw filled contours, where all possible combinations of features are associated to a Z, which 
    is 0 or 1.

    To verify that the perceptron correctly classified all the samples in the training set, plot the 
    original samples in the scatter plot and verify that they fall inside the correct region.
    """

    colors = ('red', 'blue', 'green')
    cmap = clr.ListedColormap(colors[:len(np.unique(y))])

    X0_min, X0_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    X1_min, X1_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    X0_grid, X1_grid = np.meshgrid(np.arange(X0_min, X0_max, resolution), np.arange(X1_min, X1_max, resolution))
    X0X1_combs = np.array([X0_grid.ravel(), X1_grid.ravel()]).T

    Z = classifier.predict(X0X1_combs)
    
    Z = Z.reshape(X0_grid.shape)

    plt.figure()
    plt.contourf(X0_grid, X1_grid, Z, alpha=0.3, cmap=cmap)
    plt.xlim(X0_min, X0_max)
    plt.ylim(X1_min, X1_max)

    for pos, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[pos], marker='+', label=cl)

    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], alpha=0.8, linewidth=1, c='none', marker='s', edgecolor='black', label='Test set')
