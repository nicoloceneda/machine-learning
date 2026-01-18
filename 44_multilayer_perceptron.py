# %% SETUP

# Import the libraries

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Image path

script_name = os.path.basename(__file__)
script_name = os.path.splitext(script_name)[0]

save_dir = os.path.join('images', script_name)
os.makedirs(save_dir, exist_ok=True)


# %% DATA

# Import the dataset

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# Extract the class labels

y = y.astype(int).values

# Extract the features

X = X.values

# Apply the standardization to scale the features

X_std = ((X / 255) - 0.5) * 2

# Separate the data into train, validation and test subsets with the same proportions of class labels as the input dataset

X_train_std, X_test_std, y_train, y_test = train_test_split(X_std, y, test_size=10_000, random_state=123, stratify=y)
X_train_std, X_valid_std, y_train, y_valid = train_test_split(X_train_std, y_train, test_size=5_000, random_state=123, stratify=y_train)

# Plot examples of digits 0-9

fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()

for i in range(10):

    img = X_train_std[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Digits_0_9.png'))

# Plot examples of the same digit

fix, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()

for i in range(25):

    img = X_train_std[y_train == 7][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Digits_multi_7.png'))


# %% HELPER FUNCTIONS

# Logistic sigmoid activation

def sigmoid(z):

    return 1 / (1 + np.exp(-z))

# One-hot encoder

def onehot_encode(y, n_labels):

    onehot = np.zeros((y.shape[0], n_labels))

    for i, label in enumerate(y):

        onehot[i, label] = 1

    return onehot

# Mini-batch generator

def minibatch_generator(X, y, minibatch_size):

    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    for start_idx in range(0, n_samples - minibatch_size + 1, minibatch_size):

        batch_idx = indices[start_idx:start_idx + minibatch_size]
        yield X[batch_idx], y[batch_idx]

# Mean squared error and accuracy

def compute_mse_and_acc(model, X, y, num_labels, minibatch_size):

    mse = 0.0
    n_correct = 0
    n_examples = 0

    minibatch_gen = minibatch_generator(X, y, minibatch_size)

    for i, (X_mb, y_mb) in enumerate(minibatch_gen):

        _, probas = model.forward(X_mb)

        predicted_labels = np.argmax(probas, axis=1)
        n_correct += np.sum(predicted_labels == y_mb)
        n_examples += y_mb.shape[0]

        onehot_targets = onehot_encode(y_mb, num_labels)
        loss = np.mean((onehot_targets - probas) ** 2)
        mse += loss
    
    mse = mse / i
    acc = n_correct / n_examples

    return mse, acc


# %% MODEL

# Design the multi-layer perceptron

class MultilayerPerceptron:
    """
    Multilayer perceptron classifier

    Parameters:
    ----------
    num_features : int
        Number of features.
    num_hidden : int
        Number of hidden units.
    num_classes : int
        Number of classes.
    """

    def __init__(self, num_features, num_hidden, num_classes):
        
        super().__init__()

        self.num_classes = num_classes
        rng = np.random.RandomState(123)

        # Hidden layer

        self.w_h = rng.normal(loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.b_h = np.zeros(num_hidden)

        # Output layer

        self.w_o = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.b_o = np.zeros(num_classes)

    def forward(self, X):
        """
        Forward pass

        Parameters:
        ----------
        X : array, shape = [n_examples, n_features]
            Feature values.

        Returns:
        -------
        a_h : array, shape = [n_examples, n_hidden]
            Hidden layer activations.
        a_o : array, shape = [n_examples, n_classes]
            Output layer activations.
        """

        z_h = np.dot(X, self.w_h.T) + self.b_h                                                      # [n_examples, n_hidden]
        a_h = sigmoid(z_h)

        z_o = np.dot(a_h, self.w_o.T) + self.b_o                                                    # [n_examples, n_classes]
        a_o = sigmoid(z_o)

        return a_h, a_o

    def backward(self, X, y, a_h, a_o):
        """
        Backpropagation algorithm

        Parameters:
        ----------
        X : array, shape = [n_examples, n_features]
            Feature values.
        y : array, shape = [n_examples,]
            Class labels.
        a_h : array, shape = [n_examples, n_hidden]
            Hidden layer activations.
        a_o : array, shape = [n_examples, n_classes]
            Output layer activations.
        
        Returns:
        -------
        dloss__dw_o : array, shape = [n_classes, n_hidden]
            Gradient of the loss wrt the output layer weights.
        dloss__db_o : array, shape = [n_classes,]
            Gradient of the loss wrt the output layer biases.
        dloss__dw_h : array, shape = [n_hidden, n_features]
            Gradient of the loss wrt the hidden layer weights.
        dloss__db_h : array, shape = [n_hidden,]
            Gradient of the loss wrt the hidden layer biases.
        """

        y_onehot = onehot_encode(y, self.num_classes)

        # Output layer: dloss/dw_o = dloss/da_o * da_o/dz_o * dz_o/dw_o

        dloss__da_o = 2 * (a_o - y_onehot) / y.shape[0]                                                 # [n_examples, n_classes]
        da_o__dz_o = a_o * (1 - a_o)                                                                    # [n_examples, n_classes]
        dz_o__dw_o = a_h                                                                                # [n_examples, n_hidden]

        dloss__dw_o = np.dot((dloss__da_o * da_o__dz_o).T, dz_o__dw_o)                                  # [n_classes, n_hidden]
        dloss__db_o = np.sum(dloss__da_o * da_o__dz_o, axis=0)                                          # [n_classes,]

        # Hidden layer: dloss/dw_h = dloss/da_o * da_o/dz_o * dz_o/da_h * da_h/dz_h * dz_h/dw_h

        dz_o__da_h = self.w_o                                                                           # [n_classes, n_hidden]
        da_h__dz_h = a_h * (1 - a_h)                                                                    # [n_examples, n_hidden]
        dz_h__dw_h = X                                                                                  # [n_examples, n_features]

        dloss__dw_h = np.dot((np.dot(dloss__da_o * da_o__dz_o, dz_o__da_h) * da_h__dz_h).T, dz_h__dw_h) # [n_hidden, n_features]
        dloss__db_h = np.sum(np.dot(dloss__da_o * da_o__dz_o, dz_o__da_h) * da_h__dz_h, axis=0)         # [n_hidden,]

        return dloss__dw_o, dloss__db_o, dloss__dw_h, dloss__db_h

# Initialize a multilayer perceptron object

mlp = MultilayerPerceptron(num_features=28*28, num_hidden=50, num_classes=10)


# %% TRAINING

# Model training

def train(model, X_train_std, y_train, X_valid_std, y_valid, n_epochs, learning_rate):

    epoch_train_loss = []
    epoch_valid_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []

    for epoch in range(n_epochs):

        minibatch_get = minibatch_generator(X_train_std, y_train, minibatch_size)

        for X_train_mb, y_train_mb in minibatch_get:

            # Forward pass

            a_h, a_o = model.forward(X_train_mb)

            # CBackward pass

            dloss__dw_o, dloss__db_o, dloss__dw_h, dloss__db_h = model.backward(X_train_mb, y_train_mb, a_h, a_o)

            # Update weights and biases

            model.w_h -= learning_rate * dloss__dw_h
            model.b_h -= learning_rate * dloss__db_h
            model.w_o -= learning_rate * dloss__dw_o
            model.b_o -= learning_rate * dloss__db_o

        # Compute loss and accuracy for the epoch

        train_mse, train_acc = compute_mse_and_acc(model, X_train_std, y_train, num_labels, minibatch_size)
        valid_mse, valid_acc = compute_mse_and_acc(model, X_valid_std, y_valid, num_labels, minibatch_size)

        epoch_train_loss.append(train_mse)
        epoch_valid_loss.append(valid_mse)
        epoch_train_acc.append(train_acc * 100)
        epoch_valid_acc.append(valid_acc * 100)

        print(f'Epoch: {epoch + 1}/{n_epochs}' 
              f'| Train MSE: {train_mse:.3f}' 
              f'| Valid MSE: {valid_mse:.3f}' 
              f'| Train Acc: {train_acc * 100:.2f}%' 
              f'| Valid Acc: {valid_acc * 100:.2f}%')

    return epoch_train_loss, epoch_valid_loss, epoch_train_acc, epoch_valid_acc

# Learn from the data via the train method

n_epochs = 50
learning_rate = 0.1

minibatch_size = 100
num_labels = 10
np.random.seed(123)

epoch_train_loss, epoch_valid_loss, epoch_train_acc, epoch_valid_acc = \
    train(mlp, X_train_std, y_train, X_valid_std, y_valid, n_epochs, learning_rate)

# Plot the loss on the train and validation sets

plt.figure()
plt.plot(range(1, n_epochs + 1), epoch_train_loss, label='Training loss')
plt.plot(range(1, n_epochs + 1), epoch_valid_loss, label='Validation loss')
plt.title('Loss function per epoch')
plt.xlabel('Epoch')
plt.ylabel('Mean squared errors')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'Loss_per_epoch.png'))

# Plot the accuracy on the train and validation sets

plt.figure()
plt.plot(range(1, n_epochs + 1), epoch_train_acc, label='Training accuracy')
plt.plot(range(1, n_epochs + 1), epoch_valid_acc, label='Validation accuracy')
plt.title('Accuracy per epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'Accuracy_per_epoch.png'))


# %% TESTING

# Evaluate the model on the test set

test_mse, test_acc = compute_mse_and_acc(mlp, X_test_std, y_test, num_labels, minibatch_size)
print(f'Test MSE: {test_mse:.3f}')
print(f'Test accuracy: {test_acc*100:.2f}%')

# Plot the first 25 misclassified samples

_, probas = mlp.forward(X_test_std)
y_test_pred = np.argmax(probas, axis=1)

X_test_std_miss = X_test_std[y_test != y_test_pred][:25]
y_test_pred_miss = y_test_pred[y_test != y_test_pred][:25]
y_test_miss = y_test[y_test != y_test_pred][:25]

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True, figsize=(8, 8))
ax = ax.flatten()

for i in range(25):

    img = X_test_std_miss[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title(f'{i+1}) ' 
                    f'True: {y_test_miss[i]}\n' 
                    f' Predicted: {y_test_pred_miss[i]}')
    ax[i].set_xticks([])
    ax[i].set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Misclassified_samples.png'))


# %% GENERAL

# Show plots

plt.show()
