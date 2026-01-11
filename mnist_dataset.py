""" MNIST DATASET
    -------------
    Download the mnist dataset and save the standardized features and class labels.
"""


# -------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES
# -------------------------------------------------------------------------------


import os
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------
# 1. PREPARE THE DATA
# -------------------------------------------------------------------------------


# Function to import the train and test subsets from the mnist dataset

def load_mnist(path, kind):

    labels_path = os.path.join(path, '{}-labels-idx1-ubyte'.format(kind))
    images_path = os.path.join(path, '{}-images-idx3-ubyte'.format(kind))

    with open(labels_path, 'rb') as lbpath:

        file_protocol, num_items = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as impath:

        file_protocol, num_items, rows, cols = struct.unpack('>IIII', impath.read(16))
        images = np.fromfile(impath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


# Import the train and test subsets

X_train, y_train = load_mnist(path='mnist dataset/original', kind='train')
X_test, y_test = load_mnist(path='mnist dataset/original', kind='t10k')


# Apply the standardization to scale the features

X_train_std = ((X_train / 255) - 0.5) * 2
X_test_std = ((X_test / 255) - 0.5) * 2


# Plot examples of the digits

fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()

for i in range(10):

    img = X_train_std[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.savefig('images/07_mnist dataset/Examples_of_the_digits.png')


# Plot examples of the same digit

fix, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()

for i in range(25):

    img = X_train_std[y_train == 5][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.savefig('images/07_mnist dataset/Examples_of_the_same_digit.png')


# -------------------------------------------------------------------------------
# 2. SAVE THE DATA
# -------------------------------------------------------------------------------


# Save the train and test subsets in a compressed file

np.savez_compressed('mnist dataset/compressed/mnist_std.npz', X_train_std=X_train_std, y_train=y_train, X_test_std=X_test, y_test=y_test)


# -------------------------------------------------------------------------------
# 3. GENERAL
# -------------------------------------------------------------------------------


# Show plots

plt.show()
