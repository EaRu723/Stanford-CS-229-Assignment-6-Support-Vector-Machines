import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from sklearn import svm
import plotData as pd
import visualizeBoundary as vb
import gaussianKernel as gk

plt.ion()
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})

# ===================== Part 1: Loading and Visualizing Data =====================
# 

print('Loading and Visualizing data ... ')

# Load from ex6data1:
data = scio.loadmat('ex6data1.mat') # Load dataset from MATLAB
X = data['X'] # Feature matrix
y = data['y'].flatten() # target values - flattened for simplicity
m = y.size # Number of examples in dataset

# Plot training data
pd.plot_data(X, y)

input('Program paused. Press ENTER to continue')

# ===================== Part 2: Training Linear SVM =====================
# Train a linear SVM and print the decision boundary

print('Training Linear SVM')



c = 1000 # Set regularizatoin parameter
clf = svm.SVC(C=c, kernel='linear', tol=1e-3) # Initialize SVM classifier with a linear kernel
clf.fit(X, y) # Train the SVM classifier

pd.plot_data(X, y)
vb.visualize_boundary(clf, X, 0, 4.5, 1.5, 5)

input('Program paused. Press ENTER to continue')

# ===================== Part 3: Implementing Gaussian Kernel =====================
# Evaluate and test the Gaussian kernel

print('Evaluating the Gaussian Kernel')

# Sample kernel dimensions
x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
# Compute the gaussian kernel
sim = gk.gaussian_kernel(x1, x2, sigma)

print('Gaussian kernel between x1 = [1, 2, 1], x2 = [0, 4, -1], sigma = {} : {:0.6f}\n'
      '(for sigma = 2, this value should be about 0.324652'.format(sigma, sim))

input('Program paused. Press ENTER to continue')

# ===================== Part 4: Visualizing Dataset 2 =====================
# Load and plot the seond dataset

print('Loading and Visualizing Data ...')

data = scio.loadmat('ex6data2.mat')
X = data['X']
y = data['y'].flatten()
m = y.size

# Plot training data
pd.plot_data(X, y)

input('Program paused. Press ENTER to continue')

# ===================== Part 5: Training SVM with RBF Kernel (Dataset 2) =====================
# Train SVm with Gaussian kernel and visualize the boundary

print('Training SVM with RFB(Gaussian) Kernel (this may take 1 to 2 minutes) ...')

# set regularization parameter and kernel width
c = 1
sigma = 0.1

# Define a custom RBD (gaussian kernel) function
def gaussian_kernel(x_1, x_2):
    n1 = x_1.shape[0]
    n2 = x_2.shape[0]
    result = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            result[i, j] = gk.gaussian_kernel(x_1[i], x_2[j], sigma)

    return result

# train the SVm classifier with Gaussiean Kernel
clf = svm.SVC(C=c, kernel='rbf', gamma=np.power(sigma, -2))
clf.fit(X, y)

print('Training complete!')

pd.plot_data(X, y) # Plot the data points
vb.visualize_boundary(clf, X, 0, 1, .4, 1.0) # Visualize the decision boundary

input('Program paused. Press ENTER to continue')

# ===================== Part 6: Visualizing Dataset 3 =====================
# 
#

print('Loading and Visualizing Data ...')

# Load from ex6data3:
data = scio.loadmat('ex6data3.mat')
X = data['X']
y = data['y'].flatten()
m = y.size

# Plot training data
pd.plot_data(X, y)

input('Program paused. Press ENTER to continue')

# ===================== Part 7: Visualizing Dataset 3 =====================
# Train SVM with RBF kernel on the third dataset and visualize the boundary
clf = svm.SVC(C=c, kernel='rbf', gamma=np.power(sigma, -2))
clf.fit(X, y)

pd.plot_data(X, y)
vb.visualize_boundary(clf, X, -.5, .3, -.8, .6)

input('ex6 Finished. Press ENTER to exit')
