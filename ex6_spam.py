import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from sklearn import svm

import processEmail as pe
import emailFeatures as ef

plt.ion()
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})

# ===================== Part 1: Email Preprocessing =====================
# Convert a raw email into a word incides vector

print('Preprocessing sample email (emailSample1.txt) ...')

file_contents = open('emailSample1.txt', 'r').read()
word_indices = pe.process_email(file_contents)

# Print indices of words found in the email
print('Word Indices: ')
print(word_indices)

input('Program paused. Press ENTER to continue')

# ===================== Part 2: Feature Extraction =====================
# Convert email word indices into a feature vector

print('Extracting Features from sample email (emailSample1.txt) ... ')

# Extract features from word indices
features = ef.email_features(word_indices)

# Print feature vector stats
print('Length of feature vector: {}'.format(features.size))
print('Number of non-zero entries: {}'.format(np.flatnonzero(features).size))

input('Program paused. Press ENTER to continue')

# ===================== Part 3: Train Linear SVM for Spam Classification =====================
# Train a linear SVM to classify spam emails

# Load the training data
data = scio.loadmat('spamTrain.mat')
# Exctract features and labels
X = data['X']
y = data['y'].flatten()

print('Training Linear SVM (Spam Classification)')
print('(this may take 1 to 2 minutes)')

# Train the SVM classifier
c = 0.1 # Regularization paramter
clf = svm.SVC(C=c, kernel='linear') # Initialize SVM with linear kernel
clf.fit(X, y) # Fit the model on the training data

# Calculate training accurary
p = clf.predict(X)

print('Training Accuracy: {}'.format(np.mean(p == y) * 100))

# ===================== Part 4: Test Spam Classification =====================
# Evaluate the classifier on a test set

# Load the test dataset
data = scio.loadmat('spamTest.mat')
# Extracrt feautures and labels
Xtest = data['Xtest']
ytest = data['ytest'].flatten()

print('Evaluating the trained linear SVM on a test set ...')

# Calculate the test accuracy
p = clf.predict(Xtest)

print('Test Accuracy: {}'.format(np.mean(p == ytest) * 100))

input('Program paused. Press ENTER to continue')

# ===================== Part 5: Top Predictors of Spam =====================
# Identify words with highest weights in the classifier

vocab_list = pe.get_vocab_list() # Get the word list
indices = np.argsort(clf.coef_).flatten()[::-1] # Sort the weights in descending order
print(indices)

# Display the top predictors of spam
for i in range(15):
    print('{} ({:0.6f})'.format(vocab_list[indices[i]], clf.coef_.flatten()[indices[i]]))

input('ex6_spam Finished. Press ENTER to exit')
