import numpy as np


def email_features(word_indices):
    # Total number of words in the dictionary
    n = 1899

    # Initialize a feature vector of size n + 1 with zeros (including an extra element for indexing convenience)
    features = np.zeros(n + 1)

    # Set the positino in the feature vector corresponding to word _indices  to 1
    # This indicates the presence of these words in an email
    features[word_indices -1 ] = 1

    return features
