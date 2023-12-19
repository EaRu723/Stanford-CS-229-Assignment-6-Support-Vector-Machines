import numpy as np
import re
import nltk, nltk.stem.porter


def process_email(email_contents):
    # Load the vocabulary list
    vocab_list = get_vocab_list()

    # Initalize an array to hold indices of words found in the email
    word_indices = np.array([], dtype=np.int64)

    # ===================== Preprocess Email =====================
    # Lowercasing, replaing HTML tags, numbers, URLs, email addresses, and dollar signs

    email_contents = email_contents.lower()

    email_contents = re.sub('<[^<>]+>', ' ', email_contents)

    # Any numbers get replaced with the string 'number'
    email_contents = re.sub('[0-9]+', 'number', email_contents)

    # Anything starting with http or https:// replaced with 'httpaddr'
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)

    # Strings with "@" in the middle are considered emails --> 'emailaddr'
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)

    # The '$' sign gets replaced with 'dollar'
    email_contents = re.sub('[$]+', 'dollar', email_contents)

    # ===================== Tokenize Email =====================

    # Output the processed email
    print('==== Processed Email ====')

    #Initialize the stemmer for word stemming
    stemmer = nltk.stem.porter.PorterStemmer()

    print('email contents : {}'.format(email_contents))


    # Tokenize the email into individual word tokens
    tokens = re.split('[@$/#.-:&*+=\[\]?!(){\},\'\">_<;% ]', email_contents)

    # Remove any non alphanumeric characters and perform stemming
    for token in tokens:
        token = re.sub('[^a-zA-Z0-9]', '', token)
        token = stemmer.stem(token)

        # Ignore empty tokens
        if len(token) < 1:
            continue

        # Add the index of the token to word_indices if it exists in vocab_list
        for i in range(1, len(vocab_list) + 1):
            if vocab_list[i] == token: 
                word_indices = np.append(word_indices, i)

        # Print the token
        print(token)

    print('==================')

    return word_indices


def get_vocab_list():
    # Initialize an empty dictionary for the vocabulary
    vocab_dict = {}
    with open('vocab.txt') as f:
        for line in f:
            # Split each line into value (index) and key (word)
            (val, key) = line.split()
            vocab_dict[int(val)] = key

    return vocab_dict
