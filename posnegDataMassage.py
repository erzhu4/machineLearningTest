import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import nltk
from nltk.corpus import words
# turns scentence into array
from nltk.tokenize import word_tokenize
# remove ed, ing, run -> ran etc
from nltk.stem import WordNetLemmatizer

import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 1000000

def create_lexicon(pos, neg):
    lexicon = []
    allWords = words.words()
    # for each file (pos and neg)
    for fi in [pos, neg]:
        # open file and read
        with open(fi, 'r', errors='ignore') as f:
            contents = f.readlines()
            # each line in the file
            for line in contents[:hm_lines]:
                all_words = word_tokenize(line.lower())
                lexicon += list(all_words)

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    # w_counts looks like {'the': 132, 'and': 888, 'derp': 3242, ....}

    finalLexicon = []
    for w in w_counts:
        # 'filter out super common words and super rare and special words'??
        if 1000 > w_counts[w] > 50 and w in allWords:
            finalLexicon.append(w)

    print('Size of the lexicon is: ')
    print(len(finalLexicon))
    return finalLexicon



def sample_handling(sample, lexicon, classification):
    featureset = []

    with open(sample, 'r', errors='ignore') as f:
        contents = f.readlines()
        for line in contents[:hm_lines]:
            current_words = word_tokenize(line.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon)) #creates a [0,0,0,0...] of length lexicon
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] = 1
            features = list(features)
            featureset.append([features, classification])
            # featureset looks like [
            #     [ [0,1,0,0,1], [0,1] ],
            #     [ [0,1,0,0,1], [1,0] ]
            # ]
    return featureset


def create_featuresets_and_labels(pos,neg,test_size=0.05):
    lexicon = create_lexicon(pos,neg)
    features = []
    features += sample_handling(pos, lexicon, [1,0])
    features += sample_handling(neg, lexicon, [0,1])
    random.shuffle(features)

    # turn into a numpy array
    features = np.array(features)

    testing_size = int(test_size*len(features))

    #lets get our training and testing data from these samples

    #num py notation -> take first element of each sub array in the numpy array. (so we want just the feature sets and the pos/neg labels)
    #take all our features up to the last 10%
    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])

    # the last 10% we test on
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])

    # print("get the testing data");
    # print(test_x)
    # print(test_y)

    return train_x,train_y,test_x,test_y

#running time

# if __name__ == '__main__':
#     train_x,train_y,test_x,test_y = create_featuresets_and_labels('data/pos.txt', 'data/neg.txt')
#     with open('sentiment_set.pickle', 'wb') as f:
#         pickle.dump([train_x, train_y, test_x, test_y], f)







# line = "i am in the car doing the thing that you told me to do and then i will do the other thing"

# t = word_tokenize(line.lower())
# print(t)
# print(Counter(t))