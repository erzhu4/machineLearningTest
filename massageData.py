import nltk
from nltk.corpus import words
# turns scentence into array
from nltk.tokenize import word_tokenize
# remove ed, ing, run -> ran etc
from nltk.stem import WordNetLemmatizer

import numpy as np
import random
from collections import Counter

lemmatizer = WordNetLemmatizer()

def create_lexicon(posFile, negFile):
    lexicon = []
    for fi in [posFile, negFile]:
        with open(fi, 'r', errors='ignore') as f:
            contents = f.readlines()

            for line in contents:
                all_words = word_tokenize(line.lower())
                lexicon += list(all_words)
    
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    # {'the' : 4321, 'bob': 432}
    
    finalLexicon = []
    for w in w_counts:
        finalLexicon.append(w)

    print(len(finalLexicon))
    return finalLexicon


def sample_handling(inputFile, lexicon, classification):
    featureset = []

    with open(inputFile, 'r', errors='ignore') as f:
        contents = f.readlines()

        for line in contents:
            current_words = word_tokenize(line.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]

            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] = 1

            features = list(features)
            featureset.append([features, classification])

    return featureset


def make_featuresets():

    posFile = 'data/myPos.txt'
    negFile = 'data/myNeg.txt'

    lexicon = create_lexicon(posFile, negFile)
    posSamples = sample_handling(posFile, lexicon, [1,0])
    negSamples = sample_handling(negFile, lexicon, [0,1])

    allSamples = posSamples + negSamples
    random.shuffle(allSamples)

    allSamples = np.array(allSamples)

    testing_size = int(0.1*len(allSamples))

    #lets get our training and testing data from these samples

    #num py notation -> take first element of each sub array in the numpy array. (so we want just the feature sets and the pos/neg labels)
    #take all our features up to the last 10%
    train_x = list(allSamples[:,0][:-testing_size])
    train_y = list(allSamples[:,1][:-testing_size])

    # the last 10% we test on
    test_x = list(allSamples[:,0][-testing_size:])
    test_y = list(allSamples[:,1][-testing_size:])

    # print("get the testing data");
    # print(test_x)
    # print(test_y)

    return train_x,train_y,test_x,test_y
