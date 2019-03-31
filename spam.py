# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 15:16:53 2019

@author: pushkarpathak
"""

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import random
import pandas as pd
import numpy as np
df=pd.read_csv(r"C:\Users\pushkarpathak\Downloads\spam.csv",encoding = 'latin-1')
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)
df.rename(columns = {'v1': 'labels', 'v2': 'message'}, inplace = True)
df['label'] = df['labels'].map({'ham': 0, 'spam': 1})
df.drop(['labels'], axis = 1, inplace = True)

ham_list = []
spam_list = []

def create_word_features(words):
    my_dict = dict( [ (word, True) for word in words] )
    return my_dict

for i in range(5571):
    string = df.iloc[i,0]
    words=word_tokenize(string)
    
    if(df.iloc[i,1]==0):
        ham_list.append((create_word_features(words), "ham"))
        
    if(df.iloc[i,1]==1):
        spam_list.append((create_word_features(words), "spam"))

combined_list = ham_list + spam_list
print(len(combined_list))

random.shuffle(combined_list)

training_part = int(len(combined_list) * .7)

print(len(combined_list))

training_set = combined_list[:training_part]

test_set =  combined_list[training_part:]

print (len(training_set))
print (len(test_set))

classifier = NaiveBayesClassifier.train(training_set)
accuracy = nltk.classify.util.accuracy(classifier, test_set)

print("Accuracy is: ", accuracy * 100)

#classifier.show_most_informative_features(200)
print("enter any message to check if it is ham or spam:")
msg = input()

words = word_tokenize(msg)
features = create_word_features(words)
print("Message 3 is :" ,classifier.classify(features))