# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:27:33 2019

@author: Aback
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def count_term_model(corpus, maxfeatures=1500, vocabulary=None):
    '''
    Bag-of-words kind of approach, with a CountVectorizer object,
    fit and transform a corpus into a document-term frequency matrix.
    '''
    cv = CountVectorizer(max_features=maxfeatures, vocabulary=vocabulary)
    X = cv.fit_transform(corpus).toarray()
    return cv, X


def tf_idf_model(corpus, maxfeatures=1500, vocabulary=None):
    '''
    Term-frequency-inverse document frequency model, using a TfidfVectorizer
    object, fit and transform a corpus into document-term matrix. This model
    gives less importance to less relevant terms in the documents.
    '''
    vectorizer = TfidfVectorizer(max_features=maxfeatures,
                                 vocabulary=vocabulary)
    X = vectorizer.fit_transform(corpus).toarray()
    return vectorizer, X


def docterm_matrix_toframe(vectorizer, matrix, indexes):

    df = pd.DataFrame(matrix, index=indexes, columns=vectorizer.vocabulary_.keys())

    return df.sort_index()


# Word2Vec