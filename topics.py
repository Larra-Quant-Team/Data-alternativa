# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:27:57 2019

@author: Aback
"""

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans


def lda_topics(X, k=3, maxiter=500):
    '''
    Discover topics in a corpus with the LatentDirichlet Allocation method.
    '''
    lda = LatentDirichletAllocation(n_components=k, max_iter=maxiter,
                                    learning_method='online',
                                    learning_offset=50., random_state=0)
    lda.fit(X)
    return lda


def kmeans_topics(X, k=3, maxiter=500):
    '''
    Discover topics in a corpus with the KMeans method.
    '''
    model = KMeans(n_clusters=k, init='k-means++', max_iter=maxiter, n_init=10)
    model.fit(X)
    return model
