# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 06:49:47 2020

@author: ASUS
"""
import metodos_wt as mwt
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from datetime import datetime


transcript_dict = mwt.load_obj('transcript_bbd')
count_dict = {}
dif_dict = {}

stemmer = nltk.stem.SnowballStemmer('english')
sw = nltk.corpus.stopwords.words('english')
sw_extra = ['&', 'call', 'also', 'year', 'and', 'we', 'so', 'this']
sw += sw_extra
vectorizer = CountVectorizer(stop_words='english')
desde = datetime(2020, 11, 1)

for key, data in transcript_dict.items():
    if data.shape[0] == 0:
        continue
    print(key)
    data['date'] = pd.to_datetime(data['date'])
    data = data[['date', 'text']].set_index('date').squeeze()

    # Si hay un solo transcript, pasamos
    if not isinstance(data, pd.Series):
        continue

    cv_dict = {}
    for date, text in data.iteritems():
        if not isinstance(text, str):
            continue
        text = mwt.process_raw_text(text, sw, stemmer)
        cv_dict[date] = mwt.count_vect(text, vectorizer)

    df = pd.DataFrame(cv_dict).fillna(0)
    count_dict[key] = df

    if df.columns[-1] > desde:
        dif_dict[key] = df.iloc[:, -1] - df.iloc[:, -2]

dif_dict = pd.DataFrame(dif_dict)
dif_dist = (dif_dict**2).sum()**0.5

max_dif = dif_dist.nlargest(10)
max_dif.plot.bar(color='gray')

for tick in max_dif.index:
    print(f'Principales cambios en {tick}:')
    print(dif_dict[tick].nlargest(10))
    print(dif_dict[tick].nsmallest(10))
    print('/n')




