# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 18:13:31 2020

@author: ASUS
"""

import metodos_wt as mwt
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt


transcript_dict = mwt.load_obj('transcript_bbd')
count_dict = {}

stemmer = nltk.stem.SnowballStemmer('english')
sw = nltk.corpus.stopwords.words('english')
sw_extra = ['&', 'call', 'also', 'year', 'and', 'we', 'so', 'this']
sw += sw_extra
vectorizer = CountVectorizer(stop_words='english')

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
    cols_orden = df.columns.sort_values()
    df = df[cols_orden]
    count_dict[key] = df

# Jugamos con la data
comp_list = ['NasdaqGS:MELI', 'BOVESPA:MGLU3', 'BOVESPA:BTOW3',
             'BOVESPA:LAME4', 'SNSE:FALABELLA', 'BOVESPA:LREN3',
             'BOVESPA:VVAR3', 'BMV:LIVEPOL C-1', 'SNSE:RIPLEY',
             'SNSE:TRICOT', 'BOVESPA:VALE3', 'BOVESPA:PETR4',
             'BMV:AMX L', 'BMV:WALMEX *', 'BOVESPA:ITUB4', 'BOVESPA:ABEV3']

cv_df = {}
q = '2018Q2'
for tick in comp_list:
    try:
        df_t = count_dict[tick].copy()
        df_t.columns = df_t.columns.to_period('Q')
        serie = df_t[q]
    except KeyError:
        continue
    if isinstance(serie, pd.DataFrame):
        serie = serie.iloc[:, 0]

    cv_df[tick] = serie

cv_df = pd.DataFrame(cv_df).fillna(0)

distance_matrix = pd.DataFrame(index=comp_list, columns=comp_list)

for col in cv_df.columns:
    for ind in cv_df.columns:
        if col <= ind:
            dist = np.sqrt(((cv_df[ind] - cv_df[col])**2).sum())
            distance_matrix.loc[col, ind] = dist
            distance_matrix.loc[ind, col] = dist
distance_matrix = distance_matrix.dropna(axis=1, how='all')
distance_matrix = distance_matrix.dropna(axis=0, how='all')

hierarchy_linkage = linkage(distance_matrix)
ct = 0.5 * max(hierarchy_linkage[:,2])
fig, axes = plt.subplots(1, figsize=(12, 20))
dend = dendrogram(hierarchy_linkage, p=70, truncate_mode='lastp',
                  labels=list(distance_matrix.index), leaf_rotation=0, leaf_font_size=20,
                  color_threshold=ct, orientation='right')
fig.show()

