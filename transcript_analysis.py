# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 12:17:36 2021

@author: ASUS
"""
import metodos_wt as mwt
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
import representations
from topics import lda_topics
from lmd_dicts import get_sentiment_dicts

db_path = 'C:\\Users\\ASUS\\larrainvial.com\\Equipo Quant - Documentos\\Area Estrategias Cuantitativas 2.0\\BDD\\Data Alternativa'
o_path = db_path
transcript_dict = mwt.load_obj(db_path + '\\transcript_bbd')
count_dict = {}

cluster_words = pd.read_excel(db_path + '\\transcript_clusters.xlsx')
keep_w = list(cluster_words.stack().values)

stemmer = nltk.stem.SnowballStemmer('english')
sw = nltk.corpus.stopwords.words('english')
sw_extra = ['&', 'call', 'also', 'year', 'and', 'we', 'so', 'this']
sw += sw_extra
vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))

for key, data in transcript_dict.items():
    if data.shape[0] == 0:
        continue
    print(key)
    data['date'] = pd.to_datetime(data['date'])
    data = data.loc[data['type'] == 'Earnings Call']
    data = data[['date', 'text']].set_index('date').squeeze()

    # Si hay un solo transcript, pasamos
    if not isinstance(data, pd.Series):
        continue

    cv_dict = {}
    for date, text in data.iteritems():
        if not isinstance(text, str):
            continue
        text = mwt.process_raw_text(text, sw, stemmer)
        cv_dict[date] = mwt.count_vect(text, vectorizer, keep_w=keep_w)

    df = pd.DataFrame(cv_dict).fillna(0)
    cols_orden = df.columns.sort_values()
    df = df[cols_orden]
    count_dict[key.split(':')[-1]] = df

last_dict = {c: df.iloc[:, -1] for c, df in count_dict.items()}
last_df = pd.DataFrame(last_dict)

# Conteo de palabras en total
w_sum = {k: df.sum(1) for k, df in count_dict.items()}
w_sum = pd.DataFrame(w_sum)
suma = w_sum.sum(1).sort_values(ascending=False)
bi_idx = [i for i in suma.index if len(i.split(' ')) == 2]
bi_suma = suma.loc[bi_idx]
uni_idx = [i for i in suma.index if len(i.split(' ')) == 1]
uni_suma = suma.loc[uni_idx]

# Conteo de palabras por cluster
# Podría hacerse en el for anterior, pero es prob que necesitemos normalizar
# por la suma asi que si usamos ese dato tiene que hacerse despues
clusters_dict = {}
sent_dict, negation_list = get_sentiment_dicts()
sentiment_dict = {}

for c in cluster_words.columns:
    words = cluster_words[c].values[cluster_words[c].notna()]
    cluster_dict = {}
    for k, df in count_dict.items():
        words_t = list(set(words) & set(df.index))
        subset = df.loc[words_t].sum()
        subset.index = [mwt.date_to_q(d) for d in subset.index]
        cluster_dict[k] = subset[~subset.index.duplicated(keep='first')]
        # Análisis de tono
        if c == cluster_words.columns[0]:
            sentiment_dict[k] = mwt.get_sentiment(df, sent_dict,
                                                  negation_list)
    clusters_dict[c] = pd.DataFrame(cluster_dict)

sentiments = list(next(iter(sentiment_dict.values())).columns)
sentiment_dict = {s: pd.DataFrame({k: df[s] for k, df in sentiment_dict.items()})
                  for s in sentiments}
clusters_dict = {**sentiment_dict, **clusters_dict}

# Análisis específico de empresas
company = 'MGLU3'
df = count_dict[company]

plt.style.use('fivethirtyeight')
mwt.plot_dif_evol(df, company)
#plot_word_evol(df, N=10)

# Principales cambios
mwt.main_changes(df, MM_per=4, N=15)

# K empresas más parecidas
mwt.n_nearest_companies(last_df, company, n=15)

a = {}
for c, df in transcript_dict.items():
    a[c] = df['text'].str.lower().str.contains('question and answer').sum()/df.shape[0] 





