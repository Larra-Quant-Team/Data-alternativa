# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 17:19:27 2020

@author: ASUS
"""

from requests import get
from urllib.parse import urlparse
from urllib.error import HTTPError
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import numpy as np
import pickle as pickle_rick
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import pandas as pd
from time import sleep
import re
import nltk
import json
from xml.etree.ElementTree import ParseError
from datetime import datetime


def similarGet(url):
    domain = '{uri.netloc}'.format(uri=urlparse(url))
    domain = domain.replace("www.", "")
    ENDPOINT = 'https://data.similarweb.com/api/v1/data?domain=' + domain
    resp = Request(ENDPOINT, headers={'User-Agent': 'Mozilla/5.0'})
    resp = urlopen(resp).read().decode('utf8').replace("'", '"')
    
    try:
        return json.loads(resp)
    except json.JSONDecodeError:
        return {}


def alexaGet(url):
    try:
        rank_str = BeautifulSoup(urlopen("https://www.alexa.com/minisiteinfo/" +url),'html.parser').table.a.get_text()
        rank_int = int(rank_str.replace(',', ''))
    except:
        print('Falló alexa')
        return np.nan
    return rank_int


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle_rick.load(f)


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        return pickle_rick.dump(obj, f)


def ciq_login(username, password, silent=False):
    url = 'https://www.capitaliq.com/CIQDotNet/my/dashboard.aspx'
    bot = webdriver.Chrome()
    if silent:
        bot.set_window_position(0, 0)
        bot.set_window_size(0, 0)

    bot.get(url)

    bot.find_element_by_id('username').send_keys(username)
    pwd = bot.find_element_by_id('password')
    pwd.send_keys(password)
    pwd.send_keys(Keys.RETURN)
    sleep(0.5)
    return bot


def get_transcripts(driver, c_id):
    url = 'https://www.capitaliq.com/CIQDotNet/Transcripts/Summary.aspx?CompanyId={}'
    driver.get(url.format(c_id))
    soup = BeautifulSoup(driver.page_source, 'lxml')

    # primera y última fila son ruido
    table = soup.find('table',
                      {'class': 'cTblListBody'}).tbody.find_all('tr')[1:-1]
    table_df = pd.DataFrame(columns=['date', 'title', 'type', 'link', 'text'])

    for i, row in enumerate(table):
        tds = row.find_all('td')
        serie = pd.Series()
        serie['date'] = pd.to_datetime(tds[1].text.strip())
        serie['title'] = tds[2].text.strip()
        serie['type'] = tds[3].text.strip()
        url_t = 'https://www.capitaliq.com' + row.find(href=True)['href']
        serie['link'] = url_t
        if 'void' not in url_t:
            driver.get(url_t)
            soup_t = BeautifulSoup(driver.page_source, 'lxml')
            serie['text'] = soup_t.find('table', {'class': 'cTblListBody'}).text
        table_df.loc[i] = serie


    return table_df


def get_new_transcripts(driver, c_id, last_date):

    url = 'https://www.capitaliq.com/CIQDotNet/Transcripts/Summary.aspx?CompanyId={}'
    driver.get(url.format(c_id))
    soup = BeautifulSoup(driver.page_source, 'lxml')

    # primera y última fila son ruido
    table = soup.find('table',
                      {'class': 'cTblListBody'}).tbody.find_all('tr')[1:-1]
    table_df = pd.DataFrame(columns=['date', 'title', 'type', 'link', 'text'])
    for i, row in enumerate(table):
        tds = row.find_all('td')
        try:
            date = pd.to_datetime(tds[1].text.strip())
        except ParserError:
            date = np.nan
        
        if date <= last_date:
            break

        serie = pd.Series()
        serie['date'] = date
        serie['title'] = tds[2].text.strip()
        serie['type'] = tds[3].text.strip()
        url_t = 'https://www.capitaliq.com' + row.find(href=True)['href']
        serie['link'] = url_t
        if 'void' not in url_t:
            driver.get(url_t)
            soup_t = BeautifulSoup(driver.page_source, 'lxml')
            serie['text'] = soup_t.find('table', {'class': 'cTblListBody'}).text
        table_df.loc[i] = serie

    return table_df

def colores_corporativos(colors=None):

    diccionario = {'red': (204,0,51) ,
                   'light_blue': (110,162,201),
                   'gray': (66,74,82),
                   'green':(55,95,77),
                   'yellow': (195,195,9),
                   'dark_purple': (119,28,95),
                   'blue': (42,83,113),
                   'purple': (159,37,127),
                   'light_yellow': (252,252,196),
                   'light_green': (122,178,153),
                   'light_gray':(135,146,158)
                    }
    
    for key in diccionario:
        diccionario[key] = tuple(v/255 for v in diccionario[key])   
    
    if colors is None:
        return diccionario
    else:
        aux = {col:diccionario[col] for col in colors}
        return aux


def process_raw_text(text, sw, stemmer):

    pattern = re.compile(r'\b(' + r'|'.join(sw) + r')\b\s*')
    # Eliminamos stopwords
    text = pattern.sub('', text)
    # Eliminamos números
    text = re.sub("[0-9]+", "", text)
    # Eliminamos regex
    text = re.sub(r'[^\w]', ' ', text)

    text = stemmer.stem(text)

    return text


def count_vect(text, vectorizer, min_apps=3, keep_w=[]):
    # min_apps: cuantas veces tiene que aparecer el ngram para mantenerlo
    # keep_w palabras que queremos mantener a pesar de que aparecen menos
    counts = vectorizer.fit_transform([text])
    counts = pd.Series(counts.toarray()[0], index=vectorizer.get_feature_names())
    min_apps_idx = counts >= min_apps
    keep_w_idx = [w in keep_w for w in counts.index]
    idx = min_apps_idx | keep_w_idx
    counts = counts.loc[idx]
    return counts / counts.sum()


def date_to_q(date):
    q = date.quarter
    y = date.year
    return datetime(y, q*3, 1)


def get_sentiment(df, sent_dict, negation_list):

    neg_word = list(set(sent_dict['Negative']) & set(df.index))
    pos_word = list(set(sent_dict['Positive']) & set(df.index))
    negation_word = list(set(negation_list) & set(df.index))
    negation_bigrams_pos = sum([[f'{a} {b}' for b in neg_word]
                                for a in negation_word], [])
    negation_bigrams_pos = list(set(negation_bigrams_pos) & set(df.index))
    negation_bigrams_neg = sum([[f'{a} {b}' for b in pos_word]
                                for a in negation_word], [])
    negation_bigrams_neg = list(set(negation_bigrams_neg) & set(df.index))
    sentiment = {}
    sentiment['Negative'] = df.loc[neg_word].sum()
    sentiment['Positive'] = df.loc[pos_word].sum()
    sentiment['Negative'] += 2 * df.loc[negation_bigrams_neg].sum()
    sentiment['Positive'] += 2 * df.loc[negation_bigrams_pos].sum()
    sentiment = pd.DataFrame(sentiment)
    sentiment['Delta'] = sentiment['Positive'] - sentiment['Negative']
    sentiment.index = [date_to_q(d) for d in sentiment.index]

    return sentiment



