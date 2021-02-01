# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 12:08:43 2020

@author: ASUS
"""
import matplotlib.pyplot as plt
import metodos_wt as met
import pandas as pd
from datetime import datetime
import os
import mailer_quant
from time import sleep
from modules.tables import tables

def create_key(company, url, source, field):
    company[company.isna()] = 'null'
    country = company['Country']
    investible = str(company['Invertible'])
    asset = str(company.name)
    ind_sector = company['Industry_Sector']
    ind_group = company['Industry_Group']
    ind_industry = company['Industry']
    ind_internal = company['Internal_industry']
    ind_esg = company['ESG_Industry']
    
    return '.'.join([country, asset, investible, ind_sector,
                     ind_group, ind_industry, ind_internal, ind_esg,
                     source, field])



# TODO leer información de la compañia directamente de Mongo
# Load Companies
companies = pd.read_excel('Company_Base_Definitivo.xlsx',
                          sheet_name='Compilado', engine='openpyxl')
companies.set_index('ID_Quant', inplace=True)
companies.sort_index(inplace=True)
# Filter companies that aren't investable
companies = companies.loc[companies['Invertible'] == 1]


# Por alguna razón lee mal el url_base original
dicc = pd.read_excel('url_base2.xlsx',
                     index_col='Ticker CIQ',  engine='openpyxl').squeeze()
dicc = dicc.loc[dicc.notna()]

dicc = dicc.to_dict()
data = pd.Series(index=dicc.keys(), name=datetime.today())

for tick, url in dicc.items():
    print(f'Tomando datos de {url}')
    data[tick] = met.alexaGet(url)

data_hist = met.load_obj( 'data_diaria_alexa')
data = data_hist.merge(data, how='outer', right_index=True,
                       left_index=True)

df = []
for tick, url in dicc.items():
    companieInfo = companies.loc[companies["Ticker CIQ"] == tick].iloc[0]
    series = pd.Series(data.loc[tick])
    series.index.rename("Date")
    series.name = create_key(companieInfo, url, "Alexa", "Alexa Traffic Rank")
    df.append(series)
df = pd.concat(df, axis=1)

# Send data to Mongo

eq = tables.WebInfoMaster()
keys = eq.get_keys()
eq.insert(df, keys)