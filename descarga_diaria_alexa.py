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

user = os.getlogin()
ppath = 'Data Alternativa'
db_path = f'C:/Users/{user}/larrainvial.com/Equipo Quant - Documentos/Area Estrategias Cuantitativas 2.0/BDD/' + ppath
o_path = db_path
# Por alguna razón lee mal el url_base original
dicc = pd.read_excel(db_path + '\\url_base2.xlsx',
                     index_col='Ticker CIQ').squeeze()
dicc = dicc.loc[dicc.notna()]

dicc = dicc.to_dict()
data = pd.Series(index=dicc.keys(), name=datetime.today())
for tick, url in dicc.items():
    print(f'Tomando datos de {url}')
    data[tick] = met.alexaGet(url)

data_hist = met.load_obj(db_path + '/data_diaria_alexa')
data = data_hist.merge(data, how='outer', right_index=True,
                       left_index=True)
print(data.shape)
fig, axes = plt.subplots(nrows=66, ncols=1, figsize=(30, 260))
plt.tight_layout()
data.T.plot(subplots=True, ax=axes)

#fig = data.T.plot(subplots=True, figsize=(20, 16), layout=(67,1)).get_figure()
fig.savefig('test.pdf')

met.save_obj(data, db_path + 'data_diaria_alexa')
