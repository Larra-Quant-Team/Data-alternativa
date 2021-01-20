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

#user = os.getlogin()
# Por alguna razón lee mal el url_base original
dicc = pd.read_excel('url_base2.xlsx',
                     index_col='Ticker CIQ').squeeze()
dicc = dicc.loc[dicc.notna()]

dicc = dicc.to_dict()
data = pd.Series(index=dicc.keys(), name=datetime.today())
for tick, url in dicc.items():
    print(f'Tomando datos de {url}')
    data[tick] = met.alexaGet(url)

data_hist = met.load_obj( 'data_diaria_alexa')
data = data_hist.merge(data, how='outer', right_index=True,
                       left_index=True)
print(data.shape)
fig, axes = plt.subplots(nrows=66, ncols=1, figsize=(30, 260))
plt.tight_layout()
data.T.plot(subplots=True, ax=axes)

#fig = data.T.plot(subplots=True, figsize=(20, 16), layout=(67,1)).get_figure()
fig.savefig('graficos.pdf')

met.save_obj(data,  'data_diaria_alexa')

mail_sender = mailer_quant.Mailer()
sleep(1)
mail_sender.create_message("graficos.pdf")
mails = ["fpaniagua@larrainvial.com"]
for mail in mails:
    mail_sender.send_message(mail)