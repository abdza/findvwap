#!/usr/bin/env python
import pandas as pd

datas = pd.read_csv('raw_data.csv')
datas.drop_duplicates(subset=['ticker','date'],inplace=True)
datas.to_csv('raw_data.csv',index=False)
datas = pd.read_csv('raw_data_perc.csv')
datas.drop_duplicates(subset=['ticker','date'],inplace=True)
datas.to_csv('raw_data_perc.csv',index=False)
datas = pd.read_csv('raw_data_perc_marks.csv')
datas.drop_duplicates(subset=['ticker','date'],inplace=True)
datas.to_csv('raw_data_perc_marks.csv',index=False)
