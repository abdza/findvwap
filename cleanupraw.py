#!/usr/bin/env python
import pandas as pd
import os

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

datas = pd.read_csv(os.path.join(script_dir,'raw_data.csv'))
datas.dropna(inplace=True)
datas.drop_duplicates(subset=['ticker','date'],inplace=True)
# datas = datas.loc[datas['2 Days Start']!='']
# datas = datas.loc[datas['2 Days Start']!=' ']
# datas = datas.loc[datas['2 Days Start']!=None]
datas.to_csv(os.path.join(script_dir,'raw_data.csv'),index=False)
datas = pd.read_csv(os.path.join(script_dir,'raw_data_perc.csv'))
datas.dropna(inplace=True)
datas.drop_duplicates(subset=['ticker','date'],inplace=True)
datas.to_csv(os.path.join(script_dir,'raw_data_perc.csv'),index=False)
datas = pd.read_csv(os.path.join(script_dir,'raw_data_perc_marks.csv'))
datas.dropna(inplace=True)
datas.drop_duplicates(subset=['ticker','date'],inplace=True)
datas.to_csv(os.path.join(script_dir,'raw_data_perc_marks.csv'),index=False)
