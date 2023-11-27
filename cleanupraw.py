import pandas as pd

datas = pd.read_csv('raw_data.csv')
datas.drop_duplicates(subset=['ticker','date'],inplace=True)
# datas = datas.loc[(datas['ticker']=='MSS') & (datas['date']=='2023-11-24')]
# print("Datas:",len(datas))
datas.to_csv('raw_data.csv',index=False)
