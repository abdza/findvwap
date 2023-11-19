#!/usr/bin/env python

import json
import csv
from datetime import datetime

# Opening JSON file
f = open('backtest.json')
 
# returns JSON object as 
# a dictionary
data = json.load(f)

for k,v in enumerate(data):
    # print('\n'.join([ k[0] + ' - ' + str(k[1]) for k in sorted(data[k]['prop'].items(),key=lambda x: x[1],reverse=True)]))
    data[k]['day'] = datetime.strptime(data[k]['date'],'%Y-%m-%d').strftime('%A') 
    data[k]['diff_levels'] = '\n'.join([ str(ik) + ' : ' + str(iv) for ik,iv in dict(sorted(data[k]['diff_levels'].items())).items()])
    data[k]['outstanding'] = '\n'.join([ str(ik) + ' : ' + str(iv) for ik,iv in dict(sorted(data[k]['outstanding'].items())).items()])
    # data[k]['price_levels'] = '\n'.join([ str(ik) + ' : ' + str(iv) for ik,iv in data[k]['price_levels'].items()])
    data[k]['top_prop'] = '\n'.join([ ik[0] + ' : ' + str(ik[1]) for ik in sorted(data[k]['top_prop'].items(),key=lambda x: x[1],reverse=True)])
    data[k]['prop'] = '\n'.join([ ik[0] + ' : ' + str(ik[1]) for ik in sorted(data[k]['prop'].items(),key=lambda x: x[1],reverse=True)])
    data[k]['fail_prop'] = '\n'.join([ ik[0] + ' : ' + str(ik[1]) for ik in sorted(data[k]['fail_prop'].items(),key=lambda x: x[1],reverse=True)])

fieldnames = ['day','date', 'top_ticker', 'top_price', 'top_diff','outstanding','diff_levels', 'price_levels','top_prop', 'prop', 'fail_prop']
 
with open('backtest.csv', 'w') as csvf:
    # create the csv writer
    writer = csv.DictWriter(csvf,fieldnames=fieldnames)
    writer.writeheader()
    # write a row to the csv file
    writer.writerows(data)
 
# Closing file
f.close()
