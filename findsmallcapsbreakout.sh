#!/bin/bash

source /home/zainul/abdza/findvwap/venv/bin/activate
/home/zainul/abdza/findvwap/gapup.py -i smallstocks.csv -o small_results.csv
deactivate
now=`date +"%Y-%m-%d-%H-%M"`
cp /home/zainul/abdza/findvwap/small_results.csv /home/zainul/abdza/telegram_ai_bot/
/home/zainul/abdza/telegram_ai_bot/smallstocks_message.sh
