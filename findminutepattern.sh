#!/bin/bash

source /home/zainul/abdza/findvwap/venv/bin/activate
/home/zainul/abdza/findvwap/findpattern.py -v 15m
deactivate
now=`date +"%Y-%m-%d-%H-%M"`
cp /home/zainul/abdza/findvwap/pattern.csv "/home/zainul/abdza/telegram_ai_bot/minute_pattern_${now}.csv"
cp /home/zainul/abdza/findvwap/pattern.csv /home/zainul/abdza/telegram_ai_bot/minute_pattern.csv
cp /home/zainul/abdza/findvwap/raw_data.csv "/home/zainul/abdza/telegram_ai_bot/raw_data_${now}.csv"
cp /home/zainul/abdza/findvwap/raw_data.csv /home/zainul/abdza/telegram_ai_bot/
/home/zainul/abdza/telegram_ai_bot/send_minute_message.sh
