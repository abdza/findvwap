#!/bin/bash

source /home/zainul/abdza/findvwap/venv/bin/activate
/home/zainul/abdza/findvwap/gapup.py
deactivate
now=`date +"%Y-%m-%d-%H-%M"`
cp /home/zainul/abdza/findvwap/results.csv "/home/zainul/abdza/telegram_ai_bot/results_${now}.csv"
cp /home/zainul/abdza/findvwap/results.csv /home/zainul/abdza/telegram_ai_bot/
/home/zainul/abdza/telegram_ai_bot/send_message_marks.sh
