#!/bin/bash

source /home/zainul/abdza/findvwap/venv/bin/activate
/home/zainul/abdza/findvwap/gapup.py
deactivate
now=`date +"%Y-%m-%d-%H-%M"`
cp /home/zainul/abdza/findvwap/results_profitability.csv "/home/zainul/abdza/telegram_ai_bot/result_profitabilty_${now}.csv"
cp /home/zainul/abdza/findvwap/results_profitability.csv /home/zainul/abdza/telegram_ai_bot/
/home/zainul/abdza/telegram_ai_bot/send_message_marks.sh
