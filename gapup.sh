#!/bin/bash

source /home/zainul/abdza/findvwap/venv/bin/activate
/home/zainul/abdza/findvwap/gapup.py
cp /home/zainul/abdza/findvwap/results.csv /home/zainul/abdza/telegram_ai_bot
/home/zainul/abdza/telegram_ai_bot/send_message.sh