#!/bin/bash

cd /home/zainul/abdza/findvwap
source venv/bin/activate
/home/zainul/abdza/findvwap/get_history.py
/home/zainul/abdza/findvwap/cleanupraw.py
/home/zainul/abdza/findvwap/analyze_raw.py
/home/zainul/abdza/findvwap/get_history.py
/home/zainul/abdza/findvwap/cleanupraw.py
/home/zainul/abdza/findvwap/analyze_raw.py
/home/zainul/abdza/findvwap/backtest.py
/home/zainul/abdza/findvwap/ml.py
