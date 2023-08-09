#!/bin/bash
source venv/bin/activate
./findswing.py
./findvwap.py | less
