#! /bin/bash
source /hyren/python/venv/qingdao/bin/activate
nohup python flask_main.py > output.log 2>&1 &
