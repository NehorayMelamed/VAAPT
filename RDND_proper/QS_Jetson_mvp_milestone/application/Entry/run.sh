#!/bin/bash
# bisiyata deshamayim
nohup python3 ftp_manager.py & # check nohup.out for debug information
python3 main.py
