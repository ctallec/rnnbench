#!/bin/bash
./dataset/fetch.sh
./models/fetch.sh
wget https://raw.githubusercontent.com/ctallec/bigart/master/test.lua
