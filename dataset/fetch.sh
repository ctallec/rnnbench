#!/bin/bash
./dataset/clean.sh

cd dataset
cd ptb
wget https://raw.githubusercontent.com/yoonkim/lstm-char-cnn/master/data/ptb/test.txt
wget https://raw.githubusercontent.com/yoonkim/lstm-char-cnn/master/data/ptb/train.txt
wget https://raw.githubusercontent.com/yoonkim/lstm-char-cnn/master/data/ptb/valid.txt
cd ../..

th dataset/process.lua
