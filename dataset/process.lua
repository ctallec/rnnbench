local Processer = require 'dataset.processer'
local paths = require 'paths'

local data_dir = 'dataset/ptb'

local train_file = paths.concat(data_dir, 'train.txt')
local test_file = paths.concat(data_dir, 'test.txt')
local valid_file = paths.concat(data_dir, 'valid.txt')

local train_tensor_file = paths.concat(data_dir, 'train.t7')
local test_tensor_file = paths.concat(data_dir, 'test.t7')
local valid_tensor_file = paths.concat(data_dir, 'valid.t7')
local vocab_file = paths.concat(data_dir, 'vocab.t7')

local batch_size = 64

local proc = Processer({base_file=train_file, batch_size=batch_size})
proc:createVocabulary(vocab_file)
proc:processFile(train_file, train_tensor_file)
proc:processFile(test_file, test_tensor_file)
proc:processFile(valid_file, valid_tensor_file)
