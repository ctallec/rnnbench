local paths = require 'paths'
local RNNCore = require 'models.rnnCore'

local cmd = torch.CmdLine()
cmd:option('-hiddenSize', 1000, 'hidden size')
cmd:option('-gpu', 0, 'number of gpu')
local opt = cmd:parse(arg)

local data_dir = 'dataset/ptb'
local train_file = paths.concat(data_dir, 'train.t7')
local valid_file = paths.concat(data_dir, 'valid.t7')
local vocab_file = paths.concat(data_dir, 'vocab.t7')

local train_data = torch.load(train_file)
local valid_data = torch.load(valid_file)

local train_size = train_data:size(1)
local valid_size = valid_data:size(1)

local rnncore = RNNCore{rnnType='lstm', hiddenSize=opt.hiddenSize}
local rnn = rnncore:buildRNN()
