local paths = require 'paths'
local RNNCore = require 'models.rnnCore'

local cmd = torch.CmdLine()
cmd:option('-hiddenSize', 1000, 'hidden size')
cmd:option('-gpu', 0, 'number of gpu')
local opt = cmd:parse(arg)
