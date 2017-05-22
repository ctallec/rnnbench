local paths = require 'paths'
local RNNCore = require 'models.rnnCore'
local Fitter = require 'utils.fitter'
local optim = require 'optim'

local cmd = torch.CmdLine()
cmd:option('-hiddenSize', 1000, 'hidden size')
cmd:option('-gpu', 0, 'number of gpu')
cmd:option('-t0', 100, 'sequence length')
cmd:option('-epochs', 20, 'number of epochs')
local opt = cmd:parse(arg)

local data_dir = 'dataset/ptb'
local train_file = paths.concat(data_dir, 'train.t7')
local valid_file = paths.concat(data_dir, 'valid.t7')
local vocab_file = paths.concat(data_dir, 'vocab.t7')

local save_dir = 'save'
local save_model_file = cmd:string(paths.concat(save_dir, 'model'), opt, 
                                   {gpu=true, epochs=true}) .. '.t7'

local vocab = torch.load(vocab_file)
local train_data = torch.load(train_file)
local valid_data = torch.load(valid_file)

local train_size = train_data:size(1)
local valid_size = valid_data:size(1)
local vocab_size = 0
for _, _ in pairs(vocab) do vocab_size = vocab_size + 1 end
local batch_size = train_data:size(2)

local rnncore = RNNCore{rnnType='lstm', hiddenSize=opt.hiddenSize, vocabSize=vocab_size}
local rnn = rnncore:buildCore()

if opt.gpu > 0 then
    require 'cunn'
    require 'cutorch'
    require 'cudnn'

    rnn = rnn:cuda()
end

local params, gradParams = rnn:getParameters()
local criterion = nn.ClassNLLCriterion()

local state_size = rnncore:getStateSize()
local state_shape = torch.LongStorage{batch_size, state_size}
local fitter = Fitter{state_size=state_shape, core=rnn, criterion=criterion, t0=opt.t0,
                      reset=true}

local valid_state = torch.Tensor(batch_size, state_size):zero()

local optimState = {
    learningRate=1e-3
}

if opt.gpu > 0 then
    train_data = train_data:cuda()
    valid_data = valid_data:cuda()
    fitter = fitter:cuda()
    valid_state = valid_state:cuda()
    criterion = criterion:cuda()
end

local function train()
    local index, x, y
    local function feval(params_)
        if params_~=params then
            params:copy(parasm_)
        end

        gradParams:zero()

        local loss = fitter:fit(x, y)
        gradParams:clamp(-1, 1)
        return loss, gradParams
    end

    for i=1, 1000 do
        index = torch.random(1, train_size - 100)
        x = train_data[{{index, index + 99},{}}]
        y = train_data[{{index + 1, index + 100},{}}]
        local _, loss = optim.adam(feval, params, optimState)
        io.write('\rBatch: ' .. i .. '/' .. 1000 .. ' -- Error: ' .. loss[1]/opt.t0/math.log(2))
        io.flush()
    end
    io.write("\n")
end

local function eval()
    local cumLoss = 0
    for i=1, math.floor(valid_size/opt.t0) do 
        io.write("\rValidation progress: " .. i .. "/" .. math.floor(valid_size/opt.t0))
        io.flush()
        local loss = 0
        valid_state:zero()
        local index = (i - 1) * opt.t0 + 1
        for k=index, index + opt.t0 - 1 do
            local output, state = table.unpack(rnn:forward{valid_data[k], valid_state})
            loss = loss + criterion:forward(output, valid_data[k+1])
            valid_state:copy(state)
        end
        cumLoss = cumLoss + loss / opt.t0 / math.log(2)
    end
    io.write("\n")
    return cumLoss / math.floor(valid_size/opt.t0)
end

for e=1, opt.epochs do
    print("Training: ")
    train()
    print("Validation: ")
    print(eval())
    torch.save(save_model_file, rnn)
end
