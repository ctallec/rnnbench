local classic = require 'classic'
local Processer = classic.class("Processer")

function Processer:_init(opts)
    self.base_file = opts.base_file
    self.batch_size = opts.batch_size
end

function Processer:createVocabulary(vocab_file)
    local file = torch.DiskFile(self.base_file, 'r')
    local raw = file:readString("*a")
    file:close()

    local unordered = {}
    for c in raw:gmatch('.') do
        if not unordered[c] then unordered[c] = true end
    end

    self.vocab = {}
    for k, _ in pairs(unordered) do
        self.vocab[#self.vocab+1] = k
    end

    self.inverse_vocab = {}
    for k, v in pairs(self.vocab) do
        self.inverse_vocab[v] = k
    end
    torch.save(vocab_file, self.inverse_vocab)
end

function Processer:processFile(file, save_file)
    local file = torch.DiskFile(file, 'r')
    local raw = file:readString('*a')
    
    local data = torch.ByteTensor(#raw)
    for i=1, #raw do
        data[i] = self.inverse_vocab[raw:sub(i,i)]
    end

    local data_size = data:size(1)
    local data_size = math.floor(data_size / self.batch_size) * self.batch_size
    data = data[{{1, data_size}}]
    data = data:view(self.batch_size, data_size/self.batch_size):t()
    torch.save(save_file, data)
end

-- local datafile = 'dataset/ptb/train.txt'
-- local tensorfile = 'dataset/ptb/train.t7'
-- local batch_size = 64
-- 
-- local proc = Processer({base_file=datafile, batch_size=batch_size})
-- local filestream = torch.DiskFile(datafile, 'r')
-- local raw = filestream:readString('*a')
-- for i=1, batch_size do
--     io.write(raw:sub(i,i))
-- end
-- print("Ok")
-- proc:createVocabulary()
-- proc:processFile(datafile, tensorfile)
-- local data = torch.load(tensorfile)
-- for i=1, batch_size do
--     io.write(proc.vocab[data[i][1]])
-- end

return Processer
