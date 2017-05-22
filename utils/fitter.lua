local classic = require 'classic'
local Fitter = classic.class('Fitter')

function Fitter:_init(opts)
    self.t0 = opts.t0
    self.state_size = opts.state_size
    self.core = opts.core
    self.criterion = opts.criterion
    self.reset = opts.reset
    
    local buffer_dimensions = torch.LongStorage(1+self.state_size:size(1))
    buffer_dimensions[1] = self.t0
    for i=1, self.state_size:size(1) do
        buffer_dimensions[1+i] = self.state_size[i]
    end

    self.buffer = torch.Tensor(buffer_dimensions):zero()
    self.grad_state = torch.Tensor(self.state_size):zero()
end

function Fitter:cuda()
    self.buffer = self.buffer:cuda()
    self.grad_state = self.grad_state:cuda()
    return self
end

function Fitter:fit(x, y)
    if self.reset then
        self.buffer[1]:zero()
    end
    
    self.grad_state:zero()

    local loss = 0

    for i=1, self.t0 do
        local out, state = table.unpack(self.core:forward{x[i], self.buffer[i]})
        loss = loss + criterion:forward(out, y[i])
        if i < self.t0 then
            self.buffer[i]:copy(state)
        end
    end

    for i=self.t0, 1, -1 do
        local out, _ = table.unpack(self.core:forward{x[i], self.buffer[i]})
        criterion:forward(out, y[i])
        local dloss_dout = criterion:backward(out, y[i])
        local _, previous_grad_state = table.unpack(self.core:backward{x[i], self.buffer[i]})
        self.grad_state:copy(previous_grad_state)
    end

    if not self.reset then
        local _, state = table.unpack(self.core:forward{x[self.t0], self.buffer[self.t0]})
        self.buffer[1]:copy(state)
    end
end
