require 'torch'
dofile('Csv.lua')
dofile('helpers.lua')
require 'nn'
require 'optim'
require 'string'

_csv = Csv('kinship.csv', "r")
_data = _csv:readall()
_csv:close()

people = get_people(_data)
relations = get_relations(_data)

_model = nn.Sequential()
b1 = nn.Sequential():add(nn.Linear(24,6)):add(nn.ReLU())
b2 = nn.Sequential():add(nn.Linear(12,6)):add(nn.ReLU())
pt = nn.ParallelTable():add(b1):add(b2)
_model:add(pt)
_model:add(nn.JoinTable(1,2))
_model:add(nn.Linear(12,12)):add(nn.ReLU())
_model:add(nn.Linear(12,6)):add(nn.ReLU())
_model:add(nn.Linear(6,24)):add(nn.LogSoftMax())
_confusion = optim.ConfusionMatrix(people)

print(_model)
print (one_hot(people, 'Christopher'))

_criterion = nn.ClassNLLCriterion()
optimState = {
  learningRate = 0.01,
  weightDecay = 0.0001,
  momentum = 0.9  
}
optimMethod = optim.sgd


_parameters, _gradParameters = _model:getParameters()
for t = 1, 50000 do
  idx = math.random(111)  
  x1 = _data[idx][2]
  x2 = _data[idx][1]
  y = _data[idx][3]
  print (string.format("%d: %s ---- %s ---- %s", t,x1, x2, y))
  
  target = torch.Tensor({get_class(people, y)})
  x1_in = one_hot(people, x1)
  x2_in = one_hot(relations, x2)
  -- create closure to evaluate f(X) and df/dX
  local feval = function(x)
     -- get new parameters
     if x ~= _parameters then
        _parameters:copy(x)
     end
     -- reset gradients
     _gradParameters:zero()
     -- evaluate function for complete mini batch
     -- estimate f
      local output = _model:forward({x1_in, x2_in})
      local err = _criterion:forward(output, target)
      -- estimate df/dW
      local df_do = _criterion:backward(output, target)
      _model:backward({x1_in, x2_in}, df_do)     
     -- return err and df/dX
     return err, _gradParameters
  end
  -- optimize on current mini-batch
  optimMethod(feval, _parameters, optimState)
end



for i = 1, 111 do
  -- random sample
  idx =  i 
  x1 = _data[idx][2]
  x2 = _data[idx][1]
  y = _data[idx][3]
  print (string.format("%d: %s ---- %s ---- %s", i,x1, x2, y))
  
  target = torch.Tensor({get_class(people, y)})
  x1_in = one_hot(people, x1)
  x2_in = one_hot(relations, x2)
  
  pred = _model:forward({x1_in, x2_in})
  _,idx_max = torch.max(pred,1)
  print(string.format('target: %d, pred: %d',target[1], idx_max[1]))
  _confusion:add(idx_max[1], target[1])
end

print(_confusion)






