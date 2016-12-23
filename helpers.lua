dbg = require 'debugger'

function table_print (tt, indent, done)
  done = done or {}
  indent = indent or 0
  if type(tt) == "table" then
    local sb = {}
    for key, value in pairs (tt) do
      table.insert(sb, string.rep (" ", indent)) -- indent it
      if type (value) == "table" and not done [value] then
        done [value] = true
        table.insert(sb, "{\n");
        table.insert(sb, table_print (value, indent + 2, done))
        table.insert(sb, string.rep (" ", indent)) -- indent it
        table.insert(sb, "}\n");
      elseif "number" == type(key) then
        table.insert(sb, string.format("\"%s\"\n", tostring(value)))
      else
        table.insert(sb, string.format(
            "%s = \"%s\"\n", tostring (key), tostring(value)))
       end
    end
    return table.concat(sb)
  else
    return tt .. "\n"
  end
end


function get_people(data)
  local mappeople = {}
  for i = 1, #data do
    local p1 = data[i][2]
    local p2 = data[i][3]
    mappeople[p1] = true
    mappeople[p2] = true
  end
  
  local people = {}
  for s,c in pairs(mappeople) do
    table.insert(people,s)
  end
  return people
end
 
function get_relations(data)
  local maprelations = {}
  for i = 1, #data do
    local r = data[i][1]
    maprelations[r] = true
  end  
  local relations = {}
  for s,c in pairs(maprelations) do
    table.insert(relations,s)
  end
  return relations 
end

function one_hot(map, e)
  local onehot = torch.zeros(#map)
  local cnt = 1
  for s,c in pairs(map) do
    if c == e then
      onehot[cnt] = 1
    end
    cnt = cnt + 1
  end
  return onehot
end

function get_class(map, e)
  local onehot = torch.zeros(#map)
  local cnt = 1
  for s,c in pairs(map) do
    if c == e then
      return cnt
    end
    cnt = cnt + 1
  end
  return -1
end


--for i = 0,1000000 do
--  -- random sample
--  idx = math.random(111)  
--  x1 = _data[idx][2]
--  x2 = _data[idx][1]
--  y = _data[idx][3]
--  print (string.format("%d: %s ---- %s ---- %s", i,x1, x2, y))
--  
--  target = torch.Tensor({get_class(people, y)})
--  x1_in = one_hot(people, x1)
--  x2_in = one_hot(relations, x2)
--  
--  -- feed it to the neural network and the criterion
--  _criterion:forward(_model:forward({x1_in, x2_in}), target)
--  -- train over this example in 3 steps
--  -- (1) zero the accumulation of the gradients
--  _model:zeroGradParameters()
--  -- (2) accumulate gradients
--  _model:backward({x1_in, x2_in}, _criterion:backward(_model.output, target))
--  -- (3) update parameters with a 0.01 learning rate
--  _model:updateParameters(0.01)  
--end
 
 