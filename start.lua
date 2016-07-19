require 'paths'
require 'loadcaffe'
require 'nn'
require 'cunn'
require 'cudnn'		-- gpu mode
require 'image'		-- rescaling
require 'optim'		-- confusion matrix, sgd
require 'gnuplot'

paths.dofile ('trans.lua')
paths.dofile ('test.lua')

classes = { 'airplane', 'automobile', 'bird', 'cat',
	'deer', 'dog', 'frog', 'horse', 'ship', 'truck' }

-- parse command line options
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-gpu', 1, 'use gpu or not')
cmd:option ('-epc', 60, 'number of epochs')
cmd:option ('-bat', 30, 'size of a minibatch')
cmd:option ('-iter', 49980, 'number of iterations at each training pass')
cmd:option ('-titer', 9990, 'number of iterations at each evaluation pass')
cmd:option ('-mod', 'nil', 'name of the trained model to be loaded')
cmd:text()
opt = cmd:parse(arg)

-- start Timer
timer = torch.Timer ()

-- load VGG 16 network
vgg_net = load_vgg_net ()
--vgg_net:add(nn.Linear(4096, #classes))
--vgg_net:add(nn.LogSoftMax())

if opt.gpu == 1 then	
--	vgg_net:add(nn.Linear(4096, #classes):cuda())
--	vgg_net:add(nn.LogSoftMax():cuda())
	vgg_net:add(nn.Linear(4096, #classes))
	vgg_net:add(nn.LogSoftMax())
--	cudnn.convert (vgg_net, cudnn)
	-- vgg_net.modules[39].weight:uniform(-0.8, 0.8)
	-- vgg_net.modules[39].bias:uniform(-0.8,0.8)
	vgg_net:cuda()
	print (vgg_net)
else
	vgg_net:add(nn.Linear(4096, #classes))
	vgg_net:add(nn.LogSoftMax())
end

print ('[loading vgg_net] time elapse: ' .. timer:time().real)

if opt.mod then
	-- load_model (opt.mod, vgg_net)
	vgg_net = torch.load (opt.mod)
	print (vgg_net)
end

-- load datasets
load_data (vgg_net)

-- test(vgg_net)
