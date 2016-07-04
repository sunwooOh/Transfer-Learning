require 'paths'
require 'loadcaffe'
require 'nn'
require 'cunn'
require 'cudnn'		-- gpu mode
require 'image'		-- rescaling
require 'optim'		-- confusion matrix, sgd

paths.dofile ('trans.lua')

classes = { 'airplane', 'automobile', 'bird', 'cat',
	'deer', 'dog', 'frog', 'horse', 'ship', 'truck' }

-- parse command line options
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-gpu', 0, 'do not use gpu')
cmd:text()
opt = cmd:parse(arg)

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
	vgg_net:cuda()
	print (vgg_net)
else
	vgg_net:add(nn.Linear(4096, #classes))
	vgg_net:add(nn.LogSoftMax())
end

-- load datasets
load_data (vgg_net)
