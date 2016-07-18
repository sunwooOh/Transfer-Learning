-- load data
function load_data (net)
	if not paths.filep("cifar-10-torch.tar.gz") then
	    os.execute('wget http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz')
	    os.execute('tar xvf cifar-10-torch.tar.gz')
	end
	
	trainset = {
		data = torch.Tensor (50000, 32*32*3),
		labels = torch.Tensor (50000)
	}

	for i = 0, 4 do
		subset = torch.load ('cifar-10-batches-t7/data_batch_' .. (i + 1) .. '.t7', 'ascii')
		trainset.data [ { { i*10000 + 1, (i+1)*10000 } } ] = subset.data:t()
		trainset.labels [ { { i*10000 + 1, (i+1)*10000 } } ] = subset.labels
	end

	trainset.labels = trainset.labels + 1

	subset = torch.load('cifar-10-batches-t7/test_batch.t7', 'ascii')
	testset = {
		data = subset.data:t():double(),
		labels = subset.labels[1]:double()
	}
	testset.labels:add(1)

	print ('[loading data] time elapse: ' .. timer:time().real)

	-- preprocess_data (trainset.data, 'train/processed/')
	-- preprocess_data (testset.data, 'test/processed/')

	print ('[preprocessing data] time elapse: ' .. timer:time().real)

	epochs = opt.epc
	batch_size = opt.bat

	timer:reset()
	n_inputs = math.floor (opt.iter / opt.bat)

	losses = {}
	tab = {}
	tr_acctab = {}
	te_acctab = {}

	-- tmp = {}
	-- te_acc = testing (net, testset.labels, 'test/processed/')
	-- for i = 1, #te_acc do
	-- 	tmp[i] = i
	-- end
	-- plot (tmp, te_acc, 'initial acc')

	for i = 1, epochs do
		print ('------------------------------------------------------------------------')
		print ("	[[[ Epoch  " .. i .. ' / ' .. epochs ..' ]]]')
		print ('------------------------------------------------------------------------')

		_, tmp_loss, learn_rate, tr_acc = training (net, trainset.labels, 'train/processed/')
		
		-- t_time_vals = torch.Tensor (time_vals)

		--table.insert (tab, i)
		--plot (torch.Tensor(i), loss_vals:norm(), learn_rate)

		te_acc = testing (net, testset.labels, 'test/processed/')

		for j = 1, n_inputs do
			table.insert (tab, (i-1)*n_inputs+j)
			table.insert (losses, tmp_loss[j])
			table.insert (tr_acctab, tr_acc[j])
			table.insert (te_acctab, te_acc[j])
		end

		loss_vals = torch.Tensor (losses)
		x_vals = torch.Tensor (tab)
		tr_accvals = torch.Tensor (tr_acctab)
		te_accvals = torch.Tensor (te_acctab)

		-- x_size = x_vals:size(1)
		te_size = te_accvals:size(1)
	 	nar_time = x_vals:narrow(1, 1, te_size)

		plot (x_vals, loss_vals, 'loss' .. learn_rate)
		plot (x_vals, tr_accvals, 'training_accuracy')
		plot (nar_time, te_accvals, 'test_accuracy')
	end

	netsav = net:clone('weight', 'bias') 
	torch.save ('vgg_fine_tuned.t7', netsav)


end

function preprocess_data (image_set, fname)
	for i = 1, image_set:size(1) do
		-- scaling
		base_img = image_set:select(1, i)
		res_img = base_img:resize (3,32,32)
		mod_img = image.scale (res_img, 224, 224)

		-- normalizing
		for j = 1, 3 do
			img_mean = mod_img [{ {j}, {}, {} }]:mean()
			img_std = mod_img [{ {j}, {}, {} }]:std()
			mod_img [{ {j}, {}, {} }]:add(-img_mean)
			mod_img [{ {j}, {}, {} }]:div(img_std)
		end

		-- transform: RGB to BGR
		chan_r = mod_img [{ {1}, {}, {} }]
		chan_g = mod_img [{ {2}, {}, {} }]
		chan_b = mod_img [{ {3}, {}, {} }]

		mod_img = torch.cat (torch.cat (chan_b, chan_g, 1), chan_r, 1)

		-- save
		image.save ("data/" .. fname .. i .. ".png", mod_img)
	end
end

-- resize loaded cifar-10 (32 x 32) to 224 x 224
function training (vgg_net, image_labels, fname)
	batch_size = opt.bat
	max_iter = opt.iter
	epochs = opt.epc
	conf_mat = torch.DoubleTensor (10, 10):zero()

	-- file = torch.DiskFile ('train_confusion.txt', 'w')

	print ('Start training......................................................')
	-- file:writeObject ('Start training...')
	print ('................................................................')

	if opt.gpu == 1 then
		-- c_image_labels = image_labels:cuda()
		criterion = nn.ClassNLLCriterion():cuda()
	else
		criterion = nn.ClassNLLCriterion()
	end
	
	-- confusion = optim.ConfusionMatrix(10)

	params, grad_params = vgg_net:getParameters()

	print ("[getParameters] time elapse: " .. timer:time().real)

	time_vals = {}
	loss_vals = {}
	accs = {}
	output = torch.DoubleTensor (batch_size, 10)

	-- randomize inputs and targets
	indices = torch.randperm(50000)

	for i = 1, max_iter, batch_size do
		print ("	[[[ Batch number " .. math.ceil (i/batch_size) .. ' / ' .. opt.iter/batch_size*epochs .. ' ]]]')

		inputs = torch.DoubleTensor (batch_size, 3, 224, 224)
		targets = torch.DoubleTensor (batch_size)

		for bat = i, math.min(max_iter, i+batch_size-1) do
			idx = indices[bat]

			img = image.load ('data/' .. fname .. idx .. '.png', 3, 'double')
			img = img:resize (3, 224, 224)

			if bat%batch_size ~= 0 then
				inputs[bat%batch_size]:copy (img)
				targets[bat%batch_size] = image_labels[idx]
			else
				inputs[batch_size]:copy(img)
				targets[batch_size] = image_labels[idx]
			end

		end

		c_inputs = inputs:cuda ()
		c_targets = targets:cuda ()

--		optim_state = { learning_rate = 0.00000000001, momentum = 0.9, weight_decay = 5e-4 }
		-- optim_state = { learning_rate = 0.000001 }
		-- optim_state = { learningRate = 0.0000000001, weightDecay = 0.0005 }
		weight_decay = 0.005
		learning_rate = 0.005

		-- training
		grad_params:zero()

		-- evaluate function for the entire minibatch
		output = vgg_net:forward (c_inputs)
		loss = criterion:forward (output, c_targets)

		-- estimate df/dW
		dloss_dout = criterion:backward (output, c_targets)
		vgg_net:backward (c_inputs, dloss_dout)

		-- grad_params:div(batch_size) --

		-- gradient clipping
		-- grad_params:div (grad_params:norm ()):mul (5)
		grad_params:clamp(-5, 5) --

		-- -- L2 regularization			-- uncomment to activate
		-- loss = loss + weight_decay * torch.norm (params, 2)^2
		-- grad_params:add (params:clone ():mul (weight_decay))

		-- ratio of weights:update 		-- uncomment to activate
		-- param_scale = params:norm()
		-- update_scale = (grad_params*learning_rate):norm()
		-- print ('Ratio (weights : updates) = ' .. update_scale/param_scale) 

		-- vanilla update the weights
		params:add(grad_params:mul(-learning_rate))
	
		-- measure the accuracy
		-- confusion:batchAdd (output, targets)
		print ("[before measure_acc] time elapse: " .. timer:time().real)
		conf_mat, accuracy = measure_acc (conf_mat, output, targets)

		-- if i % 600 == 0 then
			print ('params norm: ' .. params:norm())
			print ('grad_params norm: ' .. grad_params:norm())
			-- print ('................................................................' .. i+batch_size-1)
			
			-- print (confusion)
			
			-- file:writeObject ('.................................................................' .. i)
			-- file:writeObject (tostring(confusion))
			
			print ("Err : " .. loss)
			print ("Time: " .. timer:time().real)
			print ('Accuracy: ' .. accuracy)

			-- visualization
			table.insert (time_vals, timer:time().real)
			table.insert (accs, accuracy)
			
			bound = 50
			if loss < bound then
				table.insert (loss_vals, loss)
			else
				table.insert (loss_vals, bound)
			end

			-- t_time_vals = torch.Tensor (time_vals)
			-- t_loss_vals = torch.Tensor (loss_vals)
			-- plot (t_time_vals, t_loss_vals, learning_rate)

			print ('................................................................' .. i+batch_size-1)
			-- file:writeObject ("current loss: " .. loss)
		-- end
	end

	-- print (confusion)
	print ('Accuracy: ' .. accuracy)
	print ("Err : " .. loss)
	print ("Time: " .. timer:time().real)

	----------------------------------------------------------------------------------------------
	--			Testing with a small set
	----------------------------------------------------------------------------------------------
	
		-- 	print ('..............................................................')
		-- t_confusion = optim.ConfusionMatrix(10)

		-- -- test
		-- 	output = vgg_net:forward (c_inputs)
		-- 	t_confusion:batchAdd (output, targets)
		
		-- 	print (t_confusion)

	----------------------------------------------------------------------------------------------

	return time_vals, loss_vals, learning_rate, accs
end

function testing (vgg_net, image_labels, fname)
	batch_size = opt.bat
	max_iter = opt.titer

	conf_mat = torch.DoubleTensor (10, 10):zero()
	accs = {}

	-- file = torch.DiskFile ('test_confusion.txt', 'w')
	
	print ('Start testing...........................................................')
	-- file:writeObject ('Start testing...')

	if opt.gpu == 1 then
		c_image_labels = image_labels:cuda()
	end
	
	-- confusion = optim.ConfusionMatrix(10)
	output = torch.DoubleTensor (batch_size, 10)

	indices = torch.randperm(10000)

	-- for i = 1, image_set:size(1) do
	for i = 1, max_iter, batch_size do
		inputs = torch.DoubleTensor (batch_size, 3, 224, 224)
		targets = torch.DoubleTensor (batch_size)

--		for bat = 1, batch_size do
		for bat = i, math.min(max_iter, i+batch_size-1) do
			idx = bat

			img = image.load ('data/' .. fname .. idx .. '.png', 3, 'double')
			img = img:resize (3, 224, 224)
			
			if bat%batch_size ~= 0 then
				inputs[bat%batch_size]:copy (img)
				targets[bat%batch_size] = image_labels[idx]
			else
				inputs[batch_size]:copy(img)
				targets[batch_size] = image_labels[idx]
			end

		end

		c_inputs = torch.CudaTensor (batch_size, 3, 224, 224)
		c_targets = torch.CudaTensor (batch_size)
		c_inputs = inputs:cuda ()
		c_targets = targets:cuda ()
		

		-- test
		output = vgg_net:forward (c_inputs)
		-- confusion:batchAdd (output, targets)
		conf_mat, accuracy = measure_acc (conf_mat, output, targets)

		table.insert (accs, accuracy)

		print ('[ ' .. i+29 .. ' / ' .. max_iter .. ' ] Accuracy: ' .. accuracy)
		print (conf_mat)
	end
	print ('..............................................................')
	-- print (confusion)

	return accs

	-- file:writeObject ('.................................................................')
	
	-- file:close()
end

function measure_acc (mat, output, targets)		-- mat : 10 x 10

	for i = 1, opt.bat do
		max = -math.huge
		for ind = 1, 10 do 	-- for each class
			if output[i][ind] > max then
				max_ind = ind
				max = output[i][ind]
			end
		end

		mat[targets[i]][max_ind] = mat[targets[i]][max_ind] + 1
	end

	correct = 0
	for i = 1, 10 do
		correct = correct + mat[i][i]
	end
	global_correct = correct / mat:sum() * 100

	print ('[measuring accuracy] time elapse: ' .. timer:time().real)

	return mat, global_correct

end

function plot (time, val, fname)
	fpath = 'plots/'

	-- if torch.type (arb) == 'number' then
		-- fname = 'loss' .. learn_rate .. '.png'
		fname = fname .. '.png'

		gnuplot.pngfigure (fpath .. fname)

		gnuplot.plot (time, val, '-')
		-- gnuplot.xlabel ('time (s)')
		gnuplot.xlabel ('Iterations')
		gnuplot.ylabel ('Accuracy')

		gnuplot.plotflush ()
	-- else
	-- 	fname = 'accuracy.png'

	-- 	gnuplot.pngfigure (fpath .. fname)

	-- 	arb_size = arb:size(1)
	-- 	val_size = val:size(1)
	-- 	nar_time = time:narrow(1, 1, arb_size):clone()

	-- 	gnuplot.plot (
	-- 		{ 'training', time, val, '-' },
	-- 		{ 'testing', nar_time:mul(val_size/arb_size), arb, '-' }
	-- 	)
	-- 	gnuplot.xlabel ('iterations')
	-- 	gnuplot.ylabel ('accuracy')
	-- 	gnuplot.plotflush ()
	-- end
end

-- load vgg net with loadcaffe module
function load_vgg_net()
	prototxt_name = 'VGG_ILSVRC_16_layers_deploy.prototxt'
	binary_name = 'VGG_ILSVRC_16_layers.caffemodel'

	if opt.gpu == 1 then
		vgg_net = loadcaffe.load(prototxt_name, binary_name, 'cudnn')
	else
		vgg_net = loadcaffe.load(prototxt_name, binary_name, 'nn')
	end

	vgg_net:remove(vgg_net:size())
	vgg_net:remove(vgg_net:size())

	return vgg_net
end

function isNan (number)
	return tostring(number ~= number)
end