-- load data
function load_data (net)
	if (not paths.filep("cifar-10-torch.tar.gz")) then
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

	training (net, trainset.data, trainset.labels)
	test (net, testset.data, testset.labels)
end


-- resize loaded cifar-10 (32 x 32) to 224 x 224
function training (vgg_net, image_set, image_labels)
	print ('Start training......................................................')

	if opt.gpu == 1 then
		c_image_labels = image_labels:cuda()
		criterion = nn.ClassNLLCriterion():cuda()
	else
		criterion = nn.ClassNLLCriterion()
	end
	
	--learning_rate = 0.00000000001
	confusion = optim.ConfusionMatrix(10)

--	for i = 1, image_set:size(1) do
	for i = 1, 6000 do
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

		-- save
		if i == 100 then
			base_img:div(256)
			res_img:div(256)
			image.save ("base" .. i .. ".png", base_img)
			image.save ("res" .. i .. ".png", res_img)
			image.save ("mod" .. i .. ".png", mod_img)
		end

		params, grad_params = vgg_net:getParameters()
--		optim_state = { learning_rate = 0.00000000001, momentum = 0.9, weight_decay = 5e-4 }
		optim_state = { learning_rate = 0.00000000001 }

		-- training
		if opt.gpu == 1 then
			function feval (params)
				c_mod_img = mod_img:cuda()
				grad_params:zero()

				output = vgg_net:forward (c_mod_img)
				loss = criterion:forward (output, c_image_labels[i])
				dloss_dout = criterion:backward (output, c_image_labels[i])	
				vgg_net:backward (c_mod_img, dloss_dout)
				
				confusion:add (output, image_labels[i])

				return loss, grad_params
			end
--[[
			c_mod_img = mod_img:cuda()
			output = vgg_net:forward (c_mod_img)
			err = criterion:forward (output, image_labels[i])
			df_do = criterion:backward (output, image_labels[i])
		
			vgg_net:zeroGradParameters ()
			vgg_net:backward (c_mod_img, df_do)
			vgg_net:updateParameters (learning_rate)

			confusion:add (output, image_labels[i])
]]--			
		else
			output = vgg_net:forward (mod_img)
			err = criterion:forward (output, image_labels[i])
--			f = f + err
			df_do = criterion:backward (output, image_labels[i])
		
	--		vgg_net:zeroGradParameters ()
			vgg_net:backward (mod_img, df_do)
	--		vgg_net:updateParameters (learning_rate)

			confusion:add (output, image_labels[i])
		end

		-- optimization (SGD) -- only for training
		optim.sgd (feval, params, optim_state)
		
--		print (dloss_dout)
--		print (output:double())		
	
		print (i)
		if i % 50 == 0 then
			print ('................................................................' .. i)
--			print (params)
--			print (grad_params)
		--	print ('pred: ', output:max(1))
--			print ('loss: ', loss)
			print (confusion)
			confusion:zero()
		end
	end
end

function test (vgg_net, image_set, image_labels)
	print ('Start testing...........................................................')
	if opt.gpu == 1 then
		c_image_labels = image_labels:cuda()
	end
	
	confusion = optim.ConfusionMatrix(10)

--	for i = 1, image_set:size(1) do
	for i = 1, 5000 do
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

		-- test
		if opt.gpu == 1 then
			c_mod_img = mod_img:cuda()
			output = vgg_net:forward (c_mod_img)
			confusion:add (output, image_labels[i])
		else
			output = vgg_net:forward (mod_img)
			confusion:add (output, image_labels[i])
		end
		
		if i % 50 == 0 then
			print ('..............................................................' .. i)
			print (confusion)
		end
	end
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
