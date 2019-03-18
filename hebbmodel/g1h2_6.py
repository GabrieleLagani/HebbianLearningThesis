import torch.nn as nn
import params as P
import utils
import basemodel.model1
import hebbmodel.top1


class Net(nn.Module):
	# Layer names
	CONV2 = 'conv2'
	BN2 = 'bn2'
	CONV3 = 'conv3'
	POOL3 = 'pool3'
	BN3 = 'bn3'
	CONV4 = 'conv4'
	BN4 = 'bn4'
	CONV_OUTPUT = BN4  # Symbolic name for the last convolutional layer providing extracted features
	FC5 = 'fc5'
	BN5 = 'bn5'
	FC6 = 'fc6'
	CLASS_SCORES = FC6  # Symbolic name of the layer providing the class scores as output
	
	NET_G1_PATH = P.PROJECT_ROOT + '/results/gdes/config_1l/save/model0.pt'
	NET_H2_6_PATH = P.PROJECT_ROOT + '/results/hebb/top1/save/model0.pt'
	
	def __init__(self, input_shape=P.INPUT_SHAPE):
		super(Net, self).__init__()
		
		# Shape of the tensors that we expect to receive as input
		self.input_shape = input_shape
		
		# Here we define the layers of our network
		self.net_g1 = basemodel.model1.Net(input_shape)
		loaded_model = utils.load_dict(self.NET_G1_PATH)
		if loaded_model is not None: self.net_g1.load_state_dict(loaded_model)
		self.net_g1.to(P.DEVICE)
		self.net_h2_6 = hebbmodel.top1.Net(utils.get_output_fmap_shape(self.net_g1, basemodel.model1.Net.BN1))
		loaded_model = utils.load_dict(self.NET_H2_6_PATH)
		if loaded_model is not None: self.net_h2_6.load_state_dict(loaded_model)
		self.net_h2_6.to(P.DEVICE)
	
	# Here we define the flow of information through the network
	def forward(self, x):
		return self.net_h2_6(self.net_g1(x)[basemodel.model1.Net.BN1])
