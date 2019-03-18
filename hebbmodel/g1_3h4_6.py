import torch.nn as nn
import params as P
import utils
import basemodel.model3
import hebbmodel.top3


class Net(nn.Module):
	# Layer names
	CONV4 = 'conv4'
	BN4 = 'bn4'
	CONV_OUTPUT = BN4  # Symbolic name for the last convolutional layer providing extracted features
	FC5 = 'fc5'
	BN5 = 'bn5'
	FC6 = 'fc6'
	CLASS_SCORES = FC6  # Symbolic name of the layer providing the class scores as output
	
	NET_G1_3_PATH = P.PROJECT_ROOT + '/results/gdes/config_3l/save/model0.pt'
	NET_H4_6_PATH = P.PROJECT_ROOT + '/results/hebb/top3/save/model0.pt'
	
	def __init__(self, input_shape=P.INPUT_SHAPE):
		super(Net, self).__init__()
		
		# Shape of the tensors that we expect to receive as input
		self.input_shape = input_shape
		
		# Here we define the layers of our network
		self.net_g1_3 = basemodel.model3.Net(input_shape)
		loaded_model = utils.load_dict(self.NET_G1_3_PATH)
		if loaded_model is not None: self.net_g1_3.load_state_dict(loaded_model)
		self.net_g1_3.to(P.DEVICE)
		self.net_h4_6 = hebbmodel.top3.Net(utils.get_output_fmap_shape(self.net_g1_3, basemodel.model3.Net.BN3))
		loaded_model = utils.load_dict(self.NET_H4_6_PATH)
		if loaded_model is not None: self.net_h4_6.load_state_dict(loaded_model)
		self.net_h4_6.to(P.DEVICE)
	
	# Here we define the flow of information through the network
	def forward(self, x):
		return self.net_h4_6(self.net_g1_3(x)[basemodel.model3.Net.BN3])
