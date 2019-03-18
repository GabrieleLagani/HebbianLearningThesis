import torch.nn as nn
import params as P
import utils
import basemodel.model4
import hebbmodel.top4


class Net(nn.Module):
	# Layer names
	FC5 = 'fc5'
	BN5 = 'bn5'
	FC6 = 'fc6'
	CLASS_SCORES = FC6  # Symbolic name of the layer providing the class scores as output
	
	NET_G1_4_PATH = P.PROJECT_ROOT + '/results/gdes/config_4l/save/model0.pt'
	NET_H5_6_PATH = P.PROJECT_ROOT + '/results/hebb/top4/save/model0.pt'
	
	def __init__(self, input_shape=P.INPUT_SHAPE):
		super(Net, self).__init__()
		
		# Shape of the tensors that we expect to receive as input
		self.input_shape = input_shape
		
		# Here we define the layers of our network
		self.net_g1_4 = basemodel.model4.Net(input_shape)
		loaded_model = utils.load_dict(self.NET_G1_4_PATH)
		if loaded_model is not None: self.net_g1_4.load_state_dict(loaded_model)
		self.net_g1_4.to(P.DEVICE)
		self.net_h5_6 = hebbmodel.top4.Net(utils.get_output_fmap_shape(self.net_g1_4, basemodel.model4.Net.BN4))
		loaded_model = utils.load_dict(self.NET_H5_6_PATH)
		if loaded_model is not None: self.net_h5_6.load_state_dict(loaded_model)
		self.net_h5_6.to(P.DEVICE)
	
	# Here we define the flow of information through the network
	def forward(self, x):
		return self.net_h5_6(self.net_g1_4(x)[basemodel.model4.Net.BN4])
