import torch.nn as nn
import torch.nn.functional as F
import params as P
import utils


class Net(nn.Module):
	# Layer names
	FLAT = 'flat'
	FC1 = 'fc1'
	RELU1 = 'relu1'
	BN1 = 'bn1'
	FC2 = 'fc2'
	RELU2 = 'relu2'
	BN2 = 'bn2'
	FC3 = 'fc3'
	CLASS_SCORES = FC3  # Symbolic name of the layer providing the class scores as output
	
	def __init__(self, input_shape=P.INPUT_SHAPE):
		super(Net, self).__init__()
		
		# Shape of the tensors that we expect to receive as input
		self.input_shape = input_shape
		self.input_size = utils.shape2size(self.input_shape)
		
		# Here we define the layers of our network
		
		# FC Layers
		self.fc1 = nn.Linear(self.input_size, 300)  # conv_output_size-dimensional input, 300-dimensional output
		self.bn1 = nn.BatchNorm1d(300)  # Batch Norm layer
		self.fc2 = nn.Linear(300, 300)  # 300-dimensional input, 300-dimensional output
		self.bn2 = nn.BatchNorm1d(300)  # Batch Norm layer
		self.fc3 = nn.Linear(300, P.NUM_CLASSES)  # 300-dimensional input, 10-dimensional output (one per class)
	
	# Here we define the flow of information through the network
	def forward(self, x):
		out = {}
		
		# Stretch out the feature map before feeding it to the FC layers
		flat = x.view(-1, self.input_size)
		
		# Second Layer: FC with ReLU activations + batch norm
		fc1_out = self.fc1(flat)
		relu1_out = F.relu(fc1_out)
		bn1_out = self.bn1(relu1_out)
		
		# Third Layer: FC with ReLU activations + batch norm
		fc2_out = self.fc2(bn1_out)
		relu2_out = F.relu(fc2_out)
		bn2_out = self.bn2(relu2_out)
		
		# Fourth Layer: FC, outputs are the class scores
		fc3_out = self.fc3(bn2_out)
		
		# Build dictionary containing outputs from convolutional and FC layers
		out[self.FC3] = fc3_out
		return out
