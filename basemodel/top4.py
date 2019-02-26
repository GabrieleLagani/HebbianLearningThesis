import torch.nn as nn
import torch.nn.functional as F
import params as P
import utils


class Net(nn.Module):
	# Layer names
	FLAT = 'flat'
	FC5 = 'fc5'
	RELU5 = 'relu5'
	BN5 = 'bn5'
	FC6 = 'fc6'
	CLASS_SCORES = FC6 # Symbolic name of the layer providing the class scores as output
	
	def __init__(self, input_shape=P.INPUT_SHAPE):
		super(Net, self).__init__()
		
		# Shape of the tensors that we expect to receive as input
		self.input_shape = input_shape
		self.input_size = utils.shape2size(self.input_shape)
		
		# Here we define the layers of our network
		
		# FC Layers
		self.fc5 = nn.Linear(self.input_size, 300) # conv_output_size-dimensional input, 300-dimensional output
		self.bn5 = nn.BatchNorm1d(300) # Batch Norm layer
		self.fc6 = nn.Linear(300, P.NUM_CLASSES) # 300-dimensional input, 10-dimensional output (one per class)
	
	# Here we define the flow of information through the network
	def forward(self, x):
		out = {}
		
		# Stretch out the feature map before feeding it to the FC layers
		flat = x.view(-1, self.input_size)
		
		# Fifth Layer: FC with ReLU activations + batch norm
		fc5_out = self.fc5(flat)
		relu5_out = F.relu(fc5_out)
		bn5_out = self.bn5(relu5_out)
		
		# Sixth Layer: dropout + FC, outputs are the class scores
		fc6_out = self.fc6(F.dropout(bn5_out, p=0.5, training=self.training))
		
		# Build dictionary containing outputs from convolutional and FC layers
		out[self.FLAT] = flat
		out[self.FC5] = fc5_out
		out[self.RELU5] = relu5_out
		out[self.BN5] = bn5_out
		out[self.FC6] = fc6_out
		return out
