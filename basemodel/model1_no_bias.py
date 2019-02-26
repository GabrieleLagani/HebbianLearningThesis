import torch.nn as nn
import torch.nn.functional as F
import params as P
import utils


class Net(nn.Module):
	# Layer names
	CONV1 = 'conv1'
	RELU1 = 'relu1'
	POOL1 = 'pool1'
	BN1 = 'bn1'
	CONV_OUTPUT = BN1  # Symbolic name for the last convolutional layer providing extracted features
	FLAT = 'flat'
	FC2 = 'fc2'
	CLASS_SCORES = FC2  # Symbolic name of the layer providing the class scores as output
	
	def __init__(self, input_shape=P.INPUT_SHAPE):
		super(Net, self).__init__()
		
		# Shape of the tensors that we expect to receive as input
		self.input_shape = input_shape
		
		# Here we define the layers of our network
		
		# First convolutional layer
		self.conv1 = nn.Conv2d(3, 96, 5, bias=False)  # 3 input channels, 96 output channels, 5x5 convolutions
		self.bn1 = nn.BatchNorm2d(96, affine=False)  # Batch Norm layer
		
		self.conv_output_size = utils.shape2size(utils.get_conv_output_shape(self))
		
		# FC Layers
		self.fc2 = nn.Linear(self.conv_output_size, P.NUM_CLASSES, bias=False)  # conv_output_size-dimensional input, 10-dimensional output (one per class)
	
	# This function forwards an input through the convolutional layers and computes the resulting output
	def get_conv_output(self, x):
		# Layer 1: Convolutional + ReLU activations + 2x2 Max Pooling + Batch Norm
		conv1_out = self.conv1(x)
		relu1_out = F.relu(conv1_out)
		pool1_out = F.max_pool2d(relu1_out, 2)
		bn1_out = self.bn1(pool1_out)
		
		# Build dictionary containing outputs of each layer
		conv_out = {
			self.CONV1: conv1_out,
			self.RELU1: relu1_out,
			self.POOL1: pool1_out,
			self.BN1: bn1_out
		}
		return conv_out
	
	# Here we define the flow of information through the network
	def forward(self, x):
		# Compute the output feature map from the convolutional layers
		out = self.get_conv_output(x)
		
		# Stretch out the feature map before feeding it to the FC layers
		flat = out[self.CONV_OUTPUT].view(-1, self.conv_output_size)
		
		# Linear FC layer, outputs are the class scores
		fc2_out = self.fc2(flat)
		
		# Build dictionary containing outputs from convolutional and FC layers
		out[self.FLAT] = flat
		out[self.FC2] = fc2_out
		return out
