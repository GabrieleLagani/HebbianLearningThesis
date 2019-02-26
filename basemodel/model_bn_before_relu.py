import torch.nn as nn
import torch.nn.functional as F
import params as P
import utils


class Net(nn.Module):
	
	# Layer names
	CONV1 = 'conv1'
	BN1 = 'bn1'
	RELU1 = 'relu1'
	POOL1 = 'pool1'
	CONV2 = 'conv2'
	BN2 = 'bn2'
	RELU2 = 'relu2'
	CONV3 = 'conv3'
	BN3 = 'bn3'
	RELU3 = 'relu3'
	POOL3 = 'pool3'
	CONV4 = 'conv4'
	BN4 = 'bn4'
	RELU4 = 'relu4'
	CONV_OUTPUT = RELU4 # Symbolic name for the last convolutional layer providing extracted features
	FLAT = 'flat'
	FC5 = 'fc5'
	BN5 = 'bn5'
	RELU5 = 'relu5'
	FC6 = 'fc6'
	CLASS_SCORES = FC6 # Symbolic name of the layer providing the class scores as output
	
	def __init__(self, input_shape=P.INPUT_SHAPE):
		super(Net, self).__init__()
		
		# Shape of the tensors that we expect to receive as input
		self.input_shape = input_shape
		
		# Here we define the layers of our network
		
		# First convolutional layer
		self.conv1 = nn.Conv2d(3, 96, 5) # 3 input channels, 96 output channels, 5x5 convolutions
		self.bn1 = nn.BatchNorm2d(96) # Batch Norm layer
		# Second convolutional layer
		self.conv2 = nn.Conv2d(96, 128, 3) # 96 input channels, 128 output channels, 3x3 convolutions
		self.bn2 = nn.BatchNorm2d(128) # Batch Norm layer
		# Third convolutional layer
		self.conv3 = nn.Conv2d(128, 192, 3)  # 128 input channels, 192 output channels, 3x3 convolutions
		self.bn3 = nn.BatchNorm2d(192) # Batch Norm layer
		# Fourth convolutional layer
		self.conv4 = nn.Conv2d(192, 256, 3)  # 192 input channels, 256 output channels, 3x3 convolutions
		self.bn4 = nn.BatchNorm2d(256) # Batch Norm layer
		
		self.conv_output_size = utils.shape2size(utils.get_conv_output_shape(self))
		
		# FC Layers
		self.fc5 = nn.Linear(self.conv_output_size, 300) # conv_output_size-dimensional input, 300-dimensional output
		self.bn5 = nn.BatchNorm1d(300) # Batch Norm layer
		self.fc6 = nn.Linear(300, P.NUM_CLASSES) # 300-dimensional input, 10-dimensional output (one per class)
	
	# This function forwards an input through the convolutional layers and computes the resulting output
	def get_conv_output(self, x):
		# Layer 1: Convolutional + Batch Norm + ReLU activations + 2x2 Max Pooling
		conv1_out = self.conv1(x)
		bn1_out = self.bn1(conv1_out)
		relu1_out = F.relu(bn1_out)
		pool1_out = F.max_pool2d(relu1_out, 2)
		
		# Layer 2: Convolutional + Batch Norm + ReLU activations
		conv2_out = self.conv2(pool1_out)
		bn2_out = self.bn2(conv2_out)
		relu2_out = F.relu(bn2_out)
		
		# Layer 3: Convolutional + Batch Norm + ReLU activations + 2x2 Max Pooling
		conv3_out = self.conv3(relu2_out)
		bn3_out = self.bn3(conv3_out)
		relu3_out = F.relu(bn3_out)
		pool3_out = F.max_pool2d(relu3_out, 2)
		
		# Layer 4: Convolutional + Batch Norm + ReLU activations
		conv4_out = self.conv4(pool3_out)
		bn4_out = self.bn4(conv4_out)
		relu4_out = F.relu(bn4_out)
		
		# Build dictionary containing outputs of each layer
		conv_out = {
			self.CONV1: conv1_out,
			self.BN1: bn1_out,
			self.RELU1: relu1_out,
			self.POOL1: pool1_out,
			self.CONV2: conv2_out,
			self.BN2: bn2_out,
			self.RELU2: relu2_out,
			self.CONV3: conv3_out,
			self.BN3: bn3_out,
			self.RELU3: relu3_out,
			self.POOL3: pool3_out,
			self.CONV4: conv4_out,
			self.BN4: bn4_out,
			self.RELU4: relu4_out
		}
		return conv_out
	
	# Here we define the flow of information through the network
	def forward(self, x):
		# Compute the output feature map from the convolutional layers
		out = self.get_conv_output(x)
		
		# Stretch out the feature map before feeding it to the FC layers
		flat = out[self.CONV_OUTPUT].view(-1, self.conv_output_size)
		
		# Fifth Layer: FC  + Batch Norm with ReLU activations
		fc5_out = self.fc5(flat)
		bn5_out = self.bn5(fc5_out)
		relu5_out = F.relu(bn5_out)
		
		# Sixth Layer: dropout + FC, outputs are the class scores
		fc6_out = self.fc6(F.dropout(relu5_out, p=0.5, training=self.training))
		
		# Build dictionary containing outputs from convolutional and FC layers
		out[self.FLAT] = flat
		out[self.FC5] = fc5_out
		out[self.BN5] = bn5_out
		out[self.RELU5] = relu5_out
		out[self.FC6] = fc6_out
		return out
