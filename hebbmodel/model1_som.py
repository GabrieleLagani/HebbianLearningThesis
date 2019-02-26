import torch.nn as nn
import torch.nn.functional as F
import hebbmodel.hebb as H
import params as P
import utils


class Net(nn.Module):
	# Layer names
	CONV1 = 'conv1'
	POOL1 = 'pool1'
	BN1 = 'bn1'
	CONV_OUTPUT = BN1  # Symbolic name for the last convolutional layer providing extracted features
	FC2 = 'fc2'
	CLASS_SCORES = FC2  # Symbolic name of the layer providing the class scores as output
	
	def __init__(self, input_shape=P.INPUT_SHAPE):
		super(Net, self).__init__()
		
		# Shape of the tensors that we expect to receive as input
		self.input_shape = input_shape
		
		# Here we define the layers of our network
		
		# First convolutional layer
		self.conv1 = H.HebbianMap2d(
			in_channels=3,
			out_size=(8, 12),
			kernel_size=5,
			lfb_value=H.HebbianMap2d.LFB_GAUSS,
			out=H.clp_cos_sim2d,
			eta=0.1,
			lr_schedule=H.sched_exp(tau=1000, eta_min=0.01),
			tau=1000
		) # 3 input channels, 8x12=96 output channels, 5x5 convolutions
		self.bn1 = nn.BatchNorm2d(96)  # Batch Norm layer
		
		self.conv_output_shape = utils.get_conv_output_shape(self)
		
		# FC Layers (convolution with kernel size equal to the entire feature map size is like a fc layer)
		
		self.fc2 = H.HebbianMap2d(
			in_channels=self.conv_output_shape[0],
			out_size=P.NUM_CLASSES,
			kernel_size=(self.conv_output_shape[1], self.conv_output_shape[2]),
			competitive=False,
			eta=0.1,
		) # conv_output_shape-shaped input, 10-dimensional output (one per class)
	
	# This function forwards an input through the convolutional layers and computes the resulting output
	def get_conv_output(self, x):
		# Layer 1: Convolutional + 2x2 Max Pooling + Batch Norm
		conv1_out = self.conv1(x)
		pool1_out = F.max_pool2d(conv1_out, 2)
		bn1_out = self.bn1(pool1_out)
		
		# Build dictionary containing outputs of each layer
		conv_out = {
			self.CONV1: conv1_out,
			self.POOL1: pool1_out,
			self.BN1: bn1_out,
		}
		return conv_out
	
	# Here we define the flow of information through the network
	def forward(self, x):
		# Compute the output feature map from the convolutional layers
		out = self.get_conv_output(x)
		
		# Linear FC layer, outputs are the class scores
		fc2_out = self.fc2(out[self.CONV_OUTPUT]).view(-1, P.NUM_CLASSES)
		
		# Build dictionary containing outputs from convolutional and FC layers
		out[self.FC2] = fc2_out
		return out
	
	# Function for setting teacher signal for supervised hebbian learning
	def set_teacher_signal(self, y):
		self.fc2.set_teacher_signal(y)
