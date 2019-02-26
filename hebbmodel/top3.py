import torch
import torch.nn as nn
import hebbmodel.hebb as H
import params as P
import utils


class Net(nn.Module):
	# Layer names
	CONV4 = 'conv4'
	BN4 = 'bn4'
	CONV_OUTPUT = BN4  # Symbolic name for the last convolutional layer providing extracted features
	FC5 = 'fc5'
	BN5 = 'bn5'
	FC6 = 'fc6'
	CLASS_SCORES = FC6  # Symbolic name of the layer providing the class scores as output
	
	def __init__(self, input_shape=P.INPUT_SHAPE):
		super(Net, self).__init__()
		
		# Shape of the tensors that we expect to receive as input
		self.input_shape = input_shape
		
		# Here we define the layers of our network
		
		# Fourth convolutional layer
		self.conv4 = H.HebbianMap2d(
			in_channels=192,
			out_size=(16, 16),
			kernel_size=3,
			out=H.clp_cos_sim2d,
			eta=0.1,
		)  # 192 input channels, 16x16=256 output channels, 3x3 convolutions
		self.bn4 = nn.BatchNorm2d(256)  # Batch Norm layer
		
		self.conv_output_shape = utils.get_conv_output_shape(self)
		
		# FC Layers (convolution with kernel size equal to the entire feature map size is like a fc layer)
		
		self.fc5 = H.HebbianMap2d(
			in_channels=self.conv_output_shape[0],
			out_size=(15, 20),
			kernel_size=(self.conv_output_shape[1], self.conv_output_shape[2]),
			out=H.clp_cos_sim2d,
			eta=0.1,
		)  # conv_output_shape-shaped input, 15x20=300 output channels
		self.bn5 = nn.BatchNorm2d(300)  # Batch Norm layer
		
		self.fc6 = H.HebbianMap2d(
			in_channels=300,
			out_size=P.NUM_CLASSES,
			kernel_size=1,
			competitive=False,
			eta=0.1,
		) # 300-dimensional input, 10-dimensional output (one per class)
	
	# This function forwards an input through the convolutional layers and computes the resulting output
	def get_conv_output(self, x):
		# Layer 4: Convolutional + Batch Norm
		conv4_out = self.conv4(x)
		bn4_out = self.bn4(conv4_out)
		
		# Build dictionary containing outputs of each layer
		conv_out = {
			self.CONV4: conv4_out,
			self.BN4: bn4_out,
		}
		return conv_out
	
	# Here we define the flow of information through the network
	def forward(self, x):
		# Compute the output feature map from the convolutional layers
		out = self.get_conv_output(x)
		
		# Layer 5: FC + Batch Norm
		fc5_out = self.fc5(out[self.CONV_OUTPUT])
		bn5_out = self.bn5(fc5_out)
		
		# Linear FC layer, outputs are the class scores
		fc6_out = self.fc6(bn5_out).view(-1, P.NUM_CLASSES)
		
		# Build dictionary containing outputs from convolutional and FC layers
		out[self.FC5] = fc5_out
		out[self.BN5] = bn5_out
		out[self.FC6] = fc6_out
		return out
	
	# Function for setting teacher signal for supervised hebbian learning
	def set_teacher_signal(self, y):
		self.fc6.set_teacher_signal(y)
		
		if y is None:
			self.conv4.set_teacher_signal(y)
			self.fc5.set_teacher_signal(y)
		else:
			# Extend teacher signal for layer 4 and 5
			l4_knl_per_class = 24
			l5_knl_per_class = 28
			self.conv4.set_teacher_signal(
				torch.cat((
					y.view(y.size(0), y.size(1), 1).repeat(1, 1, l4_knl_per_class).view(y.size(0), -1),
					torch.ones(y.size(0), self.conv4.weight.size(0) - l4_knl_per_class * P.NUM_CLASSES, device=y.device)
				), dim=1)
			)
			self.fc5.set_teacher_signal(
				torch.cat((
					y.view(y.size(0), y.size(1), 1).repeat(1, 1, l5_knl_per_class).view(y.size(0), -1),
					torch.ones(y.size(0), self.fc5.weight.size(0) - l5_knl_per_class * P.NUM_CLASSES, device=y.device)
				), dim=1)
			)
