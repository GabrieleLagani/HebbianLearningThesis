import torch
import torch.nn as nn
import hebbmodel.hebb as H
import params as P


class Net(nn.Module):
	# Layer names
	FC5 = 'fc5'
	BN5 = 'bn5'
	FC6 = 'fc6'
	CLASS_SCORES = FC6  # Symbolic name of the layer providing the class scores as output
	
	def __init__(self, input_shape=P.INPUT_SHAPE):
		super(Net, self).__init__()
		
		# Shape of the tensors that we expect to receive as input
		self.input_shape = input_shape
		if len(input_shape) != 3: self.input_shape = (input_shape[0], 1, 1)
		
		# Here we define the layers of our network
		
		# FC Layers (convolution with kernel size equal to the entire feature map size is like a fc layer)
		
		self.fc5 = H.HebbianMap2d(
			in_channels=self.input_shape[0],
			out_size=(15, 20),
			kernel_size=(self.input_shape[1], self.input_shape[2]),
			out=H.clp_cos_sim2d,
			eta=0.1,
		)  # input_shape-shaped input, 15x20=300 output channels
		self.bn5 = nn.BatchNorm2d(300)  # Batch Norm layer
		
		self.fc6 = H.HebbianMap2d(
			in_channels=300,
			out_size=P.NUM_CLASSES,
			kernel_size=1,
			competitive=False,
			eta=0.1,
		) # 300-dimensional input, 10-dimensional output (one per class)
	
	# Here we define the flow of information through the network
	def forward(self, x):
		out = {}
		
		####    Forward stage    ####
		
		# Layer 5: FC + Batch Norm
		fc5_out = self.fc5.out(x.view(-1, *self.input_shape), self.fc5.weight)
		bn5_out = self.bn5(fc5_out)
		
		# Linear FC layer, outputs are the class scores
		fc6_out = self.fc6(bn5_out).view(-1, P.NUM_CLASSES)
		
		# Build dictionary containing outputs from convolutional and FC layers
		out[self.FC5] = fc5_out
		out[self.BN5] = bn5_out
		out[self.FC6] = fc6_out
		
		####   Backward stage   ####
		
		if self.training:
			# Layer 5
			bfb_5 = torch.matmul(
				self.fc6.weight.view(self.fc6.weight.size(0), -1).transpose(1, 0),  # 300x10 matrix
				self.fc6.teacher_signal.unsqueeze(2)  # batch of B tensors with 10x1 shape
			)  # result: batch of B tensors with shape 300x1, each representing the feedback signal on layer 5
			self.fc5.set_teacher_signal(bfb_5.view(-1, self.fc5.weight.size(0)))
			self.fc5.update(x.view(-1, *self.input_shape))
			
			self.fc5.set_teacher_signal(None)
		
		return out
	
	# Function for setting teacher signal for supervised hebbian learning
	def set_teacher_signal(self, y):
		self.fc6.set_teacher_signal(y)