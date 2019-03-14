import torch
import torch.nn as nn
import torch.nn.functional as F
import hebbmodel.hebb as H
import params as P
import utils


class Net(nn.Module):
	# Layer names
	FC1 = 'fc1'
	BN1 = 'bn1'
	FC2 = 'fc2'
	BN2 = 'bn2'
	FC3 = 'fc3'
	CLASS_SCORES = FC3  # Symbolic name of the layer providing the class scores as output
	
	def __init__(self, input_shape=P.INPUT_SHAPE):
		super(Net, self).__init__()
		
		# Shape of the tensors that we expect to receive as input
		self.input_shape = input_shape
		if len(input_shape) != 3: self.input_shape = (input_shape[0], 1, 1)
		
		# Here we define the layers of our network
		
		# FC Layers (convolution with kernel size equal to the entire feature map size is like a fc layer)
		
		self.fc1 = H.HebbianMap2d(
			in_channels=self.input_shape[0],
			out_size=(15, 20),
			kernel_size=(self.input_shape[1], self.input_shape[2]),
			out=H.clp_cos_sim2d,
			eta=0.1,
		)  # input_shape-shaped input, 15x20=300 output channels
		self.bn1 = nn.BatchNorm2d(300)  # Batch Norm layer
		
		self.fc2 = H.HebbianMap2d(
			in_channels=300,
			out_size=(15, 20),
			kernel_size=1,
			out=H.clp_cos_sim2d,
			eta=0.1,
		)  # 300-dimensional input, 15x20=300 output channels
		self.bn2 = nn.BatchNorm2d(300)  # Batch Norm layer
		
		self.fc3 = H.HebbianMap2d(
			in_channels=300,
			out_size=P.NUM_CLASSES,
			kernel_size=1,
			competitive=False,
			eta=0.1,
		)  # 300-dimensional input, 10-dimensional output (one per class)
	
	# Here we define the flow of information through the network
	def forward(self, x):
		out = {}
		
		####    Forward stage    ####
		
		# Second Layer: FC with ReLU activations + batch norm
		fc1_out = self.fc1(x.view(-1, *self.input_shape))
		bn1_out = self.bn1(fc1_out)
		
		# Third Layer: FC with ReLU activations + batch norm
		fc2_out = self.fc2(bn1_out)
		bn2_out = self.bn2(fc2_out)
		
		# Fourth Layer: FC, outputs are the class scores
		fc3_out = self.fc3(bn2_out).view(-1, P.NUM_CLASSES)
		
		# Build dictionary containing outputs from convolutional and FC layers
		out[self.FC3] = fc3_out
		
		####   Backward stage   ####
		
		if self.training:
			
			# Layer 2
			bfb_2 = torch.matmul(
					self.fc3.weight.view(self.fc3.weight.size(0), -1).transpose(1, 0), #300x10 matrix
					self.fc3.teacher_signal.unsqueeze(2) # batch of B tensors with 10x1 shape
			) # result: batch of B tensors with shape 300x1, each representing the feedback signal on layer 3
			self.fc2.set_teacher_signal(bfb_2.view(-1, self.fc2.weight.size(0)))
			self.fc2.update(bn1_out)
			
			# Layer 1
			scores_2 = self.fc2.similarity(bn1_out, self.fc2.weight).view(-1, self.fc2.weight.size(0)) * self.fc2.teacher_signal
			winner_mask_2 = (scores_2 == scores_2.max(1, keepdim=True)[0]).float()
			bfb_1 = torch.matmul(
				self.fc2.weight.view(self.fc2.weight.size(0), -1).transpose(1, 0),  # 300x300 matrix
				winner_mask_2.unsqueeze(2)  # batch of B tensors with 300x1 shape
			)  # result: batch of B tensors with shape 300x1, each representing the feedback signal on layer 2
			self.fc1.set_teacher_signal(bfb_1.view(-1, self.fc1.weight.size(0)))
			self.fc1.update(x.view(-1, *self.input_shape))
			
			self.fc2.set_teacher_signal(None)
			self.fc1.set_teacher_signal(None)
		
		return out
	
	# Function for setting teacher signal for supervised hebbian learning
	def set_teacher_signal(self, y):
		self.fc3.set_teacher_signal(y)
