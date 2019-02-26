import torch.nn as nn
import params as P
import utils


class Net(nn.Module):
	# Layer names
	FC = 'fc'
	CLASS_SCORES = FC # Symbolic name of the layer providing the class scores as output
	
	def __init__(self, input_shape=P.INPUT_SHAPE):
		super(Net, self).__init__()
		
		# Shape of the tensors that we expect to receive as input
		self.input_shape = input_shape
		self.input_size = utils.shape2size(self.input_shape)
		
		# FC Layers
		self.fc = nn.Linear(self.input_size, P.NUM_CLASSES) # input_size-dimensional input, 10-dimensional output (one per class)
	
	# Here we define the flow of information through the network
	def forward(self, x):
		out = {}
		
		# Linear FC layer, outputs are the class scores
		fc_out = self.fc(x.view(-1, self.input_size))
		
		out[self.FC] = fc_out
		return out
