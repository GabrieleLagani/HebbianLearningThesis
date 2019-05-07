import argparse
import torch

import utils
import data
import params as P
import config as C


# Evaluate a model against a batch of data, compute predicted classes, count the number of correct guesses and the
# total number of samples in the batch and optionally evaluate a loss metric. Return the number of correctly classified
# samples, the total number of samples and optionally the required loss metric
def eval_batch(net, batch, config, pre_net=None, criterion=None):
	inputs, labels = batch  # Get the inputs
	inputs, labels = inputs.to(P.DEVICE), labels.to(P.DEVICE)
	if pre_net is not None: inputs = pre_net(inputs)[config.PRE_NET_OUT]
	if hasattr(net, 'set_teacher_signal') and net.training: net.set_teacher_signal(utils.dense2onehot(labels)) # For hebbian supervised learning
	outputs = net(inputs)[net.CLASS_SCORES]  # Forward step. Take the output from the last layer (class scores)
	if hasattr(net, 'set_teacher_signal'): net.set_teacher_signal(None)  # For hebbian supervised learning
	# Compute predicted classes, count number of correct guesses and update variables for keeping track of accuracy
	# The predicted class is the argmax of the output tensor along dimension 1 (dimension 0 is the batch dimension)
	_, pred = torch.max(outputs, 1)
	batch_hits = (pred == labels).int().sum().item()
	batch_count = labels.size(0)
	
	# Evaluate loss metric if required
	loss = None
	if criterion is not None: loss = criterion(outputs, labels) # compute loss
	
	return batch_hits, batch_count, loss

# Evaluate the accuracy of a model against the specified dataset
def eval_pass(net, dataset, config, pre_net=None):
	net.eval()
	
	# Variables for computing accuracy
	hits = 0  # Number of samples processed so far and correctly classified
	count = 0  # Number of samples processed so far
	
	for batch in dataset:
		# Process batch and count number of hits and total number of samples in the batch
		batch_hits, batch_count, _ = eval_batch(net, batch, config, pre_net)
		hits += batch_hits
		count += batch_count
	
	# Compute validation accuracy
	acc = hits / count
	
	return acc

# Load models for testing or training
def load_models(config, iter_id, testing=True):
	pre_net = None
	if config.PreNet is not None:
		# Load preprocessing network if needed
		pre_net = config.PreNet()
		print("Searching for available saved model for the pre-network...")
		pre_net_state = utils.load_dict(config.PRE_NET_MDL_PATH)
		if pre_net_state is not None:
			pre_net.load_state_dict(pre_net_state)
			print("Pre-network model loaded!")
		else: print("No saved model found for the pre-network, using model initialized from scratch")
		pre_net.to(P.DEVICE)
		for p in pre_net.parameters(): p.requires_grad = False
		pre_net.eval()
	net_input_shape = P.INPUT_SHAPE
	if pre_net is not None: net_input_shape = utils.get_output_fmap_shape(pre_net, config.PRE_NET_OUT)
	net = config.Net(input_shape=net_input_shape)
	if testing:
		# Load network model to be tested
		print("Searching for available saved model...")
		loaded_model = utils.load_dict(config.MDL_PATH[iter_id])
		if loaded_model is not None:
			net.load_state_dict(loaded_model)
			print("Model loaded!")
		else: print("No saved model found, testing network initialized from scratch")
	net.to(P.DEVICE)
	return pre_net, net

# Test network specified in the configuration against the CIFAR10 test set
def run_eval_iter(config, iter_id):
	# Prepare network model to be tested
	print("Loading network models for testing...")
	pre_net, net = load_models(config, iter_id, testing=True)
	print("Model loading completed!")
	
	# Load test dataset
	print("Preparing dataset manager...")
	dataManager = data.DataManager(config)
	print("Dataset manager ready!")
	print("Preparing test dataset...")
	test_set = dataManager.get_test()
	print("Test dataset ready!")
	
	print("Testing...")
	test_acc = eval_pass(net, test_set, config, pre_net)
	print("Accuracy of the network on the test images: {:.2f}%".format(100 * test_acc))
	
	print("Saving test result...")
	utils.update_csv(str(iter_id), test_acc, config.CSV_PATH)
	print("Saved!")


def launch_experiment(run_iter_fn):
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default=P.DEFAULT_CONFIG, choices=C.CONFIGURATIONS.keys(), help='The experiment configuration you want to run.')
	args = parser.parse_args()
	
	config = C.CONFIGURATIONS[args.config]
	
	for iter_id in config.ITERATION_IDS:
		print("\n********    Starting Iteration " + str(iter_id) + "    ********\n")
		run_iter_fn(config, iter_id)
	print("\nFinished!")

if __name__ == '__main__':
	launch_experiment(run_eval_iter)
