import time
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.nn as nn

import params as P
import utils
import data
from evaluate import launch_experiment, load_models, eval_pass, eval_batch


# Perform a training pass of a model over a dataset and compute training error.
def train_pass(net, dataset, config, pre_net=None, criterion=None, optimizer=None):
	net.train()
	
	# Variables for keeping track of training progress
	total = 0  # total number of samples processed so far
	acc = None # training accuracy
	
	for batch in dataset:
		# Process batch and count number of hits and total number of samples in the batch
		batch_hits, batch_count, loss = eval_batch(net, batch, config, pre_net, criterion)
		
		# Update statistics
		total += batch_count
		batch_acc = batch_hits / batch_count
		if acc is None: acc = batch_acc
		else: acc = 0.1 * batch_acc + 0.9 * acc # Exponential running average of accuracy during epoch
		
		if optimizer is not None:
			# Update weights
			optimizer.zero_grad()  # Zero out accumulated gradients
			loss.backward()  # Backward step (compute gradients)
			optimizer.step()  # Optimize (update weights)
		
		# Estimate training progress roughly every 5000 samples (or if this is the last batch)
		if total % 5000 < config.BATCH_SIZE or total == config.VAL_SET_SPLIT:
			print("Epoch progress: " + str(total) + "/" + str(config.VAL_SET_SPLIT) + " processed samples")
	
	return acc


# Train the network specified in the configuration on the CIFAR10 train set
def run_train_iter(config, iter_id):
	if config.CONFIG_FAMILY == P.CONFIG_FAMILY_HEBB: torch.set_grad_enabled(False)
	
	# Seed rng
	torch.manual_seed(iter_id)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	
	# Load datasets
	print("Preparing dataset manager...")
	dataManager = data.DataManager(config)
	print("Dataset manager ready!")
	print("Preparing training dataset...")
	train_set = dataManager.get_train()
	print("Training dataset ready!")
	print("Preparing validation dataset...")
	val_set = dataManager.get_val()
	print("Validation dataset ready!")
	
	# Prepare network model to be trained
	print("Preparing network...")
	pre_net, net = load_models(config, iter_id, testing=False)
	criterion = None
	optimizer = None
	scheduler = None
	if config.CONFIG_FAMILY == P.CONFIG_FAMILY_GDES:
		# Instantiate optimizer if we are going to train with gradient descent
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(net.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM, weight_decay=config.L2_PENALTY, nesterov=True)
		scheduler = sched.MultiStepLR(optimizer, gamma=config.LR_DECAY, milestones=config.MILESTONES)
	print("Network ready!")
	
	# Train the network
	print("Starting training...")
	train_acc_data = []
	val_acc_data = []
	best_acc = 0.0
	best_epoch = 0
	start_time = time.time()
	for epoch in range(1, config.NUM_EPOCHS + 1):
		# Print overall progress information at each epoch
		utils.print_train_progress(epoch, config.NUM_EPOCHS, time.time()-start_time, best_acc, best_epoch)
		
		# Training phase
		print("Training...")
		train_acc = train_pass(net, train_set, config, pre_net, criterion, optimizer)
		print("Training accuracy: {:.2f}%".format(100 * train_acc))
		
		# Validation phase
		print("Validating...")
		val_acc = eval_pass(net, val_set, config, pre_net)
		print("Validation accuracy: {:.2f}%".format(100 * val_acc))
		
		# Update training statistics and saving plots
		train_acc_data += [train_acc]
		val_acc_data += [val_acc]
		utils.save_figure(train_acc_data, val_acc_data, config.ACC_PLT_PATH[iter_id])
		
		# If validation accuracy has improved update best model
		if val_acc > best_acc:
			print("Top accuracy improved! Saving new best model...")
			best_acc = val_acc
			best_epoch = epoch
			utils.save_dict(net.state_dict(), config.MDL_PATH[iter_id])
			if hasattr(net, 'conv1') and net.input_shape == P.INPUT_SHAPE: utils.plot_grid(net.conv1.weight, config.KNL_PLT_PATH[iter_id])
			if hasattr(net, 'fc') and net.input_shape == P.INPUT_SHAPE: utils.plot_grid(net.fc.weight.view(-1, *P.INPUT_SHAPE), config.KNL_PLT_PATH[iter_id])
			print("Model saved!")

		# Update LR scheduler
		if scheduler is not None: scheduler.step()
		

if __name__ == '__main__':
	launch_experiment(run_train_iter)