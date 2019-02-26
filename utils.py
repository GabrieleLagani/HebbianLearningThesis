import os
import csv
import matplotlib.pyplot as plt
import torch

import params as P


# Compute the shape of the output of the convolutional layers of a network. This is useful to correctly set the size of
# successive FC layers
def get_conv_output_shape(net):
	training = net.training
	net.eval()
	# In order to compute the shape of the output of the network convolutional layers, we can feed the network with
	# a simulated input and return the resulting output shape
	with torch.no_grad(): res = tuple(net.get_conv_output(torch.ones(1, *net.input_shape))[net.CONV_OUTPUT].size())[1:]
	net.train(training)
	return res

# Compute the shape of the output feature map from any layer of a network. This is useful to correctly set the size of
# the layers of successive network branches
def get_output_fmap_shape(net, output_layer):
	training = net.training
	net.eval()
	# In order to compute the shape of the output of the network convolutional layers, we can feed the network with
	# a simulated input and return the resulting output shape
	with torch.no_grad(): res = tuple(net(torch.ones(1, *net.input_shape, device=P.DEVICE))[output_layer].size())[1:]
	net.train(training)
	return res

# Convert tensor shape to total tensor size
def shape2size(shape):
	size = 1
	for s in shape: size *= s
	return size

# Convert dense-encoded vector to one-hot encoded
def dense2onehot(tensor, n=P.NUM_CLASSES):
	return torch.zeros(tensor.size(0), n, device=tensor.device).scatter_(1, tensor.unsqueeze(1), 1)

# Save a dictionary (e.g. representing a trained model) in the specified path
def save_dict(d, path):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	torch.save(d, path)

# Load a dictionary (e.g. representing a traied model) from the specified path
def load_dict(path):
	d = None
	try: d = torch.load(path, map_location='cpu')
	except: pass
	return d

# Return formatted string with time information
def format_time(seconds):
	seconds = int(seconds)
	minutes, seconds = divmod(seconds, 60)
	hours, minutes = divmod(minutes, 60)
	return str(hours) + "h " + str(minutes) + "m " + str(seconds) + "s"

# Print information on the training progress
def print_train_progress(current_epoch, total_epochs, elapsed_time, best_acc, best_epoch):
	print("\nEPOCH " + str(current_epoch) + "/" + str(total_epochs))
	
	elapsed_epochs = current_epoch - 1
	if elapsed_epochs == 0:
		elapsed_time_str = "-"
		avg_epoch_duration_str = "-"
		exp_remaining_time_str = "-"
	else:
		avg_epoch_duration = elapsed_time / elapsed_epochs
		remaining_epochs = total_epochs - elapsed_epochs
		elapsed_time_str = format_time(elapsed_time)
		avg_epoch_duration_str = format_time(avg_epoch_duration)
		exp_remaining_time_str = format_time(remaining_epochs * avg_epoch_duration)
	print("Elapsed time: " + elapsed_time_str)
	print("Average epoch duration: " + avg_epoch_duration_str)
	print("Expected remaining time: " + exp_remaining_time_str)
	
	print("Top accuracy so far: {:.2f}%".format(best_acc * 100) + ", at epoch: " + str(best_epoch))

# Save a figure showing train and validation error statistics in the specified file
def save_figure(train_acc_data, val_acc_data, path):
	graph = plt.axes(xlabel='Epoch', ylabel='Accuracy')
	graph.plot(range(1, len(train_acc_data)+1), train_acc_data, label='Train Acc.')
	graph.plot(range(1, len(val_acc_data)+1), val_acc_data, label='Val. Acc.')
	graph.grid(True)
	graph.legend()
	os.makedirs(os.path.dirname(path), exist_ok=True)
	graph.get_figure().savefig(path, bbox_inches='tight')
	graph.get_figure().clear()
	plt.close(graph.get_figure())

# Function to print a grid of images (e.g. representing learned kernels)
def plot_grid(tensor, path, num_rows=8, num_cols=12):
	#tensor = torch.sigmoid((tensor-tensor.mean())/tensor.std()).permute(0, 2, 3, 1).cpu().detach().numpy()
	tensor = ((tensor - tensor.min())/(tensor.max() - tensor.min())).permute(0, 2, 3, 1).cpu().detach().numpy()
	fig = plt.figure()
	for i in range(tensor.shape[0]):
		ax1 = fig.add_subplot(num_rows,num_cols,i+1)
		ax1.imshow(tensor[i])
		ax1.axis('off')
		ax1.set_xticklabels([])
		ax1.set_yticklabels([])
	plt.subplots_adjust(wspace=0.1, hspace=0.1)
	fig.savefig(path, bbox_inches='tight')
	plt.close(fig)

# Add an entry containing the seed of a training iteration and the test accuracy of the corresponding model to a csv file
def update_csv(iter_id, accuracy, path):
	d = {'iter_id': 'accuracy'}
	try:
		with open(path, 'r') as csv_file:
			reader = csv.reader(csv_file)
			d = dict(reader)
	except: pass
	d[iter_id] = accuracy
	try:
		with open(path, mode='w', newline='') as csv_file:
			writer = csv.writer(csv_file)
			for k, v in d.items(): writer.writerow([k, v])
	except: pass

