Pytorch implementation of a Hebbian-WTA learning algorithm to train
deep convolutional neural networks.
A neural network model is trained on CIFAR10 both using the 
Hebbian-WTA algorithm and SGD in order to compare the results.
Although Hebbian learning is unsupervised, I also implemented a 
technique to train the final linear classification layer using the
Hebbian algorithm in a supervised manner. This is done by applying a 
teacher signal on the final layer that provides the desired output; 
the neurons are then enforced to update their weights in order to 
follow that signal.

In order to launch a training session, type:  
`PYTHONPATH=<project root> python <project root>/train.py --config <config family>/<config name>`  
Where `<config family>` is either `gdes` or `hebb`, depending whether 
you want to run gradient descent or hebbian training, and 
`<config name>` is the name of one of the training configurations in 
the `config.py` file.  
Example:  
`PYTHONPATH=<project root> python <project root>/train.py --config gdes/config_base`  
To evaluate the network on the CIFAR10 test set, type:
`PYTHONPATH=<project root> python <project root>/evaluate.py --<config family>/<config name>`  

For further details, please refer to my thesis work:  
`(a link will be available soon)`

Author: Gabriele Lagani - gabriele.lagani@gmail.com

