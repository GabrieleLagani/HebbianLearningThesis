Pytorch implementation of Hebbian learning algorithms to train
deep convolutional neural networks.
A neural network model is trained on CIFAR10 both using 
Hebbian algorithms and SGD in order to compare the results.
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

The experiments were performed in the following environment:  
Google CoLaboratory  
Python version: 3.6  
Pytorch version: 1.0.0  
Torchvision version: 0.2.1 (N.B. With successive versions you will get an error at line 27 of the `data.py` file, which can be corrected by replacing `cifar10.train_data` with `cifar10.data`)  
  
For further details, please refer to my thesis work (N.B. The latest updates might not be covered in this document): 
_"Hebbian Learning Algorithms for Training Convolutional Neural Networks - Gabriele Lagani"_  
Link: `https://drive.google.com/file/d/1Mo-AKTzm5k3hcnO6UkVpNp0Ce0M-U7zc/view?usp=sharing`  


Author: Gabriele Lagani - gabriele.lagani@gmail.com

