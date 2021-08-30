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

You might want to give a look at the new repos as well!  
HebbianPCA: https://github.com/GabrieleLagani/HebbianPCA/blob/master/README.md  
Latest updates: https://github.com/GabrieleLagani/HebbianLearning  

In order to launch a training session, type:  
`PYTHONPATH=<project root> python <project root>/train.py --config <config family>/<config name>`  
Where `<config family>` is either `gdes` or `hebb`, depending whether 
you want to run gradient descent or hebbian training, and 
`<config name>` is the name of one of the training configurations in 
the `config.py` file.  
Example:  
`PYTHONPATH=<project root> python <project root>/train.py --config gdes/config_base`  
To evaluate the network on the CIFAR10 test set, type:  
`PYTHONPATH=<project root> python <project root>/evaluate.py --config <config family>/<config name>`

For further details, please refer to my thesis work:  
_"Hebbian Learning Algorithms for Training Convolutional Neural Networks; G. Lagani"_  
available at https://etd.adm.unipi.it/theses/available/etd-03292019-220853/unrestricted/hebbian_learning_algorithms_for_training_convolutional_neural_networks_gabriele_lagani.pdf
and the related paper:  
_"Hebbian Learning Meets Deep Convolutional Neural Networks; G. Amato, F. Carrara, F. Falchi, C. Gennaro and G. Lagani"_  
available at: http://www.nmis.isti.cnr.it/falchi/Draft/2019-ICIAP-HLMSD.pdf  

Author: Gabriele Lagani - gabriele.lagani@gmail.com

