import params as P
import basemodel.model, basemodel.model_no_bias, basemodel.model_bn_before_relu, \
	basemodel.model1, basemodel.model1_no_bias, basemodel.model2, basemodel.model3, basemodel.model4, \
	basemodel.top1, basemodel.top2, basemodel.top3, basemodel.top4, basemodel.fc, basemodel.fc_no_bias
import hebbmodel.model, hebbmodel.model1, hebbmodel.model1_som, \
	hebbmodel.top1, hebbmodel.top2, hebbmodel.top3, hebbmodel.top4, hebbmodel.fc


class Configuration:
	def __init__(self,
				 config_family,
				 config_name,
				 net_class,
				 batch_size,
				 num_epochs,
				 iteration_ids,
				 val_set_split,
				 augment_data,
				 whiten_data,
				 learning_rate=None,
				 lr_decay=None,
				 milestones=None,
				 momentum=None,
				 l2_penalty=None,
				 pre_net_class=None,
				 pre_net_mdl_path=None,
				 pre_net_out=None):
		self.CONFIG_FAMILY = config_family
		self.CONFIG_NAME = config_name
		self.CONFIG_ID = self.CONFIG_FAMILY + '/' + self.CONFIG_NAME
		
		self.Net = net_class
		
		self.BATCH_SIZE = batch_size
		self.NUM_EPOCHS = num_epochs
		self.ITERATION_IDS = iteration_ids
		
		# Paths where to save the model
		self.MDL_PATH = {}
		# Path where to save accuracy plot
		self.ACC_PLT_PATH = {}
		# Path where to save kernel images
		self.KNL_PLT_PATH = {}
		for iter_id in self.ITERATION_IDS:
			# Path where to save the model
			self.MDL_PATH[iter_id] = P.RESULT_FOLDER + '/' + self.CONFIG_ID + '/save/model' + str(iter_id) + '.pt'
			# Path where to save accuracy plot
			self.ACC_PLT_PATH[iter_id] = P.RESULT_FOLDER + '/' + self.CONFIG_ID + '/figures/accuracy' + str(iter_id) + '.png'
			# Path where to save kernel images
			self.KNL_PLT_PATH[iter_id] = P.RESULT_FOLDER + '/' + self.CONFIG_ID + '/figures/kernels' + str(iter_id) + '.png'
		# Path to the CSV where test results are saved
		self.CSV_PATH = P.RESULT_FOLDER + '/' + self.CONFIG_ID + '/test_results.csv'
		
		# Define the splitting point of the training batches between training and validation datasets
		self.VAL_SET_SPLIT = val_set_split
		
		# Define wether to apply data augmentation or whitening
		self.AUGMENT_DATA = augment_data
		self.WHITEN_DATA = whiten_data
		
		self.LEARNING_RATE = learning_rate # Initial learning rate, periodically decreased by a lr_scheduler
		self.LR_DECAY = lr_decay # LR decreased periodically by a factor of 10
		self.MILESTONES = milestones # Epochs at which LR is decreased
		self.MOMENTUM = momentum
		self.L2_PENALTY = l2_penalty
		
		self.PreNet = pre_net_class
		self.PRE_NET_MDL_PATH = pre_net_mdl_path
		self.PRE_NET_OUT = pre_net_out
	

CONFIG_LIST = [
	
	################################################################################################################
	####										GDES CONFIGURATIONS												####
	################################################################################################################
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='config_base_91', # Val/Test: 91.55
		net_class=basemodel.model.Net,
		batch_size=64,
		num_epochs=100,
		iteration_ids=[0],
		val_set_split=50000,
		augment_data=True,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.1,
		milestones=[40, 70, 90],
		momentum=0.9,
		l2_penalty=3e-2,
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='config_base_91_whiten', # Val/Test: 91.54
		net_class=basemodel.model.Net,
		batch_size=64,
		num_epochs=100,
		iteration_ids=[0],
		val_set_split=50000,
		augment_data=True,
		whiten_data=True,
		learning_rate=1e-3,
		lr_decay=0.1,
		milestones=[40, 70, 90],
		momentum=0.9,
		l2_penalty=3e-2,
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='config_base', # Val: 85.72, Test: 84.95
		net_class=basemodel.model.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=6e-2,
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='config_base_whiten', # Val: 86.15, Test: 85.77
		net_class=basemodel.model.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=True,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=6e-2,
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='config_base_no_bias', # Val: 85.17, Test: 84.66
		net_class=basemodel.model_no_bias.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=13*1e-2,
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='config_bn_before_relu', # Val: 85.46, Test: 84.88
		net_class=basemodel.model_bn_before_relu.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-2,
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='config_1l', # Val: 69.20, Test: 69.04
		net_class=basemodel.model1.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=2e-2,
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='config_1l_whiten', # Val: 68.59, Test: 68.10
		net_class=basemodel.model1.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=True,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=2e-2,
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='config_1l_no_bias', # Val: 66.21, Test: 66.38
		net_class=basemodel.model1_no_bias.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=1e-2,
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='config_2l', # Val: 77.41, Test: 76.91
		net_class=basemodel.model2.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=3e-2
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='config_3l', # Val: 81.65, Test: 81.37
		net_class=basemodel.model3.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=4e-2
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='config_4l', # Val: 84.21, Test: 83.97
		net_class=basemodel.model4.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-2
	),
	
	################################################################################################################
	####					CONFIGS: GDES CLASSIFIER ON FEATURES EXTRACTED FROM LOWER LAYERS					####
	################################################################################################################
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_gdes_conv1', # Val: 60.85, Test: 60.71
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=basemodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_base/save/model0.pt',
		pre_net_out=basemodel.model.Net.BN1
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_gdes_conv1_whiten', # Val: 59.93, Test: 60.89
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=True,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=basemodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_base_whiten/save/model0.pt',
		pre_net_out=basemodel.model.Net.BN1
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_gdes_conv1_91', # Val/Test: 60.40
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=50000,
		augment_data=True,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=basemodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_base_91/save/model0.pt',
		pre_net_out=basemodel.model.Net.BN1
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_gdes_conv1_91_whiten', # Val/Test: 60.95
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=50000,
		augment_data=True,
		whiten_data=True,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=basemodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_base_91_whiten/save/model0.pt',
		pre_net_out=basemodel.model.Net.BN1
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_gdes_conv1_no_bias', # Val: 66.60, Test: 65.62
		net_class=basemodel.fc_no_bias.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=basemodel.model_no_bias.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_base_no_bias/save/model0.pt',
		pre_net_out=basemodel.model_no_bias.Net.BN1
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_gdes_conv2', # Val: 66.30, Test: 66.30
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=basemodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_base/save/model0.pt',
		pre_net_out=basemodel.model.Net.BN2
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_gdes_conv2_91', # Val/Test: 67.95
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=50000,
		augment_data=True,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=basemodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_base_91/save/model0.pt',
		pre_net_out=basemodel.model.Net.BN2
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_gdes_conv3', # Val: 72.76, Test: 72.39
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=basemodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_base/save/model0.pt',
		pre_net_out=basemodel.model.Net.BN3
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_gdes_conv3_91', # Val/Test: 75.58
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=50000,
		augment_data=True,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=basemodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_base_91/save/model0.pt',
		pre_net_out=basemodel.model.Net.BN3
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_gdes_conv4', # Val: 83.41, Test: 82.69
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=basemodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_base/save/model0.pt',
		pre_net_out=basemodel.model.Net.BN4
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_gdes_conv4_91', # Val/Test: 88.15
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=50000,
		augment_data=True,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=basemodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_base_91/save/model0.pt',
		pre_net_out=basemodel.model.Net.BN4
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_raw_img', # Val: 40.43, Test: 40.36
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_raw_img_whiten', # Val: 39.78, Test: 39.81
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=True,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_raw_img_no_bias', # Val: 39.73, Test: 39.77
		net_class=basemodel.fc_no_bias.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
	),
	
	################################################################################################################
	####					CONFIGS: GDES CLASSIFIER ON FEATURES EXTRACTED FROM HEBB LAYERS						####
	################################################################################################################
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_hebb_conv1', # Val: 64.76, Test: 63.92
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=True,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=hebbmodel.model1.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/config_1l/save/model0.pt',
		pre_net_out=hebbmodel.model1.Net.BN1
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_hebb_conv1_no_bias', # Val: 64.43, Test: 63.65
		net_class=basemodel.fc_no_bias.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=True,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=hebbmodel.model1.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/config_1l/save/model0.pt',
		pre_net_out=hebbmodel.model1.Net.BN1
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_hebb_conv1_no_whiten', # Val: 56.59, Test: 56.57
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=hebbmodel.model1.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/config_1l_no_whiten/save/model0.pt',
		pre_net_out=hebbmodel.model1.Net.BN1
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_hebb_conv1_som', # Val: 53.23, Test: 53.40
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=hebbmodel.model1_som.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/config_1l_som/save/model0.pt',
		pre_net_out=hebbmodel.model1_som.Net.BN1
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_hebb_conv2', # Val: 64.19, Test: 63.81
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=True,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=hebbmodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/config_base/save/model0.pt',
		pre_net_out=hebbmodel.model.Net.BN2
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_hebb_conv3', # Val: 58.92, Test: 58.28
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=True,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=hebbmodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/config_base/save/model0.pt',
		pre_net_out=hebbmodel.model.Net.BN3
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_hebb_conv4', # Val: 53.47, Test: 52.99
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=True,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=hebbmodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/config_base/save/model0.pt',
		pre_net_out=hebbmodel.model.Net.BN4
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_hebb_fc5', # Val: 41.49, Test: 41.78
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=True,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=hebbmodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/config_base/save/model0.pt',
		pre_net_out=hebbmodel.model.Net.BN5
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_hebb_fc5_no_whiten', # Val: 34.01, Val: 34.39
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=hebbmodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/config_base_no_whiten/save/model0.pt',
		pre_net_out=hebbmodel.model.Net.BN5
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_hebb_fc5_no_bias', # Val: 41.50, Test: 41.79
		net_class=basemodel.fc_no_bias.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=True,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=hebbmodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/config_base/save/model0.pt',
		pre_net_out=hebbmodel.model.Net.BN5
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='top1', # Val: 85.15, Test: 84.93
		net_class=basemodel.top1.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=True,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=7e-2,
		pre_net_class=hebbmodel.model1.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/config_1l/save/model0.pt',
		pre_net_out=hebbmodel.model1.Net.BN1
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='top2', # Val: 79.15, Test: 78.61
		net_class=basemodel.top2.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=True,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=6e-2,
		pre_net_class=hebbmodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/config_base/save/model0.pt',
		pre_net_out=hebbmodel.model.Net.BN2
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='top3', # Val: 68.34, Test: 67.87
		net_class=basemodel.top3.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=True,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-2,
		pre_net_class=hebbmodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/config_base/save/model0.pt',
		pre_net_out=hebbmodel.model.Net.BN3
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='top4', # Val: 57.77, Test: 57.56
		net_class=basemodel.top4.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=True,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=hebbmodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/config_base/save/model0.pt',
		pre_net_out=hebbmodel.model.Net.BN4
	),
	
	################################################################################################################
	####										HEBB CONFIGURATIONS												####
	################################################################################################################
	
	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='config_1l', # Val: 41.65, Test: 41.84
		net_class=hebbmodel.model1.Net,
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=True,
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='config_1l_no_whiten', # Val: 26.89, Test: 27.11
		net_class=hebbmodel.model1.Net,
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='config_1l_som', # Val: 25.57, Test: 25.88
		net_class=hebbmodel.model1_som.Net,
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='config_base', # Val: 27.52, Test: 28.59
		net_class=hebbmodel.model.Net,
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=True,
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='config_base_no_whiten', # Val: 23.77, Test: 24.30
		net_class=hebbmodel.model.Net,
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
	),
	
	################################################################################################################
	####					CONFIGS: HEBB CLASSIFIER ON FEATURES EXTRACTED FROM GDES LAYERS						####
	################################################################################################################
	
	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='fc_on_gdes_fc5', # Val: 85.76, Test:84.88
		net_class=hebbmodel.fc.Net,
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		pre_net_class=basemodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_base/save/model0.pt',
		pre_net_out=basemodel.model.Net.BN5
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='fc_on_gdes_fc5_whiten', # Val: 86.12, Test: 85.67
		net_class=hebbmodel.fc.Net,
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=True,
		pre_net_class=basemodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_base_whiten/save/model0.pt',
		pre_net_out=basemodel.model.Net.BN5
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='fc_on_gdes_fc5_no_bias', # Val: 85.01, Test: 84.59
		net_class=hebbmodel.fc.Net,
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		pre_net_class=basemodel.model_no_bias.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_base_no_bias/save/model0.pt',
		pre_net_out=basemodel.model_no_bias.Net.BN5
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='fc_on_gdes_fc5_91', # Val/Test: 91.54
		net_class=hebbmodel.fc.Net,
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=50000,
		augment_data=True,
		whiten_data=False,
		pre_net_class=basemodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_base_91/save/model0.pt',
		pre_net_out=basemodel.model.Net.BN5
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='fc_on_gdes_fc5_91_whiten', # Val/Test: 91.55
		net_class=hebbmodel.fc.Net,
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=50000,
		augment_data=True,
		whiten_data=True,
		pre_net_class=basemodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_base_91_whiten/save/model0.pt',
		pre_net_out=basemodel.model.Net.BN5
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='fc_on_gdes_conv4',
		net_class=hebbmodel.fc.Net,
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		pre_net_class=basemodel.model.Net, # Val: 82.73, Test: 82.18
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_base/save/model0.pt',
		pre_net_out=basemodel.model.Net.BN4
		#pre_net_class=basemodel.model4.Net, # Val: 83.98, Test: 83.82
		#pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_4l/save/model0.pt',
		#pre_net_out=basemodel.model4.Net.BN4
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='fc_on_gdes_conv4_91', # Val/Test: 86.73
		net_class=hebbmodel.fc.Net,
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=50000,
		augment_data=True,
		whiten_data=False,
		pre_net_class=basemodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_base_91/save/model0.pt',
		pre_net_out=basemodel.model.Net.BN4
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='fc_on_gdes_conv3',
		net_class=hebbmodel.fc.Net,
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		pre_net_class=basemodel.model.Net, # Val: 67.69, Test: 67.79
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_base/save/model0.pt',
		pre_net_out=basemodel.model.Net.BN3
		#pre_net_class=basemodel.model3.Net, # Val: 78.96, Test: 78.83
		#pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_3l/save/model0.pt',
		#pre_net_out=basemodel.model3.Net.BN3
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='fc_on_gdes_conv3_91', # Val/Test: 68.71
		net_class=hebbmodel.fc.Net,
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=50000,
		augment_data=True,
		whiten_data=False,
		pre_net_class=basemodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_base_91/save/model0.pt',
		pre_net_out=basemodel.model.Net.BN3
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='fc_on_gdes_conv2',
		net_class=hebbmodel.fc.Net,
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		pre_net_class=basemodel.model.Net, # Val: 56.32, Test: 56.59
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_base/save/model0.pt',
		pre_net_out=basemodel.model.Net.BN2
		#pre_net_class=basemodel.model2.Net, # Val: 63.43, Test: 62.92
		#pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_2l/save/model0.pt',
		#pre_net_out=basemodel.model2.Net.BN2
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='fc_on_gdes_conv2_91', # Val/Test: 53.38
		net_class=hebbmodel.fc.Net,
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=50000,
		augment_data=True,
		whiten_data=False,
		pre_net_class=basemodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_base_91/save/model0.pt',
		pre_net_out=basemodel.model.Net.BN2
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='fc_on_gdes_conv1',
		net_class=hebbmodel.fc.Net,
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		pre_net_class=basemodel.model.Net, # Val: 45.43, Test: 46.58
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_base/save/model0.pt',
		pre_net_out=basemodel.model.Net.BN1
		#pre_net_class=basemodel.model1.Net, # Val: 45.72, Test: 46.01
		#pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_1l/save/model0.pt',
		#pre_net_out=basemodel.model1.Net.BN1
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='fc_on_gdes_conv1_whiten',
		net_class=hebbmodel.fc.Net,
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=True,
		pre_net_class=basemodel.model.Net, # Val: 40.54, Test: 41.43
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_base_whiten/save/model0.pt',
		pre_net_out=basemodel.model.Net.BN1
		#pre_net_class=basemodel.model1.Net, # Val: 41.79, Test: 42.01
		#pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_1l_whiten/save/model0.pt',
		#pre_net_out=basemodel.model1.Net.BN1
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='fc_on_gdes_conv1_91', # Val/Test: 42.47
		net_class=hebbmodel.fc.Net,
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=50000,
		augment_data=True,
		whiten_data=False,
		pre_net_class=basemodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_base_91/save/model0.pt',
		pre_net_out=basemodel.model.Net.BN1
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='fc_on_gdes_conv1_91_whiten', # Val/Test: 37.39
		net_class=hebbmodel.fc.Net,
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=50000,
		augment_data=True,
		whiten_data=True,
		pre_net_class=basemodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_base_91_whiten/save/model0.pt',
		pre_net_out=basemodel.model.Net.BN1
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='fc_on_gdes_conv1_no_bias',
		net_class=hebbmodel.fc.Net,
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		pre_net_class=basemodel.model_no_bias.Net, # Val: 42.40, Test: 43.01
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_base_no_bias/save/model0.pt',
		pre_net_out=basemodel.model_no_bias.Net.BN1
		#pre_net_class=basemodel.model1_no_bias.Net, # Val: 40.74, Test: 41.53
		#pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_1l_no_bias/save/model0.pt',
		#pre_net_out=basemodel.model1_no_bias.Net.BN1
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='fc_on_raw_img', # Val: 26.65, Test: 27.48
		net_class=hebbmodel.fc.Net,
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='fc_on_raw_img_whiten', # Val: 22.75, Test: 23.13
		net_class=hebbmodel.fc.Net,
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=True,
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='top1', # Val: 33.14, Test: 32.95
		net_class=hebbmodel.top1.Net,
		batch_size=32, # here I had to use smaller batch size because of memory constraints
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		pre_net_class=basemodel.model1.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_1l/save/model0.pt',
		pre_net_out=basemodel.model1.Net.BN1
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='top2', # Val: 50.73, Test: 50.43
		net_class=hebbmodel.top2.Net,
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		pre_net_class=basemodel.model2.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_2l/save/model0.pt',
		pre_net_out=basemodel.model2.Net.BN2
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='top3', # Val: 71.59, Test: 71.18
		net_class=hebbmodel.top3.Net,
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		pre_net_class=basemodel.model3.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_3l/save/model0.pt',
		pre_net_out=basemodel.model3.Net.BN3
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='top4', # Val: 83.45, Test: 83.16
		net_class=hebbmodel.top4.Net,
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		pre_net_class=basemodel.model4.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_4l/save/model0.pt',
		pre_net_out=basemodel.model4.Net.BN4
	),
	
]


CONFIGURATIONS = {}
for c in CONFIG_LIST: CONFIGURATIONS[c.CONFIG_ID] = c
