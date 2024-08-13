
"""
MoCHI model module
"""

import os
import bz2
import pickle
import _pickle as cPickle
import torch
from torch import Tensor
from torch.nn import functional as F
from pymochi.data import *
from pymochi.transformation import get_transformation
import itertools
import shutil
import functools
from sklearn.cluster import KMeans

class ConstrainedLinear(torch.nn.Linear):
    """
    A linear layer constrained to positive weights only.
    """
    def forward(
        self, 
        input):
        # return F.linear(input, self.weight.clamp(min=0, max=1000), self.bias)
        return F.linear(input, self.weight.abs(), self.bias)

class MochiWeightedL1Loss(torch.nn.L1Loss):
    """
    A weighted version of L1Loss with no reduction.
    """
    def forward(self, input: Tensor, target: Tensor, weight: Tensor) -> Tensor:
        return F.l1_loss(input, target, reduction='none') * weight

class MochiGaussianNLLLoss(torch.nn.GaussianNLLLoss):
    """
    Mochi version of GaussianNLLLoss with no reduction that accepts fitness weights (1/sigma) rather than var = sigma^2.
    """
    def forward(self, input: Tensor, target: Tensor, weight: Tensor) -> Tensor:
        return F.gaussian_nll_loss(input, target, torch.pow(weight, -2), full = True, eps = 0, reduction = "none")

class MochiModel(torch.nn.Module):
    """
    A custom model/module.
    """
    def __init__(
        self, 
        input_shape, 
        mask, 
        model_design,
        custom_transformations,
        sos_architecture,
        sos_outputlinear):
        """
        Initialize a MochiModel object.

        :param input_shape: shape of input data (required).
        :param mask: tensor to mask weights that should not be fit (required).
        :param model_design: Model design data frame with phenotype, transformation and trait columns (required).
        :param custom_transformations: dictionary of custom transformations where keys are function names and values are functions (required).
        :param sos_architecture: list of integers corresponding to number of neurons per fully-connected sumOfSigmoids hidden layer (required).
        :param sos_outputlinear: boolean indicating whether final sumOfSigmoids should be linear rather than sigmoidal (required).
        :returns: MochiModel object.
        """     
        super(MochiModel, self).__init__()
        #Model design
        self.model_design = model_design
        #Custom transformations
        self.custom_transformations = custom_transformations
        #SOS parameters
        self.sos_architecture = sos_architecture
        self.sos_outputlinear = sos_outputlinear
        #Additive traits
        n_additivetraits = len(list(set([item for sublist in list(self.model_design['trait']) for item in sublist])))
        self.additivetraits = torch.nn.ModuleList([torch.nn.Linear(input_shape, 1, bias = False, dtype=torch.float32) for i in range(n_additivetraits)])

        #Global trainable parameters
        self.globalparams = torch.nn.ModuleList(
            [torch.nn.ParameterDict({j:torch.nn.Parameter(torch.tensor(1.0), requires_grad=True) for j in get_transformation(i, custom = self.custom_transformations)().keys()}) for i in self.model_design.transformation])

        # #Shared global parameters
        # #Identify shared global parameters
        # shared_globalparams = {}
        # for i in self.globalparams:
        #     shared_globalparams = shared_globalparams|{k:v for k,v in i if k.endswith("_shared")}
        # #Set pointers to shared global parameters
        # for i in self.globalparams:
        #     for j in i.keys():
        #         if j.endswith("_shared"):
        #             i[j] = shared_globalparams[j]

        #Arbitrary non-linear transformations (arbitrary architecture) - SumOfSigmoids
        #Additive trait dimensionality of SOS phenotypes
        sos_atdim = [len(self.model_design.loc[i,'trait']) for i in range(len(self.model_design)) if self.model_design.loc[i,'transformation']=="SumOfSigmoids"]
        #SOS index of SOS phenytypes
        self.model_design.loc[self.model_design.transformation=="SumOfSigmoids",'sos_index'] = list(range(len(sos_atdim)))
        #SOS module list
        #SOS first layer
        self.sumofsigmoids_list = [torch.nn.ModuleList([torch.nn.Linear(i, self.sos_architecture[0], dtype=torch.float32) for i in sos_atdim])]
        #SOS intermediate layers
        if len(self.sos_architecture)>1:
            for i in range(len(self.sos_architecture)-1):
                self.sumofsigmoids_list += [torch.nn.ModuleList([torch.nn.Linear(self.sos_architecture[i], self.sos_architecture[i+1], dtype=torch.float32) for j in range(len(sos_atdim))])]
        #SOS last layer
        self.sumofsigmoids_list += [torch.nn.ModuleList([torch.nn.Linear(self.sos_architecture[-1], 1, dtype=torch.float32) for i in sos_atdim])]
        #SOS convert to ModuleList of ModuleLists
        self.sumofsigmoids_list = torch.nn.ModuleList(self.sumofsigmoids_list)

        #Fitness linear transformations
        n_linears = len(self.model_design)
        self.linears = torch.nn.ModuleList([ConstrainedLinear(1, 1, dtype=torch.float32) for i in range(n_linears)])
        #Training history - validation loss
        self.training_history = {"val_loss": []}
        #Training history - WT coefficients
        for i in range(len(self.additivetraits)):
            self.training_history['additivetrait'+str(i+1)+"_WT"] = []
        #Training history - WT residuals
        for i in range(len(self.model_design)):
            self.training_history['residual'+str(i+1)+"_WT"] = []
        #Mask coefficients that are impossible to fit
        self.mask = mask

    def forward(
        self, 
        select, 
        X,
        mask):
        """
        Forward pass through the model.

        :param select: Select tensor which indicates the corresponding phenotype (required).
        :param X: Feature tensor describing input sequences and interactions (required).
        :param mask: Mask tensor which sets corresponding features to zero (required).
        :returns: Output tensor.
        """  
        #Split select tesnsor into list
        select_list = [torch.narrow(select, 1, i, 1) for i in range(select.shape[1])]

        # #Split mask tensor into list
        # mask_list = [torch.narrow(mask, 0, i, 1) for i in range(mask.shape[0])]

        #Loop over all phenotypes
        observed_phenotypes = []
        for i in range(len(self.model_design)):
            #Additive traits
            # additive_traits = [self.additivetraits[j-1](torch.mul(X, mask_list[i])) for j in self.model_design.loc[i,'trait']]
            additive_traits = [self.additivetraits[j-1](torch.mul(
                X, 
                torch.reshape(torch.narrow(torch.narrow(mask, 0, j-1, 1), 1, i, 1), (1, -1)))) for j in self.model_design.loc[i,'trait']]
            #Molecular phenotypes
            if self.model_design.loc[i,'transformation'] not in ["SumOfSigmoids"]:
                globalparams = self.globalparams[i]
                transformed_trait = get_transformation(self.model_design.loc[i,'transformation'], custom = self.custom_transformations)(additive_traits, globalparams)           
            elif self.model_design.loc[i,'transformation'] == "SumOfSigmoids":
                #SumOfSigmoids layers
                transformed_trait = None
                for j in range(len(self.sumofsigmoids_list)):
                    if j == 0:
                        transformed_trait = torch.sigmoid(self.sumofsigmoids_list[j][int(self.model_design.loc[i,'sos_index'])](torch.cat(additive_traits, 1)))
                    elif j != (len(self.sumofsigmoids_list)-1):
                        transformed_trait = torch.sigmoid(self.sumofsigmoids_list[j][int(self.model_design.loc[i,'sos_index'])](transformed_trait))
                    else:
                        if self.sos_outputlinear:
                            transformed_trait = self.sumofsigmoids_list[j][int(self.model_design.loc[i,'sos_index'])](transformed_trait)
                        else:
                            transformed_trait = torch.sigmoid(self.sumofsigmoids_list[j][int(self.model_design.loc[i,'sos_index'])](transformed_trait))

            #Observed phenotypes
            observed_phenotypes += [torch.mul(self.linears[i](transformed_trait), select_list[i])]
        #Sum observed phenotypes
        return torch.stack(observed_phenotypes, dim=0).sum(dim=0)

    def weights_init_fill(
        self,
        linear_weight,
        linear_bias):
        """
        Initialize layer weights.

        :returns: Nothing.
        """ 
        #Initialize all layer weights
        for layer_list in self.children():
            for layer in layer_list:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
                elif type(layer)==torch.nn.ModuleList:
                    for ml_layer in layer:
                        if hasattr(ml_layer, 'reset_parameters'):
                            ml_layer.reset_parameters()

        #Initialize ConstrainedLinear layer weights
        for i in range(len(self.model_design)):
            # self.linears[i].weight.data.fill_(1)
            # self.linears[i].bias.data.fill_(0)
            self.linears[i].weight.data.fill_(linear_weight[i])
            self.linears[i].bias.data.fill_(linear_bias[i])

    def get_data_loaders(
        self, 
        data, 
        batch_size):
        """
        Get list of dataloaders.

        :param data: Dictionary of dictionaries of tensors as output by MochiData.get_data (required).
        :param batch_size: Minibatch size (required).
        :returns: Tuple of dataloaders.
        """ 
        dataloader_list = []
        for k in ['training', 'validation', 'test']:
            dataloader_list.append(FastTensorDataLoader(
                data[k]["select"], 
                data[k]["X"], 
                data[k]["y"], 
                data[k]["y_wt"], batch_size=batch_size, shuffle=True))
        return tuple(dataloader_list)

    def calculate_l1l2_norm(self):      
        """
        Calculate L1 and L2 norm of additive trait parameters (excluding WT).

        :returns: Tuple of L2 an L2 norms.
        """ 
        #Get weights for all additive traits
        additivetrait_parameters = [i.parameters() for i in self.additivetraits]
        #Unlist additive trait weights
        additivetrait_parameters = [item for sublist in additivetrait_parameters for item in sublist]
        l1_norm = sum(p[0][1:].abs().sum() for p in additivetrait_parameters)
        l2_norm = sum(p[0][1:].pow(2.0).sum() for p in additivetrait_parameters)

        # #Get weights for all sigmoidal layers
        # sigmoid_parameters = [
        #     [i.parameters() for i in self.sumofsigmoids1],
        #     [i.parameters() for i in self.sumofsigmoids2],
        #     [i.parameters() for i in self.sumofsigmoids3],
        #     [i.parameters() for i in self.sumofsigmoids4]]
        # sigmoid_parameters = [item for sublist in sigmoid_parameters for item in sublist]
        # sigmoid_parameters = [item for sublist in sigmoid_parameters for item in sublist]
        # l1_norm += sum(p.abs().sum() for p in sigmoid_parameters)*100
        # l2_norm += sum(p.pow(2.0).sum() for p in sigmoid_parameters)*100

        return (l1_norm, l2_norm)

    def train_model(
        self, 
        dataloader, 
        loss_function, 
        optimizer, 
        device, 
        l1_lambda = 0, 
        l2_lambda = 0):
        """
        Train model.

        :param dataloader: Dataloader (required).
        :param loss_function: Loss function (required).
        :param optimizer: Optimizer (required).
        :param device: cpu or cuda (required).
        :param l1_lambda: Lambda factor applied to L1 norm (default:0).
        :param l2_lambda: Lambda factor applied to L2 norm (default:0).
        :returns: Nothing.
        """ 
        size = dataloader.dataset_len
        self.train()
        batch = 0
        mask = self.mask.to(device)
        for select, X, y, y_wt in dataloader:
            batch += 1
            select, X, y, y_wt = select.to(device), X.to(device), y.to(device), y_wt.to(device)
            # Regularisation of additive trait parameters (excluding WT)
            l1_norm, l2_norm = self.calculate_l1l2_norm()
            # Compute prediction error (weighted by measurement error) + regularization terms
            pred = self(select, X, mask)
            loss = sum(loss_function(pred, y, y_wt))/len(y) + l1_lambda * l1_norm + l2_lambda * l2_norm
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
  
    def validate_model(
        self, 
        dataloader, 
        loss_function, 
        device, 
        l1_lambda = 0, 
        l2_lambda = 0, 
        data_WT = None):
        """
        Validate model.

        :param dataloader: Dataloader (required).
        :param loss_function: Loss function (required).
        :param optimizer: Optimizer (required).
        :param device: cpu or cuda (required).
        :param l1_lambda: Lambda factor applied to L1 norm (default:0).
        :param l2_lambda: Lambda factor applied to L2 norm (default:0).
        :param data_WT: Dictionary of tensors corresponding to WT sequences only (default:0).
        :returns: Nothing.
        """ 
        size = dataloader.dataset_len
        num_batches = len(dataloader)
        self.eval()
        val_loss = 0
        val_WT_resid = []
        with torch.no_grad():
            mask = self.mask.to(device)
            for select, X, y, y_wt in dataloader:
                select, X, y, y_wt = select.to(device), X.to(device), y.to(device), y_wt.to(device)
                # Regularisation of additive trait parameters (excluding WT)
                l1_norm, l2_norm = self.calculate_l1l2_norm()
                # Compute prediction error (weighted by measurement error) + regularization terms
                pred = self(select, X, mask)
                val_loss += (sum(loss_function(pred, y, y_wt))/len(y) + l1_lambda * l1_norm + l2_lambda * l2_norm).item()
            #Training history - WT residuals
            if data_WT!=None:
                select_WT, X_WT, y_WT = data_WT['select'].to(device), data_WT['X'].to(device), data_WT['y'].to(device)
                pred_WT = self(select_WT, X_WT, mask)
                val_WT_resid = list(np.asarray((y_WT - pred_WT).detach().cpu()))
        val_loss /= num_batches
        #Save training history - validation loss
        self.training_history["val_loss"].append(val_loss)
        #Save training history - WT coefficients
        for i in range(len(self.additivetraits)):
            self.training_history['additivetrait'+str(i+1)+"_WT"].append(self.additivetraits[i].weight.detach().cpu().numpy().flatten()[0])
        #Save training history - WT residuals
        if val_WT_resid != []:
            for i in range(len(self.model_design)):
                self.training_history['residual'+str(i+1)+"_WT"].append(float(val_WT_resid[i]))

    def get_status(self):
        """
        Get training status.

        :returns: A string describing the current status.
        """ 
        #Training history - validation loss
        status_list = [f'Avg_val_loss: {self.training_history["val_loss"][-1]:.4f}; ']
        #Training history - WT coefficients
        for i in range(len(self.additivetraits)):
            status_list += [f'{"WTcoef_"+str(i+1)}: {self.training_history["additivetrait"+str(i+1)+"_WT"][-1]:.4f}; ']
        #Training history - WT residuals
        for i in range(len(self.model_design)):
            if "residual"+str(i+1)+"_WT" in self.training_history.keys():
                status_list += [f'{"WTres_"+str(i+1)}: {self.training_history["residual"+str(i+1)+"_WT"][-1]:.4f}; ']
        return ''.join(status_list)

class MochiModelMetadata():
    """
    A simple class to store model metadata.
    """
    def __init__(
        self,
        fold,
        seed,
        grid_search,
        batch_size,
        learn_rate,
        num_epochs,
        num_epochs_grid,
        l1_regularization_factor,
        l2_regularization_factor,
        training_resample,
        early_stopping,
        scheduler_gamma,
        scheduler_epochs,
        loss_function_name,
        sos_architecture,
        sos_outputlinear):
        """
        Initialize a MochiModelMetadata object.

        :param fold: Cross-validation fold (required).
        :param seed: Random seed for both training target data resampling and shuffling training data (required).
        :param grid_search: Whether or not this model was fit during a grid search (required).
        :param batch_size: Minibatch size (required).
        :param learn_rate: Learning rate (required).
        :param num_epochs: Number of training epochs (required).
        :param num_epochs_grid: Number of grid search epochs (required).
        :param l1_regularization_factor: Lambda factor applied to L1 norm (required).
        :param l2_regularization_factor: Lambda factor applied to L2 norm (required).
        :param training_resample: Whether or not to add random noise to training target data proportional to target error (required).
        :param early_stopping: Whether or not to stop training early if validation loss not decreasing (required).
        :param scheduler_gamma: Multiplicative factor of learning rate decay (required).
        :param scheduler_epochs: Number of epochs over which to evaluate scheduler criteria (required).
        :param loss_function_name: Loss function name (required).
        :param sos_architecture: list of integers corresponding to number of neurons per fully-connected sumOfSigmoids hidden layer (required).
        :param sos_outputlinear: boolean indicating whether final sumOfSigmoids should be linear rather than sigmoidal (required).
        :returns: MochiModelMetadata object.
        """ 
        self.fold = fold
        self.seed = seed
        self.grid_search = grid_search
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.num_epochs = num_epochs
        self.num_epochs_grid = num_epochs_grid
        self.l1_regularization_factor = l1_regularization_factor
        self.l2_regularization_factor = l2_regularization_factor
        self.training_resample = training_resample
        self.early_stopping = early_stopping
        self.scheduler_gamma = scheduler_gamma
        self.scheduler_epochs = scheduler_epochs
        self.loss_function_name = loss_function_name
        self.sos_architecture = sos_architecture
        self.sos_outputlinear = sos_outputlinear

    def __str__(self):
        """
        Print object's attributes.

        :returns: A string representation of the object's attributes.
        """
        return str(self.__dict__)

class MochiTask():
    """
    A class for the storage and management of models and data related to a specific inference task.
    """
    def __init__(
        self, 
        directory,
        data = None,
        batch_size = 512,
        learn_rate = 0.05,
        num_epochs = 1000,
        num_epochs_grid = 100,
        l1_regularization_factor = 0,
        l2_regularization_factor = 0,
        training_resample = True,
        early_stopping = True,
        scheduler_gamma = 0.98,
        loss_function_name = 'WeightedL1',
        sos_architecture = [20],
        sos_outputlinear = False):
        """
        Initialize a MochiTask object.

        :param directory: Path to directory where models and results should be saved/loaded (required).
        :param data: An instance of the MochiData class (required unless 'directory' contains a saved task).
        :param batch_size: Minibatch size (default:512).
        :param learn_rate: Learning rate (default:0.05).
        :param num_epochs: Number of training epochs (default:1000).
        :param num_epochs_grid: Number of grid search epochs (default:100).
        :param l1_regularization_factor: Lambda factor applied to L1 norm (default:0).
        :param l2_regularization_factor: Lambda factor applied to L2 norm (default:0).
        :param training_resample: Whether or not to add random noise to training target data proportional to target error (default:True).
        :param early_stopping: Whether or not to stop training early if validation loss not decreasing (default:True).
        :param scheduler_gamma: Multiplicative factor of learning rate decay (default:0.98).
        :param loss_function_name: Loss function name (default:'WeightedL1').
        :param sos_architecture: list of integers corresponding to number of neurons per fully-connected sumOfSigmoids hidden layer (default:[20]).
        :param sos_outputlinear: boolean indicating whether final sumOfSigmoids should be linear rather than sigmoidal (default:False).
        :returns: MochiTask object.
        """ 
        #Get CPU or GPU device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # print(f"Using {self.device} device")
        #initialize remaining attributes
        self.directory = directory
        if data != None:
            #Create task directory
            try:
                os.mkdir(self.directory)
            except FileExistsError:
                if os.path.exists(os.path.join(self.directory, 'saved_models')):
                    print("Error: Task directory with saved models already exists.")
                    raise ValueError
            self.models = []
            self.data = data
            self.batch_size = [int(i) for i in str(batch_size).split(",")]
            self.learn_rate = [float(i) for i in str(learn_rate).split(",")]
            self.num_epochs = num_epochs
            self.num_epochs_grid = num_epochs_grid
            self.l1_regularization_factor = [float(i) for i in str(l1_regularization_factor).split(",")]
            self.l2_regularization_factor = [float(i) for i in str(l2_regularization_factor).split(",")]
            self.training_resample = training_resample
            self.early_stopping = early_stopping
            self.scheduler_gamma = scheduler_gamma
            self.loss_function_name = loss_function_name
            self.sos_architecture = sos_architecture
            self.sos_outputlinear = sos_outputlinear
        else:
            #Load saved models
            # print("Loading task.")
            self.load()

    def save(
        self,
        overwrite = False):
        """
        Save MochiTask object to disk.

        :param overwrite: Whether or not to overwrite previous saved object (default:False).
        :returns: Nothing.
        """ 

        #Check if valid MochiTask
        if 'models' not in dir(self):
            print("Error: Task cannot be saved. Invalid MochiTask instance.")
            raise ValueError

        #Output models directory
        directory = os.path.join(self.directory, 'saved_models')

        #Create output model directory
        try:
            os.mkdir(directory)
        except FileExistsError:
            if overwrite==True:
                print("Warning: Saved models directory already exists. Previous models will be overwritten.")
            else:
                print("Error: Saved models directory already exists. Set 'overwrite'=True to overwrite previous models.")
                raise ValueError

        #Delete entire directory contents and create fresh directory
        shutil.rmtree(directory)
        os.mkdir(directory)

        #Save unpickleable objects
        temp_custom_transformations = copy.deepcopy(self.data.custom_transformations)

        #Save models using torch.save
        if len(self.models)==0:
            print("Warning: No fit models. Saving metadata only.")
        for i in range(len(self.models)):
            self.models[i].custom_transformations = None
            torch.save(self.models[i], os.path.join(directory, 'model_'+str(i)+'.pth'))
            self.models[i].custom_transformations = temp_custom_transformations

        #Save remaining (non-built-in) attributes
        self.data.custom_transformations = None
        save_dict = {}
        for i in self.__dict__.keys():
            #Exclude built-in objects and torch models
            if not i.startswith("__") and i != "models":
                save_dict[i] = self.__dict__[i]
        with bz2.BZ2File(os.path.join(directory, 'data.pbz2'), 'w') as f:
            cPickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)
        self.data.custom_transformations = temp_custom_transformations

    def load(self):
        """
        Load MochiTask from disk.

        :returns: Nothing.
        """ 
        #Output models directory
        directory = os.path.join(self.directory, 'saved_models')

        #Check if model directory exists
        if not os.path.exists(directory):
            print("Error: Saved models directory does not exist.")
            raise ValueError

        #All files in directory
        files = os.listdir(directory)

        #Check data exists (no models required)
        if not ('data.pyc' in files or 'data.pbz2' in files):
            print("Error: Saved models directory structure incorrect.")
            raise ValueError

        #Load (non-built-in) attributes except models
        load_dict = None
        #Compressed pickle (preferred)
        if 'data.pbz2' in files:
            with bz2.BZ2File(os.path.join(directory, 'data.pbz2'), 'rb') as f:
                load_dict = cPickle.load(f)
        #Uncompressed pickle (for backwards compatibility)
        else:
            with open(os.path.join(directory, 'data.pyc'), 'rb') as f:
                load_dict = pickle.load(f)
        for i in load_dict.keys():
            exec("self."+i+" = load_dict['"+i+"']")
        #Get CPU or GPU device (to undo loading of this attribute)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        #Restore custom transformations
        self.data.restore_custom_transformations()

        #Load models using torch.load
        self.models = [None]*len([i for i in files if i.startswith("model_")])
        for i in files:
            if i.startswith("model_"):
                model_index = int(i[6:(len(i)-4)])
                if model_index >= len(self.models) or model_index < 0:
                    print("Error: Saved models index format incorrect.")
                    raise ValueError
                self.models[model_index] = torch.load(os.path.join(directory, i), map_location=self.device)
                #Restore custom transformations
                self.models[model_index].custom_transformations = self.data.custom_transformations
                #Add globalparams if legacy model
                if 'globalparams' not in dir(self.models[model_index]):
                    self.models[model_index].globalparams = torch.nn.ModuleList([torch.nn.ParameterDict({}) for i in self.models[model_index].model_design.transformation])

    def new_model(
        self,
        data):
        """
        Create a new MochiModel object.

        :param data: Dictionary of dictionaries of tensors as output by MochiData.get_data (required).
        :returns: A new MochiModel object.
        """ 
        #Create a new model
        model = MochiModel(
            input_shape = data['training']['X'].shape[1],
            mask = data['training']['mask'],
            model_design = self.data.model_design,
            custom_transformations = self.data.custom_transformations,
            sos_architecture = self.sos_architecture,
            sos_outputlinear = self.sos_outputlinear).to(self.device)
        return model

    def wavg(
        self,
        group, 
        absolute_value = False,
        error_only = False,
        col = 'mean',
        col_weight = 'std',
        suffix = ''):
        """
        Calculate weighted average.

        :param group: DataFrame with col+suffix and col_weight+suffix columns (required).
        :param absolute_value: Whether or not to take absolute value (default:False).
        :param error_only: Whether or not return error only (default:False).
        :param col: Column name of variable to be averaged (default:'mean').
        :param col_weight: Column name of weight variable (default:'std').
        :param suffix: Column name suffices (default:'').
        :returns: Weighted mean (or weighted mean error) numpy array.
        """ 
        d = group[col+suffix]
        w = group[col_weight+suffix]
        if absolute_value:
            d = np.abs(d)
        if error_only:
            return np.sqrt(1/sum(1/np.power(w, 2)))
        else:
            return sum(d/np.power(w, 2))/sum(1/np.power(w, 2))

    def get_global_weights(
        self,
        folds = None,
        grid_search = False):
        """
        Get global weights.

        :param folds: list of cross-validation folds (default:None i.e. all).
        :param grid_search: Whether or not to include grid_search models (default:False).
        :returns: Nothing.
        """ 

        #Check if valid MochiTask
        if 'models' not in dir(self):
            print("Error: Cannot get global weights. Invalid MochiTask instance.")
            raise ValueError

        #Output weights directory
        directory = os.path.join(self.directory, 'weights')

        #Create output weights directory
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass

        #Set folds if not supplied
        if folds is None:
            folds = [i+1 for i in range(self.data.k_folds)]

        #Model subset
        models_subset = [i for i in self.models if ((i.metadata.grid_search==grid_search) & (i.metadata.fold in folds))]
        #Check if at least one model remaining
        if len(models_subset)==0:
            print("No models satisfying criteria.")
            return

        #Construct weight list
        at_list = []
        for j in range(self.data.model_design.shape[0]): 
            param_list = []
            for i in range(len(models_subset)):
                #Get all global weights
                param_list += [{g:models_subset[i].globalparams[j][g].detach().cpu().numpy() for g in models_subset[i].globalparams[j].keys()}]
                param_list[-1] = {'fold': models_subset[i].metadata.fold} | param_list[-1]
            at_list += [pd.DataFrame(param_list)]

        #Save model weights
        for i in range(len(at_list)):
            at_list[i].to_csv(os.path.join(directory, "global_weights_"+self.data.phenotype_names[i]+".txt"), sep = "\t", index = False)

    def get_linear_weights(
        self,
        folds = None,
        grid_search = False):
        """
        Get linear weights.

        :param folds: list of cross-validation folds (default:None i.e. all).
        :param grid_search: Whether or not to include grid_search models (default:False).
        :returns: Nothing.
        """ 

        #Check if valid MochiTask
        if 'models' not in dir(self):
            print("Error: Cannot get linear weights. Invalid MochiTask instance.")
            raise ValueError

        #Output weights directory
        directory = os.path.join(self.directory, 'weights')

        #Create output weights directory
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass

        #Set folds if not supplied
        if folds is None:
            folds = [i+1 for i in range(self.data.k_folds)]

        #Model subset
        models_subset = [i for i in self.models if ((i.metadata.grid_search==grid_search) & (i.metadata.fold in folds))]
        #Check if at least one model remaining
        if len(models_subset)==0:
            print("No models satisfying criteria.")
            return

        #Construct weight list
        at_list = []
        for j in range(self.data.model_design.shape[0]): 
            param_list = []
            for i in range(len(models_subset)):
                #Get weights for all linear transformations
                linear_parameters = [item for item in models_subset[i].linears[j].parameters()]
                param_list += [
                    [models_subset[i].metadata.fold]+
                    # [linear_parameters[0][0][0].detach().cpu().numpy(), linear_parameters[1][0].detach().cpu().numpy()]]
                    [linear_parameters[0][0][0].detach().abs().cpu().numpy(), linear_parameters[1][0].detach().cpu().numpy()]]
            at_list += [pd.DataFrame(param_list, columns = ['fold', 'kernel', 'bias'])]

        #Save model weights
        for i in range(len(at_list)):
            at_list[i].to_csv(os.path.join(directory, "linears_weights_"+self.data.phenotype_names[i]+".txt"), sep = "\t", index = False)

    def add_additive_trait_statistics(
        self,
        input_list,
        RT = None):
        """
        Add statistics to additive trait weights (mean, std, ci95 etc.).

        :param input_list: list of of data frames (one per additive trait).
        :param RT: R=gas constant (in kcal/K/mol) * T=Temperature (in K) (optional).
        :returns: A list of data frames (one per additive trait).
        """ 

        #Calculate summary metrics for each additive trait (mean, std, ci95 etc.)
        for i in range(len(input_list)):
            fold_cols = [j for j in list(input_list[i].columns) if j.startswith("fold_")]
            input_list[i]['n'] = input_list[i].loc[:,fold_cols].notnull().sum(axis=1)
            input_list[i]['mean'] = input_list[i].loc[:,fold_cols].mean(axis=1)
            input_list[i]['std'] = input_list[i].loc[:,fold_cols].std(axis=1)
            input_list[i]['ci95'] = input_list[i]['std']*1.96*2
            input_list[i]['trait_name'] = self.data.additive_trait_names[i]
            if RT!=None:
                input_list[i]['mean_kcal/mol'] = input_list[i]['mean']*RT
                input_list[i]['std_kcal/mol'] = input_list[i]['std']*RT
                input_list[i]['ci95_kcal/mol'] = input_list[i]['std_kcal/mol']*1.96*2

        return input_list

    def maximize_correlations_k_means(
        self,
        dfs):
        """
        Maximizes within-DataFrame correlations using K-Means clustering, preserving column order.

        :param input_dfs: list of pandas DataFrames.
        :returns: A list of optimized DataFrames (one per additive trait).
        """

        #DataFrame parameters
        num_dfs = len(dfs)
        n = len(dfs[0].columns)
        column_names = list(dfs[0].columns)*num_dfs

        #Concatenate DataFrames along rows
        data = pd.concat([df.T.reset_index(drop=True) for df in dfs], axis=0)

        #Perform K-Means clustering while dropping coefficients that have NAs
        kmeans = KMeans(n_clusters=num_dfs, random_state=0).fit(np.array(data.dropna(axis=1, how='any')))
        labels = kmeans.labels_

        #Change labels to preserve df order
        trans_dict = {}
        count = 0
        for i in labels.reshape((num_dfs,n)):
            trans_dict[pd.Series(i).value_counts().index[0]] = count
            count += 1
        try:
            labels = np.array([trans_dict[i] for i in labels])
        except KeyError:
            print(f"Warning: Aligning additive traits using K-means clustering failed.")
            return []

        #Assign columns to DataFrames, preserving original order
        optimized_dfs = [pd.DataFrame() for _ in range(num_dfs)]
        for i, (column_index, label) in enumerate(zip(column_names, labels)):
            optimized_dfs[label][column_index] = data.iloc[i]

        #Check each DataFrame is complete
        if sum([i.shape[1]!=n for i in optimized_dfs]) != 0:
            print(f"Warning: Aligning additive traits using K-means clustering failed.")
            return []

        print(f"Aligning additive traits using K-means clustering succeeded.")

        #Sort columns by name
        for i in range(len(optimized_dfs)):
            optimized_dfs[i] = optimized_dfs[i][dfs[0].columns]

        return optimized_dfs

    def align_additive_traits(
        self,
        input_list):
        """
        Align additive trait fold columns for inferred multidimensional global epistasis ('SumOfSigmoids').

        :param input_list: list of of data frames (one per additive trait).
        :returns: A list of data frames (one per additive trait).
        """ 

        #Check if global epistasis inferred
        if sum(self.data.model_design['transformation']=='SumOfSigmoids') == 0:
            return []

        #Multidimensional global epistasis additive traits
        mge_at_list = [i for i in self.data.model_design.loc[self.data.model_design['transformation']=='SumOfSigmoids','trait'] if len(i)>1]
        
        #Check if multidimensional global epistasis inferred
        if mge_at_list == []:
            return []

        #Aligned DataFrames object
        output_list = input_list.copy()

        #Align DataFrames if associated additive traits are not shared between multiple phenotypes
        for mge_at in mge_at_list:
            #Frequency of each additive trait in model design
            at_freq_dict = pd.Series([item for sublist in self.data.model_design['trait'] for item in sublist]).value_counts().to_dict()
            #No additive traits shared between multiple phenotypes
            if [i for i in mge_at if at_freq_dict[i]>1] == []:
                #Align DataFrames
                opt_list = self.maximize_correlations_k_means(
                    dfs = [input_list[i-1][[j for j in input_list[i-1].columns if j.startswith("fold_")]] for i in mge_at])
                #Check if clustering failed
                if opt_list == []:
                    return []
                #Replace columns with aligned data
                for i in range(len(opt_list)):
                    output_list[mge_at[i]-1][[j for j in output_list[mge_at[i]-1].columns if j.startswith("fold_")]] = opt_list[i]

        return output_list

    def get_additive_trait_weights(
        self,
        folds = None,
        grid_search = False,
        RT = None,
        seq_position_offset = 0,
        aggregate = False,
        aggregate_absolute_value = False,
        save = True):
        """
        Get additive trait weights.

        :param folds: list of cross-validation folds (default:None i.e. all).
        :param grid_search: Whether or not to include grid_search models (default:False).
        :param RT: R=gas constant (in kcal/K/mol) * T=Temperature (in K) (optional).
        :param seq_position_offset: Sequence position offset (default:0).
        :param aggregate: Whether or not to aggregate trait weights per reference position by error weighted averaging (default:False).
        :param aggregate_absolute_value: Whether or not to aggregate trait weights per reference position by error weighted average of the absolute value (default:False).
        :param save: Save linear, global and additive trait weights to "weights/" (default:True).
        :returns: A list of data frames (one per additive trait).
        """ 

        #Check if valid MochiTask
        if 'models' not in dir(self):
            print("Error: Cannot get additive trait weights. Invalid MochiTask instance.")
            raise ValueError

        #Output weights directory
        directory = os.path.join(self.directory, 'weights')

        #Create output weights directory
        if save:
            try:
                os.mkdir(directory)
            except FileExistsError:
                pass

        #Save linear weights
        if save:
            self.get_linear_weights(folds = folds, grid_search = grid_search)

        #Save linear weights
        if save:
            self.get_global_weights(folds = folds, grid_search = grid_search)

        #Set folds if not supplied
        if folds is None:
            folds = [i+1 for i in range(self.data.k_folds)]

        #Model subset
        models_subset = [i for i in self.models if ((i.metadata.grid_search==grid_search) & (i.metadata.fold in folds))]
        #Check if at least one model remaining
        if len(models_subset)==0:
            print("No models satisfying criteria.")
            return

        #Construct weight list
        at_list = []
        for j in range(len(models_subset[0].additivetraits)):
            at_list += [[]]
            for i in range(len(models_subset)):
                additivetrait_parameters = models_subset[i].additivetraits[j].parameters()
                additivetrait_parameters = [item for sublist in additivetrait_parameters for item in sublist]
                additivetrait_parameters = np.asarray(additivetrait_parameters[0].detach().cpu())
                #Subset mask to phenotypes reporting on this additive trait
                phenotypes_with_trait = [pindex for pindex in range(self.data.model_design.shape[0]) if (j+1) in self.data.model_design.loc[pindex,'trait']]
                
                mask = torch.reshape(torch.narrow(models_subset[i].mask, 0, j, 1), (self.data.model_design.shape[0], -1))
                mask = pd.DataFrame(np.asarray(mask.detach().cpu())).iloc[phenotypes_with_trait,:].sum(axis=0)
                
                #Weight data frame
                at_list[-1] += [pd.DataFrame({
                    "id": np.array(list(self.data.Xohi.columns)),
                    "id_ref": ['WT']+['_'.join([smi[:1]+str(int(smi[1:-1])+seq_position_offset)+smi[-1:] for smi in mi.split("_")]) for mi in list(self.data.Xohi.columns[1:])],
                    "Pos": [None]+['_'.join([str(int(smi[1:-1])) for smi in mi.split("_")]) for mi in list(self.data.Xohi.columns[1:])],
                    "Pos_ref": [None]+['_'.join([str(int(smi[1:-1])+seq_position_offset) for smi in mi.split("_")]) for mi in list(self.data.Xohi.columns[1:])],
                    "fold_"+str(models_subset[i].metadata.fold): additivetrait_parameters})]
                #Remove weights not reported on by a single corresponding phenotype
                at_list[-1][-1] = at_list[-1][-1].loc[mask!=0,:]
            #Merge weight data frames corresponding to different folds
            at_list[-1] = functools.reduce(lambda x, y: pd.merge(x, y, how='outer', on = ['id', 'id_ref', 'Pos', 'Pos_ref']), at_list[-1])
        #Calculate summary metrics for each additive trait (mean, std, ci95 etc.)
        at_list = self.add_additive_trait_statistics(at_list, RT)

        #Aggregate weights
        if aggregate==True:
            agg_list = []
            for i in range(len(at_list)):
                grouped = at_list[i].groupby("Pos_ref")
                agg_list += [pd.merge(
                        pd.DataFrame(grouped.apply(
                            self.wavg, 
                            absolute_value = aggregate_absolute_value, 
                            suffix = ['', '_kcal/mol'][int(RT!=None)])), 
                        pd.DataFrame(grouped.apply(
                            self.wavg, 
                            error_only = True,
                            suffix = ['', '_kcal/mol'][int(RT!=None)])), 
                    on = "Pos_ref")]
                agg_list[-1] = agg_list[-1].sort_values(
                    by="Pos_ref",
                    key=lambda x: np.array([int(i) for i in agg_list[-1].index]))
                agg_list[-1].columns = ['mean', 'sigma']
            #Save aggregated model weights
            file_prefix = ["weights_agg_", "weights_agg_abs_"][int(aggregate_absolute_value)]
            for i in range(len(agg_list)):
                if save:
                    agg_list[i].to_csv(os.path.join(directory, file_prefix+self.data.additive_trait_names[i]+".txt"), sep = "\t", index = False)
        #Save model weights
        for i in range(len(at_list)):
            if save:
                at_list[i].to_csv(os.path.join(directory, "weights_"+self.data.additive_trait_names[i]+".txt"), sep = "\t", index = False)
        
        #Align additive trait fold columns for inferred multidimensional global epistasis ('SumOfSigmoids')
        aat_list = self.align_additive_traits(at_list)
        #Save if alignment successful
        if aat_list != []:
            #Calculate summary metrics for each additive trait (mean, std, ci95 etc.)
            aat_list = self.add_additive_trait_statistics(aat_list, RT)
            #Save model weights
            for i in range(len(aat_list)):
                if save:
                    aat_list[i].to_csv(os.path.join(directory, "weights_aligned_"+self.data.additive_trait_names[i]+".txt"), sep = "\t", index = False)

        #Return
        if aggregate:
            return agg_list
        else:
            return at_list

    def grid_search(
        self,
        fold = 1,
        seed = 1,
        overwrite = False,
        init_weights = None,
        fix_weights = {}):
        """
        Perform grid search over supplied hyperparameters.

        :param fold: Cross-validation fold (default:1).
        :param seed: Random seed for both training target data resampling and shuffling training data (default:1).
        :param overwrite: Whether or not to overwrite previous grid search models (default:False).
        :param init_weights: Task to use for model weight initialisation (optional).
        :param fix_weights: Dictionary of layer names to fix weights (default:empty dict i.e. no layers fixed).
        :returns: Nothing.
        """ 

        #Check if valid MochiTask
        if 'models' not in dir(self):
            print("Error: Grid search cannot be performed. Invalid MochiTask instance.")
            raise ValueError

        #Check if valid MochiData
        if not self.data.is_valid_instance():
            print("Error: Grid search cannot be performed. Invalid MochiData instance.")
            raise ValueError

        #Check if grid search has already been performed
        grid_search_models = [i for i in self.models if (i.grid_search==True) and (i.fold==fold)]
        if len(grid_search_models)!=0:
            if overwrite==False:
                print("Grid search already performed. Set 'overwrite'=True to overwrite previous grid search results.")
                return
            else:
                print("Deleting previous grid search results.")
                [self.models.remove(i) for i in grid_search_models]

        print("Performing grid search...")
        #List of hyperparameter combinations
        batch_params = list(itertools.product(
            self.batch_size,
            self.learn_rate,
            self.l1_regularization_factor,
            self.l2_regularization_factor))
        #Fit one model for each hyperparameter combination
        try:
            for b in batch_params:
                self.fit(
                    fold  = fold,
                    seed = seed,
                    grid_search = True,
                    batch_size = b[0],
                    learn_rate = b[1],
                    num_epochs = self.num_epochs,
                    num_epochs_grid = self.num_epochs_grid,
                    l1_regularization_factor = b[2],
                    l2_regularization_factor = b[3],
                    training_resample = self.training_resample,
                    early_stopping = self.early_stopping,
                    scheduler_gamma = self.scheduler_gamma,
                    loss_function_name = self.loss_function_name,
                    sos_architecture = self.sos_architecture,
                    sos_outputlinear = self.sos_outputlinear,
                    init_weights = init_weights,
                    fix_weights = fix_weights)
        except ValueError:
            print("Error: Grid search failed.")
            raise ValueError

    def adjust_WT_phenotype(
        self,
        model_id,
        epoch_proportion = 0.1):
        """
        Adjust WT phenotype such that residual phenotype from specified model is zero.
        Warning: Use with caution as this method permanently changes the MochiData object.

        :param model_id: Index of model to use for adjustment (required).
        :param epoch_proportion: Proportion of final epochs over which to average predicted WT phenotype (default:0.1).
        :returns: Nothing.
        """ 
        if len(self.models)>0 and (model_id in range(len(self.models)) or model_id == -1):
            print("Adjusting WT using model: "+str(model_id))
            for i in range(len(self.data.model_design)):
                num_epochs_avg = int(self.models[model_id].metadata.num_epochs*epoch_proportion)
                WT_correct = sum(self.models[model_id].training_history['residual'+str(i+1)+'_WT'][-num_epochs_avg:])/num_epochs_avg
                self.data.fitness.loc[(self.data.fdata.vtable['WT']==True) & self.data.phenotypes['phenotype_'+str(i+1)]==True,'fitness'] += (-WT_correct)
        else:
            print("Error: Invalid model index for WT adjustment.")
            raise ValueError

    def fit_best(
        self,
        fold = 1,
        grid_search_fold = 1,
        seed = 1,
        epoch_proportion = 0.1,
        epoch_status = 10,
        init_weights = None,
        fix_weights = {}):
        """
        Fit model using best grid search hyperparameters.

        :param fold: Cross-validation fold (default:1).
        :param grid_search_fold: Cross-validation fold of grid search models (default:1).
        :param seed: Random seed for both training target data resampling and shuffling training data (default:1).
        :param epoch_proportion: Proportion of final epochs over which to average grid search validation loss (default:0.1).
        :param epoch_status: Number of training epochs after which to print status messages (default:10).
        :param init_weights: Task to use for model weight initialisation (optional).
        :param fix_weights: Dictionary of layer names to fix weights (default:empty dict i.e. no layers fixed).
        :returns: Nothing.
        """ 

        #Check if valid MochiTask
        if 'models' not in dir(self):
            print("Error: Model cannot be fit. Invalid MochiTask instance.")
            raise ValueError

        #Check if valid MochiData
        if not self.data.is_valid_instance():
            print("Error: Model cannot be fit. Invalid MochiData instance.")
            raise ValueError

        #Check if grid search models present
        grid_search_models = [i for i in self.models if (i.metadata.grid_search==True) and (i.metadata.fold==grid_search_fold)]
        if len(grid_search_models)==0:
            print("Error: No grid search models available.")
            raise ValueError

        #Grid search model with best performance
        perf_list = np.asarray([sum(i.training_history['val_loss'][-int(i.metadata.num_epochs_grid*epoch_proportion):])/int(i.metadata.num_epochs_grid*epoch_proportion) for i in grid_search_models])
        #Check that at least one grid search model validation loss isn't NA
        if len(perf_list[~np.isnan(perf_list)])!=0:
            best_model_index = [i for i in range(len(perf_list)) if perf_list[i]==np.nanmin(perf_list)][0]
        else:
            print("Error: No valid grid search models available.")
            raise ValueError

        #Print best grid search model
        print("Best model:")
        print(grid_search_models[best_model_index].metadata)

        #Fit model using best hyperparameters
        self.fit(
            fold = fold,
            seed = seed,
            batch_size = grid_search_models[best_model_index].metadata.batch_size,
            learn_rate = grid_search_models[best_model_index].metadata.learn_rate,
            num_epochs = self.num_epochs,
            l1_regularization_factor = grid_search_models[best_model_index].metadata.l1_regularization_factor,
            l2_regularization_factor = grid_search_models[best_model_index].metadata.l2_regularization_factor,
            epoch_status = epoch_status,
            training_resample = self.training_resample,
            early_stopping = self.early_stopping,
            scheduler_gamma = self.scheduler_gamma,
            loss_function_name = self.loss_function_name,
            sos_architecture = self.sos_architecture,
            sos_outputlinear = self.sos_outputlinear,
            init_weights = init_weights,
            fix_weights = fix_weights)

    def weights_init_task(
        self,
        fold,
        model,
        input_task):
        """
        Initialize layer weights from supplied MochiTask instance.

        :param fold: Cross-validation fold (required).
        :param model: MochiModel to initialize layer weights (required).
        :param input_task: MochiTask to use for model weight initialization (required).
        :returns: Nothing.
        """ 

        #Check if valid MochiTask
        if 'models' not in dir(input_task):
            print("Error: Cannot initialize weights using supplied MochiTask. Invalid MochiTask instance.")
            raise ValueError

        #Model subset
        models_subset = [i for i in input_task.models if ((i.metadata.grid_search==False) & (i.metadata.fold==fold))]

        #Check if at least one model remaining
        if len(models_subset)==0:
            print("Error: Cannot initialize weights using supplied MochiTask. No models satisfying criteria.")
            raise ValueError

        #Check if more than one model remaining
        if len(models_subset)>1:
            print("Warning: Initialising weights using first model satisfying criteria.")

        #Model to use for model weight initialization 
        input_model = models_subset[0]
        #Find shared layers based on names
        shared_name = [i[0] for i in input_model.named_children() if i[0] in [j[0] for j in model.named_children()]]
        #Find shared layers based on names + type
        shared_name_type = [i for i in shared_name if type(input_model.get_submodule(i)) == type(model.get_submodule(i))]

        #Additive traits - copy according to additive trait names and feature names
        n_init_addt = 0
        if 'additivetraits' in shared_name_type:
            model_addt = model.get_submodule('additivetraits')
            input_addt = input_model.get_submodule('additivetraits')
            shared_traits = [i for i in self.data.additive_trait_names if i in input_task.data.additive_trait_names]
            shared_weights = [i for i in self.data.Xohi.columns if i in input_task.data.Xohi.columns]
            #Loop over shared traits
            for tname in shared_traits:
                model_ti = self.data.additive_trait_names.index(tname)
                input_ti = input_task.data.additive_trait_names.index(tname)
                #Model weights
                model_weights = model_addt[model_ti].parameters()
                model_weights = [item for sublist in model_weights for item in sublist]
                model_weights = np.asarray(model_weights[0].detach().cpu())
                #Input model weights
                input_weights = input_addt[input_ti].parameters()
                input_weights = [item for sublist in input_weights for item in sublist]
                input_weights = np.asarray(input_weights[0].detach().cpu())
                #Model weight initialization
                model_weights = [input_weights[list(input_task.data.Xohi.columns).index(i)] if i in shared_weights else model_weights[list(self.data.Xohi.columns).index(i)] for i in self.data.Xohi.columns]
                with torch.no_grad():
                    model_addt[model_ti].weight = torch.nn.Parameter(torch.tensor(np.asarray([model_weights])))
                n_init_addt += 1

        #Global parameters - copy according to phenotype names and dictionary keys
        n_init_glob = 0
        if 'globalparams' in shared_name_type:
            model_glob = model.get_submodule('globalparams')
            input_glob = input_model.get_submodule('globalparams')
            #Loop over all phenotypes
            for pname in self.data.phenotype_names:
                #Shared phenotype
                if pname in input_task.data.phenotype_names:
                    model_pi = self.data.phenotype_names.index(pname)
                    input_pi = input_task.data.phenotype_names.index(pname)
                    shared_keys = [k for k in model_glob[model_pi].keys() if k in input_glob[input_pi].keys()]
                    #Check if at least one global parameter matches
                    if len(shared_keys) == 0:
                        print("Warning: No shared global parameters for phenotype: "+pname)
                    #Copy all shared parameters
                    for k in shared_keys:
                        with torch.no_grad():
                            model_glob[model_pi][k] = copy.deepcopy(input_glob[input_pi][k])
                        n_init_glob += 1

        #Sigmoidal layers - copy according to phenotype names
        n_init_sigm = 0
        model_sigp = [input_task.data.phenotype_names[i] for i in range(len(input_task.data.model_design)) if input_task.data.model_design.loc[i,'transformation']=="SumOfSigmoids"]
        input_sigp = [self.data.phenotype_names[i] for i in range(len(self.data.model_design)) if self.data.model_design.loc[i,'transformation']=="SumOfSigmoids"]
        #Shared phenotypes with sigmoidal transformations
        shared_sigp = [i for i in model_sigp if i in model_sigp]
        for pname in shared_sigp:
            model_pi = model_sigp.index(pname)
            input_pi = input_sigp.index(pname)
            #Loop over all sigmoidal layers
            for layer_name in shared_name_type:
                if layer_name.startswith('sumofsigmoids'):
                    model_sigm = model.get_submodule(layer_name)
                    input_sigm = input_model.get_submodule(layer_name)
                    #Copy whole layer
                    with torch.no_grad():
                        model_sigm[model_pi] = copy.deepcopy(input_sigm[input_pi])
                    n_init_sigm += 1

        #Linear parameters - copy according to phenotype names
        n_init_lins = 0
        if 'linears' in shared_name_type:
            model_lins = model.get_submodule('linears')
            input_lins = input_model.get_submodule('linears')
            #Loop over all phenotypes
            for pname in self.data.phenotype_names:
                #Shared phenotype
                if pname in input_task.data.phenotype_names:
                    model_pi = self.data.phenotype_names.index(pname)
                    input_pi = input_task.data.phenotype_names.index(pname)
                    #Copy whole layer
                    with torch.no_grad():
                        model_lins[model_pi] = copy.deepcopy(input_lins[input_pi])
                    n_init_lins += 1

        #Total number of weights initialized
        print("Weights initialized from MochiTask:")
        print("Additive layers = "+str(n_init_addt)+"\nGlobal parameters = "+str(n_init_glob)+"\nSigmoidal layers = "+str(n_init_sigm)+"\nLinear layers = "+str(n_init_lins))

    def weights_require_grad(
        self,
        model,
        fix_weights):
        """
        Fix specific layer weights.

        :param model: MochiModel to fix specific layer weights (required).
        :param fix_weights: Dictionary of layer names to fix weights (required).
        :returns: Nothing.
        """ 

        #Check if dictionary has valid keys
        if sum([i for i in fix_weights.keys() if i not in ['phenotype', 'trait', 'global']])!=0:
            print("Error: Invalid fix_weights argument: layer types must be either 'phenotype', 'trait' or 'global'.")
            raise ValueError

        #Check if dictionary has valid values
        #Phenotypes
        if 'phenotype' in fix_weights.keys():
            if sum([i for i in fix_weights['phenotype'] if i not in self.data.phenotype_names])!=0:
                print("Error: Invalid fix_weights phenotype names.")
                raise ValueError
        #Traits
        if 'trait' in fix_weights.keys():
            if sum([i for i in fix_weights['trait'] if i not in self.data.additive_trait_names])!=0:
                print("Error: Invalid fix_weights trait names.")
                raise ValueError
        # #Global parameters
        # fix_weights['global']
        # if sum([i for i in fix_weights['global'] if i not in self.data.phenotype_names])!=0:
        #     print("Error: Invalid fix_weights phenotype names.")
        #     raise ValueError

        #Linear parameters
        if 'phenotype' in fix_weights.keys():
            for pname in fix_weights['phenotype']:
                model_pi = self.data.phenotype_names.index(pname)
                model_lins = model.get_submodule('linears')
                model_lins[model_pi].requires_grad_(False)

        #Traits
        if 'trait' in fix_weights.keys():
            for tname in fix_weights['trait']:
                model_ti = self.data.additive_trait_names.index(tname)
                model_addt = model.get_submodule('additivetraits')
                model_addt[model_ti].requires_grad_(False)

    def fit(
        self, 
        fold = 1,
        seed = 1,
        grid_search = False,
        batch_size = 512,
        learn_rate = 0.05,
        num_epochs = 300,
        num_epochs_grid = 100,
        l1_regularization_factor = 0,
        l2_regularization_factor = 0,
        epoch_status = 10,
        training_resample = True,
        early_stopping = True,
        scheduler_gamma = 0.98,
        scheduler_epochs = 10,
        loss_function_name = 'WeightedL1',
        sos_architecture = [20],
        sos_outputlinear = False,
        init_weights = None,
        fix_weights = {}):
        """
        Fit model.

        :param fold: Cross-validation fold (default:1).
        :param seed: Random seed for both training target data resampling and shuffling training data (default:1).
        :param grid_search: Whether or not this model is part of a grid search (default:False).
        :param batch_size: Minibatch size (default:512).
        :param learn_rate: Learning rate (default:0.05).
        :param num_epochs: Number of training epochs (default:300).
        :param num_epochs_grid: Number of grid search epochs (default:100).
        :param l1_regularization_factor: Lambda factor applied to L1 norm (default:0).
        :param l2_regularization_factor: Lambda factor applied to L2 norm (default:0).
        :param epoch_status: Number of training epochs after which to print status messages (default:10).
        :param training_resample: Whether or not to add random noise to training target data proportional to target error (default:True).
        :param early_stopping: Whether or not to stop training early if validation loss not decreasing (default:True).
        :param scheduler_gamma: Multiplicative factor of learning rate decay (default:0.98).
        :param scheduler_epochs: Number of epochs over which to evaluate scheduler criteria (default:10).
        :param loss_function_name: Loss function name (default:'WeightedL1').
        :param sos_architecture: list of integers corresponding to number of neurons per fully-connected sumOfSigmoids hidden layer (default:[20]).
        :param sos_outputlinear: boolean indicating whether final sumOfSigmoids should be linear rather than sigmoidal (default:False).
        :param init_weights: Task to use for model weight initialization (optional).
        :param fix_weights: Dictionary of layer names to fix weights (default:empty dict i.e. no layers fixed).
        :returns: Nothing.
        """ 

        #Check if valid MochiTask
        if 'models' not in dir(self):
            print("Error: Model cannot be fit. Invalid MochiTask instance.")
            raise ValueError

        #Check if valid MochiData
        if not self.data.is_valid_instance():
            print("Error: Model cannot be fit. Invalid MochiData instance.")
            raise ValueError

        #Load model data
        model_data = self.data.get_data(
            fold = fold, 
            seed = seed,
            training_resample = training_resample)

        #Load WT model data
        model_data_WT = self.data.get_data_index(
            indices = list(self.data.fdata.vtable.loc[self.data.fdata.vtable['WT']==True,:].index))

        #Set random seeds
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)

        #Instantiate model
        self.models += [self.new_model(model_data)]
        model = self.models[-1]

        #initialize model weights
        grouped = self.data.fdata.vtable.loc[:,['fitness', 'phenotype']].groupby("phenotype")
        model.weights_init_fill(
            linear_weight = list(grouped.apply(np.quantile, 0.9) - grouped.apply(np.quantile, 0.1)),
            linear_bias = list(grouped.apply(np.quantile, 0.1)))

        #initialize model weights from supplied object (init_weights)
        if type(init_weights) == MochiTask:
            self.weights_init_task(
                fold = fold,
                model = model,
                input_task = init_weights)

        #Fix weights
        self.weights_require_grad(
            model = model,
            fix_weights = fix_weights)

        #Model metadata
        model.metadata = MochiModelMetadata(
            fold = fold,
            seed = seed,
            grid_search = grid_search,
            batch_size = batch_size,
            learn_rate = learn_rate,
            num_epochs = num_epochs,
            num_epochs_grid = num_epochs_grid,
            l1_regularization_factor = l1_regularization_factor,
            l2_regularization_factor = l2_regularization_factor,
            training_resample = training_resample,
            early_stopping = early_stopping,
            scheduler_gamma = scheduler_gamma,
            scheduler_epochs = scheduler_epochs,
            loss_function_name = loss_function_name,
            sos_architecture = sos_architecture,
            sos_outputlinear = sos_outputlinear)
        print("Fitting model:")
        print(model.metadata)

        #Load model data
        train_dataloader, valid_dataloader, test_dataloader = model.get_data_loaders(model_data, batch_size)

        #Construct loss function and Optimizer
        if loss_function_name == 'WeightedL1':
            loss_function = MochiWeightedL1Loss()
        elif loss_function_name == 'GaussianNLL':
            loss_function = MochiGaussianNLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

        #Scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)

        #Train
        total_epochs = num_epochs
        if grid_search:
            total_epochs = num_epochs_grid
        for epoch in range(total_epochs):
            model.train_model(
                train_dataloader, 
                loss_function, 
                optimizer,
                self.device, 
                l1_regularization_factor, 
                l2_regularization_factor)
            model.validate_model(
                valid_dataloader, 
                loss_function, 
                self.device, 
                l1_regularization_factor, 
                l2_regularization_factor, 
                model_data_WT)

            #Check scheduler and early-stopping criteria
            if epoch >= (2*scheduler_epochs):
                #If loss not decreasing
                if min(model.training_history['val_loss'][-scheduler_epochs:]) >= min(model.training_history['val_loss'][-(2*scheduler_epochs):-scheduler_epochs]):
                    #...and WT coefficients not changing, stop training
                    if sum([np.std(model.training_history['additivetrait'+str(i+1)+"_WT"][-scheduler_epochs:])>0.001 for i in range(len(model.additivetraits))])==0:
                        if early_stopping:
                            print("Stopping early: coefficients not changing and validation loss not decreasing.")
                            break
                    #Decrease learning rate by gamma
                    scheduler.step()

            #Status
            if epoch % epoch_status == 0: 
                model_status = model.get_status()
                print(f"Epoch {epoch+1}; "+model_status)    
        print("Done!")

    def predict_all(
        self,
        folds = None,
        grid_search = False,
        data = None,
        save = True):
        """
        Model predictions on all data.

        :param output_path: Output file path (required).
        :param folds: list of cross-validation folds (default:None i.e. all).
        :param grid_search: Whether or not to include grid_search models (default:False).
        :param data: Optional MochiData object to use for prediction (default:model data).
        :param save: Save DataFrame to "predictions/predicted_phenotypes_all.txt" (default:True).
        :returns: DataFrame of variants with phenotypes predictions.
        """ 

        #Check if valid MochiTask
        if 'models' not in dir(self):
            print("Error: Cannot make predictions. Invalid MochiTask instance.")
            raise ValueError

        #Check if valid MochiData
        if not self.data.is_valid_instance():
            print("Error: Cannot make predictions. Invalid MochiData instance.")
            raise ValueError

        #Output predictions directory
        directory = os.path.join(self.directory, 'predictions')

        #Create output predictions directory
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass

        #Set folds if not supplied
        if folds is None:
            folds = [i+1 for i in range(self.data.k_folds)]

        #Model subset
        models_subset = [i for i in self.models if ((i.metadata.grid_search==grid_search) & (i.metadata.fold in folds))]
        #Check if at least one model remaining
        if len(models_subset)==0:
            print("No models satisfying criteria.")
            return

        #Model data
        if data is None:
            data = self.data
        model_data = data.get_data_index()
        model_data_loader = FastTensorDataLoader(
            model_data["select"], 
            model_data["X"], batch_size=1024, shuffle=False)

        #Predictions on all data
        pred_list = []
        at_list = []
        for i in range(len(models_subset)):
            model = models_subset[i]
            #Model predictions
            max_at = max([len(j) for j in model.model_design.trait])
            mask = model.mask.to(self.device)
            # mask_list = [torch.narrow(mask, 0, j, 1) for j in range(mask.shape[0])]
            y_pred_list = []
            at_pred_list = []
            for select, X in model_data_loader:
                select, X = select.to(self.device), X.to(self.device)
                #Predicted phenotype
                y_pred_list += [model(select, X, mask).detach().cpu().numpy().flatten()]

                #Additive traits
                additive_traits = []
                select_list = [torch.narrow(select, 1, j, 1) for j in range(select.shape[1])]
                #Loop over phenotypes
                for pi in range(len(model.model_design)):
                    #Additive traits for this phenotype
                    additive_traits_p = []
                    for at_pi in range(max_at):
                        if at_pi<len(model.model_design.loc[pi,'trait']):
                            # additive_traits_p += [model.additivetraits[model.model_design.loc[pi,'trait'][at_pi]-1](torch.mul(X, mask_list[pi]))]
                            additive_traits_p += [model.additivetraits[model.model_design.loc[pi,'trait'][at_pi]-1](torch.mul(
                                X, 
                                torch.reshape(torch.narrow(torch.narrow(mask, 0, model.model_design.loc[pi,'trait'][at_pi]-1, 1), 1, pi, 1), (1, -1))))]
                        else:
                            additive_traits_p += [torch.mul(torch.clone(additive_traits_p[0]),0)]
                    #Select
                    additive_traits += [torch.concat([torch.mul(j, select_list[pi]) for j in additive_traits_p], dim=1)]
                at_pred_list += [sum(additive_traits).detach().cpu().numpy()]
            #Concatenate
            y_pred = np.concatenate(y_pred_list)
            at_pred = np.concatenate(at_pred_list)
            #Target data frame
            y_df = pd.DataFrame({
                'fold_'+str(model.metadata.fold): y_pred})
            y_df.reset_index(drop = True, inplace = True)
            pred_list += [y_df]
            #Additive trait data frame
            additive_trait_df = pd.DataFrame(at_pred, 
                columns = ['fold_'+str(model.metadata.fold)+'_additive_trait'+str(j) for j in range(at_pred.shape[1])])
            additive_trait_df.reset_index(drop = True, inplace = True)
            at_list += [additive_trait_df]
        #Merge predictions and additive traits from all models
        pred_df = pd.concat(pred_list, axis=1)
        at_df = pd.concat(at_list, axis=1)
        fold_cols = pred_df.columns
        pred_df['mean'] = pred_df.loc[:,fold_cols].mean(axis=1)
        pred_df['std'] = pred_df.loc[:,fold_cols].std(axis=1)
        pred_df['ci95'] = pred_df['std']*1.96*2
        #Select data frame
        select_df = pd.DataFrame(
            model_data["select"].detach().cpu().numpy(), 
            columns = data.phenotype_names)
        #Fold data frame
        fold_df = pd.DataFrame({
            'Fold': np.asarray(data.cvgroups.fold)})
        #Variant data
        v_df = data.fdata.vtable
        v_df.reset_index(drop = True, inplace = True)
        select_df.reset_index(drop = True, inplace = True)
        pred_df.reset_index(drop = True, inplace = True)
        fold_df.reset_index(drop = True, inplace = True)
        at_df.reset_index(drop = True, inplace = True)

        #Merge results from all models
        result_df = pd.concat([v_df, select_df, pred_df, fold_df, at_df], axis=1)

        #Save
        if save:
            result_df.to_csv(os.path.join(directory, "predicted_phenotypes_all.txt"), sep = "\t", index = False)
        return result_df

