
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
from functools import reduce

class ConstrainedLinear(torch.nn.Linear):
    """
    A linear layer constrained to positive weights only.
    """
    def forward(
        self, 
        input):
        return F.linear(input, self.weight.clamp(min=0, max=1000), self.bias)

class MochiModel(torch.nn.Module):
    """
    A custom model/module.
    """
    def __init__(
        self, 
        input_shape, 
        mask, 
        model_design):
        """
        Initialize a MochiModel object.

        :param input_shape: shape of input data (required).
        :param mask: tensor to mask weights that should not be fit (required).
        :param model_design: Model design data frame with phenotype, transformation and trait columns (required).
        :returns: MochiModel object.
        """     
        super(MochiModel, self).__init__()
        #Model design
        self.model_design = model_design
        #Additive traits
        n_additivetraits = len(list(set([item for sublist in list(self.model_design['trait']) for item in sublist])))
        self.additivetraits = torch.nn.ModuleList([torch.nn.Linear(input_shape, 1, bias = False, dtype=torch.float32) for i in range(n_additivetraits)])

        #Global trainable parameters
        self.globalparams = torch.nn.ModuleList(
            [torch.nn.ParameterDict({j:torch.nn.Parameter(torch.tensor(1.0), requires_grad=True) for j in get_transformation(i)().keys()}) for i in self.model_design.transformation])

        #Arbitrary non-linear transformations (depth=3)
        n_sumofsigmoids = len([i for i in self.model_design.transformation if i=="SumOfSigmoids"])
        self.model_design.loc[self.model_design.transformation=="SumOfSigmoids",'s_index'] = list(range(n_sumofsigmoids))
        self.sumofsigmoids1 = torch.nn.ModuleList([torch.nn.Linear(len(self.model_design.loc[i,'trait']), 20, dtype=torch.float32) for i in range(len(self.model_design)) if self.model_design.loc[i,'transformation']=="SumOfSigmoids"])
        self.sumofsigmoids2 = torch.nn.ModuleList([torch.nn.Linear(20, 10, dtype=torch.float32) for i in range(n_sumofsigmoids)])
        self.sumofsigmoids3 = torch.nn.ModuleList([torch.nn.Linear(10, 5, dtype=torch.float32) for i in range(n_sumofsigmoids)])
        self.sumofsigmoids4 = torch.nn.ModuleList([torch.nn.Linear(5, 1, dtype=torch.float32) for i in range(n_sumofsigmoids)])
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
        #Split mask tensor into list
        mask_list = [torch.narrow(mask, 0, i, 1) for i in range(mask.shape[0])]
        #Loop over all phenotypes
        observed_phenotypes = []
        for i in range(len(self.model_design)):
            #Additive traits
            additive_traits = [self.additivetraits[j-1](torch.mul(X, mask_list[i])) for j in self.model_design.loc[i,'trait']]
            #Molecular phenotypes
            if self.model_design.loc[i,'transformation']!="SumOfSigmoids":
                globalparams = self.globalparams[i]
                transformed_trait = get_transformation(self.model_design.loc[i,'transformation'])(additive_traits, globalparams)           
            else:
                transformed_trait1 = torch.sigmoid(self.sumofsigmoids1[int(self.model_design.loc[i,'s_index'])](torch.cat(additive_traits, 1)))
                transformed_trait2 = torch.sigmoid(self.sumofsigmoids2[int(self.model_design.loc[i,'s_index'])](transformed_trait1))
                transformed_trait3 = torch.sigmoid(self.sumofsigmoids3[int(self.model_design.loc[i,'s_index'])](transformed_trait2))
                transformed_trait = self.sumofsigmoids4[int(self.model_design.loc[i,'s_index'])](transformed_trait3)
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
        #Initialize ConstrainedLinear layer weights
        for i in range(len(self.model_design)):
            # self.linears[i].weight.data.fill_(1)
            # self.linears[i].bias.data.fill_(0)
            self.linears[i].weight.data.fill_(linear_weight[i])
            self.linears[i].bias.data.fill_(linear_bias[i])

    def load_data(
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
        return (l1_norm, l2_norm)

    def train_model(
        self, 
        dataloader, 
        loss_fn, 
        optimizer, 
        device, 
        l1_lambda = 0, 
        l2_lambda = 0):
        """
        Train model.

        :param dataloader: Dataloader (required).
        :param loss_fn: Loss function (required).
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
            loss = sum(loss_fn(pred, y) * y_wt)/len(y) + l1_lambda * l1_norm + l2_lambda * l2_norm
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
  
    def validate_model(
        self, 
        dataloader, 
        loss_fn, 
        device, 
        l1_lambda = 0, 
        l2_lambda = 0, 
        data_WT = None):
        """
        Validate model.

        :param dataloader: Dataloader (required).
        :param loss_fn: Loss function (required).
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
                val_loss += (sum(loss_fn(pred, y) * y_wt)/len(y) + l1_lambda * l1_norm + l2_lambda * l2_norm).item()
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
        scheduler_epochs):
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
        num_epochs = 300,
        num_epochs_grid = 100,
        l1_regularization_factor = 0,
        l2_regularization_factor = 0,
        scheduler_gamma = 0.98):
        """
        Initialize a MochiTask object.

        :param directory: Path to directory where models and results should be saved/loaded (required).
        :param data: An instance of the MochiData class (required unless 'directory' contains a saved task).
        :param batch_size: Minibatch size (default:512).
        :param learn_rate: Learning rate (default:0.05).
        :param num_epochs: Number of training epochs (default:300).
        :param num_epochs_grid: Number of grid search epochs (default:100).
        :param l1_regularization_factor: Lambda factor applied to L1 norm (default:0).
        :param l2_regularization_factor: Lambda factor applied to L2 norm (default:0).
        :param scheduler_gamma: Multiplicative factor of learning rate decay (default:0.98).
        :returns: MochiTask object.
        """ 
        #Get CPU or GPU device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # print(f"Using {self.device} device")
        #Initialise remaining attributes
        self.directory = directory
        if data != None:
            #Create task directory
            try:
                os.mkdir(self.directory)
            except FileExistsError:
                print("Error: Task directory already exists.")
                return
            self.models = []
            self.data = data
            self.batch_size = [int(i) for i in str(batch_size).split(",")]
            self.learn_rate = [float(i) for i in str(learn_rate).split(",")]
            self.num_epochs = num_epochs
            self.num_epochs_grid = num_epochs_grid
            self.l1_regularization_factor = [float(i) for i in str(l1_regularization_factor).split(",")]
            self.l2_regularization_factor = [float(i) for i in str(l2_regularization_factor).split(",")]
            self.scheduler_gamma = scheduler_gamma
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
            print("Error: Task cannot be saved. Not a valid MochiTask.")
            return

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
                return

        #Delete entire directory contents and create fresh directory
        shutil.rmtree(directory)
        os.mkdir(directory)

        #Save models using torch.save
        if len(self.models)==0:
            print("Warning: No fit models. Saving metadata only.")
        for i in range(len(self.models)):
            torch.save(self.models[i], os.path.join(directory, 'model_'+str(i)+'.pth'))

        #Save remaining (non-built-in) attributes
        save_dict = {}
        for i in self.__dict__.keys():
            #Exclude built-in objects and torch models
            if not i.startswith("__") and i != "models":
                save_dict[i] = self.__dict__[i]
        with bz2.BZ2File(os.path.join(directory, 'data.pbz2'), 'w') as f:
            cPickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)

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
            return

        #All files in directory
        files = os.listdir(directory)

        #Check data exists (no models required)
        if not ('data.pyc' in files or 'data.pbz2' in files):
            print("Error: Saved models directory structure incorrect.")
            return

        #Load models using torch.load
        self.models = [None]*len([i for i in files if i.startswith("model_")])
        for i in files:
            if i.startswith("model_"):
                model_index = int(i[6:(len(i)-4)])
                if model_index >= len(self.models) or model_index < 0:
                    print("Error: Saved models index format incorrect.")
                    return
                self.models[model_index] = torch.load(os.path.join(directory, i), map_location=self.device)
                #Add globalparams if legacy model
                if 'globalparams' not in dir(self.models[model_index]):
                    self.models[model_index].globalparams = torch.nn.ModuleList([torch.nn.ParameterDict({}) for i in self.models[model_index].model_design.transformation])


        #Load remaining (non-built-in) attributes
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

    def new_model(
        self,
        data):
        """
        Create a new MochiModel object.

        :param data: Dictionary of dictionaries of tensors as output by MochiData.get_data (required).
        :returns: A new MochiModel object.
        """ 
        model = MochiModel(
            input_shape = data['training']['X'].shape[1],
            mask = data['training']['mask'],
            model_design = self.data.model_design).to(self.device)
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
            print("Error: Cannot get global weights. Not a valid MochiTask.")
            return

        #Output weights directory
        directory = os.path.join(self.directory, 'weights')

        #Create output weights directory
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass

        #Set folds if not supplied
        if folds==None:
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
            print("Error: Cannot get linear weights. Not a valid MochiTask.")
            return

        #Output weights directory
        directory = os.path.join(self.directory, 'weights')

        #Create output weights directory
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass

        #Set folds if not supplied
        if folds==None:
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
                    [linear_parameters[0][0][0].detach().cpu().numpy(), linear_parameters[1][0].detach().cpu().numpy()]]
            at_list += [pd.DataFrame(param_list, columns = ['fold', 'kernel', 'bias'])]

        #Save model weights
        for i in range(len(at_list)):
            at_list[i].to_csv(os.path.join(directory, "linears_weights_"+self.data.phenotype_names[i]+".txt"), sep = "\t", index = False)

    def get_additive_trait_weights(
        self,
        folds = None,
        grid_search = False,
        RT = None,
        seq_position_offset = 0,
        aggregate = False,
        aggregate_absolute_value = False):
        """
        Get additive trait weights.

        :param folds: list of cross-validation folds (default:None i.e. all).
        :param grid_search: Whether or not to include grid_search models (default:False).
        :param RT: R=gas constant (in kcal/K/mol) * T=Temperature (in K) (optional).
        :param seq_position_offset: Sequence position offset (default:0).
        :param aggregate: Whether or not to aggregate trait weights per reference position by error weighted averaging (default:False).
        :param aggregate_absolute_value: Whether or not to aggregate trait weights per reference position by error weighted average of the absolute value (default:False).
        :returns: A list of data frames (one per additive trait).
        """ 

        #Check if valid MochiTask
        if 'models' not in dir(self):
            print("Error: Cannot get additive trait weights. Not a valid MochiTask.")
            return

        #Output weights directory
        directory = os.path.join(self.directory, 'weights')

        #Create output weights directory
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass

        #Save linear weights
        self.get_linear_weights(folds = folds, grid_search = grid_search)

        #Save linear weights
        self.get_global_weights(folds = folds, grid_search = grid_search)

        #Set folds if not supplied
        if folds==None:
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
                mask = pd.DataFrame(np.asarray(models_subset[i].mask.detach().cpu())).iloc[phenotypes_with_trait,:].sum(axis=0)
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
            at_list[-1] = reduce(lambda x, y: pd.merge(x, y, how='outer', on = ['id', 'id_ref', 'Pos', 'Pos_ref']), at_list[-1])
            fold_cols = [i for i in list(at_list[-1].columns) if not i in ['id', 'id_ref', 'Pos', 'Pos_ref']]
            at_list[-1]['n'] = at_list[-1].loc[:,fold_cols].notnull().sum(axis=1)
            at_list[-1]['mean'] = at_list[-1].loc[:,fold_cols].mean(axis=1)
            at_list[-1]['std'] = at_list[-1].loc[:,fold_cols].std(axis=1)
            at_list[-1]['ci95'] = at_list[-1]['std']*1.96*2
            at_list[-1]['trait_name'] = self.data.additive_trait_names[j]
            if RT!=None:
                at_list[-1]['mean_kcal/mol'] = at_list[-1]['mean']*RT
                at_list[-1]['std_kcal/mol'] = at_list[-1]['std']*RT
                at_list[-1]['ci95_kcal/mol'] = at_list[-1]['std_kcal/mol']*1.96*2

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
                agg_list[i].to_csv(os.path.join(directory, file_prefix+self.data.additive_trait_names[i]+".txt"), sep = "\t", index = False)
        #Save model weights
        for i in range(len(at_list)):
            at_list[i].to_csv(os.path.join(directory, "weights_"+self.data.additive_trait_names[i]+".txt"), sep = "\t", index = False)
        #Return
        if aggregate:
            return agg_list
        else:
            return at_list

    def grid_search(
        self,
        fold = 1,
        seed = 1,
        overwrite = False):
        """
        Perform grid search over supplied hyperparameters.

        :param fold: Cross-validation fold (default:1).
        :param seed: Random seed for both training target data resampling and shuffling training data (default:1).
        :param overwrite: Whether or not to overwrite previous grid search models (default:False).
        :returns: Nothing.
        """ 

        #Check if valid MochiTask
        if 'models' not in dir(self):
            print("Error: Grid search cannot be performed. Not a valid MochiTask.")
            return

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
                scheduler_gamma = self.scheduler_gamma)

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

    def fit_best(
        self,
        fold = 1,
        grid_search_fold = 1,
        seed = 1,
        epoch_proportion = 0.1,
        input_model = None,
        cold = True,
        epoch_status = 10):
        """
        Fit model using best grid search hyperparameters.

        :param fold: Cross-validation fold (default:1).
        :param grid_search_fold: Cross-validation fold of grid search models (default:1).
        :param seed: Random seed for both training target data resampling and shuffling training data (default:1).
        :param epoch_proportion: Proportion of final epochs over which to average grid search validation loss (default:0.1).
        :param input_model: Model with which to continue training (optional).
        :param cold: Whether or not to reinitialise model weights (default:True).
        :param epoch_status: Number of training epochs after which to print status messages (default:10).
        :returns: Nothing.
        """ 

        #Check if valid MochiTask
        if 'models' not in dir(self):
            print("Error: Model cannot be fit. Not a valid MochiTask.")
            return

        #Check if grid search models present
        grid_search_models = [i for i in self.models if (i.metadata.grid_search==True) and (i.metadata.fold==grid_search_fold)]
        if len(grid_search_models)==0:
            print(f"Error: No grid search models available.")
            return

        #Grid search model with best performance
        perf_list = np.asarray([sum(i.training_history['val_loss'][-int(i.metadata.num_epochs_grid*epoch_proportion):])/int(i.metadata.num_epochs_grid*epoch_proportion) for i in grid_search_models])
        #Check that at least one grid search model validation loss isn't NA
        if len(perf_list[~np.isnan(perf_list)])!=0:
            best_model_index = [i for i in range(len(perf_list)) if perf_list[i]==np.nanmin(perf_list)][0]
        else:
            print(f"Error: No valid grid search models available.")
            return

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
            scheduler_gamma = self.scheduler_gamma,
            input_model = input_model,
            cold = cold,
            epoch_status = epoch_status)

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
        input_model = None,
        cold = True,
        epoch_status = 10,
        training_resample = True,
        early_stopping = True,
        scheduler_gamma = 0.98,
        scheduler_epochs = 10):
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
        :param input_model: Model with which to continue training (optional).
        :param cold: Whether or not to reinitialise model weights (default:True).
        :param epoch_status: Number of training epochs after which to print status messages (default:10).
        :param training_resample: Whether or not to add random noise to training target data proportional to target error (default:True).
        :param early_stopping: Whether or not to stop training early if validation loss not decreasing (default:True).
        :param scheduler_gamma: Multiplicative factor of learning rate decay (default:0.98).
        :param scheduler_epochs: Number of epochs over which to evaluate scheduler criteria (default:10).
        :returns: Nothing.
        """ 

        #Check if valid MochiTask
        if 'models' not in dir(self):
            print("Error: Model cannot be fit. Not a valid MochiTask.")
            return

        #Load model data
        model_data = self.data.get_data(
            fold = fold, 
            seed = seed,
            training_resample = training_resample)

        #Check for data to fit model
        if model_data == None:
            print("Error: No data to fit model.")
            return

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
        model = None
        if input_model!=None:
            self.models += [copy.deepcopy(input_model)]
            model = self.models[-1]
        else:
            self.models += [self.new_model(model_data)]
            model = self.models[-1]

        #Initialise model weights
        if cold==True or input_model==None:
            grouped = self.data.fdata.vtable.loc[:,['fitness', 'phenotype']].groupby("phenotype")
            model.weights_init_fill(
                linear_weight = list(grouped.apply(np.quantile, 0.9) - grouped.apply(np.quantile, 0.1)),
                linear_bias = list(grouped.apply(np.quantile, 0.1)))
        elif 'metadata' in model.__dict__.keys():
            #Save input model metadata if not reinitialising model weights
            model.metadata_history = copy.deepcopy(model.metadata)

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
            scheduler_epochs = scheduler_epochs)
        print("Fitting model:")
        print(model.metadata)

        #Load model data
        train_dataloader, valid_dataloader, test_dataloader = model.load_data(model_data, batch_size)

        #Construct loss function and Optimizer
        loss_fn = torch.nn.L1Loss(reduction = "none")
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
                loss_fn, 
                optimizer,
                self.device, 
                l1_regularization_factor, 
                l2_regularization_factor)
            model.validate_model(
                valid_dataloader, 
                loss_fn, 
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
        #Output predictions directory
        directory = os.path.join(self.directory, 'predictions')

        #Create output predictions directory
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass

        #Set folds if not supplied
        if folds==None:
            folds = [i+1 for i in range(self.data.k_folds)]

        #Model subset
        models_subset = [i for i in self.models if ((i.metadata.grid_search==grid_search) & (i.metadata.fold in folds))]
        #Check if at least one model remaining
        if len(models_subset)==0:
            print("No models satisfying criteria.")
            return

        #Model data
        if data==None:
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
            mask_list = [torch.narrow(mask, 0, j, 1) for j in range(mask.shape[0])]
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
                            additive_traits_p += [model.additivetraits[model.model_design.loc[pi,'trait'][at_pi]-1](torch.mul(X, mask_list[pi]))]
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

