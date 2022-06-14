
"""
MoCHI report module
"""

import os
import torch
from torch import Tensor
from pymochi.data import *
import shutil
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

class MochiReport:
    """
    A class for generating reports about MochiProject objects.
    """
    def __init__(
        self, 
        project,
        RT = None):
        """
        Initialize a MochiReport object.

        :param project: a MochiProject object (required).
        :param RT: R=gas constant (in kcal/K/mol) * T=Temperature (in K) (optional).
        :returns: MochiReport object.
        """     
        #Initialize attributes
        self.project = project
        self.RT = RT
        #Generate report
        self.report()

    def plot_loss_vs_epoch(
        self,
        output_path,
        folds = None,
        grid_search = False):
        """
        Plot validation loss for all training epochs.

        :param output_path: Output file path (required).
        :param folds: list of cross-validation folds (default:None i.e. all).
        :param grid_search: Whether or not to include grid_search models (default:False).
        :returns: Nothing.
        """ 
        #Set folds if not supplied
        if folds==None:
            folds = [i+1 for i in range(self.project.data.k_folds)]

        #Model subset
        models_subset = [i for i in self.project.models if ((i.metadata.grid_search==grid_search) & (i.metadata.fold in folds))]
        #Check if at least one model remaining
        if len(models_subset)==0:
            print("No models satisfying criteria.")
            return
        #Plot
        fig, ax = plt.subplots()  # Create a figure containing a single axis.
        for i in range(len(models_subset)):
            ax.plot(
                range(len(models_subset[i].training_history['val_loss'])), 
                models_subset[i].training_history['val_loss'], 
                label="cvfold_"+str(models_subset[i].metadata.fold)); 
        #Decorations
        ax.set_xlabel('Epoch', size = 14)
        ax.set_ylabel('Loss', size = 14)
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(output_path)

    def plot_WT_coefficient_vs_epoch(
        self,
        output_path,
        folds = None,
        grid_search = False,
        RT = None):
        """
        Plot WT coefficient for all training epochs.

        :param output_path: Output file path (required).
        :param folds: list of cross-validation folds (default:None i.e. all).
        :param grid_search: Whether or not to include grid_search models (default:False).
        :param RT: R=gas constant (in kcal/K/mol) * T=Temperature (in K) (optional).
        :returns: Nothing.
        """ 
        #Set folds if not supplied
        if folds==None:
            folds = [i+1 for i in range(self.project.data.k_folds)]

        #Set RT if not supplied
        if RT==None:
            RT = 1

        #Model subset
        models_subset = [i for i in self.project.models if ((i.metadata.grid_search==grid_search) & (i.metadata.fold in folds))]
        #Check if at least one model remaining
        if len(models_subset)==0:
            print("No models satisfying criteria.")
            return

        fig, ax = plt.subplots()  # Create a figure containing a single axiss.
        for i in range(len(models_subset[0].additivetraits)):
            for j in range(len(models_subset)):
                ax.plot(
                    range(len(models_subset[j].training_history['additivetrait'+str(i+1)+'_WT'])), 
                    np.array(models_subset[j].training_history['additivetrait'+str(i+1)+'_WT'])*RT, 
                    label=self.project.data.additive_trait_names[i]+" cvfold_"+str(j+1)); 
        #Decorations
        plt.axhline(y = 0, color = 'black', linestyle = '--')
        ax.set_xlabel('Epoch', size = 14)
        ax.set_ylabel('WT coefficient', size = 14)
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(output_path)

    def plot_WT_residual_vs_epoch(
        self,
        output_path,
        folds = None,
        grid_search = False):
        """
        Plot WT residual phenotype for all training epochs.

        :param output_path: Output file path (required).
        :param folds: list of cross-validation folds (default:None i.e. all).
        :param grid_search: Whether or not to include grid_search models (default:False).
        :returns: Nothing.
        """ 
        #Set folds if not supplied
        if folds==None:
            folds = [i+1 for i in range(self.project.data.k_folds)]

        #Model subset
        models_subset = [i for i in self.project.models if ((i.metadata.grid_search==grid_search) & (i.metadata.fold in folds))]
        #Check if at least one model remaining
        if len(models_subset)==0:
            print("No models satisfying criteria.")
            return

        fig, ax = plt.subplots()  # Create a figure containing a single axes.
        for i in range(self.project.data.model_design.shape[0]):
            for j in range(len(models_subset)):
                ax.plot(
                range(len(models_subset[j].training_history['residual'+str(i+1)+'_WT'])), 
                models_subset[j].training_history['residual'+str(i+1)+'_WT'], 
                label=self.project.data.phenotype_names[i]+" cvfold_"+str(j+1));  # Plot some data on the axes.
        plt.axhline(y = 0, color = 'black', linestyle = '--')
        #Decorations
        ax.set_xlabel('Epoch', size = 14)
        ax.set_ylabel('WT residual', size = 14)
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(output_path)

    def plot_test_performance(
        self,
        input_df,
        output_path_prefix):
        """
        Plot model performance on test data.

        :param input_df: Input DataFrame with all predictions (required).
        :param output_path_prefix: Output file path (required).
        :returns: Nothing.
        """ 

        #Subset to held out variants and relevant columns
        all_folds = list(set(input_df.loc[input_df.Fold.isna()==False,'Fold']))
        all_folds = [i for i in all_folds if 'fold_'+str(i) in input_df.columns]
        rel_cols = ['fitness', 'phenotype', 'Fold']+['fold_'+str(i) for i in all_folds]
        result_df = input_df.loc[input_df.Fold.isna()==False,rel_cols]

        #Observed phenotype
        result_df['Observed phenotype'] = result_df['fitness']
        #Predicted phenotype
        for i in all_folds:
            result_df.loc[result_df.Fold==i,'Predicted phenotype'] = result_df.loc[result_df.Fold==i,'fold_'+str(i)]

        #Plot performance for all phenotypes
        for i in list(set(result_df.phenotype)):
            #Plot
            plot_df = result_df.loc[result_df.phenotype==i,:]
            fig, ax = plt.subplots()  # Create a figure containing a single axes.
            plot_df.reset_index(drop = True, inplace = True)
            n_bin = 50
            cmap = LinearSegmentedColormap.from_list('whiteblack', ['white', 'black'], N=n_bin)
            z = plt.hexbin(
                x = 'Observed phenotype',
                y = 'Predicted phenotype',
                data = plot_df, gridsize = 100, cmap = cmap, bins = 'log')
            #Color scale
            fig.colorbar(z, ax=ax)
            #Guides
            ax.axline((0.1, 0.1), slope=1, linestyle = "dashed", color = 'black')
            plt.axhline(y = 0, color = 'black', linestyle = '--')
            plt.axvline(x = 0, color = 'black', linestyle = '--')
            #Labels
            ax.set_title(self.project.data.phenotype_names[int(i)-1])
            ax.set_xlabel('Observed phenotype', size = 14)
            ax.set_ylabel('Predicted phenotype', size = 14)
            ax.set_aspect("equal")
            #R-squared
            from matplotlib.offsetbox import AnchoredText
            cor_coef = np.corrcoef(
                result_df.loc[result_df.phenotype==i,'Observed phenotype'], 
                result_df.loc[result_df.phenotype==i,'Predicted phenotype'])[0,1]
            at = AnchoredText(
                r'$R^2 = $'+str(round(np.power(cor_coef, 2), 2)), 
                prop=dict(size=14), frameon=False, loc='upper left')
            ax.add_artist(at)
            #Save
            plt.savefig(output_path_prefix+self.project.data.phenotype_names[int(i)-1]+".pdf")

    def plot_observed_phenotype_vs_additivetrait(
        self,
        input_df,
        output_path_prefix,
        folds = None,
        RT = None):
        """
        Plot observed phenoypte versus additive trait (1-dimensional additive traits only).

        :param input_df: Input DataFrame with all predictions (required).
        :param output_path_prefix: Output file path (required).
        :param folds: list of cross-validation folds (default:None i.e. all).
        :param RT: R=gas constant (in kcal/K/mol) * T=Temperature (in K) (optional).
        :returns: Nothing.
        """ 
        #Set folds if not supplied
        if folds==None:
            folds = [i+1 for i in range(self.project.data.k_folds)]

        #Set RT if not supplied
        if RT==None:
            RT = 1

        fold = folds[0]
        observed_phenotype_col = 'fitness'

        #Plot performance for all 1-dimensional phenotypes
        for i in range(len(self.project.data.model_design)):
            #1-dimensional additive traits
            if len(self.project.data.model_design.loc[i,'trait'])==1:
                predicted_phenotype_col = 'fold_'+str(fold)
                additive_trait_col = predicted_phenotype_col+'_additive_trait0'
                plot_df_sort = input_df.loc[input_df.phenotype==str(i+1),:].sort_values(additive_trait_col)
                plot_df_sort.reset_index(drop = True, inplace = True)
                #Convert units
                plot_df_sort[additive_trait_col+"_kcal/mol"] = plot_df_sort[additive_trait_col]*RT
                additive_trait_col = additive_trait_col+"_kcal/mol"
                fig, ax = plt.subplots()
                n_bin = 50
                cmap = LinearSegmentedColormap.from_list('whiteblack', ['white', 'black'], N=n_bin)
                z = plt.hexbin(
                    x = additive_trait_col,
                    y = observed_phenotype_col,
                    data = plot_df_sort.loc[plot_df_sort.phenotype==str(i+1),:], gridsize = 100, cmap = cmap, bins = 'log')
                ax.plot(
                    plot_df_sort.loc[plot_df_sort.phenotype==str(i+1),additive_trait_col], 
                    plot_df_sort.loc[plot_df_sort.phenotype==str(i+1),predicted_phenotype_col], mfc='none', linestyle = "-", color = 'red')
                #Color scale
                fig.colorbar(z, ax=ax)
                #Guides
                plt.axhline(y = 0, color = 'black', linestyle = '--')
                plt.axvline(x = 0, color = 'black', linestyle = '--')
                #Labels
                ax.set_title(self.project.data.phenotype_names[int(i)])
                ax.set_xlabel('Additive trait', size = 14)
                ax.set_ylabel('Observed phenotype', size = 14)
                #Save
                plt.savefig(output_path_prefix+self.project.data.phenotype_names[int(i)]+".pdf")

    def report(
        self):
        """
        Produce project report.

        :returns: Nothing.
        """ 

        #Check if valid MochiProject
        if 'models' not in dir(self.project):
            print("Error: Cannot produce report. Not a valid MochiProject.")
            return

        #Output report directory
        directory = os.path.join(self.project.directory, 'report')

        #Create output report directory
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass

        #Delete entire directory contents and create fresh directory
        shutil.rmtree(directory)
        os.mkdir(directory)

        #Produce report plots
        self.plot_loss_vs_epoch(output_path = os.path.join(directory, "loss_epoch.pdf"))
        self.plot_WT_coefficient_vs_epoch(
            output_path = os.path.join(directory, "WT_coefficient_epoch.pdf"),
            folds = [1],
            RT = self.RT)
        self.plot_WT_residual_vs_epoch(
            output_path = os.path.join(directory, "WT_residual_epoch.pdf"),
            folds = [1])

        #Predictions on all data for all models
        prediction_df = self.project.predict_all()

        self.plot_test_performance(
            input_df = prediction_df,
            output_path_prefix = os.path.join(directory, "test_performance_"))

        self.plot_observed_phenotype_vs_additivetrait(
            input_df = prediction_df,
            output_path_prefix = os.path.join(directory, "observed_phenotype_vs_additivetrait_"),
            folds = [1],
            RT = self.RT)



