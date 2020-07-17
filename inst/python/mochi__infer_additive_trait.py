#!/usr/bin/env python

#######################################################################
## COMMANDLINE ARGUMENTS ##
#######################################################################

import argparse

#Create parser
parser = argparse.ArgumentParser()
 
#Add arguments to the parser
parser.add_argument("--input_file", "-i")
parser.add_argument("--output_directory", "-o")
parser.add_argument("--number_additive_traits", "-n", default = 1, help = "Number of additive traits")
parser.add_argument("--test_set_proportion", "-t", default = 0.05, help = "Test set proportion")
parser.add_argument("--l2_regularization_factor", "-l", default = 0.05, help = "L2 regularization factor for additive trait layer")
parser.add_argument("--num_nodes_swish1", default = 20, help = "Number of nodes in first swish layer")
parser.add_argument("--num_nodes_swish2", default = 10, help = "Number of nodes in second swish layer")
parser.add_argument("--num_nodes_swish3", default = 5, help = "Number of nodes in third swish layer")
parser.add_argument("--num_epochs", "-e", default = 2500, help = "Number of epochs to train the model")
parser.add_argument("--num_samples", "-s", default = 128, help = "Number of samples per gradient update")
 
#Parse the arguments
args = parser.parse_args()

input_file = args.input_file #"/users/blehner/afaure/DMS/Code/MoCHI/Global/data/S7.txt"
output_directory = args.output_directory #"/users/blehner/afaure/DMS/Code/MoCHI/Global/"
number_additive_traits = args.number_additive_traits
test_set_proportion = args.test_set_proportion
l2_regularization_factor = args.l2_regularization_factor
num_nodes_swish1 = args.num_nodes_swish1
num_nodes_swish2 = args.num_nodes_swish2
num_nodes_swish3 = args.num_nodes_swish3
num_epochs = args.num_epochs
num_samples = args.num_samples

#######################################################################
## PACKAGES ##
#######################################################################

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import random
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.offsetbox as offsetbox # for text box with position control
import pickle
import seaborn as sns
import copy
import sys
import matplotlib
import os

#######################################################################
## FUNCTIONS ##
#######################################################################

#Get sequence ID from sequence string
def get_seq_id(sq):
  return ":".join([str(i)+sq[i] for i in range(len(sq))])

#Binarise genotype according to presence/absence of mutations
def make_binary(unique_mutations, genotype):
  #Convert genotype to list
  genotypeList = genotype.split(':')
  #Get indices of mutations
  indexList = []
  for i in range(len(genotypeList)):
    indexList.append(unique_mutations.index(genotypeList[i]))
  #Initialise binary array with 0s
  line = np.zeros((1,len(unique_mutations)))
  #Set mutations in genotype to 1
  line[:,indexList] = 1.
  #Return
  return line

#Read DMS data and reformat genotypes for neural network
def read_data_all_positions(input_file):    
  #Read in the data table
  data = pd.read_table(input_file,
    dtype = {"var_seq" : object})
  #Rename columns
  data.columns = ['fitness', 'var_seq']
  #List of all sequence ids
  mutation_list = pd.Series([get_seq_id(sq) for sq in data.var_seq])
  #List of unique single mutations
  unique_mutations = set(':'.join(mutation_list).split(':'))
  unique_mutations = sorted(list(unique_mutations))
  #Remove empty blank mutations
  if '' in unique_mutations:
    unique_mutations.remove('')
  #Array of fitness values
  nn_fitness_values = data.fitness.values
  #Series of variant sequences
  var_seq = data.var_seq
  #Binary array of genotype values
  nn_genotypes_values = np.zeros((len(data), len(unique_mutations)))
  for i in range(len(mutation_list)):
    if mutation_list[i] != '':
      nn_genotypes_values[i] = make_binary(unique_mutations, mutation_list[i])[0]
  #Return
  return nn_genotypes_values, nn_fitness_values, unique_mutations, var_seq, mutation_list

#Swish activation function
def swish(x, beta = 1):
  return (x * keras.backend.sigmoid(beta * x))

#Little function that returns layer index corresponding to layer name
def get_layer_index(model, layername):
  for idx, layer in enumerate(model.layers):
    if layer.name == layername:
      return idx

#######################################################################
## SETUP ##
#######################################################################


#Output model directory
model_directory = os.path.join(output_directory, "whole_model")
#Create output model directory
try:
  os.mkdir(model_directory)
except FileExistsError:
  print("Warning: Output model directory already exists.")

#Output plot directory
plot_directory = os.path.join(output_directory, "plots")
#Create output plot directory
try:
  os.mkdir(plot_directory)
except FileExistsError:
  print("Warning: Output plot directory already exists.")

#Load data
feature_matrix, observed_fitness, unique_mutations, var_sequences, sequence_ids = read_data_all_positions(input_file)

#Re-scaling the fitness values (for some reason the neural network can only predict things between 0 and 1??)
print("rescaling the fitness values between 0 and 1...")
min_max_scaler = MinMaxScaler()
fitness_values_scaled = min_max_scaler.fit_transform(observed_fitness.reshape(-1,1))

#Split data into training and validation sets
print("splitting data into training and testing sets (95-05 split)...")
random.seed(1)
tf.random.set_seed(1)
x_train, x_valid, y_train, y_valid = train_test_split(
  feature_matrix,
  fitness_values_scaled,
  test_size = test_set_proportion)

#Create custom activation layer
keras.utils.get_custom_objects().update({'swish': keras.layers.Activation(swish)})

#######################################################################
## BUILD NEURAL NETWORK WITH FUNCTIONAL API & SPLITTING INPUT IN TWO ##
#######################################################################

#Set random seeds
random.seed(1)
tf.random.set_seed(1)

#Input layer
input_layer = keras.layers.Input(
  shape = x_train.shape[1:],
  name = "inputlayer")

#Additive trait layers
additive_trait_layer = keras.layers.Dense(
  number_additive_traits,
  input_dim = feature_matrix.shape[1],
  kernel_initializer = 'glorot_normal',
  activation = "linear",
  name = "additivetrait",
  kernel_regularizer = keras.regularizers.l2(l2_regularization_factor))(input_layer)

#Sigmoidal layers
swish_layer_1 = keras.layers.Dense(
  num_nodes_swish1,
  activation = "swish",
  name = "swishlayer1")(additive_trait_layer)
swish_layer_2 = keras.layers.Dense(
  num_nodes_swish2,
  activation = "swish",
  name = "swishlayer2")(swish_layer_1)
swish_layer_3 = keras.layers.Dense(
  num_nodes_swish3,
  activation = "swish",
  name = "swishlayer3")(swish_layer_2)

#Output layer
output_layer = keras.layers.Dense(
  1,
  activation = "linear",
  name = "outputlayer",
  kernel_initializer = 'glorot_normal')(swish_layer_3)

#Create keras model defining input and output layers
model = keras.Model(
  inputs = [input_layer], 
  outputs = [output_layer])

#Optimiser algorithm (default args)
opt = keras.optimizers.Adam()

#Compile the model
model.compile(
  optimizer = opt,
  loss = 'mean_absolute_error')
#model.summary()

#Fit the model
history = model.fit(
  x_train,
  y_train,
  validation_data = (
    x_valid,
    y_valid),
  epochs = num_epochs,
  batch_size = num_samples,
  shuffle = True,
  #callbacks = [early_stopping_monitor],
  verbose = 0)

#Plot model performance per epoch
my_figure = plt.figure(figsize = (8,8))
plt.plot(
  np.log(history.history['loss']))
plt.xlabel('Number of epochs')
plt.ylabel('Mean Absolute Error (MAE) on testing data')
my_figure.savefig(os.path.join(plot_directory, "model_performance_perepoch.pdf"), bbox_inches='tight')

#Save the entire model as a SavedModel
model.save(os.path.join(model_directory, 'my_model'))

#######################################################################
## PLOT PREDICTIONS VERSUS OBSERVATIONS ##
#######################################################################

#Model predictions on observed variants
model_predictions = model.predict(feature_matrix)

#Set font type
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#Plot model predictions versus scaled fitness values
fig = plt.figure(figsize = (8,8))
## First additive trait ##
ax = fig.add_subplot(111)
ax.plot(
  min_max_scaler.inverse_transform(model_predictions),
  observed_fitness,
  "o",
  ms = 0.51)
plt.xlabel('Predicted fitness')
plt.ylabel('Observed fitness')
fig.savefig(os.path.join(plot_directory, "model_predictions_versus_scaledfitness.pdf"), bbox_inches='tight')

#######################################################################
## PLOT ADDITIVE TRAIT ##
#######################################################################

# idx for additive trait 1 layer
layer_idx = get_layer_index(
  model = model,
  layername = "additivetrait")

#Calculate reconstructed additive trait 1
additive_traits_model = keras.Model(
  inputs = model.input,
  outputs = model.layers[layer_idx].output)
#Convert to data frame
reconstructed_additive_trait_df = pd.DataFrame(additive_traits_model.predict(feature_matrix))
reconstructed_additive_trait_df.columns = [ "trait " + str(i) for i in range(len(reconstructed_additive_trait_df.columns))]

#Plot additive trait versus observed
fig = plt.figure(figsize=(8,8))
## First additive trait ##
ax = fig.add_subplot(111)
ax.plot(
  reconstructed_additive_trait_df["trait 0"],
  observed_fitness,
  "o",
  ms = 0.51)
plt.xlabel('Additive trait')
plt.ylabel('Observed fitness')
fig.savefig(os.path.join(plot_directory, "model_first_additive_trait_versus_observed.pdf"), bbox_inches='tight')

#Plot additive trait versus predicted
fig = plt.figure(figsize = (8,8))
## First additive trait ##
ax = fig.add_subplot(111)
ax.plot(
  reconstructed_additive_trait_df["trait 0"],
  min_max_scaler.inverse_transform(model_predictions),
  "o",
  ms = 0.51)
plt.xlabel('Additive trait')
plt.ylabel('Predicted fitness')
fig.savefig(os.path.join(plot_directory, "model_first_additive_trait_versus_predicted.pdf"), bbox_inches='tight')

#######################################################################
## SAVE OBSERVATIONS, PREDICTIONS & ADDITIVE TRAIT VALUES ##
#######################################################################

#Results data frame
dataframe_to_export = pd.DataFrame({
  "seq" : var_sequences,
  "observed_fitness" : observed_fitness,
  "predicted_fitness" : min_max_scaler.inverse_transform(model_predictions).flatten(),
  "additive_trait" : reconstructed_additive_trait_df["trait 0"]})
#Save as csv file
dataframe_to_export.to_csv(
  os.path.join(output_directory, "predicted_fitness.txt"),
  sep = "\t",
  index = False)

