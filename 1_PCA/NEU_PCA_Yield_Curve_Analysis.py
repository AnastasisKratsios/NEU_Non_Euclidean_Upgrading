#!/usr/bin/env python
# coding: utf-8

# In[1]:


n_jobs = -1
n_iter = 1
N_chains = 0


# # NEU-PCA: Financial Data
# - Designed and Coded by: [Anastasis Kratsios](https://people.math.ethz.ch/~kratsioa/).
# - Some Elements of the PCA analysis are forked from [this repo](https://github.com/radmerti/MVA2-PCA/blob/master/YieldCurvePCA.ipynb).

# # What is PCA?
# PCA is a two-part algorithm.  In phase 1, high-dimensional data $\mathbb{R}^D$ is mapped into a low-dimensional space ($D\gg d$) via the optimal linear (orthogonal) projection.  In phase 2, the best $d$-dimensional embedding of the features $\mathbb{R}^d$ into $\mathbb{R}^D$ is learned and used to reconstruct (as best as is possible) the high-dimensional data from this small set of features.  

# # How does NEU-PCA function?
# Since the purpous of the reconfiguration network is to learn (non-linear) topology embeddings of low-dimensional linear space then we can apply NEU to the reconstruction map phase of PCA.  Moreover, we will see that the embedding can be infered from a low-dimensional intermediate space $\mathbb{R}^N$ with $d\leq N\ll D$.  Benefits:
# - Computationally cheap,
# - Just as effective as an Autoencoder,
# - Maintain interpretation of PCA features!

# ## Parameters

# In[2]:


## Dimension to be Reduced To
PCA_Rank = 3
## TEMPS!!
is_visuallty_verbose = True


# ---
# ---
# ---

# # 0) Initialization Phase

# ---
# ---
# ---

# ## Imports

# In[3]:


# First Round Initializations (Global Level) #
#============================================#
# Load Dependances and makes path(s)
exec(open('Initializations_Dump.py').read())
# Load Hyper( and meta) parameter(s)
exec(open('HyperParameter_Grid.py').read())
# %run Helper_Functions.ipynb
exec(open('Helper_Functions.py').read())
# Load Models
# %run Architecture_Builder.ipynb
exec(open('Architecture_Builder.py').read())
# Initialize "First Run Mode"
First_run = True


# In[4]:


import pylab as plt
import numpy as np
import seaborn as sns; sns.set()

import sklearn
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import decomposition
import scipy

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Layer, InputSpec
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import regularizers, activations, initializers, constraints, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import UnitNorm, Constraint

from numpy.random import seed

# MUSTS
import matplotlib.gridspec as gridspec


# ### Fix Seeds for Reproducability

# In[5]:


# Numpy
np.random.seed(2020)
# Tensorflow
tf.random.set_seed(2020)
# Python's Seed
random.seed(2020)


# ---
# ---
# ---

# # 1) Data-Preparation Phase

# ---
# ---
# ---

# ## Prepare Data

# #### Load Data

# In[6]:


if First_run:
    # Load Data
    yield_data = pd.read_excel('inputs/data/ust_daily.ods', engine='odf')


# #### Hardcore Maturities Vector

# In[7]:


Maturities = np.array([(1/12),.25,.5,1,2,3,5,7,10,20,30])


# #### Format Data

# In[8]:


if First_run:
    yield_data['date'] = pd.to_datetime(yield_data['date'],infer_datetime_format=True)
    yield_data.set_index('date', drop=True, inplace=True)
    yield_data.index.names = [None]
    # Remove garbage column
    yield_data.drop(columns=['BC_30YEARDISPLAY'])


# #### Subset Data

# In[9]:


if First_run:
    # # Get indices
    N_train_step = int(round(yield_data.shape[0]*Train_step_proportion,0))
    N_test_set = int(yield_data.shape[0] - round(yield_data.shape[0]*Train_step_proportion,0))
    # # Get Datasets
    X_train = yield_data[:N_train_step]
    X_test = yield_data[-N_test_set:]
    # Transpose
    X_train_T = X_train.T
    X_test_T = X_test.T
    
    
    # # Update User
    print('#================================================#')
    print(' Training Datasize: '+str(X_train.shape[0])+' and test datasize: ' + str(X_test.shape[0]) + '.  ')
    print('#================================================#')
    
    # # Set First Run to Off
    First_run = False


# #### Pre-Process Data

# In[10]:


# Initialize Scaler
scaler = MinMaxScaler()
scaler.fit(X_train)

# Train Scaler
X_train_scaled = scaler.transform(X_train)
# Map to Test Set
X_test_scaled = scaler.transform(X_test)


# ### Visualize Data

# #### Train

# In[11]:


if is_visuallty_verbose:
    print('Training Dataset Preview:')
    print(X_train.head())

X_train.head()


# #### Test

# In[12]:


if is_visuallty_verbose:
    print('Testing Dataset Preview:')
    print(X_test.head())
    
X_test.head()


# ### Time-Series

# In[13]:


plt.figure(figsize=(20,15))

plt.plot(X_train.index, X_train)
plt.xlim(X_train.index.min(), X_train.index.max())
plt.ylim(np.min(X_train.min()),np.max(X_train.max()))

plt.axhline(y=0,c="grey",linewidth=0.5,zorder=0)
for i in range(X_train.index.min().year, X_train.index.max().year+1):
    plt.axvline(x=X_train.index[X_train.index.searchsorted(DT.datetime(i,1,1))-1],
                c="grey", linewidth=0.5, zorder=0)
    
plt.legend((np.array(X_train.columns)))

# Save 
plt.savefig('outputs/plotsANDfigures/Data_Visualization_Yield_TimeSeries.pdf')


# ### Yield Curves

# In[14]:


Ncols = 6
Nrows = 10
num_years = X_train.index.max().year-X_train.index.min().year
rows = math.ceil(num_years/Ncols)

plt.figure(figsize=(24,(24/Ncols)*rows))

plt.subplot2grid((rows,Ncols), (0,0), colspan=Ncols, rowspan=Nrows)


colnum = 0
rownum = 0
for year in range(X_train.index.min().year,X_train.index.max().year):
    year_start = X_train.index[X_train.index.searchsorted(DT.datetime(year,1,1))]
    year_end = X_train.index[X_train.index.searchsorted(DT.datetime(year,12,30))]
    
    plt.subplot2grid((rows,Ncols), (rownum,colnum), colspan=1, rowspan=1)
    plt.title('{0}'.format(year))
    plt.xlim(0, len(X_train_T.index)-1)
    plt.ylim(np.min(X_train_T.values), np.max(X_train_T.values))
    plt.xticks(range(len(X_train_T.index)), X_train_T.index, size='small')
    
    plt.plot(X_train_T.loc[:,year_start:year_end].values)
    
    if colnum != Ncols-1:
        colnum += 1
    else:
        colnum = 0
        rownum += 1

# Save
plt.savefig('outputs/plotsANDfigures/Data_Visualization_Annual_Yield_Curves.pdf')


# ---

# ---
# ---
# ---

# # 2) Prediction Phase

# ---
# ---
# ---

# # Benchmark(s)
# ---

# ## Get PCAs

# In[15]:


# Reconstruct Training Data
Zpca,Zpca_test,Rpca,Rpca_test = get_PCAs(X_train_scaled=X_train_scaled.T,
                                         X_test_scaled=X_train_scaled.T,
                                         PCA_Rank=PCA_Rank)


# #### Get Reconstruction Result(s)

# In[16]:


# Get Results #
#-------------#
# Errors (Train): 
A = pd.DataFrame(Rpca)
B = pd.DataFrame(X_train.T)
train_results = B.to_numpy()-A.to_numpy()
### MSE
train_results_MSE = train_results**2
train_results_MSE_vect = np.mean(train_results_MSE,axis=1)
PCA_Reconstruction_train_results_MSE = np.mean(train_results_MSE_vect)
### MAE
train_results_MAE = np.abs(train_results)
train_results_MAE_vect = np.mean(train_results_MAE,axis=1)
PCA_Reconstruction_train_results_MAE = np.mean(train_results_MAE_vect)


# Errors (Test): One step ahead prediction errors
A = pd.DataFrame(Rpca).iloc[1:]
B = pd.DataFrame(X_train.T).iloc[:-1]
test_results = B.to_numpy()-A.to_numpy()
### MSE
test_results_MSE = test_results**2
test_results_MSE_vect = np.mean(test_results_MSE,axis=1)
PCA_Reconstruction_test_results_MSE = np.mean(test_results_MSE_vect)
### MAE
test_results_MAE = np.abs(test_results)
test_results_MAE_vect = np.mean(test_results_MAE,axis=1)
PCA_Reconstruction_test_results_MAE = np.mean(test_results_MAE_vect)


# Formatting
## Train
Performance_Results_train = pd.DataFrame([{'MAE':PCA_Reconstruction_train_results_MAE,
                                           'MSE':PCA_Reconstruction_train_results_MSE}],
                                        index=['PCA'])
## Test
Performance_Results_test = pd.DataFrame([{'MAE':PCA_Reconstruction_test_results_MAE,
                                          'MSE':PCA_Reconstruction_test_results_MSE}],
                                        index=['PCA'])

# Save Results #
#--------------#
Performance_Results_train.to_latex('outputs/tables/Fin_Performance_train.txt')
Performance_Results_test.to_latex('outputs/tables/Fin_Performance_test.txt')


# ## Get (ReLU) Auto-Encoder

# In[17]:


AE_Reconstructed_train, AE_Reconstructed_test, AE_Factors_train, AE_Factors_test = build_autoencoder(CV_folds,
                                                                                    n_jobs,
                                                                                    n_iter,
                                                                                    X_train_scaled.T,
                                                                                    X_train.T,
                                                                                    X_train_scaled.T,
                                                                                    PCA_Rank)


# #### Get Reconstruction Result(s)

# In[18]:


# Get Results #
#-------------#
# Errors (Train): 
A = pd.DataFrame(AE_Reconstructed_train)
B = pd.DataFrame(X_train.T)
train_results = B.to_numpy()-A.to_numpy()
### MSE
train_results_MSE = train_results**2
train_results_MSE_vect = np.mean(train_results_MSE,axis=1)
AE_Reconstruction_train_results_MSE = np.mean(train_results_MSE_vect)
### MAE
train_results_MAE = np.abs(train_results)
train_results_MAE_vect = np.mean(train_results_MAE,axis=1)
AE_Reconstruction_train_results_MAE = np.mean(train_results_MAE_vect)


# Errors (Test): One step ahead prediction errors
A = pd.DataFrame(AE_Reconstructed_train).iloc[1:]
B = pd.DataFrame(X_train.T).iloc[:-1]
test_results = B.to_numpy()-A.to_numpy()
### MSE
test_results_MSE = test_results**2
test_results_MSE_vect = np.mean(test_results_MSE,axis=1)
AE_Reconstruction_test_results_MSE = np.mean(test_results_MSE_vect)
### MAE
test_results_MAE = np.abs(test_results)
test_results_MAE_vect = np.mean(test_results_MAE,axis=1)
AE_Reconstruction_test_results_MAE = np.mean(test_results_MAE_vect)


# Formatting
## Train
AE_Reconstruction_Results_train = pd.DataFrame([{'MAE':PCA_Reconstruction_train_results_MAE,
                                                 'MSE':PCA_Reconstruction_train_results_MSE}],index=['AE'])
## Test
AE_Reconstruction_Results_test = pd.DataFrame([{'MAE':PCA_Reconstruction_train_results_MAE,
                                                'MSE':PCA_Reconstruction_train_results_MSE}],index=['AE'])


# Update
Performance_Results_train = pd.concat([Performance_Results_train,AE_Reconstruction_Results_train],axis=0)
Performance_Results_test = pd.concat([Performance_Results_test,AE_Reconstruction_Results_test],axis=0)


# Save Results #
#--------------#
Performance_Results_train.to_latex('outputs/tables/Fin_Performance_train.txt')
Performance_Results_test.to_latex('outputs/tables/Fin_Performance_test.txt')


# # NEU - Depricated

# In[19]:


# print('NEU-PCA: Computing...')
# NEU_PCA_Reconstruction_train, NEU_PCA_Reconstruction_test, NEU_PCA_Factors_train, NEU_PCA_Factors_test =  build_NEU_PCA(CV_folds, 
#                                                                                                                         n_jobs, 
#                                                                                                                         n_iter, 
#                                                                                                                         param_grid_in, 
#                                                                                                                         X_train_scaled.T,
#                                                                                                                         X_train.T, 
#                                                                                                                         X_train_scaled.T,
#                                                                                                                         PCA_Rank)

# print('NEU-PCA: Complete!')


# #### Get Reconstruction Result(s) - Depricated

# In[20]:


# # Get Results #
# #-------------#
# # Errors (Train): 
# A = pd.DataFrame(NEU_PCA_Reconstruction_train)
# B = pd.DataFrame(X_train.T)
# train_results = B.to_numpy()-A.to_numpy()
# ### MSE
# train_results_MSE = train_results**2
# train_results_MSE_vect = np.mean(train_results_MSE,axis=1)
# NEU_PCA_Reconstruction_train_results_MSE = np.mean(train_results_MSE_vect)
# ### MAE
# train_results_MAE = np.abs(train_results)
# train_results_MAE_vect = np.mean(train_results_MAE,axis=1)
# NEU_PCA_Reconstruction_train_results_MAE = np.mean(train_results_MAE_vect)


# # Errors (Test): One step ahead prediction errors
# A = pd.DataFrame(NEU_PCA_Reconstruction_train).iloc[1:]
# B = pd.DataFrame(X_train.T).iloc[:-1]
# test_results = B.to_numpy()-A.to_numpy()
# ### MSE
# test_results_MSE = test_results**2
# test_results_MSE_vect = np.mean(test_results_MSE,axis=1)
# NEU_PCA_Reconstruction_test_results_MSE = np.mean(test_results_MSE_vect)
# ### MAE
# test_results_MAE = np.abs(test_results)
# test_results_MAE_vect = np.mean(test_results_MAE,axis=1)
# NEU_PCA_Reconstruction_test_results_MAE = np.mean(test_results_MAE_vect)


# # Formatting
# ## Train
# NEU_Reconstruction_Results_train = pd.DataFrame([{'MAE':NEU_PCA_Reconstruction_train_results_MAE,
#                                                  'MSE':NEU_PCA_Reconstruction_train_results_MSE}],index=['NEU-PCA'])
# ## Test
# NEU_Reconstruction_Results_test = pd.DataFrame([{'MAE':NEU_PCA_Reconstruction_test_results_MAE,
#                                                 'MSE':NEU_PCA_Reconstruction_test_results_MSE}],index=['NEU-PCA'])


# # Update
# Performance_Results_train = pd.concat([Performance_Results_train,NEU_Reconstruction_Results_train],axis=0)
# Performance_Results_test = pd.concat([Performance_Results_test,NEU_Reconstruction_Results_test],axis=0)


# # Save Results #
# #--------------#
# Performance_Results_train.to_latex('outputs/tables/Fin_Performance_train.txt')
# Performance_Results_test.to_latex('outputs/tables/Fin_Performance_test.txt')


# ### NEU-Autoencoder - Depricated

# In[21]:


# print('NEU-Autoencoder: Computing...')
# NEU_AE_Reconstruction_train, NEU_AE_Reconstruction_test, NEU_AE_Factors_train, NEU_AE_Factors_test = build_NEU_Autoencoder(CV_folds, 
#                                                                                                                                n_jobs, 
#                                                                                                                                n_iter, 
#                                                                                                                                param_grid_in, 
#                                                                                                                                X_train_scaled.T,
#                                                                                                                                X_train.T, 
#                                                                                                                                X_train_scaled.T,
#                                                                                                                                PCA_Rank)

# print('NEU-Autoencoder: Complete!')


# #### Get Reconstruction Result(s) - Depricated

# In[22]:


# # Get Results #
# #-------------#
# # Errors (Train): 
# A = pd.DataFrame(NEU_AE_Reconstruction_train)
# B = pd.DataFrame(X_train.T)
# train_results = B.to_numpy()-A.to_numpy()
# ### MSE
# train_results_MSE = train_results**2
# train_results_MSE_vect = np.mean(train_results_MSE,axis=1)
# NEU_AE_Reconstruction_train_results_MSE = np.mean(train_results_MSE_vect)
# ### MAE
# train_results_MAE = np.abs(train_results)
# train_results_MAE_vect = np.mean(train_results_MAE,axis=1)
# NEU_AE_Reconstruction_train_results_MAE = np.mean(train_results_MAE_vect)


# # Errors (Test): One step ahead prediction errors
# A = pd.DataFrame(NEU_AE_Reconstruction_train).iloc[1:]
# B = pd.DataFrame(X_train.T).iloc[:-1]
# test_results = B.to_numpy()-A.to_numpy()
# ### MSE
# test_results_MSE = test_results**2
# test_results_MSE_vect = np.mean(test_results_MSE,axis=1)
# NEU_AE_Reconstruction_test_results_MSE = np.mean(test_results_MSE_vect)
# ### MAE
# test_results_MAE = np.abs(test_results)
# test_results_MAE_vect = np.mean(test_results_MAE,axis=1)
# NEU_AE_Reconstruction_test_results_MAE = np.mean(test_results_MAE_vect)


# # Formatting
# ## Train
# NEU_Reconstruction_Results_train = pd.DataFrame([{'MAE':NEU_AE_Reconstruction_train_results_MAE,
#                                                  'MSE':NEU_AE_Reconstruction_train_results_MSE}],index=['NEU-AE'])
# ## Test
# NEU_Reconstruction_Results_test = pd.DataFrame([{'MAE':NEU_AE_Reconstruction_test_results_MAE,
#                                                 'MSE':NEU_AE_Reconstruction_test_results_MSE}],index=['NEU-AE'])


# # Update
# Performance_Results_train = pd.concat([Performance_Results_train,NEU_Reconstruction_Results_train],axis=0)
# Performance_Results_test = pd.concat([Performance_Results_test,NEU_Reconstruction_Results_test],axis=0)


# # Save Results #
# #--------------#
# Performance_Results_train.to_latex('outputs/tables/Fin_Performance_train.txt')
# Performance_Results_test.to_latex('outputs/tables/Fin_Performance_test.txt')


# ---
# 
# ---
# ---
# 
# ---
# ---
# 
# ---
# ---
# 
# ---
# ---
# 
# ---
# ---
# 
# ---
# ---
# 
# ---
# ---
# 
# ---
# ---
# 
# ---
# ---
# 
# ---
# ---
# 
# ---
# ---
# 
# ---
# 

# # NEU

# ## NEU: Feature Map Generation

# In[ ]:


# Perform NEU PCA
NEU_PCA_y_hat_train_pre, NEU_PCA_y_hat_test_pre, NEU_PCA, NEU_best_params = build_NEU_PCA_2(n_folds = CV_folds, 
                                                                                            n_jobs = n_jobs, 
                                                                                            n_iter = n_iter, 
                                                                                            param_grid_in = param_grid_NEU_Feature_Only_Nets, 
                                                                                            X_train_scaled = X_train_scaled, 
                                                                                            X_train = X_train,
                                                                                            X_test_scaled = X_train_scaled,
                                                                                            PCA_Rank=PCA_Rank)

# Extract Linearizing Feature Map
#================================#
Linearizing_Feature_Map = extract_trained_feature_map_PCA(NEU_PCA.model)

# Pre-process Linearized Data #
#=============================#
# Get Linearized Predictions #
#----------------------------#
data_x_featured_train = Linearizing_Feature_Map.predict(X_train_scaled)
# data_x_featured_test = Linearizing_Feature_Map.predict(X_test_scaled)

# Update User #
#=============#
# Update User on "How much the PCA dataset has been transformed"
print('Average absolute change of PCA Dataset: ' + str(np.mean(np.abs(data_x_featured_train - X_train_scaled))))


# #### Identify optimal Parameters for NEU PCA

# In[ ]:


if N_chains != 0:
    Chaining_internalheight_per_block = NEU_best_params['feature_map_height']
    Chaining_epochs_per_block = int(np.maximum(min_epochs,NEU_best_params['epochs']/N_chains))
    Chaining_learning_rate_per_block = np.minimum(np.maximum(NEU_best_params['learning_rate']/N_chains,(10**(-8))),10**(-5))
    Chaining_batchsize_per_block = NEU_best_params['batch_size']
    Chaining_output_dimension = NEU_best_params['output_dim']
    Feature_block_depth = int(np.maximum(5,NEU_best_params['feature_map_depth']))
else:
    Feature_block_depth = 1


# ## NEU-PCA

# In[ ]:


#----------------------#
# Initialization Phase #
#----------------------#
# Initialize Best to Date
best_data_x_featured_train = data_x_featured_train
# best_data_x_featured_test = data_x_featured_test
    

# Train NEU-PCA #
#---------------#
# Reconstruct Training Data
NEU_Zpca,NEU_Zpca_test,NEU_Rpca,NEU_Rpca_test = get_PCAs(X_train_scaled = (best_data_x_featured_train.T),
                                                         X_test_scaled = (best_data_x_featured_train.T),
                                                         PCA_Rank = PCA_Rank)

# Evaluate if validation loss improved
best_to_date_NEU_PCA_MAE = np.mean(np.abs(NEU_Rpca - X_train.T))
# Record Depth
best_n_chains = 0


# #### Register NEU-PCA Result(s)

# In[ ]:


# Get Results #
#-------------#
# Errors (Train): 
A = pd.DataFrame(NEU_Rpca)
B = pd.DataFrame(X_train.T)
train_results = B.to_numpy()-A.to_numpy()
### MSE
train_results_MSE = train_results**2
train_results_MSE_vect = np.mean(train_results_MSE,axis=1)
NEU_PCA_Reconstruction_train_results_MSE = np.mean(train_results_MSE_vect)
### MAE
train_results_MAE = np.abs(train_results)
train_results_MAE_vect = np.mean(train_results_MAE,axis=1)
NEU_PCA_Reconstruction_train_results_MAE = np.mean(train_results_MAE_vect)


# Errors (Test): One step ahead prediction errors
A = pd.DataFrame(NEU_Rpca).iloc[1:]
B = pd.DataFrame(X_train.T).iloc[:-1]
test_results = B.to_numpy()-A.to_numpy()
### MSE
test_results_MSE = test_results**2
test_results_MSE_vect = np.mean(test_results_MSE,axis=1)
NEU_PCA_Reconstruction_test_results_MSE = np.mean(test_results_MSE_vect)
### MAE
test_results_MAE = np.abs(test_results)
test_results_MAE_vect = np.mean(test_results_MAE,axis=1)
NEU_PCA_Reconstruction_test_results_MAE = np.mean(test_results_MAE_vect)


# Formatting
## Train
NEU_PCA_Reconstruction_Results_train = pd.DataFrame([{'MAE':NEU_PCA_Reconstruction_train_results_MAE,
                                                 'MSE':NEU_PCA_Reconstruction_train_results_MSE}],index=['NEU-PCA'])
## Test
NEU_PCA_Reconstruction_Results_test = pd.DataFrame([{'MAE':NEU_PCA_Reconstruction_train_results_MAE,
                                                'MSE':NEU_PCA_Reconstruction_train_results_MSE}],index=['NEU-PCA'])


# Update
Performance_Results_train = pd.concat([Performance_Results_train,NEU_PCA_Reconstruction_Results_train],axis=0)
Performance_Results_test = pd.concat([Performance_Results_test,NEU_PCA_Reconstruction_Results_test],axis=0)


# Save Results #
#--------------#
Performance_Results_train.to_latex('outputs/tables/Fin_Performance_train.txt')
Performance_Results_test.to_latex('outputs/tables/Fin_Performance_test.txt')


# ## NEU-Autoencoder

# In[ ]:


print("Training NEU-Autoencoder: Begin!")
NEU_AE_Reconstructed_train, NEU_AE_Reconstructed_test, NEU_AE_Factors_train, NEU_AE_Factors_test = build_autoencoder(CV_folds,
                                                                                                                     n_jobs,
                                                                                                                     n_iter,
                                                                                                                     best_data_x_featured_train.T,
                                                                                                                     X_train.T,
                                                                                                                     best_data_x_featured_train.T,
                                                                                                                     PCA_Rank)
print("Training NEU-Autoencoder: Completed!")


# ### Register Result(s)

# In[ ]:


# Get Results #
#-------------#
# Errors (Train): 
A = pd.DataFrame(NEU_AE_Reconstructed_train)
B = pd.DataFrame(X_train.T)
train_results = B.to_numpy()-A.to_numpy()
### MSE
train_results_MSE = train_results**2
train_results_MSE_vect = np.mean(train_results_MSE,axis=1)
NEU_AE_Reconstruction_train_results_MSE = np.mean(train_results_MSE_vect)
### MAE
train_results_MAE = np.abs(train_results)
train_results_MAE_vect = np.mean(train_results_MAE,axis=1)
NEU_AE_Reconstruction_train_results_MAE = np.mean(train_results_MAE_vect)


# Errors (Test): One step ahead prediction errors
A = pd.DataFrame(NEU_AE_Reconstructed_train).iloc[1:]
B = pd.DataFrame(X_train.T).iloc[:-1]
test_results = B.to_numpy()-A.to_numpy()
### MSE
test_results_MSE = test_results**2
test_results_MSE_vect = np.mean(test_results_MSE,axis=1)
NEU_AE_Reconstruction_test_results_MSE = np.mean(test_results_MSE_vect)
### MAE
test_results_MAE = np.abs(test_results)
test_results_MAE_vect = np.mean(test_results_MAE,axis=1)
NEU_AE_Reconstruction_test_results_MAE = np.mean(test_results_MAE_vect)


# Formatting
## Train
NEU_AE_Reconstruction_Results_train = pd.DataFrame([{'MAE':NEU_AE_Reconstruction_train_results_MAE,
                                                 'MSE':NEU_AE_Reconstruction_train_results_MSE}],index=['NEU-AE'])
## Test
NEU_AE_Reconstruction_Results_test = pd.DataFrame([{'MAE':NEU_AE_Reconstruction_train_results_MAE,
                                                'MSE':NEU_AE_Reconstruction_train_results_MSE}],index=['NEU-AE'])


# Update
Performance_Results_train = pd.concat([Performance_Results_train,NEU_AE_Reconstruction_Results_train],axis=0)
Performance_Results_test = pd.concat([Performance_Results_test,NEU_AE_Reconstruction_Results_test],axis=0)


# Save Results #
#--------------#
Performance_Results_train.to_latex('outputs/tables/Fin_Performance_train.txt')
Performance_Results_test.to_latex('outputs/tables/Fin_Performance_test.txt')


# # Visualize Results

# ### Feature Space(s)

# ---

# ### Factors

# In[ ]:


plt.plot(AE_Factors_train)


# ## Numerical Summary

# #### Testing Results

# In[ ]:


print(np.round(Performance_Results_test,4))
Performance_Results_test


# #### Training Results

# In[ ]:


print(np.round(Performance_Results_train,4))
Performance_Results_train


# ---

# --- ---
# # Fin
# --- ---

# ---

# ---

# ---
