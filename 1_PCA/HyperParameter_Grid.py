#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# Verbosity Parameters
is_visuallty_verbose = False

#---------------------------------#
# General Formatting Parameter(s) #
#---------------------------------#
d = 1 # Dimension of X
D = 1 # Dimension of Y
factor = 1

#-------------------#
# Data Parameter(s) #
#-------------------#
# Test-set meta-parameters
Train_step_proportion = 1 # (i.e.: ratio of train to test-set sizes)
Extrapolation_size = 0.001 # (i.e.: size of test-train set domain (diameter/2))
# Train Data meta-parameters
N_data = 10**4 # (i.e.: N)
# Noise Parameters
noise_level = 0.25 # (i.e.: ε_i)
Distortion = 0.25 # (i.e.: δ_i)
# Generate Data
## When generating data...you may use one of the following options:
### - For evaluating non-localy patterns: "nonlocality"
### - For evaluating model performance when faced with non-stationary osculatory behaviour: "oscilatory"
### - For evaluating jump-type performance when faced with a discontinuity: "jumpdiscontinuity"
### - For a rough and noisy path: "rough"
### - For fun/debugging/sanity checking: "the_nightmare"
Option_Function = "rough"

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


trial_run = True
# This one is with larger height

# This file contains the hyper-parameter grids used to train the imprinted-tree nets.

#----------------------#
########################
# Hyperparameter Grids #
########################
#----------------------#


# Hyperparameter Grid (Readout)
#------------------------------#
if trial_run == True:

    
    # Training Parameters
    #----------------------#
    # Number of Jobs (Cores to use)
    n_jobs = 3
    # Number of Random CV Draws
    n_iter = 1
    n_iter_trees = 1#20
    # Number of CV Folds
    CV_folds = 2
    
    # JUST FOR TEST:
    N_data = 10**3

    
    # Model Parameters
    #------------------#
    ## General
    Training_dictionary = {'batch_size': [8],
                           'input_dim':[d],
                           'output_dim':[D]}
    

    ## Vanilla
    Training_Vanilla_dictionary = {'epochs': [20],
                                  'learning_rate': [0.0005]}
    
    Vanilla_ffNN_dictionary = {'height': [100],
                               'depth': [2]}
    
    ## NEU
    ### Readout
    NEU_Readout_dictionary = {'epochs': [5],
                              'learning_rate': [0.0001],
                              'homotopy_parameter': [0],
                              'readout_map_depth': [1],
                              'readout_map_height': [1],
                              'robustness_parameter': [0.0001]}
    
    ### Feature
    NEU_Feature_dictionary = {'epochs': [25],
                              'learning_rate': [0.00001],
                              'homotopy_parameter': [0],
                              'implicit_dimension': [5],
                              'feature_map_depth': [2],
                              'feature_map_height': [1],
                              'robustness_parameter': [0.01]}
                                               
    
    # Kernel Ridge #
    #--------------#
    param_grid_kernel_Ridge={"alpha": [1e0, 1e-3],
                             "gamma": np.logspace(-2, 2, 2),
                             "kernel": ["rbf", "laplacian"]}
    
    
    
    # Random Forest Grid
    #--------------------#
    Rand_Forest_Grid = {'learning_rate': [0.01],
                        'max_depth': [6],
                        'min_samples_leaf': [5],
                        'n_estimators': [400]}
                                          
    # Kernel PCA Grid 
    #-----------------#
    kPCA_grid = {'gamma': np.linspace(0.03, 0.05, 10),
                 'kernel': ['rbf', 'sigmoid', 'linear', 'poly']}
    
    # Sparse PCA Grid
    #-----------------#
    sPCA_grid = {'alpha': np.linspace(0.001, 1, 10),
                 'ridge_alpha': np.linspace(0.001,1,10)}
    
    # Autoencoder Grid #
    #------------------#
    Autoencoder_dictionary = {'batch_size': [32],
                          'epochs': [25],
                          'learning_rate': [0.0005]}
    
    NEU_PCA = {'implicit_dimension': [218],
               'feature_map_depth': [10],
               'feature_map_height': [1],
               'homotopy_parameter':[0]}



    
else:
    
    # Training Parameters
    #----------------------#
    # Number of Jobs (Cores to use)
    n_jobs = 60
    # Number of Random CV Draws
    n_iter = 40
    n_iter_trees = 40
    # Number of CV Folds
    CV_folds = 4
    
    
    # Model Parameters
    #------------------#
    ## General
    Training_dictionary = {'batch_size': [8,16,32],
                           'input_dim':[d],
                           'output_dim':[D]}
    

    ## Vanilla
    Training_Vanilla_dictionary = {'epochs': [50,100,150,200,400,600],
                                   'learning_rate': [0.0005,0.0001,0.00005,0.00001]}
    Vanilla_ffNN_dictionary = {'height': [20,50,100,150,200],
                               'depth': [1,2,3,4]}
    
    ## NEU
    ### Readout
    NEU_Readout_dictionary = {'epochs': [50,100,150,200,400,600],
                              'learning_rate': [0.0005,0.0001,0.00005,0.00001],
                              'homotopy_parameter': [0],
                              'readout_map_depth': [2],
                              'readout_map_height': [5],
                              'robustness_parameter': [0.0001,0.0005,0.001,0.005]}
    
    ### Feature
    NEU_Feature_dictionary = {'epochs': [50,100,150,200,400,600],
                              'learning_rate': [0.0005,0.0001,0.00005,0.00001],
                              'homotopy_parameter': [0],
                              'implicit_dimension': [10,25,50,100,150,200,250,300,350,400],
                              'feature_map_depth': [1,2],
                              'feature_map_height': [1,2],
                              'robustness_parameter': [0.0001,0.0005,0.001,0.005]}
    
    # Kernel Ridge #
    #--------------#
    param_grid_kernel_Ridge={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                             "gamma": np.logspace(-2, 2, 10**2),
                             "kernel": ["rbf", "laplacian", "polynomial", "cosine", "sigmoid"]}
                           
    # Random Forest Grid
    #--------------------#
    Rand_Forest_Grid = {'learning_rate': [0.0005,0.0001,0.00005,0.00001],
                        'max_depth': [3,4,5,6, 7, 8,9, 10],
                        'min_samples_leaf': [5, 9, 17, 20,50],
                        'n_estimators': [1500]}
                       
    # Kernel PCA Grid 
    #-----------------#
    kPCA_grid = {'gamma': np.linspace(0.03, 0.05, (10**2)),
                 'kernel': ['rbf', 'sigmoid', 'linear', 'poly']}
    
    # Autoencoder Grid #
    #------------------#
    Autoencoder_dictionary = {'batch_size': [8,16,32],
                              'epochs': [5,7,9,11,15,20],
                              'learning_rate': [0.005,0.001,0.0001,0.0005,0.00005]}
    
    NEU_PCA = {'implicit_dimension': [5,20,50,100],
               'feature_map_depth': [1,2,5,7,10],
               'feature_map_height': [1,2,3,5],
               'homotopy_parameter':[0]}


    
    
#==================================================================================#        
### Create NEU parameter disctionary by parameters joining model it is upgrading ###
#==================================================================================#
# Regression
param_grid_Vanilla_Nets = {**Training_dictionary,
                           **Training_Vanilla_dictionary,
                           **Vanilla_ffNN_dictionary}

param_grid_NEU_Nets = {**Training_dictionary,
                       **Vanilla_ffNN_dictionary,
                       **NEU_Readout_dictionary,
                       **NEU_Feature_dictionary,
                       **Training_Vanilla_dictionary}

param_grid_NEU_Feature_Only_Nets = {**Training_dictionary,
                                    **NEU_Feature_dictionary}

NEU_Structure_Dictionary = {**Training_dictionary,
                            **NEU_Readout_dictionary}

# PCA
param_grid_in = {**NEU_PCA,
                 **Autoencoder_dictionary}

# Update User #
#-------------#
print("Parameter Grids Build and Loaded!")
