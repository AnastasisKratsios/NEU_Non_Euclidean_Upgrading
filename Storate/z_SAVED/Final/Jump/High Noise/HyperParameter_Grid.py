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
factor = 4

#-------------------#
# Data Parameter(s) #
#-------------------#
# Test-set meta-parameters
Train_step_proportion = .75 # (i.e.: ratio of train to test-set sizes)
Extrapolation_size = -.005 # (i.e.: size of test-train set domain (diameter/2))
# Train Data meta-parameters
N_data = 10**4 # (i.e.: N)
# Noise Parameters
noise_level = 0.25 # (i.e.: ε_i)
Distortion = 0.25 # (i.e.: δ_i)

# Generate Data
Option_Function = "jumpdiscontinuity"


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


trial_run = False
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
    N_data = 10**4# (i.e.: N)

    
    # Model Parameters
    #------------------#
    ## General
    Training_dictionary = {'batch_size': [8],
                           'input_dim':[d],
                           'output_dim':[D]}
    

    ## Vanilla
    Training_Vanilla_dictionary = {'epochs': [2],
                                  'learning_rate': [0.0001]}
    Vanilla_ffNN_dictionary = {'height': [2],
                               'depth': [1]}
    
    ## NEU
    ### Readout
    NEU_Readout_dictionary = {'epochs': [2],
                              'learning_rate': [0.0001],
                              'homotopy_parameter': [0],
                              'readout_map_depth': [1],
                              'readout_map_height': [1],
                              'robustness_parameter': [0.0001]}
    
    ### Feature
    NEU_Feature_dictionary = {'epochs': [2],
                              'learning_rate': [0.00001],
                              'homotopy_parameter': [0],
                              'implicit_dimension': [1],
                              'feature_map_depth': [1],
                              'feature_map_height': [1],
                              'robustness_parameter': [0.1]}
                                               
    
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
                       'n_estimators': [400],
                       }
    
else:
    
    # Training Parameters
    #----------------------#
    # Number of Jobs (Cores to use)
    n_jobs = 60
    # Number of Random CV Draws
    n_iter = 80
    n_iter_trees = 50
    # Number of CV Folds
    CV_folds = 4
    
    
    # Model Parameters
    #------------------#
    ## General
    Training_dictionary = {'batch_size': [8,16,32],
                           'input_dim':[d],
                           'output_dim':[D]}
    

    ## Vanilla
    Training_Vanilla_dictionary = {'epochs': [50,100,150,200,250,300],
								'learning_rate': [0.0005,0.0001,0.00005,0.00001]}
    Vanilla_ffNN_dictionary = {'height': [10,20,30,40,50],
                               'depth': [1,2,3]}
    
    ## NEU
    ### Readout
    NEU_Readout_dictionary = {'epochs': [50,100,150,200,250,300],
                              'learning_rate': [0.0005,0.0001,0.00005,0.00001],
                              'homotopy_parameter': [0],
                              'readout_map_depth': [2],
                              'readout_map_height': [5],
                              'robustness_parameter': [0.0001,0.0005,0.001,0.005]}
    
    ### Feature
    NEU_Feature_dictionary = {'epochs': [50,100,150,200,250,300],
                              'learning_rate': [0.0005,0.0001,0.00005,0.00001],
                              'homotopy_parameter': [0],
                              'implicit_dimension': [5,10,15,20,25],
                              'feature_map_depth': [1,2,3],
                              'feature_map_height': [1,2,3,4],
                              'robustness_parameter': [0.0001,0.0005,0.001,0.005]}
    
    # Kernel Ridge #
    #--------------#
    param_grid_kernel_Ridge={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 10**3),
                        "kernel": ["rbf", "laplacian", "polynomial", "cosine", "sigmoid"]}
                           
    # Random Forest Grid
    #--------------------#
    Rand_Forest_Grid = {'learning_rate': [0.0005,0.0001,0.00005,0.00001],
                        'max_depth': [3,4,5,6, 7, 8,9, 10],
                        'min_samples_leaf': [4,5, 9, 17, 20,50],
                       'n_estimators': [1500]
                       }

    
    
#==================================================================================#        
### Create NEU parameter disctionary by parameters joining model it is upgrading ###
#==================================================================================#
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

# Update User #
#-------------#
print("Parameter Grids Build and Loaded!")
