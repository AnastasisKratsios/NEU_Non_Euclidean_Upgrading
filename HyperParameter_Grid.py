#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# Verbosity Parameters
is_visuallty_verbose = True

#---------------------------------#
# General Formatting Parameter(s) #
#---------------------------------#
d = 1 # Dimension of X
D = 1 # Dimension of Y

#-------------------#
# Data Parameter(s) #
#-------------------#
# Test-set meta-parameters
Train_step_proportion = .75 # (i.e.: ratio of train to test-set sizes)
Extrapolation_size = .01 # (i.e.: size of test-train set domain (diameter/2))
# Train Data meta-parameters
N_data = 10**4 # (i.e.: N)
# Noise Parameters
noise_level = .1 # (i.e.: ε_i)
Distortion = .1 # (i.e.: δ_i)



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
    N_data = 10**3 # (i.e.: N)

    
    # Model Parameters
    #------------------#
    Epochs_dictionary = {'epochs': [10]}
    NEU_Epochs_Feature_dictionary = {'epochs': [100],
                                     'homotopy_parameter': [0.1]}
    
    NEU_Epochs_dictionary = {'epochs': [5],
                            'homotopy_parameter': [0]}
    
    Training_dictionary = {'batch_size': [8],
                               'learning_rate': [0.0001],
                               'input_dim':[d],
                               'output_dim':[D]}
    
    Vanilla_ffNN_dictionary = {'height': [5],
                               'depth': [1]}

    robustness_dictionary = {'robustness_parameter': [.1]}
    
    param_grid_NEU_readout_extra_parameters = {'readout_map_depth': [10],
                                               'readout_map_height': [5]}
    
    param_grid_NEU_feature_extra_parameters = {'feature_map_depth': [20],
                                               'feature_map_height': [10]}
                                               
    
    # Kernel Ridge #
    #--------------#
    param_grid_kernel_Ridge={"alpha": [1e0, 1e-3],
                             "gamma": np.logspace(-2, 2, 2),
                             "kernel": ["rbf", "laplacian"]}
    
    
    
    # Random Forest Grid
    #--------------------#
    Rand_Forest_Grid = {'learning_rate': [0.01],
                        'max_depth': [6],
                        'min_samples_leaf': [3],
                       'n_estimators': [100],
                       }
    
else:
    
    # Training Parameters
    #----------------------#
    # Number of Jobs (Cores to use)
    n_jobs = 4
    # Number of Random CV Draws
    n_iter = 2
    n_iter_trees = 2
    # Number of CV Folds
    CV_folds = 2
    
    
    # Model Parameters
    #------------------#
    Epochs_dictionary = {'epochs': [50,100,150,200, 400, 600, 800, 1000]}
    NEU_Epochs_Feature_dictionary = {'epochs': [50,100,150,200, 400, 600, 800, 1000],
                                     'homotopy_parameter': [0,0.00001,0.0001,0.001,0.01,0.05,0.1,0.5]}
    NEU_Epochs_dictionary = {'epochs': [50,100,150,200,300],
                            'homotopy_parameter': [0,0.00001,0.0001,0.001,0.01,0.05,0.1,0.5]}
    
    Training_dictionary = {'batch_size': [8,16,32],
                               'learning_rate': [0.001,0.0005,0.00001],
                               'input_dim':[d],
                               'output_dim':[D]}
    
    Vanilla_ffNN_dictionary = {'height': [(d+D+1),5*(d+D+1),100, 200],
                               'depth': [1,2,3,4]}

    robustness_dictionary = {'robustness_parameter': [1000,100,25,1,0.01,0.001,0.0001]}
    
    param_grid_NEU_readout_extra_parameters = {'readout_map_depth': [10,20,30,40],
                                               'readout_map_height': [(2*(d+1)),(3*(d+1)),2*(d+D+2), 50]}
    
    param_grid_NEU_feature_extra_parameters={'feature_map_depth': [10,20,30,40],
                                               'feature_map_height': [(2*(d+1)),(3*(d+1)),2*(d+D+2), 50]}
    
    # Kernel Ridge #
    #--------------#
    param_grid_kernel_Ridge={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5),
                        "kernel": ["rbf", "laplacian", "polynomial", "cosine", "sigmoid"]}
                           
    # Random Forest Grid
    #--------------------#
    Rand_Forest_Grid = {'learning_rate': [0.0001,0.0005,0.005, 0.01],
                        'max_depth': [3,4,5,6, 7, 8,9, 10, 25, 50],
                        'min_samples_leaf': [3,4, 5, 9, 17, 20,50],
                       'n_estimators': [5, 10, 25, 50, 100, 200, 400, 600]
                       }
                       
#==================================================================================#        
### Create NEU parameter disctionary by parameters joining model it is upgrading ###
#==================================================================================#
param_grid_Vanilla_Nets = {**Training_dictionary,
                       **Vanilla_ffNN_dictionary,
                       **Epochs_dictionary}

param_grid_NEU_Nets = {**Training_dictionary,
                       **robustness_dictionary,
                       **Vanilla_ffNN_dictionary,
                       **param_grid_NEU_readout_extra_parameters,
                       **param_grid_NEU_feature_extra_parameters,
                       **NEU_Epochs_Feature_dictionary}

param_grid_NEU_Feature_Only_Nets = {**Training_dictionary,
                                    **robustness_dictionary,
                                    **param_grid_NEU_feature_extra_parameters,
                                    **NEU_Epochs_Feature_dictionary}

NEU_Structure_Dictionary = {**Training_dictionary,
                            **robustness_dictionary,
                            **param_grid_NEU_readout_extra_parameters,
                            **NEU_Epochs_dictionary}

# Update User #
#-------------#
print("Parameter Grids Build and Loaded!")
