#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#---------------------------------#
# General Formatting Parameter(s) #
#---------------------------------#
d = 1 # Dimension of X
D = 1 # Dimension of Y

#------------------------------------#
# Training/Optimization Parameter(s) #
#------------------------------------#
# Robustness Parameter
robustness_parameter = .005


#---------------------------#
# Architecture Parameter(s) #
#---------------------------#
# 1) Base Model
#---------------#
Initial_Depth = 2
Initial_Height = 50
# 2) Feature Map
#---------------#
Feature_map_depth = 100
Feature_map_height = 20
# 3) Readout Map
#---------------#
# Reconfiguration Parameters
N_Reconfigurations = 100
# Depth & Height Per Reconfiguration
Depth_per_reconfig = 50
Height_per_reconfig = 20

#-------------------#
# Data Parameter(s) #
#-------------------#
# Test-set meta-parameters
Train_step_proportion = .75 # (i.e.: ratio of train to test-set sizes)
Extrapolation_size = .01 # (i.e.: size of test-train set domain (diameter/2))
# Train Data meta-parameters
N_data = 10**3 # (i.e.: N)
# Noise Parameters
noise_level = .5 # (i.e.: ε_i)
Distortion = .1 # (i.e.: δ_i)
# Unknown Function:
def unknown_f(x):
    return -.1*x*np.sin(x) + .1*(x**2)*np.cos(x) + .9*x*np.exp(-np.abs(x)) + .5*np.sin(4*x + 3*x**2) + .8*np.cos(7*(x**2))+ (x % .5)



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
    n_jobs = 4
    # Number of Random CV Draws
    n_iter = 1
    n_iter_trees = 1#20
    # Number of CV Folds
    CV_folds = 4

    
    # Model Parameters
    #------------------#
    param_grid_Vanilla_Nets = {'batch_size': [16],
                               'epochs': [200],
                               'learning_rate': [0.0014],
                               'height': [40],
                               'depth': [2],
                               'input_dim':[d],
                               'output_dim':[D]}

    param_grid_NEU_Nets = {'batch_size': [16],
                           'epochs': [200],
                           'learning_rate': [0.0014],
                           'height': [40],
                           'depth': [1],
                           'input_dim':[d],
                           'output_dim':[D],
                           'feature_map_depth': [4],
                           'readout_map_depth': [3]}
                       
    # Random Forest Grid
    #--------------------#
    Rand_Forest_Grid = {'learning_rate': [0.1],
                        'max_depth': [2],
                        'min_samples_leaf': [1],
                       'n_estimators': [10],
                       }
    
else:
    
    # Training Parameters
    #----------------------#
    # Number of Jobs (Cores to use)
    n_jobs = 70
    # Number of Random CV Draws
    n_iter = 50
    n_iter_trees = 50
    # Number of CV Folds
    CV_folds = 4
    
    
    # Model Parameters
    #------------------#
    param_grid_Vanilla_Nets = {'batch_size': [16,32,64],
                        'epochs': [200, 400, 800, 1000, 1200, 1400],
                          'learning_rate': [0.0001,0.0005,0.005],
                          'height': [100,200, 400, 600, 800],
                           'depth': [1,2],
                          'input_dim':[d],
                           'output_dim':[D]}

    param_grid_Deep_Classifier = {'batch_size': [16,32,64],
                        'epochs': [200, 400, 800, 1000, 1200, 1400],
                        'learning_rate': [0.0001,0.0005,0.005, 0.01],
                        'height': [100,200, 400, 500,600],
                        'depth': [1,2,3,4],
                        'input_dim':[d],
                        'output_dim':[D]}
                           
    # Random Forest Grid
    #--------------------#
    Rand_Forest_Grid = {'learning_rate': [0.0001,0.0005,0.005, 0.01],
                        'max_depth': [1,2,3,4,5, 7, 10, 25, 50, 75,100, 150, 200, 300, 500],
                        'min_samples_leaf': [1,2,3,4, 5, 9, 17, 20,50,75, 100],
                       'n_estimators': [5, 10, 25, 50, 100, 200, 250]
                       }
                       