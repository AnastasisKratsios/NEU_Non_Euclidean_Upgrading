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
# Training meta-parameters
Pre_Epochs = 200
Full_Epochs = 600

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
Extrapolation_size = .1 # (i.e.: size of test-train set domain (diameter/2))
# Train Data meta-parameters
N_data = 10**3 # (i.e.: N)
# Noise Parameters
noise_level = .5 # (i.e.: ε_i)
Distortion = .1 # (i.e.: δ_i)
# Unknown Function:
def unknown_f(x):
    return -.1*x*np.sin(x) + .1*(x**2)*np.cos(x) + .9*x*np.exp(-np.abs(x)) + .5*np.sin(4*x + 3*x**2) + .8*np.cos(7*(x**2))+ (x % .5)

