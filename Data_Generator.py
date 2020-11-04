#!/usr/bin/env python
# coding: utf-8

# # Data Generator (Backend)

# #### Imports

# In[2]:


import numpy as np
import pandas as pd
# For Plotting
import matplotlib.pyplot as plt
# For quick standardization
from sklearn.preprocessing import StandardScaler


# ## Generate Data

# ### The Unknown Function
# ---
#  - 1) $\min\{\exp(\frac{-1}{(1+x)^2}),x+\cos(x)\}$. Reason: Evaluate performance for pasted functions and general badness.
#  - 2) $\cos(\exp(-x))$.  Reason: Evaluate performance for non-periodic osculations.
#  - 3) $I_{(-\infty,\frac1{2})}$.  Reason: Evaluation performance on a single jump.  
#  
#  ---

# In[ ]:


if Option_Function == "nonlocality":
    # Option 1
    def unknown_f(x):
        unknown_out = np.minimum(np.exp(-1/(1+x)**2),x+np.cos(x))
        return unknown_out

if Option_Function == "oscilatory":
    # Option 2
    def unknown_f(x):
        unknown_out = np.cos(np.exp(2+x))
        return unknown_out


if Option_Function == "jumpdiscontinuity":
    # Option 3
    def unknown_f(x):
        unknown_out = np.maximum(0,np.sign(x))
        return unknown_out
    

if Option_Function == "the_nightmare":
    # For fun: The Nightmare
    def unknown_f(x):
        unknown_out = np.min(np.exp(-1/(1+x)**2),x+np.cos(x))*np.cos(np.exp(2+x))*np.maximum(0,np.sign(x-.5))
        return unknown_out


# #### Generate Data

# In[ ]:


# Test Data #
#-----------#
step_size = (1- (-1))/N_data
data_x_test = np.arange((-1-Extrapolation_size),(1+Extrapolation_size),step_size)
data_y_test = unknown_f(data_x_test)

# Training Data #
#---------------#
data_x = np.sort(np.random.uniform(-1,1,round(N_data*Train_step_proportion)))
data_y = unknown_f(data_x)*np.random.uniform((1-Distortion),(1+Distortion),len(data_x)) + np.random.normal(0,noise_level,len(data_x))


# ### Preprocess
# 
# Coerce Data into proper shape.

# In[74]:


data_x = pd.DataFrame(data_x)
data_x_test = pd.DataFrame(data_x_test)


# Rescale Data

# In[ ]:


# Initialize Scaler
sc = StandardScaler()

# Preprocess Training Data
data_x = sc.fit_transform(data_x)

# Preprocess Test Data
data_x_test = sc.transform(data_x_test)


# *NEU Format - InDeV*

# In[76]:


data_NEU = np.concatenate((data_x,data_y.reshape(-1,D)),axis = 1)


# ## Plot Data vs. True Function

# In[77]:


# Initialize Plot #
#-----------------#
plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')

# Generate Plots #
#----------------#

# Plot Signal
plt.plot(np.array(data_x_test).reshape(-1,),data_y_test,color='gray',label='f(x)',linestyle='--')
# Plot Data
plt.scatter(np.array(data_x).reshape(-1,),data_y,color='gray',label='train', marker = '2')
plt.scatter(np.array(data_x_test).reshape(-1,),data_y_test,color='black',label='test', marker = '.')

# Format Plot #
#-------------#
plt.legend(loc="upper left")
plt.title("Model Predictions")
# Show Plot
if is_visuallty_verbose == True:
    plt.show(block=False)


# ## Report Simulation Configuration to User:

# In[ ]:


print("Simulation Confiugration Information:")
print(" ")
print("=========================================================================================================================================================")
print("We're plotting the function: " +str(Option_Function)+" with "+
      str(noise_level)+" additive noise, a distortion/model uncertainty level of"+
      str(Distortion)+", and an out-of sample window on either side of the input space of:"+
      str(Extrapolation_size)+".  We train using "+
      str(N_data)+" datapoints and have a test set conisting of "+
      str(Train_step_proportion)+"% percent of the total generated data.")
print("=========================================================================================================================================================")
print(" ")


# ---

# **Fin**
