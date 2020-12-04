#!/usr/bin/env python
# coding: utf-8

# # NEU-PCA: MNIST
# - Designed and Coded by: [Anastasis Kratsios](https://people.math.ethz.ch/~kratsioa/).
# - Some Elements of the PCA analysis are forked from [this repo](https://github.com/radmerti/MVA2-PCA/blob/master/YieldCurvePCA.ipynb).

# # What is PCA?
# PCA is a two-part algorithm.  In phase 1, high-dimensional data $\mathbb{R}^D$ is mapped into a low-dimensional space ($D\gg d$) via the optimal linear (orthogonal) projection.  In phase 2, the best $d$-dimensional embedding of the features $\mathbb{R}^d$ into $\mathbb{R}^D$ is learned and used to reconstruct (as best as is possible) the high-dimensional data from this small set of features.  

# # How does NEU-PCA function?
# Since the purpous of the reconfiguration network is to learn (non-linear) topology embeddings of low-dimensional linear space then we can apply NEU to the reconstruction map phase of PCA.  Moreover, we will see that the embedding can be infered from a low-dimensional intermediate space $\mathbb{R}^N$ with $d\leq N\ll D$.  Benefits:
# - Computationally cheap,
# - Just as effective as an Autoencoder,
# - Maintain interpretation of PCA features!

# $$
# \mbox{Data}:\mathbb{R}^D \rightarrow 
# \mbox{Principal Features}: \mathbb{R}^d 
# \rightarrow 
# \mbox{Reconstructing Feature Space}: \mathbb{R}^N
# \rightarrow 
# \mbox{Embedding - Reconstruction}: \mathbb{R}^D
# .
# $$

# ## Parameters

# In[17]:


PCA_Rank = 2


# ## Imports

# In[18]:


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


# In[19]:


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

# In[20]:


# Numpy
np.random.seed(2020)
# Tensorflow
tf.random.set_seed(2020)
# Python's Seed
random.seed(2020)


# ## Load Data

# In[21]:


from tensorflow.keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784) / 255
X_test = X_test.reshape(10000, 784) / 255


# #### Pre-Process Data

# In[22]:


# Initialize Scaler
scaler = MinMaxScaler()
scaler.fit(X_train)

# Train Scaler
X_train_scaled = scaler.transform(X_train)
# Map to Test Set
X_test_scaled = scaler.transform(X_test)


# ---

# # Benchmark(s)
# ---

# ## Get PCAs

# In[23]:


print('PCA: Computing...')
# Reconstruct Training Data
Zpca,Zpca_test,Rpca,Rpca_test = get_PCAs(X_train_scaled=X_train_scaled,
                                         X_test_scaled=X_test_scaled,
                                         PCA_Rank=PCA_Rank)
print('PCA: Complete!')


# #### Get Reconstruction Result(s)

# In[24]:


# Get Results #
#-------------#
## Compute
PCA_Reconstruction_results = reporter_array(Rpca,Rpca_test,X_train,X_test)
## Organize
### Train
Performance_Results_train = pd.DataFrame(PCA_Reconstruction_results['Train'],index=PCA_Reconstruction_results.index)
Performance_Results_train.columns=['PCA']
### Test
Performance_Results_test = pd.DataFrame(PCA_Reconstruction_results['Test'],index=PCA_Reconstruction_results.index)
Performance_Results_test.columns=['PCA']

# Update Total Results #
#----------------------#
# N/A

# Save Results #
#--------------#
Performance_Results_train.to_latex('outputs/tables/MNIST_Performance_train.txt')
Performance_Results_test.to_latex('outputs/tables/MNIST_Performance_test.txt')


# ## Get (ReLU) Auto-Encoder

# In[9]:


print('Auto-encoder: Computing...')
AE_Reconstructed_train, AE_Reconstructed_test, AE_Factors_train, AE_Factors_test = build_autoencoder(CV_folds,
                                                                                                     n_jobs,
                                                                                                     n_iter,
                                                                                                     X_train_scaled,
                                                                                                     X_train,
                                                                                                     X_test_scaled,
                                                                                                     PCA_Rank)

print('Auto-encoder: Complete!')


# #### Get Reconstruction Result(s)

# In[25]:


# Get Results #
#-------------#
## Compute
AE_Reconstruction_results = reporter_array(AE_Reconstructed_train,AE_Reconstructed_test,X_train,X_test)
## Organize
### Train
AE_Performance_Results_train = pd.DataFrame(AE_Reconstruction_results['Train'],index=AE_Reconstruction_results.index)
AE_Performance_Results_train.columns=['AE']
### Test
AE_Performance_Results_test = pd.DataFrame(AE_Reconstruction_results['Test'],index=AE_Reconstruction_results.index)
AE_Performance_Results_test.columns=['AE']

# Update Total Results #
#----------------------#
Performance_Results_train = pd.concat([Performance_Results_train,AE_Performance_Results_train],axis=1)
Performance_Results_test = pd.concat([Performance_Results_test,AE_Performance_Results_test],axis=1)

# Save Results #
#--------------#
Performance_Results_train.to_latex('outputs/tables/MNIST_Performance_train.txt')
Performance_Results_test.to_latex('outputs/tables/MNIST_Performance_test.txt')


# # NEU - PCA

# In[26]:


print('NEU-PCA: Computing...')
NEU_PCA_Reconstruction_train, NEU_PCA_Reconstruction_test, NEU_PCA_Factors_train, NEU_PCA_Factors_test =  build_NEU_PCA(CV_folds, 
                                                                                                                        n_jobs, 
                                                                                                                        n_iter, 
                                                                                                                        param_grid_in, 
                                                                                                                        X_train_scaled,
                                                                                                                        X_test_scaled, 
                                                                                                                        X_train,
                                                                                                                        PCA_Rank)

print('NEU-PCA: Complete!')


# #### Get Reconstruction Result(s)

# In[27]:


# Get Results #
#-------------#
## Compute
NEU_Reconstruction_Results = reporter_array(NEU_PCA_Reconstruction_train,NEU_PCA_Reconstruction_test,X_train,X_test)
## Organize
### Train
NEU_Reconstruction_Results_train = pd.DataFrame(NEU_Reconstruction_Results['Train'],index=NEU_Reconstruction_Results.index)
NEU_Reconstruction_Results_train.columns=['NEU-PCA']
### Test
NEU_Reconstruction_Results_test = pd.DataFrame(NEU_Reconstruction_Results['Test'],index=NEU_Reconstruction_Results.index)
NEU_Reconstruction_Results_test.columns=['NEU-PCA']

# Update Total Results #
#----------------------#
Performance_Results_train = pd.concat([Performance_Results_train,NEU_Reconstruction_Results_train],axis=1)
Performance_Results_test = pd.concat([Performance_Results_test,NEU_Reconstruction_Results_test],axis=1)

# Save Results #
#--------------#
Performance_Results_train.to_latex('outputs/tables/MNIST_Performance_train.txt')
Performance_Results_test.to_latex('outputs/tables/MNIST_Performance_test.txt')


# # NEU Autoencoder

# In[ ]:


print('NEU-AE: Computing...')
NEU_PCA_Reconstruction_train, NEU_PCA_Reconstruction_test, NEU_PCA_Factors_train, NEU_PCA_Factors_test =  build_NEU_Autoencoder(CV_folds, 
                                                                                                                                n_jobs, 
                                                                                                                                n_iter, 
                                                                                                                                param_grid_in, 
                                                                                                                                X_train_scaled,
                                                                                                                                X_test_scaled, 
                                                                                                                                X_train,
                                                                                                                                PCA_Rank)
print('NEU-AE: Complete!')


# #### Get Reconstruction Result(s)

# In[ ]:


# Get Results #
#-------------#
## Compute
NEU_Reconstruction_Results2 = reporter_array(NEU_PCA_Reconstruction_train,NEU_PCA_Reconstruction_test,X_train,X_test)
## Organize
### Train
NEU_Reconstruction_Results_train2 = pd.DataFrame(NEU_Reconstruction_Results2['Train'],index=NEU_Reconstruction_Results2.index)
NEU_Reconstruction_Results_train2.columns=['NEU-PCA2']
### Test
NEU_Reconstruction_Results_test2 = pd.DataFrame(NEU_Reconstruction_Results2['Test'],index=NEU_Reconstruction_Results2.index)
NEU_Reconstruction_Results_test2.columns=['NEU-PCA2']

# Update Total Results #
#----------------------#
Performance_Results_train = pd.concat([Performance_Results_train,NEU_Reconstruction_Results_train2],axis=1)
Performance_Results_test = pd.concat([Performance_Results_test,NEU_Reconstruction_Results_test2],axis=1)

# Save Results #
#--------------#
Performance_Results_train.to_latex('outputs/tables/MNIST_Performance_train.txt')
Performance_Results_test.to_latex('outputs/tables/MNIST_Performance_test.txt')


# # Visualize Results

# ### Feature Space(s)

# In[ ]:


fig2 = plt.figure(constrained_layout=True, figsize=(16,16))
spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig2)

fig2_ax0 = fig2.add_subplot(spec2[0, 0])
fig2_ax0.set_title('PCA')
# plt.title('PCA')
fig2_ax0.scatter(Zpca[:5000,0], Zpca[:5000,1], c=Y_train[:5000], s=8, cmap='tab10')
fig2_ax0.get_xaxis().set_ticklabels([])
fig2_ax0.get_yaxis().set_ticklabels([])

# plt.subplot(122)
fig2_ax1 = fig2.add_subplot(spec2[0, 1])
# plt.title('Autoencoder')
fig2_ax1.set_title('Autoencoder')
fig2_ax1.scatter(AE_Factors_train[:5000,0], AE_Factors_train[:5000,1], c=Y_train[:5000], s=8, cmap='tab10')
fig2_ax1.get_xaxis().set_ticklabels([])
fig2_ax1.get_yaxis().set_ticklabels([])

# plt.subplot(223)
fig2_ax2 = fig2.add_subplot(spec2[1, 0])
# plt.title('NEU-PCA')
fig2_ax2.set_title('NEU-PCA')
fig2_ax2.scatter(NEU_PCA_Factors_train[:5000,0], NEU_PCA_Factors_train[:5000,1], c=Y_train[:5000], s=8, cmap='tab10')
fig2_ax2.get_xaxis().set_ticklabels([])
fig2_ax2.get_yaxis().set_ticklabels([])


plt.tight_layout()

# Save Results
fig2.savefig('outputs/plotsANDfigures/Results_Visualization_MNIST.pdf')


# ## Reconstruction(s)

# #### Testing

# In[ ]:


plt.figure(figsize=(9,4))
toPlot = (X_test, Rpca_test, AE_Reconstructed_test, NEU_PCA_Reconstruction_test)
for i in range(10):
    for j in range(4):
        ax = plt.subplot(4, 10, 10*j+i+1)
        plt.imshow(toPlot[j][i,:].reshape(28,28), interpolation="nearest", 
                   vmin=0, vmax=1)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

plt.tight_layout()

# Save Results
plt.savefig('outputs/plotsANDfigures/Results_Visualization_MNIST_Reconstruction_test.pdf')


# #### Training

# In[ ]:


plt.figure(figsize=(9,4))
toPlot = (X_train, Rpca, AE_Reconstructed_train, NEU_PCA_Reconstruction_train)
for i in range(10):
    for j in range(4):
        ax = plt.subplot(4, 10, 10*j+i+1)
        plt.imshow(toPlot[j][i,:].reshape(28,28), interpolation="nearest", 
                   vmin=0, vmax=1)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

plt.tight_layout()

# Save Results
plt.savefig('outputs/plotsANDfigures/Results_Visualization_MNIST_Reconstruction_train.pdf')


# ---

# ## Numerical Summary

# #### Testing Results

# In[ ]:


print(np.round(Performance_Results_test,4))
Performance_Results_test


# #### Training Results

# In[ ]:


print(np.round(Performance_Results_train,4))
Performance_Results_train


# --- ---
# # Fin
# --- ---
