#!/usr/bin/env python
# coding: utf-8

# # Helper Functions Depot
# This little script contains all the custom helper functions required to run any segment of the NEU and its benchmarks.

# In[1]:


# # Load Dependances and makes path(s)
# exec(open('Initializations_Dump.py').read())
# # Load Hyper( and meta) parameter(s)
# exec(open('HyperParameter_Grid.py').read())


# ---

# ## Loss Functions

# ### Mean Absolute Percentage Error

# In[30]:


# MAPE, between 0 and 100
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    y_true.shape = (y_true.shape[0], 1)
    y_pred.shape = (y_pred.shape[0], 1)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# ### Expected Shortfall Loss
# *These loss functions also have a robust representation but are only used for debugging/sanitychecking and are not included in the final version of the code or the paper itself.*

# In[2]:


def above_percentile(x, p): #assuming the input is flattened: (n,)

    samples = Kb.cast(Kb.shape(x)[0], Kb.floatx()) #batch size
    p =  (100. - p)/100.  #100% will return 0 elements, 0% will return all elements

    #selected samples
    values, indices = tf.math.top_k(x, samples)

    return values

def Robust_MSE_ES(p):
    def ES_p_loss(y_true, y_predicted):
        ses = Kb.pow(y_true-y_predicted,2)
        above = above_percentile(Kb.flatten(ses), p)
        return Kb.mean(above)
    return loss


# ### Robust MSE
# These next loss-functions are the ones used in the paper; they solve the approximate robust MSE problem:
# $$
# MSE_{robust}(x,y)\triangleq \operatorname{argmax}_{w\in \Delta_N} \sum_{n=1}^N w_n\|x_n-y_n\|^2 - \lambda \sum_{n=1}^N w_n \log\left(\frac1{N}\right)
# ;
# $$
# where $\Delta_N\triangleq \left\{w \in \mathbb{R}^N:\, 0\leq w_n\leq 1\mbox{ and } \sum_{n=1}^N w_n =1\right\}$ is the probability simplex!

# #### The Tensorflow Version

# In[1]:


# Tensorflow Version (Formulation with same arg-minima)
# @tf.function
def Entropic_Risk(y_true, y_pred):
    # Compute Exponential Utility
    loss_out = tf.math.abs((y_true - y_pred))
    loss_out = tf.math.exp(robustness_parameter*loss_out)
    loss_out = tf.math.reduce_sum(loss_out)

    # Return Value
    return loss_out

# def Robust_MSE(y_true, y_pred):
#     # Compute Exponential Utility
#     loss_out = tf.math.abs((y_true - y_pred))
#     loss_out = tf.math.exp(robustness_parameter*loss_out)
#     loss_out = tf.math.reduce_sum(loss_out)

#     # Return Value
#     return loss_out
def Robust_MSE(robustness_parameter=0.05):
    def loss(y_true, y_pred):
        # Initialize Loss
        absolute_errors_eval = tf.math.abs((y_true - y_pred))

        # Compute Exponential        
        loss_out_expweights = tf.math.exp(robustness_parameter*absolute_errors_eval)
        loss_out_expweights_totals = tf.math.reduce_sum(loss_out_expweights)
        loss_out_weighted = loss_out_expweights/tf.math.reduce_sum(loss_out_expweights)
        loss_out_weighted = loss_out_weighted*absolute_errors_eval
        loss_out_weighted = tf.math.reduce_sum(loss_out_weighted)

        # Compute Average Loss
        #loss_average = tf.math.reduce_mean(absolute_errors_eval)

        # Return Value
        loss_out = loss_out_weighted# - loss_average

        return loss_out
    return loss


# Homotopic Version:

# In[ ]:


# def Robust_MSE_homotopic(robustness_parameter=0.05, homotopy_parameter=1, input_tensor):
#     def loss(y_true, y_pred):
#         # Initialize Loss
#         absolute_errors_eval = tf.math.abs((y_true - y_pred))

#         # Compute Exponential        
#         loss_out_expweights = tf.math.exp(robustness_parameter*absolute_errors_eval)
#         loss_out_expweights_totals = tf.math.reduce_sum(loss_out_expweights)
#         loss_out_weighted = loss_out_expweights/tf.math.reduce_sum(loss_out_expweights)
#         loss_out_weighted = loss_out_weighted*absolute_errors_eval
#         loss_out_weighted = tf.math.reduce_sum(loss_out_weighted)

#         # Compute Homotopy Penalty
#         absolute_errors_eval = tf.math.abs((y_true - y_pred))

#         # Return Value
#         loss_out = loss_out_weighted# - loss_average

#         return loss_out
#     return loss


# #### The Numpy Version

# In[4]:


# Numpy Version (Full dual Version)
def Robust_MSE_numpy(y_true, y_pred):
    # Compute Exponential Utility
    loss_out = np.abs((y_true - y_pred))
    loss_out = np.exp(robustness_parameter*loss_out)
    loss_out = np.mean(loss_out)
    loss_out = np.log(loss_out)/robustness_parameter
    # Return Value
    return loss_out


# ---
# ---
# ---
# ---
# ---
# ---
# ---
# ---
# ---
# ---

# # Neural Network Related

# ---
# ---
# ---
# ---
# ---
# ---
# ---
# ---
# ---
# ---

# ---
# ## Custom Layers
# ---

# ### Classical Feed-Forward Layers

# In[5]:


class fullyConnected_Dense(tf.keras.layers.Layer):

    def __init__(self, units=16, input_dim=32):
        super(fullyConnected_Dense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(name='Weights_ffNN',
                                 shape=(input_shape[-1], self.units),
                               initializer='random_normal',
                               trainable=True)
        self.b = self.add_weight(name='bias_ffNN',
                                 shape=(self.units,),
                               initializer='random_normal',
                               trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


# ### Homeomorphism Layers:
# - Shift
# - Euclidean Group
# - Special Affine Group
# - Affine Group
# 
# - *Reconfiguration Unit*

# ## Shift $\mathbb{R}^d$  Layers

# $x \mapsto x +b$ for some trainable $b\in \mathbb{R}^{d}$

# In[6]:


class Shift_Layers(tf.keras.layers.Layer):
    
    def __init__(self, units=16, input_dim=32):
        super(Shift_Layers, self).__init__()
        self.units = units
    
    def build(self, input_shape):
        #------------------------------------------------------------------------------------#
        # Euclidean Parameters
        #------------------------------------------------------------------------------------#
        self.b = self.add_weight(name='location_parameter',
                                 shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=False)
        # Wrap things up!
        super().build(input_shape)
        
    def call(self, input):        
        # Exponentiation and Action
        #----------------------------#
        x_out = input
        x_out = x_out + self.b
        
        # Return Output
        return x_out


# ## $\operatorname{E}_{d}(\mathbb{R}) \cong \mathbb{R}^d \rtimes \operatorname{O}_{d}(\mathbb{R})$  Layers
# This is the group of all isometries of $\mathbb{R}^d$.

# In[7]:


class Euclidean_Layer(tf.keras.layers.Layer):
    
    def __init__(self, units=16, input_dim=32):
        super(Euclidean_Layer, self).__init__()
        self.units = units
    
    def build(self, input_shape):
        #------------------------------------------------------------------------------------#
        # Tangential Parameters
        #------------------------------------------------------------------------------------#
        # For Numerical Stability (problems with Tensorflow's Exp rounding)
        self.Id = self.add_weight(name='Identity_Matrix',
                                   shape=(input_shape[-1],input_shape[-1]),
                                   initializer='identity',
                                   trainable=False)
        # Element of gld
        self.glw = self.add_weight(name='Tangential_Weights',
                                   shape=(input_shape[-1],input_shape[-1]),
                                   initializer='GlorotUniform',
                                   trainable=True)
        
        #------------------------------------------------------------------------------------#
        # Euclidean Parameters
        #------------------------------------------------------------------------------------#
        self.b = self.add_weight(name='location_parameter',
                                 shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=False)
        # Wrap things up!
        super().build(input_shape)
        
    def call(self, input):
        # Build Tangential Feed-Forward Network (Bonus)
        #-----------------------------------------------#
        On = tf.linalg.matmul((self.Id + self.glw),tf.linalg.inv(self.Id - self.glw))
        
        # Exponentiation and Action
        #----------------------------#
        x_out = input
        x_out = tf.linalg.matvec(On,x_out)
        x_out = x_out + self.b
        
        # Return Output
        return x_out


# ## $\operatorname{SAff}_{d}(\mathbb{R}) \cong \mathbb{R}^d \rtimes \operatorname{SL}_{d}(\mathbb{R})$  Layers

# Note: $A \in \operatorname{SL}_d(\mathbb{R})$ if and only if $A=\frac1{\sqrt[d]{\det(\exp(X))}} \exp(X)$ for some $d\times d$ matrix $X$.  
# 
# *Why?*... We use the fact that $\det(k A) = k^d \det(A)$ for any $k \in \mathbb{R}$ and any $d\times d$ matrix A.

# In[8]:


class Special_Affine_Layer(tf.keras.layers.Layer):
    
    def __init__(self, units=16, input_dim=32):
        super(Special_Affine_Layer, self).__init__()
        self.units = units
    
    def build(self, input_shape):
        #------------------------------------------------------------------------------------#
        # Tangential Parameters
        #------------------------------------------------------------------------------------#
        # For Numerical Stability (problems with Tensorflow's Exp rounding)
        self.Id = self.add_weight(name='Identity_Matrix',
                                   shape=(input_shape[-1],input_shape[-1]),
                                   initializer='identity',
                                   trainable=False)
#         self.num_stab_param = self.add_weight(name='matrix_exponential_stabilizer',
#                                               shape=[1],
#                                               initializer=RandomUniform(minval=0.0, maxval=0.01),
#                                               trainable=True,
#                                               constraint=tf.keras.constraints.NonNeg())
        # Element of gld
        self.glw = self.add_weight(name='Tangential_Weights',
                                   shape=(input_shape[-1],input_shape[-1]),
                                   initializer='GlorotUniform',
                                   trainable=True)
        
        #------------------------------------------------------------------------------------#
        # Euclidean Parameters
        #------------------------------------------------------------------------------------#
        self.b = self.add_weight(name='location_parameter',
                                 shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=False)
        # Wrap things up!
        super().build(input_shape)
        
    def call(self, input):
        # Build Tangential Feed-Forward Network (Bonus)
        #-----------------------------------------------#
        GLN = tf.linalg.expm(self.glw)
        GLN_det = tf.linalg.det(GLN)
        GLN_det = tf.pow(tf.abs(GLN_det),(1/(d+D)))
        SLN = tf.math.divide(GLN,GLN_det)
        
        # Exponentiation and Action
        #----------------------------#
        x_out = input
        x_out = tf.linalg.matvec(SLN,x_out)
        x_out = x_out + self.b
        
        # Return Output
        return x_out


# ## Deep GLd Layer:
# $$
# \begin{aligned}
# \operatorname{Deep-GL}_d(x) \triangleq& f^{Depth}\circ \dots \circ f^1(x)\\
# f^i(x)\triangleq &\exp(A_2) \operatorname{Leaky-ReLU}\left(
# \exp(A_1)x + b_1
# \right)+ b_2
# \end{aligned}
# $$
# where $A_i$ are $d\times d$ matrices and $b_i \in \mathbb{R}^d$. 

# In[9]:


class Deep_GLd_Layer(tf.keras.layers.Layer):
    
    def __init__(self, units=16, input_dim=32):
        super(Deep_GLd_Layer, self).__init__()
        self.units = units
    
    def build(self, input_shape):
        #------------------------------------------------------------------------------------#
        # Tangential Parameters
        #------------------------------------------------------------------------------------#
        # For Numerical Stability (problems with Tensorflow's Exp rounding)
#         self.Id = self.add_weight(name='Identity_Matrix',
#                                    shape=(input_shape[-1],input_shape[-1]),
#                                    initializer='identity',
#                                    trainable=False)
#         self.num_stab_param = self.add_weight(name='matrix_exponential_stabilizer',
#                                               shape=[1],
#                                               initializer=RandomUniform(minval=0.0, maxval=0.01),
#                                               trainable=True,
#                                               constraint=tf.keras.constraints.NonNeg())
#         Element of gl_d
        self.glw = self.add_weight(name='Tangential_Weights',
                                   shape=(input_shape[-1],input_shape[-1]),
                                   initializer='GlorotUniform',
                                   trainable=True)
        self.glw2 = self.add_weight(name='Tangential_Weights2',
                                       shape=(input_shape[-1],input_shape[-1]),
                                       initializer='GlorotUniform',
                                       trainable=True)
        
        #------------------------------------------------------------------------------------#
        # Euclidean Parameters
        #------------------------------------------------------------------------------------#
        self.b = self.add_weight(name='location_parameter',
                                 shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=False)
        self.b2 = self.add_weight(name='location_parameter2',
                                 shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=False)
        # Wrap things up!
        super().build(input_shape)
        
    def call(self, input):
        # Build Tangential Feed-Forward Network (Bonus)
        #-----------------------------------------------#
        GLN = tf.linalg.expm(self.glw)
        GLN2 = tf.linalg.expm(self.glw2)
        
        # Exponentiation and Action
        #----------------------------#
        x_out = input

        x_out = tf.linalg.matvec(GLN,x_out)
        x_out = x_out + self.b
        x_out = tf.nn.leaky_relu(x_out)
        x_out = tf.linalg.matvec(GLN2,x_out)
        x_out = x_out + self.b2
        
        # Return Output
        return x_out


# A simple version is the following:
# This is the code use in the [background article](https://arxiv.org/abs/2006.02341).

# In[ ]:


class fullyConnected_Dense_Invertible(tf.keras.layers.Layer):

    def __init__(self, units=16, input_dim=32):
        super(fullyConnected_Dense_Invertible, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.Id = self.add_weight(name='Identity_Matrix',
                                   shape=(input_shape[-1],input_shape[-1]),
                                   initializer='identity',
                                   trainable=False)
        self.w = self.add_weight(name='Weights_ffNN',
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer='zeros',
                                 trainable=True)
        self.b = self.add_weight(name='bias_ffNN',
                                 shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)
        # Numerical Stability Parameter(s)
        #----------------------------------#
        self.num_stab_param = self.add_weight(name='weight_exponential',
                                         shape=[1],
                                         initializer='ones',
                                         trainable=False,
                                         constraint=tf.keras.constraints.NonNeg())

    def call(self, inputs):
        ## Initiality Numerical Stability
        numerical_stabilizer = tf.eye(D)*(self.num_stab_param*10**(-3))
        expw_log = self.w + numerical_stabilizer
#         rescaler = tf.norm(expw_log)
        # Cayley Transform
        expw = tf.linalg.matmul((self.Id + self.w),tf.linalg.inv(self.Id - self.w)) 
        # Lie Version
#         expw_log = expw_log /rescaler
        expw = tf.linalg.expm(expw_log)
        return tf.matmul(inputs, expw) + self.b


# ---
# ## NEU Layers
# ---

# ## New NEU
# - Feature map version (only differs up to dimension constant)
# - Readout map version
# - Constrained parametereized swish with $\beta \in \left[-\frac1{2},\frac1{2}\right]$.

# **Implemented Version:**: *Lie Version:* $$
# x \mapsto \exp\left(
# %\psi(a\|x\|+b)
# \operatorname{Skew}_d\left(
#     F(\|x\|)
# \right)
# \right) x.
# $$
# 
# **Non-Implemented Variant:**  *Cayley version:*
# $$
# \begin{aligned}
# \operatorname{Cayley}(A(x)):\,x \mapsto & \left[(I_d + A(x))(I_d- A(x))^{-1}\right]x
# \\
# A(x)\triangleq &%\psi(a\|x\|+b)
# \operatorname{Skew}_d\left(
#     F(\|x\|)\right).
# \end{aligned}
# $$
# 
# Note that the inverse of the Cayley transform of $A(x)$ is:
# $$
# \begin{aligned}
# \operatorname{Cayley}^{-1}(A(x)):\,x \mapsto & \left[(I_d - A(x))(I_d+ A(x))^{-1}\right]x
# .
# \end{aligned}
# $$

# In[94]:


class Reconfiguration_unit(tf.keras.layers.Layer):
    
    def __init__(self, units=16, input_dim=32, home_space_dim = d, homotopy_parameter = 0):
        super(Reconfiguration_unit, self).__init__()
        self.units = units
        self.home_space_dim = home_space_dim
        self.homotopy_parameter = homotopy_parameter
    
    def build(self, input_shape):
        #------------------------------------------------------------------------------------#
        # Center
        #------------------------------------------------------------------------------------#
        self.location = self.add_weight(name='location',
                                    shape=(self.home_space_dim,),
                                    initializer='random_normal',
                                    trainable=True)
        
        
        #------------------------------------------------------------------------------------#
        #====================================================================================#
        #------------------------------------------------------------------------------------#
        #====================================================================================#
        #                                  Decay Rates                                       #
        #====================================================================================#
        #------------------------------------------------------------------------------------#
        #====================================================================================#
        #------------------------------------------------------------------------------------#
        
        
        #------------------------------------------------------------------------------------#
        # Bump Function
        #------------------------------------------------------------------------------------#
        self.sigma = self.add_weight(name='bump_threshfold',
                                        shape=[1],
                                        initializer=RandomUniform(minval=.5, maxval=1),
                                        trainable=False,
                                        constraint=tf.keras.constraints.NonNeg())
        self.a = self.add_weight(name='bump_scale',
                                        shape=[1],
                                        initializer='ones',
                                        trainable=False)
        self.b = self.add_weight(name='bump_location',
                                        shape=[1],
                                        initializer='zeros',
                                        trainable=False)
        
        #------------------------------------------------------------------------------------#
        # Exponential Decay
        #------------------------------------------------------------------------------------#
        self.exponential_decay = self.add_weight(name='exponential_decay_rate',
                                                 shape=[1],
                                                 initializer=RandomUniform(minval=.5, maxval=10),
                                                 trainable=True,
                                                 constraint=tf.keras.constraints.NonNeg())
        
        #------------------------------------------------------------------------------------#
        # Mixture
        #------------------------------------------------------------------------------------#
        self.m_w1 = self.add_weight(name='no_decay',
                                    shape=[1],
                                    initializer='zeros',
                                    trainable=True,
                                    constraint=tf.keras.constraints.NonNeg(),
                                    regularizer=tf.keras.regularizers.L2(self.homotopy_parameter))
        self.m_w2 = self.add_weight(name='weight_exponential',
                                    shape=[1],
                                    initializer='zeros',
                                    trainable=True,
                                    constraint=tf.keras.constraints.NonNeg(),
                                    regularizer=tf.keras.regularizers.L2(self.homotopy_parameter))
        self.m_w3 = self.add_weight(name='bump',
                                    shape=[1],
                                    initializer='zeros',#RandomUniform(minval=.5, maxval=1),
                                    trainable=True,
                                    constraint=tf.keras.constraints.NonNeg(),
                                    regularizer=tf.keras.regularizers.L2(self.homotopy_parameter))
        
        #------------------------------------------------------------------------------------#
        # Tangential Map
        #------------------------------------------------------------------------------------#
        self.Id = self.add_weight(name='Identity_Matrix',
                                   shape=(self.home_space_dim,self.home_space_dim),
                                   initializer='identity',
                                   trainable=False)
        # No Decay
        self.Tw1 = self.add_weight(name='Tangential_Weights_1',
                                   shape=(self.units,(self.home_space_dim**2)),
                                   initializer='GlorotUniform',
                                   trainable=True)        
        self.Tw2 = self.add_weight(name='Tangential_Weights_2',
                                   shape=((self.home_space_dim**2),self.units),
                                   initializer='GlorotUniform',
                                   trainable=True)
        self.Tb1 = self.add_weight(name='Tangential_basies_1',
                                   shape=((self.home_space_dim**2),1),
                                   initializer='GlorotUniform',
                                   trainable=True)
        self.Tb2 = self.add_weight(name='Tangential_basies_1',
                                   shape=(self.home_space_dim,self.home_space_dim),
                                   initializer='GlorotUniform',
                                   trainable=True)
        # Exponential Decay
        self.Tw1_b = self.add_weight(name='Tangential_Weights_1_b',
                           shape=(self.units,(self.home_space_dim**2)),
                           initializer='GlorotUniform',
                           trainable=True)        
        self.Tw2_b = self.add_weight(name='Tangential_Weights_2_b',
                                   shape=((self.home_space_dim**2),self.units),
                                   initializer='GlorotUniform',
                                   trainable=True)
        self.Tb1_b = self.add_weight(name='Tangential_basies_1_b',
                                   shape=((self.home_space_dim**2),1),
                                   initializer='GlorotUniform',
                                   trainable=True)
        self.Tb2_b = self.add_weight(name='Tangential_basies_1_b',
                                   shape=(self.home_space_dim,self.home_space_dim),
                                   initializer='GlorotUniform',
                                   trainable=True)
        # Bump
        self.Tw1_c = self.add_weight(name='Tangential_Weights_1_c',
                           shape=(self.units,(self.home_space_dim**2)),
                           initializer='GlorotUniform',
                           trainable=True)        
        self.Tw2_c = self.add_weight(name='Tangential_Weights_2_c',
                                   shape=((self.home_space_dim**2),self.units),
                                   initializer='GlorotUniform',
                                   trainable=True)
        self.Tb1_c = self.add_weight(name='Tangential_basies_1_c',
                                   shape=((self.home_space_dim**2),1),
                                   initializer='GlorotUniform',
                                   trainable=True)
        self.Tb2_c = self.add_weight(name='Tangential_basies_1_c',
                                   shape=(self.home_space_dim,self.home_space_dim),
                                   initializer='GlorotUniform',
                                   trainable=True)
        
        
        # Numerical Stability Parameter(s)
        #----------------------------------#
        self.num_stab_param = self.add_weight(name='weight_exponential',
                                         shape=[1],
                                         initializer='ones',
                                         trainable=False,
                                         constraint=tf.keras.constraints.NonNeg())
        
        # Wrap things up!
        super().build(input_shape)

    # C^{\infty} bump function (numerically unstable...) #
    #----------------------------------------------------#
#     def bump_function(self, x):
#         return tf.math.exp(-self.sigma / (self.sigma - x))
    # C^1 bump function (numerically stable??) #
    #----------------------------------------------------#
    def bump_function(self, x):
#         return tf.math.pow(x-self.sigma,2)*tf.math.pow(x+self.sigma,2)
        bump_out = tf.math.pow(x-self.sigma,2)*tf.math.pow(x+self.sigma,2)
        bump_out = tf.math.pow(bump_out,(1/4))
        return bump_out

        
    def call(self, input):
        #------------------------------------------------------------------------------------#
        # Initializations
        #------------------------------------------------------------------------------------#
        norm_inputs = tf.norm(input) #WLOG if norm is squared!
        
        #------------------------------------------------------------------------------------#
        # Decay Rate Functions
        #------------------------------------------------------------------------------------#
        # Bump Function (Local Behaviour)
        bump_input = self.a*norm_inputs + self.b
        greater = tf.math.greater(bump_input, -self.sigma)
        less = tf.math.less(bump_input, self.sigma)
        condition = tf.logical_and(greater, less)

        bump_decay = tf.where(
            condition, 
            self.bump_function(bump_input),
            0.0)
#         bump_decay = 1
        
        # Exponential Decay
        exp_decay = tf.math.exp(-self.exponential_decay*norm_inputs)
        
        
        
        
        #------------------------------------------------------------------------------------#
        # Tangential Map
        #------------------------------------------------------------------------------------#
        # Build Radial, Tangent-Space Valued Function, i.e.: C(R^d,so_d) st. f(x)=f(y) if |x|=|y|
        
        
        # Build Tangential Feed-Forward Network (Bonus)
        #-----------------------------------------------#
        # No Decay
        tangential_ffNN = norm_inputs*self.Id
        tangential_ffNN = tf.reshape(tangential_ffNN,[(self.home_space_dim**2),1])
        tangential_ffNN = tangential_ffNN + self.Tb1
        
        tangential_ffNN = tf.linalg.matmul(self.Tw1,tangential_ffNN)         
        tangential_ffNN = tf.nn.relu(tangential_ffNN)
        tangential_ffNN = tf.linalg.matmul(self.Tw2,tangential_ffNN)
        tangential_ffNN = tf.reshape(tangential_ffNN,[self.home_space_dim,self.home_space_dim])
        tangential_ffNN = tangential_ffNN + self.Tb2
        
        # Exponential Decay
        tangential_ffNN_b = norm_inputs*exp_decay*self.Id
        tangential_ffNN_b = tf.reshape(tangential_ffNN_b,[(self.home_space_dim**2),1])
        tangential_ffNN_b = tangential_ffNN_b + self.Tb1_b
        
        tangential_ffNN_b = tf.linalg.matmul(self.Tw1_b,tangential_ffNN_b)         
        tangential_ffNN_b = tf.nn.relu(tangential_ffNN_b)
        tangential_ffNN_b = tf.linalg.matmul(self.Tw2_b,tangential_ffNN_b)
        tangential_ffNN_b = tf.reshape(tangential_ffNN_b,[self.home_space_dim,self.home_space_dim])
        tangential_ffNN_b = tangential_ffNN_b + self.Tb2_b
        
        # Bump (Local Aspect)
        tangential_ffNN_c = bump_decay*norm_inputs*self.Id
        tangential_ffNN_c = tf.reshape(tangential_ffNN_c,[(self.home_space_dim**2),1])
        tangential_ffNN_c = tangential_ffNN_c + self.Tb1_c
        
        tangential_ffNN_c = tf.linalg.matmul(self.Tw1_c,tangential_ffNN_c)         
        tangential_ffNN_c = tf.nn.relu(tangential_ffNN_c)
        tangential_ffNN_c = tf.linalg.matmul(self.Tw2_c,tangential_ffNN_c)
        tangential_ffNN_c = tf.reshape(tangential_ffNN_c,[self.home_space_dim,self.home_space_dim])
        tangential_ffNN_c = tangential_ffNN_c + self.Tb2_c
    
        # Map to Rotation-Matrix-Valued Function #
        #----------------------------------------#
        # No Decay
        tangential_ffNN = (tf.transpose(tangential_ffNN) - tangential_ffNN) 
        tangential_ffNN_b = (tf.transpose(tangential_ffNN_b) - tangential_ffNN_b) 
        tangential_ffNN_c = (tf.transpose(tangential_ffNN_c) - tangential_ffNN_c) 
        # Decay
        tangential_ffNN = (self.m_w1*tangential_ffNN) + (self.m_w2*tangential_ffNN_b) + (self.m_w3*tangential_ffNN_c) 
            
            
        # NUMERICAL STABILIZER
        #----------------------#
#         tangential_ffNN = tangential_ffNN + tf.eye(self.home_space_dim) *(self.num_stab_param*10**(-3))
        tangential_ffNN = tf.math.maximum(tf.math.minimum(-tangential_ffNN,10**(15)),-(10**(15)))
    
        # Lie Parameterization:  
        tangential_ffNN = tf.linalg.expm(tangential_ffNN)
        # Cayley Transformation (Stable):
#         tangential_ffNN = tf.linalg.matmul((self.Id + tangential_ffNN),tf.linalg.pinv(self.Id - tangential_ffNN)) 
        
        # Exponentiation and Action
        #----------------------------#
        x_out = tf.linalg.matvec(tangential_ffNN,input) + self.location
#         x_out = tf.linalg.matvec(tangential_ffNN,input)
        
        # Return Output
        return x_out


# #### Fully-connected Feed-forward layer with $GL_{d}$-connections

# In[ ]:


class fullyConnected_Dense_Invertible(tf.keras.layers.Layer):

    def __init__(self, units=16, input_dim=32):
        super(fullyConnected_Dense_Invertible, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.Id = self.add_weight(name='Identity_Matrix',
                                   shape=(input_shape[-1],input_shape[-1]),
                                   initializer='identity',
                                   trainable=False)
        self.w = self.add_weight(name='Weights_ffNN',
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer='zeros',
                                 trainable=True)
        self.b = self.add_weight(name='bias_ffNN',
                                 shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)
        # Numerical Stability Parameter(s)
        #----------------------------------#
        self.num_stab_param = self.add_weight(name='weight_exponential',
                                         shape=[1],
                                         initializer='ones',
                                         trainable=False,
                                         constraint=tf.keras.constraints.NonNeg())

    def call(self, inputs):
        ## Initiality Numerical Stability
        numerical_stabilizer = tf.eye(D)*(self.num_stab_param*10**(-3))
        expw_log = self.w + numerical_stabilizer
#         rescaler = tf.norm(expw_log)
        # Cayley Transform
#         expw = tf.linalg.matmul((self.Id + self.w),tf.linalg.inv(self.Id - self.w)) 
        # Lie Version
#         expw_log = expw_log /rescaler
        expw = tf.linalg.expm(expw_log)
        return tf.matmul(inputs, expw) + self.b


# $\sigma_{\operatorname{rescaled-swish-trainable}}:x\mapsto 2 \frac{x}{1+ \exp(-\beta x)};\qquad \beta \in [0,\infty)$.

# In[ ]:


class rescaled_swish_trainable(tf.keras.layers.Layer):
    
    def __init__(self, units=16, input_dim=32, homotopy_parameter = 0):
        super(rescaled_swish_trainable, self).__init__()
        self.units = units
        self.homotopy_parameter = homotopy_parameter
        
    def build(self, input_shape):
        self.relulevel = self.add_weight(name='relu_level',
                                         shape=[1],
                                         initializer='zeros',
                                         trainable=True,
                                         constraint=tf.keras.constraints.NonNeg(),
                                         regularizer=tf.keras.regularizers.L2(self.homotopy_parameter))
#                                  constraint=tf.keras.constraints.MinMaxNorm(min_value=-0.5, max_value=0.5))
                                
    def call(self,inputs):
        swish_numerator_rescaled = 2*inputs
        parameter = self.relulevel
        swish_denominator_trainable = tf.math.sigmoid(parameter*inputs)
        swish_trainable_out = tf.math.multiply(swish_numerator_rescaled,swish_denominator_trainable)
        return swish_trainable_out


# #### Projection Layers
# This layer maps $(x,y)\mapsto y$.

# In[12]:


projection_layer = tf.keras.layers.Lambda(lambda x: x[:, -D:])


# ---
# # Architecture Building Utilities:
# ## NEU-Related:
# ### Feature Extractor:
# This little function pops the final layer of a tensorflow model and replaces it with the identity.  When we apply it the the NEU-OLS we obtain only the trained linearizing feature map. 

# In[ ]:


def extract_trained_feature_map(model):

    # Dissasemble Network
    layers = [l for l in model.layers]

    # Define new reconfiguration unit to be added
    output_layer_new  = tf.identity(layers[len(layers)-2].output)

    for i in range(len(layers)-1):
        layers[i].trainable = False


    # build model
    new_model = tf.keras.Model(inputs=[layers[0].input], outputs=output_layer_new)
    
    # Return Output
    return new_model


# ---
# ---
# ---
# ---
# ---
# ---
# ---
# ---
# ---
# ---

# ---
# ---
# ---
# ---
# ---
# ---
# ---
# ---
# ---

# ---
# ## Reporters and Summarizers
# ---

# ---
# ---
# ---
# ---
# ---
# ---
# ---
# ---
# ---

# We empirically estimate the standard error and confidence intervals or the relevant error distributions using the method of this paper: [Bootstrap Methods for Standard Errors, Confidence Intervals, and Other Measures of Statistical Accuracy - by: B. Efron and R. Tibshirani ](https://www.jstor.org/stable/2245500?casa_token=w_8ZaRuo1qwAAAAA%3Ax5kzbYXzxGSWj-EZaC10XyOVADJyKQGXOVA9huJejP9Tt7fgMNhmPhj-C3WdgbB9AEZdqjT5q_azPmBLH6pDq61jzVFxV4XxqBuerQRBLaaOFKcyr0s&seq=1#metadata_info_tab_contents).

# In[ ]:


def bootstrap(data, n=1000, func=np.mean):
    """
    Generate `n` bootstrap samples, evaluating `func`
    at each resampling. `bootstrap` returns a function,
    which can be called to obtain confidence intervals
    of interest.
    """
    simulations = list()
    sample_size = len(data)
    xbar_init = np.mean(data)
    for c in range(n):
        itersample = np.random.choice(data, size=sample_size, replace=True)
        simulations.append(func(itersample))
    simulations.sort()
    def ci(p):
        """
        Return 2-sided symmetric confidence interval specified
        by p.
        """
        u_pval = (1+p)/2.
        l_pval = (1-u_pval)
        l_indx = int(np.floor(n*l_pval))
        u_indx = int(np.floor(n*u_pval))
        return(simulations[l_indx],xbar_init,simulations[u_indx])
    return(ci)


# #### Graphical Summarizers

# In[ ]:


def get_Error_distribution_plots(test_set_data,
                                 model_test_results,
                                 NEU_model_test_results,
                                 model_name):
    # Initialization
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")

    # Initialize NEU-Model Name
    NEU_model_name = "NEU-"+model_name
    plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')

    # Initialize Errors
    Er = data_x_test_raw.reshape(-1,) - model_test_results
    NEU_Er = data_x_test_raw.reshape(-1,) - NEU_model_test_results
    
    # Internal Computations 
    xbar_init = np.mean(Er)
    NEU_xbar_init = np.mean(NEU_Er)

    # Generate Plots #
    #----------------#
    # generate 5000 resampled sample means  =>
    means = [np.mean(np.random.choice(Er,size=len(Er),replace=True)) for i in range(5000)]
    NEU_means = [np.mean(np.random.choice(NEU_Er,size=len(NEU_Er),replace=True)) for i in range(5000)]
    sns.distplot(means, color='r', kde=True, hist_kws=dict(edgecolor="r", linewidth=.675),label=model_name)
    sns.distplot(NEU_means, color='b', kde=True, hist_kws=dict(edgecolor="b", linewidth=.675),label=NEU_model_name)
    plt.xlabel("Initial Sample Mean: {}".format(xbar_init))
    plt.title("Distribution of Sample Mean")
    plt.axvline(x=xbar_init) # vertical line at xbar_init
    plt.legend(loc="upper left")
    plt.title("Model Predictions")
    # Save Plot
    plt.savefig('./outputs/plotsANDfigures/'+model_name+'.pdf', format='pdf')
    # Show Plot
    if is_visuallty_verbose == True:
        plt.show(block=False)


# #### Numerical Summarizers

# In[17]:


#-------------------------------#
#=### Results & Summarizing ###=#
#-------------------------------#
def reporter(y_train_hat_in,y_test_hat_in,y_train_in,y_test_in, N_bootstraps=(10**4)):
    # Initialize Errors
    train_set_errors = y_train_in - y_train_hat_in
    test_set_errors = y_test_in - y_test_hat_in
    
    # Initalize and Compute: Bootstraped Confidence Intervals
    train_boot = bootstrap(train_set_errors, n=N_bootstraps)
    train_boot = np.asarray(train_boot(.95))
    test_boot = bootstrap(test_set_errors, n=N_bootstraps)
    test_boot = np.asarray(test_boot(.95))
    
    # Training Performance
    Training_performance = np.array([mean_absolute_error(y_train_hat_in,y_train_in),
                                     mean_squared_error(y_train_hat_in,y_train_in),
                                     mean_absolute_percentage_error(y_train_hat_in,y_train_in)])
    Training_performance = np.concatenate([train_boot,Training_performance])
    
    # Testing Performance
    Test_performance = np.array([mean_absolute_error(y_test_hat_in,y_test_in),
                                 mean_squared_error(y_test_hat_in,y_test_in),
                                 mean_absolute_percentage_error(y_test_hat_in,y_test_in)])
    Test_performance = np.concatenate([test_boot,Test_performance])
    
    # Organize into Dataframe
    Performance_dataframe = pd.DataFrame({'Train': Training_performance,'Test': Test_performance})
    Performance_dataframe.index = ["Er. 95L","Er. Mean","Er. 95U","MAE","MSE","MAPE"]
    return Performance_dataframe


# ---
# # Fin
# ---
