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
# \operatorname{Deep-GL}_d(x) \triangleq& f^{Depth}\circ \dots f^1(x)\\
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

# In[35]:


class Reconfiguration_unit_Feature(tf.keras.layers.Layer):
    
    def __init__(self, units=16, input_dim=32):
        super(Reconfiguration_unit_Feature, self).__init__()
        self.units = units
    
    def build(self, input_shape):
        #------------------------------------------------------------------------------------#
        # Center
        #------------------------------------------------------------------------------------#
        self.location = self.add_weight(name='location',
                                    shape=(input_shape[-1],),
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
                                        trainable=True,
                                        constraint=tf.keras.constraints.NonNeg())
        self.a = self.add_weight(name='bump_scale',
                                        shape=[1],
                                        initializer='ones',
                                        trainable=True)
        self.b = self.add_weight(name='bump_location',
                                        shape=[1],
                                        initializer='zeros',
                                        trainable=True)
        
        #------------------------------------------------------------------------------------#
        # Exponential Decay
        #------------------------------------------------------------------------------------#
        self.exponential_decay = self.add_weight(name='exponential_decay_rate',
                                                 shape=[1],
                                                 initializer=RandomUniform(minval=.5, maxval=1),
                                                 trainable=True,
                                                 constraint=tf.keras.constraints.NonNeg())
        
        #------------------------------------------------------------------------------------#
        # Mixture
        #------------------------------------------------------------------------------------#
        self.m_w2 = self.add_weight(name='weight_exponential',
                                         shape=[1],
                                         initializer='zeros',
                                         trainable=True,
                                         constraint=tf.keras.constraints.NonNeg())
        self.m_w3 = self.add_weight(name='bump',
                                     shape=[1],
                                     initializer=RandomUniform(minval=.5, maxval=1),
                                     trainable=True,
                                     constraint=tf.keras.constraints.NonNeg())
        
        #------------------------------------------------------------------------------------#
        # Tangential Map
        #------------------------------------------------------------------------------------#
        self.Id = self.add_weight(name='Identity_Matrix',
                                   shape=(input_shape[-1],input_shape[-1]),
                                   initializer='identity',
                                   trainable=False)

        # Exponential Decay (Semi-Local)  
        self.Tw2_b = self.add_weight(name='Tangential_Weights_2_b',
                                   shape=((d**2),self.units),
                                   initializer='GlorotUniform',
                                   trainable=True)
        # Bump (Localizer)  
        self.Tw2_c = self.add_weight(name='Tangential_Weights_2_c',
                                   shape=((d**2),self.units),
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

    def bump_function(self, x):
        return tf.math.exp(-self.sigma / (self.sigma - x))

        
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
        
        # Exponential Decay
        exp_decay = tf.math.exp(-self.exponential_decay*norm_inputs)
        
        
        
        
        #------------------------------------------------------------------------------------#
        # Tangential Map
        #------------------------------------------------------------------------------------#
        # Build Radial, Tangent-Space Valued Function, i.e.: C(R^d,so_d) st. f(x)=f(y) if |x|=|y|
        
        
        # Build Tangential Feed-Forward Network (Bonus)
        #-----------------------------------------------#
        # Exponential Decay
        tangential_ffNN_b = norm_inputs*exp_decay*self.Id
        tangential_ffNN_b = tf.reshape(tangential_ffNN_b,[(d**2),1])
        tangential_ffNN_b = tf.linalg.matmul(self.Tw2_b,tangential_ffNN_b)
        tangential_ffNN_b = tf.reshape(tangential_ffNN_b,[d,d])
        
        # Bump (Local Aspect)
        tangential_ffNN_c = bump_decay*norm_inputs*self.Id
        tangential_ffNN_c = tf.reshape(tangential_ffNN_c,[(d**2),1])
        tangential_ffNN_c = tf.linalg.matmul(self.Tw2_c,tangential_ffNN_c)
        tangential_ffNN_c = tf.reshape(tangential_ffNN_c,[d,d])
    
        # Map to Rotation-Matrix-Valued Function #
        #----------------------------------------#
        # No Decay
        tangential_ffNN = (tf.transpose(tangential_ffNN) - tangential_ffNN) 
        tangential_ffNN_b = (tf.transpose(tangential_ffNN_b) - tangential_ffNN_b) 
        tangential_ffNN_c = (tf.transpose(tangential_ffNN_c) - tangential_ffNN_c) 
        # Decay
        tangential_ffNN = (self.m_w2*tangential_ffNN_b) + (self.m_w3*tangential_ffNN_c) 
        
        
        # Map Infinitesimal Transformation to Matrix #
        # ------------------------------------------ #
        ## Initiality Numerical Stability
        numerical_stabilizer = tf.eye(d)*(self.num_stab_param*10**(-4))
        tangential_ffNN = tangential_ffNN + numerical_stabilizer
#         rescaler = tf.norm(tangential_ffNN)
#         tangential_ffNN = tangential_ffNN /rescaler
        ## Map to Matrix
        # Cayley Transformation (Stable):
        tangential_ffNN = tf.linalg.matmul((self.Id + tangential_ffNN),tf.linalg.inv(self.Id - tangential_ffNN)) 
        # Lie Parameterization (Numerically Unstable):  
#         tangential_ffNN = tf.linalg.expm(tangential_ffNN)
#         tangential_ffNN = tangential_ffNN * tf.math.log(rescaler)
        
        # Exponentiation and Action
        #----------------------------#
        x_out = tf.linalg.matvec(tangential_ffNN,input)
        # Apply affine shift
        x_out = x_out + self.location
        
        # Return Output
        return x_out


# In[ ]:


class fullyConnected_Reconfiguration_Unit(tf.keras.layers.Layer):

    def __init__(self, units=16, input_dim=32):
        super(fullyConnected_Dense_Invertible, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(name='Weights_ffNN',
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer='zeros',
                                 trainable=True)
        self.b = self.add_weight(name='bias_ffNN',
                                 shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs):
        expw = tf.linalg.expm(self.w)
        return tf.matmul(inputs, expw) + self.b


# In[36]:


class Reconfiguration_unit_Readout(tf.keras.layers.Layer):
    
    def __init__(self, units=16, input_dim=32):
        super(Reconfiguration_unit_Feature, self).__init__()
        self.units = units
    
    def build(self, input_shape):
        #------------------------------------------------------------------------------------#
        # Center
        #------------------------------------------------------------------------------------#
        self.location = self.add_weight(name='location',
                                    shape=(input_shape[-1],),
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
                                        trainable=True,
                                        constraint=tf.keras.constraints.NonNeg())
        self.a = self.add_weight(name='bump_scale',
                                        shape=[1],
                                        initializer='ones',
                                        trainable=True)
        self.b = self.add_weight(name='bump_location',
                                        shape=[1],
                                        initializer='zeros',
                                        trainable=True)
        
        #------------------------------------------------------------------------------------#
        # Exponential Decay
        #------------------------------------------------------------------------------------#
        self.exponential_decay = self.add_weight(name='exponential_decay_rate',
                                                 shape=[1],
                                                 initializer=RandomUniform(minval=.5, maxval=1),
                                                 trainable=True,
                                                 constraint=tf.keras.constraints.NonNeg())
        
        #------------------------------------------------------------------------------------#
        # Mixture
        #------------------------------------------------------------------------------------#
        self.m_w1 = self.add_weight(name='no_decay',
                                         shape=[1],
                                         initializer='zeros',
                                         trainable=True,
                                         constraint=tf.keras.constraints.NonNeg())
        self.m_w2 = self.add_weight(name='weight_exponential',
                                         shape=[1],
                                         initializer='zeros',
                                         trainable=True,
                                         constraint=tf.keras.constraints.NonNeg())
        self.m_w3 = self.add_weight(name='bump',
                                     shape=[1],
                                     initializer=RandomUniform(minval=.5, maxval=1),
                                     trainable=True,
                                     constraint=tf.keras.constraints.NonNeg())
        
        #------------------------------------------------------------------------------------#
        # Tangential Map
        #------------------------------------------------------------------------------------#
        self.Id = self.add_weight(name='Identity_Matrix',
                                   shape=(input_shape[-1],input_shape[-1]),
                                   initializer='identity',
                                   trainable=False)
        # No Decay (Global)
        self.Tw2 = self.add_weight(name='Tangential_Weights_2',
                                   shape=(((D)**2),self.units),
                                   initializer='GlorotUniform',
                                   trainable=True)
        # Exponential Decay (Semi-Local)  
        self.Tw2_b = self.add_weight(name='Tangential_Weights_2_b',
                                   shape=(((D)**2),self.units),
                                   initializer='GlorotUniform',
                                   trainable=True)
        # Bump (Localizer)  
        self.Tw2_c = self.add_weight(name='Tangential_Weights_2_c',
                                   shape=(((D)**2),self.units),
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

    def bump_function(self, x):
        return tf.math.exp(-self.sigma / (self.sigma - x))

        
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
        
        # Exponential Decay
        exp_decay = tf.math.exp(-self.exponential_decay*norm_inputs)
        
        
        
        
        #------------------------------------------------------------------------------------#
        # Tangential Map
        #------------------------------------------------------------------------------------#
        # Build Radial, Tangent-Space Valued Function, i.e.: C(R^d,so_d) st. f(x)=f(y) if |x|=|y|
        
        
        # Build Tangential Feed-Forward Network (Bonus)
        #-----------------------------------------------#        
        # Exponential Decay
        tangential_ffNN_b = norm_inputs*exp_decay*self.Id
        tangential_ffNN_b = tf.reshape(tangential_ffNN_b,[((D)**2),1])
        tangential_ffNN_b = tf.linalg.matmul(self.Tw2_b,tangential_ffNN_b)
        tangential_ffNN_b = tf.reshape(tangential_ffNN_b,[(D),(D)])
        
        # Bump (Local Aspect)
        tangential_ffNN_c = bump_decay*norm_inputs*self.Id
        tangential_ffNN_c = tf.reshape(tangential_ffNN_c,[((D)**2),1])
        tangential_ffNN_c = tf.linalg.matmul(self.Tw2_c,tangential_ffNN_c)
        tangential_ffNN_c = tf.reshape(tangential_ffNN_c,[(D),(D)])
    
        # Map to Rotation-Matrix-Valued Function #
        #----------------------------------------#
        # No Decay
        tangential_ffNN = (tf.transpose(tangential_ffNN) - tangential_ffNN) 
        tangential_ffNN_b = (tf.transpose(tangential_ffNN_b) - tangential_ffNN_b) 
        tangential_ffNN_c = (tf.transpose(tangential_ffNN_c) - tangential_ffNN_c) 
        # Decay
        tangential_ffNN = (self.m_w1*tangential_ffNN) + (self.m_w2*tangential_ffNN_b) + (self.m_w3*tangential_ffNN_c) 
        
        
        # Map Infinitesimal Transformation to Matrix #
        # ------------------------------------------ #
        ## Initiality Numerical Stability
        numerical_stabilizer = tf.eye(D)*(self.num_stab_param*10**(-4))
        tangential_ffNN = tangential_ffNN + numerical_stabilizer
#         rescaler = tf.norm(tangential_ffNN)
#         tangential_ffNN = tangential_ffNN /rescaler
        ## Map to Matrix
        # Cayley Transformation (Stable):
        tangential_ffNN = tf.linalg.matmul((self.Id + tangential_ffNN),tf.linalg.inv(self.Id - tangential_ffNN)) 
        # Lie Parameterization (Numerically Unstable):  
#         tangential_ffNN = tf.linalg.expm(tangential_ffNN)
        
        # Exponentiation and Action
        #----------------------------#
        x_out = tf.linalg.matvec(tangential_ffNN,input)
        # Apply affine shift
        x_out = x_out + self.location
        
        # Return Output
        return x_out


# $\sigma_{\operatorname{rescaled-swish-trainable}}:x\mapsto 2 \frac{x}{1+ \exp(-\beta x)};\qquad \beta \in [0,\infty)$.

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


# In[ ]:


class rescaled_swish_trainable(tf.keras.layers.Layer):
    
    def __init__(self, units=16, input_dim=32):
        super(rescaled_swish_trainable, self).__init__()
        self.units = units
        
    def build(self, input_shape):
        self.relulevel = self.add_weight(name='relu_level',
                                 shape=[1],
                                 initializer='ones',
                                 trainable=True,
                                 constraint=tf.keras.constraints.MinMaxNorm(min_value=-0.5, max_value=0.5))
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


# ### Reconfiguration Unit
# *Lie Version:* $$
# x \mapsto \exp\left(
# %\psi(a\|x\|+b)
# \operatorname{Skew}_d\left(
#     F(\|x\|)
# \right)
# \right) x.
# $$
# 
# *Cayley version:*
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

# ### Archictecture Builders
# This next snippet builds a (Vanilla) feed-forward network.
# - **Inputs**: *Height, Depth, Learning Rate + Input/Output Dimension Specifications.*
# 
# #### Compiles Feed-forward network with usual MAE.

# ### Readout Map Version

# In[23]:


class Reconfiguration_unit_Readout(tf.keras.layers.Layer):
    
    def __init__(self, units=16, input_dim=32):
        super(Reconfiguration_unit_Readout, self).__init__()
        self.units = units
    
    def build(self, input_shape):
        #------------------------------------------------------------------------------------#
        # Center
        #------------------------------------------------------------------------------------#
        self.location = self.add_weight(name='location',
                                    shape=(input_shape[-1],),
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
                                        trainable=True,
                                        constraint=tf.keras.constraints.NonNeg())
        self.a = self.add_weight(name='bump_scale',
                                        shape=[1],
                                        initializer='ones',
                                        trainable=True)
        self.b = self.add_weight(name='bump_location',
                                        shape=[1],
                                        initializer='zeros',
                                        trainable=True)
        
        #------------------------------------------------------------------------------------#
        # Exponential Decay
        #------------------------------------------------------------------------------------#
        self.exponential_decay = self.add_weight(name='exponential_decay_rate',
                                                 shape=[1],
                                                 initializer=RandomUniform(minval=.5, maxval=1),
                                                 trainable=True,
                                                 constraint=tf.keras.constraints.NonNeg())
        
        #------------------------------------------------------------------------------------#
        # Mixture
        #------------------------------------------------------------------------------------#
        self.m_w1 = self.add_weight(name='no_decay',
                                         shape=[1],
                                         initializer='zeros',
                                         trainable=True,
                                         constraint=tf.keras.constraints.NonNeg())
        self.m_w2 = self.add_weight(name='weight_exponential',
                                         shape=[1],
                                         initializer='zeros',
                                         trainable=True,
                                         constraint=tf.keras.constraints.NonNeg())
        self.m_w3 = self.add_weight(name='bump',
                                     shape=[1],
                                     initializer=RandomUniform(minval=.5, maxval=1),
                                     trainable=True,
                                     constraint=tf.keras.constraints.NonNeg())
        
        #------------------------------------------------------------------------------------#
        # Tangential Map
        #------------------------------------------------------------------------------------#
        self.Id = self.add_weight(name='Identity_Matrix',
                                   shape=(D,D),
                                   initializer='identity',
                                   trainable=False)
        # No Decay
        self.Tw1 = self.add_weight(name='Tangential_Weights_1',
                                   shape=(self.units,(D**2)),
                                   initializer='GlorotUniform',
                                   trainable=True)        
        self.Tw2 = self.add_weight(name='Tangential_Weights_2',
                                   shape=((D**2),self.units),
                                   initializer='GlorotUniform',
                                   trainable=True)
        self.Tb1 = self.add_weight(name='Tangential_basies_1',
                                   shape=((D**2),1),
                                   initializer='GlorotUniform',
                                   trainable=True)
        self.Tb2 = self.add_weight(name='Tangential_basies_1',
                                   shape=(D,D),
                                   initializer='GlorotUniform',
                                   trainable=True)
        # Exponential Decay
        self.Tw1_b = self.add_weight(name='Tangential_Weights_1_b',
                           shape=(self.units,(D**2)),
                           initializer='GlorotUniform',
                           trainable=True)        
        self.Tw2_b = self.add_weight(name='Tangential_Weights_2_b',
                                   shape=((D**2),self.units),
                                   initializer='GlorotUniform',
                                   trainable=True)
        self.Tb1_b = self.add_weight(name='Tangential_basies_1_b',
                                   shape=((D**2),1),
                                   initializer='GlorotUniform',
                                   trainable=True)
        self.Tb2_b = self.add_weight(name='Tangential_basies_1_b',
                                   shape=(D,D),
                                   initializer='GlorotUniform',
                                   trainable=True)
        # Bump
        self.Tw1_c = self.add_weight(name='Tangential_Weights_1_c',
                           shape=(self.units,(D**2)),
                           initializer='GlorotUniform',
                           trainable=True)        
        self.Tw2_c = self.add_weight(name='Tangential_Weights_2_c',
                                   shape=((D**2),self.units),
                                   initializer='GlorotUniform',
                                   trainable=True)
        self.Tb1_c = self.add_weight(name='Tangential_basies_1_c',
                                   shape=(((input_shape[-1])**2),1),
                                   initializer='GlorotUniform',
                                   trainable=True)
        self.Tb2_c = self.add_weight(name='Tangential_basies_1_c',
                                   shape=(D,D),
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

    def bump_function(self, x):
        return tf.math.exp(-self.sigma / (self.sigma - x))

        
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
        tangential_ffNN = tf.reshape(tangential_ffNN,[(D**2),1])
        tangential_ffNN = tangential_ffNN + self.Tb1
        
        tangential_ffNN = tf.linalg.matmul(self.Tw1,tangential_ffNN)         
        tangential_ffNN = tf.nn.relu(tangential_ffNN)
        tangential_ffNN = tf.linalg.matmul(self.Tw2,tangential_ffNN)
        tangential_ffNN = tf.reshape(tangential_ffNN,[D,D])
        tangential_ffNN = tangential_ffNN + self.Tb2
        
        # Exponential Decay
        tangential_ffNN_b = norm_inputs*exp_decay*self.Id
        tangential_ffNN_b = tf.reshape(tangential_ffNN_b,[(D**2),1])
        tangential_ffNN_b = tangential_ffNN_b + self.Tb1_b
        
        tangential_ffNN_b = tf.linalg.matmul(self.Tw1_b,tangential_ffNN_b)         
        tangential_ffNN_b = tf.nn.relu(tangential_ffNN_b)
        tangential_ffNN_b = tf.linalg.matmul(self.Tw2_b,tangential_ffNN_b)
        tangential_ffNN_b = tf.reshape(tangential_ffNN_b,[D,D])
        tangential_ffNN_b = tangential_ffNN_b + self.Tb2_b
        
        # Bump (Local Aspect)
        tangential_ffNN_c = bump_decay*norm_inputs*self.Id
        tangential_ffNN_c = tf.reshape(tangential_ffNN_c,[(D**2),1])
        tangential_ffNN_c = tangential_ffNN_c + self.Tb1_c
        
        tangential_ffNN_c = tf.linalg.matmul(self.Tw1_c,tangential_ffNN_c)         
        tangential_ffNN_c = tf.nn.relu(tangential_ffNN_c)
        tangential_ffNN_c = tf.linalg.matmul(self.Tw2_c,tangential_ffNN_c)
        tangential_ffNN_c = tf.reshape(tangential_ffNN_c,[D,D])
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
        tangential_ffNN = tangential_ffNN + tf.eye(D) *(self.num_stab_param*10**(-3))
        # Cayley Transformation (Stable):
#         tangential_ffNN = tf.linalg.matmul((self.Id + tangential_ffNN),tf.linalg.inv(self.Id - tangential_ffNN)) 
        # Lie Parameterization (Numerically Unstable):  
        tangential_ffNN = tf.linalg.expm(tangential_ffNN)
        
        # Exponentiation and Action
        #----------------------------#
        x_out = tf.linalg.matvec(tangential_ffNN,input) + self.location
#         x_out = tf.linalg.matvec(tangential_ffNN,input)
        
        # Return Output
        return x_out


# ### Feature Map Version

# In[11]:


class Reconfiguration_unit_Feature(tf.keras.layers.Layer):
    
    def __init__(self, units=16, input_dim=32):
        super(Reconfiguration_unit_Feature, self).__init__()
        self.units = units
    
    def build(self, input_shape):
        #------------------------------------------------------------------------------------#
        # Center
        #------------------------------------------------------------------------------------#
        self.location = self.add_weight(name='location',
                                    shape=(input_shape[-1],),
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
                                        trainable=True,
                                        constraint=tf.keras.constraints.NonNeg())
        self.a = self.add_weight(name='bump_scale',
                                        shape=[1],
                                        initializer='ones',
                                        trainable=True)
        self.b = self.add_weight(name='bump_location',
                                        shape=[1],
                                        initializer='zeros',
                                        trainable=True)
        
        #------------------------------------------------------------------------------------#
        # Exponential Decay
        #------------------------------------------------------------------------------------#
        self.exponential_decay = self.add_weight(name='exponential_decay_rate',
                                                 shape=[1],
                                                 initializer=RandomUniform(minval=.5, maxval=1),
                                                 trainable=True,
                                                 constraint=tf.keras.constraints.NonNeg())
        
        #------------------------------------------------------------------------------------#
        # Mixture
        #------------------------------------------------------------------------------------#
        self.m_w1 = self.add_weight(name='no_decay',
                                         shape=[1],
                                         initializer='zeros',
                                         trainable=True,
                                         constraint=tf.keras.constraints.NonNeg())
        self.m_w2 = self.add_weight(name='weight_exponential',
                                         shape=[1],
                                         initializer='zeros',
                                         trainable=True,
                                         constraint=tf.keras.constraints.NonNeg())
        self.m_w3 = self.add_weight(name='bump',
                                     shape=[1],
                                     initializer=RandomUniform(minval=.5, maxval=1),
                                     trainable=True,
                                     constraint=tf.keras.constraints.NonNeg())
        
        #------------------------------------------------------------------------------------#
        # Tangential Map
        #------------------------------------------------------------------------------------#
        self.Id = self.add_weight(name='Identity_Matrix',
                                   shape=(d,d),
                                   initializer='identity',
                                   trainable=False)
        # No Decay
        self.Tw1 = self.add_weight(name='Tangential_Weights_1',
                                   shape=(self.units,(d**2)),
                                   initializer='GlorotUniform',
                                   trainable=True)        
        self.Tw2 = self.add_weight(name='Tangential_Weights_2',
                                   shape=((d**2),self.units),
                                   initializer='GlorotUniform',
                                   trainable=True)
        self.Tb1 = self.add_weight(name='Tangential_basies_1',
                                   shape=((d**2),1),
                                   initializer='GlorotUniform',
                                   trainable=True)
        self.Tb2 = self.add_weight(name='Tangential_basies_1',
                                   shape=(d,d),
                                   initializer='GlorotUniform',
                                   trainable=True)
        # Exponential Decay
        self.Tw1_b = self.add_weight(name='Tangential_Weights_1_b',
                           shape=(self.units,(d**2)),
                           initializer='GlorotUniform',
                           trainable=True)        
        self.Tw2_b = self.add_weight(name='Tangential_Weights_2_b',
                                   shape=((d**2),self.units),
                                   initializer='GlorotUniform',
                                   trainable=True)
        self.Tb1_b = self.add_weight(name='Tangential_basies_1_b',
                                   shape=((d**2),1),
                                   initializer='GlorotUniform',
                                   trainable=True)
        self.Tb2_b = self.add_weight(name='Tangential_basies_1_b',
                                   shape=(d,d),
                                   initializer='GlorotUniform',
                                   trainable=True)
        # Bump
        self.Tw1_c = self.add_weight(name='Tangential_Weights_1_c',
                           shape=(self.units,(d**2)),
                           initializer='GlorotUniform',
                           trainable=True)        
        self.Tw2_c = self.add_weight(name='Tangential_Weights_2_c',
                                   shape=((d**2),self.units),
                                   initializer='GlorotUniform',
                                   trainable=True)
        self.Tb1_c = self.add_weight(name='Tangential_basies_1_c',
                                   shape=((d**2),1),
                                   initializer='GlorotUniform',
                                   trainable=True)
        self.Tb2_c = self.add_weight(name='Tangential_basies_1_c',
                                   shape=(d,d),
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

    def bump_function(self, x):
        return tf.math.exp(-self.sigma / (self.sigma - x))

        
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
        tangential_ffNN = tf.reshape(tangential_ffNN,[((d)**2),1])
        tangential_ffNN = tangential_ffNN + self.Tb1
        
        tangential_ffNN = tf.linalg.matmul(self.Tw1,tangential_ffNN)         
        tangential_ffNN = tf.nn.relu(tangential_ffNN)
        tangential_ffNN = tf.linalg.matmul(self.Tw2,tangential_ffNN)
        tangential_ffNN = tf.reshape(tangential_ffNN,[(d),(d)])
        tangential_ffNN = tangential_ffNN + self.Tb2
        
        # Exponential Decay
        tangential_ffNN_b = norm_inputs*exp_decay*self.Id
        tangential_ffNN_b = tf.reshape(tangential_ffNN_b,[((d)**2),1])
        tangential_ffNN_b = tangential_ffNN_b + self.Tb1_b
        
        tangential_ffNN_b = tf.linalg.matmul(self.Tw1_b,tangential_ffNN_b)         
        tangential_ffNN_b = tf.nn.relu(tangential_ffNN_b)
        tangential_ffNN_b = tf.linalg.matmul(self.Tw2_b,tangential_ffNN_b)
        tangential_ffNN_b = tf.reshape(tangential_ffNN_b,[(d),(d)])
        tangential_ffNN_b = tangential_ffNN_b + self.Tb2_b
        
        # Bump (Local Aspect)
        tangential_ffNN_c = bump_decay*norm_inputs*self.Id
        tangential_ffNN_c = tf.reshape(tangential_ffNN_c,[((d)**2),1])
        tangential_ffNN_c = tangential_ffNN_c + self.Tb1_c
        
        tangential_ffNN_c = tf.linalg.matmul(self.Tw1_c,tangential_ffNN_c)         
        tangential_ffNN_c = tf.nn.relu(tangential_ffNN_c)
        tangential_ffNN_c = tf.linalg.matmul(self.Tw2_c,tangential_ffNN_c)
        tangential_ffNN_c = tf.reshape(tangential_ffNN_c,[(d),(d)])
        tangential_ffNN_c = tangential_ffNN_c + self.Tb2_c
    
        # Map to Rotation-Matrix-Valued Function #
        #----------------------------------------#
        # No Decay
        tangential_ffNN = (tf.transpose(tangential_ffNN) - tangential_ffNN) 
        tangential_ffNN_b = (tf.transpose(tangential_ffNN_b) - tangential_ffNN_b) 
        tangential_ffNN_c = (tf.transpose(tangential_ffNN_c) - tangential_ffNN_c) 
        # Decay
        tangential_ffNN = (self.m_w1*tangential_ffNN) + (self.m_w2*tangential_ffNN_b) + (self.m_w3*tangential_ffNN_c) 
        
        # Numerical Stability
        tangential_ffNN = tangential_ffNN+ tf.eye(d)*(self.num_stab_param*10**(-3))
            
        # Cayley Transformation (Stable):
        tangential_ffNN = tf.linalg.matmul((self.Id + tangential_ffNN),tf.linalg.inv(self.Id - tangential_ffNN)) 
        # Lie Parameterization (Numerically Unstable):  
#         tangential_ffNN = tf.linalg.expm(tangential_ffNN)
        
        # Exponentiation and Action
        #----------------------------#
        x_out = tf.linalg.matvec(tangential_ffNN,input) + self.location
#         x_out = tf.linalg.matvec(tangential_ffNN,input)
        
        # Return Output
        return x_out


# #### Compiles Feed-forward network with Robust MSE.

# In[32]:


class Swish(tf.keras.layers.Layer):

    def __init__(self, beta=1.0, trainable=True, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta = beta
        self.trainable = trainable

    def build(self, input_shape):
        self.beta_factor = K.variable(self.beta,
                                      dtype=K.floatx(),
                                      name='beta_factor')
        if self.trainable:
            self._trainable_weights.append(self.beta_factor)

        super(Swish, self).build(input_shape)

    def call(self, inputs, mask=None):
        return swish(inputs, self.beta_factor)

    def get_config(self):
        config = {'beta': self.get_weights()[0] if self.trainable else self.beta,
                  'trainable': self.trainable}
        base_config = super(Swish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


# ### Benchmark ffNNs

# ### Benchmark Geometric Deep Learning Feed-Forward Architectures

# Next we implement the NEU but without using reconfiguration networks for the feature and readout maps... Instead we use the (homeomorphic) feed-forward architecture with *sub-minimal width* feed-forward architecture introduced in: [Bilokopytov and Kratsios](https://arxiv.org/pdf/2006.02341.pdf).  

# In[ ]:


def get_NAIVE_NEU_ffNN(feature_map_depth, feature_map_height, ## NEU-Feature Map Hyper-Parameter(s)
                       height, depth, ## ffNN Parameter(s)
                       readout_map_depth, readout_map_height,
                       learning_rate, input_dim, output_dim): ## Training Parameters

    
    #--------------------------------------------------#
    # Build Regular Arch.
    #--------------------------------------------------#
    #-###################-#
    # Define Model Input -#
    #-###################-#
    inputs_ffNN = tf.keras.Input(shape=(d,))
    
    
    #-###############-#
    # NEU Feature Map #
    #-###############-#
    
    # Initial Features
    inputs_ffNN_feature = Deep_GLd_Layer(d)(inputs_ffNN)
    # Higher-Order Feature Depth
    if feature_map_depth > 0:
        inputs_ffNN_feature = Deep_GLd_Layer(d)(inputs_ffNN)

    
    
    #-##############################################################-#
    #### - - - (Reparameterization of) Feed-Forward Network - - - ####
    #-##############################################################-#
    # First ffNN Layer: Reconfigured inputs -> Hidden Neurons
    x_ffNN = fullyConnected_Dense(height)(inputs_ffNN_feature)
    # Higher-Order Deep Layers: Hidden Neurons -> Hidden Neurons
    for i in range(depth):
        #----------------------#
        # Choice of Activation #
        #----------------------#
        # ReLU Activation
        x_ffNN = tf.nn.relu(x_ffNN)
        
        #-------------#
        # Dense Layer #
        #-------------#
        x_ffNN = fullyConnected_Dense(height)(x_ffNN)
    # Last ffNN Layer: Hidden Neurons -> Output Space
    x_ffNN = fullyConnected_Dense(D)(x_ffNN)     
    
    
    
    #-###########-#
    # Readout Map #
    #-###########-#
    # Input -> Input x ffNN output
    output_layer_new = tf.concat([inputs_ffNN, x_ffNN], axis=1)
    
    # Add Depth to Readout Map
    if readout_map_depth > 0:
        output_layer_new = Deep_GLd_Layer(d+D)(output_layer_new)

    # Project down from graph space to output space (from: Input x Outputs -> Outputs)
    output_layer = projection_layer(output_layer_new)
    
    
    # Define Model Output
    ffNN = tf.keras.Model(inputs_ffNN, output_layer)
    #--------------------------------------------------#
    # Define Optimizer & Compile Archs.
    #----------------------------------#
    opt = Adam(lr=learning_rate)
    ffNN.compile(optimizer=opt, loss=Robust_MSE, metrics=["mse", "mae", "mape"])

    return ffNN


# ---
# ---
# ---
# ---
# ## NEU - Network Builder(s)
# ---
# ---
# ---
# ---
# ---

# # Architecture Trainers

# #### Build and Train Feed-Forward Network

# ### NEU-ffNN Builder
# This next snippet builds the NEU for the feed-forward network; i.e.:
# $$
# f_{NEU} \triangleq \rho \circ f_{ffNN}\circ \phi
# ,
# $$
# where $\rho=p\circ \xi$, $\xi,\phi$ are reconfiguration networks, and $f_{ffNN}$ is a feed-forward network.  

# ---

# In[ ]:


#------------------------------------------------------------------------------------------------#
#                                      Define Predictive Model                                   #
#------------------------------------------------------------------------------------------------#
def get_ffNN(height, depth, learning_rate, input_dim, output_dim):
    #----------------------------#
    # Maximally Interacting Layer #
    #-----------------------------#
    # Initialize Inputs
    input_layer = tf.keras.Input(shape=(input_dim,))
   
    
    #------------------#
    #   Core Layers    #
    #------------------#
    core_layers = fullyConnected_Dense(height)(input_layer)
    # Activation
    core_layers = tf.nn.swish(core_layers)
    # Train additional Depth?
    if depth>1:
        # Add additional deep layer(s)
        for depth_i in range(1,depth):
            core_layers = fullyConnected_Dense(height)(core_layers)
            # Activation
            core_layers = tf.nn.swish(core_layers)
    
    #------------------#
    #  Readout Layers  #
    #------------------# 
    # Affine (Readout) Layer (Dense Fully Connected)
    output_layers = fullyConnected_Dense(output_dim)(core_layers)  
    # Define Input/Output Relationship (Arch.)
    trainable_layers_model = tf.keras.Model(input_layer, output_layers)
    
    
    #----------------------------------#
    # Define Optimizer & Compile Archs.
    #----------------------------------#
    opt = Adam(lr=learning_rate)
    trainable_layers_model.compile(optimizer=opt, loss="mae", metrics=["mse", "mae", "mape"])

    return trainable_layers_model


# In[21]:


def build_ffNN(n_folds , n_jobs, n_iter, param_grid_in, X_train, y_train,X_test):

    # Deep Feature Network
    ffNN_CV = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=get_ffNN, 
                                                            verbose=True)
    
    # Randomized CV
    ffNN_CVer = RandomizedSearchCV(estimator=ffNN_CV, 
                                    n_jobs=n_jobs,
                                    cv=KFold(n_folds, random_state=2020, shuffle=True),
                                    param_distributions=param_grid_in,
                                    n_iter=n_iter,
                                    return_train_score=True,
                                    random_state=2020,
                                    verbose=10)
    
    # Fit Model #
    #-----------#
    ffNN_CVer.fit(X_train,y_train)

    # Write Predictions #
    #-------------------#
    y_hat_train = ffNN_CVer.predict(X_train)
    y_hat_test = ffNN_CVer.predict(X_test)
    
    # Counter number of parameters #
    #------------------------------#
    # Extract Best Model
    best_model = ffNN_CVer.best_estimator_
    # Count Number of Parameters
    N_params_best_ffNN = np.sum([np.prod(v.get_shape().as_list()) for v in best_model.model.trainable_variables])
    
    
    #-----------------#
    # Save Full-Model #
    #-----------------#
    print('Benchmark-Model: Saving')
#     joblib.dump(best_model, './outputs/models/Benchmarks/ffNN_trained_CV.pkl', compress = 1)
    ffNN_CVer.best_params_['N_Trainable_Parameters'] = N_params_best_ffNN
    pd.DataFrame.from_dict(ffNN_CVer.best_params_,orient='index').to_latex("./outputs/models/Benchmarks/Best_Parameters.tex")
    print('Benchmark-Model: Saved')
    
    # Return Values #
    #---------------#
    return y_hat_train, y_hat_test

# Update User
#-------------#
print('Deep Feature Builder - Ready')


# #### Build and Train NEU-ffNN

# In[ ]:


def get_NEU_ffNN(height, depth, learning_rate, input_dim, output_dim, feature_map_depth, readout_map_depth, robustness_parameter):
    #--------------------------------------------------#
    # Build Regular Arch.
    #--------------------------------------------------#
    #-###################-#
    # Define Model Input -#
    #-###################-#
    input_layer = tf.keras.Input(shape=(input_dim,))
    
    
    #-###############-#
    # NEU Feature Map #
    #-###############-#
    deep_feature_map  = Reconfiguration_unit_Feature(height)(input_layer)
#     deep_feature_map = fullyConnected_Dense_Invertible(input_dim)(input_layer)
    for i_feature_depth in range(feature_map_depth):
#        # First Layer
#         deep_feature_map = Shift_Layers(input_dim)(deep_feature_map)
        deep_feature_map  = Reconfiguration_unit_Feature(height)(deep_feature_map)
        deep_feature_map = rescaled_swish_trainable()(deep_feature_map)
#         deep_feature_map = fullyConnected_Dense_Invertible(input_dim)(deep_feature_map)
            
    
    
    #------------------#
    #   Core Layers    #
    #------------------#
    core_layers = fullyConnected_Dense(height)(input_layer)
    # Activation
    core_layers = tf.nn.swish(core_layers)
    # Train additional Depth?
    if depth>1:
        # Add additional deep layer(s)
        for depth_i in range(1,depth):
            core_layers = fullyConnected_Dense(height)(core_layers)
            # Activation
            core_layers = tf.nn.swish(core_layers)
    
    #------------------#
    #  Readout Layers  #
    #------------------# 
    # Affine (Readout) Layer (Dense Fully Connected)
    core_layers = fullyConnected_Dense(output_dim)(core_layers)  
    
    
    #-###############-#
    # NEU Readout Map #
    #-###############-#
    deep_readout_map  = Reconfiguration_unit_Readout(height)(core_layers)
    for i_readout_depth in range(readout_map_depth):
        deep_readout_map = rescaled_swish_trainable()(deep_readout_map)
#         deep_readout_map = fullyConnected_Dense_Invertible(input_dim)(deep_readout_map)
        deep_readout_map  = Reconfiguration_unit_Readout(height)(deep_readout_map)
    
    
    # Define Input/Output Relationship (Arch.)
    trainable_layers_model = tf.keras.Model(input_layer, deep_readout_map)
    #--------------------------------------------------#
    # Define Optimizer & Compile Archs.
    #----------------------------------#
    opt = Adam(lr=learning_rate)
    trainable_layers_model.compile(optimizer=opt, loss=Robust_MSE(0.05), metrics=["mse", "mae", "mape"])

    return trainable_layers_model


# In[ ]:


def build_NEU_ffNN(n_folds , n_jobs, n_iter, param_grid_in, X_train, y_train, X_test):

    # Deep Feature Network
    NEU_ffNN_CV = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=get_NEU_ffNN, 
                                                            verbose=True)
    
    # Randomized CV
    NEU_ffNN_CV = RandomizedSearchCV(estimator=NEU_ffNN_CV, 
                                    n_jobs=n_jobs,
                                    cv=KFold(n_folds, random_state=2020, shuffle=True),
                                    param_distributions=param_grid_in,
                                    n_iter=n_iter,
                                    return_train_score=True,
                                    random_state=2020,
                                    verbose=10)
    
    # Fit Model #
    #-----------#
    NEU_ffNN_CV.fit(X_train,y_train)

    # Write Predictions #
    #-------------------#
    y_hat_train = NEU_ffNN_CV.predict(X_train)
    y_hat_test = NEU_ffNN_CV.predict(X_test)
    
    # Counter number of parameters #
    #------------------------------#
    # Extract Best Model
    best_model = NEU_ffNN_CV.best_estimator_
    # Count Number of Parameters
    N_params_best_ffNN = np.sum([np.prod(v.get_shape().as_list()) for v in best_model.model.trainable_variables])
    print('NEU-ffNN: Trained!')
    
    #-----------------#
    # Save Full-Model #
    #-----------------#
    print('NEU-ffNN: Saving')
#     joblib.dump(best_model, './outputs/models/Benchmarks/ffNN_trained_CV.pkl', compress = 1)
    NEU_ffNN_CV.best_params_['N_Trainable_Parameters'] = N_params_best_ffNN
    pd.DataFrame.from_dict(NEU_ffNN_CV.best_params_,orient='index').to_latex("./outputs/models/NEU/Best_Parameters.tex")
    print('NEU-ffNN: Saved')
    
    # Return Values #
    #---------------#
    return y_hat_train, y_hat_test

# Update User
#-------------#
print('Complete NEU-ffNN Training Procedure!!!')


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

# In[17]:


#-------------------------------#
#=### Results & Summarizing ###=#
#-------------------------------#
def reporter(y_train_hat_in,y_test_hat_in,y_train_in,y_test_in):
    # Training Performance
    Training_performance = np.array([mean_absolute_error(y_train_hat_in,y_train_in),
                                mean_squared_error(y_train_hat_in,y_train_in),
                                   mean_absolute_percentage_error(y_train_hat_in,y_train_in)])
    # Testing Performance
    Test_performance = np.array([mean_absolute_error(y_test_hat_in,y_test_in),
                                mean_squared_error(y_test_hat_in,y_test_in),
                                   mean_absolute_percentage_error(y_test_hat_in,y_test_in)])
    # Organize into Dataframe
    Performance_dataframe = pd.DataFrame({'train': Training_performance,'test': Test_performance})
    Performance_dataframe.index = ["MAE","MSE","MAPE"]
    # return output
    return Performance_dataframe


# ---
# # Fin
# ---
