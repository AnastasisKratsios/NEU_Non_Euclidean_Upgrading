#!/usr/bin/env python
# coding: utf-8

# # Architecture Builder
# This little script contains all the architecutre builders used in benchmarking the NEU.

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

# # Reconfiguration Networks

# ## Readout Network

# In[1]:


def get_Reconfiguration_Network_Readout(learning_rate, input_dim, output_dim, readout_map_depth,readout_map_height,robustness_parameter,homotopy_parameter):
    #--------------------------------------------------#
    # Build Regular Arch.
    #--------------------------------------------------#
    #-###################-#
    # Define Model Input -#
    #-###################-#
    input_layer = tf.keras.Input(shape=((input_dim),))
    
    
    
    #-###############-#
    # NEU Readout Map #
    #-###############-#
    deep_readout_map  = Reconfiguration_unit(units=readout_map_height,home_space_dim=(input_dim), homotopy_parameter = homotopy_parameter)(input_layer)
    for i_readout_depth in range(readout_map_depth):
        deep_readout_map = rescaled_swish_trainable(homotopy_parameter = homotopy_parameter)(deep_readout_map)
        deep_readout_map  = Reconfiguration_unit(units=readout_map_height,home_space_dim=(input_dim), homotopy_parameter = homotopy_parameter)(deep_readout_map)
        
    # Projection Layer
#     output_layer = projection_layer(deep_readout_map)
    # Trainable Output Layer
    output_layer = fullyConnected_Dense(output_dim)(deep_readout_map)
    
    
    # Define Input/Output Relationship (Arch.)
    trainable_layers_model = tf.keras.Model(input_layer, output_layer)
    #--------------------------------------------------#
    # Define Optimizer & Compile Archs.
    #----------------------------------#
    opt = Adam(lr=learning_rate)
    if robustness_parameter == 0:
        trainable_layers_model.compile(optimizer=opt, loss='mae', metrics=["mse", "mae", "mape"])
    else:
        trainable_layers_model.compile(optimizer=opt, loss=Robust_MSE(robustness_parameter), metrics=["mse", "mae", "mape"])

    return trainable_layers_model


# In[2]:


def build_NEU_Structure(n_folds , n_jobs, n_iter, param_grid_in, X_train, y_train, X_test):
    # Update Dictionary
    param_grid_in_internal = param_grid_in
    param_grid_in_internal['input_dim'] = [(X_train.shape[1])]

    # Deep Feature Network
    NEU_Structure_CV = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=get_Reconfiguration_Network_Readout, 
                                                                verbose=True)
    
    # Randomized CV
    NEU_Structure_CV = RandomizedSearchCV(estimator=NEU_Structure_CV, 
                                    n_jobs=n_jobs,
                                    cv=KFold(n_folds, random_state=2020, shuffle=True),
                                    param_distributions=param_grid_in,
                                    n_iter=n_iter,
                                    return_train_score=True,
                                    random_state=2020,
                                    verbose=10)
    
    # Pipe Standard Scaler 
    NEU_Structure_CV_mmxscaler_piped = NEU_Structure_CV
    #Pipeline([('scaler', MinMaxScaler()), ('model', NEU_Structure_CV)])
    
    # Fit Model #
    #-----------#
    NEU_Structure_CV_mmxscaler_piped.fit(X_train,y_train)

    # Write Predictions #
    #-------------------#
    y_hat_train = NEU_Structure_CV_mmxscaler_piped.predict(X_train)
    y_hat_test = NEU_Structure_CV_mmxscaler_piped.predict(X_test)
    
    # Counter number of parameters #
    #------------------------------#
    # Extract Best Model
    best_model = NEU_Structure_CV.best_estimator_
    # Count Number of Parameters
    N_params_best_ffNN = np.sum([np.prod(v.get_shape().as_list()) for v in best_model.model.trainable_variables])
    print('NEU-Structure Map: Trained!')
    
    #-----------------#
    # Save Full-Model #
    #-----------------#
    print('NEU-Structure Map: Saving')
#     joblib.dump(best_model, './outputs/models/Benchmarks/ffNN_trained_CV.pkl', compress = 1)
#     NEU_Structure_CV.best_params_['N_Trainable_Parameters'] = N_params_best_ffNN
#     pd.DataFrame.from_dict(NEU_Structure_CV.best_params_,orient='index').to_latex("./outputs/models/NEU/Best_Parameters.tex")
    print('NEU-Structure: Saved')
    
    # Return Values #
    #---------------#
    return y_hat_train, y_hat_test

# Update User
#-------------#
print('Complete NEU-Structure Building Procedure!!!')


# ## For Just Readout Version:
# In the case where $D>1$ and $\mathcal{F}$ is a universal approximator.  

# In[3]:


def get_Reconfiguration_Network_Readout_no_project(learning_rate, input_dim, output_dim, readout_map_depth,readout_map_height,robustness_parameter,homotopy_parameter):
    #--------------------------------------------------#
    # Build Regular Arch.
    #--------------------------------------------------#
    #-###################-#
    # Define Model Input -#
    #-###################-#
    input_layer = tf.keras.Input(shape=((input_dim),))
    
    
    
    #-###############-#
    # NEU Readout Map #
    #-###############-#
    deep_readout_map  = Reconfiguration_unit(units=readout_map_height,home_space_dim=(input_dim), homotopy_parameter = homotopy_parameter)(input_layer)
    for i_readout_depth in range(readout_map_depth):
        deep_readout_map = rescaled_swish_trainable(homotopy_parameter = homotopy_parameter)(deep_readout_map)
        deep_readout_map  = Reconfiguration_unit(units=readout_map_height,home_space_dim=(input_dim), homotopy_parameter = homotopy_parameter)(deep_readout_map)
    
    
    # Define Input/Output Relationship (Arch.)
    trainable_layers_model = tf.keras.Model(input_layer, deep_readout_map)
    #--------------------------------------------------#
    # Define Optimizer & Compile Archs.
    #----------------------------------#
    opt = Adam(lr=learning_rate)
    if robustness_parameter == 0:
        trainable_layers_model.compile(optimizer=opt, loss='mae', metrics=["mse", "mae", "mape"])
    else:
        trainable_layers_model.compile(optimizer=opt, loss=Robust_MSE(robustness_parameter), metrics=["mse", "mae", "mape"])

    return trainable_layers_model


# In[4]:


def build_NEU_Readout(n_folds , n_jobs, n_iter, param_grid_in, X_train, y_train, X_test):
    # Update Dictionary
    param_grid_in_internal = param_grid_in
    param_grid_in_internal['input_dim'] = [(X_train.shape[1])]

    # Deep Feature Network
    NEU_Structure_CV = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=get_Reconfiguration_Network_Readout_no_project, 
                                                                verbose=True)
    
    # Randomized CV
    NEU_Structure_CV = RandomizedSearchCV(estimator=NEU_Structure_CV, 
                                    n_jobs=n_jobs,
                                    cv=KFold(n_folds, random_state=2020, shuffle=True),
                                    param_distributions=param_grid_in,
                                    n_iter=n_iter,
                                    return_train_score=True,
                                    random_state=2020,
                                    verbose=10)
    
    # Pipe Standard Scaler 
    NEU_Structure_CV_mmxscaler_piped = NEU_Structure_CV
    #Pipeline([('scaler', MinMaxScaler()), ('model', NEU_Structure_CV)])
    
    # Fit Model #
    #-----------#
    NEU_Structure_CV_mmxscaler_piped.fit(X_train,y_train)

    # Write Predictions #
    #-------------------#
    y_hat_train = NEU_Structure_CV_mmxscaler_piped.predict(X_train)
    y_hat_test = NEU_Structure_CV_mmxscaler_piped.predict(X_test)
    
    # Counter number of parameters #
    #------------------------------#
    # Extract Best Model
    best_model = NEU_Structure_CV.best_estimator_
    # Count Number of Parameters
    N_params_best_ffNN = np.sum([np.prod(v.get_shape().as_list()) for v in best_model.model.trainable_variables])
    print('NEU-Structure Map: Trained!')
    
    #-----------------#
    # Save Full-Model #
    #-----------------#
    print('NEU-Structure Map: Saving')
#     joblib.dump(best_model, './outputs/models/Benchmarks/ffNN_trained_CV.pkl', compress = 1)
#     NEU_Structure_CV.best_params_['N_Trainable_Parameters'] = N_params_best_ffNN
#     pd.DataFrame.from_dict(NEU_Structure_CV.best_params_,orient='index').to_latex("./outputs/models/NEU/Best_Parameters.tex")
    print('NEU-Structure: Saved')
    
    # Return Values #
    #---------------#
    return y_hat_train, y_hat_test

# Update User
#-------------#
print('Complete NEU-Structure Building Procedure!!!')


# # Linear Models

# Get NEU-OLS

# In[5]:


import numpy as np


# In[6]:


def get_NEU_OLS(learning_rate, input_dim, output_dim, feature_map_depth, feature_map_height,robustness_parameter, homotopy_parameter,implicit_dimension):
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
    ##Random Embedding
    ### Compute Required Dimension
    embedding_dimension = 2*np.maximum(np.maximum(input_dim,output_dim),implicit_dimension)
    ### Execute Random Embedding
    deep_feature_map_prep = fullyConnected_Dense(embedding_dimension)(input_layer)
    deep_feature_map = tf.concat([input_layer, deep_feature_map_prep], axis=1)
    ## Homeomorphic Part
    dimension_lifted = (input_dim + embedding_dimension)
    for i_feature_depth in range(feature_map_depth):
        # First Layer
        ## Spacial-Dependent part of reconfiguration unit
        deep_feature_map  = Reconfiguration_unit(units=feature_map_height,home_space_dim=dimension_lifted, homotopy_parameter = homotopy_parameter)(deep_feature_map)
        ## Constant part of reconfiguration unit
#         deep_feature_map = fullyConnected_Dense_Invertible(input_dim)(deep_feature_map)
        ## Non-linear part of reconfiguration unit
        deep_feature_map = rescaled_swish_trainable(homotopy_parameter = homotopy_parameter)(deep_feature_map)
            
    
    
    #------------------#
    #   Core Layers    #
    #------------------#
    # Linear Readout (Really this is the OLS model)
    OLS_Layer_output = fullyConnected_Dense(output_dim)(deep_feature_map)
    
    
    
    # Define Input/Output Relationship (Arch.)
    trainable_layers_model = tf.keras.Model(input_layer, OLS_Layer_output)
    #--------------------------------------------------#
    # Define Optimizer & Compile Archs.
    #----------------------------------#
    opt = Adam(lr=learning_rate)
    if robustness_parameter == 0:
        trainable_layers_model.compile(optimizer=opt, loss='mae', metrics=["mse", "mae", "mape"])
    else:
        trainable_layers_model.compile(optimizer=opt, loss=Robust_MSE(robustness_parameter), metrics=["mse", "mae", "mape"])

    return trainable_layers_model


# Build NEU-OLS

# In[7]:


def build_NEU_OLS(n_folds , n_jobs, n_iter, param_grid_in, X_train, y_train, X_test):
    # Update Dictionary
    param_grid_in_internal = param_grid_in
    param_grid_in_internal['input_dim'] = [(X_train.shape[1])]

    # Deep Feature Network
    NEU_OLS_CV = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=get_NEU_OLS, verbose=True)
    
    # Randomized CV
    NEU_OLS_CV = RandomizedSearchCV(estimator=NEU_OLS_CV, 
                                    n_jobs=n_jobs,
                                    cv=KFold(n_folds, random_state=2020, shuffle=True),
                                    param_distributions=param_grid_in_internal,
                                    n_iter=n_iter,
                                    return_train_score=True,
                                    random_state=2020,
                                    verbose=10)
    
    # Fit Model #
    #-----------#
    NEU_OLS_CV.fit(X_train,y_train)

    # Write Predictions #
    #-------------------#
    y_hat_train = NEU_OLS_CV.predict(X_train)
    y_hat_test = NEU_OLS_CV.predict(X_test)
    
    # Counter number of parameters #
    #------------------------------#
    # Extract Best Model
    best_model = NEU_OLS_CV.best_estimator_
    # Count Number of Parameters
    N_params_best_ffNN = np.sum([np.prod(v.get_shape().as_list()) for v in best_model.model.trainable_variables])
    print('NEU-OLS: Trained!')
    
    #-----------------#
    # Save Full-Model #
    #-----------------#
    print('NEU-OLS: Saving')
#     joblib.dump(best_model, './outputs/models/Benchmarks/ffNN_trained_CV.pkl', compress = 1)
    NEU_OLS_CV.best_params_['N_Trainable_Parameters'] = N_params_best_ffNN
    Path('./outputs/models/NEU/NEU_OLS/').mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_dict(NEU_OLS_CV.best_params_,orient='index').to_latex("./outputs/models/NEU/NEU_OLS/Best_Parameters.tex")
    print('NEU-OLS: Saved')
    
    # Return Values #
    #---------------#
    return y_hat_train, y_hat_test, best_model, NEU_OLS_CV.best_params_

# Update User
#-------------#
print('Complete NEU-ffNN Training Procedure!!!')


# ---

# # Non-Linear Models: Neural Networks

# ## (Vanilla) Feed-forward neural network

# In[8]:


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


# In[9]:


def build_ffNN(n_folds , n_jobs, n_iter, param_grid_in, X_train, y_train,X_test):
    # Update Dictionary
    param_grid_in_internal = param_grid_in
    param_grid_in_internal['input_dim'] = [(X_train.shape[1])]
    
    # Deep Feature Network
    ffNN_CV = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=get_ffNN, 
                                                            verbose=True)
    
    # Randomized CV
    ffNN_CVer = RandomizedSearchCV(estimator=ffNN_CV, 
                                    n_jobs=n_jobs,
                                    cv=KFold(n_folds, random_state=2020, shuffle=True),
                                    param_distributions=param_grid_in_internal,
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


# ## NEU-Feed-forward Neural Network
# This next snippet builds the NEU for the feed-forward network; i.e.:
# $$
# f_{NEU} \triangleq \rho \circ f_{ffNN}\circ \phi
# ,
# $$
# where $\rho=p\circ \xi$, $\xi,\phi$ are reconfiguration networks, and $f_{ffNN}$ is a feed-forward network.  

# ---

# #### Build and Train NEU-ffNN

# In[10]:


def get_NEU_ffNN(height, depth, learning_rate, input_dim, output_dim, feature_map_depth, readout_map_depth, feature_map_height,readout_map_height,robustness_parameter,homotopy_parameter,implicit_dimension):

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
    ##Random Embedding
    ### Compute Required Dimension
    embedding_dimension = 2*np.maximum(np.maximum(input_dim,output_dim),implicit_dimension)
    ### Execute Random Embedding
    deep_feature_map_prep = fullyConnected_Dense(embedding_dimension)(input_layer)
    deep_feature_map = tf.concat([input_layer, deep_feature_map_prep], axis=1)
    ## Homeomorphic Part
    dimension_lifted = (input_dim + embedding_dimension)    
    ### Execute Random Embedding
    for i_feature_depth in range(feature_map_depth):
#        # First Layer
        deep_feature_map  = Reconfiguration_unit(units=feature_map_height,home_space_dim=dimension_lifted, homotopy_parameter = homotopy_parameter)(deep_feature_map)
        deep_feature_map = fullyConnected_Dense_Invertible(embedding_dimension)(input_layer)
        deep_feature_map = rescaled_swish_trainable(homotopy_parameter = homotopy_parameter)(deep_feature_map)
            
    
    
    #------------------#
    #   Core Layers    #
    #------------------#
    core_layers = fullyConnected_Dense(height)(deep_feature_map)
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
    deep_readout_map  = Reconfiguration_unit(units=readout_map_height,home_space_dim=output_dim, homotopy_parameter = homotopy_parameter)(core_layers)
    for i_readout_depth in range(readout_map_depth):
        deep_readout_map = rescaled_swish_trainable(homotopy_parameter = homotopy_parameter)(deep_readout_map)
        deep_readout_map  = Reconfiguration_unit(units=readout_map_height,home_space_dim=output_dim, homotopy_parameter = homotopy_parameter)(deep_readout_map)
    
    
    # Define Input/Output Relationship (Arch.)
    trainable_layers_model = tf.keras.Model(input_layer, deep_readout_map)
    #--------------------------------------------------#
    # Define Optimizer & Compile Archs.
    #----------------------------------#
    opt = Adam(lr=learning_rate)
    if robustness_parameter == 0:
        trainable_layers_model.compile(optimizer=opt, loss='mae', metrics=["mse", "mae", "mape"])
    else:
        trainable_layers_model.compile(optimizer=opt, loss=Robust_MSE(robustness_parameter), metrics=["mse", "mae", "mape"])

    return trainable_layers_model


# In[11]:


def build_NEU_ffNN(n_folds , n_jobs, n_iter, param_grid_in, X_train, y_train, X_test):
    # Update Dictionary
    param_grid_in_internal = param_grid_in
    param_grid_in_internal['input_dim'] = [(X_train.shape[1])]
    

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


# ## Alternative NEU-ffNN

# In[12]:


def get_NEU_ffNN_w_proj(height, depth, learning_rate, input_dim, output_dim, feature_map_depth, readout_map_depth, feature_map_height,readout_map_height,robustness_parameter,homotopy_parameter,implicit_dimension):

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
        #-###############-#
    # NEU Feature Map #
    #-###############-#
    ##Random Embedding
    ### Compute Required Dimension
    embedding_dimension = 2*np.maximum(np.maximum(input_dim,output_dim),implicit_dimension)
    ### Execute Random Embedding
    deep_feature_map_prep = fullyConnected_Dense(embedding_dimension)(input_layer)
    deep_feature_map = tf.concat([input_layer, deep_feature_map_prep], axis=1)
    ## Homeomorphic Part
    dimension_lifted = (input_dim + embedding_dimension)    
    ### Execute Random Embedding
    for i_feature_depth in range(feature_map_depth):
#        # First Layer
        deep_feature_map  = Reconfiguration_unit(units=feature_map_height,home_space_dim=dimension_lifted, homotopy_parameter = homotopy_parameter)(deep_feature_map)
        deep_feature_map = fullyConnected_Dense_Invertible(embedding_dimension)(input_layer)
        deep_feature_map = rescaled_swish_trainable(homotopy_parameter = homotopy_parameter)(deep_feature_map)
            
    
    
    #------------------#
    #   Core Layers    #
    #------------------#
    core_layers = fullyConnected_Dense(height)(deep_feature_map)
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
    
    deep_readout_map = tf.concat([input_layer, core_layers], axis=1)
    
    #-###############-#
    # NEU Readout Map #
    #-###############-#
    deep_readout_map  = Reconfiguration_unit(units=readout_map_height,home_space_dim=(output_dim+input_dim), homotopy_parameter = homotopy_parameter)(deep_readout_map)
    for i_readout_depth in range(readout_map_depth):
        deep_readout_map = rescaled_swish_trainable(homotopy_parameter = homotopy_parameter)(deep_readout_map)
        deep_readout_map  = Reconfiguration_unit(units=readout_map_height,home_space_dim=(output_dim+input_dim), homotopy_parameter = homotopy_parameter)(deep_readout_map)
    
    # Projection Layer
    output_layer = projection_layer(deep_readout_map)
    
    
    # Define Input/Output Relationship (Arch.)
    trainable_layers_model = tf.keras.Model(input_layer, output_layer)
    #--------------------------------------------------#
    # Define Optimizer & Compile Archs.
    #----------------------------------#
    opt = Adam(lr=learning_rate)
    if robustness_parameter == 0:
        trainable_layers_model.compile(optimizer=opt, loss='mae', metrics=["mse", "mae", "mape"])
    else:
        trainable_layers_model.compile(optimizer=opt, loss=Robust_MSE(robustness_parameter), metrics=["mse", "mae", "mape"])

    return trainable_layers_model


# In[13]:


def build_NEU_ffNN_w_proj(n_folds , n_jobs, n_iter, param_grid_in, X_train, y_train, X_test):
    # Update Dictionary
    param_grid_in_internal = param_grid_in
    param_grid_in_internal['input_dim'] = [(X_train.shape[1])]
    

    # Deep Feature Network
    NEU_ffNN_CV = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=get_NEU_ffNN_w_proj, 
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


# #### Get NEU-ffNN (Only Feature Map)

# In[14]:


def get_NEU_ffNN_w_feature_only(height, depth, learning_rate, input_dim, output_dim, feature_map_depth, readout_map_depth, feature_map_height,readout_map_height,robustness_parameter,homotopy_parameter,implicit_dimension):

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
    ##Random Embedding
    ### Compute Required Dimension
    embedding_dimension = 2*np.maximum(np.maximum(input_dim,output_dim),implicit_dimension)
    ### Execute Random Embedding
    deep_feature_map_prep = fullyConnected_Dense(embedding_dimension)(input_layer)
    deep_feature_map = tf.concat([input_layer, deep_feature_map_prep], axis=1)
    ## Homeomorphic Part
    dimension_lifted = (input_dim + embedding_dimension)    
    ### Execute Random Embedding
    for i_feature_depth in range(feature_map_depth):
#        # First Layer
        deep_feature_map  = Reconfiguration_unit(units=feature_map_height,home_space_dim=dimension_lifted, homotopy_parameter = homotopy_parameter)(deep_feature_map)
        deep_feature_map = fullyConnected_Dense_Invertible(embedding_dimension)(input_layer)
        deep_feature_map = rescaled_swish_trainable(homotopy_parameter = homotopy_parameter)(deep_feature_map)
            
    
    
    #------------------#
    #   Core Layers    #
    #------------------#
    core_layers = fullyConnected_Dense(height)(deep_feature_map)
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
    output_layer = fullyConnected_Dense(output_dim)(core_layers)  
    
    
    # Define Input/Output Relationship (Arch.)
    trainable_layers_model = tf.keras.Model(input_layer, output_layer)
    #--------------------------------------------------#
    # Define Optimizer & Compile Archs.
    #----------------------------------#
    opt = Adam(lr=learning_rate)
    if robustness_parameter == 0:
        trainable_layers_model.compile(optimizer=opt, loss='mae', metrics=["mse", "mae", "mape"])
    else:
        trainable_layers_model.compile(optimizer=opt, loss=Robust_MSE(robustness_parameter), metrics=["mse", "mae", "mape"])

    return trainable_layers_model


# In[15]:


def build_NEU_ffNN_w_feature_only(n_folds , n_jobs, n_iter, param_grid_in, X_train, y_train, X_test):
    # Update User
    print("Training NEU-ffNN (Fully Coupled)")
    
    # Update Dictionary
    param_grid_in_internal = param_grid_in
    param_grid_in_internal['input_dim'] = [(X_train.shape[1])]
    

    # Deep Feature Network
    NEU_ffNN_CV = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=get_NEU_ffNN_w_feature_only, 
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
    print('NEU-ffNN (Fully Coupled): Trained!')
    
    #-----------------#
    # Save Full-Model #
    #-----------------#
    print('NEU-ffNN (Fully Coupled): Saving')
#     joblib.dump(best_model, './outputs/models/Benchmarks/ffNN_trained_CV.pkl', compress = 1)
    NEU_ffNN_CV.best_params_['N_Trainable_Parameters'] = N_params_best_ffNN
    pd.DataFrame.from_dict(NEU_ffNN_CV.best_params_,orient='index').to_latex("./outputs/models/NEU/Best_Parameters.tex")
    print('NEU-ffNN (Fully Coupled): Saved')
    
    # Return Values #
    #---------------#
    return y_hat_train, y_hat_test

# Update User
#-------------#
print('Complete NEU-ffNN (Fully Coupled) Training Procedure!!!')


# ## Naive NEU-ffNN
# Next we implement the NEU but without using reconfiguration networks for the feature and readout maps... Instead we use the (homeomorphic) feed-forward architecture with *sub-minimal width* feed-forward architecture introduced in: [Bilokopytov and Kratsios](https://arxiv.org/pdf/2006.02341.pdf).  

# In[16]:


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


# In[17]:


def build_NAIVE_NEU_ffNN(n_folds , n_jobs, n_iter, param_grid_in, X_train, y_train, X_test):

    # Deep Feature Network
    NAIVE_NEU_ffNN_CV = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=get_NAIVE_NEU_ffNN, 
                                                            verbose=True)
    
    # Randomized CV
    NAIVE_NEU_ffNN_CV = RandomizedSearchCV(estimator=NAIVE_NEU_ffNN_CV, 
                                    n_jobs=n_jobs,
                                    cv=KFold(n_folds, random_state=2020, shuffle=True),
                                    param_distributions=param_grid_in,
                                    n_iter=n_iter,
                                    return_train_score=True,
                                    random_state=2020,
                                    verbose=10)
    
    # Fit Model #
    #-----------#
    NAIVE_NEU_ffNN_CV.fit(X_train,y_train)

    # Write Predictions #
    #-------------------#
    y_hat_train = NAIVE_NEU_ffNN_CV.predict(X_train)
    y_hat_test = NAIVE_NEU_ffNN_CV.predict(X_test)
    
    # Counter number of parameters #
    #------------------------------#
    # Extract Best Model
    best_model = NAIVE_NEU_ffNN_CV.best_estimator_
    # Count Number of Parameters
    N_params_best_ffNN = np.sum([np.prod(v.get_shape().as_list()) for v in best_model.model.trainable_variables])
    print('NEU-ffNN: Trained!')
    
    #-----------------#
    # Save Full-Model #
    #-----------------#
    print('NAIVE_NEU-ffNN: Saving')
#     joblib.dump(best_model, './outputs/models/Benchmarks/ffNN_trained_CV.pkl', compress = 1)
    NAIVE_NEU_ffNN_CV.best_params_['N_Trainable_Parameters'] = N_params_best_ffNN
    pd.DataFrame.from_dict(NAIVE_NEU_ffNN_CV.best_params_,orient='index').to_latex("./outputs/models/Naive_NEU/Best_Parameters.tex")
    print('NAIVE_NEU-ffNN: Saved')
    
    # Return Values #
    #---------------#
    return y_hat_train, y_hat_test

# Update User
#-------------#
print('Complete NEU-ffNN Training Procedure!!!')


# ## Fully Coupled NEU-Models

# In[18]:


def get_NEU_OLS_FullyCoupled(learning_rate, input_dim, output_dim, feature_map_depth, readout_map_depth, feature_map_height,readout_map_height,robustness_parameter,homotopy_parameter,implicit_dimension):

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
    ##Random Embedding
    ### Compute Required Dimension
    embedding_dimension = 2*np.maximum(np.maximum(input_dim,output_dim),implicit_dimension)
    ### Execute Random Embedding
    deep_feature_map_prep = fullyConnected_Dense(embedding_dimension)(input_layer)
    deep_feature_map = tf.concat([input_layer, deep_feature_map_prep], axis=1)
    ## Homeomorphic Part
    dimension_lifted = (input_dim + embedding_dimension)    
    ### Execute Random Embedding
    for i_feature_depth in range(feature_map_depth):
#        # First Layer
        deep_feature_map  = Reconfiguration_unit(units=feature_map_height,home_space_dim=dimension_lifted, homotopy_parameter = homotopy_parameter)(deep_feature_map)
        deep_feature_map = fullyConnected_Dense_Invertible(embedding_dimension)(input_layer)
        deep_feature_map = rescaled_swish_trainable(homotopy_parameter = homotopy_parameter)(deep_feature_map)
            
    
    #------------------#
    #  Readout Layers  #
    #------------------# 
    # Affine (Readout) Layer (Dense Fully Connected)
    core_layers = fullyConnected_Dense(output_dim)(deep_feature_map)  
    
    deep_readout_map = tf.concat([input_layer, core_layers], axis=1)
    
    #-###############-#
    # NEU Readout Map #
    #-###############-#
    deep_readout_map  = Reconfiguration_unit(units=readout_map_height,home_space_dim=(output_dim+input_dim), homotopy_parameter = homotopy_parameter)(deep_readout_map)
    for i_readout_depth in range(readout_map_depth):
        deep_readout_map = rescaled_swish_trainable(homotopy_parameter = homotopy_parameter)(deep_readout_map)
        deep_readout_map  = Reconfiguration_unit(units=readout_map_height,home_space_dim=(output_dim+input_dim), homotopy_parameter = homotopy_parameter)(deep_readout_map)
    
    # Projection Layer
    output_layer = projection_layer(deep_readout_map)
    
    
    # Define Input/Output Relationship (Arch.)
    trainable_layers_model = tf.keras.Model(input_layer, output_layer)
    #--------------------------------------------------#
    # Define Optimizer & Compile Archs.
    #----------------------------------#
    opt = Adam(lr=learning_rate)
    if robustness_parameter == 0:
        trainable_layers_model.compile(optimizer=opt, loss='mae', metrics=["mse", "mae", "mape"])
    else:
        trainable_layers_model.compile(optimizer=opt, loss=Robust_MSE(robustness_parameter), metrics=["mse", "mae", "mape"])

    return trainable_layers_model


# In[19]:


def build_NEU_OLS_FullyCoupled(n_folds , n_jobs, n_iter, param_grid_in, X_train, y_train, X_test):
    # Update Dictionary
    param_grid_in_internal = param_grid_in
    param_grid_in_internal['input_dim'] = [(X_train.shape[1])]
    

    # Deep Feature Network
    NEU_ffNN_CV = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=get_NEU_OLS_FullyCoupled, 
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
# # Specialized Architecture(s)
# ---

# ## For Principal Component Analysis

# In[20]:


def get_NEU_PCA(input_dim,
                learning_rate,
                PCA_Rank,
                implicit_dimension,
                feature_map_depth,
                feature_map_height,
                homotopy_parameter):
    #--------------------------------------------------#
    # Build Regular Arch.
    #--------------------------------------------------#
    #-###################-#
    # Define Model Input -#
    #-###################-#
    input_layer = tf.keras.Input(shape=(PCA_Rank,))

    #----------------------#
    # Core Layers: PCA #
    #----------------------#
#     # PCA
#     encoder = fullyConnected_Dense(PCA_Rank)(input_layer)
    # Reconstructor

    #-###############-#
    # NEU Feature Map #
    #-###############-#
    ##Random Embedding
    ### Compute Required Dimension
    embedding_dimension = 2*np.maximum(PCA_Rank,implicit_dimension)
    ### Execute Random Embedding
    deep_feature_map_prep = fullyConnected_Dense(embedding_dimension)(input_layer)
    deep_feature_map = tf.concat([input_layer, deep_feature_map_prep], axis=1)
    ## Homeomorphic Part
    dimension_lifted = (PCA_Rank + embedding_dimension)
    for i_feature_depth in range(feature_map_depth):
        # First Layer
        ## Spacial-Dependent part of reconfiguration unit
        deep_feature_map  = Reconfiguration_unit(units=feature_map_height,home_space_dim=dimension_lifted, homotopy_parameter = homotopy_parameter)(deep_feature_map)
        ## Constant part of reconfiguration unit
        deep_feature_map = rescaled_swish_trainable(homotopy_parameter = homotopy_parameter)(deep_feature_map)

    # PCA Readout (Really this is the OLS model)
    decoder = fullyConnected_Dense(input_dim)(deep_feature_map)


    # Define Input/Output Relationship (Arch.)
    NEU_PCA = tf.keras.Model(input_layer, decoder)
    #--------------------------------------------------#
    # Define Optimizer & Compile Archs.
    #----------------------------------#
    opt = Adam(lr=learning_rate)
    NEU_PCA.compile(metrics=['accuracy'],loss='mean_squared_error',optimizer='Adam')

    # Return NEU PCA
    return NEU_PCA


# In[21]:


def build_NEU_PCA(n_folds, 
                  n_jobs, 
                  n_iter, 
                  param_grid_in, 
                  X_train_scaled,
                  X_test_scaled, 
                  X_train,
                  PCA_Rank):
    # Update Dictionary
    param_grid_in_internal = param_grid_in
    param_grid_in_internal['input_dim'] = [(X_train.shape[1])]
    param_grid_in_internal['PCA_Rank'] = [PCA_Rank]

    print('Performing PCA')
    #----------------------#
    # Core Layers: PCA     #
    #----------------------#
    # Get Low-Dimensional Representation
    X_train_internal,X_test_internal, Rpca, Rpca_test = get_PCAs(X_train_scaled=X_train_scaled,
                                                                 X_test_scaled=X_test_scaled,
                                                                 PCA_Rank=PCA_Rank)
    
    X_train_internal = X_train_internal[:,:PCA_Rank]
    X_test_internal = X_test_internal[:,:PCA_Rank]
    #-------------------------------------------------#
    print('Performing PCAs Computed')
    
    # Deep Feature Network
    NEU_PCA_CV = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=get_NEU_PCA, verbose=True)

    # Randomized CV
    NEU_PCA_CV = RandomizedSearchCV(estimator=NEU_PCA_CV, 
                                    n_jobs=n_jobs,
                                    cv=KFold(n_folds, random_state=2020,shuffle=True),
                                    param_distributions=param_grid_in_internal,
                                    n_iter=n_iter,
                                    return_train_score=True,
                                    random_state=2020,
                                    verbose=10)

    # Fit Model #
    #-----------#
    print('Training NEU-Feature Map!')
    NEU_PCA_CV.fit(X_train_internal,X_train)

    # Write Predictions #
    #-------------------#
    NEU_PCA_Reconstruction_train = NEU_PCA_CV.predict(X_train_internal)
    NEU_PCA_Reconstruction_test = NEU_PCA_CV.predict(X_test_internal)

    # Counter number of parameters #
    #------------------------------#
    # Extract Best Model
    best_model = NEU_PCA_CV.best_estimator_
    # Count Number of Parameters
    N_params_best_NEU_PCA = np.sum([np.prod(v.get_shape().as_list()) for v in best_model.model.trainable_variables])
    print('NEU-PCA: Trained!')

    #-----------------#
    # Save Full-Model #
    #-----------------#
    print('NEU-PCA: Saving')
    #     joblib.dump(best_model, './outputs/models/Benchmarks/ffNN_trained_CV.pkl', compress = 1)
    NEU_PCA_CV.best_params_['N_Trainable_Parameters'] = N_params_best_NEU_PCA
    Path('./outputs/models/NEU/NEU_PCA/').mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_dict(NEU_PCA_CV.best_params_,orient='index').to_latex("./outputs/models/NEU/NEU_PCA/Best_Parameters.tex")
    print('NEU-PCA: Saved')

    # Get Factor(s) #
    # --------------#
#     # Extract Auto-Encoder Layer
#     encoder_layer = tf.keras.Model(inputs=best_model.model.inputs, outputs=best_model.model.layers[1].output)
#     # Get Feature(s)
#     # ## Train
#     NEU_PCA_Factors_train = np.array(encoder_layer.predict(X_train_scaled))
#     NEU_PCA_Factors_test = np.array(encoder_layer.predict(X_test_scaled))

    # Update User
    #-------------#
    print('Complete NEU-PCA Training Procedure!!!')

    # Return Values #
    # ---------------#
    return NEU_PCA_Reconstruction_train, NEU_PCA_Reconstruction_test, X_train_internal, X_test_internal


# ## Autoencoder

# In[22]:


def get_autoencoder(input_dim,
                    learning_rate,
                    PCA_Rank):
    
    # Initialization(s) #
    #-------------------#
    # Get encoder depth
    Encoder_depth = 5
    print('We use a DNN of depth: '+str(3+2*Encoder_depth))
    
    #--------------------------------------------------#
    # Build Regular Arch.
    #--------------------------------------------------#
    #-###################-#
    # Define Model Input -#
    #-###################-#
    input_layer = tf.keras.Input(shape=(input_dim,))


    #----------------------#
    # Core Layers: Encoder #
    #----------------------#
    encoder = fullyConnected_Dense(512)(input_layer)
    encoder = tf.nn.relu(encoder)
    encoder = fullyConnected_Dense(128)(encoder)
    encoder = tf.nn.relu(encoder)
    encoder = fullyConnected_Dense(PCA_Rank)(encoder)

    #----------------------#
    # Core Layers: Decoder #
    #----------------------#
    # PCA Readout (Really this is the OLS model)
    decoder = fullyConnected_Dense(PCA_Rank)(encoder)
    decoder = tf.nn.relu(decoder)
    decoder = fullyConnected_Dense(128)(decoder)
    decoder = tf.nn.relu(decoder)
    decoder = fullyConnected_Dense(512)(decoder)
    decoder = tf.nn.relu(decoder)
    # Readout Layer #
    #---------------#
    decoder = fullyConnected_Dense(input_dim)(decoder)
#     decoder = tf.nn.sigmoid(decoder)


    # Define Input/Output Relationship (Arch.)
    autoencoder = tf.keras.Model(input_layer, decoder)
    #--------------------------------------------------#
    # Define Optimizer & Compile Archs.
    #----------------------------------#
    opt = Adam(lr=learning_rate)
    autoencoder.compile(metrics=['accuracy'],loss='mean_squared_error',optimizer='Adam')
    
    # Give Auto-Encoder
    return autoencoder


# In[23]:


def build_autoencoder(n_folds, 
                      n_jobs, 
                      n_iter, 
                      X_train_scaled, 
                      X_train, 
                      X_test_scaled,
                      PCA_Rank):
    
    print('Begin autoencoder Training')
    # Update Dictionary
    Encoder_depth = 5
    param_grid_in_internal = Autoencoder_dictionary
    param_grid_in_internal['input_dim'] = [(X_train_scaled.shape[1])]
    param_grid_in_internal['PCA_Rank'] = [int(PCA_Rank)]

    # # Deep Feature Network
    AE_CV = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=get_autoencoder, verbose=True)

    # Randomized CV
    AE_CV = RandomizedSearchCV(estimator=AE_CV, 
                               n_jobs=n_jobs,
                               cv=KFold(n_folds, random_state=2020, shuffle=True),
                               param_distributions=param_grid_in_internal,
                               n_iter=n_iter,
                               return_train_score=True,
                               random_state=2020,
                               verbose=10)

    # Fit Model #
    #-----------#
    AE_CV.fit(X_train,X_train)

    # Write Predictions #
    # -------------------#
    AE_Reconstructed_train = AE_CV.predict(X_train_scaled)
    AE_Reconstructed_test = AE_CV.predict(X_test_scaled)

    # Counter number of parameters #
    #------------------------------#
    # Extract Best Model
    best_model = AE_CV.best_estimator_
    # Count Number of Parameters
    N_params_best_AE = np.sum([np.prod(v.get_shape().as_list()) for v in best_model.model.trainable_variables])
    print('Autoencoder: Trained!')

    #-----------------#
    # Save Full-Model #
    #-----------------#
    print('Autoencoder: Saving')
    #     joblib.dump(best_model, './outputs/models/Benchmarks/ffNN_trained_CV.pkl', compress = 1)
    AE_CV.best_params_['N_Trainable_Parameters'] = N_params_best_AE
    Path('./outputs/models/Autoencoder/').mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_dict(AE_CV.best_params_,orient='index').to_latex("./outputs/models/Autoencoder/Best_Parameters.tex")
    print('Autoencoder: Saved')

    # Update User
    #-------------#
    print('Complete Autoencoder Training Procedure!!!')

    # Get Factor(s) #
    # --------------#
    # Extract Auto-Encoder Layer
    encoder_layer = tf.keras.Model(inputs=best_model.model.inputs, outputs=best_model.model.layers[Encoder_depth].output)
    # Get Feature(s)
    # ## Train
    AE_Factors_train = np.array(encoder_layer.predict(X_train_scaled))
    AE_Factors_test = np.array(encoder_layer.predict(X_test_scaled))

    # Return Values #
    #---------------#
    return AE_Reconstructed_train, AE_Reconstructed_test, AE_Factors_train, AE_Factors_test


# ## NEU Autoencoder

# In[20]:


def get_NEU_Autoencoder(input_dim,
                        learning_rate,
                        PCA_Rank,
                        implicit_dimension,
                        feature_map_depth,
                        feature_map_height,
                        homotopy_parameter):
    #--------------------------------------------------#
    # Build Regular Arch.
    #--------------------------------------------------#
    #-###################-#
    # Define Model Input -#
    #-###################-#
    input_layer = tf.keras.Input(shape=(input_dim,))

    #----------------------#
    # Core Layers: PCA #
    #----------------------#
    # PCA
    encoder = fullyConnected_Dense(PCA_Rank)(input_layer)
    for i in range(feature_map_depth):
        # Reconstructor
        deep_feature_map  = Reconfiguration_unit(units=feature_map_height,
                                                 home_space_dim=PCA_Rank, 
                                                 homotopy_parameter = homotopy_parameter)(encoder)
        ## Constant part of reconfiguration unit
        deep_feature_map_out = rescaled_swish_trainable(homotopy_parameter = homotopy_parameter)(deep_feature_map)
    
    
    #-###############-#
    # NEU Feature Map #
    #-###############-#
    ##Random Embedding
    ### Compute Required Dimension
    embedding_dimension = 2*np.maximum(PCA_Rank,implicit_dimension)
    ### Execute Random Embedding
    deep_feature_map_prep = fullyConnected_Dense(embedding_dimension)(deep_feature_map_out)
    deep_feature_map = tf.concat([deep_feature_map_out, deep_feature_map_prep], axis=1)
    ## Homeomorphic Part
    dimension_lifted = (PCA_Rank + embedding_dimension)
    for i_feature_depth in range(feature_map_depth):
        # First Layer
        ## Spacial-Dependent part of reconfiguration unit
        deep_feature_map  = Reconfiguration_unit(units=feature_map_height,
                                                 home_space_dim=dimension_lifted, 
                                                 homotopy_parameter = homotopy_parameter)(deep_feature_map)
        ## Constant part of reconfiguration unit
        deep_feature_map = rescaled_swish_trainable(homotopy_parameter = homotopy_parameter)(deep_feature_map)

    # PCA Readout (Really this is the OLS model)
    decoder = fullyConnected_Dense(input_dim)(deep_feature_map)


    # Define Input/Output Relationship (Arch.)
    NEU_PCA = tf.keras.Model(input_layer, decoder)
    #--------------------------------------------------#
    # Define Optimizer & Compile Archs.
    #----------------------------------#
    opt = Adam(lr=learning_rate)
    NEU_PCA.compile(metrics=['accuracy'],loss='mean_squared_error',optimizer='Adam')

    # Return NEU PCA
    return NEU_PCA


# In[21]:


def build_NEU_Autoencoder(n_folds, 
                          n_jobs, 
                          n_iter, 
                          param_grid_in, 
                          X_train_scaled,
                          X_test_scaled, 
                          X_train,
                          PCA_Rank):
    # Update Dictionary
    param_grid_in_internal = param_grid_in
    param_grid_in_internal['input_dim'] = [(X_train.shape[1])]
    param_grid_in_internal['PCA_Rank'] = [PCA_Rank]

    print('Performing AE')
    # Deep Feature Network
    NEU_PCA_CV = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=get_NEU_Autoencoder, verbose=True)

    # Randomized CV
    NEU_PCA_CV = RandomizedSearchCV(estimator=NEU_PCA_CV, 
                                    n_jobs=n_jobs,
                                    cv=KFold(n_folds, random_state=2020,shuffle=True),
                                    param_distributions=param_grid_in_internal,
                                    n_iter=n_iter,
                                    return_train_score=True,
                                    random_state=2020,
                                    verbose=10)

    # Fit Model #
    #-----------#
    print('Training NEU-Feature Map!')
    NEU_PCA_CV.fit(X_train_scaled,X_train)

    # Write Predictions #
    #-------------------#
    NEU_PCA_Reconstruction_train = NEU_PCA_CV.predict(X_train_scaled)
    NEU_PCA_Reconstruction_test = NEU_PCA_CV.predict(X_test_scaled)

    # Counter number of parameters #
    #------------------------------#
    # Extract Best Model
    best_model = NEU_PCA_CV.best_estimator_
    # Count Number of Parameters
    N_params_best_NEU_PCA = np.sum([np.prod(v.get_shape().as_list()) for v in best_model.model.trainable_variables])
    print('NEU-AE: Trained!')

    #-----------------#
    # Save Full-Model #
    #-----------------#
    print('NEU-AE: Saving')
    #     joblib.dump(best_model, './outputs/models/Benchmarks/ffNN_trained_CV.pkl', compress = 1)
    NEU_PCA_CV.best_params_['N_Trainable_Parameters'] = N_params_best_NEU_PCA
    Path('./outputs/models/NEU/NEU_PCA/').mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_dict(NEU_PCA_CV.best_params_,orient='index').to_latex("./outputs/models/NEU/NEU_PCA/Best_Parameters.tex")
    print('NEU-AE: Saved')

    # Get Factor(s) #
    # --------------#
    # Extract Auto-Encoder Layer
    Encoder_Depth = 1 + 2*NEU_PCA_CV.best_params_['feature_map_depth']
    encoder_layer = tf.keras.Model(inputs=best_model.model.inputs, outputs=best_model.model.layers[Encoder_Depth].output)
    # Get Feature(s)
    # ## Train
    NEU_PCA_Factors_train = np.array(encoder_layer.predict(X_train_scaled))
    NEU_PCA_Factors_test = np.array(encoder_layer.predict(X_test_scaled))

    # Update User
    #-------------#
    print('Complete NEU-AE Training Procedure!!!')

    # Return Values #
    # ---------------#
    return NEU_PCA_Reconstruction_train, NEU_PCA_Reconstruction_test, NEU_PCA_Factors_train, NEU_PCA_Factors_test


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
# # Fin
# ---
