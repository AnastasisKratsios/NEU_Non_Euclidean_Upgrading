#!/usr/bin/env python
# coding: utf-8

# ---
# # Benchmark Regression Models (Univariate)
# ---
# This little script implements multiple semi-classical regression model, functionally.  The functions are then ported into the NEU; thus we obtain benchmarks and a simple way to generate the NEU-version of any such model.  

# ## Ordinary Linear Regression

# In[ ]:


# Block warnings that spam when performing coordinate descent (by default) in 1-d.
import warnings
warnings.filterwarnings("ignore")

#====================================#
# Ordinary Linear Regression Version #
#====================================#
# # Initialize OLS Model
# lin_reg = LinearRegression()
# lin_reg.fit(data_x,data_y)
# # Generate OLS Predictions
# OLS_y_hat_train = lin_reg.predict(data_x)
# OLS_y_hat_test = lin_reg.predict(data_x_test)

#=====================#
# Elastic Net Version #
#=====================#
# Initialize Elastic Net Regularization Model
if trial_run == True:
    Elastic_Net = ElasticNetCV(cv=5, random_state=0, alphas = np.linspace(0,(10**2),(2)),
                           l1_ratio=np.linspace(0,1,(2)))
else:
    Elastic_Net = ElasticNetCV(cv=5, random_state=0, alphas = np.linspace(0,(10**2),(10**2)),
                               l1_ratio=np.linspace(0,1,(10**2)))

# Fit Elastic Net Model
Elastic_Net.fit(data_x,data_y)
# Get Prediction(s)
ENET_OLS_y_hat_train = Elastic_Net.predict(data_x)
ENET_OLS_y_hat_test = Elastic_Net.predict(data_x_test)


# ## LOWESS

# Training Function

# In[ ]:


def get_LOESS(data_x,data_x_test,data_y):
    # Initializations #
    #-----------------#
    from scipy.interpolate import interp1d
    import statsmodels.api as sm
    
    # Coerce Data #
    #=============#
    # Training Data
    data_y_vec = np.array(data_y)
    data_x_vec = np.array(data_x).reshape(-1,)
    # Testing Data
    data_x_test_vec = np.array(data_x_test).reshape(-1,)
    
    # Train LOESS #
    #=============#
    LOESS = sm.nonparametric.lowess
    f_hat_LOESS = LOESS(data_y_vec,data_x_vec.reshape(-1,))
    LOESS_x = list(zip(*f_hat_LOESS))[0]
    f_hat_LOESS = list(zip(*f_hat_LOESS))[1]
    
    # Get LOESS Prediction(s) #
    #-------------------------#
    # Train
    f = interp1d(LOESS_x, f_hat_LOESS, bounds_error=False)
    LOESS_prediction_train = f(data_x_vec)
    LOESS_prediction_train = np.nan_to_num(LOESS_prediction_train)
    # Test
    LOESS_prediction_test = f(data_x_test_vec)
    LOESS_prediction_test = np.nan_to_num(LOESS_prediction_test)
    
    # Return LOESS Outputs
    return LOESS_prediction_train, LOESS_prediction_test


# Get predictions from training function

# In[ ]:


if Option_Function != "SnP":
    LOESS_prediction_train, LOESS_prediction_test = get_LOESS(data_x,data_x_test,data_y)


# ## Smoothing Splines

# Training Function

# In[ ]:


def get_smooting_splines(data_x,data_x_test,data_y):
    # Imports #
    #---------#
    import rpy2.robjects as robjects # Work directly from R (since smoothing splines packages is better)

    # Coercion #
    #----------#
    # Training Data
    data_y_vec = np.array(data_y)
    data_x_vec = np.array(data_x).reshape(-1,)
    # Testing Data
    data_x_test_vec = np.array(data_x_test).reshape(-1,)
    r_y = robjects.FloatVector(data_y_vec)
    r_x = robjects.FloatVector(data_x_vec)

    # Training #
    #----------#
    r_smooth_spline = robjects.r['smooth.spline'] #extract R function# run smoothing function
    spline1 = r_smooth_spline(x=r_x, y=r_y, spar=0.7)
    f_hat_smoothing_splines=np.array(robjects.r['predict'](spline1,robjects.FloatVector(data_x_vec)).rx2('y'))

    # Prediction #
    #------------#
    # Train
    f_hat_smoothing_splines_train=np.array(robjects.r['predict'](spline1,robjects.FloatVector(data_x_vec)).rx2('y'))
    # Test
    f_hat_smoothing_splines_test=np.array(robjects.r['predict'](spline1,robjects.FloatVector(data_x_test_vec)).rx2('y'))

    # Return Outputs
    return f_hat_smoothing_splines_train, f_hat_smoothing_splines_test


#  Get Predictions from training function

# In[ ]:


if Option_Function != "SnP":
    f_hat_smoothing_splines_train, f_hat_smoothing_splines_test = get_smooting_splines(data_x,data_x_test,data_y)


# # Kernel Regression

# Get Kernel Ridge Regressor Model.  

# In[ ]:


def get_Kernel_Ridge_Regressor(data_x_in,data_x_test_in,data_y_in):
    # Imports
    from sklearn.svm import SVR
    from sklearn.kernel_ridge import KernelRidge

    # Initialize Randomized Gridsearch
    kernel_ridge_CVer = RandomizedSearchCV(estimator = KernelRidge(),
                                           n_jobs=n_jobs,
                                           cv=KFold(CV_folds, random_state=2020, shuffle=True),
                                           param_distributions=param_grid_kernel_Ridge,
                                           n_iter=n_iter,
                                           return_train_score=True,
                                           random_state=2020,
                                           verbose=10)
    kernel_ridge_CVer.fit(data_x_in,data_y_in)

    # Get best Kernel ridge regressor
    best_kernel_ridge_model = kernel_ridge_CVer.best_estimator_

    # Get Predictions
    f_hat_kernel_ridge_train = best_kernel_ridge_model.predict(data_x_in)
    f_hat_kernel_ridge_test = best_kernel_ridge_model.predict(data_x_test_in)


    Path('./outputs/models/Kernel_Ridge/').mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_dict(kernel_ridge_CVer.best_params_,orient='index').to_latex("./outputs/models/Kernel_Ridge/Best_Parameters.tex")
    
    # Return
    return f_hat_kernel_ridge_train, f_hat_kernel_ridge_test, best_kernel_ridge_model


# Get Kernel ridge regressor predictions.  

# In[ ]:


f_hat_kernel_ridge_train, f_hat_kernel_ridge_test, best_kernel_ridge_model = get_Kernel_Ridge_Regressor(data_x,data_x_test,data_y)


# # Gradient-Boosted Regression Trees

# In[1]:


def get_GBRF(X_train,X_test,y_train):

    # Run Random Forest Util
    rand_forest_model_grad_boosted = GradientBoostingRegressor()

    # Grid-Search CV
    Random_Forest_GridSearch = RandomizedSearchCV(estimator = rand_forest_model_grad_boosted,
                                                  n_iter=n_iter_trees,
                                                  cv=KFold(CV_folds, random_state=2020, shuffle=True),
                                                  param_distributions=Rand_Forest_Grid,
                                                  return_train_score=True,
                                                  random_state=2020,
                                                  verbose=10,
                                                  n_jobs=n_jobs)

    random_forest_trained = Random_Forest_GridSearch.fit(X_train,y_train)
    random_forest_trained = random_forest_trained.best_estimator_

    #--------------------------------------------------#
    # Write: Model, Results, and Best Hyper-Parameters #
    #--------------------------------------------------#

    # Save Model
    # pickle.dump(random_forest_trained, open('./outputs/models/Gradient_Boosted_Tree/Gradient_Boosted_Tree_Best.pkl','wb'))

    # Save Readings
    cur_path = os.path.expanduser('./outputs/tables/best_params_Gradient_Boosted_Tree.txt')
    with open(cur_path, "w") as f:
        f.write(str(Random_Forest_GridSearch.best_params_))

    best_params_table_tree = pd.DataFrame({'N Estimators': [Random_Forest_GridSearch.best_params_['n_estimators']],
                                        'Min Samples Leaf': [Random_Forest_GridSearch.best_params_['min_samples_leaf']],
                                        'Learning Rate': [Random_Forest_GridSearch.best_params_['learning_rate']],
                                        'Max Depth': [Random_Forest_GridSearch.best_params_['max_depth']],
                                        })
    
    # Count Number of Parameters in Random Forest Regressor
    N_tot_params_per_tree = [ (x[0].tree_.node_count)*random_forest_trained.n_features_ for x in random_forest_trained.estimators_]
    N_tot_params_in_forest = sum(N_tot_params_per_tree)
    best_params_table_tree['N_parameters'] = [N_tot_params_in_forest]
    # Write Best Parameter(s)
    best_params_table_tree.to_latex('./outputs/tables/Best_params_table_Gradient_Boosted_Tree.txt')
    #---------------------------------------------#
    
    # Generate Prediction(s) #
    #------------------------#
    y_train_hat_random_forest_Gradient_boosting = random_forest_trained.predict(X_train)
    y_test_hat_random_forest_Gradient_boosting = random_forest_trained.predict(X_test)
    
    # Return Predictions #
    #--------------------#
    return y_train_hat_random_forest_Gradient_boosting, y_test_hat_random_forest_Gradient_boosting, random_forest_trained


# In[ ]:


GBRF_y_hat_train, GBRF_y_hat_test, GBRF_model = get_GBRF(data_x,data_x_test,data_y)


# # Feed-Forward Neural Network

# In[ ]:


ffNN_y_hat_train,ffNN_y_hat_test = build_ffNN(n_folds = CV_folds, 
                                             n_jobs = n_jobs, 
                                             n_iter = n_iter, 
                                             param_grid_in = param_grid_Vanilla_Nets, 
                                             X_train = data_x, 
                                             y_train = data_y,
                                             X_test = data_x_test)

