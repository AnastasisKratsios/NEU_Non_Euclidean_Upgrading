#!/usr/bin/env python
# coding: utf-8

# # Data Processor - Apple Stock Tracker

# In[ ]:


#------------------------#
# Run External Notebooks #
#------------------------#
if Option_Function == "SnP":
    #--------------#
    # Get S&P Data #
    #--------------#
    #=# SnP Constituents #=#
    # Load Data
    snp_data = pd.read_csv('inputs/data/snp500_data/snp500-adjusted-close.csv')
    # Format Data
    ## Index by Time
    snp_data['date'] = pd.to_datetime(snp_data['date'],infer_datetime_format=True)
    #-------------------------------------------------------------------------------#

    #=# SnP Index #=#
    ## Read Regression Target
    snp_index_target_data = pd.read_csv('inputs/data/snp500_data/GSPC.csv')
    ## Get (Reference) Dates
    dates_temp = pd.to_datetime(snp_data['date'],infer_datetime_format=True).tail(600)
    ## Format Target
    snp_index_target_data = pd.DataFrame({'SnP_Index': snp_index_target_data['Close'],'date':dates_temp.reset_index(drop=True)})
    snp_index_target_data['date'] = pd.to_datetime(snp_index_target_data['date'],infer_datetime_format=True)
    snp_index_target_data.set_index('date', drop=True, inplace=True)
    snp_index_target_data.index.names = [None]
    #-------------------------------------------------------------------------------#
    
    ## Get Rid of Rubbish
    snp_data.set_index('date', drop=True, inplace=True)
    snp_data.index.names = [None]
    ## Get Rid of NAs and Expired Trends
    snp_data = (snp_data.tail(600)).dropna(axis=1).fillna(0)
    
    # Apple
    snp_index_target_data = snp_data[{'AAPL'}]
    snp_data = snp_data[{'IBM','QCOM','MSFT','CSCO','ADI','MU','MCHP','NVR','NVDA','GOOGL','GOOG'}]
    # Get Return(s)
    snp_data_returns = snp_data.diff().iloc[1:]
    snp_index_target_data_returns = snp_index_target_data.diff().iloc[1:]
    #--------------------------------------------------------#
    
    #-------------#
    # Subset Data #
    #-------------#
    # Get indices
    N_train_step = int(round(snp_index_target_data_returns.shape[0]*Train_step_proportion,0))
    N_test_set = int(snp_index_target_data_returns.shape[0] - round(snp_index_target_data_returns.shape[0]*Train_step_proportion,0))
    # # Get Datasets
    X_train = snp_data_returns[:N_train_step]
    X_test = snp_data_returns[-N_test_set:]
    ## Coerce into format used in benchmark model(s)
    data_x = X_train
    data_x_test = X_test
    # Get Targets 
    data_y = snp_index_target_data_returns[:N_train_step]
    data_y_test = snp_index_target_data_returns[-N_test_set:]
    
    # Scale Data
    scaler = StandardScaler()
    data_x = scaler.fit_transform(data_x)
    data_x_test = scaler.transform(data_x_test)

    # # Update User
    print('#================================================#')
    print(' Training Datasize: '+str(X_train.shape[0])+' and test datasize: ' + str(X_test.shape[0]) + '.  ')
    print('#================================================#')

    # # Set First Run to Off
    First_run = False

    #-----------#
    # Plot Data #
    #-----------#
    fig = snp_data_returns.plot(figsize=(16, 16))
    fig.get_legend().remove()
    plt.title("S&P Data Returns")

    # SAVE Figure to .eps
    plt.savefig('./outputs/plotsANDfigures/SNP_Data_returns.pdf', format='pdf')
else:
    # Simulate Data using the data-generator:
    get_ipython().run_line_magic('run', 'Data_Generator.ipynb')
    exec(open('Data_Generator.py').read())

