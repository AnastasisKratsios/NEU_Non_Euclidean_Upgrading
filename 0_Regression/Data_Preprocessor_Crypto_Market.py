#!/usr/bin/env python
# coding: utf-8

# # Data Processor - CryptoMarket

# In[ ]:


#------------------------#
# Run External Notebooks #
#------------------------#
if Option_Function == "crypto":
    #--------------#
    # Prepare Data #
    #--------------#
    # Read Dataset
    crypto_data = pd.read_csv('inputs/data/cryptocurrencies/Cryptos_All_in_one.csv')
    # Format Date-Time
    crypto_data['Date'] = pd.to_datetime(crypto_data['Date'],infer_datetime_format=True)
    crypto_data.set_index('Date', drop=True, inplace=True)
    crypto_data.index.names = [None]

    # Remove Missing Data
    crypto_data = crypto_data[crypto_data.isna().any(axis=1)==False]

    # Get Returns
    crypto_returns = crypto_data.diff().iloc[1:]

    # Parse Regressors from Targets
    ## Get Regression Targets
    crypto_target_data = pd.DataFrame({'BITCOIN-closing':crypto_returns['BITCOIN-Close']})
    ## Get Regressors
    crypto_data_returns = crypto_returns.drop('BITCOIN-Close', axis=1)  

    #-------------#
    # Subset Data #
    #-------------#
    # Get indices
    N_train_step = int(round(crypto_data_returns.shape[0]*Train_step_proportion,0))
    N_test_set = int(crypto_data_returns.shape[0] - round(crypto_data_returns.shape[0]*Train_step_proportion,0))
    # # Get Datasets
    X_train = crypto_data_returns[:N_train_step]
    X_test = crypto_data_returns[-N_test_set:]

    ## Coerce into format used in benchmark model(s)
    data_x = X_train
    data_x_test = X_test
    # Get Targets 
    data_y = crypto_target_data[:N_train_step]
    data_y_test = crypto_target_data[-N_test_set:]

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
    fig = crypto_data_returns.plot(figsize=(16, 16))
    fig.get_legend().remove()
    plt.title("Crypto_Market Returns")

    # SAVE Figure to .eps
    plt.savefig('./outputs/plotsANDfigures/Crypto_Data_returns.pdf', format='pdf')
    
    
    # Set option to SnP to port rest of pre-processing automatically that way
    Option_Function = "SnP"
else:
    # Simulate Data using the data-generator:
    get_ipython().run_line_magic('run', 'Data_Generator.ipynb')
    exec(open('Data_Generator.py').read())

