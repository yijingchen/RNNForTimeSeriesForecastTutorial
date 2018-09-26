
# coding: utf-8

# # Multi step model (vector output approach)
# 
# Download zipfile from https://www.dropbox.com/s/pqenrr2mcvl0hk9/GEFCom2014.zip?dl=0 and store in the data folder.
# 
# In this notebook, we demonstrate how to:
# - prepare time series data for training a RNN forecasting model
# - get data in the required shape for the keras API
# - implement a RNN model in keras to predict the next 24 steps ahead (time *t+1* to *t+24*) in the time series. This model uses recent values of temperature and load as the model input. The model will be trained to output a vector, the elements of which are ordered predictions for future time steps.
# - enable early stopping to reduce the likelihood of model overfitting
# - evaluate the model on a test dataset
# 
# The data in this example is taken from the GEFCom2014 forecasting competition<sup>1</sup>. It consists of 3 years of hourly electricity load and temperature values between 2012 and 2014. The task is to forecast future values of electricity load.
# 
# <sup>1</sup>Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli and Rob J. Hyndman, "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond", International Journal of Forecasting, vol.32, no.3, pp 896-913, July-September, 2016.

# In[1]:

import sys
import os
import numpy as np
import pandas as pd
import datetime as dt
from glob import glob
from collections import UserDict
import itertools

pd.options.display.float_format = '{:,.2f}'.format
np.set_printoptions(precision=2)


# In[2]:

get_ipython().magic('run -i common/utils.py')


# Load data into Pandas dataframe

# In[3]:

if not os.path.exists(os.path.join('data', 'energy.csv')):
    get_ipython().magic('run common/extract_data.py')
energy = load_data()
energy.head()


# In[4]:

valid_start_dt = '2014-09-01 00:00:00'
test_start_dt = '2014-11-01 00:00:00'

HORIZON = 24
N_EXPERIMENTS = 2


# In[5]:

train = energy.copy()[energy.index < valid_start_dt][['load', 'temp']]


# In[6]:

from sklearn.preprocessing import MinMaxScaler

y_scaler = MinMaxScaler()
y_scaler.fit(train[['load']])

X_scaler = MinMaxScaler()
train[['load', 'temp']] = X_scaler.fit_transform(train)


# ## Implement the RNN

# In[7]:

from keras.models import Model, Sequential
from keras.layers import GRU, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop
from keras import losses, regularizers


# fixed parameters

# In[8]:

EPOCHS = 50

# tunable parameters

# In[9]:

LATENT_DIMS = [5, 10, 15]
BATCH_SIZES = [8, 16, 32]
LEARNING_RATES = [0.01, 0.001, 0.0001]
ALPHA_VALS = [0.1, 0.001, 0]
ALL_PARAMS = [LATENT_DIMS, BATCH_SIZES, LEARNING_RATES, ALPHA_VALS]


# In[10]:

T_VALUES = [72, 168, 336]


# In[11]:

parameters = [list(enumerate(x)) for x in ALL_PARAMS]
grid = list(itertools.product(*parameters))
lengths = [len(T_VALUES)]
lengths.extend([len(x) for x in ALL_PARAMS])
mapes = np.empty(tuple(lengths))
st_errs = np.empty_like(mapes) 


# In[12]:

def create_input(T):
    
    tensor_structure = {'X':(range(-T+1, 1), ['load', 'temp'])}
    train_inputs = TimeSeriesTensor(train, 'load', HORIZON, tensor_structure)
    X_train = train_inputs.dataframe.as_matrix()[:,HORIZON:]
    Y_train = train_inputs['target']
    
    # Construct validation set (keeping T hours from the training set in order to construct initial features)
    look_back_dt = dt.datetime.strptime(valid_start_dt, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=T-1)
    valid = energy.copy()[(energy.index >=look_back_dt) & (energy.index < test_start_dt)][['load', 'temp']]
    valid[['load', 'temp']] = X_scaler.transform(valid)
    valid_inputs = TimeSeriesTensor(valid, 'load', HORIZON, tensor_structure)
    X_valid = valid_inputs.dataframe.as_matrix()[:,HORIZON:]
    Y_valid = valid_inputs['target']
    
    # Construct test set (keeping T hours from the validation set in order to construct initial features)
    look_back_dt = dt.datetime.strptime(test_start_dt, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=T-1)
    test = energy.copy()[test_start_dt:][['load', 'temp']]
    test[['load', 'temp']] = X_scaler.transform(test)
    test_inputs = TimeSeriesTensor(test, 'load', HORIZON, tensor_structure)
    X_test = test_inputs.dataframe.as_matrix()[:,HORIZON:]
    
    return X_train, Y_train, X_valid, Y_valid, X_test, test_inputs


# In[13]:

def get_model(LATENT_DIM, LEARNING_RATE, T, ALPHA):
    model = Sequential()
    model.add(Dense(LATENT_DIM, activation="relu", input_shape=(2*T,), kernel_regularizer=regularizers.l2(ALPHA), bias_regularizer=regularizers.l2(ALPHA)))
    model.add(Dense(HORIZON, kernel_regularizer=regularizers.l2(ALPHA), bias_regularizer=regularizers.l2(ALPHA)))
    optimizer = RMSprop(lr=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mse')
    
    return model


# In[14]:

for t, T_val in enumerate(T_VALUES):
    
    X_train, Y_train, X_valid, Y_valid, X_test, test_inputs = create_input(T_val)
      
    for (i,LATENT_DIM), (j,BATCH_SIZE), (k,LEARNING_RATE), (l, ALPHA) in grid:
    
        mapes_param = np.empty(N_EXPERIMENTS)
        for ii in range(N_EXPERIMENTS):
    
            # Initialize the model
            model = get_model(BATCH_SIZE, LEARNING_RATE, T_val, ALPHA)
            earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)
            best_val = ModelCheckpoint('model_{epoch:02d}.h5', save_best_only=True, mode='min', period=1)
    
            # Train the model
            history = model.fit(X_train, Y_train,
                                batch_size=BATCH_SIZE,
                                epochs=EPOCHS,
                                validation_data=(X_valid, Y_valid),
                                callbacks=[earlystop, best_val],
                                verbose=0)
    
            # load the model with the smallest MAPE
            best_epoch = np.argmin(np.array(history.history['val_loss']))+1
            model.load_weights("model_{:02d}.h5".format(best_epoch))
    
            predictions = model.predict(X_test)
    
            # Compute MAPE for each forecast horizon
            eval_df = create_evaluation_df(predictions, test_inputs, HORIZON, y_scaler)
            eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
    
            # Compute MAPE across all predictions
            mapes_param[ii] = mape(eval_df['prediction'], eval_df['actual'])
            print('{0:.4f}'.format(mapes_param[ii]))
    
            for f in glob('model_*.h5'):
                os.remove(f)
            
        mapes[t,i,j,k,l] = np.mean(mapes_param)
        st_errs[t,i,j,k,l] = np.std(mapes_param)/np.sqrt(N_EXPERIMENTS)
    
        params = 'T={0}, LATENT_DIM={1}, BATCH_SIZE={2}, LR={3}, ALPHA={4}'.format(T_val, LATENT_DIM, BATCH_SIZE, LEARNING_RATE, ALPHA)
        print(params)
        result = 'Mean MAPE = {0:.4f} +/- {1:.4f}'.format(np.mean(mapes_param), np.std(mapes_param)/np.sqrt(N_EXPERIMENTS))
        print(result)
        sys.stdout.flush()


# In[15]:

print(mapes.shape)


# In[16]:

print(mapes)


# In[18]:

print(T_val)


# In[ ]:



