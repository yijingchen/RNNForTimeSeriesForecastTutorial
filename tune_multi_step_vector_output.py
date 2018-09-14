
import os
import numpy as np
import pandas as pd
import datetime as dt
from glob import glob

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument("--T", type=int)
parser.add_argument("--LATENT_DIM_1", type=int)
parser.add_argument("-LATENT_DIM_2", type=int)
args = parser.parse_args()

pd.options.display.float_format = '{:,.2f}'.format
np.set_printoptions(precision=2)

from common.utils import load_data, mape, TimeSeriesTensor, create_evaluation_df

# Load data into Pandas dataframe

energy = load_data(args.data_dir)
energy.head()

valid_start_dt = '2014-09-01 00:00:00'
test_start_dt = '2014-11-01 00:00:00'

T = args.T
HORIZON = 24

print("T: ", T)
print("HORIZON: ", HORIZON)

train = energy.copy()[energy.index < valid_start_dt][['load', 'temp']]

from sklearn.preprocessing import MinMaxScaler

y_scaler = MinMaxScaler()
y_scaler.fit(train[['load']])

X_scaler = MinMaxScaler()
train[['load', 'temp']] = X_scaler.fit_transform(train)


# Use the TimeSeriesTensor convenience class to:
# 1. Shift the values of the time series to create a Pandas dataframe containing all the data for a single training example
# 2. Discard any samples with missing values
# 3. Transform this Pandas dataframe into a numpy array of shape (samples, time steps, features) for input into Keras
# 
# The class takes the following parameters:
# 
# - **dataset**: original time series
# - **H**: the forecast horizon
# - **tensor_structure**: a dictionary discribing the tensor structure in the form { 'tensor_name' : (range(max_backward_shift, max_forward_shift), [feature, feature, ...] ) }
# - **freq**: time series frequency
# - **drop_incomplete**: (Boolean) whether to drop incomplete samples

tensor_structure = {'X':(range(-T+1, 1), ['load', 'temp'])}
train_inputs = TimeSeriesTensor(train, 'load', HORIZON, tensor_structure)


# Construct validation set (keeping W hours from the training set in order to construct initial features)

look_back_dt = dt.datetime.strptime(valid_start_dt, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=T-1)
valid = energy.copy()[(energy.index >=look_back_dt) & (energy.index < test_start_dt)][['load', 'temp']]
valid[['load', 'temp']] = X_scaler.transform(valid)
valid_inputs = TimeSeriesTensor(valid, 'load', HORIZON, tensor_structure)

look_back_dt = dt.datetime.strptime(test_start_dt, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=T-1)
test = energy.copy()[test_start_dt:][['load', 'temp']]
test[['load', 'temp']] = X_scaler.transform(test)
test_inputs = TimeSeriesTensor(test, 'load', HORIZON, tensor_structure)


# ## Implement the RNN

# We will implement a RNN forecasting model with the following structure:

from keras.models import Model, Sequential
from keras.layers import GRU, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint


def get_model(LATENT_DIM_1, LATENT_DIM_2=0):
    model = Sequential()
    
    if LATENT_DIM_2:
        model.add(GRU(LATENT_DIM_1, input_shape=(T, 2), return_sequences=True))
        model.add(GRU(LATENT_DIM_2))
    else:
        model.add(GRU(LATENT_DIM_1, input_shape=(T, 2)))

    model.add(Dense(HORIZON))
    model.compile(optimizer='RMSprop', loss='mse')
    
    return model

N_EXPERIMENTS = 1
BATCH_SIZE = 32
EPOCHS = 500
LATENT_DIM_1 = args.LATENT_DIM_1
LATENT_DIM_2 = args.LATENT_DIM_2

print("LATENT_DIM_1: ", LATENT_DIM_1)
print("LATENT_DIM_2: ", LATENT_DIM_2)


mapes = np.empty(N_EXPERIMENTS)
for i in range(N_EXPERIMENTS):
    
    # Initialize the model
    model = get_model(LATENT_DIM_1, LATENT_DIM_2)
    model.summary()
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)
    best_val = ModelCheckpoint('model_{epoch:02d}.h5', save_best_only=True, mode='min', period=1)
    
    # Train the model
    history = model.fit(train_inputs['X'],
                        train_inputs['target'],
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(valid_inputs['X'], valid_inputs['target']),
                        callbacks=[earlystop, best_val],
                        verbose=1)
    
    # load the model with the smallest MAPE
    best_epoch = np.argmin(np.array(history.history['val_loss']))+1
    model.load_weights("model_{:02d}.h5".format(best_epoch))
    
    predictions = model.predict(test_inputs['X'])
    
    # Compute MAPE for each forecast horizon
    eval_df = create_evaluation_df(predictions, test_inputs, HORIZON, y_scaler)
    eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
    
    # Compute MAPE across all predictions
    mapes[i] = mape(eval_df['prediction'], eval_df['actual'])
    print('{0:.4f}'.format(mapes[i]))
    
    for f in glob('model_*.h5'):
        os.remove(f)

result = 'Mean MAPE = {0:.4f} +/- {1:.4f}'.format(np.mean(mapes), np.std(mapes)/np.sqrt(N_EXPERIMENTS))
print(result)