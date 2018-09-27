
import os
import warnings
import numpy as np
import pandas as pd
import datetime as dt
from glob import glob

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--T", type=int)
parser.add_argument("--ENCODER_DIM_1", type=int)
parser.add_argument("--DECODER_DIM_1", type=int)
parser.add_argument("-ENCODER_DIM_2", type=int)
parser.add_argument("-DECODER_DIM_2", type=int)
args = parser.parse_args()

from common.utils import load_data, mape, TimeSeriesTensor, create_evaluation_df

pd.options.display.float_format = '{:,.2f}'.format
np.set_printoptions(precision=2)

energy = load_data(args.data_dir)
energy.head()

valid_start_dt = '2014-09-01 00:00:00'
test_start_dt = '2014-11-01 00:00:00'

T = args.T
HORIZON = 24

print("T: ", T)
print("HORIZON: ", HORIZON)

train = energy.copy()[energy.index < valid_start_dt][['load']]

from sklearn.preprocessing import MinMaxScaler

y_scaler = MinMaxScaler()
y_scaler.fit(train[['load']])

X_scaler = MinMaxScaler()
train[['load']] = X_scaler.fit_transform(train)

tensor_structure = {'encoder_input':(range(-T+1, 1), ['load']), 'decoder_input':(range(0, HORIZON), ['load'])}
train_inputs = TimeSeriesTensor(train, 'load', HORIZON, tensor_structure)
train_inputs.dataframe.head()

look_back_dt = dt.datetime.strptime(valid_start_dt, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=T-1)
valid = energy.copy()[(energy.index >=look_back_dt) & (energy.index < test_start_dt)][['load']]
valid[['load']] = X_scaler.transform(valid)
valid_inputs = TimeSeriesTensor(valid, 'load', HORIZON, tensor_structure)

look_back_dt = dt.datetime.strptime(test_start_dt, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=T-1)
test = energy.copy()[test_start_dt:][['load']]
test[['load']] = X_scaler.transform(test)
test_inputs = TimeSeriesTensor(test, 'load', HORIZON, tensor_structure)

from keras.models import Model, Sequential
from keras.layers import GRU, Dense, RepeatVector, TimeDistributed, Flatten, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint


BATCH_SIZE = 32
EPOCHS = 1
ENCODER_DIM_1 = args.ENCODER_DIM_1
ENCODER_DIM_2 = args.ENCODER_DIM_2
DECODER_DIM_1 = args.DECODER_DIM_1
DECODER_DIM_2 = args.DECODER_DIM_2

# Training model

encoder_input = Input(shape=(None, 1))

if ENCODER_DIM_2:
    encoder = GRU(ENCODER_DIM_1, return_sequences=True)(encoder_input)
    encoder, state_h = GRU(ENCODER_DIM_2, return_state=True)(encoder)
else:
    encoder, state_h = GRU(ENCODER_DIM_1, return_state=True)(encoder_input)
encoder_states = [state_h]

decoder_input = Input(shape=(None, 1))

if DECODER_DIM_2:
    decoder = GRU(DECODER_DIM_1, return_sequences=True)(decoder_input, initial_state=encoder_states)
    decoder, _ = GRU(DECODER_DIM_2, return_sequences=True, return_state=True)(decoder)
else:
    decoder, _ = GRU(DECODER_DIM_1, return_sequences=True, return_state=True)(decoder_input, initial_state=encoder_states)

decoder_dense = TimeDistributed(Dense(1))
decoder_output = decoder_dense(decoder)

model = Model([encoder_input, decoder_input], decoder_output)
model.compile(optimizer='RMSprop', loss='mse')

model.summary()
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)
best_val = ModelCheckpoint('model_{epoch:02d}.h5', save_best_only=True, mode='min', period=1)


train_target = train_inputs['target'].reshape(train_inputs['target'].shape[0], train_inputs['target'].shape[1], 1)
valid_target = valid_inputs['target'].reshape(valid_inputs['target'].shape[0], valid_inputs['target'].shape[1], 1)

history = model.fit([train_inputs['encoder_input'], train_inputs['decoder_input']],
                    train_target,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=([valid_inputs['encoder_input'], valid_inputs['decoder_input']], valid_target),
                    callbacks=[earlystop, best_val],
                    verbose=1)


# Inference model

# build inference encoder model
encoder_model = Model(encoder_input, encoder_states)

# build inference decoder model
decoder_state_input_h = Input(shape=(DECODER_DIM_1,))
decoder_states_input = [decoder_state_input_h]

decoder_output, state_h = decoder(decoder_input, initial_state=decoder_states_input)
decoder_states = [state_h]
decoder_output = decoder_dense(decoder_output)
decoder_model = Model([decoder_input] + decoder_states_input, [decoder_output] + decoder_states)


# Define the funtion to make single sequence prediction 
# based on scoring encoder-decoder
def predict_single_sequence(single_input_seq, horizon, n_features):
    # apply encoder model to the input_seq to get state
    states_value = encoder_model.predict(single_input_seq)
    
    # get input for decoder's first time step (which is encoder input at time t)
    dec_input = np.zeros((1, 1, n_features))
    dec_input[0, 0, 0] = single_input_seq[0, -1, :]
    
    # create final output placeholder
    output = list()
    # collect predictions
    for t in range(horizon):
        # predict next value
        yhat, h = decoder_model.predict([dec_input] + [states_value])
        # store prediction
        output.append(yhat[0,0,:])
        # update state
        state = [h]
        # update decoder input to be used as input for next prediction
        dec_input[0, 0, 0] = yhat
        
    return np.array(output)


# Define the funtion to make multiple sequence prediction 
# based on scoring encoder-decoder
def predict_multi_sequence(input_seq_multi, horizon, n_features):
    # create output placeholder
    predictions_all = list()
    for seq_index in range(input_seq_multi.shape[0]):       
        # Take one sequence for decoding
        input_seq = input_seq_multi[seq_index: seq_index + 1]
        # Generate prediction for the single sequence
        predictions = predict_single_sequence(input_seq, horizon, n_features)
        # store all the sequence prediction
        predictions_all.append(predictions)
        
    return np.array(predictions_all)


# Evaluate the model

look_back_dt = dt.datetime.strptime(test_start_dt, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=T-1)
test = energy.copy()[test_start_dt:][['load']]
test[['load']] = y_scaler.transform(test)
test_inputs = TimeSeriesTensor(test, 'load', HORIZON, tensor_structure)

# example of multiple sequence prediction based on validation data
test_predictions_all = predict_multi_sequence(test_inputs['encoder_input'], HORIZON, 1)

test_predictions_all_eval = test_predictions_all.reshape(test_predictions_all.shape[0], test_predictions_all.shape[1])

eval_df = create_evaluation_df(test_predictions_all_eval, test_inputs, HORIZON, y_scaler)
eval_df.head()

eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']

# Compute MAPE across all predictions
MAPE = '{0:.4f}'.format(mape(eval_df['prediction'], eval_df['actual']))
print(MAPE)

for f in glob('model_*.h5'):
    os.remove(f)

with open(os.path.join(args.output_dir, "Output.txt"), "w") as file:
    file.write(MAPE)