
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

train = energy.copy()[energy.index < valid_start_dt][['load', 'temp']]

from sklearn.preprocessing import MinMaxScaler

y_scaler = MinMaxScaler()
y_scaler.fit(train[['load']])

X_scaler = MinMaxScaler()
train[['load', 'temp']] = X_scaler.fit_transform(train)

tensor_structure = {'X':(range(-T+1, 1), ['load', 'temp'])}
train_inputs = TimeSeriesTensor(train, 'load', HORIZON, {'X':(range(-T+1, 1), ['load', 'temp'])})

look_back_dt = dt.datetime.strptime(valid_start_dt, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=T-1)
valid = energy.copy()[(energy.index >=look_back_dt) & (energy.index < test_start_dt)][['load', 'temp']]
valid[['load', 'temp']] = X_scaler.transform(valid)
valid_inputs = TimeSeriesTensor(valid, 'load', HORIZON, tensor_structure)

look_back_dt = dt.datetime.strptime(test_start_dt, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=T-1)
test = energy.copy()[test_start_dt:][['load', 'temp']]
test[['load', 'temp']] = X_scaler.transform(test)
test_inputs = TimeSeriesTensor(test, 'load', HORIZON, tensor_structure)

from keras.models import Model, Sequential
from keras.layers import GRU, Dense, RepeatVector, TimeDistributed, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint


def get_model(ENCODER_DIM_1, DECODER_DIM_1, ENCODER_DIM_2=0, DECODER_DIM_2=0):
    model = Sequential()

    if ENCODER_DIM_2:
        model.add(GRU(ENCODER_DIM_1, input_shape=(T, 2), return_sequences=True))
        model.add(GRU(ENCODER_DIM_2))
    else:
        model.add(GRU(ENCODER_DIM_1, input_shape=(T, 2)))
    
    model.add(RepeatVector(HORIZON))

    model.add(GRU(DECODER_DIM_1, return_sequences=True))
    if DECODER_DIM_2:
        model.add(GRU(DECODER_DIM_2, return_sequences=True))
    
    model.add(TimeDistributed(Dense(1)))
    model.add(Flatten())
    model.compile(optimizer='RMSprop', loss='mse')

    return model


BATCH_SIZE = 32
EPOCHS = 100
ENCODER_DIM_1 = args.ENCODER_DIM_1
ENCODER_DIM_2 = args.ENCODER_DIM_2
DECODER_DIM_1 = args.DECODER_DIM_1
DECODER_DIM_2 = args.DECODER_DIM_2

# Initialize the model
model = get_model(ENCODER_DIM_1, DECODER_DIM_1, ENCODER_DIM_2, DECODER_DIM_2)
model.summary()
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)
#best_val = ModelCheckpoint(os.path.join(args.output_dir, 'model_{epoch:02d}.h5'), save_best_only=True, mode='min', period=1)

# Train the model
history = model.fit(train_inputs['X'],
                    train_inputs['target'],
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(valid_inputs['X'], valid_inputs['target']),
                    callbacks=[earlystop], # best_val
                    verbose=1)


# load the model with the smallest MAPE
#best_epoch = np.argmin(np.array(history.history['val_loss']))+1
#model.load_weights(os.path.join(args.output_dir, "model_{:02d}.h5".format(best_epoch)))

predictions = model.predict(test_inputs['X'])

# Compute MAPE for each forecast horizon
eval_df = create_evaluation_df(predictions, test_inputs, HORIZON, y_scaler)
eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']

# Compute MAPE across all predictions
MAPE = '{0:.4f}'.format(mape(eval_df['prediction'], eval_df['actual']))
print(MAPE)

#for f in glob('model_*.h5'):
#    os.remove(f)

with open(os.path.join(args.output_dir, "Output.txt"), "w") as file:
    file.write(MAPE)