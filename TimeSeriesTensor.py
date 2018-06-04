class TimeSeriesTensor(UserDict):
    
    # A dictionary of tensors for input into the RNN model
    
    # Use this class to:
    #   1. Shift the values of the time series to create a Pandas dataframe containing all the data
    #      for a single training example
    #   2. Discard any samples with missing values
    #   3. Transform this Pandas dataframe into a numpy array of shape 
    #      (samples, time steps, features) for input into Keras

    # The class takes the following parameters:
    #    - **dataset**: original time series
    #    - **H**: the forecast horizon
    #    - **tensor_structures**: a dictionary discribing the tensor structure with
    #      the form { 'tensor_name' : (range(max_backward_shift, max_forward_shift), [feature, feature, ...] ) }
    #    - **freq**: time series frequency
    #    - **drop_incomplete**: (Boolean) whether to drop incomplete samples
    
    def __init__(self, dataset, target, H, tensor_structure, freq='H', drop_incomplete=True):
        self.dataset = dataset
        self.target = target
        self.tensor_structure = tensor_structure
        self.tensor_names = list(tensor_structure.keys())
        
        self.shifted_df = self.shift_data(H, freq, drop_incomplete)
        self.data = self.df2tensors(self.shifted_df)
    
    
    def shift_data(self, H, freq, drop_incomplete):
        
        df = self.dataset.copy()
        
        idx_tuples = []
        for t in range(1, H+1):
            df['t+'+str(t)] = df[self.target].shift(t*-1, freq=freq)
            idx_tuples.append(('target', 'y', 't+'+str(t)))

        for name, structure in self.tensor_structure.items():
            rng = structure[0]
            dataset_cols = structure[1]
            
            for col in dataset_cols:
                    
                for t in rng:
                    sign = '+' if t > 0 else ''
                    shift = str(t) if t != 0 else ''
                    period = 't'+sign+shift
                    shifted_col = name+'_'+col+'_'+period
                    df[shifted_col] = df[col].shift(t*-1, freq=freq)
                    idx_tuples.append((name, col, period))
                
        df = df.drop(self.dataset.columns, axis=1)
        idx = pd.MultiIndex.from_tuples(idx_tuples, names=['tensor', 'feature', 'time step'])
        df.columns = idx

        if drop_incomplete:
            df = df.dropna(how='any')

        return df
    
    
    def df2tensors(self, shifted_df):
    
        inputs = {}
        y = shifted_df['target']
        y = y.as_matrix()
        inputs['target'] = y

        for name, structure in self.tensor_structure.items():
            rng = structure[0]
            cols = structure[1]
            tensor = shifted_df[name][cols].as_matrix()
            tensor = tensor.reshape(tensor.shape[0], len(cols), len(rng))
            tensor = np.transpose(tensor, axes=[0, 2, 1])
            inputs[name] = tensor

        return inputs
    
    def subset_data(self, new_shifted_df):
        
        self.shifted_df = new_shifted_df
        self.data = self.df2tensors(self.shifted_df)