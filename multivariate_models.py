from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import darts
from scipy.stats import norm
from tqdm import trange
from darts.models import RNNModel


class MultivariateVaR(ABC):
    '''Parent class'''
    def __init__(self, alpha=99, window_size=300, **kwargs):
        self.name = 'MultivariateVaR'
        self.alpha = alpha
        self.window_size = window_size
    
    @abstractmethod
    def fit(self, ts):
        raise NotImplementedError("fit method must be implemented in child class")
    
    @abstractmethod
    def predict_var_one_day(self, returns, weights):
        raise NotImplementedError("predict_var_one_day method must be implemented in child class")
    
    def predict_var_rolling_window(self, ts, weights):
        var_values = []
        for i in trange(self.window_size, len(ts)):
            current_date = ts.index[i].date()
            current_returns = ts.iloc[i - self.window_size:i]
            current_var = self.predict_var_one_day(current_returns, weights)
            var_values.append({'Date': current_date, 'VaR': current_var})
        var_values_df = pd.DataFrame(var_values)
        return var_values_df


class HistoricalSimulation(MultivariateVaR):
    def __init__(self, alpha=99, window_size=300):
        super().__init__(alpha=alpha, window_size=window_size)
        self.name = 'HistoricalSimulation'
    
    def fit(self, ts):
        pass
    
    def hs(self, returns, alpha):
        '''Historical Simulation VaR'''
        return np.percentile(returns, 100 - alpha)
    
    def predict_var_one_day(self, returns, weights):
        R = returns.corr()
        V = np.zeros(len(weights))
        for i in range(len(weights)):
            if weights[i] < 0:
                V[i] = weights[i] * self.hs(returns.iloc[:, i], alpha=self.alpha)
            else:
                V[i] = weights[i] * self.hs(returns.iloc[:, i], alpha=100 - self.alpha)
        return -np.sqrt(V @ R @ V.T)


class DeepVaR(MultivariateVaR):
    '''Class for fitting and predicting with Darts DeepAR model'''

    def __init__(self, context_length=15, alpha=99, window_size=300,
                 epochs=5, n_layers=2, dropout=0.1):
        super().__init__(alpha=alpha, window_size=window_size)
        self.name = 'DeepVaR'
        self.context_length = context_length
        self.epochs = epochs
        self.n_layers = n_layers
        self.dropout = dropout
        self.model = None

    def fit(self, ts):
        '''Expects a pandas DataFrame with datetime index and columns as asset returns.'''
        ts_darts = darts.TimeSeries.from_dataframe(ts, fill_missing_dates=True, freq='D')
        self.model = RNNModel(
            model='LSTM',
            input_chunk_length=self.context_length,
            output_chunk_length=1,
            n_rnn_layers=self.n_layers,
            dropout=self.dropout,
            n_epochs=self.epochs,
            batch_size=32,
            random_state=42
        )
        self.model.fit(ts_darts)

    def predict_ts(self, ts):
        '''Expects a pandas DataFrame with datetime index and columns as asset returns.'''
        ts_darts = darts.TimeSeries.from_dataframe(ts, fill_missing_dates=True, freq='D')
        return self.model.predict(n=1, series=ts_darts, num_samples=1000)

    def predict_var_one_day(self, returns, weights):
        V = np.zeros(len(weights))
        predictions = self.predict_ts(returns)
        
        for i in range(len(weights)):
            pred_samples = predictions[i].values()
            if weights[i] < 0:
                V[i] = weights[i] * np.percentile(pred_samples, self.alpha)
            else:
                V[i] = weights[i] * np.percentile(pred_samples, 100 - self.alpha)
        
        R = returns.corr()
        return -np.sqrt(V @ R @ V.T)