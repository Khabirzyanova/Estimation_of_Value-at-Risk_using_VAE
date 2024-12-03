import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize

class HistoricalSimulation:
    """
    Historical Simulation (HS) Value at Risk Model.
    """
    def __init__(self, confidence_level=0.99, window_size=300):
        self.confidence_level = confidence_level
        self.window_size = window_size

    def calculate_var(self, historical_returns):
        alpha = 1 - self.confidence_level
        return np.percentile(historical_returns, alpha * 100)

    def rolling_var(self, returns):
        var_values = []
        for i in range(self.window_size, len(returns)):
            window = returns.iloc[i - self.window_size:i]
            var = self.calculate_var(window)
            var_values.append({'Date': returns.index[i], 'VaR': var})
        return pd.DataFrame(var_values).set_index('Date')


class DCCGARCH:
    """
    Dynamic Conditional Correlation (DCC) GARCH Model.
    """
    def __init__(self, window_size=300, confidence_level=0.99):
        self.window_size = window_size
        self.confidence_level = confidence_level

    def fit_garch(self, returns):
        model = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
        fitted = model.fit(disp='off')
        return fitted.conditional_volatility

    def fit_dcc(self, returns):
        conditional_volatilities = {}
        for column in returns.columns:
            conditional_volatilities[column] = self.fit_garch(returns[column])

        standardized_returns = returns.copy()
        for column in returns.columns:
            standardized_returns[column] = (
                returns[column] / conditional_volatilities[column]
            )

        def dcc_likelihood(params):
            a, b = params
            T, N = standardized_returns.shape
            Q = np.cov(standardized_returns.T)
            Q_bar = Q.copy()
            likelihood = 0
            for t in range(T):
                Z_t = standardized_returns.iloc[t].values.reshape(-1, 1)
                Q = (1 - a - b) * Q_bar + a * np.outer(Z_t, Z_t.T) + b * Q
                R = np.diag(1 / np.sqrt(np.diag(Q))) @ Q @ np.diag(1 / np.sqrt(np.diag(Q)))
                likelihood += np.log(np.linalg.det(R)) + Z_t.T @ np.linalg.inv(R) @ Z_t
            return likelihood[0, 0]

        initial_params = [0.01, 0.97]
        bounds = [(1e-6, 1), (1e-6, 1)]
        result = minimize(dcc_likelihood, initial_params, bounds=bounds)
        a, b = result.x

        T, N = standardized_returns.shape
        Q = np.cov(standardized_returns.T)
        Q_bar = Q.copy()
        dcc_matrices = []
        for t in range(T):
            Z_t = standardized_returns.iloc[t].values.reshape(-1, 1)
            Q = (1 - a - b) * Q_bar + a * np.outer(Z_t, Z_t.T) + b * Q
            R = np.diag(1 / np.sqrt(np.diag(Q))) @ Q @ np.diag(1 / np.sqrt(np.diag(Q)))
            dcc_matrices.append(R)
        return np.array(dcc_matrices)

    def predict_var(self, returns, weights):
        dcc_matrices = self.fit_dcc(returns)
        conditional_volatilities = np.array(
            [self.fit_garch(returns[col]) for col in returns.columns]
        )
        portfolio_volatility = np.sqrt(
            np.sum(weights[:, None] * conditional_volatilities**2, axis=0)
            + 2 * np.sum(np.triu(weights[:, None] @ weights[None, :], k=1) * dcc_matrices, axis=(1, 2))
        )
        alpha = 1 - self.confidence_level
        return -np.percentile(portfolio_volatility, alpha * 100)