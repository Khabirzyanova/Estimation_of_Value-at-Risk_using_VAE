from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import norm, t
import pandas as pd

from gluonts.dataset.common import ListDataset
from gluonts.mx import Trainer
from gluonts.model.deepar import DeepAREstimator



class MultivariateVaR(ABC):
    """
    Base class for multivariate Value-at-Risk (VaR) models.
    """

    @abstractmethod
    def __init__(self, alpha: float, weights: np.ndarray, window_size: int) -> None:
        """
        Initialize the multivariate VaR model.

        Parameters:
            alpha: Confidence level for the VaR calculation (e.g., 0.05 for 95%).
            weights: Array of portfolio weights. The sum of weights must equal 1.
            window_size: Rolling window size for historical returns.
        """
        self.alpha = alpha
        self.weights = weights
        self.window_size = window_size

        # Ensure that weights sum to 1
        if not np.isclose(self.weights.sum(), 1.0):
            raise ValueError("Portfolio weights must sum to 1.")

        # Ensure window_size is positive
        if self.window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        super().__init__()

    @abstractmethod
    def forecast(self, returns: pd.DataFrame) -> float:
        """
        Forecasts VaR based on the returns of assets.

        Parameters:
            returns: A DataFrame with returns, where the index represents dates,
                     and columns represent assets.

        Returns:
            A float representing the calculated VaR for the portfolio.
        """
        # Ensure the returns DataFrame has two dimensions and at least two assets
        if len(returns.shape) != 2 or returns.shape[1] < 2:
            raise ValueError("The returns DataFrame must have at least two assets (columns).")

        # Ensure the number of weights matches the number of assets
        if self.weights.shape[0] != returns.shape[1]:
            raise ValueError("The number of weights must match the number of assets.")

        pass



class HistoricalSimulation(MultivariateVaR):
    """
    A class for calculating Value-at-Risk (VaR) using the Historical Simulation method.
    """

    def __init__(self, alpha: float, weights: np.ndarray, window_size: int) -> None:
        """
        Initialize the Historical Simulation model.

        Parameters:
            alpha: Confidence level for the VaR calculation (e.g., 0.05 for 95%).
            weights: Array of portfolio weights. The sum of weights must equal 1.
            window_size: Rolling window size for historical returns.
        """
        super().__init__(alpha=alpha, weights=weights, window_size=window_size)
        self.name = "HistoricalSimulation"
    

    def calculate_var(self, returns: pd.Series, alpha: float) -> float:
        """
        Calculate Historical Simulation VaR for a single asset.

        Parameters:
            returns: A Series of historical returns for a single asset.
            alpha: Confidence level for the VaR calculation.

        Returns:
            The calculated VaR for the asset.
        """
        return np.percentile(returns, alpha * 100)
    
    # def rolling_var(self, returns: pd.DataFrame) -> pd.DataFrame:
    #     var_values = []
    #     for i in range(self.window_size, len(returns)):
    #         window = returns.iloc[i - self.window_size:i]
    #         var = self.calculate_var(window)
    #         var_values.append({'Date': returns.index[i], 'VaR': var})
    #     return pd.DataFrame(var_values).set_index('Date')


    def forecast(self, returns: pd.DataFrame) -> float:
        """
        Forecast portfolio VaR using the Historical Simulation method.

        Parameters:
            returns: A DataFrame with historical returns, where the index represents dates,
                     and columns represent assets.

        Returns:
            The calculated portfolio VaR.
        """
        # Rolling window
        returns = returns.tail(self.window_size)

        # Correlation matrix of returns
        R = returns.corr()

        # Compute individual VaR contributions
        V = np.zeros(len(self.weights))
        for i in range(len(self.weights)):
            if self.weights[i] < 0:  # Short position
                V[i] = self.weights[i] * self.calculate_var(returns.iloc[:, i], alpha=self.alpha)
            else:  # Long position
                V[i] = self.weights[i] * self.calculate_var(returns.iloc[:, i], alpha=1 - self.alpha)

        # Aggregate portfolio VaR
        portfolio_var = -np.sqrt(V @ R @ V.T)
        return portfolio_var


class DCCGARCH(MultivariateVaR):
    """
    A class for Dynamic Conditional Correlation (DCC-GARCH) multivariate VaR model
    with support for normal or t-Student distributions.
    """

    def __init__(self, alpha: float, weights: np.ndarray, window_size: int, distribution: str = "normal") -> None:
        """
        Initialize the DCC-GARCH model.

        Parameters:
            alpha: Confidence level for the VaR calculation (e.g., 0.05 for 95%).
            weights: Array of portfolio weights. The sum of weights must equal 1.
            window_size: Rolling window size for historical returns.
            distribution: The type of distribution to use ("normal" or "t").
        """
        super().__init__(alpha=alpha, weights=weights, window_size=window_size)
        if distribution not in {"normal", "t"}:
            raise ValueError("Invalid distribution type. Choose 'normal' or 't'.")
        self.distribution = distribution
        self.name = "DCCGARCH"

    def dcc_likelihood(self, params, standardized_returns):
        """
        Calculate the likelihood for the DCC model.

        Parameters:
            params: The DCC parameters (a, b).
            standardized_returns: Standardized returns used for DCC fitting.

        Returns:
            The negative log-likelihood for the DCC model.
        """
        a, b = params
        T, N = standardized_returns.shape
        Q = np.cov(standardized_returns.T)
        Q_bar = Q.copy()
        likelihood = 0
        for t in range(T):
            Z_t = standardized_returns.iloc[t].values.reshape(-1, 1)
            Q = (1 - a - b) * Q_bar + a * np.outer(Z_t, Z_t.T) + b * Q
            R = np.diag(1 / np.sqrt(np.diag(Q))) @ Q @ np.diag(1 / np.sqrt(np.diag(Q)))
            if self.distribution == "normal":
                likelihood += np.log(np.linalg.det(R)) + Z_t.T @ np.linalg.inv(R) @ Z_t
            elif self.distribution == "t":
                df = 10  # Degrees of freedom for t-distribution (can be made a parameter)
                likelihood += np.log(np.linalg.det(R)) + (df + N) * np.log(
                    1 + (Z_t.T @ np.linalg.inv(R) @ Z_t) / df
                )
        return likelihood[0, 0]

    def fit_dcc(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Fit the DCC model to a set of returns.

        Parameters:
            returns: A DataFrame of asset returns.

        Returns:
            A 3D array of DCC correlation matrices over time.
        """
        conditional_volatilities = {
            column: self.fit_garch(returns[column]) for column in returns.columns
        }
        standardized_returns = returns.copy()
        for column in returns.columns:
            standardized_returns[column] = (
                returns[column] / conditional_volatilities[column]
            )

        initial_params = [0.01, 0.97]
        bounds = [(1e-6, 1), (1e-6, 1)]
        result = minimize(
            self.dcc_likelihood, initial_params, bounds=bounds, args=(standardized_returns,)
        )
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

    def forecast(self, returns: pd.DataFrame) -> float:
        """
        Forecast portfolio VaR using the DCC-GARCH model.

        Parameters:
            returns: A DataFrame with historical returns, where the index represents dates,
                     and columns represent assets.

        Returns:
            The calculated portfolio VaR.
        """
        returns = returns.tail(self.window_size)
        dcc_matrices = self.fit_dcc(returns)
        conditional_volatilities = np.array(
            [self.fit_garch(returns[col]) for col in returns.columns]
        )
        portfolio_volatility = np.sqrt(
            np.sum(self.weights[:, None] * conditional_volatilities**2, axis=0)
            + 2 * np.sum(
                np.triu(self.weights[:, None] @ self.weights[None, :], k=1)
                * dcc_matrices,
                axis=(1, 2),
            )
        )

        if self.distribution == "normal":
            alpha_percentile = (1 - self.alpha) * 100
            return -np.percentile(portfolio_volatility, alpha_percentile)
        elif self.distribution == "t":
            df = 10  # Degrees of freedom for t-distribution
            alpha_percentile = student_t.ppf(self.alpha, df)
            return -np.percentile(portfolio_volatility, alpha_percentile)
        

class DeepVaR(MultivariateVaR):
    """
    Class for fitting and predicting with the GluonTS DeepAR estimator, inheriting from MultivariateVaR.
    """

    def __init__(self, alpha: float, window_size: int, context_length: int = 15, 
                 epochs: int = 5, learning_rate: float = 1e-4, 
                 n_layers: int = 2, dropout: float = 0.1):
        """
        Initialize the DeepVaR model.

        Parameters:
            alpha (float): Confidence level for VaR.
            window_size (int): Rolling window size for returns.
            context_length (int): Context length for the DeepAR model.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the trainer.
            n_layers (int): Number of LSTM layers in the DeepAR model.
            dropout (float): Dropout rate for the LSTM.
        """
        super().__init__(alpha=alpha, window_size=window_size)
        self.name = 'DeepVaR'
        self.context_length = context_length
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.n_layers = n_layers
        self.dropout = dropout
        self.estimator = None

    def df_to_np(self, ts: pd.DataFrame) -> np.ndarray:
        """
        Convert pandas DataFrame to numpy array.
        """
        return ts.to_numpy().T

    def list_dataset(self, ts: pd.DataFrame, train: bool = True) -> ListDataset:
        """
        Convert pandas DataFrame to a GluonTS ListDataset for training or testing.

        Parameters:
            ts (pd.DataFrame): Time series data with datetime index and asset returns as columns.
            train (bool): If True, prepare training dataset; otherwise, prepare test dataset.
        """
        custom_dataset = self.df_to_np(ts)
        start = pd.Timestamp(ts.index[0])
        if train:
            return ListDataset([{'target': x, 'start': start} for x in custom_dataset[:, :-1]], freq='1D')
        return ListDataset([{'target': x, 'start': start} for x in custom_dataset], freq='1D')

    def fit(self, ts: pd.DataFrame):
        """
        Fit the DeepAR model on the training data.

        Parameters:
            ts (pd.DataFrame): Time series data with datetime index and asset returns as columns.
        """
        estimator = DeepAREstimator(
            prediction_length=1,
            context_length=self.context_length,
            freq='1D',
            trainer=Trainer(epochs=self.epochs, ctx="cpu", learning_rate=self.learning_rate, num_batches_per_epoch=50),
            num_layers=self.n_layers,
            dropout_rate=self.dropout,
            cell_type='lstm',
            num_cells=50
        )
        list_ds = self.list_dataset(ts, train=True)
        self.estimator = estimator.train(list_ds)

    def predict_ts(self, ts: pd.DataFrame):
        """
        Predict future time series values using the trained DeepAR model.

        Parameters:
            ts (pd.DataFrame): Time series data with datetime index and asset returns as columns.
        """
        test_ds = self.list_dataset(ts, train=False)
        return self.estimator.predict(test_ds, num_samples=1000)

    def forecast(self, returns: pd.DataFrame) -> float:
        """
        Forecast VaR for one day based on the fitted DeepAR model.

        Parameters:
            returns (pd.DataFrame): Asset returns data with datetime index and asset returns as columns.
        """
        V = np.zeros(len(self.weights))
        predictions_it = self.predict_ts(returns)
        predictions = list(predictions_it)
        for i in range(len(self.weights)):
            if self.weights[i] < 0:
                V[i] = self.weights[i] * np.percentile(predictions[i].samples[:, 0], self.alpha)
            else:
                V[i] = self.weights[i] * np.percentile(predictions[i].samples[:, 0], 100 - self.alpha)
        R = returns.corr()
        return -np.sqrt(V @ R @ V.T)


class  TempVaR(MultivariateVaR):
    pass




# class VarianceCovariance(MultivariateVaR):
#     def __init__(
#             self, 
#             alpha : float, 
#             weights: np.ndarray,
#         ):
#         """
#         The variance-covariance method is used to calculate VaR based on returns 
#         of each asset in a portfolio.

#         Parameters:
#             alpha:
#                 VaR confidence level
#             weights:
#                 An array of weights of assets in a portfolio, should be summed up to 1.
#                 If not provided, the dataloader iterates over all assets' returns.
#         """
#         super().__init__(alpha, weights)

#     def forecast(self, returns: pd.DataFrame):
#         super().forecast(returns)
#         cov_matrix = returns.cov()
#         loc = returns.mean() @ self.weights
#         scale = np.sqrt(self.weights.T @ cov_matrix @ self.weights)
#         return norm.ppf(1-self.alpha, loc=loc, scale=scale)