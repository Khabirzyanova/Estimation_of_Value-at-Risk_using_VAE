import numpy as np
from scipy.stats import chi2, norm
import pandas as pd

def kupiec_pof_test(predicted_var: np.ndarray, actual_returns: np.ndarray, confidence_level: float = 0.99) -> float:
    """
    Kupiec's Proportion of Failure (POF) Test.
    Tests whether the observed proportion of exceptions matches the expected proportion
    for a given confidence level of Value at Risk (VaR).

    Parameters:
        predicted_var (np.ndarray): Predicted Value at Risk (VaR) values.
        actual_returns (np.ndarray): Observed returns corresponding to the VaR estimates.
        confidence_level (float): Confidence level of VaR (e.g., 0.99 for 99% confidence). Default is 0.99.

    Returns:
        float: p-value of the POF test. A low p-value indicates the model is inconsistent with the observed failures.
    """
    # Check for exceptions where actual returns exceed the predicted VaR
    exceptions = actual_returns < predicted_var  # Boolean array
    num_exceptions = exceptions.sum()  # Total exceptions
    total_observations = len(actual_returns)  # Total data points

    # Expected exception probability
    expected_probability = 1 - confidence_level

    # Likelihood ratio for the POF test
    likelihood_ratio = -2 * (
        num_exceptions * np.log(expected_probability / (num_exceptions / total_observations)) +
        (total_observations - num_exceptions) * np.log((1 - expected_probability) / (1 - num_exceptions / total_observations))
    )

    # Compute the p-value from the chi-squared distribution
    p_value = 1 - chi2.cdf(likelihood_ratio, df=1)

    return p_value

def berkowitz_test(var: np.ndarray, target: np.ndarray, alpha: float = 0.99) -> float:
    """
    Berkowitz Test assesses the calibration of VaR forecasts. It tests whether the realized
    losses are consistent with the predicted VaR level.

    Parameters:
        var: Predicted VaRs.
        target: Corresponding returns.
        alpha: VaR confidence level. Default is 0.99.

    Returns:
        p-value of the Berkowitz test.
    """
    z = (target - var.mean()) / var.std()
    lr_berkowitz = -2 * (np.log(norm.cdf(z)).sum() - np.log(alpha) * (target < var).sum() - np.log(1 - alpha) * (target >= var).sum())
    pvalue = 1 - chi2.cdf(lr_berkowitz, df=1)
    return pvalue

def quantile_loss(var: np.ndarray, target: np.ndarray, alpha: float = 0.99) -> float:
    """
    Quantile loss also known as Pinball loss. Measures the discrepancy between
    true values and a corresponding 1-alpha quantile.

    Parameters:
        var: Predicted VaRs.
        target: Corresponding returns.
        alpha: VaR confidence level. Default is 0.99.

    Returns:
        The average value of the quantile loss function.
    """
    return np.where(target < var, alpha * (var - target), (1 - alpha) * (target - var)).mean()

def quadratic_loss(var: np.ndarray, target: np.ndarray, alpha: float = 0.99, a: float = 1.0) -> float:
    """
    Quadratic Loss measures the squared difference between the predicted VaR and returns,
    penalizing negative returns with weight (return - VaR)^2 and negative VaRs with
    weight -a * VaR.

    Parameters:
        var: Predicted VaRs.
        target: Corresponding returns.
        alpha: Weight parameter for return - VaR. Default is 0.99.
        a: Weight parameter for negative VaRs. Default is 1.

    Returns:
        Quadratic Loss value.
    """
    return np.where(target < var, (target - var)**2, -a * var).mean()

def smooth_loss(var: np.ndarray, target: np.ndarray, alpha: float = 0.99, d: float = 25.0) -> float:
    """
    Smooth Loss penalizes observations for which return - VaR < 0 more heavily with weight (1-alpha).

    Parameters:
        var: Predicted VaRs.
        target: Corresponding returns.
        alpha: Weight parameter. Default is 0.99.
        d: Parameter Default is 25.

    Returns:
        Smooth Loss value.
    """
    return ((alpha - (1 + np.exp(d * (target - var)))**(-1)) * (target - var)).mean()

def tick_loss(var: np.ndarray, target: np.ndarray, alpha: float = 0.99) -> float:
    """
    Tick Loss penalizes exceedances with weight alpha and non-exceedances with weight 1 - alpha.

    Parameters:
        var: Predicted VaRs.
        target: Corresponding returns.
        alpha: Weight parameter. Default is 0.99.

    Returns:
        Tick Loss value.
    """
    return ((alpha - (target < var).astype(float)) * (target - var)).mean()

def avg_exceedances(var: np.ndarray, target: np.ndarray) -> float:
    """
    Calculate the average number of exceedances.

    Parameters:
        var: Predicted VaRs.
        target: Corresponding returns.

    Returns:
        Average number of exceedances.
    """
    exceedances = (target < var).sum()
    return exceedances / len(target)

def regulatory_loss(var: np.ndarray, target: np.ndarray) -> float:
    """
    Regulatory Loss Function penalizes exceedances with squared difference and non-exceedances with 0.

    Parameters:
        var: Predicted VaRs.
        target: Corresponding returns.

    Returns:
        Regulatory Loss value.
    """
    return np.where(target < var, (var - target)**2, 0).mean()

def metrics(var: np.ndarray, target: np.ndarray, alpha: float = 0.99) -> dict:
    metrics_dict = {}

    metrics_dict['POF Test p-value'] = kupiec_pof_test(var, target, alpha)
    metrics_dict['Berkowitz Test p-value'] = berkowitz_test(var, target, alpha)
    metrics_dict['Quantile Loss'] = quantile_loss(var, target, alpha)
    metrics_dict['Quadratic Loss'] = quadratic_loss(var, target, alpha)
    metrics_dict['Smooth Loss'] = smooth_loss(var, target, alpha)
    metrics_dict['Tick Loss'] = tick_loss(var, target, alpha)
    metrics_dict['Average Exceedances'] = avg_exceedances(var, target)
    metrics_dict['Regulatory Loss'] = regulatory_loss(var, target)

    return metrics_dict

