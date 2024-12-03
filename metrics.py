import numpy as np
from scipy.stats import chi2

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
