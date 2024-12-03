import numpy as np
from metrics import kupiec_pof_test

class ModelEvaluator:
    """
    Class for evaluating Value at Risk (VaR) models.
    """

    def __init__(self, model, returns, weights=None):
        """
        Initialize the evaluator.

        Parameters:
            model: The VaR model to evaluate.
            returns: Historical returns for evaluation.
            weights: Portfolio weights (if applicable).
        """
        self.model = model
        self.returns = returns
        self.weights = weights

    def evaluate(self, confidence_level=0.99):
        """
        Evaluate the model using Kupiec's Proportion of Failures (POF) Test.

        Parameters:
            confidence_level: Confidence level for the VaR calculation.

        Returns:
            p-value from the POF test.
        """
        if hasattr(self.model, 'predict_var'):
            predicted_var = self.model.predict_var(self.returns, self.weights)
        else:
            predicted_var = self.model.rolling_var(self.returns)['VaR']

        actual_returns = self.returns.iloc[self.model.window_size:]
        return kupiec_pof_test(predicted_var, actual_returns, confidence_level)
    