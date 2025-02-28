import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

class ModelEvaluator:
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """
        Calculate common evaluation metrics for time series predictions
        """
        metrics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred)
        }
        return metrics
    
    @staticmethod
    def compare_models(actual, predictions_dict):
        """
        Compare multiple models' predictions
        """
        results = {}
        for model_name, predictions in predictions_dict.items():
            results[model_name] = ModelEvaluator.calculate_metrics(actual, predictions)
        return results 