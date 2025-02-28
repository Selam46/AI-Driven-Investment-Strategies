import numpy as np
import pandas as pd

class FinancialMetrics:
    def __init__(self):
        self.risk_free_rate = 0.02  # Assuming 2% risk-free rate
        
    def calculate_sharpe_ratio(self, returns):
        """
        Calculate the Sharpe Ratio
        """
        excess_returns = returns - self.risk_free_rate/252  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    def calculate_var(self, returns, confidence_level=0.95):
        """
        Calculate Value at Risk
        """
        return np.percentile(returns.dropna(), (1 - confidence_level) * 100)
    
    def calculate_rolling_metrics(self, df, window=252):
        """
        Calculate rolling financial metrics
        """
        # Rolling Sharpe Ratio
        rolling_returns = df['Daily_Return'].rolling(window=window)
        rolling_std = df['Daily_Return'].rolling(window=window).std()
        df['Rolling_Sharpe'] = (rolling_returns.mean() - self.risk_free_rate/252) / rolling_std * np.sqrt(252)
        
        # Rolling VaR
        df['Rolling_VaR'] = df['Daily_Return'].rolling(window=window).quantile(0.05)
        
        return df 