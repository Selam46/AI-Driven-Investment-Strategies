import pandas as pd
import numpy as np
from scipy import stats

class DataPreprocessor:
    def __init__(self):
        pass
    
    def clean_data(self, df):
        """
        Clean the dataframe by handling missing values and outliers
        """
        # Handle missing values
        df = df.fillna(method='ffill')  # Forward fill
        df = df.fillna(method='bfill')  # Backward fill for any remaining NAs
        
        return df
    
    def calculate_returns(self, df):
        """
        Calculate daily and log returns
        """
        df['Daily_Return'] = df['Close'].pct_change()
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        return df
    
    def detect_outliers(self, df, column='Daily_Return', z_threshold=3):
        """
        Detect outliers using z-score method
        """
        z_scores = stats.zscore(df[column].dropna())
        outliers = np.abs(z_scores) > z_threshold
        df['Is_Outlier'] = False
        df.loc[df[column].dropna().index[outliers], 'Is_Outlier'] = True
        return df
    
    def calculate_volatility(self, df, window=20):
        """
        Calculate rolling volatility
        """
        df['Volatility'] = df['Daily_Return'].rolling(window=window).std() * np.sqrt(252)
        return df 