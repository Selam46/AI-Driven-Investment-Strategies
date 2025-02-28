import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class FinancialDataLoader:
    def __init__(self):
        self.tickers = ['TSLA', 'BND', 'SPY']
        
    def download_data(self, start_date='2010-01-01', end_date=None):
        """
        Download historical data for specified tickers
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        data = {}
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)
                data[ticker] = df
            except Exception as e:
                print(f"Error downloading {ticker}: {e}")
                
        return data
    
    def load_local_data(self, filepath):
        """
        Load data from local CSV file
        """
        try:
            df = pd.read_csv(filepath)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            return df
        except Exception as e:
            print(f"Error loading data from {filepath}: {e}")
            return None 