import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class TimeSeriesAnalyzer:
    def __init__(self):
        self.sarima_model = None
        self.lstm_model = None
        self.scaler = MinMaxScaler()
        
    def check_stationarity(self, timeseries):
        """
        Perform Augmented Dickey-Fuller test for stationarity
        """
        result = adfuller(timeseries)
        return {
            'ADF Statistic': result[0],
            'p-value': result[1],
            'Critical values': result[4]
        }
    
    def decompose_series(self, data, period=252):
        """
        Decompose time series into trend, seasonal, and residual components
        """
        return seasonal_decompose(data, period=period)
    
    def prepare_data(self, data, train_size=0.8):
        """
        Split data into training and testing sets
        """
        train_size = int(len(data) * train_size)
        train_data = data[:train_size]
        test_data = data[train_size:]
        return train_data, test_data

class SARIMAModel:
    def __init__(self):
        self.model = None
        
    def train(self, train_data, seasonal=True, m=252):
        """
        Train SARIMA model using auto_arima
        """
        self.model = auto_arima(train_data,
                               seasonal=seasonal,
                               m=m,
                               start_p=0, start_q=0,
                               max_p=3, max_q=3,
                               start_P=0, start_Q=0,
                               max_P=2, max_Q=2,
                               d=1, D=1,
                               trace=True,
                               error_action='ignore',
                               suppress_warnings=True,
                               stepwise=True)
        return self.model
    
    def predict(self, n_periods):
        """
        Generate predictions
        """
        return self.model.predict(n_periods=n_periods)

class LSTMModel:
    def __init__(self, look_back=60):
        self.model = None
        self.look_back = look_back
        self.scaler = MinMaxScaler()
        
    def prepare_sequences(self, data):
        """
        Prepare data sequences for LSTM
        """
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        X, y = [], []
        for i in range(self.look_back, len(scaled_data)):
            X.append(scaled_data[i-self.look_back:i, 0])
            y.append(scaled_data[i, 0])
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """
        Build LSTM model architecture
        """
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        self.model = model
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
        """
        Train LSTM model
        """
        return self.model.fit(X_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=validation_split,
                            verbose=1)
    
    def predict(self, data):
        """
        Generate predictions using the trained model
        """
        predictions = []
        current_batch = data[-self.look_back:]
        current_batch = current_batch.reshape((1, self.look_back, 1))
        
        for _ in range(len(data)):
            prediction = self.model.predict(current_batch)[0]
            predictions.append(prediction)
            current_batch = np.roll(current_batch, -1)
            current_batch[0, -1, 0] = prediction
            
        return self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)) 