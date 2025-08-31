import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import ta
import warnings
warnings.filterwarnings('ignore')

class StockDirectionPredictor:
    """Main class for stock direction prediction"""
    
    def __init__(self, symbol='AAPL'):
        self.symbol = symbol
        self.model = None
        self.feature_columns = [
            'SMA_10', 'SMA_20', 'EMA_12', 'RSI', 'MACD',
            'price_position', 'gap', 'intraday_return', 'volume_ratio',
            'return_1d', 'return_2d', 'return_3d', 'return_5d'
        ]
        self.is_trained = False
        
    def get_stock_data(self, period='3y'):
        """Get stock data from Yahoo Finance"""
        ticker = yf.Ticker(self.symbol)
        data = ticker.history(period=period)
        return data
    
    def create_features(self, df):
        """Create technical features"""
        df = df.copy()
        
        # Technical indicators
        df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        
        # Price features
        df['price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['intraday_return'] = (df['Close'] - df['Open']) / df['Open']
        
        # Volume features
        df['volume_sma_20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
        
        # Lagged returns
        for lag in [1, 2, 3, 5]:
            df[f'return_{lag}d'] = df['Close'].pct_change(lag)
        
        # Target variable
        df['next_return'] = df['Close'].shift(-1) / df['Close'] - 1
        df['target'] = (df['next_return'] > 0).astype(int)
        
        return df.dropna()
    
    def train(self, period='3y'):
        """Train the model"""
        # Get and prepare data
        raw_data = self.get_stock_data(period)
        self.featured_data = self.create_features(raw_data)
        
        # Prepare features and target
        X = self.featured_data[self.feature_columns].values
        y = self.featured_data['target'].values
        
        # Split chronologically
        self.split_index = int(len(X) * 0.8)
        X_train = X[:self.split_index]
        y_train = y[:self.split_index]
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        return self
    
    def predict_tomorrow(self):
        """Predict tomorrow's direction"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
            
        latest_features = self.featured_data[self.feature_columns].iloc[-1:].values
        prediction = self.model.predict(latest_features)[0]
        probability = self.model.predict_proba(latest_features)[0]
        confidence = probability[prediction]
        
        return {
            'prediction': prediction,
            'direction': '📈 UP' if prediction == 1 else '📉 DOWN',
            'confidence': confidence,
            'probability_up': probability[1],
            'probability_down': probability[0]
        }
    
    def evaluate(self):
        """Evaluate model performance"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
            
        # Test data
        X = self.featured_data[self.feature_columns].values
        y = self.featured_data['target'].values
        X_test = X[self.split_index:]
        y_test = y[self.split_index:]
        
        # Predictions
        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        
        # Strategy performance
        test_data = self.featured_data.iloc[self.split_index:].copy()
        test_data['predictions'] = predictions
        test_data['strategy_return'] = np.where(
            test_data['predictions'] == 1,
            test_data['next_return'],
            -test_data['next_return']
        )
        
        total_return = (1 + test_data['strategy_return']).cumprod().iloc[-1] - 1
        buy_hold_return = (1 + test_data['next_return']).cumprod().iloc[-1] - 1
        sharpe = test_data['strategy_return'].mean() / test_data['strategy_return'].std() * np.sqrt(252)
        
        return {
            'accuracy': accuracy,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'sharpe_ratio': sharpe,
            'predictions': predictions,
            'probabilities': probabilities,
            'test_data': test_data
        }
    
    def get_feature_importance(self):
        """Get feature importance"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
            
        return pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
