import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import ta
import warnings
warnings.filterwarnings('ignore')

print("🚀 Starting Stock Direction Predictor")
print("=" * 50)

# ================== STEP 3: DATA COLLECTION ==================

def get_stock_data(symbol='AAPL', period='2y'):
    """Get stock data from Yahoo Finance"""
    print(f"📊 Collecting data for {symbol}...")
    
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period)
    
    print(f"✅ Collected {len(data)} days of data")
    print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
    
    return data

# Collect data for Apple stock
stock_data = get_stock_data('AAPL', '3y')  # 3 years of data
print(f"Data shape: {stock_data.shape}")
print("\nFirst 5 rows:")
print(stock_data.head())

# ================== STEP 4: FEATURE ENGINEERING ==================

def create_features(df):
    """Create features for machine learning"""
    print("\n🔧 Engineering features...")
    
    df = df.copy()
    
    # 1. Basic Technical Indicators
    df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
    
    # 2. Momentum Indicators
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['Close'])
    
    # 3. Price Position Features
    df['price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df['intraday_return'] = (df['Close'] - df['Open']) / df['Open']
    
    # 4. Volume Features
    df['volume_sma_20'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
    
    # 5. Lagged Returns (previous days' performance)
    for lag in [1, 2, 3, 5]:
        df[f'return_{lag}d'] = df['Close'].pct_change(lag)
    
    # 6. Create Target Variable (what we want to predict)
    df['next_day_return'] = df['Close'].shift(-1) / df['Close'] - 1
    df['target'] = (df['next_day_return'] > 0).astype(int)  # 1 if up, 0 if down
    
    # Remove rows with missing data
    df = df.dropna()
    
    print(f"✅ Created features. Data shape: {df.shape}")
    return df

# Create features
featured_data = create_features(stock_data)

# Show feature summary
feature_columns = ['SMA_10', 'SMA_20', 'EMA_12', 'RSI', 'MACD', 'price_position', 
                   'gap', 'intraday_return', 'volume_ratio', 'return_1d', 'return_2d', 
                   'return_3d', 'return_5d']

print(f"\n📈 Features created: {len(feature_columns)}")
print("Feature list:", feature_columns)

# ================== STEP 5: DATA PREPARATION ==================

def prepare_data_for_modeling(df, feature_columns):
    """Prepare data for machine learning"""
    print("\n🎯 Preparing data for modeling...")
    
    # Select features and target
    X = df[feature_columns].values
    y = df['target'].values
    dates = df.index
    
    # Split data chronologically (last 20% for testing)
    split_index = int(len(X) * 0.8)
    
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    
    train_dates = dates[:split_index]
    test_dates = dates[split_index:]
    
    print(f"Training data: {len(X_train)} samples ({train_dates[0].date()} to {train_dates[-1].date()})")
    print(f"Testing data: {len(X_test)} samples ({test_dates[0].date()} to {test_dates[-1].date()})")
    
    # Check class balance
    print(f"\nTraining set balance:")
    print(f"Up days (1): {(y_train == 1).sum()} ({(y_train == 1).mean():.1%})")
    print(f"Down days (0): {(y_train == 0).sum()} ({(y_train == 0).mean():.1%})")
    
    return X_train, X_test, y_train, y_test, test_dates

# Prepare data
X_train, X_test, y_train, y_test, test_dates = prepare_data_for_modeling(featured_data, feature_columns)

# ================== STEP 6: MODEL TRAINING ==================

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    """Train and evaluate our model"""
    print("\n🤖 Training Random Forest model...")
    
    # Create and train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    test_probabilities = model.predict_proba(X_test)
    
    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    print(f"✅ Model trained!")
    print(f"Training accuracy: {train_accuracy:.3f}")
    print(f"Testing accuracy: {test_accuracy:.3f}")
    
    return model, test_predictions, test_probabilities

# Train model
model, predictions, probabilities = train_and_evaluate_model(X_train, y_train, X_test, y_test)

# ================== STEP 7: DETAILED EVALUATION ==================

print("\n📊 DETAILED MODEL EVALUATION")
print("=" * 40)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=['Down', 'Up']))

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)
print(f"\nConfusion Matrix:")
print(f"Predicted:  Down  Up")
print(f"Actual Down: {cm[0,0]:3d}  {cm[0,1]:3d}")
print(f"Actual Up:   {cm[1,0]:3d}  {cm[1,1]:3d}")

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n🎯 Top 5 Most Important Features:")
for i, row in feature_importance.head().iterrows():
    print(f"{row['feature']:15s}: {row['importance']:.3f}")

# ================== STEP 8: FINANCIAL PERFORMANCE ==================

def calculate_strategy_performance(test_data, predictions, probabilities):
    """Calculate how profitable our strategy would be"""
    print("\n💰 STRATEGY PERFORMANCE ANALYSIS")
    print("=" * 40)
    
    # Get test period data
    test_df = test_data.copy()
    test_df['predictions'] = predictions
    test_df['confidence'] = np.max(probabilities, axis=1)
    
    # Calculate actual returns
    test_df['actual_return'] = test_df['Close'].pct_change().shift(-1)
    
    # Strategy 1: Always follow predictions
    test_df['strategy_return'] = np.where(
        test_df['predictions'] == 1, 
        test_df['actual_return'],      # Go long if predicting up
        -test_df['actual_return']      # Go short if predicting down
    )
    
    # Strategy 2: Only trade when confident (> 60% probability)
    confidence_threshold = 0.6
    test_df['confident_strategy'] = np.where(
        test_df['confidence'] > confidence_threshold,
        test_df['strategy_return'],
        0  # No trade if not confident
    )
    
    # Calculate performance metrics
    strategy_returns = test_df['strategy_return'].dropna()
    confident_returns = test_df['confident_strategy'].dropna()
    buy_hold_returns = test_df['actual_return'].dropna()
    
    # Performance summary
    results = {
        'Strategy (All Trades)': {
            'total_return': (1 + strategy_returns).cumprod().iloc[-1] - 1,
            'sharpe_ratio': strategy_returns.mean() / strategy_returns.std() * np.sqrt(252),
            'win_rate': (strategy_returns > 0).mean(),
            'num_trades': len(strategy_returns)
        },
        'Strategy (Confident Only)': {
            'total_return': (1 + confident_returns).cumprod().iloc[-1] - 1,
            'sharpe_ratio': confident_returns.mean() / confident_returns.std() * np.sqrt(252) if confident_returns.std() > 0 else 0,
            'win_rate': (confident_returns > 0).mean(),
            'num_trades': (test_df['confidence'] > confidence_threshold).sum()
        },
        'Buy & Hold': {
            'total_return': (1 + buy_hold_returns).cumprod().iloc[-1] - 1,
            'sharpe_ratio': buy_hold_returns.mean() / buy_hold_returns.std() * np.sqrt(252),
            'win_rate': (buy_hold_returns > 0).mean(),
            'num_trades': len(buy_hold_returns)
        }
    }
    
    for strategy, metrics in results.items():
        print(f"\n{strategy}:")
        print(f"  Total Return: {metrics['total_return']:8.1%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:8.2f}")
        print(f"  Win Rate:     {metrics['win_rate']:8.1%}")
        print(f"  Trades:       {metrics['num_trades']:8d}")
    
    return test_df, results

# Calculate performance
test_period_data = featured_data.iloc[len(X_train):]
backtest_df, performance_results = calculate_strategy_performance(test_period_data, predictions, probabilities)

# ================== STEP 9: VISUALIZATION ==================

def create_visualizations(backtest_df):
    """Create helpful visualizations"""
    print("\n📈 Creating visualizations...")
    
    # Set up the plotting
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Stock Direction Prediction Results', fontsize=16, fontweight='bold')
    
    # 1. Cumulative Returns Comparison
    cumulative_strategy = (1 + backtest_df['strategy_return']).cumprod()
    cumulative_confident = (1 + backtest_df['confident_strategy']).cumprod()
    cumulative_buy_hold = (1 + backtest_df['actual_return']).cumprod()
    
    axes[0,0].plot(backtest_df.index, cumulative_strategy, label='Strategy (All)', linewidth=2)
    axes[0,0].plot(backtest_df.index, cumulative_confident, label='Strategy (Confident)', linewidth=2)
    axes[0,0].plot(backtest_df.index, cumulative_buy_hold, label='Buy & Hold', linewidth=2)
    axes[0,0].set_title('Cumulative Returns Comparison')
    axes[0,0].set_ylabel('Cumulative Return')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Prediction Confidence Distribution
    axes[0,1].hist(backtest_df['confidence'], bins=20, alpha=0.7, edgecolor='black')
    axes[0,1].axvline(0.6, color='red', linestyle='--', label='Confidence Threshold')
    axes[0,1].set_title('Prediction Confidence Distribution')
    axes[0,1].set_xlabel('Confidence Score')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Feature Importance
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True).tail(8)  # Top 8 features
    
    axes[1,0].barh(importance_df['feature'], importance_df['importance'])
    axes[1,0].set_title('Top Feature Importance')
    axes[1,0].set_xlabel('Importance Score')
    
    # 4. Monthly Win Rate
    monthly_data = backtest_df.copy()
    monthly_data['month'] = monthly_data.index.to_period('M')
    monthly_data['correct'] = (monthly_data['predictions'] == (monthly_data['actual_return'] > 0).astype(int))
    monthly_win_rate = monthly_data.groupby('month')['correct'].mean()
    
    axes[1,1].plot(monthly_win_rate.index.to_timestamp(), monthly_win_rate.values, marker='o')
    axes[1,1].axhline(0.5, color='red', linestyle='--', label='Random (50%)')
    axes[1,1].set_title('Monthly Prediction Accuracy')
    axes[1,1].set_ylabel('Accuracy')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Create visualizations
create_visualizations(backtest_df)

# ================== STEP 10: NEXT DAY PREDICTION ==================

def predict_tomorrow(model, latest_data, feature_columns):
    """Predict tomorrow's direction"""
    print("\n🔮 TOMORROW'S PREDICTION")
    print("=" * 30)
    
    # Get latest features
    latest_features = latest_data[feature_columns].iloc[-1:].values
    
    # Make prediction
    prediction = model.predict(latest_features)[0]
    probability = model.predict_proba(latest_features)[0]
    
    direction = "📈 UP" if prediction == 1 else "📉 DOWN"
    confidence = probability[prediction]
    
    print(f"Prediction for next trading day: {direction}")
    print(f"Confidence: {confidence:.1%}")
    
    if confidence > 0.6:
        print("🟢 High confidence prediction")
    elif confidence > 0.55:
        print("🟡 Moderate confidence prediction")
    else:
        print("🔴 Low confidence prediction - consider avoiding this trade")
    
    return prediction, confidence

# Make tomorrow's prediction
tomorrow_pred, tomorrow_conf = predict_tomorrow(model, featured_data, feature_columns)

# ================== STEP 11: MODEL VALIDATION ==================

def cross_validate_model(X, y):
    """Perform time-series cross validation"""
    print("\n🔄 CROSS-VALIDATION RESULTS")
    print("=" * 35)
    
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        # Split data
        X_train_cv = X[train_idx]
        X_val_cv = X[val_idx]
        y_train_cv = y[train_idx]
        y_val_cv = y[val_idx]
        
        # Train model
        cv_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        cv_model.fit(X_train_cv, y_train_cv)
        
        # Evaluate
        val_pred = cv_model.predict(X_val_cv)
        accuracy = accuracy_score(y_val_cv, val_pred)
        cv_scores.append(accuracy)
        
        print(f"Fold {fold}: {accuracy:.3f}")
    
    print(f"\nCross-validation mean: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores)*2:.3f})")
    
    return cv_scores

# Perform cross-validation
X_all = featured_data[feature_columns].values
y_all = featured_data['target'].values
cv_scores = cross_validate_model(X_all, y_all)

# ================== STEP 12: SUMMARY AND NEXT STEPS ==================

print("\n" + "="*60)
print("🎉 PROJECT SUMMARY")
print("="*60)

print(f"\n📊 Model Performance:")
print(f"   Test Accuracy: {accuracy_score(y_test, predictions):.1%}")
print(f"   Cross-val Mean: {np.mean(cv_scores):.1%}")

print(f"\n💼 Strategy Performance:")
total_return = performance_results['Strategy (All Trades)']['total_return']
sharpe = performance_results['Strategy (All Trades)']['sharpe_ratio']
print(f"   Total Return: {total_return:.1%}")
print(f"   Sharpe Ratio: {sharpe:.2f}")

print(f"\n🚀 NEXT STEPS TO IMPROVE:")
print("1. Try different stocks (MSFT, GOOGL, TSLA)")
print("2. Add more features (news sentiment, economic indicators)")
print("3. Experiment with other models (XGBoost, Neural Networks)")
print("4. Implement proper risk management")
print("5. Add real-time data updates")

print(f"\n📝 Quick Start Commands:")
print("# Try different stock:")
print("stock_data = get_stock_data('MSFT', '3y')")
print("# Retrain with new data and compare results!")

print(f"\n⚠️  Important Reminders:")
print("- This is for educational purposes only")
print("- Past performance doesn't guarantee future results")
print("- Always consider transaction costs in real trading")
print("- Start with paper trading before using real money")
print("\nThank you for using the Stock Direction Predictor! Happy Trading! 📈🚀")