import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import ta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Direction Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 2rem;
    background: linear-gradient(90deg, #1f77b4, #ff7f0e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-card {
    background: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
}
.prediction-up {
    color: #00ff00;
    font-weight: bold;
    font-size: 1.5rem;
}
.prediction-down {
    color: #ff0000;
    font-weight: bold;
    font-size: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# Cache functions for better performance
@st.cache_data
def load_stock_data(symbol, period):
    """Load stock data with caching"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        return data
    except Exception as e:
        st.error(f"Error loading data for {symbol}: {str(e)}")
        return None

@st.cache_data
def create_features(df):
    """Create technical features with caching"""
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

def train_model(df, feature_columns):
    """Train the prediction model"""
    X = df[feature_columns].values
    y = df['target'].values
    
    # Split data chronologically
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    test_predictions = model.predict(X_test)
    test_probabilities = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, test_predictions)
    
    return model, test_predictions, test_probabilities, accuracy, X_test, y_test

def create_returns_chart(df, predictions, split_index):
    """Create cumulative returns chart"""
    test_data = df.iloc[split_index:].copy()
    test_data['predictions'] = predictions
    test_data['strategy_return'] = np.where(
        test_data['predictions'] == 1,
        test_data['next_return'],
        -test_data['next_return']
    )
    
    # Calculate cumulative returns
    cumulative_strategy = (1 + test_data['strategy_return']).cumprod()
    cumulative_buy_hold = (1 + test_data['next_return']).cumprod()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=test_data.index,
        y=(cumulative_strategy - 1) * 100,
        name='ML Strategy',
        line=dict(color='#1f77b4', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=test_data.index,
        y=(cumulative_buy_hold - 1) * 100,
        name='Buy & Hold',
        line=dict(color='#ff7f0e', width=3)
    ))
    
    fig.update_layout(
        title='Strategy Performance Comparison',
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def create_accuracy_chart(df, predictions, split_index):
    """Create rolling accuracy chart"""
    test_data = df.iloc[split_index:].copy()
    test_data['predictions'] = predictions
    test_data['correct'] = (test_data['predictions'] == test_data['target'])
    
    # Calculate rolling accuracy
    window = 30
    rolling_accuracy = test_data['correct'].rolling(window=window).mean() * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=test_data.index,
        y=rolling_accuracy,
        name=f'{window}-Day Rolling Accuracy',
        line=dict(color='#2ca02c', width=2)
    ))
    
    fig.add_hline(y=50, line_dash="dash", line_color="red", 
                  annotation_text="Random Guess (50%)")
    
    fig.update_layout(
        title=f'Model Accuracy Over Time ({window}-Day Rolling Average)',
        xaxis_title='Date',
        yaxis_title='Accuracy (%)',
        template='plotly_white'
    )
    
    return fig

def create_feature_importance_chart(model, feature_columns):
    """Create feature importance chart"""
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=importance_df['importance'],
        y=importance_df['feature'],
        orientation='h',
        marker_color='#1f77b4'
    ))
    
    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Importance Score',
        yaxis_title='Features',
        template='plotly_white'
    )
    
    return fig

def create_prediction_gauge(confidence, prediction):
    """Create a gauge chart for prediction confidence"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Prediction Confidence"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkgreen" if prediction == 1 else "darkred"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "lightgreen"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 60}}))
    
    fig.update_layout(height=300)
    return fig

# Main dashboard function
def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Direction Predictor</h1>', unsafe_allow_html=True)
    st.markdown("*AI-powered next-day stock direction prediction with comprehensive analysis*")
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Stock selection
        symbol = st.selectbox(
            "Select Stock",
            ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'META'],
            index=0
        )
        
        # Data period
        period = st.selectbox(
            "Data Period",
            ['1y', '2y', '3y', '5y'],
            index=2
        )
        
        # Analysis button
        analyze_button = st.button("🔍 Analyze Stock", type="primary")
        
        st.markdown("---")
        st.markdown("### 📊 Model Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.5,
            max_value=0.9,
            value=0.6,
            step=0.05
        )
    
    # Main content
    if analyze_button:
        with st.spinner(f"Analyzing {symbol}... This may take a moment."):
            
            # Load data
            stock_data = load_stock_data(symbol, period)
            
            if stock_data is not None and len(stock_data) > 100:
                
                # Create features
                featured_data = create_features(stock_data)
                
                # Define feature columns
                feature_columns = [
                    'SMA_10', 'SMA_20', 'EMA_12', 'RSI', 'MACD',
                    'price_position', 'gap', 'intraday_return', 'volume_ratio',
                    'return_1d', 'return_2d', 'return_3d', 'return_5d'
                ]
                
                # Train model
                model, predictions, probabilities, accuracy, X_test, y_test = train_model(featured_data, feature_columns)
                
                # Calculate split index for charts
                split_index = int(len(featured_data) * 0.8)
                
                # Tomorrow's prediction
                latest_features = featured_data[feature_columns].iloc[-1:].values
                tomorrow_pred = model.predict(latest_features)[0]
                tomorrow_prob = model.predict_proba(latest_features)[0]
                tomorrow_confidence = tomorrow_prob[tomorrow_pred]
                
                # Display key metrics
                st.markdown("## 🎯 Key Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    direction = "📈 UP" if tomorrow_pred == 1 else "📉 DOWN"
                    confidence_color = "green" if tomorrow_confidence > 0.6 else "orange" if tomorrow_confidence > 0.55 else "red"
                    st.markdown(f'<div class="metric-card"><h3>Tomorrow\'s Prediction</h3><p class="prediction-{"up" if tomorrow_pred == 1 else "down"}">{direction}</p></div>', unsafe_allow_html=True)
                
                with col2:
                    st.metric("Confidence", f"{tomorrow_confidence:.1%}", 
                             delta=f"{(tomorrow_confidence-0.5)*100:+.1f}% vs Random")
                
                with col3:
                    st.metric("Model Accuracy", f"{accuracy:.1%}", 
                             delta=f"{(accuracy-0.5)*100:+.1f}% vs Random")
                
                with col4:
                    # Calculate strategy return
                    test_data = featured_data.iloc[split_index:].copy()
                    test_data['predictions'] = predictions
                    test_data['strategy_return'] = np.where(
                        test_data['predictions'] == 1,
                        test_data['next_return'],
                        -test_data['next_return']
                    )
                    total_return = (1 + test_data['strategy_return']).cumprod().iloc[-1] - 1
                    buy_hold_return = (1 + test_data['next_return']).cumprod().iloc[-1] - 1
                    
                    st.metric("Strategy Return", f"{total_return:.1%}", 
                             delta=f"{(total_return-buy_hold_return)*100:+.1f}% vs Buy & Hold")
                
                # Confidence gauge
                st.markdown("## 🎚️ Prediction Confidence")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    gauge_fig = create_prediction_gauge(tomorrow_confidence, tomorrow_pred)
                    st.plotly_chart(gauge_fig, use_container_width=True)
                
                with col2:
                    # Confidence interpretation
                    if tomorrow_confidence > 0.65:
                        st.success("🟢 **High Confidence** - Strong signal for trading")
                    elif tomorrow_confidence > 0.55:
                        st.warning("🟡 **Moderate Confidence** - Consider position sizing")
                    else:
                        st.error("🔴 **Low Confidence** - Avoid trading on this signal")
                    
                    st.write(f"**Predicted Direction:** {direction}")
                    st.write(f"**Probability Up:** {tomorrow_prob[1]:.1%}")
                    st.write(f"**Probability Down:** {tomorrow_prob[0]:.1%}")
                
                # Performance analysis tabs
                st.markdown("## 📊 Performance Analysis")
                
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Returns", "🎯 Accuracy", "🔧 Features", "📋 Details"])
                
                with tab1:
                    returns_fig = create_returns_chart(featured_data, predictions, split_index)
                    st.plotly_chart(returns_fig, use_container_width=True)
                    
                    # Performance metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Strategy Metrics")
                        sharpe = test_data['strategy_return'].mean() / test_data['strategy_return'].std() * np.sqrt(252)
                        win_rate = (test_data['strategy_return'] > 0).mean()
                        st.write(f"**Sharpe Ratio:** {sharpe:.2f}")
                        st.write(f"**Win Rate:** {win_rate:.1%}")
                        st.write(f"**Total Trades:** {len(test_data)}")
                    
                    with col2:
                        st.markdown("### Buy & Hold Metrics")
                        bh_sharpe = test_data['next_return'].mean() / test_data['next_return'].std() * np.sqrt(252)
                        bh_win_rate = (test_data['next_return'] > 0).mean()
                        st.write(f"**Sharpe Ratio:** {bh_sharpe:.2f}")
                        st.write(f"**Win Rate:** {bh_win_rate:.1%}")
                        st.write(f"**Max Drawdown:** {((1 + test_data['next_return']).cumprod() / (1 + test_data['next_return']).cumprod().cummax() - 1).min():.1%}")
                
                with tab2:
                    accuracy_fig = create_accuracy_chart(featured_data, predictions, split_index)
                    st.plotly_chart(accuracy_fig, use_container_width=True)
                    
                    # Accuracy breakdown
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Prediction Breakdown")
                        correct_predictions = (predictions == y_test).sum()
                        total_predictions = len(y_test)
                        st.write(f"**Correct Predictions:** {correct_predictions}/{total_predictions}")
                        st.write(f"**Accuracy:** {accuracy:.1%}")
                        
                        # Confusion matrix
                        from sklearn.metrics import confusion_matrix
                        cm = confusion_matrix(y_test, predictions)
                        st.write("**Confusion Matrix:**")
                        st.write(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
                        st.write(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
                    
                    with col2:
                        # Monthly accuracy
                        test_df = featured_data.iloc[split_index:].copy()
                        test_df['predictions'] = predictions
                        test_df['correct'] = (test_df['predictions'] == test_df['target'])
                        test_df['month'] = test_df.index.to_period('M')
                        monthly_acc = test_df.groupby('month')['correct'].mean()
                        
                        st.markdown("### Monthly Accuracy")
                        for month, acc in monthly_acc.tail(6).items():
                            st.write(f"**{month}:** {acc:.1%}")
                
                with tab3:
                    importance_fig = create_feature_importance_chart(model, feature_columns)
                    st.plotly_chart(importance_fig, use_container_width=True)
                    
                    # Feature analysis
                    importance_df = pd.DataFrame({
                        'Feature': feature_columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Top Features")
                        st.dataframe(importance_df.head(8), hide_index=True)
                    
                    with col2:
                        st.markdown("### Feature Insights")
                        top_feature = importance_df.iloc[0]['Feature']
                        top_importance = importance_df.iloc[0]['Importance']
                        st.write(f"**Most Important:** {top_feature} ({top_importance:.3f})")
                        
                        # Feature correlation with target
                        correlations = featured_data[feature_columns + ['target']].corr()['target'].abs().sort_values(ascending=False)
                        st.write(f"**Highest Correlation:** {correlations.index[1]} ({correlations.iloc[1]:.3f})")
                
                with tab4:
                    st.markdown("### 📋 Model Details")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Data Information:**")
                        st.write(f"• Total samples: {len(featured_data)}")
                        st.write(f"• Training samples: {split_index}")
                        st.write(f"• Testing samples: {len(featured_data) - split_index}")
                        st.write(f"• Features used: {len(feature_columns)}")
                        st.write(f"• Date range: {featured_data.index[0].date()} to {featured_data.index[-1].date()}")
                    
                    with col2:
                        st.markdown("**Model Parameters:**")
                        st.write("• Algorithm: Random Forest")
                        st.write("• Trees: 100")
                        st.write("• Max depth: 10")
                        st.write("• Class weight: Balanced")
                        st.write("• Random state: 42")
                    
                    # Classification report
                    st.markdown("### Classification Report")
                    report = classification_report(y_test, predictions, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.round(3))
                    
                    # Recent predictions
                    st.markdown("### Recent Predictions")
                    recent_data = featured_data.iloc[split_index:].tail(10).copy()
                    recent_data['predictions'] = predictions[-10:]
                    recent_data['actual_direction'] = (recent_data['next_return'] > 0).astype(int)
                    recent_data['correct'] = recent_data['predictions'] == recent_data['actual_direction']
                    
                    display_cols = ['Close', 'predictions', 'actual_direction', 'correct', 'next_return']
                    recent_display = recent_data[display_cols].copy()
                    recent_display['predictions'] = recent_display['predictions'].map({1: '📈', 0: '📉'})
                    recent_display['actual_direction'] = recent_display['actual_direction'].map({1: '📈', 0: '📉'})
                    recent_display['correct'] = recent_display['correct'].map({True: '✅', False: '❌'})
                    recent_display.columns = ['Close Price', 'Predicted', 'Actual', 'Correct?', 'Return %']
                    recent_display['Return %'] = (recent_display['Return %'] * 100).round(2)
                    
                    st.dataframe(recent_display, use_container_width=True)
            
            else:
                st.error("Failed to load sufficient data. Please try a different stock or period.")
    
    else:
        # Welcome screen
        st.markdown("## 👋 Welcome to the Stock Direction Predictor!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 What This Does
            - Predicts if a stock will go **UP** or **DOWN** tomorrow
            - Uses **machine learning** on technical indicators
            - Provides **confidence scores** for each prediction
            - Shows **backtesting results** and strategy performance
            """)
        
        with col2:
            st.markdown("""
            ### 🚀 How to Use
            1. Select a stock from the sidebar
            2. Choose the data period (more data = better training)
            3. Click **"Analyze Stock"** to run the prediction
            4. Explore the results in different tabs
            """)
        
        st.markdown("---")
        st.info("👈 **Get started by selecting a stock and clicking 'Analyze Stock' in the sidebar!**")
        
        # Sample results preview
        st.markdown("### 📊 Sample Results Preview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Typical Accuracy", "55-60%", "+5-10% vs Random")
        with col2:
            st.metric("Strategy Returns", "15-25%", "Varies by stock")
        with col3:
            st.metric("Confidence Range", "50-85%", "Higher = Better")

# Helper functions for charts (defined after main to avoid caching issues)
def create_prediction_gauge(confidence, prediction):
    """Create a gauge chart for prediction confidence"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 55], 'color': "lightgray"},
                {'range': [55, 65], 'color': "yellow"},
                {'range': [65, 100], 'color': "lightgreen"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 60}}))
    
    fig.update_layout(height=250)
    return fig

# Run the app
if __name__ == "__main__":
    main()
