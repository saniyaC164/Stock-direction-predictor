import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from stock_predictor import StockDirectionPredictor  # Import your model
import pandas as pd
import numpy as np

# Page config
st.set_page_config(
    page_title="Stock Direction Predictor",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
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
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor(symbol, period):
    """Load and train predictor with caching"""
    predictor = StockDirectionPredictor(symbol)
    predictor.train(period)
    return predictor

def create_returns_chart(test_data):
    """Create returns comparison chart"""
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
        title='Cumulative Returns Comparison',
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        template='plotly_white'
    )
    
    return fig

def main():
    """Main dashboard"""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Direction Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        symbol = st.selectbox("Select Stock", ['AAPL', 'MSFT', 'GOOGL', 'TSLA'])
        period = st.selectbox("Data Period", ['1y', '2y', '3y'])
        
        if st.button("🔍 Analyze Stock", type="primary"):
            st.session_state.analyze = True
            st.session_state.symbol = symbol
            st.session_state.period = period
    
    # Main content
    if hasattr(st.session_state, 'analyze') and st.session_state.analyze:
        
        with st.spinner("Training model and generating predictions..."):
            # Load predictor
            predictor = load_predictor(st.session_state.symbol, st.session_state.period)
            
            # Get predictions
            tomorrow = predictor.predict_tomorrow()
            evaluation = predictor.evaluate()
            
            # Display results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Tomorrow's Prediction", tomorrow['direction'])
            with col2:
                st.metric("Confidence", f"{tomorrow['confidence']:.1%}")
            with col3:
                st.metric("Model Accuracy", f"{evaluation['accuracy']:.1%}")
            with col4:
                st.metric("Strategy Return", f"{evaluation['total_return']:.1%}")
            
            # Charts
            st.subheader("📊 Performance Analysis")
            
            tab1, tab2 = st.tabs(["Returns", "Features"])
            
            with tab1:
                returns_fig = create_returns_chart(evaluation['test_data'])
                st.plotly_chart(returns_fig, use_container_width=True)
            
            with tab2:
                importance_df = predictor.get_feature_importance()
                fig = px.bar(importance_df.head(10), 
                           x='importance', y='feature', 
                           orientation='h', title='Feature Importance')
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("👈 Select a stock and click 'Analyze Stock' to get started!")

if __name__ == "__main__":
    main()

    