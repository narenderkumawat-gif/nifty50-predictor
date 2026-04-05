import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="NIFTY 50 Predictor", page_icon="📈", layout="wide")

# Title
st.title("📈 NIFTY 50 Price Level Prediction")
st.markdown("**AI-Powered LSTM Model for Indian Stock Market**")

# Sidebar
st.sidebar.header("Settings")
forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 1, 30, 5)
sequence_length = st.sidebar.slider("Sequence Length (Days)", 30, 90, 60)
epochs = st.sidebar.slider("Training Epochs", 10, 50, 20)

# Cache data loading
@st.cache_data
def load_data():
    """Load NIFTY 50 data from Yahoo Finance"""
    nifty = yf.download('^NSEI', start='2010-01-01', end=datetime.now().strftime('%Y-%m-%d'), progress=False)
    return nifty

# Build and train model
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Main app logic
try:
    with st.spinner('Loading NIFTY 50 data...'):
        data = load_data()
    
    if data.empty:
        st.error("Failed to load data. Please try again later.")
    else:
        st.success(f"✅ Loaded {len(data)} days of NIFTY 50 data")
        
        # Display current data
        col1, col2, col3 = st.columns(3)
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2]
        change = current_price - prev_price
        change_pct = (change / prev_price) * 100
        
        col1.metric("Current NIFTY 50", f"₹{current_price:.2f}", f"{change:.2f} ({change_pct:+.2f}%)")
        col2.metric("High (Today)", f"₹{data['High'].iloc[-1]:.2f}")
        col3.metric("Low (Today)", f"₹{data['Low'].iloc[-1]:.2f}")
        
        # Prepare data
        prices = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_prices = scaler.fit_transform(prices)
        
        # Create sequences
        X, y = create_sequences(scaled_prices, sequence_length)
        split = int(0.9 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Train model
        if st.button("🚀 Train Model & Generate Forecast"):
            with st.spinner(f'Training LSTM model ({epochs} epochs)...'):
                model = build_lstm_model((X_train.shape[1], 1))
                history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0, validation_split=0.1)
                
                st.success("✅ Model trained successfully!")
                
                # Make predictions
                last_sequence = scaled_prices[-sequence_length:].reshape(1, sequence_length, 1)
                predictions = []
                
                for _ in range(forecast_days):
                    pred = model.predict(last_sequence, verbose=0)
                    predictions.append(pred[0, 0])
                    last_sequence = np.append(last_sequence[:, 1:, :], [[[pred[0, 0]]]], axis=1)
                
                # Inverse transform
                predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
                
                # Generate future dates
                last_date = data.index[-1]
                future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
                
                # Create forecast dataframe
                forecast_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Price': predictions.flatten()
                })
                
                # Display forecast
                st.subheader("📊 Forecast Results")
                st.dataframe(forecast_df.style.format({'Predicted Price': '₹{:.2f}'}), use_container_width=True)
                
                # Plot
                fig, ax = plt.subplots(figsize=(14, 6))
                
                # Historical prices (last 180 days)
                hist_data = data['Close'].iloc[-180:]
                ax.plot(hist_data.index, hist_data.values, label='Historical Prices', color='blue', linewidth=2)
                
                # Predicted prices
                ax.plot(future_dates, predictions, label=f'{forecast_days}-Day Forecast', color='red', linestyle='--', marker='o', linewidth=2)
                
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('NIFTY 50 Price (₹)', fontsize=12)
                ax.set_title(f'NIFTY 50: Historical vs Predicted ({forecast_days} Days Ahead)', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(alpha=0.3)
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Summary
                st.subheader("📈 Summary")
                final_pred = predictions[-1][0]
                total_change = final_pred - current_price
                total_change_pct = (total_change / current_price) * 100
                
                col1, col2 = st.columns(2)
                col1.metric(f"Predicted Price (Day {forecast_days})", f"₹{final_pred:.2f}", f"{total_change:.2f} ({total_change_pct:+.2f}%)")
                col2.metric("Model Training Loss", f"{history.history['loss'][-1]:.6f}")
                
                # Disclaimer
                st.warning("⚠️ **Disclaimer**: This is an AI-generated forecast based on historical patterns. Not financial advice. Markets are unpredictable.")
                
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please refresh the page and try again.")
