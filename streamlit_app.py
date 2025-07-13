import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.preprocessing import preprocess_data
from model.lstm_model import create_sequences, build_lstm_model
from model.garch_model import forecast_garch_var
from tensorflow.keras.callbacks import EarlyStopping

# Page Setup
st.set_page_config(page_title="FinCaster", layout="wide", page_icon="💡")
st.title("💡 FinCaster - Financial Forecasting Dashboard")

# File Upload
uploaded_file = st.file_uploader("📥 Upload your OHLCV CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully.")
        st.write("🔎 Data Preview", df.head())

        # Preprocessing: Add RSI, MACD, Returns, Log_Volume, etc.
        df = preprocess_data(df)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        st.stop()

    st.subheader("📊 Cleaned Data Sample")
    st.dataframe(df.tail())

    tab1, tab2, tab3 = st.tabs(["📈 LSTM Forecast", "📉 GARCH Risk", "⬇️ Download CSV"])

    with tab1:
        st.subheader("🔮 Multivariate LSTM Price Forecast")
        try:
            features = ['Open', 'High', 'Low', 'Close', 'Log_Volume', 'RSI', 'MACD', 'Returns']
            X, y = create_sequences(df[features], target_col='Close')

            if len(X) < 30:
                st.warning("⚠️ Not enough data for LSTM prediction.")
            else:
                split = int(len(X) * 0.8)
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]

                model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
                model.fit(X_train, y_train, epochs=10, batch_size=16,
                          validation_data=(X_test, y_test),
                          callbacks=[EarlyStopping(patience=3)],
                          verbose=0)
                preds = model.predict(X_test).flatten()

                st.plotly_chart(
                    go.Figure([
                        go.Scatter(y=y_test[:100], name="Actual", line=dict(color="green")),
                        go.Scatter(y=preds[:100], name="Predicted", line=dict(color="orange"))
                    ]).update_layout(title="Actual vs Predicted Prices")
                )

                st.success("✅ Forecast complete.")

        except Exception as e:
            st.error(f"LSTM Error: {e}")

    with tab2:
        st.subheader("📉 GARCH Volatility Forecast + VaR")
        try:
            vol_forecast, var_1d = forecast_garch_var(df)

            st.metric("1-Day VaR (95%)", f"{var_1d:.2f}%")
            st.plotly_chart(
                go.Figure([go.Scatter(y=vol_forecast, name="Forecasted Volatility")])
                .update_layout(title="GARCH Volatility Forecast")
            )

            st.info(f"""
            **Interpretation**:
            - This chart shows expected daily volatility.
            - The 1-day Value at Risk (VaR) is {abs(var_1d):.2f}% at 95% confidence.
            """)

        except Exception as e:
            st.error(f"GARCH Error: {e}")

    with tab3:
        st.subheader("⬇️ Download Cleaned Data with Indicators")
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", csv_data, file_name="processed_data.csv")

else:
    st.info("📤 Please upload a valid OHLCV CSV file to begin.")
