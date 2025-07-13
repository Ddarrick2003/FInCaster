import streamlit as st
import pandas as pd
from utils.preprocessing import preprocess_data
from model.lstm_model import create_sequences, build_lstm_model
from model.garch_model import forecast_garch_var
from tensorflow.keras.callbacks import EarlyStopping

st.set_page_config(page_title="FinCaster", layout="wide")
st.title("🌞 FinCaster - Smart Financial Forecaster")

uploaded_file = st.file_uploader("📥 Upload your OHLCV CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Fill missing returns
        df['Returns'] = df['Close'].pct_change().fillna(0)

        # Clean and transform
        df, _, _ = preprocess_data(df)

        st.success(f"✅ Data loaded and cleaned — shape: {df.shape}")
        st.dataframe(df.head())

        tab1, tab2 = st.tabs(["📈 LSTM Forecasting", "📉 GARCH Risk"])

        with tab1:
            st.subheader("LSTM Model Forecasting")
            try:
                features = ['Open', 'High', 'Low', 'Close', 'Log_Volume', 'RSI', 'MACD', 'Returns']
                X, y = create_sequences(df[features], target_col='Close')
                if len(X) == 0:
                    st.warning("⚠️ Not enough data to train.")
                else:
                    split = int(len(X) * 0.8)
                    X_train, X_test = X[:split], X[split:]
                    y_train, y_test = y[:split], y[split:]

                    model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
                    model.fit(X_train, y_train, epochs=10, batch_size=16,
                              validation_data=(X_test, y_test),
                              callbacks=[EarlyStopping(patience=3)], verbose=0)

                    preds = model.predict(X_test).flatten()
                    st.line_chart({"Actual": y_test[:100], "Predicted": preds[:100]})

                    # Downloadable CSV
                    df_preds = pd.DataFrame({
                        "Actual": y_test.flatten(),
                        "Predicted": preds
                    })
                    st.download_button("📥 Download Predicted Prices", df_preds.to_csv(index=False),
                                       file_name="lstm_predictions.csv")
            except Exception as e:
                st.error(f"❌ LSTM error: {e}")

        with tab2:
            st.subheader("GARCH-Based Risk Forecasting")
            try:
                vol_forecast, var_1d = forecast_garch_var(df)

                st.metric(label="1-Day Value at Risk (95%)", value=f"{var_1d:.2f}%")
                st.line_chart(vol_forecast.values)

                st.markdown("### 📘 Interpretation")
                st.info(f"""
                - The volatility chart shows future market risk.
                - Spikes = high uncertainty. Flat = stability.
                - 1-Day VaR of {abs(var_1d):.2f}% means with 95% confidence, losses should not exceed this in a day.
                """)
            except Exception as e:
                st.error(f"❌ GARCH error: {e}")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("👆 Upload a CSV file with OHLCV columns to begin.")
