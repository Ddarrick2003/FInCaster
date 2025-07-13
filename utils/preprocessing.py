import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def preprocess_data(df):
    df = df.copy()

    # Convert columns to numeric
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)

    # Log Volume
    df['Volume'] = df['Volume'].fillna(1)
    df['Log_Volume'] = np.log(df['Volume'].clip(lower=1))

    # Returns
    df['Returns'] = df['Close'].pct_change().fillna(0)

    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD'] = macd - signal

    df.dropna(inplace=True)

    # Normalize
    features = ['Open', 'High', 'Low', 'Close', 'Log_Volume', 'RSI', 'MACD', 'Returns']
    minmax = MinMaxScaler()
    standard = StandardScaler()
    df[features] = minmax.fit_transform(df[features])

    return df, minmax, standard
