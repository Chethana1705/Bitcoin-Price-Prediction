import pandas as pd

def preprocess_data(csv_path):
    df = pd.read_csv(csv_path, delimiter=';')

    # Convert to datetime and sort by date
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # Feature engineering
    df['price_range'] = df['high'] - df['low']
    df['avg_price'] = (df['open'] + df['close']) / 2
    df['return'] = df['close'].pct_change()
    df['ma_7'] = df['close'].rolling(window=7).mean()
    df['ma_14'] = df['close'].rolling(window=14).mean()

    # Targets
    df['next_close'] = df['close'].shift(-1)
    df['price_direction'] = (df['next_close'] > df['close']).astype(int)

    # Drop rows with NaN
    df = df.dropna()
    return df
