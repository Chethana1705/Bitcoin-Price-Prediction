from prophet import Prophet
import pandas as pd
import joblib
import os

def train_forecasting_model(df):
    prophet_df = df[['timestamp', 'close']].rename(columns={'timestamp': 'ds', 'close': 'y'})
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/prophet_model.pkl")
    return model

def make_forecast(model, periods=7):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
