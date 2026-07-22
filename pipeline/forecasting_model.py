import joblib
import os
from prophet import Prophet


def make_forecast(df):
    # Check the columns
    print("Raw columns:", df.columns)

    # Rename columns if necessary (adjust according to your dataset)
    df = df.rename(columns={
        "timestamp": "ds",  # Replace with your actual column name
        "price": "y"  # Replace with your actual column name
    })

    # Ensure datetime column is proper and timezone-free
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)

    # Train Prophet model
    model = Prophet()
    model.fit(df)

    # Save the trained model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/prophet_model.pkl")

    return model
