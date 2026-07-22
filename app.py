import streamlit as st
st.set_page_config(layout="wide", page_title="Bitcoin ML Explorer", page_icon="📈")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Bitcoin_3_11_2025-5_12_2025_historical_data_coinmarketcap.csv", delimiter=";")
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

df = load_data()
features = ['open', 'high', 'low', 'close', 'volume', 'marketCap']

# Streamlit UI
st.title("📈 Bitcoin ML Model Explorer")
model_option = st.selectbox("Select Model Type", ["Regression", "Forecasting", "Classification"])

# --- Regression Model ---
if model_option == "Regression":
    st.subheader(":chart_with_upwards_trend: Regression: Predict Next Day Close Price")

    df['target_close'] = df['close'].shift(-1)
    df = df.dropna()

    X = df[features]
    y = df['target_close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    score = model.score(X_test, y_test)
    next_input = df[features].iloc[-1:].values
    next_pred = model.predict(next_input)[0]

    st.write(f"**R² Score:** {score:.4f}")

    prev_close = df['close'].iloc[-1]
    delta = next_pred - prev_close
    trend = "🔝 Increase" if delta > 0 else "🔽 Decrease"
    delta_color = "green" if delta > 0 else "red"

    st.markdown("### 📌 Predicted Price for Tomorrow")
    st.markdown("**Predicted Close Price**")
    st.markdown(f"<span style='font-size:36px; color:lime'><b>${next_pred:,.2f}</b></span>", unsafe_allow_html=True)
    st.markdown(f"<span style='color:{delta_color}'> {delta:+.2f}</span>", unsafe_allow_html=True)
    st.markdown(f"**Trend:** 🧽 {trend}", unsafe_allow_html=True)
    st.markdown(f"Based on latest available data from: <span style='color:lightgray'>{df['timestamp'].dt.date.iloc[-1]}</span>", unsafe_allow_html=True)

    st.line_chart(pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred}))

# --- Forecasting with ARIMA ---
elif model_option == "Forecasting":
    st.subheader(":calendar: Forecasting with ARIMA")

    ts = df.set_index('timestamp')['close']

    # Fixed ARIMA parameters
    p, d, q = 5, 1, 0
    st.write(f"Using ARIMA({p}, {d}, {q}) for forecasting")

    try:
        model = ARIMA(ts, order=(p, d, q))
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=15)
        forecast_dates = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=15)

        next_day_value = forecast.iloc[0]
        last_close = ts.iloc[-1]
        delta = next_day_value - last_close
        trend = "🔝 Increase" if delta > 0 else "🔽 Decrease"
        delta_color = "green" if delta > 0 else "red"

        st.markdown("### 📌 Forecast for Tomorrow")
        st.markdown(f"**Predicted Close on {forecast_dates[0].date()}**")
        st.markdown(f"<span style='font-size:36px; color:orange'><b>${next_day_value:,.2f}</b></span>", unsafe_allow_html=True)
        st.markdown(f"<span style='color:{delta_color}'> {delta:+.2f}</span>", unsafe_allow_html=True)
        st.markdown(f"**Trend:** 🧽 {trend}", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        ts.plot(label='Historical', ax=ax)
        forecast.plot(label='Forecast', ax=ax, style='--')
        plt.legend()
        st.pyplot(fig)

        forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast_Close': forecast.values})
        st.dataframe(forecast_df)

    except Exception as e:
        st.error(f"ARIMA model failed to fit: {e}")

# --- Classification Model ---
elif model_option == "Classification":
    st.subheader(":mag: Classification: Will Price Go Up Tomorrow?")

    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()

    X = df[features]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {acc:.2f}")

    next_input = df[features].iloc[-1:].values
    next_class = clf.predict(next_input)[0]
    proba = clf.predict_proba(next_input)[0][next_class]
    direction = "🔽 Price Expected to Go Down" if next_class == 0 else "🔝 Price Expected to Go Up"
    color = "red" if next_class == 0 else "green"

    st.markdown("### 📌 Prediction for Tomorrow")
    st.markdown("**Price Movement**")
    st.markdown(f"<span style='font-size:28px; color:{color}'><b>{direction}</b></span>", unsafe_allow_html=True)
    st.markdown(f"<span style='color:lightgreen'>✔ Confidence: {proba*100:.2f}%</span>", unsafe_allow_html=True)
    st.markdown(f"Based on latest data from: <span style='color:lightgray'>{df['timestamp'].dt.date.iloc[-1]}</span>", unsafe_allow_html=True)

    st.write("Confusion Matrix:")
    cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())