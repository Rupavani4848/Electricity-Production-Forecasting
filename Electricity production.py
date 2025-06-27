import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import streamlit as st 
from statsmodels.tsa.stattools import adfuller, acf, pacf 
from statsmodels.tsa.arima.model import ARIMA 
from sklearn.metrics import mean_squared_error 
from scipy.stats import boxcox 
from statsmodels.tsa.seasonal import seasonal_decompose 

st.set_page_config(page_title="Electricity Forecasting", layout="wide") 
st.title("Electricity Production Forecasting")

st.markdown(""" 
Welcome to the interactive time series analysis dashboard for *US Electricity Production*.  
Use the sidebar to explore data transformations, model performance, and forecasting methods. 
""") 

# Sidebar
with st.sidebar: 
    st.header("Analysis Menu") 
    option = st.radio("Choose a section:", [
        "Rolling Statistics", "ADF Test", "Log Transformation", "Moving Average",
        "Exponential Decay Transformation", "Seasonality Decomposition",
        "Autocorrelation and PACF", "Persistence Model", "ARIMA Models", "MSE Comparison"
    ])

# Load Data
try: 
    df = pd.read_csv('Electric_Production.csv', parse_dates=['DATE'], index_col='DATE') 
    df.columns = ['value'] 
except FileNotFoundError: 
    st.error("The file 'Electric_Production.csv' was not found. Please upload it to continue.") 
    st.stop() 

sns.set_style('darkgrid') 

# Rolling Statistics
if option == "Rolling Statistics": 
    rolling_mean = df.rolling(window=12).mean() 
    rolling_std = df.rolling(window=12).std() 
    fig, ax = plt.subplots(figsize=(10, 6)) 
    ax.plot(df, label='Original', color='cornflowerblue') 
    ax.plot(rolling_mean, label='Rolling Mean', color='firebrick') 
    ax.plot(rolling_std, label='Rolling Std', color='limegreen') 
    ax.set_title('Rolling Statistics') 
    ax.legend() 
    st.pyplot(fig) 

# ADF Test
elif option == "ADF Test": 
    def adfuller_test(ts, window=12): 
        movingAverage = ts.rolling(window).mean() 
        movingSTD = ts.rolling(window).std() 
        fig, ax = plt.subplots(figsize=(10, 6)) 
        ax.plot(ts, label='Original', color='cornflowerblue') 
        ax.plot(movingAverage, label='Rolling Mean', color='firebrick') 
        ax.plot(movingSTD, label='Rolling Std', color='limegreen') 
        ax.set_title('Rolling Statistics') 
        ax.legend() 
        st.pyplot(fig) 
        adf = adfuller(ts, autolag='AIC') 
        with st.expander("ADF Test Results"): 
            st.write(f"ADF Statistic: {round(adf[0], 3)}") 
            st.write(f"p-value: {round(adf[1], 3)}") 
            st.write("Critical Values:") 
            for key, val in adf[4].items(): 
                st.write(f"- {key}: {round(val, 3)}") 
            if adf[0] > adf[4]["5%"]: 
                st.error("Time series is non-stationary.") 
            else: 
                st.success("Time series is stationary.") 
    adfuller_test(df['value'])

# Log Transformation
elif option == "Log Transformation": 
    df_log = df.copy() 
    df_log['value'] = boxcox(df_log['value'], lmbda=0.0) 
    fig, ax = plt.subplots(figsize=(10, 6)) 
    ax.plot(df_log, color='cornflowerblue') 
    ax.set_title("After Logarithmic Transformation") 
    st.pyplot(fig)

# Moving Average
elif option == "Moving Average": 
    df_log = df.copy() 
    df_log['value'] = boxcox(df_log['value'], lmbda=0.0) 
    moving_avg = df_log.rolling(window=12).mean() 
    df_log_ma = df_log - moving_avg 
    df_log_ma.dropna(inplace=True) 
    fig, ax = plt.subplots(figsize=(10, 6)) 
    ax.plot(df_log_ma, color='cornflowerblue') 
    ax.set_title("After Moving Average") 
    st.pyplot(fig)

# Exponential Decay Transformation
elif option == "Exponential Decay Transformation": 
    df_log = df.copy() 
    df_log['value'] = boxcox(df_log['value'], lmbda=0.0) 
    moving_avg = df_log.rolling(window=12).mean() 
    df_log_ma = df_log - moving_avg 
    df_log_ma.dropna(inplace=True) 
    df_log_ewm = df_log_ma.ewm(halflife=12).mean() 
    diff = df_log_ma - df_log_ewm 
    fig, ax = plt.subplots(figsize=(10, 6)) 
    ax.plot(diff, color='cornflowerblue') 
    ax.set_title("After Exponential Decay Transformation") 
    st.pyplot(fig)

# Seasonality Decomposition
elif option == "Seasonality Decomposition": 
    df_log = df.copy() 
    df_log['value'] = boxcox(df_log['value'], lmbda=0.0) 
    moving_avg = df_log.rolling(window=12).mean() 
    df_log_ma = df_log - moving_avg 
    df_log_ma.dropna(inplace=True) 
    decomposition = seasonal_decompose(df_log_ma, model='additive') 
    fig = decomposition.plot() 
    st.pyplot(fig)

# Autocorrelation and PACF
elif option == "Autocorrelation and PACF": 
    df_log = df.copy() 
    df_log['value'] = boxcox(df_log['value'], lmbda=0.0) 
    moving_avg = df_log.rolling(window=12).mean() 
    df_log_ma = df_log - moving_avg 
    df_log_ma.dropna(inplace=True) 
    acf_vals = acf(df_log_ma, nlags=20) 
    pacf_vals = pacf(df_log_ma, nlags=20) 
    fig, axs = plt.subplots(1, 2, figsize=(12, 5)) 
    axs[0].stem(acf_vals) 
    axs[0].set_title('ACF') 
    axs[1].stem(pacf_vals) 
    axs[1].set_title('PACF') 
    st.pyplot(fig)

# Persistence Model
elif option == "Persistence Model": 
    df_log = df.copy() 
    df_log['value'] = boxcox(df_log['value'], lmbda=0.0) 
    train_size = int(len(df_log) * 0.66) 
    train, test = df_log[:train_size], df_log[train_size:] 
    history = train['value'].tolist() 
    predictions = [history[-1] for _ in range(len(test))] 
    mse = mean_squared_error(test['value'], predictions) 
    fig, ax = plt.subplots(figsize=(10, 6)) 
    ax.plot(test.index, test['value'], label='Actual', color='cornflowerblue') 
    ax.plot(test.index, predictions, label='Predicted', color='orange') 
    ax.set_title('Persistence Model Forecast') 
    ax.legend() 
    st.pyplot(fig) 
    st.metric("Mean Squared Error", round(mse, 4))

# ARIMA Model
elif option == "ARIMA Models": 
    df_log = df.copy() 
    df_log['value'] = boxcox(df_log['value'], lmbda=0.0) 
    train_size = int(len(df_log) * 0.66) 
    train, test = df_log[:train_size], df_log[train_size:] 
    model = ARIMA(train, order=(2, 1, 2)) 
    model_fit = model.fit() 
    predictions = model_fit.forecast(steps=len(test)) 
    mse = mean_squared_error(test['value'], predictions) 
    fig, ax = plt.subplots(figsize=(10, 6)) 
    ax.plot(test.index, test['value'], label='Actual', color='cornflowerblue') 
    ax.plot(test.index, predictions, label='ARIMA Predicted', color='orange') 
    ax.set_title('ARIMA Forecast') 
    ax.legend() 
    st.pyplot(fig) 
    st.metric("Mean Squared Error", round(mse, 4))

# MSE Comparison
elif option == "MSE Comparison": 
    df_log = df.copy() 
    df_log['value'] = boxcox(df_log['value'], lmbda=0.0) 
    train_size = int(len(df_log) * 0.66) 
    train, test = df_log[:train_size], df_log[train_size:] 
    # Persistence
    history = train['value'].tolist() 
    persistence_preds = [history[-1] for _ in range(len(test))] 
    persistence_mse = mean_squared_error(test['value'], persistence_preds) 
    # ARIMA
    model = ARIMA(train, order=(2, 1, 2)) 
    model_fit = model.fit() 
    arima_preds = model_fit.forecast(steps=len(test)) 
    arima_mse = mean_squared_error(test['value'], arima_preds) 
    st.subheader("MSE Comparison") 
    st.write(f"Persistence Model MSE: {round(persistence_mse, 4)}") 
    st.write(f"ARIMA Model MSE: {round(arima_mse, 4)}") 
    fig, ax = plt.subplots(figsize=(6, 4)) 
    sns.barplot(x=['Persistence', 'ARIMA'], y=[persistence_mse, arima_mse], palette='viridis', ax=ax) 
    ax.set_title("Mean Squared Error Comparison") 
    ax.set_ylabel("MSE") 
    st.pyplot(fig)
