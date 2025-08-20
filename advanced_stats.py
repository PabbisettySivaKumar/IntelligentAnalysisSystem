import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
 
# 📊 Bayesian Estimation
def run_bayesian_estimation(data):
    mean = np.mean(data)
    std = np.std(data)
    bayes_mean = np.random.normal(loc=mean, scale=std / np.sqrt(len(data)), size=1000)
    ci_lower, ci_upper = np.percentile(bayes_mean, [2.5, 97.5])
    return f"Estimated mean: {mean:.2f}\n95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]"

# 📈 Time Series Forecasting using ARIMA
def run_time_series_forecast(df, date_col, value_col):
    df = df[[date_col, value_col]].dropna()
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    ts = df[value_col]

    model = ARIMA(ts, order=(1, 1, 1))
    fit = model.fit()
    forecast = fit.forecast(steps=30)

    fig, ax = plt.subplots(figsize=(10, 4))
    ts.plot(ax=ax, label="Actual")
    forecast.plot(ax=ax, label="Forecast", color="red")
    ax.legend()
    return fig, forecast

# 🔬 Causal Inference using Linear Regression
def run_causal_inference(df, treatment_col, outcome_col):
    try:
        X = df[[treatment_col]].dropna()
        y = df[outcome_col].loc[X.index]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        summary = model.summary().as_text()
        return summary
    except Exception as e:
        return f"Error in causal inference: {e}"