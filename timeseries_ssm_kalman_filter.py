"""
Advanced Time Series Forecasting with State Space Models and Kalman Filtering
----------------------------------------------------------------------------

This script:
1. Generates a synthetic multivariate time series with trend, seasonality, and noise
2. Implements a State Space Model (SSM)
3. Implements a Kalman Filter from scratch
4. Estimates noise covariance matrices (Q, R) via Maximum Likelihood
5. Performs multi-step forecasting
6. Compares performance against a SARIMA benchmark
7. Reports RMSE, MAE, and MAPE

Author: AI Expert Solution
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

np.random.seed(42)

# ---------------------------------------------------------------------
# 1. SYNTHETIC DATA GENERATION
# ---------------------------------------------------------------------

def generate_time_series(n=600, seasonal_period=12):
    """
    True underlying state structure:
    State vector: [level, trend, seasonal]
    
    x_t = F x_{t-1} + w_t
    y_t = H x_t + v_t
    """
    level = np.zeros(n)
    trend = np.zeros(n)
    seasonal = np.zeros(n)
    y = np.zeros(n)

    # True noise covariances
    Q_true = np.diag([0.05, 0.01, 0.02])
    R_true = np.array([[0.1]])

    level[0], trend[0], seasonal[0] = 10, 0.05, 2

    for t in range(1, n):
        level[t] = level[t-1] + trend[t-1] + np.random.normal(0, np.sqrt(Q_true[0, 0]))
        trend[t] = trend[t-1] + np.random.normal(0, np.sqrt(Q_true[1, 1]))
        seasonal[t] = 2 * np.sin(2 * np.pi * t / seasonal_period) + np.random.normal(0, np.sqrt(Q_true[2, 2]))
        y[t] = level[t] + seasonal[t] + np.random.normal(0, np.sqrt(R_true[0, 0]))

    return y.reshape(-1, 1), Q_true, R_true


# ---------------------------------------------------------------------
# 2. KALMAN FILTER IMPLEMENTATION
# ---------------------------------------------------------------------

class KalmanFilter:
    """
    Custom Kalman Filter implementation
    """

    def __init__(self, F, H, Q, R, x0, P0):
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, y):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (y - self.H @ self.x)
        self.P = (np.eye(len(self.P)) - K @ self.H) @ self.P

    def filter(self, observations):
        estimates = []
        for y in observations:
            self.predict()
            self.update(y)
            estimates.append(self.x.copy())
        return np.array(estimates)


# ---------------------------------------------------------------------
# 3. LOG-LIKELIHOOD FOR PARAMETER ESTIMATION
# ---------------------------------------------------------------------

def negative_log_likelihood(params, y, F, H):
    q1, q2, q3, r = np.exp(params)
    Q = np.diag([q1, q2, q3])
    R = np.array([[r]])

    x0 = np.zeros((3, 1))
    P0 = np.eye(3)

    kf = KalmanFilter(F, H, Q, R, x0, P0)

    log_likelihood = 0
    for obs in y:
        kf.predict()
        S = H @ kf.P @ H.T + R
        residual = obs - H @ kf.x
        log_likelihood += -0.5 * (
            np.log(np.linalg.det(S)) +
            residual.T @ np.linalg.inv(S) @ residual
        )
        kf.update(obs)

    return -log_likelihood.squeeze()


# ---------------------------------------------------------------------
# 4. FORECASTING
# ---------------------------------------------------------------------

def forecast(kf, steps):
    forecasts = []
    for _ in range(steps):
        kf.predict()
        forecasts.append((kf.H @ kf.x).item())
    return np.array(forecasts)


# ---------------------------------------------------------------------
# 5. MAIN EXECUTION
# ---------------------------------------------------------------------

if __name__ == "__main__":

    # Generate data
    y, Q_true, R_true = generate_time_series()
    train, test = y[:500], y[500:]

    # State Space Matrices
    F = np.array([
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    H = np.array([[1, 0, 1]])

    # Parameter estimation
    init_params = np.log([0.1, 0.1, 0.1, 0.1])
    result = minimize(
        negative_log_likelihood,
        init_params,
        args=(train, F, H),
        method="L-BFGS-B"
    )

    q1, q2, q3, r = np.exp(result.x)
    Q_est = np.diag([q1, q2, q3])
    R_est = np.array([[r]])

    print("\nEstimated Process Noise Q:\n", Q_est)
    print("\nEstimated Measurement Noise R:\n", R_est)

    # Kalman Filtering
    kf = KalmanFilter(F, H, Q_est, R_est, np.zeros((3, 1)), np.eye(3))
    kf.filter(train)

    # Forecast
    ssm_forecast = forecast(kf, len(test))

    # -----------------------------------------------------------------
    # 6. BENCHMARK MODEL (SARIMA)
    # -----------------------------------------------------------------

    sarima = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_fit = sarima.fit(disp=False)
    sarima_forecast = sarima_fit.forecast(len(test))

    # -----------------------------------------------------------------
    # 7. EVALUATION METRICS
    # -----------------------------------------------------------------

    def metrics(true, pred):
        rmse = np.sqrt(mean_squared_error(true, pred))
        mae = mean_absolute_error(true, pred)
        mape = np.mean(np.abs((true - pred) / true)) * 100
        return rmse, mae, mape

    ssm_rmse, ssm_mae, ssm_mape = metrics(test.flatten(), ssm_forecast)
    sar_rmse, sar_mae, sar_mape = metrics(test.flatten(), sarima_forecast)

    print("\n--- Forecast Accuracy Comparison ---")
    print("SSM  -> RMSE:", ssm_rmse, " MAE:", ssm_mae, " MAPE:", ssm_mape)
    print("SARIMA -> RMSE:", sar_rmse, " MAE:", sar_mae, " MAPE:", sar_mape)
