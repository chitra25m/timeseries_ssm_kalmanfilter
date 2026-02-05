import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

np.random.seed(42)

def generate_data(n=600, season_period=12):
    level = np.zeros(n)
    trend = np.zeros(n)
    seasonal = np.zeros(n)
    y = np.zeros(n)

    Q_true = np.diag([0.05, 0.01, 0.02])
    R_true = np.array([[0.1]])

    level[0] = 10
    trend[0] = 0.05

    for t in range(1, n):
        level[t] = level[t-1] + trend[t-1] + np.random.normal(0, np.sqrt(Q_true[0, 0]))
        trend[t] = trend[t-1] + np.random.normal(0, np.sqrt(Q_true[1, 1]))
        seasonal[t] = 2 * np.sin(2 * np.pi * t / season_period) + np.random.normal(0, np.sqrt(Q_true[2, 2]))
        y[t] = level[t] + seasonal[t] + np.random.normal(0, np.sqrt(R_true[0, 0]))

    return y.reshape(-1, 1)

class KalmanFilter:
    def __init__(self, F, H, Q, R):
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.x = np.zeros((F.shape[0], 1))
        self.P = np.eye(F.shape[0])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, y):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (y - self.H @ self.x)
        self.P = (np.eye(len(self.P)) - K @ self.H) @ self.P

def neg_log_likelihood(params, y, F, H):
    q1, q2, q3, r = np.exp(params)
    Q = np.diag([q1, q2, q3])
    R = np.array([[r]])

    kf = KalmanFilter(F, H, Q, R)
    ll = 0

    for obs in y:
        kf.predict()
        S = H @ kf.P @ H.T + R
        e = obs - H @ kf.x
        ll += np.log(np.linalg.det(S)) + e.T @ np.linalg.inv(S) @ e
        kf.update(obs)

    return ll.squeeze()

def forecast(kf, steps):
    preds = []
    for _ in range(steps):
        kf.predict()
        preds.append((kf.H @ kf.x).item())
    return np.array(preds)

def metrics(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }

if __name__ == "__main__":

    data = generate_data()
    train, test = data[:500], data[500:]

    F = np.array([[1, 1, 0],
                  [0, 1, 0],
                  [0, 0, 1]])

    H = np.array([[1, 0, 1]])

    init_params = np.log([0.1, 0.1, 0.1, 0.1])
    res = minimize(neg_log_likelihood, init_params, args=(train, F, H), method="L-BFGS-B")

    q1, q2, q3, r = np.exp(res.x)
    Q_est = np.diag([q1, q2, q3])
    R_est = np.array([[r]])

    print("\nEstimated Process Noise Matrix Q")
    print(Q_est)

    print("\nEstimated Measurement Noise Matrix R")
    print(R_est)

    kf = KalmanFilter(F, H, Q_est, R_est)
    for y in train:
        kf.predict()
        kf.update(y)

    ssm_forecast = forecast(kf, len(test))

    sarima = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
    sarima_fit = sarima.fit(disp=False)
    sarima_forecast = sarima_fit.forecast(len(test))

    ssm_metrics = metrics(test.flatten(), ssm_forecast)
    sarima_metrics = metrics(test.flatten(), sarima_forecast)

    comparison = pd.DataFrame([ssm_metrics, sarima_metrics],
                              index=["State Space Model", "SARIMA"])

    print("\nForecast Performance Comparison")
    print(comparison)

