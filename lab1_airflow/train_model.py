from sklearn.preprocessing import StandardScaler, PowerTransformer
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


def scale_frame(frame):
    df = frame.copy()
    X, y = df.drop(columns=['median_house_value']), df['median_house_value']
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scale = scaler.fit_transform(X.values)
    Y_scale = power_trans.fit_transform(y.values.reshape(-1, 1))
    return X_scale, Y_scale, power_trans


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train():
    df = pd.read_csv("./df_clear.csv")
    X, Y, power_trans = scale_frame(df)
    X_train, X_val, y_train, y_val = train_test_split(X, Y,
                                                        test_size=0.3,
                                                        random_state=42)

    params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
              'l1_ratio': [0.001, 0.05, 0.01, 0.2],
              "penalty": ["l1", "l2", "elasticnet"],
              "loss": ['squared_error', 'huber', 'epsilon_insensitive'],
              "fit_intercept": [False, True],
              }

    lr = SGDRegressor(random_state=42)
    clf = GridSearchCV(lr, params, cv=3, n_jobs=4)
    clf.fit(X_train, y_train.reshape(-1))
    best = clf.best_estimator_

    y_pred = best.predict(X_val)
    y_price_pred = power_trans.inverse_transform(y_pred.reshape(-1, 1))
    (rmse, mae, r2) = eval_metrics(power_trans.inverse_transform(y_val), y_price_pred)

    print(f"Best params: {clf.best_params_}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2: {r2}")

    with open("lr_house.pkl", "wb") as file:
        joblib.dump(best, file)

    with open("power_trans.pkl", "wb") as file:
        joblib.dump(power_trans, file)

    metrics = {"rmse": rmse, "mae": mae, "r2": r2, "best_params": clf.best_params_}

    import json
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4, default=str)

    return metrics
