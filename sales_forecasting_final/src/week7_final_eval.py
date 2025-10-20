from src.utils import get_project_root, ensure_dir, setup_logger


def run():
    # Initialize a logger for Week 7
    logger = setup_logger("WEEK7")

    # Define main project directories
    ROOT = get_project_root()
    DATA_DIR = ROOT / "data"
    RAW = DATA_DIR / "raw"
    PROCESSED = DATA_DIR / "processed"
    MODELS = ROOT / "models"
    REPORTS = ROOT / "reports"
    NOTEBOOKS = ROOT / "notebooks"

    # Ensure that all directories exist
    ensure_dir(RAW)
    ensure_dir(PROCESSED)
    ensure_dir(MODELS)
    ensure_dir(REPORTS)
    ensure_dir(NOTEBOOKS)

    # Import necessary libraries for model evaluation and visualization
    import sqlalchemy
    import pandas as pd
    import numpy as np
    import joblib
    import xgboost as xgb
    from tensorflow.keras import models
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    import seaborn as sns
    import shap
    import json
    import optuna
    from prophet import Prophet
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import warnings
    warnings.filterwarnings("ignore")

    # Week 7 – Final Model Comparison and Ensembling

    # Connect to the SQLite database
    DB_PATH = DATA_DIR / "sales_data.db"
    engine = sqlalchemy.create_engine(f"sqlite:///{DB_PATH}")

    # Load processed dataset from the database
    monthly = pd.read_sql("SELECT * FROM monthly_sales_features", engine, parse_dates=["month"]).dropna()
    print(" Loaded dataset:", monthly.shape)

    # Define feature set and target variable
    features = [c for c in monthly.columns if "lag" in c or "roll" in c or c in ["oil_price", "holiday_count", "month_of_year"]]
    target = "sales"
    X = monthly[features]
    y = monthly[target]

    # Create train-validation split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    # Define RMSE function for model evaluation
    rmse = lambda y, yhat: np.sqrt(mean_squared_error(y, yhat))

    # Load pre-trained models
    rf = joblib.load(ROOT / "models/best_random_forest.pkl")
    rf_preds = rf.predict(X_val)
    rf_rmse = rmse(y_val, rf_preds)

    xgb_model = xgb.Booster()
    xgb_model.load_model(str(ROOT / "models/best_xgboost.json"))
    xgb_preds = xgb_model.predict(xgb.DMatrix(X_val))
    xgb_rmse = rmse(y_val, xgb_preds)

    lstm = models.load_model(ROOT / "models/lstm_week5.keras")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_val_s = X_scaled[split:].reshape((X_val.shape[0], 1, X_val.shape[1]))
    lstm_preds = lstm.predict(X_val_s, verbose=0).flatten()
    lstm_rmse = rmse(y_val.values, lstm_preds)

    # Combine predictions for ensemble (average of all models)
    ensemble_preds = (rf_preds + xgb_preds + lstm_preds) / 3
    ensemble_rmse = rmse(y_val, ensemble_preds)

    # Save RMSE comparison results
    results = pd.DataFrame([
        {"model": "RandomForest", "rmse": rf_rmse},
        {"model": "XGBoost", "rmse": xgb_rmse},
        {"model": "LSTM", "rmse": lstm_rmse},
        {"model": "Ensemble", "rmse": ensemble_rmse},
    ])

    leaderboard_path = PROCESSED / "model_leaderboard.csv"
    results.to_csv(leaderboard_path, index=False)
    print("\n Week 7 complete – results saved to:", leaderboard_path)
    print(results)

    # Visualization 1: Predicted vs Actual sales (last 50 validation samples)
    plt.figure(figsize=(12, 6))
    plt.plot(y_val.values[-50:], label="Actual", marker="o")
    plt.plot(rf_preds[-50:], label="RF", linestyle="--")
    plt.plot(xgb_preds[-50:], label="XGB", linestyle="--")
    plt.plot(lstm_preds[-50:], label="LSTM", linestyle="--")
    plt.plot(ensemble_preds[-50:], label="Ensemble", linewidth=2)
    plt.legend()
    plt.title("Predicted vs Actual (Last 50 Validation Points)")
    plt.ylabel("Sales")
    plt.tight_layout()
    plt.show()

    # Visualization 2: RMSE leaderboard comparison
    plt.figure(figsize=(8, 5))
    sns.barplot(x="model", y="rmse", data=results.sort_values("rmse"))
    plt.title("Model RMSE Leaderboard")
    plt.ylabel("Validation RMSE")
    plt.tight_layout()
    plt.show()

    # SHAP Feature Importance for XGBoost model
    try:
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(xgb.DMatrix(X_val))
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_val, plot_type="bar", max_display=10, show=True)
    except Exception as e:
        print(" SHAP importance skipped:", e)

    # Prophet Forecast vs Actual Sales (aggregated monthly)
    agg_sales = monthly.groupby("month")["sales"].sum().reset_index().rename(columns={"month": "ds", "sales": "y"})
    m = Prophet(yearly_seasonality=True)
    m.fit(agg_sales)
    future = m.make_future_dataframe(periods=6, freq="MS")
    fcst = m.predict(future)

    plt.figure(figsize=(12, 6))
    plt.plot(agg_sales["ds"], agg_sales["y"], label="Actual")
    plt.plot(fcst["ds"], fcst["yhat"], label="Prophet Forecast", linestyle="--")
    plt.axvline(x=agg_sales["ds"].max(), color="red", linestyle=":", label="Forecast Start")
    plt.title("Prophet Forecast vs Actual (Aggregated Monthly Sales)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Baseline Models: Naive Lag-12 and ARIMA
    cutoff = monthly["month"].max() - pd.DateOffset(months=6)
    val = monthly[monthly["month"] > cutoff]
    val["pred_naive"] = val["sales_lag_12"]
    naive_rmse = rmse(val["sales"], val["pred_naive"])
    print(f"Naive Lag-12 RMSE: {naive_rmse:.2f}")

    # ARIMA model on a sample store and family
    sample = monthly[(monthly.store_nbr == 1) & (monthly.family == monthly["family"].unique()[0])].set_index("month").asfreq("MS")
    train_arima = sample[sample.index <= cutoff]
    val_arima = sample[sample.index > cutoff]
    model = SARIMAX(train_arima["sales"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    res = model.fit(disp=False)
    arima_preds = res.forecast(len(val_arima))
    arima_rmse = rmse(val_arima["sales"], arima_preds)
    print(f"ARIMA RMSE: {arima_rmse:.2f}")

    print("\n Final Evaluation Completed Successfully!")
