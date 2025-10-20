from src.utils import get_project_root, ensure_dir, setup_logger


def run():
    # Initialize a logger for Week 5
    logger = setup_logger("WEEK5")

    # Define project directories
    ROOT = get_project_root()
    DATA_DIR = ROOT / "data"
    RAW = DATA_DIR / "raw"
    PROCESSED = DATA_DIR / "processed"
    MODELS = ROOT / "models"
    REPORTS = ROOT / "reports"
    NOTEBOOKS = ROOT / "notebooks"

    # Ensure that all required folders exist
    ensure_dir(RAW)
    ensure_dir(PROCESSED)
    ensure_dir(MODELS)
    ensure_dir(REPORTS)
    ensure_dir(NOTEBOOKS)

    # Import dependencies for modeling and tuning
    import sqlalchemy
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb
    from tensorflow.keras import layers, models
    from sklearn.preprocessing import MinMaxScaler
    import joblib
    import json
    import optuna
    import warnings
    warnings.filterwarnings("ignore")

    # Connect to SQLite database
    DB_PATH = DATA_DIR / "sales_data.db"
    engine = sqlalchemy.create_engine(f"sqlite:///{DB_PATH}")

    # Week 5 – Advanced Models (Random Forest, XGBoost, and LSTM)
    # Load the preprocessed monthly dataset
    monthly = pd.read_sql("SELECT * FROM monthly_sales_features", engine, parse_dates=["month"]).dropna()
    print(" Loaded monthly features:", monthly.shape)

    # Select relevant features and target variable
    features = [c for c in monthly.columns if (
        "lag" in c or "roll" in c or c in ["oil_price", "holiday_count", "month_of_year"]
    )]
    target = "sales"

    # Split data into training and validation sets (time-based)
    X = monthly[features].fillna(0)
    y = monthly[target]
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    # Define RMSE metric
    rmse = lambda y, yhat: np.sqrt(mean_squared_error(y, yhat))

    # Train a Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_val)
    print(" RF RMSE:", rmse(y_val, rf_preds))
    joblib.dump(rf, MODELS / "rf_week5.pkl")

    # Train an XGBoost model
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = {
        "objective": "reg:squarederror",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    xgb_model = xgb.train(params, dtrain, num_boost_round=300, evals=[(dval, "val")], verbose_eval=False)
    xgb_preds = xgb_model.predict(dval)
    print(" XGB RMSE:", rmse(y_val, xgb_preds))
    xgb_model.save_model(str(MODELS / "xgb_week5.json"))

    # Train an LSTM (Long Short-Term Memory) Neural Network
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_seq = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    X_train_s, X_val_s = X_seq[:split], X_seq[split:]
    y_train_s, y_val_s = y_train.values, y_val.values

    lstm = models.Sequential([
        layers.LSTM(64, input_shape=(1, X_scaled.shape[1])),
        layers.Dense(1)
    ])
    lstm.compile(optimizer="adam", loss="mse")

    # Train the LSTM model for 5 epochs
    history = lstm.fit(X_train_s, y_train_s, epochs=5, batch_size=64, verbose=1)
    lstm.save(str(MODELS / "lstm_week5.keras"))

    # Save training history for later visualization
    lstm_history = {
        "loss": history.history.get("loss", []),
        "val_loss": history.history.get("val_loss", [])
    }
    with open(MODELS / "lstm_history.json", "w") as f:
        json.dump(lstm_history, f)

    # Evaluate the LSTM model
    lstm_preds = lstm.predict(X_val_s).flatten()
    print(" LSTM RMSE:", rmse(y_val_s, lstm_preds))

    # Define objective function for Random Forest hyperparameter tuning with Optuna
    def rf_objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        max_depth = trial.suggest_int("max_depth", 5, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return rmse(y_val, preds)

    print("\n Tuning Random Forest with Optuna...")
    rf_study = optuna.create_study(
        direction="minimize",
        study_name="rf_study",
        storage=f"sqlite:///{ROOT}/optuna_studies.db",
        load_if_exists=True
    )
    rf_study.optimize(rf_objective, n_trials=20)
    print(" Best RF Params:", rf_study.best_params)
    print(" Best RF RMSE:", rf_study.best_value)

    # Train and save the best Random Forest model
    best_rf = RandomForestRegressor(**rf_study.best_params, random_state=42, n_jobs=-1)
    best_rf.fit(X_train, y_train)
    joblib.dump(best_rf, MODELS / "best_random_forest.pkl")

    # Define objective function for XGBoost tuning
    def xgb_objective(trial):
        params = {
            "objective": "reg:squarederror",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        }
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)
        preds = model.predict(dval)
        return rmse(y_val, preds)

    print("\n Tuning XGBoost with Optuna...")
    xgb_study = optuna.create_study(
        direction="minimize",
        study_name="xgb_study",
        storage=f"sqlite:///{ROOT}/optuna_studies.db",
        load_if_exists=True
    )
    xgb_study.optimize(xgb_objective, n_trials=20)
    print(" Best XGB Params:", xgb_study.best_params)
    print(" Best XGB RMSE:", xgb_study.best_value)

    # Train and save the best XGBoost model
    best_xgb = xgb.train(xgb_study.best_params, xgb.DMatrix(X, label=y), num_boost_round=200)
    best_xgb.save_model(str(MODELS / "best_xgboost.json"))

    print("\n Week 5 complete – all models and tuning results saved.")
