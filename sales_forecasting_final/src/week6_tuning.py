from src.utils import get_project_root, ensure_dir, setup_logger


def run():
    # Initialize a logger for Week 6
    logger = setup_logger("WEEK6")

    # Define main project directories
    ROOT = get_project_root()
    DATA_DIR = ROOT / "data"
    RAW = DATA_DIR / "raw"
    PROCESSED = DATA_DIR / "processed"
    MODELS = ROOT / "models"
    REPORTS = ROOT / "reports"
    NOTEBOOKS = ROOT / "notebooks"

    # Ensure necessary directories exist
    ensure_dir(RAW)
    ensure_dir(PROCESSED)
    ensure_dir(MODELS)
    ensure_dir(REPORTS)
    ensure_dir(NOTEBOOKS)

    # Import libraries for modeling, optimization, and data handling
    import sqlalchemy
    import pandas as pd
    import numpy as np
    import optuna
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    import xgboost as xgb
    import joblib
    import warnings
    warnings.filterwarnings("ignore")

    # Connect to SQLite database
    DB_PATH = DATA_DIR / "sales_data.db"
    engine = sqlalchemy.create_engine(f"sqlite:///{DB_PATH}")

    # Week 6 – Hyperparameter Tuning (Random Forest + XGBoost)
    # Load the prepared monthly feature dataset
    monthly = pd.read_sql("SELECT * FROM monthly_sales_features", engine, parse_dates=["month"]).dropna()
    print(" Loaded dataset from DB:", monthly.shape)

    # Select model features and target variable
    features = [
        c for c in monthly.columns
        if "lag" in c or "roll" in c or c in ["oil_price", "holiday_count", "month_of_year"]
    ]
    target = "sales"

    # Split the dataset into training and validation portions
    X = monthly[features]
    y = monthly[target]
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    # Define RMSE metric for evaluation
    rmse = lambda y, yhat: np.sqrt(mean_squared_error(y, yhat))

    # Random Forest Hyperparameter Optimization using Optuna
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
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return rmse(y_val, preds)

    print("\n Tuning Random Forest with Optuna...")
    rf_study = optuna.create_study(
        direction="minimize",
        study_name="rf_study",
        storage=f"sqlite:///{ROOT}/optuna_studies.db",
        load_if_exists=True,
    )
    rf_study.optimize(rf_objective, n_trials=20)
    print(" Best RF Params:", rf_study.best_params)
    print(" Best RF RMSE:", rf_study.best_value)

    # Train and save the best Random Forest model
    best_rf = RandomForestRegressor(**rf_study.best_params, random_state=42, n_jobs=-1)
    best_rf.fit(X_train, y_train)
    joblib.dump(best_rf, MODELS / "best_random_forest_week6.pkl")

    # XGBoost Hyperparameter Optimization using Optuna
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
        load_if_exists=True,
    )
    xgb_study.optimize(xgb_objective, n_trials=20)
    print(" Best XGB Params:", xgb_study.best_params)
    print(" Best XGB RMSE:", xgb_study.best_value)

    # Train and save the best XGBoost model
    best_xgb = xgb.train(xgb_study.best_params, xgb.DMatrix(X, label=y), num_boost_round=200)
    best_xgb.save_model(str(MODELS / "best_xgboost_week6.json"))

    # Display completion message
    print("\n Week 6 complete – tuned models saved to /models/")
    logger.info("Week 6 complete – Optuna tuning finished and models saved.")
