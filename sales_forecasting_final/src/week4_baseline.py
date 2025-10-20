from src.utils import get_project_root, ensure_dir, setup_logger


def run():
    # Initialize a logger for Week 4
    logger = setup_logger("WEEK4")

    # Define main directory paths
    ROOT = get_project_root()
    DATA_DIR = ROOT / "data"
    RAW = DATA_DIR / "raw"
    PROCESSED = DATA_DIR / "processed"
    MODELS = ROOT / "models"
    REPORTS = ROOT / "reports"
    NOTEBOOKS = ROOT / "notebooks"

    # Ensure all required directories exist
    ensure_dir(RAW)
    ensure_dir(PROCESSED)
    ensure_dir(MODELS)
    ensure_dir(REPORTS)
    ensure_dir(NOTEBOOKS)

    # Import libraries for analysis and modeling
    import sqlalchemy
    from sqlalchemy import text
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from prophet import Prophet
    import warnings
    warnings.filterwarnings("ignore")

    # Connect to SQLite database
    DB_PATH = DATA_DIR / "sales_data.db"
    engine = sqlalchemy.create_engine(f"sqlite:///{DB_PATH}")

    # Week 4 – Exploratory Data Analysis (EDA) and Baseline Models
    # Load the preprocessed dataset from the database
    monthly = pd.read_sql("SELECT * FROM monthly_sales_features", engine, parse_dates=["month"])
    print("Loaded monthly features:", monthly.shape)

    # Visualize total monthly sales trend over time
    agg = monthly.groupby("month")["sales"].sum().reset_index()
    plt.figure(figsize=(12, 4))
    sns.lineplot(data=agg, x="month", y="sales")
    plt.title("Total Monthly Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Boxplot to visualize seasonality in monthly sales
    plt.figure(figsize=(10, 5))
    sns.boxplot(
        x="month_of_year",
        y="sales",
        data=monthly.sample(min(5000, len(monthly)), random_state=42)
    )
    plt.title("Seasonality by Month")
    plt.tight_layout()
    plt.show()

    # Establish a naive baseline model using lag-12 sales (same month last year)
    monthly["pred_naive"] = monthly["sales_lag_12"]
    cutoff = monthly["month"].max() - pd.DateOffset(months=6)
    val = monthly[monthly["month"] > cutoff]

    # Define RMSE (Root Mean Squared Error) for evaluation
    rmse = lambda y, yhat: np.sqrt(((y - yhat) ** 2).mean())
    print("Naive lag-12 RMSE:", rmse(val["sales"], val["pred_naive"]))

    # Fit ARIMA model on one sample time series (store 1, one product family)
    sample = monthly[monthly["store_nbr"] == 1].copy()
    sample = sample[sample["family"] == sample["family"].unique()[0]].set_index("month").asfreq("MS")
    series = sample["sales"].fillna(0)
    model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    res = model.fit(disp=False)
    print("ARIMA forecast (6 months):\n", res.forecast(6))

    # Build a Prophet model for overall sales forecasting
    agg_sales = (
        monthly.groupby("month")["sales"]
        .sum()
        .reset_index()
        .rename(columns={"month": "ds", "sales": "y"})
    )
    m = Prophet(yearly_seasonality=True)
    m.fit(agg_sales)
    future = m.make_future_dataframe(periods=6, freq="MS")
    fcst = m.predict(future)
    print("Prophet tail:\n", fcst[["ds", "yhat"]].tail())

    # Save baseline metrics (Naive model RMSE)
    metrics_path = PROCESSED / "baseline_metrics.csv"
    pd.DataFrame(
        [{"model": "naive_lag12", "rmse": rmse(val["sales"], val["pred_naive"])}]
    ).to_csv(metrics_path, index=False)

    print("Week 4 complete. Metrics saved to:", metrics_path)
    logger.info("Week 4 complete")
