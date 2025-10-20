from src.utils import get_project_root, ensure_dir, setup_logger

def run():
    # Initialize a logger for Week 3
    logger = setup_logger("WEEK3")

    # Define main directories
    ROOT = get_project_root()
    DATA_DIR = ROOT / "data"
    RAW = DATA_DIR / "raw"
    PROCESSED = DATA_DIR / "processed"
    MODELS = ROOT / "models"
    REPORTS = ROOT / "reports"
    NOTEBOOKS = ROOT / "notebooks"

    # Ensure that required folders exist
    ensure_dir(RAW)
    ensure_dir(PROCESSED)
    ensure_dir(MODELS)
    ensure_dir(REPORTS)
    ensure_dir(NOTEBOOKS)

    # Import dependencies
    import sqlalchemy
    import pandas as pd
    import numpy as np
    from sqlalchemy import text

    # Define database path and output folder
    DB_PATH = DATA_DIR / "sales_data.db"
    OUT_DIR = PROCESSED
    engine = sqlalchemy.create_engine(f"sqlite:///{DB_PATH}")

    # Week 3 - Feature Engineering
    # Create aggregated tables at a monthly level for modeling
    with engine.begin() as conn:
        # Aggregate monthly sales per store and family
        conn.execute(text("DROP TABLE IF EXISTS monthly_sales"))
        conn.execute(text("""
            CREATE TABLE monthly_sales AS
            SELECT store_nbr, family,
                   date(substr(date,1,7)||'-01') as month,
                   SUM(sales) as sales
            FROM train_clean
            GROUP BY store_nbr, family, month
        """))

        # Aggregate monthly average oil price
        conn.execute(text("DROP TABLE IF EXISTS monthly_oil"))
        conn.execute(text("""
            CREATE TABLE monthly_oil AS
            SELECT date(substr(date,1,7)||'-01') as month,
                   AVG(oil_price) as oil_price
            FROM oil_clean
            GROUP BY month
        """))

        # Aggregate monthly holiday counts
        conn.execute(text("DROP TABLE IF EXISTS monthly_holidays"))
        conn.execute(text("""
            CREATE TABLE monthly_holidays AS
            SELECT date(substr(date,1,7)||'-01') as month,
                   COUNT(*) as holiday_count
            FROM holidays_clean
            GROUP BY month
        """))

        # Aggregate monthly transactions per store
        conn.execute(text("DROP TABLE IF EXISTS monthly_transactions"))
        conn.execute(text("""
            CREATE TABLE monthly_transactions AS
            SELECT store_nbr,
                   date(substr(date,1,7)||'-01') as month,
                   SUM(transactions) as transactions
            FROM transactions_clean
            GROUP BY store_nbr, month
        """))

    # Merge all monthly tables into one dataset
    monthly = pd.read_sql("""
        SELECT m.store_nbr, m.family, m.month, m.sales,
               s.city, s.state, s.type as store_type, s.cluster,
               o.oil_price, h.holiday_count, t.transactions
        FROM monthly_sales m
        LEFT JOIN stores_clean s ON m.store_nbr = s.store_nbr
        LEFT JOIN monthly_oil o ON m.month = o.month
        LEFT JOIN monthly_holidays h ON m.month = h.month
        LEFT JOIN monthly_transactions t ON m.store_nbr = t.store_nbr AND m.month = t.month
    """, engine, parse_dates=['month'])

    # Add time-based features
    monthly = monthly.sort_values(['store_nbr', 'family', 'month'])
    monthly['month_of_year'] = monthly['month'].dt.month
    monthly['year'] = monthly['month'].dt.year

    # Create lag features for past sales trends
    for lag in [1, 2, 3, 6, 12]:
        monthly[f"sales_lag_{lag}"] = monthly.groupby(['store_nbr', 'family'])['sales'].shift(lag)

    # Create rolling averages to smooth sales patterns
    for r in [3, 6]:
        monthly[f"sales_roll_{r}"] = (
            monthly.groupby(['store_nbr', 'family'])['sales']
            .transform(lambda x: x.shift(1).rolling(r).mean())
        )

    # Handle missing values to maintain data consistency
    monthly['oil_price'] = monthly['oil_price'].ffill().bfill()
    monthly['holiday_count'] = monthly['holiday_count'].fillna(0)
    monthly['transactions'] = monthly['transactions'].fillna(0)

    # Save processed dataset to local directory
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        OUT_PATH = OUT_DIR / "monthly_sales_features.parquet"
        monthly.to_parquet(OUT_PATH, index=False)
        print(" Saved dataset to Parquet:", OUT_PATH)
    except Exception as e:
        OUT_PATH = OUT_DIR / "monthly_sales_features.csv"
        monthly.to_csv(OUT_PATH, index=False)
        print(" Parquet save failed. Saved as CSV instead:", OUT_PATH)
        print("Error:", e)

    # Store the final dataset in the database for next steps
    monthly.to_sql("monthly_sales_features", engine, if_exists="replace", index=False)
    print(" Saved dataset to database table: monthly_sales_features")

    # Display completion info
    print("\n Week 3 complete")
    print("Shape:", monthly.shape)
    print("Columns:", monthly.columns.tolist())
