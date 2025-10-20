from src.utils import get_project_root, ensure_dir, setup_logger


def run():
    # Initialize a logger for Week 2
    logger = setup_logger("WEEK2")

    # Define main project directories
    ROOT = get_project_root()
    DATA_DIR = ROOT / "data"
    RAW = DATA_DIR / "raw"
    PROCESSED = DATA_DIR / "processed"
    MODELS = ROOT / "models"
    REPORTS = ROOT / "reports"
    NOTEBOOKS = ROOT / "notebooks"

    # Ensure that the key directories exist
    ensure_dir(RAW)
    ensure_dir(PROCESSED)
    ensure_dir(MODELS)
    ensure_dir(REPORTS)
    ensure_dir(NOTEBOOKS)

    # Import required libraries
    import sqlalchemy
    from sqlalchemy import text
    import pandas as pd, numpy as np
    import matplotlib.pyplot as plt, seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")

    # Define database path and create connection engine
    DB_PATH = DATA_DIR / "sales_data.db"
    engine = sqlalchemy.create_engine(f"sqlite:///{DB_PATH}")

    # Week 2: ETL, cleaning, and initial EDA
    # Load raw CSVs, clean them, and write back to SQLite

    # Define input files
    files = {
        "train": "train.csv",
        "stores": "stores.csv",
        "oil": "oil.csv",
        "holidays": "holidays_events.csv",
        "transactions": "transactions.csv"
    }

    # Step 1: Load raw CSVs into database
    for tbl, fname in files.items():
        path = RAW / fname
        if path.exists():
            print(f"Loading {fname} -> {tbl}")
            df = pd.read_csv(path, low_memory=False)
            df.to_sql(tbl, engine, if_exists="replace", index=False)
        else:
            print(f" Missing raw file: {path}")

    # Step 2: Create cleaned versions of all tables
    with engine.begin() as conn:
        # Clean training data
        conn.execute(text("DROP TABLE IF EXISTS train_clean"))
        conn.execute(text("""
            CREATE TABLE train_clean AS
            SELECT
                CAST(id AS INTEGER) AS id,
                date(date) AS date,
                CAST(store_nbr AS INTEGER) AS store_nbr,
                family,
                CAST(sales AS REAL) AS sales,
                CAST(onpromotion AS INTEGER) AS onpromotion
            FROM train
            WHERE sales IS NOT NULL AND sales >= 0
        """))

        # Clean oil data
        conn.execute(text("DROP TABLE IF EXISTS oil_clean"))
        conn.execute(text("""
            CREATE TABLE oil_clean AS
            SELECT date(date) AS date,
                   CAST(dcoilwtico AS REAL) AS oil_price
            FROM oil
        """))

        # Clean holiday events data
        conn.execute(text("DROP TABLE IF EXISTS holidays_clean"))
        conn.execute(text("""
            CREATE TABLE holidays_clean AS
            SELECT date(date) AS date,
                   type,
                   locale,
                   locale_name,
                   description,
                   CASE WHEN lower(transferred) IN ('true','1','t') THEN 1 ELSE 0 END AS transferred
            FROM holidays
        """))

        # Clean transactions data
        conn.execute(text("DROP TABLE IF EXISTS transactions_clean"))
        conn.execute(text("""
            CREATE TABLE transactions_clean AS
            SELECT date(date) AS date,
                   CAST(store_nbr AS INTEGER) AS store_nbr,
                   CAST(transactions AS INTEGER) AS transactions
            FROM transactions
            WHERE transactions IS NOT NULL
        """))

        # Clean store metadata
        conn.execute(text("DROP TABLE IF EXISTS stores_clean"))
        conn.execute(text("""
            CREATE TABLE stores_clean AS
            SELECT DISTINCT
                CAST(store_nbr AS INTEGER) AS store_nbr,
                city, state, type, CAST(cluster AS INTEGER) AS cluster
            FROM stores
        """))

    print("Base clean tables created in DB")

    # Step 3: Interpolate oil prices to ensure daily coverage
    oil = pd.read_sql("SELECT * FROM oil_clean", engine, parse_dates=['date'])
    oil = oil.set_index('date').asfreq('D')
    oil['oil_price'] = oil['oil_price'].interpolate().bfill().ffill()
    oil = oil.reset_index()
    oil.to_sql("oil_clean", engine, if_exists="replace", index=False)
    print("Oil interpolated to daily and saved to DB")

    # Step 4: Clean holidays by removing transferred entries
    hol = pd.read_sql("SELECT * FROM holidays_clean", engine, parse_dates=['date'])
    hol['transferred'] = hol['transferred'].astype(int)
    hol = hol[hol['transferred'] == 0].copy()
    hol['type'] = hol['type'].str.strip().str.upper()
    hol.to_sql("holidays_clean", engine, if_exists="replace", index=False)
    print("Holidays filtered and saved to DB")

    # Step 5: Add promotional flags and counts to training data
    train = pd.read_sql("SELECT * FROM train_clean", engine, parse_dates=['date'])
    train['promo_count'] = train['onpromotion'].fillna(0).astype(int)
    train['promo_flag'] = (train['promo_count'] > 0).astype(int)
    train.to_sql("train_clean", engine, if_exists="replace", index=False)
    print("Train promo fields added and saved to DB")

    # Step 6: Quick exploratory data analysis and sample plots
    print("\nQuick summaries:")
    print("Train rows:", len(train))
    print("Zero sales ratio:", (train['sales'] == 0).mean())

    # Monthly sales trend
    monthly_total = train.groupby(pd.Grouper(key='date', freq='M'))['sales'].sum()
    plt.figure(figsize=(10, 4))
    plt.plot(monthly_total.index, monthly_total.values)
    plt.title("Monthly total sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Compare sales with and without promotions
    print("\nAverage sales with vs without promo:")
    print(train.groupby('promo_flag')['sales'].mean())

    print("\nWeek 2 complete: cleaned tables written to DB at", DB_PATH)
