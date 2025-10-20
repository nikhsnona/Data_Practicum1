from src.utils import get_project_root, ensure_dir, setup_logger


def run():
    # Initialize a logger specifically for Week 1
    logger = setup_logger("WEEK1")

    # Define key directory paths relative to the project root
    ROOT = get_project_root()
    DATA_DIR = ROOT / "data"
    RAW = DATA_DIR / "raw"
    PROCESSED = DATA_DIR / "processed"
    MODELS = ROOT / "models"
    REPORTS = ROOT / "reports"
    NOTEBOOKS = ROOT / "notebooks"

    # Ensure that all necessary directories exist (create them if missing)
    ensure_dir(RAW)
    ensure_dir(PROCESSED)
    ensure_dir(MODELS)
    ensure_dir(REPORTS)
    ensure_dir(NOTEBOOKS)

    # Create a SQLite database connection for storing and managing data
    import sqlalchemy
    engine = sqlalchemy.create_engine(f"sqlite:///{DATA_DIR / 'sales_data.db'}")

    # Create essential folder structure for the forecasting project
    dirs = [RAW, PROCESSED, MODELS, REPORTS, NOTEBOOKS, ROOT / "src"]
    for d in dirs:
        # Create each directory (including parents if missing)
        d.mkdir(parents=True, exist_ok=True)
        # Add a .gitkeep file so empty folders can be tracked by Git
        (d / ".gitkeep").write_text("")

    # Initialize Python package structure for source code
    init_file = ROOT / "src" / "__init__.py"
    init_file.write_text("# src package initializer\n")

    # Generate a minimal README to describe the project
    README = ROOT / "README.md"
    README.write_text(
        """# Multi-Source Sales Forecasting

This project forecasts sales using the Kaggle dataset 
**Store Sales - Time Series Forecasting**. It integrates multiple sources 
(sales, stores, oil, holidays, transactions) into a single forecasting pipeline.

## Project Structure
- data/raw        : Raw Kaggle CSVs
- data/processed  : Cleaned & feature-engineered datasets
- models/         : Trained model artifacts
- reports/        : Outputs, figures, evaluation
- notebooks/      : Jupyter notebooks for weekly progress
- src/            : Reusable Python modules (data, features, models, utils)

## How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Run ETL and feature engineering via `main.py`
3. Explore models and results in `notebooks/`
"""
    )

    # Final confirmation message and log entry
    logger.info(f"Project structure created under: {ROOT}")
    print(f" Project structure created under: {ROOT}")
