# Multi-Source Sales Forecasting

This project forecasts sales using the Kaggle dataset 
**Store Sales – Time Series Forecasting**. It integrates multiple sources 
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
