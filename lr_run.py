import os
# Limit OS-level process affinity to exclusively use Core 0.
# This strictly physically constraints Polars, PyTorch, and everything else in this notebook to 1 core.
os.sched_setaffinity(0, {2,3})

# Also keep your GPU and general thread limits just in case
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["POLARS_MAX_THREADS"] = "2" # Must be set before 'import polars'

import polars as pl
import importlib
import lr_rf_cuml_based
from datetime import date, datetime
from lr_rf_cuml_based import mach_lern
import json

def main():
    df_aapl = pl.scan_csv('/mnt/windows/windows_hanka_bcthesis/full_stock_prices/AAPL.csv').with_columns(pl.col('date').str.to_date('%Y-%m-%d')).collect()
    df_msft = pl.scan_csv('/mnt/windows/windows_hanka_bcthesis/full_stock_prices/MSFT.csv').with_columns(pl.col('date').str.to_date('%Y-%m-%d')).collect()
    df_googl = pl.scan_csv('/mnt/windows/windows_hanka_bcthesis/full_stock_prices/GOOGL.csv').with_columns(pl.col('date').str.to_date('%Y-%m-%d')).collect()
    df_amzn = pl.scan_csv('/mnt/windows/windows_hanka_bcthesis/full_stock_prices/AMZN.csv').with_columns(pl.col('date').str.to_date('%Y-%m-%d')).collect()
    df_nvda = pl.scan_csv('/mnt/windows/windows_hanka_bcthesis/full_stock_prices/NVDA.csv').with_columns(pl.col('date').str.to_date('%Y-%m-%d')).collect() # Or FB.csv

    # Package them up
    dict_of_dfs = {
        "AAPL": df_aapl.select(['date', 'close']),
        "MSFT": df_msft.select(['date', 'close']),
        "GOOGL": df_googl.select(['date', 'close']),
        "AMZN": df_amzn.select(['date', 'close']),
        "NVDA": df_nvda.select(['date', 'close'])
    }

    df_tfidf = (
        pl.scan_parquet('/mnt/windows/windows_hanka_bcthesis/full_news/tfidf_nasdaq.parquet')
        # Use whatever the date column is actually called in this file
        .filter(pl.col("trading_session_date_utc").is_between(pl.date(2006, 10, 20), pl.date(2019, 12, 31)))
        .collect()
    )

    df_sent = (
        pl.scan_parquet('/mnt/red/red_hanka_bcthesis/full_news/finbert_nasdaq_2006-2023_avg_sentiment.parquet')
        # Use whatever the date column is actually called in this file
        .filter(pl.col("trading_session_date_utc").is_between(pl.date(2006, 10, 20), pl.date(2019, 12, 31)))
        .collect()
    )

    ml_object = mach_lern(test_size=0.3)

    ml_object.load_and_prepare_multiple_price_data(dict_of_dfs, start_date=date(2006, 10, 20), end_date=date(2019, 12, 31)) 
    ml_object.load_tfidf_data(df_tfidf)
    ml_object.load_finbert_data(df_sent)

    ml_object.start_dask_cluster()

    #for searching for optimal params
    #modes = ['price', 'tfidf', 'finbert', 'hybrid_tfidf', 'hybrid_finbert']
    # param_grid = {
    #     'logr__C': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
    #     'logr__l1_ratio': [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    #     "logr__class_weight": [None, "balanced"],
    #     "logr__tol": [1e-4, 1e-3],
    # }

    
    try:
        #searching for optimal parameters
        #for mode in modes:
        # Step 1: Load all optimal parameters
        with open("optimal_parameters.jsonl", "r") as f:
            lines = f.readlines()

        # Step 2: Filter for LR entries after 2026-03-10 21:00:00
        filtered_entries = []
        for line in lines:
            entry = json.loads(line)
            if entry.get("model_type") == "LR":
                ts = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")
                if ts >= datetime(2026, 3, 10, 21, 0, 0) and ts <= datetime(2026, 3, 10, 23, 59, 59):
                    filtered_entries.append(entry)

        # Step 3: Evaluate each set of parameters
        nmb = 1
        for entry in filtered_entries:
            mode = entry["mode"]
            param_grid = entry["hyperparameters"]
            param_grid = {k: [v] for k, v in param_grid.items()}
            acc = entry["best_cv_accuracy"]
            auc = entry["best_cv_auc"]
            print(f"\n{nmb }. Evaluating LR mode: {mode} with params: {param_grid}")
            print(f"\nTrain accuracy {acc} and AUC: {auc}")
            # You may need to modify train_logistic_regression to accept params
            # Convert all param values to lists for GridSearchCV

            best_pipeline, x_test, y_test, active_features = ml_object.train_logistic_regression(mode=mode, param_grid=param_grid)
            y_pred, y_probs = ml_object.evaluate(
                mode=mode, 
                best_pipeline=best_pipeline, 
                x_test=x_test, 
                y_test=y_test, 
                active_features=active_features
            )
            # Print or plot results as needed
            nmb += 1
    finally:
        ml_object.stop_dask_cluster()

if __name__ == "__main__":
    main()