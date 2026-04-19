import os

# # Limit OS-level process affinity to exclusively use Core 0.
# # This strictly physically constraints Polars, PyTorch, and everything else in this notebook to 1 core.
os.sched_setaffinity(0, {0,1})

# Also keep your GPU and general thread limits just in case
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["POLARS_MAX_THREADS"] = "2" # Must be set before 'import polars'

from mlp_lstm import DataPreparation, LightningLSTM, train_lstm_model
import optuna
import polars as pl
from datetime import datetime, date

df_tfidf = (
    pl.scan_parquet('/mnt/windows/windows_hanka_bcthesis/full_news/tfidf_nasdaq.parquet')
    # Use whatever the date column is actually called in this file
    .filter(pl.col("trading_session_date_utc").is_between(pl.date(2006, 10, 20), pl.date(2026, 12, 31)))
    .collect()
)

# Use scan_parquet() -> filter() -> collect()
df_sent = (
    pl.scan_parquet('/mnt/red/red_hanka_bcthesis/full_news/finbert_nasdaq_2006-2023_avg_sentiment.parquet')
    # Use whatever the date column is actually called in this file
    .filter(pl.col("trading_session_date_utc").is_between(pl.date(2006, 10, 20), pl.date(2026, 12, 31)))
    .collect()
)

df_emb = (
    pl.scan_parquet('/mnt/red/red_hanka_bcthesis/full_news/finbert_nasdaq_2006-2023_avg_embeddings.parquet')
    # Use whatever the date column is actually called in this file
    .filter(pl.col("trading_session_date_utc").is_between(pl.date(2006, 10, 20), pl.date(2026, 12, 31)))
    .collect()
)

# 1. Load the 5 individual stock dataframes
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

DataPrepObject = DataPreparation()
DataPrepObject.load_and_prepare_multiple_price_data(dict_of_dfs, start_date=date(2006, 10, 20), end_date=date(2026, 12, 31)) 
DataPrepObject.load_finbert_embeddings_data(df_emb, n_components=60)
DataPrepObject.load_tfidf_data(df_tfidf, n_components=100)
DataPrepObject.load_finbert_sentiment_data(df_sent)

# test 6th Optuna optimized run (LSTM) results from 26.3.
storage_name = "sqlite:///optuna_universal_stock.db"
study_name = "universal_stock_v4" # Change this if your 1000-trial study had a different name
study = optuna.load_study(study_name=study_name, storage=storage_name)

# Extract all trials into a DataFrame
df_trials = study.trials_dataframe()

# Filter out PRUNED or FAILED trials, keep only COMPLETE
df_completed = df_trials[df_trials['state'] == 'COMPLETE']

# Sort by best Validation Accuracy (descending) and grab Top 5
df_optuna_lstm_top = df_completed.sort_values(by='value', ascending=False).head(5)

# Clean up the output to only show metrics and parameters
cols_to_show = ['value'] + \
               [c for c in df_optuna_lstm_top.columns if c.startswith('params_')]

# Rename 'value' to 'val_acc' for readability
df_optuna_lstm_top = df_optuna_lstm_top[cols_to_show].rename(columns={'value': 'val_acc'})

# Display nicely in Jupyter
print('Top 5 LSTM Optuna Trials:')
print(df_optuna_lstm_top)

# test 6th Optuna optimized run (LSTM) results from 26.3.
# select top 5 models by validation accuracy and test them
top_optuna_lstm_models = df_optuna_lstm_top.head(5)

models_to_test = []
for idx, row in top_optuna_lstm_models.iterrows():
    models_to_test.append({
        "name": f"({row['val_acc']*100:.2f}%) - nmb {idx}",
        "hidden_size": row['params_hidden_size'],
        "dropout": row['params_dropout'],
        "learning_rate": row['params_learning_rate'],
        "weight_decay": row['params_weight_decay'],
        "batch_size": row['params_batch_size'],
        "mode": row['params_mode'],
        "num_layers": row['params_num_layers']
    })

results = []

for config in models_to_test:
    mode = config["mode"]
    run_name = f"trial {config['name']} | mode={mode}"
    print(f"\n{'='*80}\nOOS TESTING: {run_name}\n{'='*80}")
    
    # Fetch Tensors and Split (Reserves the chronological end for testing)
    df_master, X_tensor, y_tensor = DataPrepObject.get_lstm_tensors(mode=mode, seq_length=5)
    train_loader, val_loader, test_loader, num_features = DataPrepObject.split_lstm_data(
        X_tensor, y_tensor, 
        train_ratio=0.60, # ~2006-2013
        val_ratio=0.15,   # ~2014-2015
        batch_size=config["batch_size"]
    )
    
    # Train the Model
    model, trainer = train_lstm_model(
    train_loader=train_loader, 
    val_loader=val_loader, 
    num_features=num_features, 
    hidden_size=config["hidden_size"],
    num_layers=config["num_layers"],
    weight_decay=config["weight_decay"],
    learning_rate=config["learning_rate"],
    dropout=config["dropout"],
    max_epochs=120,
    verbose=False   # Keep console clean
    )

    # RUN THE OUT OF SAMPLE TEST! 
    print("Running strict Out-Of-Sample evaluation on untouched data...")
    test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False)[0]

    # Store the final truth
    results.append({
        "Run": run_name,
        "Test_Acc": test_metrics.get("test_acc", 0.0),
        "Test_AUC": test_metrics.get("test_auroc", 0.0),
        "Best_Val_Acc": trainer.callbacks[0].best_score.item()
    })

    # Print Leaderboard
print("\n" + "="*50)
print("FINAL OUT-OF-SAMPLE (OOS) LEADERBOARD")
print("="*50)
for r in sorted(results, key=lambda x: x["Test_Acc"], reverse=True):
    print(f"OOS Accuracy: {r['Test_Acc']:.4f} | OOS AUC: {r['Test_AUC']:.4f} | (Val Acc: {r['Best_Val_Acc']:.4f}) -> {r['Run']}")