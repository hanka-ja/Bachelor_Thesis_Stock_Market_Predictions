import os
# Limit OS-level process affinity to exclusively use Core 0.
# This strictly physically constraints Polars, PyTorch, and everything else in this notebook to 1 core.
os.sched_setaffinity(0, {2,3})

# Also keep your GPU and general thread limits just in case
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["POLARS_MAX_THREADS"] = "2" # Must be set before 'import polars'

import torch
import pyarrow
import warnings
import optuna
import polars as pl
import lightning as L
from datetime import date, datetime
import json
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from mlp_lstm import DataPreparation, MultiLayerPerceptron

warnings.filterwarnings("ignore", ".*does not have many workers.*")
torch.set_float32_matmul_precision('medium')

df_tfidf = (
    pl.scan_parquet('/mnt/windows/windows_hanka_bcthesis/full_news/tfidf_nasdaq.parquet')
    # Use whatever the date column is actually called in this file
    .filter(pl.col("trading_session_date_utc").is_between(pl.date(2006, 10, 20), pl.date(2019, 12, 31)))
    .collect()
)

# Use scan_parquet() -> filter() -> collect()
df_sent = (
    pl.scan_parquet('/mnt/red/red_hanka_bcthesis/full_news/finbert_nasdaq_2006-2023_avg_sentiment.parquet')
    # Use whatever the date column is actually called in this file
    .filter(pl.col("trading_session_date_utc").is_between(pl.date(2006, 10, 20), pl.date(2019, 12, 31)))
    .collect()
)

df_emb = (
    pl.scan_parquet('/mnt/red/red_hanka_bcthesis/full_news/finbert_nasdaq_2006-2023_avg_embeddings.parquet')
    # Use whatever the date column is actually called in this file
    .filter(pl.col("trading_session_date_utc").is_between(pl.date(2006, 10, 20), pl.date(2019, 12, 31)))
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
DataPrepObject.load_and_prepare_multiple_price_data(dict_of_dfs, start_date=date(2006, 10, 20), end_date=date(2019, 12, 31)) 
DataPrepObject.load_finbert_embeddings_data(df_emb, n_components=60)
DataPrepObject.load_tfidf_data(df_tfidf, n_components=100)
DataPrepObject.load_finbert_sentiment_data(df_sent)

# 3. Defining the Optuna Trial
def objective(trial):
    # Let Optuna pick the dataset type dynamically!
    mode = trial.suggest_categorical("mode", [
        'price', 'tfidf', 'finbert_sent', 'finbert_emb', 
        'tfidf_hybrid', 'finbert_sent_hybrid', 'finbert_emb_hybrid'
    ]) 

    # Build the specific tensors for this trial's mode
    df_master, X_tensor, y_tensor = DataPrepObject.get_mlp_tensors(mode=mode)

    
    # MLP relies on wide networks natively, test bigger node counts
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.3, 0.6)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    
    # MLP Split Data - Note: batch_size is intentionally hardcoded safely at 64 in your class design
    train_loader, val_loader, test_loader, num_features = DataPrepObject.split_mlp_data(
        X_tensor, y_tensor, 
        train_ratio=0.60, 
        val_ratio=0.15,
        batch_size=batch_size
    )
    
    model = MultiLayerPerceptron(
        input_size=num_features,
        hidden_size=hidden_size,
        dropout_rate=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    early_stop = EarlyStopping(
        monitor="val_acc", 
        min_delta=0.00, 
        patience=30, 
        mode="max",
        verbose=False
    )
    
    trainer = L.Trainer(
        max_epochs=150,
        accelerator="gpu",   
        devices=[0], 
        callbacks=[early_stop],
        enable_progress_bar=False,
        logger=False # Keeps terminal fully clean overnight
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    best_val_acc = trainer.callbacks[0].best_score.item() 
    
    # Grab the final recorded metrics
    val_auc = trainer.callback_metrics.get("val_auroc", torch.tensor(0.0)).item()
    val_loss = trainer.callback_metrics.get("val_loss", torch.tensor(1.0)).item()
    
    # Save the AUROC as a hidden user attribute so we can review it later
    trial.set_user_attr("val_auc", val_auc)
    trial.set_user_attr("val_loss", val_loss)
    
    return best_val_acc

# 4. Spin up the Trial and Save to DB 
study_name = "universal_mlp_v1"
storage_name = "sqlite:///optuna_universal_mlp.db"

# Creates a SEPARATE dataset record in the same sqlite database!
study = optuna.create_study(
    study_name=study_name, 
    storage=storage_name, 
    load_if_exists=True, 
    direction="maximize"
)

print(f"Starting Multi-Layer Perceptron (MLP) Overnight Sweep...")
study.optimize(objective, n_trials=300) 
#study.optimize(objective, n_trials=300) 

best_trial = study.best_trial

# Create a dictionary of the results
results = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_type": "MLP",
    "best_accuracy": best_trial.value,
    "best_val_auc": best_trial.user_attrs.get('val_auc', None),
    "best_val_loss": best_trial.user_attrs.get('val_loss', None),
    "hyperparameters": best_trial.params,
    "stocks_used": list(dict_of_dfs.keys()),   
    "mode": best_trial.params["mode"]
}

# Append to a JSONL file (so you don't overwrite previous nights' runs)
try:
    with open("optimal_parameters.jsonl", "a") as f:
        json.dump(results, f)
        f.write("\n")
    print("\nResults successfully saved to optimal_parameters.jsonl")
except Exception as e:
    print(f"\nFailed to save JSON: {e}")

print("\n" + "="*50)
print("OPTUNA QUICK TEST FINISHED")
print("="*50)

# Pulling out both the optimization target AND the secret tracked metrics
best_trial = study.best_trial
print(f"Best Accuracy Achieved: {best_trial.value:.4f}")
print(f"Associated AUROC:       {best_trial.user_attrs['val_auc']:.4f}")
print(f"Associated Loss:        {best_trial.user_attrs['val_loss']:.4f}")

print("\nBest Parameters:")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")