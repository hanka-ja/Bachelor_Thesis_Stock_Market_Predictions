import os

# Limit OS-level process affinity to exclusively use Core 0.
# This strictly physically constraints Polars, PyTorch, and everything else in this notebook to 1 core.
os.sched_setaffinity(0, {0,1})

# Also keep your GPU and general thread limits just in case
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["POLARS_MAX_THREADS"] = "2" # Must be set before 'import polars'

from mlp_lstm import DataPreparation, train_lstm_model
import torch
import polars as pl
import optuna
# We can turn off Lightning's progress bar for Optuna to keep the console clean
import warnings
import json
from datetime import datetime, date
warnings.filterwarnings("ignore", ".*does not have many workers.*")

#print(f"PyTorch Version: {torch.__version__}")
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

# 2. Define the Optuna Objective
def objective(trial):
    # Let Optuna pick the dataset type dynamically!
    mode = trial.suggest_categorical("mode", [
        'price', 'tfidf', 'finbert_sent', 'finbert_emb', 
        'tfidf_hybrid', 'finbert_sent_hybrid', 'finbert_emb_hybrid'
    ]) 

    # Rebuild tensors for the LSTM sequence logic
    df_master, X_tensor, y_tensor = DataPrepObject.get_lstm_tensors(mode=mode, seq_length=5)
    
    # PyTorch Architectural Hyperparameters
    hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 3)    
    dropout = trial.suggest_float("dropout", 0.3, 0.6)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)    
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    
    # Split Data
    train_loader, val_loader, test_loader, num_features = DataPrepObject.split_lstm_data(
        X_tensor, y_tensor, 
        train_ratio=0.60, 
        val_ratio=0.15,   
        batch_size=batch_size
    )
    
    # Train Model (Keep verbose=False so it doesn't spam your screen)
    model, trainer = train_lstm_model(
        train_loader=train_loader, 
        val_loader=val_loader, 
        num_features=num_features, 
        hidden_size=hidden_size,
        num_layers=num_layers,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        dropout=dropout,
        max_epochs=150, # Early stopping will catch it if it plateaus
        verbose=False   
    )
    
    best_val_acc = trainer.callbacks[0].best_score.item() 
    
    # Grab the final recorded metrics
    val_auc = trainer.callback_metrics.get("val_auroc", torch.tensor(0.0)).item()
    val_loss = trainer.callback_metrics.get("val_loss", torch.tensor(1.0)).item()
    
    # Save the AUROC as a hidden user attribute so we can review it later
    trial.set_user_attr("val_auc", val_auc)
    trial.set_user_attr("val_loss", val_loss)
    
    # Tell Optuna to optimize for Validation Accuracy
    return best_val_acc

# 3. Create Persistent SQLite Study
# The magic trick: `load_if_exists=True` lets you stop and restart the notebook anytime
study_name = "universal_stock_v3"
storage_name = "sqlite:///optuna_universal_stock.db"

study = optuna.create_study(
    study_name=study_name, 
    storage=storage_name, 
    load_if_exists=True, 
    direction="maximize"
)

print(f"Starting sweep. Baseline to beat is 0.5211 (Always Up).")
# Set n_trials high. When you want a break, just hit the "Interrupt" stop button in VS Code.
study.optimize(objective, n_trials=400) #400

best_trial = study.best_trial

# Create a dictionary of the results
results = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_type": "LSTM",
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

# # 1. Prepare Data (Outside the loop so it only loads once!)
# DataPrepObject = DataPreparation()
# DataPrepObject.load_and_prepare_price_data(df_prices) 
# DataPrepObject.load_tfidf_data(df_tfidf, n_components=100) 
# DataPrepObject.load_finbert_sentiment_data(df_sent) 
# DataPrepObject.load_finbert_embeddings_data(df_emb, n_components=30) 

# def objective(trial):
#     # 1. Expand the Search Space
#     seq_len = trial.suggest_categorical("seq_length", [3, 5, 10, 14, 21]) # Added 3 and 14
#     hidden = trial.suggest_categorical("hidden_size", [16, 32, 64, 128]) # Added 128
#     drop = trial.suggest_float("dropout", 0.1, 0.6, step=0.1) # Expanded to 0.6
#     lr = trial.suggest_float("learning_rate", 5e-5, 2e-2, log=True) # Wider learning rate bounds
    
#     # --- PHASE 2 UNLOCKED PARAMETERS ---
#     batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
#     num_layers = trial.suggest_int("num_layers", 1, 3)
#     weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

#     # Let Optuna pick the dataset type!
#     # (Uncomment the others once you load their data above)
#     mode = trial.suggest_categorical("mode", [
#         'price', 
#         'tfidf', 
#         'finbert_sent', 
#         'finbert_emb', 
#         'tfidf_hybrid',
#         'finbert_sent_hybrid',
#         'finbert_emb_hybrid'
#     ]) 

#     # 2. Get data for this trial's specific mode and sequence length
#     try:
#         _, X_t, y_t = DataPrepObject.get_lstm_tensors(mode=mode, seq_length=seq_len)
#         train_loader, val_loader, _, num_features = DataPrepObject.split_lstm_data(X_t, y_t, batch_size=batch_size)
#     except Exception as e:
#         # If a mode fails because data wasn't loaded, tell Optuna to prune this trial
#         raise optuna.TrialPruned()
    
#     #df_m, X_t, y_t = DataPrepObject.get_lstm_tensors(mode=mode, seq_length=seq_len)
#     #train_loader, val_loader, test_loader, num_features = DataPrepObject.split_lstm_data(X_t, y_t)

#     # 3. Call your centralized function silently
#     # (Requires adding verbose=False capability to mlp_lstm.py as discussed previously)
#     model, trainer = train_lstm_model(
#         train_loader=train_loader, 
#         val_loader=val_loader, 
#         num_features=num_features, 
#         hidden_size=hidden,
#         num_layers=num_layers,
#         weight_decay=weight_decay,
#         learning_rate=lr,
#         dropout=drop,
#         max_epochs=120,
#         verbose=False   
#     )
    
#     # 4. Extract all metrics safely
#     # Grab the BEST accuracy recorded by the EarlyStopper
#     val_acc = trainer.callbacks[0].best_score.item() 
    
#     val_auc = trainer.callback_metrics.get("val_auroc", torch.tensor(0.0)).item()
#     val_loss = trainer.callback_metrics.get("val_loss", torch.tensor(1.0)).item()
    
#     # 5. Save the extra metrics for our own records
#     trial.set_user_attr("val_auc", val_auc)
#     trial.set_user_attr("val_loss", val_loss)
    
#     # 6. Return the main metric for Optuna to maximize
#     return val_acc

# study = optuna.create_study(
#     direction="maximize", 
#     study_name="LSTM_DeepSweep_v2",
#     storage="sqlite:///optuna_lstm_v2.db", # <-- MAGIC LINE
#     load_if_exists=True                 # <-- MAGIC LINE 2 (resumes if interrupted!)
# )
# #study.optimize(objective, n_trials=1000)
# try:
#     print("\nStarting optimization. Press Ctrl+C at any time to stop and save results!")
#     study.optimize(objective, n_trials=600, timeout=3600) # Stop after 1 hour even if not all trials are done
# except KeyboardInterrupt:
#     print("\nOptimization manually interrupted by user! Fetching the best results so far...")


# best_trial = study.best_trial

# # Create a dictionary of the results
# results = {
#     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#     "model_type": "LSTM",
#     "best_accuracy": best_trial.value,
#     "best_val_auc": best_trial.user_attrs.get('val_auc', None),
#     "best_val_loss": best_trial.user_attrs.get('val_loss', None),
#     "hyperparameters": best_trial.params
# }

# # Append to a JSONL file (so you don't overwrite previous nights' runs)
# try:
#     with open("optimal_parameters.jsonl", "a") as f:
#         json.dump(results, f)
#         f.write("\n")
#     print("\nResults successfully saved to optimal_parameters.jsonl")
# except Exception as e:
#     print(f"\nFailed to save JSON: {e}")

# print("\n" + "="*50)
# print("OPTUNA QUICK TEST FINISHED")
# print("="*50)

# # Pulling out both the optimization target AND the secret tracked metrics
# best_trial = study.best_trial
# print(f"Best Accuracy Achieved: {best_trial.value:.4f}")
# print(f"Associated AUROC:       {best_trial.user_attrs['val_auc']:.4f}")
# print(f"Associated Loss:        {best_trial.user_attrs['val_loss']:.4f}")

# print("\nBest Parameters:")
# for key, value in best_trial.params.items():
#     print(f"    {key}: {value}")