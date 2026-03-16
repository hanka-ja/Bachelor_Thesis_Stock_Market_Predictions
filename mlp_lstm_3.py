import torch
import torch.nn as nn
import lightning as L
from torch.optim import Adam
import polars as pl
from torch.utils.data import TensorDataset, DataLoader
from IPython.display import display
import os
from sklearn.decomposition import PCA
import numpy as np
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC

#os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Make both the 3080 and 3070 visible
torch.set_float32_matmul_precision('medium')

class DataPreparation():
    def __init__(self):
        self.df_master = None
        # Initialize lists to track feature groups
        self.price_cols = []
        self.tfidf_cols = []
        self.finbert_sent_cols = []
        self.finbert_emb_cols = []  

    def _standardize_date(self, df, current_date_col_name):
        """Safely renames and formats the date column without touching TF-IDF word columns."""
        
        # Only rename if the column isn't already 'date_utc'
        if current_date_col_name != "trading_date_utc":
            df = df.rename({current_date_col_name: "trading_date_utc"})
            
        # Parse strings to dates
        if df["trading_date_utc"].dtype == pl.String:
            df = df.with_columns(pl.col("trading_date_utc").str.to_date("%Y-%m-%d"))
            
        return df.sort("trading_date_utc")    

    def load_and_prepare_price_data(self, df_prices, start_date=None, end_date=None):
        """
        Loads prices, creates Lags 1-5, and generates the Target variable.
        This forms the core timeline of df_master.
        """
        print("Loading prices, creating lags, and generating targets...")
        
        # Explicitly tell it to look for the lowercase 'date'
        df_prices = self._standardize_date(df_prices, "date")

        if start_date is not None and end_date is not None:
            df_prices = df_prices.filter(pl.col("trading_date_utc").is_between(start_date, end_date))

        # 2. Calculate return_lag0
        df_prices = df_prices.select([pl.col("trading_date_utc"), pl.col("close")])
        df_prices = df_prices.with_columns(pl.col("close").pct_change().alias("return_lag0"))       

        # 3. Create Target (Did price go UP tomorrow?)
        df_prices = df_prices.with_columns(
            (pl.col("return_lag0").shift(-1) > 0).cast(pl.Int8).alias("target_next_day")
        )        

        # 4. Create History (Lags 1-5)
        lags = 5
        lag_cols = ["return_lag0"] # Put return_lag0 back in the feature list
        for i in range(1, lags + 1):
            col_name = f"return_lag{i}"
            df_prices = df_prices.with_columns(
                pl.col("return_lag0").shift(i).alias(col_name)
            )
            lag_cols.append(col_name)

        self.price_cols = lag_cols

        # 5. Build Master DF
        cols_to_keep = ["trading_date_utc", "target_next_day"] + self.price_cols
        self.df_master = df_prices.select(cols_to_keep).drop_nulls()

        print(f"Price timeline established. Shape: {self.df_master.shape}")
        return self.df_master

    def load_and_prepare_multiple_price_data(self, dict_of_dfs, start_date=None, end_date=None):
        """
        Loads multiple stock dataframes (e.g., AAPL, TSLA).
        Calculates lags locally for each stock, and combines them vertically.
        """
        print(f"Loading prices for multiple stocks: {list(dict_of_dfs.keys())}...")
        all_dfs = []
        lag_cols = []
        
        for ticker, df in dict_of_dfs.items():
            df = self._standardize_date(df, "date")
            if start_date is not None and end_date is not None:
                df = df.filter(pl.col("trading_date_utc").is_between(start_date, end_date))

            # Calculate returns and target
            df = df.select([pl.col("trading_date_utc"), pl.col("close")])
            df = df.with_columns(pl.col("close").pct_change().alias("return_lag0"))       
            df = df.with_columns((pl.col("return_lag0").shift(-1) > 0).cast(pl.Int8).alias("target_next_day"))        

            # Create Lags
            lags = 5
            current_lag_cols = ["return_lag0"]
            for i in range(1, lags + 1):
                col_name = f"return_lag{i}"
                df = df.with_columns(pl.col("return_lag0").shift(i).alias(col_name))
                current_lag_cols.append(col_name)

            lag_cols = current_lag_cols 

            # NEW: Assign the ticker column so we know which stock is which
            df = df.with_columns(pl.lit(ticker).alias("ticker"))
            
            # Select final columns and clean nulls
            cols_to_keep = ["trading_date_utc", "ticker", "target_next_day"] + lag_cols
            df = df.select(cols_to_keep).drop_nulls()
            all_dfs.append(df)

        self.price_cols = lag_cols
        
        # Stack them all vertically. 
        self.df_master = pl.concat(all_dfs)
        print(f"Master multi-stock timeline established. Shape: {self.df_master.shape}")
        return self.df_master

    def load_tfidf_data(self, df_tfidf, n_components=100):
        """
        Loads TF-IDF features and joins them to the stock target.
        """
        if self.df_master is None: raise ValueError("Run load_and_prepare_price_data first!")
            
        print("Loading TF-IDF data and aligning with timeline...")
        df_tfidf = self._standardize_date(df_tfidf, "trading_session_date_utc")       
        
        raw_tfidf_cols = [c for c in df_tfidf.columns if c != "trading_date_utc"]
        
        # 2. Extract out just the numbers and replace any NaNs with 0 
        # (scikit-learn PCA cannot handle null/NaN values)
        X_tfidf_raw = df_tfidf.select(raw_tfidf_cols).fill_null(0).to_numpy()
        
        # 3. Apply PCA via scikit-learn
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_tfidf_raw)

        # 4. Create new column names for the PCA outputs (e.g., pca_0, pca_1, ...)
        pca_cols = [f"pca_{i}" for i in range(n_components)]
        self.tfidf_cols = pca_cols # Update the object's tracked feature names
        
        # 5. Convert the PCA numpy array back into a Polars DataFrame
        df_pca = pl.DataFrame(X_pca, schema=pca_cols)
        
        # 6. Re-attach the date column so we can join it to the main dataset
        df_pca = df_pca.with_columns(df_tfidf["trading_date_utc"])
        
        # 7. Join to the master dataframe (Left join on the price timeline)
        self.df_master = self.df_master.join(df_pca, on="trading_date_utc", how="left")
        
        # 8. Fill days where there was no news with 0s
        self.df_master = self.df_master.fill_null(0)
        
        print(f"TF-IDF aligned & PCA applied. Total columns in master: {len(self.df_master.columns)}")
        return self.df_master    
    
    def load_finbert_sentiment_data(self, df_finb_sent):
        """
        Loads FinBERT sentiment features and joins them to the stock timeline.
        """
        if self.df_master is None: 
            raise ValueError("Run load_and_prepare_price_data first!")
            
        print("Loading FinBERT sentiment data and aligning with timeline...")
        
        # Determine the date column name flexibly
        date_col = "trading_session_date_utc" if "trading_session_date_utc" in df_finb_sent.columns else "date"
        if date_col in df_finb_sent.columns:
            df_finb_sent = self._standardize_date(df_finb_sent, date_col)
            
        self.finbert_sent_cols = [c for c in df_finb_sent.columns if c != "trading_date_utc"]
        
        # Join to the master dataframe (Left join to preserve pure price timeline)
        self.df_master = self.df_master.join(df_finb_sent, on="trading_date_utc", how="left")
        
        # Fill days where there was no news with 0s (neutral sentiment)
        self.df_master = self.df_master.fill_null(0.0)
        
        print(f"FinBERT aligned. Total columns in master: {len(self.df_master.columns)}")
        return self.df_master

    def load_finbert_embeddings_data(self, df_finb_embeddings, n_components=30):
        """
        Loads 768-D FinBERT embeddings, applies PCA to prevent overfitting, 
        and joins them to the stock timeline.
        """
        if self.df_master is None: 
            raise ValueError("Run load_and_prepare_price_data first!")
            
        print(f"Loading FinBERT embeddings. Applying PCA to reduce to {n_components} dimensions...")
        
        # Standardize date
        date_col = "trading_session_date_utc" if "trading_session_date_utc" in df_finb_embeddings.columns else "date"
        if date_col in df_finb_embeddings.columns:
            df_finb_embeddings = self._standardize_date(df_finb_embeddings, date_col)
            
        # Isolate embedding columns
        emb_cols = [c for c in df_finb_embeddings.columns if c != "trading_date_utc"][0]
        
        # Fast, single-pass conversion to 2D NumPy array
        X_emb = np.array(df_finb_embeddings[emb_cols].to_list(), dtype=np.float32)
        print(f"Embeddings shape: {X_emb.shape}")  # Should be (n_rows, 768)
        
        # Safely cap n_components if your rows are fewer than n_components
        actual_components = min(n_components, X_emb.shape[0], X_emb.shape[1])
        print(f"PCA reducing to {actual_components} components...")

        # Apply PCA
        pca = PCA(n_components=actual_components, random_state=42)
        X_pca = pca.fit_transform(X_emb)
        
        # Create new DataFrame with PCA features
        pca_cols = [f"emb_pca_{i}" for i in range(actual_components)]
        self.finbert_emb_cols = pca_cols
        df_pca = pl.DataFrame(X_pca, schema=pca_cols)
        
        # Add date column back
        df_pca = df_pca.with_columns(df_finb_embeddings["trading_date_utc"])
        
        # Join to Master dataframe and fill NaNs with 0.0 (origin of PCA space)
        self.df_master = self.df_master.join(df_pca, on="trading_date_utc", how="left")
        self.df_master = self.df_master.fill_null(0.0)
        
        print(f"Embeddings PCA applied. Total cols in master: {len(self.df_master.columns)}")
        return self.df_master

    def get_mlp_tensors(self, mode):
        """
        Finalizes the data and converts to PyTorch Tensors.
        Call this AFTER loading all feature sets (Price + TFIDF + FinBERT).
        """
        #FIX: stacking logic for multiple stocks - follow get_lstm_tensors() approach to ensure we don't mix different stocks together in the same sequence.
        # Determine which features exist in df_master
        # all_cols = self.df_master.columns
        
        if self.df_master is None: raise ValueError("No data loaded!")
        # 1. Select active features based on mode
        active_features = []
        
        if mode == 'price':
            active_features = self.price_cols
        elif mode == 'tfidf':
            active_features = self.tfidf_cols
        elif mode == 'finbert_sent':
            active_features = self.finbert_sent_cols
        elif mode == 'finbert_emb':
            active_features = self.finbert_emb_cols
        elif mode == 'tfidf_hybrid':
            active_features = self.price_cols + self.tfidf_cols
        elif mode == 'finbert_sent_hybrid':
            active_features = self.price_cols + self.finbert_sent_cols
        elif mode == 'finbert_emb_hybrid':
            active_features = self.price_cols + self.finbert_emb_cols
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        # Validation
        if not active_features:
            raise ValueError(f"No features found for mode '{mode}'. Did you load the specific data?")

        #feature_cols = [c for c in all_cols if c not in ["trading_date_utc", "target_next_day"]]
        # Check if features actually exist in df_master (in case of partial loading)
        missing_cols = [c for c in active_features if c not in self.df_master.columns]
        if missing_cols:
             raise ValueError(f"Missing columns for mode '{mode}': {missing_cols[:5]}...")

        print(f"Preparing Tensors for mode '{mode}' with {len(active_features)} features...")
        
        # Extract Data
        # Sort by date first to ensure time order is correct before converting to numpy
        df_sorted = self.df_master.sort("trading_date_utc")

        # Extract Data
        X_data = df_sorted.select(active_features).to_numpy()
        y_data = df_sorted.select("target_next_day").to_numpy().flatten()

        # Convert to Float32 Tensors (Required for PyTorch)
        X_tensor = torch.tensor(X_data, dtype=torch.float32)
        y_tensor = torch.tensor(y_data, dtype=torch.float32)

        return self.df_master, X_tensor, y_tensor

    def get_lstm_tensors(self, mode, seq_length=5):
        """
        Retrieves data exactly like get_tensors(), but reshapes the price lags 
        into a 3D sequence (batch, sequence_length, features) for the LSTM.
        """
        if self.df_master is None: raise ValueError("No data loaded!")
        
        # 1. Define the absolute base price feature (IGNORE the fake lags)
        base_price = ["return_lag0"]

        # 2. Select active base features based on mode
        active_features = []
        if mode == 'price':
            active_features = base_price
        elif mode == 'tfidf':
            active_features = self.tfidf_cols
        elif mode == 'finbert_sent':
            active_features = self.finbert_sent_cols
        elif mode == 'finbert_emb':
            active_features = self.finbert_emb_cols
        elif mode == 'tfidf_hybrid':
            active_features = base_price + self.tfidf_cols
        elif mode == 'finbert_sent_hybrid':
            active_features = base_price + self.finbert_sent_cols
        elif mode == 'finbert_emb_hybrid':
            active_features = base_price + self.finbert_emb_cols
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        # Validation
        missing_cols = [c for c in active_features if c not in self.df_master.columns]
        if missing_cols:
             raise ValueError(f"Missing columns for mode '{mode}': {missing_cols[:5]}...")

        print(f"Applying real sliding window (Seq={seq_length}) for mode '{mode}' with {len(active_features)} features...")

        # Ensure we sort chronologically 
        df_sorted = self.df_master.sort("trading_date_utc")
        
        # Check if we have multiple stocks loaded
        has_ticker = "ticker" in df_sorted.columns
        tickers = df_sorted["ticker"].unique().to_list() if has_ticker else [None]

        X_seq, y_seq, date_seq = [], [], []
        
        # Step 1: Create sequences without mixing different stocks together!
        for ticker in tickers:
            # Filter to just one stock
            if has_ticker:
                df_ticker = df_sorted.filter(pl.col("ticker") == ticker)
            else:
                df_ticker = df_sorted
                
            data_2d = df_ticker.select(active_features).to_numpy()
            labels_1d = df_ticker.select("target_next_day").to_numpy().flatten()
            dates_1d = df_ticker.select("trading_date_utc").to_series().to_list()
            
            # Build sliding windows for this specific stock
            for i in range(len(data_2d) - seq_length + 1):
                window = data_2d[i : i + seq_length]
                target = labels_1d[i + seq_length - 1]
                target_date = dates_1d[i + seq_length - 1]
                
                X_seq.append(window)
                y_seq.append(target)
                date_seq.append(target_date)

        # Step 2: Now that sequences are safely built, sort EVERYTHING chronologically
        # This interleaves AAPL, TSLA, MSFT day-by-day so the model learns across the market chronologically.
        combined = sorted(zip(date_seq, X_seq, y_seq), key=lambda x: x[0])
        
        X_seq_sorted = [x for _, x, _ in combined]
        y_seq_sorted = [y for _, _, y in combined]

        # Convert to PyTorch Tensors
        X_tensor = torch.tensor(np.array(X_seq_sorted), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y_seq_sorted), dtype=torch.float32)     

        print(f"LSTM Tensors created. Shape: {X_tensor.shape} -> (Batch, Seq_Len, Features)")
        
        return self.df_master, X_tensor, y_tensor     

        # Sort chronologically
        # df_sorted = self.df_master.sort("trading_date_utc")
        
        # # 4. Extract into flat 2D arrays
        # data_2d = df_sorted.select(active_features).to_numpy()
        # labels_1d = df_sorted.select("target_next_day").to_numpy().flatten()
        
        # # 5. Build the Sliding Window Sequences
        # X_seq, y_seq = [], []
        
        # # We add + 1 to grab the exact very last window
        # for i in range(len(data_2d) - seq_length + 1):
        #     # The sequence is the window of days [i : i + seq_length]
        #     window = data_2d[i : i + seq_length]
            
        #     # The target is the label of the LAST day in this specific window
        #     # (Because 'target_next_day' on that last day represents the day AFTER the window)
        #     target = labels_1d[i + seq_length - 1]
            
        #     X_seq.append(window)
        #     y_seq.append(target)   

        # # 6. Convert to PyTorch Tensors
        # # Wrapping in np.array() first is much faster for PyTorch than direct list conversion
        # X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
        # y_tensor = torch.tensor(np.array(y_seq), dtype=torch.float32)     

        # print(f"LSTM Tensors created. Shape: {X_tensor.shape} -> (Batch, Seq_Len, Features)")
        
        # return self.df_master, X_tensor, y_tensor        

    def split_mlp_data(self, X_tensor, y_tensor, train_ratio=0.7, val_ratio=0.15, batch_size=64):
        # 1. Chronological Split (No Shuffling!)
        # We need to slice the tensors manually to respect time order.

        total_len = len(X_tensor)
        train_size = int(total_len * train_ratio)
        val_size = int(total_len * val_ratio)
        test_size = total_len - train_size - val_size

        # Slicing tensors maintains order (0 to 70%, 70% to 85%, 85% to 100%)
        X_train, y_train = X_tensor[:train_size], y_tensor[:train_size]
        X_val, y_val = X_tensor[train_size:train_size+val_size], y_tensor[train_size:train_size+val_size]
        X_test, y_test = X_tensor[train_size+val_size:], y_tensor[train_size+val_size:]

        # --- NEW SCALING LOGIC ---
        # Calculate mean and std ONLY on the training set to prevent data leakage
        X_mean = X_train.mean(dim=0, keepdim=True)
        X_std = X_train.std(dim=0, keepdim=True)

        # Apply the same scaler to all three sets
        X_train = (X_train - X_mean) / (X_std + 1e-7)
        X_val = (X_val - X_mean) / (X_std + 1e-7)
        X_test = (X_test - X_mean) / (X_std + 1e-7)

        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        # 2. Create Datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

        # 3. Create DataLoaders
        # shuffle=False is MANDATORY for Validation and Test to check performance correctly.
        # shuffle=True is *technically* allowed for Training data ONLY (because the order inside the training set doesn't matter as much as long as it doesn't see future data), 
        # BUT for strict financial time-series, keeping shuffle=False is often safer to preserve regime structures.
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=False)

        num_features = X_train.shape[1]

        return train_loader, val_loader, test_loader, num_features

    def split_lstm_data(self, X_tensor, y_tensor, train_ratio=0.7, val_ratio=0.15, batch_size=64):
        total_len = len(X_tensor)
        train_size = int(total_len * train_ratio)
        val_size = int(total_len * val_ratio)

        # Slice chronologically
        X_train, y_train = X_tensor[:train_size], y_tensor[:train_size]
        X_val, y_val = X_tensor[train_size:train_size+val_size], y_tensor[train_size:train_size+val_size]
        X_test, y_test = X_tensor[train_size+val_size:], y_tensor[train_size+val_size:]

        # For 3D Tensors (Batch, Seq, Feature), calculate mean/std across Batch AND Seq (dims 0 and 1)
        # This gives us the global mean/std of ALL historical returns in the train set.
        X_mean = X_train.mean(dim=(0, 1), keepdim=True)
        X_std = X_train.std(dim=(0, 1), keepdim=True)

        # Scale data
        X_train = (X_train - X_mean) / (X_std + 1e-7)
        X_val   = (X_val - X_mean) / (X_std + 1e-7)
        X_test  = (X_test - X_mean) / (X_std + 1e-7)

        # Create DataLoaders
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size= batch_size, shuffle=False, num_workers=0)
        val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size= batch_size, shuffle=False, num_workers=0)
        test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size= batch_size, shuffle=False, num_workers=0)

        # We return 1 (for feature size) so the LSTM model initializes correctly
        num_features = X_train.shape[2] 
        return train_loader, val_loader, test_loader, num_features

class MultiLayerPerceptron(L.LightningModule):
    def __init__(self, input_size, learning_rate=0.001, hidden_size=64, dropout_rate=0.2, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters() # Saves input_size and learning_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Instead of hardcoding w00, b00, etc., nn.Linear creates them automatically.
        # nn.Sequential runs the data through these layers in order.
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),           # Drops 20% of nodes to prevent overfitting
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size//2, 1)           # 1 output node (raw score/logit for Up or Down)
        )
        
        # BCEWithLogitsLoss is the standard loss function for binary classification (0 or 1)
        self.loss_fn = nn.BCEWithLogitsLoss()        

        # Initialize Metrics!
        # These stateful metrics automatically calculate totals at the end of every epoch.
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()
        self.train_auroc = BinaryAUROC()
        self.val_auroc = BinaryAUROC()
        self.test_auroc = BinaryAUROC()

        self.example_input_array = torch.zeros(1, input_size) 

    def forward(self, x):
        # x will be your batch of returns (and later, FinBERT embeddings)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        
        # 1. Run data through the model
        logits = self.forward(inputs).squeeze(dim=1) # squeeze() ensures shape matches labels        
        # 2. Calculate loss
        loss = self.loss_fn(logits, labels.float())
        
       # Calculate metrics using sigmoid probabilities
        preds = torch.sigmoid(logits)
        self.train_acc(preds, labels)
        self.train_auroc(preds, labels)

        # Log to TensorBoard
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_auroc", self.train_auroc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.forward(inputs).squeeze(dim=1)
        loss = self.loss_fn(logits, labels.float())

        preds = torch.sigmoid(logits)
        self.val_acc(preds, labels)
        self.val_auroc(preds, labels)
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.forward(inputs).squeeze(dim=1)
        loss = self.loss_fn(logits, labels.float())
        
        preds = torch.sigmoid(logits)
        self.test_acc(preds, labels)
        self.test_auroc(preds, labels)
        
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_auroc", self.test_auroc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Adam is generally better and faster than SGD for standard MLPs
        optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer       
    
class LightningLSTM(L.LightningModule):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout_rate=0.2, learning_rate=0.001, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_acc = BinaryAccuracy()
        self.test_auroc = BinaryAUROC()
        
        # batch_first=True is CRITICAL because our data is structured as (Batch, Seq, Feature)
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0 # LSTM dropout only applies between multiple LSTM layers
        )
        
        # We add a dropout layer manually for the output of the LSTM
        self.dropout = nn.Dropout(dropout_rate)
        
        # The Final Output Layer (condenses the LSTM hidden state down to 1 prediction: Up or Down)
        self.linear = nn.Linear(hidden_size, 1)
        
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Initialize Metrics!
        # These stateful metrics automatically calculate totals at the end of every epoch.
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.train_auroc = BinaryAUROC()
        self.val_auroc = BinaryAUROC()

    def forward(self, x):
        # x shape: (Batch Size, Sequence Length, Input Size)
        # lstm_out contains the output features from all time steps.
        # hidden_state (h_n) contains the final summary memory after reading the full sequence.
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # We only care about the FINAL memory state (what the LSTM thinks after reading day 0)
        # h_n shape is (num_layers, batch, hidden_size). We grab the last layer's hidden state.
        final_memory = h_n[-1]
        
        dropped_out = self.dropout(final_memory)
        predictions = self.linear(dropped_out)
        return predictions

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.forward(inputs).squeeze(dim=1) 
        loss = self.loss_fn(logits, labels.float())
        
        # Calculate metrics (need probability between 0 and 1, so we use sigmoid)
        preds = torch.sigmoid(logits)
        self.train_acc(preds, labels)
        self.train_auroc(preds, labels)

        # Log to TensorBoard
        self.log("train_loss", loss, prog_bar=True)
        # on_epoch=True ensures it perfectly averages across the whole epoch
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_auroc", self.train_auroc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.forward(inputs).squeeze(dim=1)
        
        loss = self.loss_fn(logits, labels.float())
        preds = torch.sigmoid(logits)
        
        self.val_acc(preds, labels)
        self.val_auroc(preds, labels)
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.forward(inputs).squeeze(dim=1)
        
        loss = self.loss_fn(logits, labels.float())
        preds = torch.sigmoid(logits)
        
        self.test_acc(preds, labels)
        self.test_auroc(preds, labels)
        
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_auroc", self.test_auroc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

def train_lstm_model(train_loader, val_loader, num_features, hidden_size=16, num_layers=1,weight_decay=1e-4, max_epochs=200, dropout=0.3, learning_rate = 0.001, verbose=True):
    """
    Helper function to initialize, configure, and train the LSTM model.
    """
    # 1. Init Model
    model = LightningLSTM(
        input_size=num_features, 
        hidden_size=hidden_size, 
        num_layers=num_layers, 
        dropout_rate=dropout, 
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    # 2. Setup Callbacks and Logger
    early_stop = EarlyStopping(
        monitor='val_acc', 
        min_delta=0.00, 
        patience=30, 
        mode='max',
        verbose=True
    )
    logger = TensorBoardLogger(save_dir="/mnt/red/red_hanka_bcthesis/lightning_logs", name="lstm_runs") if verbose else False
    
    # 3. Setup Trainer
    trainer = L.Trainer(
        max_epochs=max_epochs, 
        callbacks=[early_stop],
        logger=logger,
        accelerator="gpu",
        devices=[0],
        enable_progress_bar=verbose,
        enable_model_summary=verbose
    )
    
    # 4. Train
    print("\n--- Starting Training ---")
    trainer.fit(model, train_loader, val_loader)
    
    # 5. Get metrics
    best_val_loss = early_stop.best_score
    if best_val_loss is not None:
        print(f"\nTraining stopped! Best Accuracy achieved: {best_val_loss.item():.4f}")
    else:
        print("\nTraining stopped, but failed to log val_loss.")
    
    return model, trainer