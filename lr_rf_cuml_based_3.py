import os
import warnings
from xmlrpc import client
from scipy import cluster
#from torch import mode
#os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0" is the 3080. "1" is the 3070.

import polars as pl
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from dask_ml.model_selection import GridSearchCV as DaskGridSearchCV
import dask_cudf
import cudf

from cuml.preprocessing import StandardScaler
from cuml.linear_model import LogisticRegression
from cuml.model_selection import train_test_split
from cuml.metrics import accuracy_score, roc_auc_score
from cuml.ensemble import RandomForestClassifier

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import RocCurveDisplay
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay
#from sklearn.metrics import make_scorer

class mach_lern():
    def __init__(self, test_size=0.3):
         self.test_size = test_size
         self.df_master = None
         
         # Distinct lists for each feature type
         self.price_cols = []
         self.tfidf_cols = []
         self.finbert_cols = []

    def _standardize_date(self, df, current_date_col_name):
        """Safely renames and formats the date column without touching TF-IDF word columns."""
        
        # Only rename if the column isn't already 'date_utc'
        if current_date_col_name != "trading_date_utc":
            df = df.rename({current_date_col_name: "trading_date_utc"})
            
        # Parse strings to dates
        if df["trading_date_utc"].dtype == pl.String:
            df = df.with_columns(pl.col("trading_date_utc").str.to_date("%Y-%m-%d"))
            
        return df.sort("trading_date_utc")
    
    def _reorder_columns(self):
        """Forces readability: Date -> Target -> Price -> FinBERT -> TF-IDF"""
        if self.df_master is None:
            return
            
        # Create the ideal order
        ideal_order = ["trading_date_utc", "target_next_day", "nyse_ticker"] + self.price_cols + self.finbert_cols + self.tfidf_cols
        
        # Filter the ideal order to only include columns we have actually loaded so far
        final_cols = [col for col in ideal_order if col in self.df_master.columns]
        final_cols = list(dict.fromkeys(final_cols))  # preserves order, removes duplicates
        self.df_master = self.df_master.select(final_cols)

    def _save_optimal_params(self, model_type, mode, n_rows, n_cols, best_params, cv_accuracy, cv_auc):
        """Saves the optimal parameters from GridSearchCV to a JSON file."""
        file_path = "optimal_parameters.jsonl"
        
        stocks_used = self.df_master["nyse_ticker"].unique().to_list() if "nyse_ticker" in self.df_master.columns else []

        log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": model_type,
        "mode": mode,
        "best_cv_accuracy": round(cv_accuracy, 4),
        "best_cv_auc": round(cv_auc, 4),
        "rows": n_rows,
        "features": n_cols,
        "stocks_used": stocks_used,
        "hyperparameters": best_params
    }
        
        # 'a' means append mode. It won't overwrite previous runs.
        with open(file_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
        print(f"Optimal parameters saved to {file_path}")

    @staticmethod
    def gpu_auc_scorer(estimator, X, y):
        """
        A direct scorer that bypasses Scikit-Learn's internal index handling.
        """
        # 1. Get probabilities directly from the estimator on the GPU
        # y_probs will be a cuDF DataFrame with 2 columns
        y_probs = estimator.predict_proba(X)
        
        # 2. Slice the second column (class 1) using .values 
        # .values converts it to a CuPy array, stripping the problematic index
        # Safely extract column 1
        # Extract column 1 (probability of class "Up")
        if hasattr(y_probs, 'iloc'):  # cuDF DataFrame
            y_score = y_probs.iloc[:, 1].to_numpy()
        elif hasattr(y_probs, 'get'):  # CuPy array
            y_score = y_probs[:, 1].get()
        else:
            y_score = y_probs[:, 1]

        # Convert y to numpy
        if hasattr(y, 'get'):  # CuPy array
            y_np = y.get()
        elif hasattr(y, 'to_numpy'):  # cuDF Series
            y_np = y.to_numpy()
        else:
            y_np = np.array(y)
            
        try:
            return roc_auc_score(y_np, y_score)
        except ValueError as e:
            # If the chronological fold happens to be a 100% "Up" or 100% "Down" streak, 
            # AUC is mathematically impossible. We return 0.5 (a neutral random guess) 
            # so the GridSearch doesn't crash.
            if "only one class" in str(e).lower():
                return 0.5 
            else:
                raise e # If it's a different ValueError, we still want to know about it

    @staticmethod
    def gpu_accuracy_scorer(estimator, X, y):
        """Accuracy scorer that handles GPU arrays (CuPy/cuDF)."""
        y_pred = estimator.predict(X)
            # Convert y_pred to numpy - it's a CuPy array
        if hasattr(y_pred, 'get'):  # CuPy array
            y_pred_np = y_pred.get()
        elif hasattr(y_pred, 'to_numpy'):  # cuDF Series
            y_pred_np = y_pred.to_numpy()
        else:
            y_pred_np = np.array(y_pred)

        # Convert y to numpy - it's likely a cuDF Series from DaskGridSearchCV
        if hasattr(y, 'get'):  # CuPy array
            y_np = y.get()
        elif hasattr(y, 'to_numpy'):  # cuDF Series
            y_np = y.to_numpy()
        else:
            y_np = np.array(y)

        from sklearn.metrics import accuracy_score
        return accuracy_score(y_np, y_pred_np)

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
        
        self._reorder_columns()
        print(f"Price timeline established. Shape: {self.df_master.shape}")
        #display(self.df_master)
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
            df = df.with_columns(pl.lit(ticker).alias("nyse_ticker"))
            
            # Select final columns and clean nulls
            cols_to_keep = ["trading_date_utc", "nyse_ticker", "target_next_day"] + lag_cols
            df = df.select(cols_to_keep).drop_nulls()
            all_dfs.append(df)

        self.price_cols = lag_cols
        
        # Stack them all vertically. 
        self.df_master = pl.concat(all_dfs).sort("trading_date_utc")
        print(f"Master multi-stock timeline established. Shape: {self.df_master.shape}")
        return self.df_master
    
    def load_tfidf_data(self, df_tfidf):
        """
        Loads TF-IDF features and joins them to the stock target.
        """
        if self.df_master is None: raise ValueError("Run load_and_prepare_price_data first!")
            
        print("Loading TF-IDF data and aligning with timeline...")
        df_tfidf = self._standardize_date(df_tfidf, "trading_session_date_utc")       
        
        self.tfidf_cols = [c for c in df_tfidf.columns if c != "trading_date_utc"]
        
        self.df_master = self.df_master.join(df_tfidf, on="trading_date_utc", how="left")
        self.df_master = self.df_master.fill_null(0)
        
        self._reorder_columns()
        print(f"TF-IDF aligned. Total columns in master: {len(self.df_master.columns)}")
        #display(self.df_master)
        return self.df_master
    
    def load_finbert_data(self, df_finbert):
        if self.df_master is None: raise ValueError("Run load_and_prepare_price_data first!")
            
        print("Loading FinBERT data and aligning with timeline...")
        df_finbert = self._standardize_date(df_finbert, "trading_session_date_utc")       
        
        self.finbert_cols = [c for c in df_finbert.columns if c != "trading_date_utc"]
        
        self.df_master = self.df_master.join(df_finbert, on="trading_date_utc", how="left")
        self.df_master = self.df_master.fill_null(0)
        
        self._reorder_columns()
        print(f"FinBERT aligned. Total columns in master: {len(self.df_master.columns)}")
        #display(self.df_master)
        return self.df_master

    def start_dask_cluster(self):
        """Starts the Dask CUDA cluster if it isn't already running."""
        # Check if we already created a client to avoid duplicates
        if not hasattr(self, 'client') or self.client is None:
            print("Booting up LocalCUDACluster...")
            self.cluster = LocalCUDACluster()
            self.client = Client(self.cluster)
            print(f"Dask dashboard available at: {self.client.dashboard_link}")
        else:
            print(f"Cluster is already running at: {self.client.dashboard_link}")

    def stop_dask_cluster(self):
        """Safely shuts down the Dask cluster and frees GPU memory."""
        if hasattr(self, 'client') and self.client is not None:
            print("Shutting down Dask cluster and freeing ports...")
            self.client.close()
            self.cluster.close()
            self.client = None
            self.cluster = None
            print("Cluster successfully shut down.")
        else:
            print("No active cluster to shut down.")

    def train_logistic_regression(self, mode, param_grid):
        """
        Valid modes: 'price', 'tfidf', 'finbert', 'hybrid_tfidf', 'hybrid_finbert'
        """
        if self.df_master is None: raise ValueError("No data found.")
        
        # 1. Mode Selector
        active_features = []
        if mode == 'price':
            active_features = self.price_cols
        elif mode == 'tfidf':
            active_features = self.tfidf_cols
        elif mode == 'finbert':
            active_features = self.finbert_cols
        elif mode == 'hybrid_tfidf':
            active_features = self.price_cols + self.tfidf_cols 
        elif mode == 'hybrid_finbert':
            active_features = self.price_cols + self.finbert_cols
        else:
            raise ValueError(f"Invalid mode: {mode}")
        #Since you are in the experimentation phase, keep it for now. It acts as a control variable for "Activity Level." 
        #Just keep an eye on your Confusion Matrix. If your model starts predicting "Up" every time the article count is 
        #high, you'll know it's found a "cheat" rather than a real signal.    
        if not active_features:
            raise ValueError(f"No features found for mode '{mode}'. Did you load the data?")
        
        print(f"\nStarting Experiment: {mode.upper()} ({len(active_features)} features)")
        
        # 1. THE FAST LANE: Move Features to GPU via Arrow
        # We bypass Pandas/NumPy entirely. This stays in Arrow format.
        X_arrow = self.df_master.select(active_features).to_arrow()
        X_gpu = cudf.DataFrame.from_arrow(X_arrow).to_cupy()
        
        # 2. THE FAST LANE: Move Target to GPU
        # We let Polars handle the Int32 cast natively before sending to GPU
        y_arrow = self.df_master["target_next_day"].cast(pl.Int32).to_arrow()
        y_gpu = cudf.Series.from_arrow(y_arrow).to_cupy()
        X_train, x_test, y_train, y_test = train_test_split(X_gpu, y_gpu, test_size=self.test_size, shuffle=False)

        X_train_gdf = cudf.DataFrame(X_train)
        y_train_gs  = cudf.Series(y_train)
        # 3. Pipeline & GridSearch
        tscv = TimeSeriesSplit(n_splits=5)#2
        #note: consider switching to 10 folds in the future, but stick to  5 for now
        
        # We specify penalty='elasticnet' and solver='qn' for GPU optimization.
        pipeline = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ('logr', LogisticRegression(penalty='elasticnet', solver='qn', max_iter=4000))#500
                                        #class_weight='balanced'))
        ])       
        #Why it's good: If your training data has 60% "Up" days and 40% "Down" days, a lazy model might
        # just guess "Up" every time to get 60% accuracy. balanced forces the model to treat a correct 
        # "Down" guess as more valuable, preventing this bias.[class_weight='balanced']
        # param_grid = {
        #     'logr__C': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
        #     'logr__l1_ratio': [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
        #     "logr__class_weight": [None, "balanced"],
        #     "logr__tol": [1e-4, 1e-3],
        # }       
        #debugging grid
        # param_grid = {
        #     'logr__C': [0.1, 1.0, 10.0],  # instead of 9 values
        #     'logr__l1_ratio': [0.2, 0.8],  # instead of 7 values
        #     "logr__class_weight": [None],   # just one instead of both
        #     "logr__tol": [1e-3],            # just one instead of both
        # }
        # This reduces 252 combinations down to 3×2×1×1 = 6 combinations 
        #when having enough RAM and GPU power, consider adding more values to the grid, like C=0.01 or 1000, and l1_ratio=0.25, 0.75, etc. 
        # But start with a smaller grid for faster iteration.

        #Small C (e.g., 0.1): Strong "handcuffs." It forces the weights to be small. This makes the model "simpler" and less likely to overfit.
        #Large C (e.g., 100.0): Weak "handcuffs." The model is allowed to use huge weights to fit the training data as perfectly
        # as possible. In trading, this often leads to great "backtest" results but terrible real-world performance.

        if not hasattr(self, 'client') or self.client is None:
            print("No active Dask cluster found. Starting one automatically...")
            self.start_dask_cluster()

        print(f"\n--- Running GPU GridSearchCV ({tscv.n_splits} splits) ---")

        # Dask automatically handles the parallelization safely on the GPU
        grid_search = DaskGridSearchCV(
            estimator=pipeline, 
            param_grid=param_grid, 
            cv=tscv, 
            scoring=mach_lern.gpu_accuracy_scorer
        )
        #note: consider leaving out scoring parameter and then grid_search 
        #will go for default scorer which is accuracy (i think). this custom
        #made method might not be fully correct yet, roc auc should be better parameter

        # Now dask_cudf will happily partition them
        X_dask = dask_cudf.from_cudf(X_train_gdf, npartitions=1) # Use fewer partitions for small data
        y_dask = dask_cudf.from_cudf(y_train_gs, npartitions=1)
        grid_search.fit(X_dask, y_dask)

        # 4. Extract Results
        best_pipeline = grid_search.best_estimator_      
        best_cv_accuracy = grid_search.best_score_
        print("Calculating CV AUC for best model...")
        tscv_for_auc = TimeSeriesSplit(n_splits=5)#2
        cv_auc_scores = []
        for train_idx, val_idx in tscv_for_auc.split(X_train_gdf):
            X_train_fold = X_train_gdf.iloc[train_idx]
            X_val_fold = X_train_gdf.iloc[val_idx]
            y_train_fold = y_train_gs.iloc[train_idx]
            y_val_fold = y_train_gs.iloc[val_idx]
            
            best_pipeline.fit(X_train_fold, y_train_fold)
            
            # Predict Probas & send to numpy properly to avoid bugs
            y_probs = best_pipeline.predict_proba(X_val_fold)
            # cuML RF output is a CuPy array. Get class 1 (up)
            if hasattr(y_probs, 'iloc'):  # cuDF DataFrame
                y_score = y_probs.iloc[:, 1].to_numpy()  # Get column 1 (Up class) positionally
            elif hasattr(y_probs, 'get'):  # CuPy array
                y_score = y_probs[:, 1].get()
            else:  # Already numpy
                y_score = y_probs[:, 1]            
                
            y_np = y_val_fold.to_numpy() # y_val_fold is cuDF Series
            
            try:
                fold_auc = roc_auc_score(y_np, y_score)
                cv_auc_scores.append(fold_auc)
            except ValueError as e:
                pass # Only happens if 100% of the slice is the same class
        best_cv_auc = np.mean(cv_auc_scores) if cv_auc_scores else 0.5

        print(f"Best Params: {grid_search.best_params_}")
        print(f"Best CV Accuracy: {best_cv_accuracy:.4f} | Best CV AUC: {best_cv_auc:.4f}")
        
       #L1 (Lasso): The "Executioner." It forces useless features to have a weight of exactly zero. It effectively performs feature selection for you.
        total_rows = len(X_gpu)
        total_features = len(active_features)
        self._save_optimal_params(
            model_type="LR",
            mode=mode,
            n_rows=total_rows,
            n_cols=total_features,
            best_params=grid_search.best_params_,
            cv_accuracy=best_cv_accuracy,
            cv_auc=best_cv_auc
        )
        
        return best_pipeline, x_test, y_test, active_features

    def train_random_forest(self, mode, param_grid):
        if self.df_master is None: 
            raise ValueError("No data found.")
        
        # 1. Mode Selector (Same as LR)
        active_features = []
        if mode == 'price': 
            active_features = self.price_cols
        elif mode == 'tfidf': 
            active_features = self.tfidf_cols
        elif mode == 'finbert': 
            active_features = self.finbert_cols
        elif mode == 'hybrid_tfidf': 
            active_features = self.price_cols + self.tfidf_cols 
        elif mode == 'hybrid_finbert': 
            active_features = self.price_cols + self.finbert_cols
        else: raise ValueError(f"Invalid mode: {mode}")
        
        print(f"\nStarting Experiment: RF {mode.upper()} ({len(active_features)} features)")
        
        # 2. Fast Lane Data Transfer
        X_arrow = self.df_master.select(active_features).to_arrow()
        X_gpu = cudf.DataFrame.from_arrow(X_arrow).to_cupy()
        y_arrow = self.df_master["target_next_day"].cast(pl.Int32).to_arrow()
        y_gpu = cudf.Series.from_arrow(y_arrow).to_cupy()
        
        from sklearn.model_selection import train_test_split
        X_train, x_test, y_train, y_test = train_test_split(X_gpu, y_gpu, test_size=self.test_size, shuffle=False)

        X_train_gdf = cudf.DataFrame(X_train)
        y_train_gs  = cudf.Series(y_train)
        
        # 3. Pipeline & GridSearch (Unique to Random Forest!)
        tscv = TimeSeriesSplit(n_splits=5)#2
        
        # Random Forests don't strictly need scaling, but it keeps the pipeline consistent
        pipeline = Pipeline([
            ('rf', RandomForestClassifier(random_state=42, n_streams=8)) 
        ])       
        
        # moved to being input param Tailored Grid for GPU Random Forest
        # param_grid = {
        #     'rf__n_estimators': [200, 400], # More trees = smoother, more stable predictions
        #     'rf__max_depth': [6, 10, 14],             # Constrained depth prevents tracing noise
        #     'rf__max_features': ['sqrt', 'log2'],     # log2 is incredibly fast for high dimensionality
        #     'rf__min_samples_leaf': [5, 10]  # Standard options are fine here
        # }        
        #debugging grid
        # param_grid = {
        #     'rf__n_estimators': [200],         # Just one value
        #     'rf__max_depth': [10],             # Just one value
        #     'rf__max_features': ['sqrt'],      # Just one value
        #     'rf__min_samples_leaf': [5]        # Just one value
        # }

        if not hasattr(self, 'client') or self.client is None:
            self.start_dask_cluster()

        print(f"\n--- Running GPU GridSearchCV ({tscv.n_splits} splits) ---")
        
        grid_search = DaskGridSearchCV(
            estimator=pipeline, 
            param_grid=param_grid, 
            cv=tscv, 
            scoring=mach_lern.gpu_accuracy_scorer
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The number of bins")
            X_dask = dask_cudf.from_cudf(X_train_gdf, npartitions=1)
            y_dask = dask_cudf.from_cudf(y_train_gs, npartitions=1)
            
            grid_search.fit(X_dask, y_dask)

        # 4. Extract Results
        best_pipeline = grid_search.best_estimator_  
        best_cv_accuracy = grid_search.best_score_    

        print("Calculating CV AUC for best model...")
        tscv_for_auc = TimeSeriesSplit(n_splits=5)
        cv_auc_scores = []
        for train_idx, val_idx in tscv_for_auc.split(X_train_gdf):
            X_train_fold = X_train_gdf.iloc[train_idx]
            X_val_fold = X_train_gdf.iloc[val_idx]
            y_train_fold = y_train_gs.iloc[train_idx]
            y_val_fold = y_train_gs.iloc[val_idx]
            
            best_pipeline.fit(X_train_fold, y_train_fold)
            
            # Predict Probas & send to numpy properly to avoid bugs
            y_probs = best_pipeline.predict_proba(X_val_fold)
            # cuML RF output is a CuPy array. Get class 1 (up)
            if hasattr(y_probs, 'iloc'):  # cuDF DataFrame
                y_score = y_probs.iloc[:, 1].to_numpy()  # Get column 1 (Up class) positionally
            elif hasattr(y_probs, 'get'):  # CuPy array
                y_score = y_probs[:, 1].get()
            else:  # Already numpy
                y_score = y_probs[:, 1]            
                
            y_np = y_val_fold.to_numpy() # y_val_fold is cuDF Series
            
            try:
                fold_auc = roc_auc_score(y_np, y_score)
                cv_auc_scores.append(fold_auc)
            except ValueError as e:
                pass # Only happens if 100% of the slice is the same class
        best_cv_auc = np.mean(cv_auc_scores) if cv_auc_scores else 0.5

        print(f"Best Params: {grid_search.best_params_}")
        print(f"Best CV Accuracy: {best_cv_accuracy:.4f} | Best CV AUC: {best_cv_auc:.4f}")
        
        self._save_optimal_params(
            model_type="RF",
            mode=mode,
            n_rows=len(X_gpu),
            n_cols=len(active_features),
            best_params=grid_search.best_params_,
            cv_accuracy=best_cv_accuracy,
            cv_auc=best_cv_auc
        )        
        return best_pipeline, x_test, y_test, active_features

    def evaluate(self, mode, best_pipeline, x_test, y_test, active_features):
        """
        Evaluates the trained pipeline on the test set and prints statistics.
        Returns the raw predictions and probabilities.
        """
        #1. Get predictions and probabilities
        y_pred = best_pipeline.predict(x_test)
        y_probs = best_pipeline.predict_proba(x_test)
        #accuracy = accuracy_score(y_test, y_pred)

        # To handle cuML arrays, we might need to cast to numpy for Sklearn's classification_report
        y_test_np = y_test.get() if hasattr(y_test, 'get') else y_test
        y_pred_np = y_pred.get() if hasattr(y_pred, 'get') else y_pred
        #has both classes predictions (Up and down)
        
        # Handle y_probs: extract column 1 (positive class) and move to CPU
        if hasattr(y_probs, 'iloc'): # If it's cuDF
            y_probs_np = y_probs.iloc[:, 1].values.get()
        else: # If it's CuPy
            y_probs_np = y_probs[:, 1].get()
            #probs of class 1 (Up)

        print("\n--- Final Model Reality Check (Untouched Test Set) ---")
        #print(f"First 5 Class Predictions: {y_pred[:5]}")
        print(f"Min Prob: {y_probs_np.min():.4f}, Max Prob: {y_probs_np.max():.4f}, Mean Prob: {y_probs_np.mean():.4f}")
        
        print(f"Final Test Accuracy: {accuracy_score(y_test_np, y_pred_np):.4f}")
        test_auc = roc_auc_score(y_test_np, y_probs_np)
        print(f"Final Test ROC AUC: {test_auc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test_np, y_pred_np, target_names=['Down (0)', 'Up (1)'], zero_division=0))

        # 2. Universal Feature Analysis
        print("\n--- Weight/Importance Analysis ---")        
        # Extract the actual model from the end of the pipeline
        best_model = best_pipeline[-1] if hasattr(best_pipeline, 'named_steps') else best_pipeline
        
        # Step A: Grab the raw data, whatever type it is
        if hasattr(best_model, 'coef_'):
            raw_data = best_model.coef_
            print("Metric: Linear Coefficients (Magnitude = Impact, Sign = Direction)")
        elif hasattr(best_model, 'feature_importances_'):
            raw_data = best_model.feature_importances_
            print("Metric: Gini Importance (Magnitude = How often it was useful)")
        else:
            print("Warning: Model has no feature importance attribute.")
            return y_pred, y_probs

        # Step B: Safely strip the cuDF/CuPy structure and flatten to a CPU NumPy array
        if hasattr(raw_data, 'to_numpy'):
            # If it's a cuDF DataFrame or Series
            importances = raw_data.to_numpy().flatten()
        elif hasattr(raw_data, 'get'):
            # If it's a CuPy GPU array
            importances = raw_data.get().flatten()
        else:
            # Fallback for standard CPU arrays
            importances = np.array(raw_data).flatten()

        # # 2. Dynamic Weight Analysis
        # best_model = best_pipeline.named_steps['logr']   
        # coefs = best_model.coef_.values.flatten()
        # if hasattr(coefs, 'get'): 
        #     coefs = coefs.get() 
        
        print("--- Weight Analysis ---")        
        price_weights, text_weights = np.array([]), np.array([])
        text_type = "TF-IDF" if 'tfidf' in mode else "FinBERT"
        
        if mode == 'price':
            price_weights = importances
        elif mode in ['tfidf', 'finbert']:
            text_weights = importances
        elif mode in ['hybrid_tfidf', 'hybrid_finbert']:
            num_price = len(self.price_cols)
            price_weights = importances[:num_price]
            text_weights = importances[num_price:]

        # 3. Calculate and print stats
        if len(price_weights) > 0:
            #print(f"Price Features Count: {len(price_weights)}")
            print(f"Average Absolute Price Weight ({len(price_weights)} features): {np.mean(np.abs(price_weights)):.6f}")
        else:
            print("Price Features: [None in this mode]")

        if len(text_weights) > 0:
            print(f"Average Absolute {text_type} Importance ({len(text_weights)} features): {np.mean(np.abs(text_weights)):.6f}")

        max_idx = np.argmax(np.abs(importances))
        print(f"Most influential feature: '{active_features[max_idx]}' (Importance: {importances[max_idx]:.4f})")
        
        # 1. Confusion matrix
        conf_matrix = confusion_matrix(y_test_np, y_pred_np)
        print("\nConfusion Matrix:")
        print(conf_matrix)

        return y_pred, y_probs
    
    def draw_charts(self, y_test, y_pred, y_probs):
        """
        Visualizes the model's performance via ROC Curve and Cumulative Returns.
        """
        # Ensure all plotting data is on the CPU
        #x_test_np = x_test.get() if hasattr(x_test, 'get') else x_test
        y_test_np = y_test.get() if hasattr(y_test, 'get') else y_test
        y_pred_np = y_pred.get() if hasattr(y_pred, 'get') else y_pred
        
        # Handle y_probs: We need it on CPU, and we ONLY want column 1 (Probability of "Up")
        # Step A: Move to CPU first
        y_probs_temp = y_probs.get() if hasattr(y_probs, 'get') else y_probs
        
        # Step B: Slice it if it has 2 columns (Down, Up)
        if y_probs_temp.ndim == 2:
            y_probs_np = y_probs_temp[:, 1]
        else:
            y_probs_np = y_probs_temp
        
        # 1. Confusion matrix
        conf_matrix = confusion_matrix(y_test_np, y_pred_np)
        # print("\nConfusion Matrix:")
        # print(conf_matrix)

        plt.figure(figsize=(6, 5))
        cmd = ConfusionMatrixDisplay(
            confusion_matrix=conf_matrix,
            display_labels=['Down (0)', 'Up (1)']
        )
        cmd.plot(cmap='Blues', values_format='d') # 'd' for decimal integers
        plt.title("Confusion Matrix: Market Direction Predictions")
        plt.show()
        
        # 2. ROC Curve ---
        plt.figure(figsize=(8, 6))
        RocCurveDisplay.from_predictions(y_test_np, y_probs_np)
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray") # Random guess line
        plt.title("ROC Curve: Signal vs. Noise")
        plt.show()

        # 3. Money Chart        
        # Grab the actual FUTURE returns (shift -1 means "look at tomorrow's return")
        future_returns = self.df_master["return_lag0"].shift(-1).fill_null(0.0)        
        
        # Slice only the test period
        test_returns = future_returns.tail(len(y_pred_np))
        test_returns = test_returns.head(-1).to_numpy()
        y_pred_np = y_pred_np[:-1]
        # Strategy: If pred == 1, hold stock. If pred == 0, hold cash.
        strategy_returns = y_pred_np * test_returns

        # 4. Calculate Cumulative Growth
        cum_strategy = np.insert((1 + strategy_returns).cumprod(), 0, 1.0)
        cum_market = np.insert((1 + test_returns).cumprod(), 0, 1.0)

        plt.figure(figsize=(10, 5))
        plt.plot(cum_strategy, label="Model Strategy", linewidth=2, color="blue")
        plt.plot(cum_market, label="Market (Buy & Hold)", linestyle="--", color="orange")
        plt.title("Cumulative Returns: Model vs. Market")
        plt.xlabel("Trading Days")
        plt.ylabel("Growth of $1.00")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()