import polars as pl
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import os
from datetime import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
# import seaborn as sns ## seaborn makes it easier to draw nice-looking graphs.

class Log_Regr():
    def __init__(self, test_size=0.3):
         self.test_size = test_size
         self.df_master = None
         
         # Distinct lists for each feature type
         self.price_cols = []
         self.tfidf_cols = []
         self.finbert_cols = []

    def _standardize_date(self, df, current_date_col_name):
        """Safely renames and formats the date column without touching TF-IDF word columns."""
        
        # Only rename if the column isn't already 'date_article'
        if current_date_col_name != "date_article":
            df = df.rename({current_date_col_name: "date_article"})
            
        # Parse strings to dates
        if df["date_article"].dtype == pl.String:
            df = df.with_columns(pl.col("date_article").str.to_date("%Y-%m-%d"))
            
        return df.sort("date_article")
    
    def _reorder_columns(self):
        """Forces readability: Date -> Target -> Price -> FinBERT -> TF-IDF"""
        if self.df_master is None:
            return
            
        # Create the ideal order
        ideal_order = ["date_article", "target_next_day"] + self.price_cols + self.finbert_cols + self.tfidf_cols
        
        # Filter the ideal order to only include columns we have actually loaded so far
        final_cols = [col for col in ideal_order if col in self.df_master.columns]
        
        self.df_master = self.df_master.select(final_cols)

    def _save_optimal_params(self, mode, n_rows, n_cols, best_params, cv_score):
        """Saves the optimal parameters from GridSearchCV to a JSON file."""
        file_path = "optimal_parameters.jsonl"
        
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment": mode,
            "rows": n_rows,
            "features": n_cols,
            "best_cv_accuracy": round(cv_score, 4),
            "parameters": best_params
        }
        
        # 'a' means append mode. It won't overwrite previous runs.
        with open(file_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
        print(f"Optimal parameters saved to {file_path}")

    def load_and_prepare_price_data(self, df_prices, start_date=None, end_date=None):
        """
        Loads prices, creates Lags 1-5, and generates the Target variable.
        This forms the core timeline of df_master.
        """
        print("Loading prices, creating lags, and generating targets...")
        
        # Explicitly tell it to look for the lowercase 'date'
        df_prices = self._standardize_date(df_prices, "date")

        if start_date is not None and end_date is not None:
            df_prices = df_prices.filter(pl.col("date_article").is_between(start_date, end_date))

        # 2. Calculate return_lag0
        df_prices = df_prices.select([pl.col("date_article"), pl.col("close")])
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
        cols_to_keep = ["date_article", "target_next_day"] + self.price_cols
        self.df_master = df_prices.select(cols_to_keep).drop_nulls()
        
        self._reorder_columns()
        print(f"Price timeline established. Shape: {self.df_master.shape}")
        return self.df_master
    
    def load_tfidf_data(self, df_tfidf):
        """
        Loads TF-IDF features and joins them to the stock target.
        """
        if self.df_master is None: raise ValueError("Run load_and_prepare_price_data first!")
            
        print("Loading TF-IDF data and aligning with timeline...")
        df_tfidf = self._standardize_date(df_tfidf, "Date")       
        
        self.tfidf_cols = [c for c in df_tfidf.columns if c != "date_article"]
        
        self.df_master = self.df_master.join(df_tfidf, on="date_article", how="left")
        self.df_master = self.df_master.fill_null(0)
        
        self._reorder_columns()
        print(f"TF-IDF aligned. Total columns in master: {len(self.df_master.columns)}")
        return self.df_master

    def load_finbert_data(self, df_finbert):
        if self.df_master is None: raise ValueError("Run load_and_prepare_price_data first!")
            
        print("Loading FinBERT data and aligning with timeline...")
        df_finbert = self._standardize_date(df_finbert, "Date")       
        
        self.finbert_cols = [c for c in df_finbert.columns if c != "date_article"]
        
        self.df_master = self.df_master.join(df_finbert, on="date_article", how="left")
        self.df_master = self.df_master.fill_null(0)
        
        self._reorder_columns()
        print(f"FinBERT aligned. Total columns in master: {len(self.df_master.columns)}")
        #display(self.df_master)
        return self.df_master

    def train_and_evaluate(self, mode):
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
            
        if not active_features:
            raise ValueError(f"No features found for mode '{mode}'. Did you load the data?")
        
        print(f"\nStarting Experiment: {mode.upper()} ({len(active_features)} features)")
        
        # 2. Arrays & Splits
        x = self.df_master.select(active_features).to_numpy()
        y = self.df_master["target_next_day"].to_numpy()
        X_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size, shuffle=False)

        # 3. Pipeline & GridSearch
        tscv = TimeSeriesSplit(n_splits=5)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('logr', LogisticRegression(solver='saga', class_weight='balanced', max_iter=2000))
        ])
        
        param_grid = {
            'logr__C': [0.1, 1.0, 10.0, 100.0],
            'logr__l1_ratio': [0.0, 0.1, 0.5, 0.9, 1.0] 
        }        
        print(f"\n--- Running GridSearchCV ({tscv.n_splits} splits) ---")
        
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # 4. Extract Results
        best_pipeline = grid_search.best_estimator_
        best_model = best_pipeline.named_steps['logr'] 
        
        print(f"Best Params: {grid_search.best_params_} | Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")
        total_rows = len(x)
        total_features = len(active_features)
        self._save_optimal_params(mode, total_rows, total_features, grid_search.best_params_, grid_search.best_score_)

        y_pred = best_pipeline.predict(x_test)
        y_probs = best_pipeline.predict_proba(x_test)
        #accuracy = accuracy_score(y_test, y_pred)

        print("\n--- Final Model Reality Check (Untouched Test Set) ---")
        #print(f"First 5 Class Predictions: {y_pred[:5]}")
        print(f"Min Prob: {y_probs[:,1].min():.4f}, Max Prob: {y_probs[:,1].max():.4f}, Mean Prob: {y_probs[:,1].mean():.4f}")
        print(f"Final Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Down (0)', 'Up (1)'], zero_division=0))
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(conf_matrix)

        # 5. Dynamic Weight Analysis
        coefs = best_model.coef_[0] 
        print("--- Weight Analysis ---")        
        price_weights, text_weights = np.array([]), np.array([])
        text_type = "None"

        if mode == 'price':
            price_weights = coefs
        elif mode == 'tfidf':
            text_weights = coefs
            text_type = "TF-IDF"
        elif mode == 'finbert':
            text_weights = coefs
            text_type = "FinBERT"
        elif mode == 'hybrid_tfidf':
            # Price is first, so we slice the first N columns for price
            num_price = len(self.price_cols)
            price_weights = coefs[:num_price]
            text_weights = coefs[num_price:]
            text_type = "TF-IDF"
        elif mode == 'hybrid_finbert':
            # Price is first, so we slice the first N columns for price
            num_price = len(self.price_cols)
            price_weights = coefs[:num_price]
            text_weights = coefs[num_price:]
            text_type = "FinBERT"

        # 6. Calculate and print stats
        if len(price_weights) > 0:
            print(f"Price Features Count: {len(price_weights)}")
            print(f"Average Absolute Price Weight: {np.mean(np.abs(price_weights)):.6f}")
        else:
            print("Price Features: [None in this mode]")

        if len(price_weights) > 0:
            print(f"Average Absolute Price Weight ({len(price_weights)} cols): {np.mean(np.abs(price_weights)):.6f}")
        if len(text_weights) > 0:
            print(f"Average Absolute {text_type} Weight ({len(text_weights)} cols): {np.mean(np.abs(text_weights)):.6f}")

        max_idx = np.argmax(np.abs(coefs))
        print(f"Most influential feature: '{active_features[max_idx]}' (Weight: {coefs[max_idx]:.4f})")

        # --- 1. ROC Curve ---
        plt.figure(figsize=(8, 6))
        RocCurveDisplay.from_estimator(best_pipeline, x_test, y_test)
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray") # Random guess line
        plt.title("ROC Curve: Signal vs. Noise")
        plt.show()

        # --- 2. Money Chart (Simple Version) ---
        # We need the actual returns from the test period
        # Assuming your df_master is still aligned with the test set
        test_returns = self.df_master.tail(len(y_pred))["return_lag0"].to_numpy()

        # Strategy: If pred == 1, hold stock. If pred == 0, hold cash.
        strategy_returns = y_pred * test_returns

        # Calculate Cumulative Growth (1 + r).cumprod()
        cum_strategy = np.insert((1 + strategy_returns).cumprod(), 0, 1.0)
        cum_market = np.insert((1 + test_returns).cumprod(), 0, 1.0)

        plt.figure(figsize=(10, 5))
        plt.plot(cum_strategy, label="Model Strategy", linewidth=2)
        plt.plot(cum_market, label="Market (Buy & Hold)", linestyle="--")
        plt.title("Cumulative Returns: Model vs. Market")
        plt.ylabel("Growth of $1")
        plt.legend()
        plt.show()
        
        return x_test, y_pred