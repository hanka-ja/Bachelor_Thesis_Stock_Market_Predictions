import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import NewsPreprocessing
from datetime import date, timedelta
import numpy as np
from tqdm.auto import tqdm
import torch
from IPython.display import display
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel.scatter_gather import scatter_kwargs
from dotenv import load_dotenv
import pandas_market_calendars as mcal
import glob
import time
import gc

#os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 1. Create a lightweight Dataset class
class NewsDataset(Dataset):
    def __init__(self, polars_series):
        # We store the Polars Series directly. Zero memory copy.
        self.texts = polars_series
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return str(self.texts[idx])

class BalancedDataParallel(torch.nn.DataParallel):
    def __init__(self, gpu0_bs, *args, **kwargs):
        self.gpu0_bs = gpu0_bs
        super().__init__(*args, **kwargs)

    def scatter(self, inputs, kwargs, device_ids):
        # We manually define the sizes for the 3080 (GPU 0) and 3070 (GPU 1)
        # If batch is 1024, and gpu0_bs is 640, GPU 1 gets 384.
        if 'input_ids' in kwargs:
            total_bs = kwargs['input_ids'].size(0)
        elif inputs:
            total_bs = inputs[0].size(0)
        else:
            raise ValueError("Could not determine batch size from inputs or kwargs")
        # Handle the last batch (which might be smaller than your 640 limit)
        if total_bs <= self.gpu0_bs:
            # If the remaining batch is tiny, just give it all to the 3080
            chunk_sizes = [total_bs, 0]
        else:
            chunk_sizes = [self.gpu0_bs, total_bs - self.gpu0_bs]
        
        return scatter_kwargs(
            inputs, kwargs, device_ids, dim=self.dim, chunk_sizes=chunk_sizes
        )

class TfIdfVectorizer():
    def __init__(self, dataset, max_features=5000):
        self.max_features = max_features
        self.df_cleaned = dataset
        self.vectorizer = None

    def split_data(self, cleaned_dataset, train_ratio = 0.7, date_col = "trading_session_date_utc"): 
        # 1. Sort data to ensure chronological order
        df_sorted = cleaned_dataset.sort(date_col)        
        # 2. Get the list of dates
        dates = df_sorted[date_col]
        total_rows = len(dates)        
        # 3. Find the index where the split should happen based on COUNT
        split_index = int(total_rows * train_ratio)        
        # 4. Identify the specific date at that index to serve as the cutoff
        # Everything BEFORE this date represents 70% of the ROWS
        self.train_cutoff = dates[split_index]
        # Split Data to test and train data using the cutoff
        self.train_text = cleaned_dataset.filter(pl.col(date_col) < self.train_cutoff)["preprocessed_for_tfidf"]
        self.test_text = cleaned_dataset.filter(pl.col(date_col) >= self.train_cutoff)["preprocessed_for_tfidf"]

        print(f"Ready (Row-Count based split)!")
        print(f"Total Unique Days (Rows): {total_rows}")
        print(f"Train Ratio: {train_ratio:.1%}")
        print(f"Split Index: {split_index}")
        print(f"Cutoff Date: {self.train_cutoff}")
        print(f"Train Period: {dates.min()} to {self.train_cutoff} (Exclusive)")
        print(f"Train samples: {len(self.train_text)}, Test samples: {len(self.test_text)}")
        
        #with pl.Config(fmt_str_lengths=1000, tbl_width_chars=1000, tbl_rows=5):
        display(self.train_text)
        return self.train_text, self.test_text, cleaned_dataset
    
    def fit_transform(self, cleaned_dataset, train_text, test_text, date_col = "trading_session_date_utc"):

        if self.vectorizer is None:
            print(f"Initializing TfidfVectorizer (max_features={self.max_features})...")
            self.vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words='english')

        print("Fitting TF-IDF on training data...")
        # Check if training data is empty to prevent errors
        if len(train_text) > 0:
            X_train = self.vectorizer.fit_transform(train_text.to_list())
            print(f"Train Shape: {X_train.shape}")
        else:
            print("Warning: Training set is empty. Check your date cutoff.")
            X_train = None

        print("Transforming test data...")
        if len(test_text) > 0 and X_train is not None:
            X_test = self.vectorizer.transform(test_text.to_list())
            print(f"Test Shape:  {X_test.shape}")
        else:
            print("Warning: Test set is empty or training failed.")

        # 1. Recover Date columns (since we filtered them out for vectorization)
        train_dates = cleaned_dataset.filter(pl.col(date_col) < self.train_cutoff)[date_col]
        test_dates = cleaned_dataset.filter(pl.col(date_col) >= self.train_cutoff)[date_col]

        # 2. Create Train DataFrame
        # We convert the matrix (X_train) to a DataFrame and add the 'Date' column back
        df_train = pl.DataFrame(
            X_train.toarray(), 
            schema=list(self.vectorizer.get_feature_names_out())
        ).with_columns(train_dates.alias(date_col))

        # 3. Create Test DataFrame
        df_test = pl.DataFrame(
            X_test.toarray(), 
            schema=list(self.vectorizer.get_feature_names_out())
        ).with_columns(test_dates.alias(date_col))

        # 4. Concatenate (Stack Vertically)
        # We put the timeline back together: 2020... -> ...2023
        df_tfidf_result = pl.concat([df_train, df_test])

        # 5. Move Date to the first column for better readability
        cols = df_tfidf_result.columns
        cols.remove(date_col)
        df_tfidf_result = df_tfidf_result.select([date_col] + cols)
        return df_tfidf_result

    def analyse_results(self, df_to_analyse, date_col = "trading_session_date_utc"):    
        # Display results
        df_to_analyse = df_to_analyse
        print(f"Final Shape: {df_to_analyse.shape}")
        with pl.Config(fmt_str_lengths=1000, tbl_width_chars=1000, tbl_cols=-1, tbl_rows=5):
            print("Combined TF-IDF DataFrame:")
            display(df_to_analyse)

        # Calculate mean TF-IDF score per word (column) using Polars
        mean_scores = df_to_analyse.drop(date_col).mean()

        # Convert to a DataFrame with 'word' and 'average_tfidf' columns
        word_scores = pl.DataFrame({
            "word": mean_scores.columns,
            "average_tfidf": mean_scores.row(0)
        })

        # Sort and display top 20 words
        top_words = word_scores.sort("average_tfidf", descending=True).head(20)

        print("\n--- TOP 20 WORDS BY IMPORTANCE ---")
        with pl.Config(tbl_rows=20):
            print(top_words)

        # Sanity check for 'the'
        if "the" in word_scores["word"].to_list():
            print("WARNING: The word 'the' is still in your dataset. Preprocessing (Stop Word Removal) failed.")
        else:
            print("SUCCESS: The word 'the' was successfully removed.")

class OptimizedFinBERT():
    def __init__(self, model_name="ProsusAI/finbert", hf_token=None, parallel_gpus=False, device_id=0):
        self.hf_token = hf_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=self.hf_token)
        
        # 1. Load the model temporarily into RAM first
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, output_hidden_states=True, token=self.hf_token
        )
        
        if parallel_gpus and torch.cuda.device_count() > 1:
            self.device = torch.device("cuda:0") # Main orchestrating device
            print(f"⚖️ Using BalancedDataParallel (640/384 split) across {torch.cuda.device_count()} GPUs!")
            self.model = BalancedDataParallel(gpu0_bs=640, module=model)
        else:
            if torch.cuda.is_available() and device_id < torch.cuda.device_count():
                self.device = torch.device(f"cuda:{device_id}")
                print(f"🎯 Locking FinBERT strictly to GPU {device_id} ({torch.cuda.get_device_name(device_id)})")
            else:
                self.device = torch.device("cpu")
                print("⚠️ CUDA unavailable or invalid device_id. Using CPU.")
            self.model = model
        
        self.model = self.model.to(self.device)
        self.model.eval()

    def _group_articles_by_trading_day(self, df_or_lazy, cutoff_hour=20):
        """Grouping articles by trading session day 
        Strategy: NYSE closes at 4PM EST.
        - Articles BEFORE 4PM EST on day D -> assigned to day D (predict D+1 closing price)
        - Articles AFTER  4PM EST on day D -> assigned to day D+1 (predict D+2 closing price)        
        Args:
            cutoff_hour: Market close hour in EST (default: 16 = 4PM) used for TF-IDF preprocessing."""
        if isinstance(df_or_lazy, pl.DataFrame):
            lf = df_or_lazy.lazy()
        elif isinstance(df_or_lazy, pl.LazyFrame):
            lf = df_or_lazy
        else:
            raise ValueError("Input must be Polars DataFrame or LazyFrame")

        # 2. Check if already done
        schema = lf.collect_schema()
        if "trading_session_date_utc" in schema.names():
            return lf
        print("   Mapping to NYSE trading sessions (Lazy)...")
        
        bounds = lf.select([
            pl.col("date_utc").min().alias("min"),
            pl.col("date_utc").max().alias("max")
        ]).collect() # This triggers a scan, but returns only 2 values. Safe.
        min_date = bounds["min"][0].date()
        max_date = bounds["max"][0].date() + timedelta(days=10) # Buffer for weekends

        nyse = mcal.get_calendar("NYSE")
        sched = nyse.schedule(start_date=min_date, end_date=max_date)


        valid_days = pl.DataFrame({
            "trading_date": [d.date() for d in sched.index]
        }).lazy().set_sorted("trading_date")

        # First, shift late articles to the next calendar day
        lf = lf.with_columns(
            pl.when(pl.col("date_utc").dt.hour() < cutoff_hour)
            .then(pl.col("date_utc").dt.date())
            .otherwise(pl.col("date_utc").dt.date() + pl.duration(days=1))
            .alias("candidate_date")
        )
        lf = lf.sort("candidate_date")

        # Perform AsOf Join to find the next valid trading day
        # strategy="forward" means if candidate_date matches exactly, take it.
        # If not (e.g. Saturday), take the NEXT available date (Monday).
        lf = lf.join_asof(
            valid_days,
            left_on="candidate_date",
            right_on="trading_date",
            strategy="forward"
        )
        
        # Cleanup
        return lf.rename({"trading_date": "trading_session_date_utc"}).drop("candidate_date")

    def process_batch(self, inputs, pooling='mean'):
        # Move pre-tokenized inputs to the target device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
            # This calls BalancedDataParallel.forward, which calls .scatter
            outputs = self.model(**inputs)

        # 1. Get Sentiment Probabilities (Softmax)
        probs = torch.softmax(outputs.logits, dim=1).to(torch.float32).cpu().numpy()
        # 2. Get Embeddings based on Pooling Strategy
        hidden_states = outputs.hidden_states[-1].to(torch.float32) # Last layer: [batch, seq_len, 768]
        
        if pooling == 'cls':
            embeddings = hidden_states[:, 0, :]
        elif pooling == 'mean':
             # 1. Unsqueeze to [Batch, Seq_Len, 1]
             # We do NOT use .expand(). We let PyTorch broadcast automatically.
             mask_expanded = inputs['attention_mask'].unsqueeze(-1)             
             # 2. Sum the embeddings (Broadcasting happens here)
             # [B, 512, 768] * [B, 512, 1] -> Works perfectly without extra RAM
             sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)             
             # 3. Sum the mask to count valid tokens (avoid division by zero)
             # Result is [Batch, 1] (or [Batch, 768] if broadcasted, but we want the count)
             sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)             
             embeddings = sum_embeddings / sum_mask
        
        return embeddings.cpu().numpy(), probs
    
    def apply_model(self, dataset, output_file, backup_dir="/mnt/red/red_hanka_bcthesis/finbert_backups", rows_limit = 0, num_workers=0):
        os.makedirs(backup_dir, exist_ok=True)
        
        # 0. SAFETY CHECK: Ensure backup directory is empty before starting
        existing_chunks = glob.glob(f"{backup_dir}/backup_chunk_*.parquet")
        if existing_chunks:
            print(
                f"\n WARNING: Found {len(existing_chunks)} existing backup chunks in {backup_dir}!\n"
                f"To prevent data corruption or SchemaErrors, please manually delete, move, or rename "
                f"these files before starting a new model run."
            )
            print("Exiting method safely. No data was processed.")
            return None
        
        finBert_df = dataset
        rows_limit = rows_limit
        if rows_limit > 0:
            finBert_df = finBert_df.head(rows_limit)

        BATCH_SIZE = 128
        all_texts = finBert_df["Article_title"]

        # Define a collate function to tokenize on the CPU workers
        def collate_fn(batch):
            return self.tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
        
        # Create the DataLoader with workers to pre-fetch and tokenize
        dataloader = DataLoader(
            NewsDataset(all_texts), 
            batch_size=BATCH_SIZE, 
            num_workers=num_workers,          # Adjust based on your Ryzen core count (4 is safe)
            collate_fn=collate_fn, 
            pin_memory=True         # Speeds up RAM to VRAM transfer
        )

        current_chunk_dfs = [] # Store tiny chunks of DataFrames
        offset = 0   # Track our position in the main DataFrame
        chunk_index = 0
        
        # TQDM gives a progress bar
        print(f"Starting processing of {len(all_texts)} articles...")

        for inputs in tqdm(dataloader, desc="Processing Batches"):
            batch_emb, batch_prob = self.process_batch(inputs, pooling='mean')            
            
            current_batch_size = batch_emb.shape[0]
            batch_df = finBert_df.slice(offset, current_batch_size)
            
            batch_df = batch_df.with_columns([
                pl.Series("finbert_embedding", batch_emb.tolist()), 
                pl.Series("sent_pos", batch_prob[:, 0]),
                pl.Series("sent_neg", batch_prob[:, 1]),
                pl.Series("sent_neu", batch_prob[:, 2])
            ])
            
            current_chunk_dfs.append(batch_df)
            offset += current_batch_size
            
            # 2. CHECKPOINT LOGIC: Every ~50,000 rows, flush to disk!
            if offset >= (chunk_index + 1) * 50000:
                # Stitch the small chunk together
                temp_df = pl.concat(current_chunk_dfs)
                # Save it to the hard drive
                temp_df.write_parquet(f"{backup_dir}/backup_chunk_{chunk_index}.parquet")
                
                # 3. FREE THE RAM
                current_chunk_dfs = [] 
                chunk_index += 1

        # 4. Save any remaining rows at the very end of the loop
        if current_chunk_dfs:
            temp_df = pl.concat(current_chunk_dfs)
            temp_df.write_parquet(f"{backup_dir}/backup_chunk_{chunk_index}.parquet")

        print("All batches processed! Reloading and stitching chunks from disk...")
        
        valid_files = []
        
        chunk_files = sorted(glob.glob(f"{backup_dir}/backup_chunk_*.parquet"), 
                             key=lambda x: int(x.split('_')[-1].split('.')[0]))

        for file in chunk_files:
            try:
                # Try to read the file schema to ensure it's not corrupted
                pl.read_parquet_schema(file)
                valid_files.append(file)
            except Exception as e:
                print(f" Skipping {os.path.basename(file)} (Likely corrupted/incomplete from crash)")

        if valid_files:
            print(f"\nStitching {len(valid_files)} valid chunks out-of-core (low RAM)...")
            # scan_parquet creates a LazyFrame, sink_parquet streams it directly to disk
            pl.scan_parquet(valid_files).sink_parquet(output_file)

            print(f" Successfully saved stitched data to: {output_file}")            
            # Return a LazyFrame pointing to the completed file
            finbert_result_lazy = pl.scan_parquet(output_file)
            final_count = pl.scan_parquet(output_file).select(pl.len()).collect().item()
            print(f"\nStitched DataFrame Total Rows: {final_count}")
            print("Done!")
            return finbert_result_lazy
        else:
            print("No valid chunks were found!")
            return None
    
    def test_predictions(self, dataset_results):
        # 1. Select just 2 examples from your actual dataset
        finBert_df_results = dataset_results
        verification_df = finBert_df_results.head(2)
        test_texts = verification_df["Article_title"].to_list()

        print(f"--- Starting Verification on {len(test_texts)} samples from dataset ---")

        # 3. Process the batch (Using 'mean' pooling as decided)
        embeddings, sentiments = self.process_batch(test_texts, pooling='mean')

        # --- CHECK 1: The Shapes ---
        print(f"\n OUTPUT SHAPES:")
        print(f"Embeddings Shape: {embeddings.shape}")
        print(f"Sentiments Shape: {sentiments.shape}")

        # ASSERTIONS
        # We expect (2 samples, 768 features)
        assert embeddings.shape == (2, 768), f"ERROR: Expected embedding shape (2, 768), got {embeddings.shape}"
        # We expect (2 samples, 3 sentiment classes)
        assert sentiments.shape == (2, 3), f"ERROR: Expected sentiment shape (2, 3), got {sentiments.shape}"

        # --- CHECK 2: The Logic (Do the numbers make sense?) ---
        print(f"\nLOGIC CHECK:")

        # Check if sentiment probabilities sum to 1.0
        sums = np.sum(sentiments, axis=1)
        print(f"Probability Sums (Should be ~1.0): {sums}")
        assert np.allclose(sums, 1.0, atol=1e-5), " ERROR: Sentiment probabilities do not sum to 1!"

        # FinBERT Output Order: [Positive, Negative, Neutral] (Standard for ProsusAI/finbert)
        labels = ["Positive", "Negative", "Neutral"]
        print("\n--- DETAILED INSPECTION ---")
        for i, text in enumerate(test_texts):
            top_class_index = np.argmax(sentiments[i]) # Which score is highest?
            print(f"Text: '{text}'")
            print(f" -> Prediction: {labels[top_class_index]} ({sentiments[i][top_class_index]:.4f})")
            print(f" -> Scores: POS={sentiments[i][0]:.2f}, NEG={sentiments[i][1]:.2f}, NEU={sentiments[i][2]:.2f}")

        print("\n VERIFICATION SUCCESSFUL: Pipeline works on real data.")

    def test_predictions_dates(self, dataset_results, dataset_name = 'finBert_results'):
        dataset_results = dataset_results
        date_counts_results = (
            dataset_results
            .group_by("date_utc")
            .agg(pl.len().alias("row_count"))
            .sort("date_utc")
        )
        print(f"--- {dataset_name}: Rows per Date ---")
        display(date_counts_results)

    def aggregate_daily_sentiment(self, df_or_path, use_trading_days=True, cutoff_hour=20):
        """
        Aggregates article-level sentiment scores into Daily Averages.
        Useful for Logistic Regression and Random Forest.

        Args:
        use_trading_days: If True, groups by NYSE trading sessions instead of calendar days
        cutoff_hour: Market close hour (only used if use_trading_days=True)
        """
        print(" Aggregating sentiment scores by Date...")
        
        # Handle input: If it's a string (path), scan it. If it's a DF, make it Lazy.
        if isinstance(df_or_path, str):
            lf = pl.scan_parquet(df_or_path)
        else:
            lf = df_or_path.lazy()

        # 🚀 FIX 1: Immediately drop the heavy embeddings! 
        # We only keep the 4 columns we actually need for sentiment math.
        # This instantly turns a 25GB operation into a ~50MB operation.
        columns_to_keep = ["date_utc", "sent_pos", "sent_neg", "sent_neu"]
        
        # Also keep the trading session column if it already exists
        schema = lf.collect_schema()
        if "trading_session_date_utc" in schema.names():
            columns_to_keep.append("trading_session_date_utc")
            has_trading_col = True
        else:
            has_trading_col = False
            
        # Select only the lightweight columns
        lf = lf.select(columns_to_keep)

        # Optional: Map to trading days first
        if use_trading_days:
            if has_trading_col:
                print(" Using existing 'trading_session_date_utc' column...")
                df_trading = lf
                group_col = "trading_session_date_utc"
            else:
                print(" Mapping to NYSE trading sessions...")
                # 🚀 FIX 2: Pass the LazyFrame (lf) directly! Do NOT use .collect() here.
                # Your _group_articles_by_trading_day is already written to handle LazyFrames perfectly.
                df_trading = self._group_articles_by_trading_day(lf, cutoff_hour=cutoff_hour)
                group_col = "trading_session_date_utc"
        else:
            # If not using trading days, we need a Date column to group by
            df_trading = lf.with_columns(pl.col("date_utc").dt.date().alias("date_utc"))
            group_col = "date_utc"


        daily_df = (
            df_trading
            .group_by(group_col)
            .agg([
                # Calculate the Mean (Average) for each probability
                pl.col("sent_pos").mean().alias("avg_pos"),
                pl.col("sent_neg").mean().alias("avg_neg"),
                pl.col("sent_neu").mean().alias("avg_neu"),
                
                # Count how many articles appeared that day
                # (Volume is a strong signal for volatility)
                pl.len().alias("daily_article_count") 
            ])
            .sort(group_col)
            .collect()
        )
        
        print(f"Aggregated into {len(daily_df)} daily rows.")
        display(daily_df)
        return daily_df
    
    def aggregate_daily_embeddings(self, df_or_path, use_trading_days=True, cutoff_hour=20):
        """
        Aggregates [N_articles, 768] into [1_day, 768] using Mean Pooling.
        Pure Polars implementation (No Pandas).
        """
        print(" Aggregating Embeddings by Date (Mean Pooling via Polars)...")
        # 1. Define the schema explicitly (FinBERT = 768 dimensions)
        # We name them e_0, e_1... e_767

        # 1. Handle input safely
        if isinstance(df_or_path, str):
            lf = pl.scan_parquet(df_or_path)
        else:
            lf = df_or_path.lazy()

        embedding_size = 768
        field_names = [f"e_{i}" for i in range(embedding_size)]
        
           # --- STRATEGY: Lightweight Indexing ---
        # We don't want to shuffle 768-float vectors around just to find dates.
        # We create a lightweight version of the data to calculate the mapping.
        
        print("   1. Building lightweight date map (Ignoring embeddings)...")
        
        # Create a lightweight Frame with just Row ID and Date
        lf_map = lf.select(pl.col("date_utc"))
        
        group_col = "Date"
        
        if use_trading_days:
            # Apply the heavy logic (Sort + JoinAsOf) on the LIGHTWEIGHT frame
            # This is fast because it only moves dates and ints, not embeddings.
            lf_map = self._group_articles_by_trading_day(lf_map, cutoff_hour)
            group_col = "trading_session_date_utc"
        else:
            lf_map = lf_map.with_columns(pl.col("date_utc").dt.date().alias("Date"))

        # Collect the map to memory (It's small: ~2M rows * 2 columns = ~30MB)
        print("   2. Collecting date map to memory...")
        df_map = lf_map.collect()
        
        # Get unique dates from the map
        unique_dates = df_map[group_col].unique().sort()
        print(f"   Found {len(unique_dates)} unique {group_col}s.")
        
        daily_results = []
        
        print(f"   3. Processing days individually (Fetching by Row ID)...")
        
        #daily_results = [] # List to hold tiny 1-row dataframes

        # --- CHANGE 2: The Loop (Crucial for RAM) ---
        # OLD CODE: df_daily_emb = lf.select(...).unnest()...
        # NEW CODE: We iterate through 'unique_dates' and touch only one day at a time.
        #print(f"   Processing {len(unique_dates)} days or trading days individually...")
        
        for single_date in tqdm(unique_dates, desc="Daily Aggregation"):
           # A. Find which row indices belong to this date (using the small DF)
            # This is instant
            #target_row_ids = df_map.filter(pl.col(group_col) == single_date)["row_idx"]
            
            #if len(target_row_ids) == 0: continue

            # B. Fetch ONLY those rows from the heavy LazyFrame
            # slicing by row_index is efficient in Parquet if chunks align, 
            # but filtering by ID is safe and strictly filters BEFORE loading embeddings.
            
            day_map = df_map.filter(pl.col(group_col) == single_date)
            if len(day_map) == 0: continue
            
            min_dt = day_map["date_utc"].min()
            max_dt = day_map["date_utc"].max()

            daily_mean = (
                lf
                #.with_row_index("row_idx") # Re-generate indices on the heavy frame
                #.filter(pl.col("row_idx").is_in(target_row_ids))
                .filter(pl.col("date_utc").is_between(min_dt, max_dt))
                .select([
                    # pl.col("Date"),
                    # 1. Convert List to Struct, then Unnest to create 768 columns (field_0, field_1...)
                    pl.col("finbert_embedding")
                        .list.to_struct(fields=field_names)
                        .struct.unnest()
                ])
                .mean()# Collapses the 200 rows into 1 single average row

                # --- CHANGE 4: Re-pack immediately ---
                # We zip the 768 columns back into a list RIGHT NOW.
                # This keeps the result tiny (1 row, 1 column) before we store it.
                .select([
                    pl.lit(single_date).alias(group_col), # Add the date label back
                    pl.concat_list(pl.col(r"^e_\d+$")).alias("daily_embedding")
                ])
                .collect() # Execute this tiny chunk immediately
            )
            daily_results.append(daily_mean)

            gc.collect()
        # --- CHANGE 5: Stitch at the end ---
        # Instead of one giant .collect() at the end, we concat the tiny results.
        print("   Stitching (trading) days together...")
        if not daily_results:
            print(" No data was processed!")
            return None
        
        final_df = pl.concat(daily_results)        
        print(f"Created (trading) Daily Embeddings: {len(final_df)} (trading) days.")
        display(final_df)
        return final_df
    
    def manual_sentiment_average_check(self, df_average_sentiment, df_original_finbert_results):
        # Check the output structure
        daily_sentiment = df_average_sentiment
        df_original_finbert_results = df_original_finbert_results
        print(f"Shape: {daily_sentiment.shape}")
        print(f"Columns: {daily_sentiment.columns}")

        # VERIFY SENTIMENT PROBABILITIES SUM TO 1.0
        print("\n--- Probability Sum Check ---")
        daily_sentiment_with_sum = daily_sentiment.with_columns([
            (pl.col("avg_pos") + pl.col("avg_neg") + pl.col("avg_neu")).alias("probability_sum")
        ])
        print(daily_sentiment_with_sum.select(["date_utc", "probability_sum"]))

        # Should all be ~1.0
        assert all(daily_sentiment_with_sum["probability_sum"].is_between(0.99, 1.01)), "Probabilities don't sum to 1!"

        # CHECK ARTICLE COUNTS MAKE SENSE
        print("\n--- Article Count Verification ---")
        # Compare aggregated count vs original
        original_count = df_original_finbert_results.group_by("date_utc").agg(pl.len().alias("count")).sort("date_utc")
        agg_count = daily_sentiment.select(["date_utc", "daily_article_count"]).sort("date_utc")

        comparison = original_count.join(agg_count, on="date_utc")
        print(comparison)

        # SPOT CHECK A SPECIFIC DAY
        print("\n--- Spot Check Single Day ---")
        test_date = df_original_finbert_results["date_utc"][0]
        print(f"Testing date: {test_date}")

        # Manual calculation
        day_data = df_original_finbert_results.filter(pl.col("date_utc") == test_date)
        manual_pos = day_data["sent_pos"].mean()
        manual_neg = day_data["sent_neg"].mean()
        manual_neu = day_data["sent_neu"].mean()
        manual_count = len(day_data)

        # From aggregation
        agg_data = daily_sentiment.filter(pl.col("date_utc") == test_date).row(0)
        agg_pos, agg_neg, agg_neu, agg_count = agg_data[1], agg_data[2], agg_data[3], agg_data[4]

        print(f"Manual - POS: {manual_pos:.6f}, NEG: {manual_neg:.6f}, NEU: {manual_neu:.6f}, COUNT: {manual_count}")
        print(f"Agg    - POS: {agg_pos:.6f}, NEG: {agg_neg:.6f}, NEU: {agg_neu:.6f}, COUNT: {agg_count}")

        # VISUAL CHECK
        print("\n--- First 10 Days ---")
        with pl.Config(fmt_str_lengths=1000, tbl_width_chars=1000):
            display(daily_sentiment.head(10))

    def verify_embedding_aggregation(self):
            """
            Sanity Check: Creates fake data with KNOWN values to prove 
            that the averaging logic works mathematically.
            """
            print("\n STARTING UNIT TEST: Embedding Aggregation...")
            
            # --- SCENARIO 1: The Math Check ---
            # Date: 2024-01-01
            # Article A: All values are 0.0
            # Article B: All values are 2.0
            # EXPECTED RESULT: All values must be 1.0 ((0+2)/2)
            vec_zeros = np.zeros(768).tolist()
            vec_twos  = (np.ones(768) * 2).tolist()
            
            # --- SCENARIO 2: The Identity Check ---
            # Date: 2024-01-02
            # Article C: Random numbers
            # EXPECTED RESULT: Exact same random numbers (Average of 1 item is itself)
            vec_random = np.random.rand(768).tolist()
            
            # Create Dummy DataFrame
            dummy_df = pl.DataFrame({
                "date_utc": ["2024-01-01", "2024-01-01", "2024-01-02"],
                "finbert_embedding": [vec_zeros, vec_twos, vec_random]
            }).with_columns(pl.col("date_utc").cast(pl.Date))
            
            print(" Created dummy data. Running aggregation...")
            
            # RUN YOUR FUNCTION
            result_df = self.aggregate_daily_embeddings(dummy_df)
            
            # --- ASSERTIONS (The Proof) ---
            
            # 1. Check Dimensions
            assert len(result_df) == 2, f" Failed: Expected 2 days, got {len(result_df)}"
            print("   Dimension Check Passed (2 Unique Days found).")
            
            # 2. Check Math (Date 1)
            # Get the vector for 2024-01-01. It sits in a list in the first row.
            res_day1 = np.array(result_df.filter(pl.col("date_utc") == date(2024, 1, 1))["daily_embedding"][0])
            
            # We expect all 1.0s. We use allclose to handle floating point tiny errors.
            if np.allclose(res_day1, 1.0):
                print("   Math Check Passed: Average of [0,0..] and [2,2..] is [1,1..]")
            else:
                print(f"   Math Check Failed! Expected 1.0, got mean: {np.mean(res_day1)}")
                return # Stop
                
            # 3. Check Identity (Date 2)
            res_day2 = np.array(result_df.filter(pl.col("date_utc") == date(2024, 1, 2))["daily_embedding"][0])
            
            if np.allclose(res_day2, np.array(vec_random)):
                print("   Identity Check Passed: Single article average matches original.")
            else:
                print("   Identity Check Failed! Values changed unexpectedly.")
                return

            print("\n TEST PASSED: Your aggregation logic is 100% correct.\n")

    def test_sentiment_results(self):
        #test average sentiment on fake df
        # Create fake sentiment data with easily calculatable values
        fake_sentiment_df = pl.DataFrame({
            "date_utc": [
                date(2024, 1, 1), 
                date(2024, 1, 1), 
                date(2024, 1, 1), 
                date(2024, 1, 1), 
                date(2024, 1, 1),
                date(2024, 1, 2)
            ],
            "sent_pos": [0.1, 0.3, 0.5, 0.2, 0.4, 0.6],
            "sent_neg": [0.2, 0.1, 0.2, 0.3, 0.1, 0.2],
            "sent_neu": [0.7, 0.6, 0.3, 0.5, 0.5, 0.2],
            "Article_title": ["Article 1", "Article 2", "Article 3", "Article 4", "Article 5", "Article 6"]
        })

        print("Fake Sentiment Data:")
        display(fake_sentiment_df)

        # Apply aggregation
        result_df = finbert.aggregate_daily_sentiment(fake_sentiment_df)

        print("\nAggregation Result:")
        display(result_df)

        # --- ASSERTIONS ---
        print("\nRunning Assertions...")

        # Test 1: Check shape (2 days)
        assert len(result_df) == 2, f"Expected 2 days, got {len(result_df)}"
        print("Test 1 Passed: Correct number of days")

        # Test 2: Check article counts
        day1_count = result_df.filter(pl.col("date_utc") == date(2024, 1, 1))["daily_article_count"][0]
        day2_count = result_df.filter(pl.col("date_utc") == date(2024, 1, 2))["daily_article_count"][0]

        assert day1_count == 5, f"Expected 5 articles on day 1, got {day1_count}"
        assert day2_count == 1, f"Expected 1 article on day 2, got {day2_count}"
        print("Test 2 Passed: Article counts are correct")

        # Test 3: Verify Day 1 averages (manual calculation)
        # pos: (0.1 + 0.3 + 0.5 + 0.2 + 0.4) / 5 = 1.5 / 5 = 0.3
        # neg: (0.2 + 0.1 + 0.2 + 0.3 + 0.1) / 5 = 0.9 / 5 = 0.18
        # neu: (0.7 + 0.6 + 0.3 + 0.5 + 0.5) / 5 = 2.6 / 5 = 0.52

        day1_data = result_df.filter(pl.col("date_utc") == date(2024, 1, 1)).row(0)
        avg_pos, avg_neg, avg_neu, count = day1_data[1], day1_data[2], day1_data[3], day1_data[4]

        assert np.isclose(avg_pos, 0.3, atol=1e-6), f"Expected avg_pos=0.3, got {avg_pos}"
        assert np.isclose(avg_neg, 0.18, atol=1e-6), f"Expected avg_neg=0.18, got {avg_neg}"
        assert np.isclose(avg_neu, 0.52, atol=1e-6), f"Expected avg_neu=0.52, got {avg_neu}"
        print("Test 3 Passed: Day 1 averages are mathematically correct")
        print(f"   POS: {avg_pos:.2f}, NEG: {avg_neg:.2f}, NEU: {avg_neu:.2f}")

        # Test 4: Verify Day 2 identity (single article, values unchanged)
        day2_data = result_df.filter(pl.col("date_utc") == date(2024, 1, 2)).row(0)
        avg_pos_2, avg_neg_2, avg_neu_2 = day2_data[1], day2_data[2], day2_data[3]

        assert np.isclose(avg_pos_2, 0.6, atol=1e-6), f"Expected avg_pos=0.6, got {avg_pos_2}"
        assert np.isclose(avg_neg_2, 0.2, atol=1e-6), f"Expected avg_neg=0.2, got {avg_neg_2}"
        assert np.isclose(avg_neu_2, 0.2, atol=1e-6), f"Expected avg_neu=0.2, got {avg_neu_2}"
        print("Test 4 Passed: Day 2 identity check (single article remains unchanged)")

        # Test 5: Probabilities sum to ~1.0
        sum_day1 = avg_pos + avg_neg + avg_neu
        sum_day2 = avg_pos_2 + avg_neg_2 + avg_neu_2

        assert np.isclose(sum_day1, 1.0, atol=1e-5), f"Day 1 probabilities don't sum to 1: {sum_day1}"
        assert np.isclose(sum_day2, 1.0, atol=1e-5), f"Day 2 probabilities don't sum to 1: {sum_day2}"
        print("Test 5 Passed: Probabilities sum to 1.0")

        print("\nALL TESTS PASSED! The aggregate_daily_sentiment function works correctly.")

