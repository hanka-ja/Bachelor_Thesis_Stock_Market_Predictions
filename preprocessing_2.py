import polars as pl
import matplotlib.pyplot as plt
from datetime import date, timedelta, timezone, datetime
from zoneinfo import ZoneInfo
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from IPython.display import display
from datasets import load_dataset
import hashlib
from tqdm.auto import tqdm
import pandas_market_calendars as mcal
import os
# A simpler way with PyArrow Dataset
import pyarrow.dataset as ds
import pyarrow.parquet as pq

class NewsPreprocessing:
    def __init__(self, file_path, stock_symbol=None):
        """
        Initializes the processor, sets up NLTK resources, and defines the file path.
        """
        self.file_path = file_path
        self.stock_symbol = stock_symbol
        self.df = None
        self.dfs = {}
        self.source_names = {}
        
        # Initialize NLP tools
        print("Initializing NLTK resources...")
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
            
        self.stemmer = PorterStemmer()
           
    def load_data(self, file_paths=None, start=None, end=None, 
                col_to_check='Article_title', stock_symbol=None, 
                filter_russian=False, date_col = 'Date'): 
        """
        Step 1: Lazy Load and Filter. 
        Supports CSV/Parquet, multiple files, and optional Russian filtering.
        """
        targets = file_paths if file_paths is not None else self.file_path
        
        # Normalize input to list
        if isinstance(targets, str):
            path_list = [targets]
            is_single_mode = True
        else:
            path_list = targets
            is_single_mode = False
            
        loaded_dfs = []
        self.dfs = {} 
        self.source_names = {} 
        print(f"\n Loading data process started...")

        for i, path in enumerate(path_list):
            filename = str(path).split("/")[-1]
            print(f"   Source: {path}")
            
            # Determine actual stock symbol to use
            target_symbol = stock_symbol if stock_symbol is not None else self.stock_symbol
            print(f"   Filtering for stock: {target_symbol}")

            try:
                # 1. Lazy Scan
                if path.lower().endswith(".csv"):
                    # scan_csv with default safety
                    lazy_frame = pl.scan_csv(path, try_parse_dates=False)                    
                    lazy_frame = lazy_frame.with_columns(
                        pl.col(date_col)
                        .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S UTC", strict=False)                      
                    ).rename({date_col: "date_utc"})                    
                else:
                    lazy_frame = pl.scan_parquet(path)
                    lazy_frame = lazy_frame.with_columns(
                    pl.col(date_col)
                    ).rename({date_col: "date_utc"})

                lazy_frame = lazy_frame.with_columns(
                    pl.col("date_utc").dt.replace_time_zone("UTC")
                )

                lazy_frame = lazy_frame.with_columns(
                    pl.col("date_utc")
                    .dt.convert_time_zone("America/New_York")
                    .alias("date_est")
                )

                # Move date_est to the first column
                lazy_frame = lazy_frame.select(["date_est", pl.exclude("date_est")])
                
                # 2. Apply Stock Filter
                if target_symbol is not None:
                    lazy_frame = lazy_frame.filter(pl.col("Stock_symbol") == target_symbol)

                # 3. Apply Time Filter
                if start:
                    print(f"   Applying start filter: >= {start}")
                    start_utc = start.replace(tzinfo=timezone.utc)
                    lazy_frame = lazy_frame.filter(pl.col("date_utc") >= start_utc)
                if end:
                    print(f"   Applying end filter: <= {end}")
                    end_utc = end.replace(tzinfo=timezone.utc)
                    lazy_frame = lazy_frame.filter(pl.col("date_utc") <= end_utc)

                # 4. Apply Filter to Filter out Russian Text (Conditional)
                if filter_russian:
                    print(f"    Filtering out Russian text in '{col_to_check}'...")
                    lazy_frame = lazy_frame.filter(
                        ~pl.col(col_to_check).str.contains(r"[а-яА-Я]")
                    )

                # 5. Collect (Streaming)
                df = lazy_frame.collect(engine="streaming")
                
                loaded_dfs.append(df)
                
                # Store metadata
                key_name = f"source_{i+1}"
                self.dfs[key_name] = df
                self.source_names[key_name] = filename
                print(f"Loaded {len(df)} rows into '{key_name}'.")

            except Exception as e:
                print(f"Error loading {path}: {e}")
                loaded_dfs.append(None)

        # Return Logic
        if is_single_mode:
            self.df = loaded_dfs[0]
            return self.df
        else:
            if loaded_dfs and loaded_dfs[0] is not None:
                self.df = loaded_dfs[0]
            return loaded_dfs
        
    def load_huggingface_api(self, limit_rows=90):
        """
        Load rows from Hugging Face by STREAMING (Client-Side Filtering).
        """
        
        print(f"\n  Streaming from Hugging Face (Robust Mode)...")
        print(f"   Target: {self.stock_symbol} | Limit: {limit_rows} rows")

        try:
            # 1. Initialize Stream (No Download happens here)
            ds = load_dataset(
                "Zihan1004/FNSPID", 
                data_files="Stock_news/nasdaq_exteral_data.csv", 
                split="train", 
                streaming=True
            )
            
            # 2. Filter Client-Side
            # It scans the file line-by-line over the internet.
            filtered_stream = ds.filter(lambda row: row['Stock_symbol'] == self.stock_symbol)
            
            # 3. Collect Rows
            collected_rows = []
            print("Scanning stream (this works even if API fails)...")
            
            for i, row in enumerate(filtered_stream):
                collected_rows.append(row)
                if i % 1000 == 0:
                    temp_df = pl.DataFrame(collected_rows)
                    temp_df.write_parquet(f"checkpoint_{i}.parquet")
                    print(f"Checkpoint saved at row {i}")
                if len(collected_rows) >= limit_rows:
                    break
            
            if collected_rows:
                # 4. Convert to Polars
                df = pl.DataFrame(collected_rows, infer_schema_length=None)
                
                # Store in dictionary
                key_name = "source_hf_stream"
                self.dfs[key_name] = df
                self.source_names[key_name] = f"HF Stream (Verified: {self.stock_symbol})"
                self.df = df
                
                print(f"Success! Streamed {len(df)} rows.")
                print(f"Focus switched to {key_name}")
                display (df)
                return df
            else:
                print(f"Stream finished but found no rows for {self.stock_symbol}")
                return None

        except Exception as e:
            print(f"Error during streaming: {e}")
            return None

    def check_features(self, target_df=None):
        """Checks features of the provided dataframe or self.df"""
        df = target_df if target_df is not None else self.df
        if df is None: return

        print(f"\nInspecting Data ({len(df)} rows)...")

        if "date_est" in df.columns:
            date_expr = pl.col("date_est").cast(pl.Date)
        # elif "date_utc" in df.columns:
        #     date_expr = pl.col("date_utc").cast(pl.Date)
        # else:
        #     date_expr = pl.col("Date").cast(pl.Date)

        try:
            # Compute stats on the fly without modifying original DF
            stats = df.select([
                date_expr.n_unique().alias("distinct_days"),
                date_expr.min().alias("min_date"),
                date_expr.max().alias("max_date")
            ]).to_dict(as_series=False)
            
            num_days = stats["distinct_days"][0]
            min_d = stats["min_date"][0]
            max_d = stats["max_date"][0]             
            
            print(f"Number of distinct days: {num_days}")
            print(f"Date range: {min_d} to {max_d}")
            if min_d and max_d:
                time_span = (max_d - min_d).days
                print(f"Time span: {time_span} days")
                
        except Exception as e:
            print(f"Could not calculate strict date stats: {e}")
            print(f"Raw Date Col Type: {df.schema.get('date_est')}")

        print("\nSample Data:")
#        with pl.Config():
        display(df.sort("date_est"))

    def visualize_distribution(self, target_df=None, label="Article Count", start=None, end=None):
        """Plot Daily Article Counts with Dynamic Y-Axis Scaling"""

        df = target_df if target_df is not None else self.df
        if df is None:
            print("No data to visualize.")
            return
        
        print(f"\nGenerating Article Distribution Chart ({label})...")

        if "date_est" in df.columns:
            date_expr = pl.col("date_est").cast(pl.Date)
        # elif "date_utc" in df.columns:
        #     date_expr = pl.col("date_utc").cast(pl.Date)
        # elif df.schema.get("Date") == pl.Utf8:
        #     date_expr = (
        #         pl.col("Date")
        #         .str.replace(r"\+.*$", "")
        #         .str.replace(" UTC", "")
        #         .str.strptime(pl.Datetime, strict=False)
        #         .cast(pl.Date)
        #     )
        # else:
        #     date_expr = pl.col("Date").cast(pl.Date)
        
        # 2. Group and Count
        daily_counts = (
            df
            .with_columns(date_expr.alias("date_est"))
            .group_by("date_est")
            .len()
            .rename({"len": "count"})
            .sort("date_est")
        )

        # 3. Apply Filters (Crucial Step)
        # We handle string inputs for start/end just in case
        if start is not None:
            if isinstance(start, str):
                start = datetime.strptime(start, "%Y-%m-%d").date()
            daily_counts = daily_counts.filter(pl.col("date_est") >= start)
            
        if end is not None:
            if isinstance(end, str):
                end = datetime.strptime(end, "%Y-%m-%d").date()
            daily_counts = daily_counts.filter(pl.col("date_est") <= end)

        # Check if data exists after filtering
        if daily_counts.is_empty():
            print(f"No data found in range {start} to {end}.")
            return

        # 4. Extract Data for Plotting
        dates = daily_counts["date_est"].to_numpy()
        counts = daily_counts["count"].to_numpy()

        plt.figure(figsize=(12, 5)) # Slightly wider for better visibility
        plt.bar(dates, counts, color="steelblue", label=label)
        
        plt.title(f"{label} per Day ({self.stock_symbol})")
        plt.xlabel("Date_EST")
        plt.ylabel("Count")

        # --- 5. Dynamic Y-Axis Scaling ---
        # Calculate the max count ONLY within the visible filtered range
        visible_max = counts.max()
        
        # Set Y-limit to 10% higher than the max bar for better aesthetics
        # If max is 0 (unlikely here due to check above), default to 1
        top_limit = visible_max * 1.1 if visible_max > 0 else 1.0
        plt.ylim(0, top_limit)
        
        # Apply X-limits (Optional, since we filtered data, but good for empty gap visualization)
        if start or end:
            plt.xlim(left=start, right=end)

        plt.grid(axis='y', linestyle='--', alpha=0.5) # Grid helps readability
        plt.tight_layout()
        plt.show()

    def check_for_duplicates(self, source_1, source_2):
        """Check for duplicate articles between two sources within a date range - used to test whether 
        data news from two sources differ or not."""
        print("\nIncluding new row to test for identification of the differences...")
        _ext_cols = source_1.columns
        new_row = {c: None for c in _ext_cols}
        new_row["date_est"] = datetime(2020, 2, 1, 16, 0, 5, tzinfo=ZoneInfo("America/New_York"))
        new_row["Article_title"] = "MANUALLY ADDED TEST ARTICLE FOR DUPLICATE CHECKING"

        dt_col_type = source_1.schema["date_est"]

        source_1 = pl.concat(
            [
                source_1,
                pl.DataFrame([new_row]).with_columns(pl.col("date_est").cast(dt_col_type))
            ],
            how="vertical"
        )

        print("Added 1 test article to external source on 2020-02-01 (dtype aligned).")

        # 1) Filter both sources for the date window (inclusive)
        start = date(2020, 1, 1, 16, 0, 5,)
        end   = date(2020, 7, 1, 16, 0, 5,)
        #NOTE: need to be fixed to be set dynamically based on data range

        nasdaq_win = source_1.filter(
            (pl.col("date_est").cast(pl.Datetime) >= start) & (pl.col("date_est").cast(pl.Datetime) <= end)
        )
        external_win = source_2.filter(
            (pl.col("date_est").cast(pl.Datetime) >= start) & (pl.col("date_est").cast(pl.Datetime) <= end)
        )

        # 2) Compare by Article_title
        titles_nasdaq = set(nasdaq_win["Article_title"].to_list())
        titles_ext    = set(external_win["Article_title"].to_list())

        intersection = titles_nasdaq & titles_ext
        same_count = len(intersection)

        # 3) Collect differing rows (align columns before concat)
        diff_nasdaq = (
            nasdaq_win
            .filter(~pl.col("Article_title").is_in(intersection))
            .select([pl.col("date_est").cast(pl.Datetime).alias("date_est"), "Article_title"])
            .with_columns(pl.lit("nasdaq").alias("source"))
        )

        diff_ext = (
            external_win
            .filter(~pl.col("Article_title").is_in(intersection))
            .select([pl.col("date_est").cast(pl.Datetime).alias("date_est"), "Article_title"])
            .with_columns(pl.lit("external").alias("source"))
        )

        diff_df = pl.concat([diff_nasdaq, diff_ext], how="vertical")
        diff_count = len(diff_df)

        print(f"Same articles (by title): {same_count}")
        print(f"Different articles: {diff_count}")

#        with pl.Config(fmt_str_lengths=200, tbl_width_chars=1000, tbl_rows=20):
        display(diff_df)

    def save_data_into_new_file(self, target_df=None, filename="mistake.parquet"):
        """Step 5: Save to Parquet"""
        df = target_df if target_df is not None else self.df
        if df is None: return

        default_name = f"{self.stock_symbol}_ready_for_tfidf.parquet"
        print(f"\n Save Result")
        print("   Please specify the filename and ending either .parquet or .csv")
        filename = filename.strip()
        
        if not filename: 
            filename = default_name
        filepath = os.path.join("full_news", filename)
        try:
            if filename.lower().endswith(".csv"):                
                df_csv = df.with_columns(
                    pl.col("date_utc").cast(pl.Utf8).alias("date_utc")
                )
                df_csv.write_csv(filepath)
                print(f"Saved to CSV: {filepath}")
                print(f"Note: date_utc column converted to string format for CSV compatibility.")
            elif filename.lower().endswith(".parquet"):
                df.write_parquet(filepath)
                print(f"Saved to Parquet: {filepath}")
            else:
                # Default fallback if no extension provided or unrecognized
                if "." not in filename:
                    filename += ".parquet"
                    df.write_parquet(filepath)
                    print(f"No extension found. Defaulted to Parquet: {filepath}")
                else:
                    print("Unsupported extension. Please use .csv or .parquet")
        except Exception as e:
            print(f"Error saving file: {e}")

    def group_articles_by_trading_day(self, output_path, target_df=None, cutoff_hour=20):
        """Grouping articles by trading session day 
               Strategy: NYSE closes at 4PM EST.
        - Articles BEFORE 4PM EST on day D -> assigned to day D (predict D+1 closing price)
        - Articles AFTER  4PM EST on day D -> assigned to day D+1 (predict D+2 closing price)        
        Args:
            cutoff_hour: Market close hour in EST (default: 16 = 4PM) used for TF-IDF preprocessing."""
        
        input_data = target_df if target_df is not None else self.df
        if input_data is None: return None

        # 1. Convert to LazyFrame if it isn't already to avoid RAM spikes
        if isinstance(input_data, pl.DataFrame):
            lf = input_data.lazy()
        elif isinstance(input_data, pl.LazyFrame):
            lf = input_data
        elif isinstance(input_data, str):
            if input_data.endswith('.csv'):
                lf = pl.scan_csv(input_data)
            else:
                lf = pl.scan_parquet(input_data)
        else:
            print("Unsupported data type for target_df")
            return None

        # build NYSE trading day lookup covering your data range
        # 2. Extract boundaries lazily
        bounds = lf.select([
            pl.col("date_utc").min().alias("min"),
            pl.col("date_utc").max().alias("max")
        ]).collect() # Small safe execution limit

        min_dt = bounds["min"][0]
        max_dt = bounds["max"][0]

        nyse = mcal.get_calendar("NYSE")
        sched = nyse.schedule(
            start_date=min_dt.date() - timedelta(days=5), 
            end_date=(max_dt.date() + timedelta(days=10))
        )

        # Convert to DataFrame for joining
        trading_days_df = pl.DataFrame({
            "trading_date": [d.date() for d in sched.index]
        }).lazy().set_sorted("trading_date")

        print(f"\n Grouping articles by trading session (cutoff: {cutoff_hour}:00 UTC)...")
        
        grouped_lf = (
            lf.with_columns(
                pl.when(pl.col("date_utc").dt.hour() < cutoff_hour)
                .then(pl.col("date_utc").dt.date())
                .otherwise(pl.col("date_utc").dt.date() + pl.duration(days=1))
                .alias("candidate_session_date")
            )
            .sort("candidate_session_date")
            .join_asof(
                trading_days_df,
                left_on="candidate_session_date",
                right_on="trading_date",
                strategy="forward"
            )
            .rename({"trading_date": "trading_session_date_utc"})
            .group_by("trading_session_date_utc")
            .agg(
                pl.col("Article_title").str.join(delimiter=" ").alias("daily_text")
            )
            .sort("trading_session_date_utc")
        )
        print("Executing lazy join and grouping operations via Stream Engine to conserve RAM...")

        # 4. Stream evaluation (keeps memory flat instead of blowing up) - still pointlessly too big for now
        #grouped_df = grouped_lf.collect(streaming=True)

        print("Returning LazyFrame.")
        grouped_lf.sink_parquet(output_path)

        print(f"Streaming complete. Data saved to {output_path}")

        if target_df is None:
            self.df = grouped_lf
            
        return grouped_lf

    def _clean_stem_logic(self, text):
        """Internal helper for NLTK operations (runs inside map_elements)"""
        if not text: 
            return ""
        
        # 1. Manual check (Polars did lowercase & regex, but map_elements might need safeguards)
        # Since we did regex in Polars, 'text' here is already clean string
        words = text.split()

        # 2. Remove Stopwords & Stem
        # Logic: Keep word ONLY IF it is NOT in stop_words
        filtered = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
        
        return " ".join(filtered)

    def clean_text(self, input_path=None, output_path=None, chunk_size=50000):
        """Clean provided df or self.df"""
        #NOTE: maybe in preprocessing consider using some bert tokenizer or something for preprocessing? 
        #maybe the results will be better?
        # Determine input
        if input_path is None:
            print("Please provide an input_path referencing your joined parquet/csv file.")
            return None
        
        if output_path is None:
            output_path = input_path.replace(".parquet", "_cleaned.parquet").replace(".csv", "_cleaned.parquet")

        print(f"\n Starting Chunker Advanced Text Cleaning (Regex + Stemming)...")
        
        # We will create a temporary folder to hold chunks
        base_dir = os.path.dirname(output_path)
        base_name = os.path.basename(output_path).split('.')[0]
        chunk_dir = os.path.join(base_dir, f"{base_name}_chunks")
        os.makedirs(chunk_dir, exist_ok=True)
        
        print(f" Reading from: {input_path}")
        print(f" Saving to: {output_path}")

        # Lazy scan the input
        if input_path.endswith('.csv'):
            lf = pl.scan_csv(input_path)
        else:
            lf = pl.scan_parquet(input_path)

        # Check if the column exists
        schema = lf.collect_schema()
        if "daily_text" not in schema.names():
            print("\n Column 'daily_text' not found in the file!")
            return None
        
                # Use PyArrow to stream the parquet/csv file in batches
        dataset = ds.dataset(input_path, format="csv" if input_path.endswith('.csv') else "parquet")
        
        # We need a writer
        #writer = None
        
        total_rows = dataset.count_rows()
        print(f" Total rows to process: {total_rows}. Processing in batches of {chunk_size}...")

        # Process batch by batch
        with tqdm(total=total_rows, desc="NLTK Cleaning Chunks") as pbar:
            for i, batch in enumerate(dataset.to_batches(batch_size=chunk_size)):
                chunk_file = os.path.join(chunk_dir, f"chunk_{i:04d}.parquet")                

                # If this chunk already exists from a previous run, skip it!
                if os.path.exists(chunk_file):
                    pbar.update(len(batch))
                    continue

                chunk_df = pl.from_arrow(batch)

                # Apply fast regex
                chunk_df = chunk_df.with_columns(
                    pl.col("daily_text")
                    .str.to_lowercase()
                    .str.replace_all(r"\d+", "num")
                    .str.replace_all(r"[^a-z\s]", "")
                    .alias("temp_clean")
                )
        
                # Slow NLTK pass
                temp_clean_list = chunk_df["temp_clean"].to_list()
                processed = [self._clean_stem_logic(text) for text in temp_clean_list]
                
                # Update DataFrame
                chunk_df = chunk_df.with_columns(
                    pl.Series("preprocessed_for_tfidf", processed)
                ).drop("temp_clean")

                # The file handles opening, writing the footer, and closing instantly.
                chunk_df.write_parquet(chunk_file)
                pbar.update(len(batch))

        print(f" All chunks processed. Stitching them together...")
        
        # Stitch perfectly
        lazy_stitch = pl.scan_parquet(os.path.join(chunk_dir, "*.parquet"))
        lazy_stitch.sink_parquet(output_path)
        
        print(f" Text cleaning complete. Final file saved to: {output_path}")
        
        # Return a lazy frame of the result
        return pl.scan_parquet(output_path)

    def file_hash(filepath):
        """Calculate SHA256 hash of a file without loading it into memory - check for integrity."""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            # Read in 64KB chunks (efficient for disk I/O)
            while chunk := f.read(65536):
                sha256.update(chunk)
        return sha256.hexdigest()

    def run_interactive_pipeline(self):
        """The Interactive Wizard for Preprocessing Steps"""
        print(f"--- STARTING PREPROCESSING FOR {self.stock_symbol} ---", flush=True)
        
        # Load data (handles single or multiple)
        print("   [Optional] Filter for specific stock:")
        stock_symbol = input("   Stock Symbol [Enter for All]: ").strip() or None

        print("   [Optional] Filter Time Period (EST, format: YYYY-MM-DD HH:MM:SS):")  #FIX: Update prompt
        s_txt = input("   Start Datetime (EST, e.g. 2020-01-01 00:00:00) [Enter for All]: ").strip()
        e_txt = input("   End   Datetime (EST, e.g. 2020-12-31 23:59:59) [Enter for All]: ").strip()
            
        s_date, e_date = None, None
        try:
            if s_txt: s_date = datetime.strptime(s_txt, "%Y-%m-%d %H:%M:%S")  #FIX: Require full datetime
            if e_txt: e_date = datetime.strptime(e_txt, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            print(" Invalid format! Using full range.")
        dfs_loaded = self.load_data(start=s_date, end=e_date, stock_symbol=stock_symbol, filter_russian=True)
        
        if not self.dfs and self.df is None:
            print("\n CRITICAL ERROR: No data could be loaded. Please check file paths.", flush=True)
            return None

        # Initial Focus Selection
        if isinstance(dfs_loaded, list) and len(dfs_loaded) > 1:
            print("\n Multiple files loaded:", flush=True)
            for k, v in self.dfs.items():
                fname = self.source_names.get(k, "Unknown File")
                print(f"   - {k} ({fname}): {len(v)} rows")
            
            choice = input(f"Which source to process initially? (default: source_1): ").lower()
            if choice in self.dfs:
                self.df = self.dfs[choice]
                print(f" Focused on {choice}")
            else:
                self.df = self.dfs.get("source_1")
                print(f" Defaulting focus to source_1")
        
        try:
            # Interactive Loop - Wrapped in Try/Except for robust Exit
            while True:
                print("\n------------------------------------------------")
                # Find current source name for display
                current_name = "Unknown"
                for k, v in self.dfs.items():
                    if v is self.df: # Identity check
                        current_name = f"{k} [{self.source_names.get(k, '')}]"
                        break

                if self.df is not None:
                    print(f"🎯 Current Focus: {current_name} ({len(self.df)} rows)")
                else:
                    print("🎯 Current Focus: [NO DATA SELECTED]")

                # Menu reflects order of class methods
                print("1️⃣  Check Features (Inspect Data)")
                print("2️⃣  Visualize Data Distribution")
                print("3️⃣  Check for Duplicates (Compare source_1 vs source_2)")
                print("4️⃣  Save Result")
                print("5️⃣  Group Articles by Day")
                print("6️⃣  Run Text Cleaning")
                print("7️⃣  Switch Active Source Dataframe")
                print("8️⃣  Load External Data from Hugging Face")
                print("❌ Exit (or write X/exit/quit/esc)")
                
                # Capture input safely
                action = input("\nChoose an action: ").strip().lower()

                if not action: continue

                # Logic for Exit (Manual typing)
                if action in ['x', 'exit', 'quit', 'esc', 'escape']:
                    break
                
                target_df = None
                if action in ['1', '2', '4', '5', '6']:
                    print(f"   (Optional) Enter source to target (e.g. source_1).")
                    print(f"   Available: {list(self.dfs.keys())}")
                    target_key = input(f"   Press Enter to use current focus ({current_name.split()[0]}): ").strip()
                    
                    if target_key:
                        if target_key in self.dfs:
                            target_df = self.dfs[target_key]
                            print(f" Targeting '{target_key}' temporarily.")
                        else:
                            print(f" '{target_key}' not found. Using default focus.")

                if action == '1':
                     self.check_features(target_df=target_df)

                elif action == '2':
                    # Ask for optional date range                    
                    print("   [Optional] Filter Time Period (EST, format: YYYY-MM-DD HH:MM:SS):")  #FIX: Update prompt
                    s_txt = input("   Start Datetime (EST, e.g. 2020-01-01 00:00:00) [Enter for All]: ").strip()
                    e_txt = input("   End   Datetime (EST, e.g. 2020-12-31 23:59:59) [Enter for All]: ").strip()
                        
                    s_date, e_date = None, None
                    try:
                        if s_txt: s_date = datetime.strptime(s_txt, "%Y-%m-%d %H:%M:%S")
                        if e_txt: e_date = datetime.strptime(e_txt, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        print(" Invalid format! Using full range.")
                            
                    self.visualize_distribution(target_df=target_df, start=s_date, end=e_date)

                elif action == '3':
                    if len(self.dfs) >= 2:
                        print(f"   Available sources: {list(self.dfs.keys())}")
                        
                        # Custom input for Source 1
                        k1 = input(f"   Enter name for Source 1 (default: source_1): ").strip()
                        if not k1: k1 = "source_1"
                        
                        # Custom input for Source 2
                        k2 = input(f"   Enter name for Source 2 (default: source_2): ").strip()
                        if not k2: k2 = "source_2"

                        if k1 in self.dfs and k2 in self.dfs:
                            s1_name = f"{k1} ({self.source_names.get(k1)})"
                            s2_name = f"{k2} ({self.source_names.get(k2)})"
                            print(f"\n Comparing {s1_name} vs {s2_name}...")
                            self.check_for_duplicates(self.dfs[k1], self.dfs[k2])
                        else:
                            print(f" One or both keys ('{k1}', '{k2}') not found in available sources.")
                    else:
                        print(" Not enough sources loaded for comparison.")

                elif action == '4':
                    filename = input(f"   Enter filename (default: 'mistake.parquet'): ").strip()
                    if not filename:
                        filename = "mistake.parquet"
                    self.save_data_into_new_file(target_df=target_df, filename=filename)

                elif action == '5':
                    self.group_articles_by_trading_day(target_df=target_df)
                
                elif action == '6':
                    self.clean_text(target_df=target_df)
                        
                elif action == '7' or 'switch' in action or action == '🔄':
                    if not self.dfs:
                        print(" No multiple sources available to switch.")
                        continue
                    
                    print("\nAvailable Sources:")
                    for k in self.dfs.keys():
                        fname = self.source_names.get(k, "Unknown File")
                        print(f" - {k} : {fname}")
                    
                    new_source = input("Enter name of source to switch to (e.g., source_1): ").strip()
                    if new_source in self.dfs:
                        self.df = self.dfs[new_source]
                        print(f" Switched focus to {new_source}")
                    else:
                        print(" Invalid source name.")
                
                elif action == '8':
                        self.load_huggingface_api()
                                
                else:
                    print(f"Invalid input: '{action}'. Try again.")                

        except (KeyboardInterrupt, EOFError):
            print("\n Exiting pipeline via Cancel/Esc...")
            
            
        print("\n Preprocessing Finished.")
        return self.df