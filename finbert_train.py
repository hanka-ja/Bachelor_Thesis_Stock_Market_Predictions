import os
os.sched_setaffinity(0, {0})
# Also keep your GPU and general thread limits just in case
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1" # Must be set before 'import polars'

from dotenv import load_dotenv
import polars as pl
from datetime import datetime
from tfidf_finbert import OptimizedFinBERT
from preprocessing import NewsPreprocessing

# 1. Setup Data
preprocessor = NewsPreprocessing("/mnt/windows/windows_hanka_bcthesis/full_news/cleaned_news_external.parquet")
finb_news = preprocessor.load_data(
    file_paths="/mnt/windows/windows_hanka_bcthesis/full_news/nasdaq_external_news.parquet", 
    start=datetime(2023, 1, 1),
    col_to_check='Article_title', 
    filter_russian=True
)

finbert_df = finb_news.with_row_index("id").select([
    pl.col("id"), pl.col("date_utc").cast(pl.Datetime), pl.col("Article_title"),
]).filter(pl.col("Article_title").is_not_null()).sort("date_utc")

# 2. Setup FinBERT
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
finbert = OptimizedFinBERT(hf_token=hf_token)

# 3. Execute
output_file = "/mnt/red/red_hanka_bcthesis/full_news/finbert_nasdaq_2023_raw.parquet"
print("Starting FinBERT processing...")
finbert.apply_model(finbert_df, output_file=output_file)

# 4. Aggregations
finbert.aggregate_daily_sentiment(output_file).write_parquet(
    "/mnt/red/red_hanka_bcthesis/full_news/finbert_nasdaq_2023_avg_sentiment.parquet"
)
finbert.aggregate_daily_embeddings(output_file).write_parquet(
    "/mnt/red/red_hanka_bcthesis/full_news/finbert_nasdaq_2023_avg_embeddings.parquet"
)
print("Finished completely.")