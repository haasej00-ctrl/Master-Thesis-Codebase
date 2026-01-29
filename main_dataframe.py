import wrds
import pickle
import os
import pandas as pd
from dotenv import load_dotenv
from main_functions import calc_quantity_and_sentiment, calc_quality, calc_CAR

# Function to check if quarter2 immediately follows quarter1
def is_preceding_quarter(q1, y1, q2, y2):
    """
    Check if (q2, y2) immediately follows (q1, y1)
    q1, y1: previous quarter and year
    q2, y2: current quarter and year
    """
    # Q1 follows Q4 of previous year
    if q1 == 4 and q2 == 1 and y2 == y1 + 1:
        return True
    # Q2/Q3/Q4 follow Q1/Q2/Q3 of same year
    elif q2 == q1 + 1 and y2 == y1:
        return True
    return False

# 0.) Initialize
db = wrds.Connection()
load_dotenv()

with open('data/init_dict_with_transcripts.pkl', 'rb') as f:
    init_dict_with_transcripts = pickle.load(f)

# Drop rows with empty transcript lists
print(f"Before dropping empty transcripts: {len(init_dict_with_transcripts)} rows")
init_dict_with_transcripts = init_dict_with_transcripts[
    init_dict_with_transcripts['transcript'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
]
print(f"After dropping empty transcripts: {len(init_dict_with_transcripts)} rows")

# Check and drop tickers with fewer than 3 rows
print(f"\nChecking for tickers with insufficient observations...")
ticker_counts = init_dict_with_transcripts['ticker'].value_counts()
tickers_to_drop = ticker_counts[ticker_counts < 3].index.tolist()
if tickers_to_drop:
    rows_before_ticker_drop = len(init_dict_with_transcripts)
    init_dict_with_transcripts = init_dict_with_transcripts[~init_dict_with_transcripts['ticker'].isin(tickers_to_drop)]
    rows_dropped_ticker = rows_before_ticker_drop - len(init_dict_with_transcripts)
    print(f"Dropping {len(tickers_to_drop)} tickers with <3 observations (removed {rows_dropped_ticker} rows)")
    print(f"Dropped tickers: {', '.join(sorted(tickers_to_drop))}")
else:
    print(f"All remaining tickers have at least 3 observations")

# 1.) Group by ticker and sort by announcement_date
df = init_dict_with_transcripts.sort_values(['ticker', 'announcement_date']).reset_index(drop=True)

# Initialize new columns
new_columns = {
    'AI_quantity': None, 'AI_related_text_elements': None, 'negativity_score': None,
    'neg_change': None, 'AI_quality': None, 'returns_volatility': None, 'avg_share_turnover': None,
    'CAR_1': None, 'CAR_3': None, 'CAR_15': None, 'CAR_30': None
}
for col in new_columns:
    df[col] = None

print('Initialized new columns.')
print(f'Looping over {len(df)} rows...')

# 2.) Loop over rows
previous_ticker = None
previous_negativity_score = None

for idx, row in df.iterrows():
    # Reconnect to database every 100 rows to prevent timeout
    if (idx + 1) % 100 == 0:
        print(f"  Refreshing database connection...")
        db.close()
        db = wrds.Connection(
            wrds_username=os.getenv('WRDS_username'),
            wrds_password=os.getenv('WRDS_password')
        )
    
    # Print progress every 100 rows or on the last row
    if (idx + 1) % 100 == 0 or (idx + 1) == len(df):
        print("=" * 60, flush=True)
        print(f"PROGRESS: {idx+1}/{len(df)} completed ({(idx+1)/len(df)*100:.1f}%)")
        print("=" * 60, flush=True)
    
    # 2.1) Calculate quantity and sentiment
    relative_keyword_count, AI_related_text_elements, negativity_score = calc_quantity_and_sentiment(
        text_elements=row['transcript']
    )

    df.at[idx, 'AI_quantity'] = relative_keyword_count
    df.at[idx, 'AI_related_text_elements'] = AI_related_text_elements
    df.at[idx, 'negativity_score'] = negativity_score

    # 2.2) Check if row has preceding quarter
    if idx == 0:
        # First row overall - no previous quarter
        previous_ticker = row['ticker']
        previous_negativity_score = negativity_score
        continue

    prev_row = df.iloc[idx - 1]

    if row['ticker'] != prev_row['ticker']:
        # First occurrence of this ticker - no previous quarter for this ticker
        previous_ticker = row['ticker']
        previous_negativity_score = negativity_score
        continue

    # Same ticker - check if previous quarter is the immediate predecessor
    if not is_preceding_quarter(prev_row['fiscal_quarter'], prev_row['fiscal_year'],
                                row['fiscal_quarter'], row['fiscal_year']):
        # Gap in quarters - skip this row
        previous_ticker = row['ticker']
        previous_negativity_score = negativity_score
        continue

    # If we reach here, we have a valid preceding quarter

    # 2.3) Calculate neg_change
    # Check if either current or previous negativity_score is None
    if previous_negativity_score is None or negativity_score is None:
        df.at[idx, 'neg_change'] = None
    else:
        df.at[idx, 'neg_change'] = negativity_score - previous_negativity_score
    
    # 2.4) Calculate quality
    try:
        df.at[idx, 'AI_quality'] = calc_quality(
            AI_related_text_elements=AI_related_text_elements,
            call_date=row['call_date']
        )
    except Exception as e:
        print(f"  Error calculating quality: {e}")
    
    # Set AI_quality to 0 if it's None (for non-first rows only)
    if df.at[idx, 'AI_quality'] is None:
        df.at[idx, 'AI_quality'] = 0
    
    # 2.5) Calculate CAR
    try:
        CAR_data, crsp_fund = calc_CAR(
            date=row['call_date'],
            ticker=row['ticker'],
            db=db
        )
        
        # Store CAR data for each time window
        for car_entry in CAR_data:
            days = int(car_entry['Time window'].split(':')[1].split()[0])
            df.at[idx, f'CAR_{days}'] = car_entry['CAR']
        
        # Store fundamental data
        df.at[idx, 'returns_volatility'] = crsp_fund['returns_volatility']
        df.at[idx, 'avg_share_turnover'] = crsp_fund['avg_share_turnover']
    except Exception as e:
        print(f"  Error calculating CAR: {e}")
    
    # Update tracking variables for next iteration
    previous_ticker = row['ticker']
    previous_negativity_score = negativity_score

# Save dataframe before dropping rows
df.to_pickle('data/main_dataframe_before_dropping.pkl')

# Drop all rows where period_year equals 2014
print(f"\nBefore filtering period_year: {len(df)} rows")
df = df[df['period_year'] != 2014].copy()
print(f"After filtering period_year == 2014: {len(df)} rows")

# Drop all rows with any None values
print(f"Before dropping rows with None values: {len(df)} rows")
df = df.dropna()
print(f"After dropping rows with None values: {len(df)} rows")

# Check and drop tickers with fewer than 2 rows (2 are needed for fixed effects)
print(f"\nChecking for tickers with insufficient observations...")
ticker_counts = df['ticker'].value_counts()
tickers_to_drop = ticker_counts[ticker_counts < 2].index.tolist()

if tickers_to_drop:
    rows_before_ticker_drop = len(df)
    df = df[~df['ticker'].isin(tickers_to_drop)]
    rows_dropped_ticker = rows_before_ticker_drop - len(df)
    print(f"Dropping {len(tickers_to_drop)} tickers with <2 observations (removed {rows_dropped_ticker} rows)")
    print(f"Dropped tickers: {', '.join(sorted(tickers_to_drop))}")
else:
    print(f"All tickers have at least 2 observations for fixed effects regression")

# Convert call_date to datetime format
df['call_date'] = pd.to_datetime(df['call_date'])

# Save final dataframe
df.to_pickle('data/main_dataframe.pkl')
print(f"\nComplete. Dataset: {len(df)} rows, {len(df.columns)} columns")