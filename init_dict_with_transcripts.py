import pandas as pd
import requests
import os
from dotenv import load_dotenv
import pickle
from datetime import datetime, timedelta

# Load environment variables from .env file
load_dotenv()
API_Ninjas_key = os.getenv('API_Ninjas_key')

# Load init_dict
with open('data/init_dict.pkl', 'rb') as f:
    init_dict = pickle.load(f)

# Rename to init_dict_with_transcripts for processing
init_dict_with_transcripts = init_dict.copy()

API_url = 'https://api.api-ninjas.com/v1/earningstranscript'

def get_transcript(ticker, quarter, year):
    """
    Get earnings call transcript from API.
    Returns the full API response or None if empty/error.
    """
    params = {
        'ticker': ticker,
        'quarter': quarter,
        'year': year
    }
    headers = {'X-Api-Key': API_Ninjas_key}
    
    try:
        response = requests.get(API_url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            # Check if response is empty
            if data:
                return data
    except Exception as e:
        print(f"Error fetching {ticker} Q{quarter} {year}: {e}")
    
    return None

def extract_executive_texts(transcript_dict):
    """
    Extract all text elements from company executives in an earnings call transcript.
    Excludes operators, conference operators, analysts, and entries with empty roles.
    
    Args:
        transcript_dict: Dictionary containing earnings call transcript
                        Expected structure:
                        {
                            'transcript_split': [
                                {
                                    'speaker': 'Speaker Name',
                                    'role': 'Speaker Role',
                                    'text': 'What they said...'
                                },
                                ...
                            ]
                        }
    
    Returns:
        list: List of text strings from company executives only, or None if transcript is None
    """
    
    # Handle None transcript
    if transcript_dict is None:
        return None
    
    executive_texts = []
    
    # Get transcript_split from the dictionary
    transcript_split = transcript_dict.get('transcript_split', [])
    
    # Define roles to exclude (in lowercase for comparison)
    excluded_roles = ['operator', 'analyst', 'conference operator']
    
    for element in transcript_split:
        # Get role and convert to lowercase for comparison
        role = element.get('role', '').lower()
        
        # Skip if role is empty or in excluded list
        if not role or role in excluded_roles:
            continue
        
        # Add text to executive texts
        executive_texts.append(element.get('text', ''))
    
    return executive_texts

# Initialize new columns for transcripts and call date
init_dict_with_transcripts['transcript'] = None
init_dict_with_transcripts['call_date'] = None

# Tracking variables
empty_count = 0
rows_to_drop = []
corrected_tickers = set()  # Track tickers that have been corrected
ticker_rows_processed = {}  # Track which rows have been processed for each ticker

# Iterate through each row and get transcript
total_rows = len(init_dict_with_transcripts)
print(f"Fetching transcripts for {total_rows} earnings calls...")
print("=" * 60)

# Use enumerate to get actual iteration count
for iteration_num, (idx, row) in enumerate(init_dict_with_transcripts.iterrows(), start=1):
    # Progress update every 500 calls
    if iteration_num % 500 == 0:
        print("=" * 60, flush=True)
        print(f"PROGRESS: {iteration_num}/{total_rows} completed ({iteration_num/total_rows*100:.1f}%)", flush=True)
        print("=" * 60, flush=True)
    
    # Get transcript
    transcript = get_transcript(
        ticker=row['ticker'],
        quarter=row['fiscal_quarter'],
        year=init_dict_with_transcripts.at[idx, 'fiscal_year']  # Read directly from DataFrame, not from row
    )
    
    # Store result
    if transcript is None:
        empty_count += 1
    else:
        # Extract and store the date
        init_dict_with_transcripts.at[idx, 'call_date'] = transcript.get('date')
    
    init_dict_with_transcripts.at[idx, 'transcript'] = transcript
    
    # Check if dates are within acceptable range or if fiscal year needs correction
    call_date = init_dict_with_transcripts.at[idx, 'call_date']
    announcement_date = row['announcement_date']
    
    if call_date is not None and announcement_date is not None:
        try:
            # Convert to datetime if they're strings
            if isinstance(call_date, str):
                call_date_dt = pd.to_datetime(call_date)
            else:
                call_date_dt = call_date
            
            if isinstance(announcement_date, str):
                announcement_date_dt = pd.to_datetime(announcement_date)
            else:
                announcement_date_dt = announcement_date
            
            # Calculate difference in days
            date_diff = abs((call_date_dt - announcement_date_dt).days)
            
            # Check if fiscal year is off by one year (345-385 days difference)
            if 345 <= date_diff <= 385 and row['ticker'] not in corrected_tickers:
                print(f"\n{'='*60}")
                print(f"FISCAL YEAR CORRECTION DETECTED for {row['ticker']}")
                print(f"Date difference: {date_diff} days")
                print(f"Reducing fiscal_year by 1 for all rows with ticker {row['ticker']}")
                print(f"{'='*60}\n")
                
                # Update fiscal_year for all rows with this ticker
                ticker_mask = init_dict_with_transcripts['ticker'] == row['ticker']
                init_dict_with_transcripts.loc[ticker_mask, 'fiscal_year'] = \
                    init_dict_with_transcripts.loc[ticker_mask, 'fiscal_year'] - 1
                
                # Mark this ticker as corrected
                corrected_tickers.add(row['ticker'])
                
                # Re-fetch transcripts for all previously processed rows of this ticker
                if row['ticker'] in ticker_rows_processed:
                    print(f"Re-fetching transcripts for {len(ticker_rows_processed[row['ticker']])} previously processed rows of {row['ticker']}")
                    for prev_idx in ticker_rows_processed[row['ticker']]:
                        prev_row = init_dict_with_transcripts.loc[prev_idx]
                        prev_transcript = get_transcript(
                            ticker=prev_row['ticker'],
                            quarter=prev_row['fiscal_quarter'],
                            year=prev_row['fiscal_year']  # Now uses corrected fiscal_year
                        )
                        
                        if prev_transcript is None:
                            empty_count += 1
                        else:
                            init_dict_with_transcripts.at[prev_idx, 'call_date'] = prev_transcript.get('date')
                        
                        init_dict_with_transcripts.at[prev_idx, 'transcript'] = prev_transcript
                
                # Re-fetch transcript for current row with corrected fiscal_year
                print(f"Re-fetching transcript for current row (idx {idx})")
                transcript = get_transcript(
                    ticker=row['ticker'],
                    quarter=row['fiscal_quarter'],
                    year=init_dict_with_transcripts.at[idx, 'fiscal_year']  # Uses corrected fiscal_year
                )
                
                if transcript is None:
                    empty_count += 1
                else:
                    init_dict_with_transcripts.at[idx, 'call_date'] = transcript.get('date')
                
                init_dict_with_transcripts.at[idx, 'transcript'] = transcript
                
                # Recalculate date_diff with new transcript
                call_date = init_dict_with_transcripts.at[idx, 'call_date']
                if call_date is not None:
                    if isinstance(call_date, str):
                        call_date_dt = pd.to_datetime(call_date)
                    else:
                        call_date_dt = call_date
                    date_diff = abs((call_date_dt - announcement_date_dt).days)
            
            # Mark row for dropping if difference > 5 days (after any corrections)
            if date_diff > 5:
                rows_to_drop.append(idx)
                print(f"Marking row {idx} (iteration {iteration_num}) for removal: {row['ticker']} Q{row['fiscal_quarter']} {init_dict_with_transcripts.at[idx, 'fiscal_year']} - date difference: {date_diff} days")
        
        except Exception as e:
            print(f"Error comparing dates for row {idx} (iteration {iteration_num}): {e}")
    
    # Track this row as processed for this ticker
    if row['ticker'] not in ticker_rows_processed:
        ticker_rows_processed[row['ticker']] = []
    ticker_rows_processed[row['ticker']].append(idx)

# Save the dataframe before dropping rows
with open("data/init_dict_with_transcripts_before_drop.pkl", "wb") as f:
    pickle.dump(init_dict_with_transcripts, f)

# Drop rows that don't meet the date criteria
if rows_to_drop:
    print(f"\nDropping {len(rows_to_drop)} rows due to date mismatch (>5 days difference)")
    init_dict_with_transcripts = init_dict_with_transcripts.drop(rows_to_drop)

# Drop rows with empty transcripts
rows_before_empty_drop = len(init_dict_with_transcripts)
init_dict_with_transcripts = init_dict_with_transcripts[init_dict_with_transcripts['transcript'].notna()]
rows_dropped_empty = rows_before_empty_drop - len(init_dict_with_transcripts)

if rows_dropped_empty > 0:
    print(f"Dropping {rows_dropped_empty} rows with empty transcripts")

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

# Print summary
print(f"\n{'='*60}")
print(f"PROCESSING SUMMARY")
print(f"{'='*60}")
print(f"Total earnings calls processed: {len(init_dict)}")
print(f"Tickers with fiscal year corrected: {len(corrected_tickers)}")
if corrected_tickers:
    print(f"Corrected tickers: {', '.join(sorted(corrected_tickers))}")
print(f"Rows dropped due to date mismatch: {len(rows_to_drop)}")
print(f"Rows dropped due to empty transcripts: {rows_dropped_empty}")
print(f"Remaining earnings calls: {len(init_dict_with_transcripts)}")
print(f"Empty responses encountered: {empty_count}")
print(f"Successful fetches in final dataset: {len(init_dict_with_transcripts)}")

# Extract executive texts from transcripts
print(f"\nExtracting executive texts from transcripts...")
init_dict_with_transcripts['transcript'] = init_dict_with_transcripts['transcript'].apply(extract_executive_texts)

# Count how many successful extractions we have
successful_extractions = init_dict_with_transcripts['transcript'].notna().sum()
print(f"Successfully extracted executive texts from {successful_extractions} transcripts")

# Save the final dataframe
with open("data/init_dict_with_transcripts.pkl", "wb") as f:
    pickle.dump(init_dict_with_transcripts, f)

print("\nData saved to data/init_dict_with_transcripts.pkl")