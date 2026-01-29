import init_functions
import pandas as pd
import wrds
import pickle
import json

# Initialize WRDS API
db = wrds.Connection()

# Load earnings_dict
earnings_dict = pd.read_json('data/earnings_dict.json', orient='records')
earnings_dict['announcement_date'] = pd.to_datetime(earnings_dict['announcement_date'])
print('Loaded earnings_dict.')

# Create list of all tickers
list_of_all_tickers = earnings_dict['ticker'].unique().tolist()
print('Created list of tickers.')
print('Creating permno_dict...')

# Create and save permno_dict
permno_dict = init_functions.get_permno(list_of_all_tickers, db)

with open("data/permno_dict.json", "w") as f:
    json.dump(permno_dict, f, indent=2)

print('Created permno_dict.')
print('Creating trading_days...')

# Create and save trading_days
study_period = ('2014-05-01', '2024-10-31')
trading_days = init_functions.get_trading_days(study_period, db)

with open("data/trading_days.json", "w") as f:
    json.dump(trading_days, f, indent=2)

print('Created trading_days.')
print('Getting earnings_data_dict and analyst_coverage...')

# Get earnings_data_dict and analyst_coverage
earnings_data_dict, analyst_coverage = init_functions.get_UE_and_EPS_change(earnings_dict, db)

with open("data/earnings_data_dict.pkl", "wb") as f:
    pickle.dump(earnings_data_dict, f)
    
with open("data/analyst_coverage.pkl", "wb") as f:
    pickle.dump(analyst_coverage, f)
    
print('Saved earnings_data_dict and analyst_coverage.')
print('Getting fund_data_dict (incl. discretionary accruals)...')

# Get fund_data_dict
compustat_fund = init_functions.get_compustat_fund(earnings_dict, db)
print('compustat_fund created.')
modified_jones_results = init_functions.calc_modified_jones(compustat_fund)
print('modified_jones_results created.')
fund_data_dict = init_functions.calc_discretionary_accruals(compustat_fund, modified_jones_results)
print('fund_data_dict created.')

with open("data/fund_data_dict.pkl", "wb") as f:
    pickle.dump(fund_data_dict, f)

print('Saved fund_data_dict (incl. discretionary accruals).')

# Create init_dict
print('Creating init_dict...')
init_dict = earnings_dict.copy()

# Add analyst_coverage column
print('Adding analyst_coverage column...')
analyst_coverage_values = []
for idx, row in init_dict.iterrows():
    ticker = row['ticker']
    announcement_date = row['announcement_date']
    
    # Check if ticker exists in analyst_coverage
    if ticker in analyst_coverage:
        # Check if announcement_date exists for this ticker
        if announcement_date in analyst_coverage[ticker]:
            analyst_coverage_values.append(analyst_coverage[ticker][announcement_date])
        else:
            analyst_coverage_values.append(None)
    else:
        analyst_coverage_values.append(None)

init_dict['analyst_coverage'] = analyst_coverage_values
print('Added analyst_coverage column.')

# Add discretionary_accruals_scaled column
print('Adding discretionary_accruals_scaled column...')
discretionary_accruals_values = []
for idx, row in init_dict.iterrows():
    ticker = row['ticker']
    announcement_date = row['announcement_date']
    
    # Check if ticker exists in fund_data_dict
    if ticker in fund_data_dict:
        # Check if announcement_date exists for this ticker
        if announcement_date in fund_data_dict[ticker]:
            discretionary_accruals_values.append(fund_data_dict[ticker][announcement_date].get('discretionary_accruals_scaled'))
        else:
            discretionary_accruals_values.append(None)
    else:
        discretionary_accruals_values.append(None)

init_dict['discretionary_accruals_scaled'] = discretionary_accruals_values
print('Added discretionary_accruals_scaled column.')

# Add leverage column
print('Adding leverage column...')
leverage_values = []
for idx, row in init_dict.iterrows():
    ticker = row['ticker']
    announcement_date = row['announcement_date']
    
    # Check if ticker exists in fund_data_dict
    if ticker in fund_data_dict:
        # Check if announcement_date exists for this ticker
        if announcement_date in fund_data_dict[ticker]:
            leverage_values.append(fund_data_dict[ticker][announcement_date].get('leverage'))
        else:
            leverage_values.append(None)
    else:
        leverage_values.append(None)

init_dict['leverage'] = leverage_values
print('Added leverage column.')

# Add book_to_market column
print('Adding book_to_market column...')
book_to_market_values = []
for idx, row in init_dict.iterrows():
    ticker = row['ticker']
    announcement_date = row['announcement_date']
    
    # Check if ticker exists in fund_data_dict
    if ticker in fund_data_dict:
        # Check if announcement_date exists for this ticker
        if announcement_date in fund_data_dict[ticker]:
            book_to_market_values.append(fund_data_dict[ticker][announcement_date].get('book_to_market'))
        else:
            book_to_market_values.append(None)
    else:
        book_to_market_values.append(None)

init_dict['book_to_market'] = book_to_market_values
print('Added book_to_market column.')

# Add SUE column
print('Adding SUE column...')
init_dict['SUE'] = init_dict.apply(lambda row: init_functions.calc_SUE(row['ticker'], row['announcement_date'], earnings_data_dict), axis=1)
print('Added SUE column.')

# Add stand_eps_change column
print('Adding stand_eps_change column...')
init_dict['stand_eps_change'] = init_dict.apply(lambda row: init_functions.calc_standardized_eps_change(row['ticker'], row['announcement_date'], earnings_data_dict), axis=1)
print('Added stand_eps_change column.')

# Add market_cap
print('Adding market_cap column...')
init_dict['market_cap'] = init_functions.get_market_cap_bulk(init_dict, permno_dict, trading_days, db)
print('Added market_cap column.')

# Remove rows with any null values
print('\nRemoving rows with null values...')
rows_before = len(init_dict)
init_dict = init_dict.dropna()
rows_after = len(init_dict)
print(f'Removed {rows_before - rows_after} rows with null values.')
print(f'Remaining rows: {rows_after}')

# Check and drop tickers with fewer than 3 rows
print(f"\nChecking for tickers with insufficient observations...")
ticker_counts = init_dict['ticker'].value_counts()
tickers_to_drop = ticker_counts[ticker_counts < 3].index.tolist()

if tickers_to_drop:
    rows_before_ticker_drop = len(init_dict)
    init_dict = init_dict[~init_dict['ticker'].isin(tickers_to_drop)]
    rows_dropped_ticker = rows_before_ticker_drop - len(init_dict)
    print(f"Dropping {len(tickers_to_drop)} tickers with <3 observations (removed {rows_dropped_ticker} rows)")
    print(f"Examples of dropped tickers: {', '.join(sorted(tickers_to_drop)[:10])}")
    if len(tickers_to_drop) > 10:
        print(f"... and {len(tickers_to_drop) - 10} more tickers")
else:
    print(f"All tickers have at least 3 observations")

print(f'Final rows after ticker filter: {len(init_dict)}')

with open("data/init_dict.pkl", "wb") as f:
    pickle.dump(init_dict, f)

print('Saved init_dict.')

# Print summary statistics
print('\n' + '='*80)
print('INIT_DICT SUMMARY STATISTICS')
print('='*80)
print(f'\nDataset shape: {init_dict.shape[0]} rows × {init_dict.shape[1]} columns')
print(f'\nColumns: {list(init_dict.columns)}')
print(f'\nDate range: {init_dict["announcement_date"].min()} to {init_dict["announcement_date"].max()}')
print(f'Number of unique tickers: {init_dict["ticker"].nunique()}')
print(f'\nDescriptive statistics for numeric columns:')
print(init_dict.describe())

print('\n' + '='*80)
print('SAMPLE DATA (first 10 rows)')
print('='*80)
print(init_dict.head(10))

print('\n' + '='*80)
print('SAMPLE DATA (last 10 rows)')
print('='*80)
print(init_dict.tail(10))