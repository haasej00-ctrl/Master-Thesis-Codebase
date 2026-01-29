import pandas as pd
import numpy as np
from datetime import timedelta
import requests
import wrds
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
API_Ninjas_key = os.getenv('API_Ninjas_key')

# Initialize WRDS API
db = wrds.Connection()

# Getting list of tickers
url = 'https://api.api-ninjas.com/v1/earningscalltranscriptslist'
headers = {
    'X-Api-Key': API_Ninjas_key
}
response = requests.get(url, headers=headers)
data = response.json() 
list_of_all_tickers = [item['ticker'] for item in data]


print("=" * 80)
print("CREATING EARNINGS_DICT V1")
print("=" * 80)

# ============================================================================
# V1: Initial query for all earnings announcements with fiscal data and NAICS
# ============================================================================

# Convert ticker list to SQL string
tickers_str = "', '".join(list_of_all_tickers)

# SQL query - joining ibes with fundq and dsenames
query = f"""
WITH all_data AS (
    SELECT 
        a.*,
        f.fyearq,      
        f.fqtr,        
        f.fyr,
        f.datacqtr,
        f.datafqtr,
        f.datadate,
        d.naics
    FROM ibes.actu_epsus a
    LEFT JOIN comp.fundq f
        ON a.oftic = f.tic
        AND a.pends = f.datadate
    INNER JOIN (
        SELECT DISTINCT ON (ticker) ticker, shrcd, naics
        FROM crsp.dsenames
        WHERE ticker IN ('{tickers_str}')
            AND shrcd IN (10, 11)
        ORDER BY ticker, nameendt DESC
    ) d
        ON a.oftic = d.ticker
    WHERE a.anndats BETWEEN '2014-05-01' AND '2024-10-31'
        AND a.pdicity = 'QTR'
        AND a.measure = 'EPS'
        AND a.value IS NOT NULL
        AND a.curr_act = 'USD'
        AND a.usfirm = 1
        AND a.oftic IN ('{tickers_str}')
        AND f.fyr IS NOT NULL
        AND d.naics IS NOT NULL
),
most_recent_cusip AS (
    SELECT DISTINCT ON (oftic) 
        oftic, 
        cusip
    FROM all_data
    ORDER BY oftic, anndats DESC
)
SELECT ad.*
FROM all_data ad
INNER JOIN most_recent_cusip mrc
    ON ad.oftic = mrc.oftic 
    AND ad.cusip = mrc.cusip
ORDER BY ad.anndats
"""

# Execute query
print("Executing initial query from ibes.actu_epsus...")
earnings_dict_v1 = db.raw_sql(query)
print(f"Total entries retrieved: {len(earnings_dict_v1)}")

# Calculate fiscal quarter and fiscal year
print("Calculating fiscal periods...")
earnings_dict_v1['period_month'] = pd.to_datetime(earnings_dict_v1['pends']).dt.month
earnings_dict_v1['period_year'] = pd.to_datetime(earnings_dict_v1['pends']).dt.year
earnings_dict_v1['months_after_fye'] = (earnings_dict_v1['period_month'] - earnings_dict_v1['fyr']) % 12
earnings_dict_v1['fiscal_qtr'] = ((earnings_dict_v1['months_after_fye'] + 11) % 12) // 3 + 1
earnings_dict_v1['fiscal_year'] = earnings_dict_v1['period_year'] + (earnings_dict_v1['period_month'] > earnings_dict_v1['fyr']).astype(int)

# Extract 2-digit and 4-digit NAICS codes
earnings_dict_v1['naics_2digit'] = earnings_dict_v1['naics'].astype(str).str[:2]
earnings_dict_v1['naics_4digit'] = earnings_dict_v1['naics'].astype(str).str[:4]

# Remove rows with missing NAICS
earnings_dict_v1 = earnings_dict_v1[
    earnings_dict_v1['naics_2digit'].notna() & 
    (earnings_dict_v1['naics_2digit'] != 'na')
].copy()
print(f"Entries after removing missing NAICS: {len(earnings_dict_v1)}")

# Create fiscal year periods (July 1 to June 30)
# If announcement is Jan-Oct, fiscal_year_period = current year
# If announcement is Nov-Dec, fiscal_year_period = next year
earnings_dict_v1['anndats_dt'] = pd.to_datetime(earnings_dict_v1['anndats'])
earnings_dict_v1['fiscal_year_period'] = earnings_dict_v1['anndats_dt'].apply(
    lambda x: x.year if x.month < 11 else x.year + 1
)

print(f"✓ V1 complete: {len(earnings_dict_v1)} entries")

print("\n" + "=" * 80)
print("CREATING EARNINGS_DICT V2")
print("=" * 80)

# ============================================================================
# V2: Filter by NAICS and year-industry requirements
# ============================================================================

# Filter out NAICS = 99 (unclassified industry)
earnings_dict_v2 = earnings_dict_v1[earnings_dict_v1['naics_2digit'] != '99'].copy()
print(f"Entries after removing NAICS=99: {len(earnings_dict_v2)}")

# Check year-industry combinations (need at least 30 entries per combination)
print("\nChecking year-industry combinations (need ≥30 entries per year-industry)...")
year_industry_counts = earnings_dict_v2.groupby(['fiscal_year_period', 'naics_2digit']).size()

# Find industries that don't meet the 30-entry requirement for ALL years (2015-2024)
# 2014 is not considered here because it only represents half a year (01.05.2014 to 31.10.2014) and is not relevant for core regression
industries_to_check = earnings_dict_v2['naics_2digit'].unique()
years_to_check = range(2015, 2025)  # Fiscal years 2015-2024

industries_to_exclude = set()
excluded_info = []

for industry in industries_to_check:
    min_count = float('inf')
    years_below_threshold = []
    
    for year in years_to_check:
        count = year_industry_counts.get((year, industry), 0)
        if count < 30:
            years_below_threshold.append((year, count))
        min_count = min(min_count, count)
    
    # Exclude industry if ANY year has < 30 entries
    if years_below_threshold:
        industries_to_exclude.add(industry)
        excluded_info.append({
            'naics_2digit': industry,
            'min_entries': min_count,
            'years_below_30': len(years_below_threshold)
        })

# Print excluded industries
if excluded_info:
    print(f"\n⚠ Excluding {len(excluded_info)} industries that don't meet 30-entry requirement:")
    for info in sorted(excluded_info, key=lambda x: x['min_entries']):
        print(f"  NAICS {info['naics_2digit']}: min entries = {info['min_entries']}, "
              f"years below 30: {info['years_below_30']}/10")
else:
    print("\n✓ All industries meet the 30-entry requirement for all years.")

# Filter out excluded industries
earnings_dict_v2 = earnings_dict_v2[
    ~earnings_dict_v2['naics_2digit'].isin(industries_to_exclude)
].copy()

print(f"\n✓ V2 complete: {len(earnings_dict_v2)} entries")
print(f"  Remaining industries: {earnings_dict_v2['naics_2digit'].nunique()}")
print(f"  Remaining tickers: {earnings_dict_v2['oftic'].nunique()}")

print("\n" + "=" * 80)
print("CREATING EARNINGS_DICT V3")
print("=" * 80)

# ============================================================================
# V3: Filter for sufficient historical data (need 8 previous earnings calls)
# ============================================================================

# Get unique tickers for querying statsum_epsus
unique_tickers = earnings_dict_v2['oftic'].unique()
tickers_str_v3 = "', '".join(unique_tickers)

# Calculate date range for comprehensive query
# Need to look back ~3 years before earliest date to find 8 historical quarters
min_date = earnings_dict_v2['anndats_dt'].min()
max_date = earnings_dict_v2['anndats_dt'].max()
query_start = (min_date - timedelta(days=1100)).strftime('%Y-%m-%d')  # ~3 years back
query_end = max_date.strftime('%Y-%m-%d')

print(f"Querying ibes.statsum_epsus for historical announcement dates...")
print(f"  Using oftic (official ticker) for consistency across databases")
print(f"Date range: {query_start} to {query_end}")
print(f"Tickers: {len(unique_tickers)}")

# Create ticker-cusip mapping from V2
ticker_cusip_map = earnings_dict_v2.groupby('oftic')['cusip'].first().to_dict()

# Modified historical query to include cusip
historical_query = f"""
SELECT DISTINCT 
    oftic,
    cusip,
    anndats_act as announcement_date
FROM ibes.statsum_epsus
WHERE
    oftic IN ('{tickers_str_v3}')
    AND anndats_act BETWEEN '{query_start}' AND '{query_end}'
    AND measure = 'EPS'
    AND fiscalp = 'QTR'
    AND actual IS NOT NULL
    AND meanest IS NOT NULL
ORDER BY oftic, anndats_act
"""

all_announcements = db.raw_sql(historical_query)
all_announcements['announcement_date'] = pd.to_datetime(all_announcements['announcement_date'])
print(f"Retrieved {len(all_announcements)} historical announcements")

# Filter to only keep announcements matching the cusip from V2
all_announcements = all_announcements[
    all_announcements.apply(
        lambda row: ticker_cusip_map.get(row['oftic']) == row['cusip'],
        axis=1
    )
]
print(f"After CUSIP filtering: {len(all_announcements)} announcements")

# Create a dictionary mapping oftic to sorted list of announcement dates
print("Building historical announcement index...")
ticker_announcements = {}
for ticker in unique_tickers:
    dates = all_announcements[all_announcements['oftic'] == ticker]['announcement_date'].sort_values().tolist()
    ticker_announcements[ticker] = dates

print(f"Indexed {len(ticker_announcements)} tickers")

# Diagnostic: Check how many V2 tickers have matching data
v2_tickers_with_data = sum(1 for t in earnings_dict_v2['oftic'].unique() 
                            if ticker_announcements.get(t, []))
print(f"V2 tickers with matching historical data: {v2_tickers_with_data}/{len(earnings_dict_v2['oftic'].unique())}")

# Check each entry in V2 for sufficient historical data
print("Checking sufficient historical data for each entry...")
valid_indices = []
no_match_count = 0
insufficient_history_count = 0

for idx, row in earnings_dict_v2.iterrows():
    ticker = row['oftic']  # Use official ticker (oftic)
    target_date = row['anndats_dt']
    
    # Get announcements for this ticker
    announcements = ticker_announcements.get(ticker, [])
    
    if len(announcements) < 9:  # Need at least 9 total (8 historical + 1 current)
        insufficient_history_count += 1
        continue
    
    # Find exact matching announcement date (both from IBES, should match exactly)
    if target_date not in announcements:
        no_match_count += 1
        continue
    
    # Count historical announcements before this date
    historical_count = sum(1 for d in announcements if d < target_date)
    
    if historical_count >= 8:  # Need 8 historical
        valid_indices.append(idx)
    else:
        insufficient_history_count += 1

earnings_dict_v3 = earnings_dict_v2.loc[valid_indices].copy()

print(f"\nFiltering results:")
print(f"  Total entries checked: {len(earnings_dict_v2)}")
print(f"  No exact matching announcement found: {no_match_count}")
print(f"  Insufficient historical data (<9 quarters): {insufficient_history_count}")
print(f"  Valid entries: {len(valid_indices)}")
print(f"  Retention rate: {len(valid_indices)/len(earnings_dict_v2)*100:.1f}%")

print(f"\n✓ V3 complete: {len(earnings_dict_v3)} entries (filtered from {len(earnings_dict_v2)})")

print("\n" + "=" * 80)
print("CREATING EARNINGS_DICT V4")
print("=" * 80)

# ============================================================================
# V4: Filter out tickers with fewer than 3 entries (need at least 3: discard first, then 2 for fixed effects)
# ============================================================================

print("Removing tickers with fewer than 3 entries...")
ticker_counts_v3 = earnings_dict_v3.groupby('oftic').size()
tickers_to_keep = ticker_counts_v3[ticker_counts_v3 >= 3].index
earnings_dict_v4 = earnings_dict_v3[earnings_dict_v3['oftic'].isin(tickers_to_keep)].copy()

print(f"\n✓ V4 complete: {len(earnings_dict_v4)} entries (filtered from {len(earnings_dict_v3)})")
print(f"  Remaining tickers: {earnings_dict_v4['oftic'].nunique()}")
print(f"  Tickers removed: {earnings_dict_v3['oftic'].nunique() - earnings_dict_v4['oftic'].nunique()}")

print("\n" + "=" * 80)
print("CREATING EARNINGS_DICT V5")
print("=" * 80)

# ============================================================================
# V5: Random sample of ~1,000 companies maintaining year-industry constraints
# ============================================================================

# Set random seed for reproducibility
np.random.seed(85)

# Create year-industry identifier
earnings_dict_v4['year_industry'] = (
    earnings_dict_v4['fiscal_year_period'].astype(str) + '_' + 
    earnings_dict_v4['naics_2digit']
)

# Get company counts per year-industry group
company_year_industry = earnings_dict_v4.groupby(['year_industry', 'oftic']).size().reset_index(name='count')
year_industry_company_counts = company_year_industry.groupby('year_industry')['oftic'].nunique()

print(f"Number of year-industry combinations: {len(year_industry_company_counts)}")
print(f"Total unique companies in V4: {earnings_dict_v4['oftic'].nunique()}")

# Strategy: Sample companies proportionally from year-industry groups to maintain 30+ calls per group
target_companies = 1000
min_calls_per_group = 30

# Calculate how many companies to sample from each year-industry group
# to maintain proportionality while ensuring minimum calls
year_industry_groups = earnings_dict_v4.groupby('year_industry')

# First, calculate average calls per company in each group
group_stats = []
for group_name, group_df in year_industry_groups:
    companies = group_df['oftic'].unique()
    n_companies = len(companies)
    n_calls = len(group_df)
    avg_calls_per_company = n_calls / n_companies if n_companies > 0 else 0
    
    group_stats.append({
        'year_industry': group_name,
        'n_companies': n_companies,
        'n_calls': n_calls,
        'avg_calls_per_company': avg_calls_per_company
    })

group_stats_df = pd.DataFrame(group_stats)

# Calculate minimum companies needed per group to reach 30 calls
group_stats_df['min_companies_needed'] = np.ceil(
    min_calls_per_group / group_stats_df['avg_calls_per_company']
).astype(int)

# Calculate proportional allocation of remaining companies
total_min_companies = group_stats_df['min_companies_needed'].sum()
remaining_companies = target_companies - total_min_companies

if remaining_companies < 0:
    print(f"⚠ Warning: Need {total_min_companies} companies minimum, but target is {target_companies}")
    print(f"  Adjusting to {total_min_companies} companies")
    target_companies = total_min_companies
    group_stats_df['companies_to_sample'] = group_stats_df['min_companies_needed']
else:
    # Distribute remaining proportionally
    group_stats_df['proportion'] = group_stats_df['n_companies'] / group_stats_df['n_companies'].sum()
    group_stats_df['additional_companies'] = np.floor(
        group_stats_df['proportion'] * remaining_companies
    ).astype(int)
    group_stats_df['companies_to_sample'] = (
        group_stats_df['min_companies_needed'] + group_stats_df['additional_companies']
    )
    
    # Handle any remaining due to rounding
    companies_allocated = group_stats_df['companies_to_sample'].sum()
    if companies_allocated < target_companies:
        shortage = target_companies - companies_allocated
        # Add to groups with most companies available
        largest_groups = group_stats_df.nlargest(shortage, 'n_companies').index
        group_stats_df.loc[largest_groups, 'companies_to_sample'] += 1

print(f"\nSampling strategy:")
print(f"  Target companies: {target_companies}")
print(f"  Total groups: {len(group_stats_df)}")
print(f"  Companies to sample per group: min={group_stats_df['companies_to_sample'].min()}, "
      f"max={group_stats_df['companies_to_sample'].max()}")

# Sample companies from each group
sampled_companies = []
for group_name, group_df in year_industry_groups:
    n_to_sample = group_stats_df[group_stats_df['year_industry'] == group_name]['companies_to_sample'].iloc[0]
    companies_in_group = group_df['oftic'].unique()
    
    # Ensure we don't try to sample more than available
    n_to_sample = min(n_to_sample, len(companies_in_group))
    
    selected = np.random.choice(companies_in_group, size=n_to_sample, replace=False)
    sampled_companies.extend(selected)

# Remove duplicates (a company might appear in multiple year-industry groups)
sampled_companies = list(set(sampled_companies))
print(f"Unique companies sampled: {len(sampled_companies)}")

# Get all rows for sampled companies
earnings_dict_v5_temp = earnings_dict_v4[earnings_dict_v4['oftic'].isin(sampled_companies)].copy()
print(f"Total earnings calls from sampled companies: {len(earnings_dict_v5_temp)}")

# Verify constraints
print("\nVerifying company-level constraint...")
company_counts = earnings_dict_v5_temp.groupby('oftic').size()
companies_below_threshold = (company_counts < 3).sum()
print(f"  Companies with fewer than 3 calls: {companies_below_threshold}")
print(f"  All companies have ≥3 calls: {companies_below_threshold == 0}")

# Create final output dataframe with required columns only
earnings_dict_v5 = earnings_dict_v5_temp[[
    'oftic',
    'anndats',
    'fiscal_qtr',
    'fiscal_year',
    'fiscal_year_period',
    'naics_2digit',
    'naics_4digit'
]].copy()

# Rename columns for clarity
earnings_dict_v5.columns = [
    'ticker',
    'announcement_date',
    'fiscal_quarter',
    'fiscal_year',
    'period_year',
    'naics_2digit',
    'naics_4digit'
]

# Sort by announcement date
earnings_dict_v5 = earnings_dict_v5.sort_values('announcement_date').reset_index(drop=True)

# Verify final constraints
print("\nVerifying final constraints...")
final_year_industry_counts = earnings_dict_v5.groupby(
    ['period_year', 'naics_2digit']
).size()

min_group_size = final_year_industry_counts.min()
max_group_size = final_year_industry_counts.max()

print(f"\n✓ V5 complete: {len(earnings_dict_v5)} entries")
print(f"\nFinal statistics:")
print(f"  Unique tickers: {earnings_dict_v5['ticker'].nunique()}")
print(f"  Unique industries (2-digit NAICS): {earnings_dict_v5['naics_2digit'].nunique()}")
print(f"  Unique industries (4-digit NAICS): {earnings_dict_v5['naics_4digit'].nunique()}")
print(f"  Unique year-industry combinations: {len(final_year_industry_counts)}")
print(f"  Entries per year-industry: min={min_group_size}, max={max_group_size}")
print(f"  Date range: {earnings_dict_v5['announcement_date'].min()} to {earnings_dict_v5['announcement_date'].max()}")

# Verify no group has less than 30 entries
if min_group_size < 30:
    print(f"\n⚠ WARNING: Minimum group size ({min_group_size}) is below 30!")
    small_groups = final_year_industry_counts[final_year_industry_counts < 30]
    print(f"Groups with <30 entries: {len(small_groups)}")
    for (year, naics), count in small_groups.items():
        print(f"  Year {year}, NAICS {naics}: {count} entries")
else:
    print(f"\n✓ All year-industry combinations have ≥30 entries")

print("\n" + "=" * 80)
print("CREATING EARNINGS_DICT V6")
print("=" * 80)

# ============================================================================
# V6: Exclude companies with less than 30 trading days between ANY earnings calls
# ============================================================================

print("Filtering companies to ensure minimum 30 trading days between all earnings calls...")
print("(This prevents confounding events from affecting CAR calculations)")

# Sort by ticker and announcement date
earnings_dict_v5_sorted = earnings_dict_v5.sort_values(['ticker', 'announcement_date']).reset_index(drop=True)

# Calculate trading days between consecutive earnings calls for each ticker
valid_tickers = []
excluded_tickers = []

for ticker in earnings_dict_v5_sorted['ticker'].unique():
    ticker_data = earnings_dict_v5_sorted[earnings_dict_v5_sorted['ticker'] == ticker].copy()
    
    # Get announcement dates
    dates = pd.to_datetime(ticker_data['announcement_date']).sort_values()
    
    # Check gaps between consecutive announcements
    min_trading_days = float('inf')
    gaps_below_threshold = []
    
    for i in range(len(dates) - 1):
        # Calculate trading days between consecutive announcements
        # np.busday_count excludes weekends (but not US market holidays)
        trading_days = np.busday_count(
            dates.iloc[i].date(),
            dates.iloc[i+1].date()
        )
        
        if trading_days < 30:
            gaps_below_threshold.append(trading_days)
        
        min_trading_days = min(min_trading_days, trading_days)
    
    # Keep ticker only if ALL gaps are >= 30 trading days
    if min_trading_days >= 30:
        valid_tickers.append(ticker)
    else:
        excluded_tickers.append({
            'ticker': ticker,
            'min_trading_days': min_trading_days,
            'n_calls': len(dates),
            'n_gaps_below_30': len(gaps_below_threshold)
        })

print(f"\nFiltering results:")
print(f"  Total tickers in V5: {earnings_dict_v5_sorted['ticker'].nunique()}")
print(f"  Tickers with all gaps ≥30 trading days: {len(valid_tickers)}")
print(f"  Tickers excluded: {len(excluded_tickers)}")
print(f"  Retention rate: {len(valid_tickers)/earnings_dict_v5_sorted['ticker'].nunique()*100:.1f}%")

if excluded_tickers:
    excluded_df = pd.DataFrame(excluded_tickers)
    print(f"\nExcluded ticker statistics:")
    print(f"  Average minimum gap: {excluded_df['min_trading_days'].mean():.1f} trading days")
    print(f"  Smallest gap found: {excluded_df['min_trading_days'].min()} trading days")
    print(f"  Average calls per excluded ticker: {excluded_df['n_calls'].mean():.1f}")

# Create V6 with only valid tickers
earnings_dict_v6 = earnings_dict_v5_sorted[
    earnings_dict_v5_sorted['ticker'].isin(valid_tickers)
].reset_index(drop=True)

# Verify final constraints from previous versions still hold
print("\nVerifying year-industry constraints after V6 filtering...")
final_year_industry_counts = earnings_dict_v6.groupby(
    ['period_year', 'naics_2digit']
).size()

min_group_size = final_year_industry_counts.min() if len(final_year_industry_counts) > 0 else 0
max_group_size = final_year_industry_counts.max() if len(final_year_industry_counts) > 0 else 0

print(f"\n✓ V6 complete: {len(earnings_dict_v6)} entries (filtered from {len(earnings_dict_v5)})")
print(f"\nFinal statistics:")
print(f"  Unique tickers: {earnings_dict_v6['ticker'].nunique()}")
print(f"  Unique industries (2-digit NAICS): {earnings_dict_v6['naics_2digit'].nunique()}")
print(f"  Unique industries (4-digit NAICS): {earnings_dict_v6['naics_4digit'].nunique()}")
print(f"  Unique year-industry combinations: {len(final_year_industry_counts)}")
print(f"  Entries per year-industry: min={min_group_size}, max={max_group_size}")
print(f"  Date range: {earnings_dict_v6['announcement_date'].min()} to {earnings_dict_v6['announcement_date'].max()}")

# Check if any year-industry groups fell below 30 entries
if min_group_size < 30 and len(final_year_industry_counts) > 0:
    print(f"\n⚠ WARNING: After V6 filtering, {(final_year_industry_counts < 30).sum()} year-industry groups now have <30 entries")
    small_groups = final_year_industry_counts[final_year_industry_counts < 30]
    print(f"\nGroups with <30 entries:")
    for (year, naics), count in small_groups.items():
        print(f"  Year {year}, NAICS {naics}: {count} entries")
else:
    print(f"\n✓ All year-industry combinations still have ≥30 entries")

# Verify minimum 30 trading days between all calls
print("\nVerifying 30-day trading gap constraint...")
gap_violations = 0
for ticker in earnings_dict_v6['ticker'].unique():
    ticker_data = earnings_dict_v6[earnings_dict_v6['ticker'] == ticker].copy()
    dates = pd.to_datetime(ticker_data['announcement_date']).sort_values()
    
    for i in range(len(dates) - 1):
        trading_days = np.busday_count(dates.iloc[i].date(), dates.iloc[i+1].date())
        if trading_days < 30:
            gap_violations += 1
            print(f"  ⚠ {ticker}: gap of {trading_days} days between {dates.iloc[i].date()} and {dates.iloc[i+1].date()}")

if gap_violations == 0:
    print("✓ All consecutive earnings calls have ≥30 trading days separation")
else:
    print(f"⚠ Found {gap_violations} gap violations - this should not happen!")

print("\n" + "=" * 80)
print("EARNINGS_DICT CREATION COMPLETE")
print("=" * 80)

# Display sample of final dataframe
print("\nSample of final earnings_dict_v6:")
print(earnings_dict_v6.head(10))
print("\nDataframe info:")
print(earnings_dict_v6.info())

# Save to JSON
earnings_dict_v6.to_json('data/earnings_dict.json', orient='records', date_format='iso')