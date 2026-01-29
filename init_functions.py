from typing import Optional, List, Dict, Tuple
import pandas as pd
from datetime import date, datetime, timedelta
import statsmodels.api as sm
import numpy as np

from typing import List, Dict

def get_permno(tickers: List[str], db) -> Dict[str, int]:
    """
    Get PERMNO codes for a list of tickers from CRSP.
    
    Args:
        tickers: List of ticker symbols.
        db: Database connection object with raw_sql method.
    
    Returns:
        Dictionary with ticker as key and PERMNO as value.
    """
    permno_dict = {}
    
    # Among entries with the most recent nameendt for the ticker,
    # choose the PERMNO whose latest (by date) shrout in crsp.dsf is highest.
    permno_query = """
    WITH candidates AS (
        SELECT permno, ticker, nameendt
        FROM crsp.dsenames
        WHERE ticker = %s
          AND shrcd IN (10, 11)
    ),
    max_nameendt AS (
        SELECT MAX(nameendt) AS nameendt FROM candidates
    ),
    filtered AS (
        -- only rows at the most recent nameendt (handles multiple share classes)
        SELECT c.permno, c.ticker
        FROM candidates c
        JOIN max_nameendt m ON c.nameendt = m.nameendt
    ),
    latest_shares AS (
        -- latest shrout per permno from daily file
        SELECT permno, shrout
        FROM (
            SELECT
                d.permno,
                d.shrout,
                d.date,
                ROW_NUMBER() OVER (PARTITION BY d.permno ORDER BY d.date DESC) AS rn
            FROM crsp.dsf d
            WHERE d.permno IN (SELECT permno FROM filtered)
        ) s
        WHERE rn = 1
    )
    SELECT f.permno, f.ticker
    FROM filtered f
    LEFT JOIN latest_shares ls USING (permno)
    ORDER BY ls.shrout DESC NULLS LAST
    LIMIT 1
    """
    
    for ticker in tickers:
        result = db.raw_sql(permno_query, params=(ticker,))
        
        if len(result) > 0:
            permno_dict[ticker] = int(result['permno'].iloc[0])
        else:
            print(f"Warning: No PERMNO found for {ticker}")
            permno_dict[ticker] = None
    
    return permno_dict

def get_trading_days(
    study_period: Tuple[str, str],
    db,
    estimation_window: Tuple[int, int] = (-250, -15),
    CAR_time_windows: List[int] = [1, 3, 30, 60, 90]
) -> List[str]:
    """
    Get all trading days in the study period plus buffer periods for estimation and CAR windows.

    Args:
        study_period: Tuple of (start_date, end_date) as strings in ISO format.
                      Format: ('YYYY-MM-DD', 'YYYY-MM-DD')
                      Example: ('2018-01-01', '2020-12-31')
        db: Database connection object with raw_sql method.
        estimation_window: Tuple of (start, end) trading days relative to event date.
                          Used to calculate buffer before study period start.
                          Default: (-250, -15)
        CAR_time_windows: List of post-event window lengths in trading days.
                         Used to calculate buffer after study period end.
                         Default: [1, 3, 30, 60, 90]
    
    Returns:
        List of trading day strings in ISO format (YYYY-MM-DD), sorted chronologically.
    """
    # Convert string dates to date objects
    start_date = datetime.strptime(study_period[0], '%Y-%m-%d').date()
    end_date = datetime.strptime(study_period[1], '%Y-%m-%d').date()
    
    # Calculate buffer periods with 2x multiplier for calendar days
    days_before = abs(estimation_window[0]) * 2  # Take absolute value of negative start
    days_after = max(CAR_time_windows) * 2
    
    # Calculate query date range with buffers
    query_start = start_date - timedelta(days=days_before)
    query_end = end_date + timedelta(days=days_after)
    
    # Query to get all available trading dates from Fama-French data
    trading_dates_query = """
    SELECT DISTINCT date
    FROM ff.fivefactors_daily
    WHERE date >= %s AND date <= %s
    ORDER BY date
    """
    
    trading_dates_df = db.raw_sql(trading_dates_query, params=(query_start, query_end))
    
    # Ensure date column is datetime type, then convert to string list
    trading_dates_df['date'] = pd.to_datetime(trading_dates_df['date'])
    trading_days = trading_dates_df['date'].dt.strftime('%Y-%m-%d').tolist()
    
    return trading_days

def get_UE_and_EPS_change(
    earnings_dict: pd.DataFrame, db
) -> Tuple[Dict[str, Dict[datetime, Dict[str, float]]], Dict[str, Dict[datetime, int]]]:
    """
    Calculate unexpected earnings for companies across multiple earnings announcement dates.
    
    Args:
        earnings_dict: DataFrame with columns: ticker (oftic), announcement_date, fiscal_quarter, 
                      fiscal_year, period_year, naics_2digit
    
    Returns:
        Tuple containing:
        1. earnings_data_output: Dictionary with ticker as key and dict as value. Inner dict has:
           - key: earnings call date (from incoming earnings_dict)
           - value: dict with "unexpected_earnings" and "eps_change" keys
        2. analyst_coverage: Dictionary with ticker as key and dict as value. Inner dict has:
           - key: earnings call date (from incoming earnings_dict)
           - value: analyst coverage (int)
    """
    
    if earnings_dict.empty:
        return {}, {}
    
    # Convert announcement_date to datetime
    earnings_dict = earnings_dict.copy()
    earnings_dict['announcement_date'] = pd.to_datetime(earnings_dict['announcement_date'])
    
    # Get all unique tickers (oftic) and date range
    all_tickers = earnings_dict['ticker'].unique().tolist()
    earliest_date = earnings_dict['announcement_date'].min()
    latest_date = earnings_dict['announcement_date'].max()
    data_start = earliest_date - timedelta(days=1100)  # ~3 years before
    data_end = latest_date + timedelta(days=5)
    
    # Single query for ALL announcement dates across ALL tickers
    all_announcements_query = """
        SELECT DISTINCT 
            oftic,
            anndats_act as announcement_date
        FROM ibes.statsum_epsus
        WHERE
            oftic IN %s
            AND anndats_act >= %s
            AND anndats_act <= %s
            AND measure = 'EPS'
            AND fiscalp = 'QTR'
            AND actual IS NOT NULL
            AND meanest IS NOT NULL
        ORDER BY oftic, anndats_act DESC
    """
    
    announcements_result = db.raw_sql(
        all_announcements_query,
        params=(tuple(all_tickers), data_start.strftime('%Y-%m-%d'), data_end.strftime('%Y-%m-%d'))
    )
    
    if announcements_result.empty:
        print("No earnings announcements found for any ticker")
        return {}, {}
    
    announcements_result['announcement_date'] = pd.to_datetime(announcements_result['announcement_date'])
    
    # Single query for ALL earnings data across ALL tickers
    earnings_query = """
        SELECT
            oftic,
            statpers as statistics_date,
            fpedats as fiscal_period_end,
            anndats_act as announcement_date,
            meanest as consensus_forecast_eps,
            numest as num_estimates,
            actual as actual_eps
        FROM ibes.statsum_epsus
        WHERE
            oftic IN %s
            AND anndats_act <= %s
            AND anndats_act >= %s
            AND measure = 'EPS'
            AND fiscalp = 'QTR'
            AND actual IS NOT NULL
            AND meanest IS NOT NULL
        ORDER BY oftic, anndats_act DESC, statpers DESC
    """
    
    results = db.raw_sql(
        earnings_query,
        params=(tuple(all_tickers), data_end.strftime('%Y-%m-%d'), data_start.strftime('%Y-%m-%d'))
    )
    
    if results.empty:
        print("No earnings data found for any ticker")
        return {}, {}
    
    # Convert date columns
    results['announcement_date'] = pd.to_datetime(results['announcement_date'])
    results['statistics_date'] = pd.to_datetime(results['statistics_date'])
    results['fiscal_period_end'] = pd.to_datetime(results['fiscal_period_end'])
    
    # Deduplicate: keep earliest announcement per fiscal period, latest statpers
    results = results.sort_values(
        ['oftic', 'fiscal_period_end', 'announcement_date', 'statistics_date'],
        ascending=[True, False, True, False]
    )
    results = results.groupby(['oftic', 'fiscal_period_end'], as_index=False).first()
    
    # Output dictionaries
    earnings_data_output = {}
    analyst_coverage = {}
    
    # Process each ticker
    for ticker in all_tickers:
        # Get input dates for this ticker (exact dates only)
        ticker_input = earnings_dict[earnings_dict['ticker'] == ticker]
        input_dates = set(ticker_input['announcement_date'])
        
        if not input_dates:
            continue
        
        # Get all announcements for this ticker
        ticker_announcements = announcements_result[announcements_result['oftic'] == ticker]['announcement_date'].tolist()
        
        if not ticker_announcements:
            print(f"No earnings announcements found for {ticker}")
            continue
        
        # Check which input dates have exact matches in announcements
        ticker_announcements_set = set(ticker_announcements)
        valid_dates = sorted([d for d in input_dates if d in ticker_announcements_set])
        
        if not valid_dates:
            print(f"No exact matching earnings announcements found for {ticker}")
            continue
        
        # Check which valid dates have sufficient historical data (8 quarters)
        earliest_valid_date = None
        
        for d in valid_dates:
            lookback_start = d - timedelta(days=1100)
            num_historical = sum(1 for ann_date in ticker_announcements if lookback_start <= ann_date < d)
            
            if num_historical >= 8:
                earliest_valid_date = d
                break
        
        if not earliest_valid_date:
            print(f"No dates with sufficient historical data for {ticker}")
            continue
        
        valid_dates = [d for d in valid_dates if d >= earliest_valid_date]
        
        if not valid_dates:
            continue
        
        # Get earnings data for this ticker
        ticker_results = results[results['oftic'] == ticker].copy()
        ticker_results = ticker_results.sort_values('announcement_date', ascending=False)
        
        # Filter for relevant announcements: input dates + 8 historical quarters before earliest
        earliest_date_ticker = min(valid_dates)
        latest_date_ticker = max(valid_dates)
        
        # Get announcements that match our input dates
        in_range_announcements = ticker_results[ticker_results['announcement_date'].isin(valid_dates)]
        
        # Get the 8 quarters before the earliest valid date
        historical_announcements = ticker_results[
            ticker_results['announcement_date'] < earliest_date_ticker
        ].head(8)
        
        all_relevant_announcements = pd.concat([in_range_announcements, historical_announcements])
        all_relevant_announcements = all_relevant_announcements.drop_duplicates(subset=['announcement_date'])
        all_relevant_announcements = all_relevant_announcements.sort_values('announcement_date', ascending=True)
        
        # Calculate unexpected earnings and eps_change
        earnings_data_dict = {}
        analyst_coverage_dict = {}
        prev_actual_eps = None
        valid_dates_set = set(valid_dates)
        
        for _, row in all_relevant_announcements.iterrows():
            announcement_date = row['announcement_date']
            current_actual_eps = float(row['actual_eps'])
            unexpected = current_actual_eps - float(row['consensus_forecast_eps'])
            num_forecasts = int(row['num_estimates'])
            
            # Use announcement_date as key (converted to datetime if needed)
            key_date = announcement_date.to_pydatetime() if hasattr(announcement_date, 'to_pydatetime') else announcement_date
            
            eps_change = current_actual_eps - prev_actual_eps if prev_actual_eps is not None else None
            
            earnings_data_dict[key_date] = {
                "unexpected_earnings": unexpected,
                "eps_change": eps_change
            }
            
            # Only add to analyst_coverage if this date is in the input
            if announcement_date in valid_dates_set:
                analyst_coverage_dict[key_date] = num_forecasts
            
            prev_actual_eps = current_actual_eps
        
        earnings_data_output[ticker] = earnings_data_dict
        analyst_coverage[ticker] = analyst_coverage_dict
    
    return earnings_data_output, analyst_coverage

def get_compustat_fund(earnings_dict, db):
    """
    Retrieve Compustat quarterly fundamentals for earnings call dates.
    
    Args:
        earnings_dict: pandas DataFrame with columns:
                      - ticker: company ticker
                      - announcement_date: earnings call date
                      - naics_2digit: 2-digit NAICS industry code
                      - period_year: year classification for grouping
                      - fiscal_quarter, fiscal_year (optional)
        db: Active WRDS database connection
    
    Returns:
        dict[ticker][date][data_point] = value (or None if data is NA)
    """
    # Convert announcement_date to datetime if not already
    earnings_dict = earnings_dict.copy()
    earnings_dict['announcement_date'] = pd.to_datetime(earnings_dict['announcement_date'])
    
    # Prepare date range and ticker list
    all_dates = earnings_dict['announcement_date'].tolist()
    tickers = earnings_dict['ticker'].unique().tolist()
    min_date = (min(all_dates) - timedelta(days=120)).strftime('%Y-%m-%d')
    max_date = (max(all_dates) + timedelta(days=60)).strftime('%Y-%m-%d')
    
    # Query Compustat - no longer need to join with company table for NAICS
    query = f"""
    SELECT f.tic, f.datadate, f.ceqq, f.prccq, f.cshoq, f.dlcq, f.dlttq, f.atq,
           f.actq, f.lctq, f.cheq, f.dpq, f.revtq, f.rectq, f.ppegtq
    FROM comp.fundq f
    WHERE f.tic IN {tuple(tickers) if len(tickers) > 1 else f"('{tickers[0]}')"}
    AND f.datadate BETWEEN '{min_date}' AND '{max_date}'
    ORDER BY f.tic, f.datadate
    """
    
    df = db.raw_sql(query)
    
    # Convert datadate to datetime
    df['datadate'] = pd.to_datetime(df['datadate'])
    
    # Initialize output dictionary
    comp_fund_output = {ticker: {} for ticker in tickers}
    
    # Process each row in earnings_dict DataFrame
    for _, row in earnings_dict.iterrows():
        ticker = row['ticker']
        call_date = row['announcement_date']
        naics_2digit = row.get('naics_2digit')  # Get NAICS from earnings_dict
        period_year = row.get('period_year')  # Get period_year from earnings_dict
        
        ticker_data = df[df['tic'] == ticker].sort_values('datadate').reset_index(drop=True)
        
        if ticker_data.empty:
            continue
        
        # Find most recent quarter on or before call date
        mask = ticker_data['datadate'] <= call_date
        if not mask.any():
            continue
        
        idx = mask[::-1].idxmax()  # Index of most recent quarter
        current = ticker_data.loc[idx]
        previous = ticker_data.loc[idx - 1] if idx > 0 else None
        
        data = {}
        
        # Book-to-market ratio
        if pd.notna(current['ceqq']) and pd.notna(current['prccq']) and pd.notna(current['cshoq']):
            market_val = current['prccq'] * current['cshoq']
            if market_val > 0:
                data['book_to_market'] = current['ceqq'] / market_val
            else:
                data['book_to_market'] = None
        else:
            data['book_to_market'] = None
        
        # Industry - use 2-digit NAICS from earnings_dict
        if pd.notna(naics_2digit):
            data['2naics'] = str(naics_2digit)
        else:
            data['2naics'] = None
        
        # Period year - use from earnings_dict for year-industry grouping
        if pd.notna(period_year):
            data['period_year'] = int(period_year)
        else:
            data['period_year'] = None
        
        # Leverage - only calculate if both debt components and assets are available
        if pd.notna(current['atq']) and current['atq'] > 0 and \
           pd.notna(current['dlcq']) and pd.notna(current['dlttq']):
            total_debt = current['dlcq'] + current['dlttq']
            data['leverage'] = total_debt / current['atq']
        else:
            data['leverage'] = None
        
        # Total assets from previous quarter
        if previous is not None and pd.notna(previous['atq']):
            data['total_assets_previous_quarter'] = previous['atq']
        else:
            data['total_assets_previous_quarter'] = None
        
        # Modified Jones variables - calculate changes if previous quarter exists
        if previous is not None:
            # Calculate total accruals
            # TA = (ΔCA - ΔCash) - (ΔCL - ΔSTD) - Depreciation
            if all(pd.notna(current[v]) and pd.notna(previous[v]) for v in ['actq', 'cheq', 'lctq', 'dlcq']) and pd.notna(current['dpq']):
                delta_ca = current['actq'] - previous['actq']
                delta_cash = current['cheq'] - previous['cheq']
                delta_cl = current['lctq'] - previous['lctq']
                delta_std = current['dlcq'] - previous['dlcq']
                data['total_accruals'] = (delta_ca - delta_cash) - (delta_cl - delta_std) - current['dpq']
            else:
                data['total_accruals'] = None
            
            # Calculate deltas for individual variables
            for var, name in [
                ('actq', 'delta_current_assets'),
                ('lctq', 'delta_current_liabilities'),
                ('cheq', 'delta_cash'),
                ('dlcq', 'delta_short_term_debt'),
                ('revtq', 'delta_revenue'),
                ('rectq', 'delta_receivables')
            ]:
                if pd.notna(current[var]) and pd.notna(previous[var]):
                    data[name] = current[var] - previous[var]
                else:
                    data[name] = None
        else:
            # No previous quarter available
            data['total_accruals'] = None
            for name in ['delta_current_assets', 'delta_current_liabilities', 'delta_cash',
                        'delta_short_term_debt', 'delta_revenue', 'delta_receivables']:
                data[name] = None
        
        # Current period values
        data['depreciation'] = current['dpq'] if pd.notna(current['dpq']) else None
        data['gross_ppe'] = current['ppegtq'] if pd.notna(current['ppegtq']) else None
        
        # Modified Jones regression variables (scaled by Assets(t-1))
        if previous is not None and pd.notna(previous['atq']) and previous['atq'] > 0:
            assets_t_minus_1 = previous['atq']
            
            # Total accruals / Assets(t-1)
            if data['total_accruals'] is not None:
                data['total_accruals_scaled'] = data['total_accruals'] / assets_t_minus_1
            else:
                data['total_accruals_scaled'] = None
            
            # 1 / Assets(t-1)
            data['inverse_assets'] = 1 / assets_t_minus_1
            
            # (ΔRevenue - ΔReceivables) / Assets(t-1)
            if data.get('delta_revenue') is not None and data.get('delta_receivables') is not None:
                data['delta_rev_minus_rec_scaled'] = (data['delta_revenue'] - data['delta_receivables']) / assets_t_minus_1
            else:
                data['delta_rev_minus_rec_scaled'] = None
            
            # PPE / Assets(t-1)
            if pd.notna(current['ppegtq']):
                data['ppe_scaled'] = current['ppegtq'] / assets_t_minus_1
            else:
                data['ppe_scaled'] = None
        else:
            # No valid previous quarter assets
            data['total_accruals_scaled'] = None
            data['inverse_assets'] = None
            data['delta_rev_minus_rec_scaled'] = None
            data['ppe_scaled'] = None
        
        comp_fund_output[ticker][call_date] = data
    
    return comp_fund_output

def calc_modified_jones(fund_data_dict):
    """
    Conducts Modified Jones regression for calculating discretionary accruals.
    
    The Modified Jones model estimates normal (non-discretionary) accruals using:
    TA_t / A_{t-1} = β0*(1/A_{t-1}) + β1*((ΔRev - ΔRec)/A_{t-1}) + β2*(PPE/A_{t-1}) + ε
    
    Industries with fewer than 30 observations in ANY year are blacklisted and excluded
    from regressions across ALL years to maintain a balanced panel.
    
    Args:
        fund_data_dict: dict with structure {ticker: {date: {data_point: value}}}
             Required data points: total_accruals_scaled, inverse_assets,
             delta_rev_minus_rec_scaled, ppe_scaled, 2naics, period_year
    
    Returns:
        dict with structure {period_year_naics: {coefficient_name: coefficient_value}}
        Example: {'2024_33': {'beta_0': 0.05, 'beta_1': 0.10, 'beta_2': -0.02}}
    """
    
    # Convert nested dictionary to long-form DataFrame
    rows = []
    for ticker, dates in fund_data_dict.items():
        for date, data in dates.items():
            # Skip observations with missing required variables
            required_vars = ['total_accruals_scaled', 'inverse_assets', 
                           'delta_rev_minus_rec_scaled', 'ppe_scaled', '2naics', 'period_year']
            if all(data.get(var) is not None for var in required_vars):
                row = {'ticker': ticker, 'date': date}
                row.update(data)
                rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Check if we have any valid observations
    if df.empty:
        print("WARNING: No valid observations for Modified Jones regression")
        return {}
    
    # First pass: identify industries with insufficient observations in any year
    blacklisted_industries = set()
    for (period_year, naics), group in df.groupby(['period_year', '2naics']):
        if len(group) < 30 and period_year != 2014:
            blacklisted_industries.add(naics)
            print(f"DEBUG: Blacklisting industry {naics} due to insufficient observations in {period_year} ({len(group)} < 30)")
    
    # Store regression coefficients for each industry-period_year combination
    results = {}
    
    # Run separate regression for each period_year and 2-digit NAICS industry
    for (period_year, naics), group in df.groupby(['period_year', '2naics']):
        # Skip if industry is blacklisted
        if naics in blacklisted_industries:
            print(f"DEBUG: Skipping {period_year}_{naics} - industry is blacklisted")
            continue
        
        try:
            # Prepare dependent and independent variables (already scaled)
            y = group['total_accruals_scaled']
            X = group[['inverse_assets', 'delta_rev_minus_rec_scaled', 'ppe_scaled']]
            
            # Run OLS regression (missing='drop' handles any NaN values)
            model = sm.OLS(y, X, missing='drop')
            fitted = model.fit()
            
            # Store coefficients with industry-period_year key
            key = f"{int(period_year)}_{naics}"
            results[key] = {
                'beta_0': fitted.params['inverse_assets'],                  # Intercept coefficient
                'beta_1': fitted.params['delta_rev_minus_rec_scaled'],      # Revenue coefficient
                'beta_2': fitted.params['ppe_scaled'],                      # PPE coefficient
                'period_year': int(period_year),
                'naics': naics,
                'n_obs': len(group),                                        # Number of observations
                'r_squared': fitted.rsquared                                # Model fit
            }
        except Exception as e:
            # Skip if regression fails (e.g., multicollinearity, insufficient data)
            print(f"DEBUG: Regression failed for {period_year}_{naics} - Error: {str(e)}")
            continue
    
    return results

def calc_discretionary_accruals(fund_data_dict, modified_jones_results):
    """
    Calculate discretionary accruals for each earnings call and add to fund_data_dict.
    
    Discretionary accruals are calculated as the difference between total accruals
    and non-discretionary accruals (predicted by the Modified Jones model).
    
    Modified Jones Model:
    NDA = β₀ * (1/Assets) + β₁ * ((ΔRev - ΔRec)/Assets) + β₂ * (PPE/Assets)
    Discretionary Accruals = Total Accruals - NDA
    
    Args:
        fund_data_dict: Nested dict {ticker: {date: {data_point: value}}}
                       Must contain: 'total_accruals_scaled', 'inverse_assets',
                       'delta_rev_minus_rec_scaled', 'ppe_scaled', '2naics', 'period_year'
        modified_jones_results: Dict of regression coefficients by year and NAICS code
                               Key format: 'YYYY_NN' (e.g., '2024_33')
                               Values: dict with 'beta_0', 'beta_1', 'beta_2'
    
    Returns:
        Updated fund_data_dict with 'discretionary_accruals_scaled' field added (or None if can't be calculated)
    """
    # Loop over each ticker
    for ticker, dates_dict in fund_data_dict.items():
        # Loop over each earnings call date
        for date, data in dates_dict.items():
            # Check if we have all required variables
            required_vars = ['total_accruals_scaled', 'inverse_assets', 
                           'delta_rev_minus_rec_scaled', 'ppe_scaled', '2naics', 'period_year']
            
            if any(data.get(var) is None for var in required_vars):
                data['discretionary_accruals_scaled'] = None
                continue
            
            # Create lookup key using PERIOD_YEAR (not date.year) and 2-digit NAICS code
            # This must match how keys are created in calc_modified_jones
            key = f"{int(data.get('period_year'))}_{data.get('2naics')}"
            
            # Skip if regression coefficients not available for this year-industry
            if key not in modified_jones_results:
                data['discretionary_accruals_scaled'] = None
                continue
            
            # Get regression coefficients for this year-industry combination
            coeffs = modified_jones_results[key]
            
            # Calculate non-discretionary accruals (NDA) using Modified Jones model
            # NDA = β₀ * (1/Assets) + β₁ * ((ΔRev-ΔRec)/Assets) + β₂ * (PPE/Assets)
            nda = (coeffs['beta_0'] * data['inverse_assets'] + 
                   coeffs['beta_1'] * data['delta_rev_minus_rec_scaled'] + 
                   coeffs['beta_2'] * data['ppe_scaled'])
            
            # Calculate discretionary accruals (DA = Total Accruals - NDA)
            # This represents the "abnormal" portion of accruals not explained by the model
            data['discretionary_accruals_scaled'] = data['total_accruals_scaled'] - nda
    
    return fund_data_dict

def calc_SUE(ticker: str, target_date: datetime, earnings_data: Dict[str, Dict[datetime, Dict[str, float]]]):
    """
    Calculate standardized unexpected earnings (SUE) for a single earnings announcement.
    
    Args:
        ticker: Stock ticker symbol.
        target_date: Date of the earnings announcement.
        earnings_data: Dictionary with structure:
                      {ticker: {date: {'unexpected_earnings': value, 'eps_change': value}}}
    
    Returns:
        SUE value as float, or None if calculation is not possible.
    """
    
    # Check if ticker exists in earnings_data
    if ticker not in earnings_data:
        return None
    
    # Get all UE data for this ticker and convert to sorted list
    ticker_data = earnings_data[ticker]
    ue_data = []
    
    for date, metrics in ticker_data.items():
        ue = metrics.get('unexpected_earnings')
        if ue is not None:  # Only include if unexpected_earnings exists
            ue_data.append((date, ue))
    
    # Sort by date ascending
    ue_data.sort(key=lambda x: x[0])
    
    # Find UE for the target date and get prior 8 earnings
    current_ue = None
    prior_ues = []
    
    for i, (date, ue) in enumerate(ue_data):
        if date == target_date:
            current_ue = ue
            # Get the 8 earnings announcements before this date
            start_idx = max(0, i - 8)
            prior_ues = [ue_data[j][1] for j in range(start_idx, i)]
            break
    
    # Return None if we don't have the current UE or enough historical data
    if current_ue is None or len(prior_ues) < 8:
        return None
    
    # Calculate standard deviation of prior 8 UEs
    std_dev = np.std(prior_ues, ddof=1)  # Sample standard deviation
    
    # Return None if standard deviation is zero
    if std_dev == 0:
        return None
    
    # Calculate and return SUE
    sue = current_ue / std_dev
    return sue

def calc_standardized_eps_change(ticker: str, date: datetime, earnings_data: Dict[str, Dict[datetime, Dict[str, float]]]):
    """
    Calculate standardized earnings change for a single earnings announcement.
    
    Args:
        ticker: Stock ticker symbol.
        date: Date of the earnings announcement.
        earnings_data: Dictionary with structure:
                      {ticker: {date: {'unexpected_earnings': value, 'eps_change': value}}}
    
    Returns:
        Standardized earnings change value as float, or None if calculation is not possible.
    """
    
    # Check if ticker exists in earnings_data
    if ticker not in earnings_data:
        return None
    
    # Get all eps_change data for this ticker and convert to sorted list
    ticker_data = earnings_data[ticker]
    eps_change_data = []
    
    for announcement_date, metrics in ticker_data.items():
        eps_change = metrics.get('eps_change')
        if eps_change is not None:  # Only include if eps_change exists
            eps_change_data.append((announcement_date, eps_change))
    
    # Sort by date ascending
    eps_change_data.sort(key=lambda x: x[0])
    
    # Find eps_change for the target date and get prior 8 earnings
    current_eps_change = None
    prior_eps_changes = []
    
    for i, (announcement_date, eps_change) in enumerate(eps_change_data):
        if announcement_date == date:
            current_eps_change = eps_change
            # Get the 8 earnings announcements before this date
            start_idx = max(0, i - 8)
            prior_eps_changes = [eps_change_data[j][1] for j in range(start_idx, i)]
            break
    
    # Return None if we don't have the current eps_change or enough historical data
    if current_eps_change is None or len(prior_eps_changes) < 8:
        return None
    
    # Calculate standard deviation of prior 8 eps_changes
    std_dev = np.std(prior_eps_changes, ddof=1)  # Sample standard deviation
    
    # Return None if standard deviation is zero
    if std_dev == 0:
        return None
    
    # Calculate and return standardized earnings change
    standardized_eps_change = current_eps_change / std_dev
    
    return standardized_eps_change

def get_market_cap_bulk(
    init_dict: pd.DataFrame,
    permno_dict: Dict[str, int],
    trading_days: List[str],
    db
) -> pd.Series:
    """
    Get market capitalization 15 trading days before earnings announcements for all rows in init_dict.
    Makes bulk database queries in batches to avoid query size limits.
    
    Args:
        init_dict: DataFrame containing ticker and announcement_date columns.
        permno_dict: Dictionary with ticker as key and PERMNO as value.
        trading_days: List of trading day strings in format 'YYYY-MM-DD'.
        db: Database connection object with raw_sql method.
    
    Returns:
        pandas Series with market cap values (or None) for each row in init_dict.
    """
    
    # Convert trading_days list to date objects and sort
    trading_days_dates = sorted([datetime.strptime(d, '%Y-%m-%d').date() for d in trading_days])
    
    # Step 1: Pre-compute target dates (15 trading days before) for each earnings call
    target_dates_list = []
    
    for idx, row in init_dict.iterrows():
        ticker = row['ticker']
        earnings_date = row['announcement_date']
        
        # Get PERMNO for the ticker
        permno = permno_dict.get(ticker)
        if permno is None:
            target_dates_list.append((idx, None, None))
            continue
        
        # Convert earnings_date to date object if needed
        earnings_date_key = earnings_date.date() if isinstance(earnings_date, datetime) else earnings_date
        
        # Find all trading days on or before the earnings date
        valid_trading_days_before_earnings = [d for d in trading_days_dates if d <= earnings_date_key]
        
        # Check if we have at least 15 trading days
        if len(valid_trading_days_before_earnings) < 15:
            target_dates_list.append((idx, None, None))
            continue
        
        # Go back 15 trading days (not calendar days)
        selected_trading_day = valid_trading_days_before_earnings[-15]
        
        target_dates_list.append((idx, permno, selected_trading_day))
    
    # Step 2: Collect all valid (permno, date) pairs
    valid_pairs = [(permno, target_date) for idx, permno, target_date in target_dates_list 
                   if permno is not None and target_date is not None]
    
    # If no valid pairs, return Series of None values
    if not valid_pairs:
        return pd.Series([None] * len(init_dict), index=init_dict.index)
    
    # Step 3: Make bulk SQL queries IN BATCHES to avoid query size limits
    market_cap_lookup = {}
    BATCH_SIZE = 1000  # Process 1000 pairs at a time
    
    print(f'Fetching market cap data for {len(valid_pairs)} observations in batches of {BATCH_SIZE}...')
    
    total_records_found = 0
    
    for i in range(0, len(valid_pairs), BATCH_SIZE):
        batch = valid_pairs[i:i+BATCH_SIZE]
        
        # Create SQL query for this batch
        pairs_str = ', '.join([f"({permno}, '{date.strftime('%Y-%m-%d')}')" for permno, date in batch])
        
        crsp_query = f"""
        SELECT
            date,
            permno,
            ABS(prc) * shrout * 1000 AS market_cap
        FROM crsp.dsf
        WHERE (permno, date) IN ({pairs_str})
        """
        
        result = db.raw_sql(crsp_query)
        
        total_records_found += len(result)
        
        # Store results in lookup dictionary
        if not result.empty:
            for _, row_data in result.iterrows():
                permno_val = row_data['permno']
                date_val = row_data['date']
                
                # Convert date to date object for consistent lookup
                # Handle multiple possible types from SQL
                if isinstance(date_val, pd.Timestamp):
                    date_val = date_val.date()
                elif isinstance(date_val, str):
                    date_val = datetime.strptime(date_val, '%Y-%m-%d').date()
                # If already a date object, leave as is
                
                market_cap = float(row_data['market_cap']) if pd.notna(row_data['market_cap']) else None
                market_cap_lookup[(permno_val, date_val)] = market_cap
        
        # Progress indicator
        if (i + BATCH_SIZE) % 5000 == 0:
            print(f'  Processed {min(i + BATCH_SIZE, len(valid_pairs))} / {len(valid_pairs)} observations')
    
    print(f'Completed fetching market cap data. Found {total_records_found} records out of {len(valid_pairs)} requested.')
    
    # Step 4: Map market cap values back to init_dict rows
    market_cap_values = []
    for idx, permno, target_date in target_dates_list:
        if permno is None or target_date is None:
            market_cap_values.append(None)
        else:
            market_cap = market_cap_lookup.get((permno, target_date), None)
            market_cap_values.append(market_cap)
    
    return pd.Series(market_cap_values, index=init_dict.index)
