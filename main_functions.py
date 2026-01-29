import json
import pickle
from openai import OpenAI
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime, timedelta
from difflib import SequenceMatcher

# Initialize OpenAI client
load_dotenv()
OpenAI_api_key = os.getenv('OpenAI_api_key')
OpenAI_client_import = OpenAI(api_key=OpenAI_api_key)

# Load data variables
sentiment_list = pd.read_csv("C:/Users/haase/Downloads/Loughran-McDonald_MasterDictionary_1993-2024.csv")
positive_words_import = sentiment_list.loc[sentiment_list['Positive'] > 0, 'Word'].str.lower().tolist()
negative_words_import = sentiment_list.loc[sentiment_list['Negative'] > 0, 'Word'].str.lower().tolist()

with open('data/consolidated_papers.json', 'r') as f:
    consolidated_papers_import = json.load(f)

with open('data/permno_dict.json', 'r') as f:
    permno_dict_import = json.load(f)

with open('data/trading_days.json', 'r') as f:
    trading_days_import = json.load(f)

with open('data/keywords_list.pkl', 'rb') as f:
    keywords_list_import = pickle.load(f)

# Main independent variables
def calc_quantity_and_sentiment(text_elements, keywords=keywords_list_import, positive_words=positive_words_import, negative_words=negative_words_import, similarity_threshold=0.93, element_threshold=0.005):
    """
    Count total words and keyword occurrences across multiple text elements.
    Uses fuzzy matching with caching for better performance.
    Also performs sentiment analysis and filters texts by keyword density.
    
    Args:
        text_elements: List of strings containing text to analyze
        keywords: List of strings containing keywords to count
        positive_words: List of positive words for sentiment analysis (optional)
        negative_words: List of negative words for sentiment analysis (optional)
        similarity_threshold: Float between 0-1 for fuzzy matching sensitivity (default 0.85)
                            Higher = stricter matching, Lower = more lenient
        element_threshold: Float for minimum keyword density per text element (default 0.001)
                         Texts with relative_keyword_count > threshold are included in output
    
    Returns:
        tuple: (relative_keyword_count, AI_related_text_elements, negativity_score)
            - relative_keyword_count: float, % of keywords from total word count (as decimal)
            - AI_related_text_elements: list of strings that exceed the element_threshold
            - negativity_score: float, sentiment score between -1 (positive) and 1 (negative)
    """
    total_word_count = 0
    total_keyword_count = 0
    AI_related_text_elements = []
    positive_word_count = 0
    negative_word_count = 0
    
    # Convert keywords to lowercase for case-insensitive matching
    keywords_lower = [keyword.lower() for keyword in keywords]
    
    # Cache for fuzzy matching results to avoid redundant calculations
    similarity_cache = {}
    
    def is_similar(word1, word2, threshold):
        """Check if two words are similar using sequence matching with caching."""
        cache_key = (word1, word2)
        if cache_key not in similarity_cache:
            similarity_cache[cache_key] = SequenceMatcher(None, word1, word2).ratio() >= threshold
        return similarity_cache[cache_key]
    
    # Loop through each text element
    for text in text_elements:
        # Reset per-element counters
        element_word_count = 0
        element_keyword_count = 0
        
        # Split text into words
        words = text.split()
        
        # Count words for this element
        element_word_count = len(words)
        
        # Add to total word count
        total_word_count += element_word_count
        
        # Count keywords in this text element
        for word in words:
            # Remove common punctuation and convert to lowercase
            clean_word = word.strip('.,!?;:"()[]{}').lower()
            
            # Skip empty strings
            if not clean_word:
                continue
            
            # Sentiment analysis: count positive and negative words
            if clean_word in positive_words:
                positive_word_count += 1
            elif clean_word in negative_words:
                negative_word_count += 1
                
            # AI keyword quantity: First check for exact match (much faster than similarity calculation)
            if clean_word in keywords_lower:
                total_keyword_count += 1
                element_keyword_count += 1
                continue
            
            # Only do fuzzy matching if no exact match found
            for keyword in keywords_lower:
                if is_similar(clean_word, keyword, similarity_threshold):
                    total_keyword_count += 1
                    element_keyword_count += 1
                    break  # Count each word only once even if it matches multiple keywords
        
        # Check if this text element exceeds the threshold
        if element_word_count > 0:
            element_relative_keyword_count = element_keyword_count / element_word_count
            if element_relative_keyword_count > element_threshold:
                AI_related_text_elements.append(text)
    
    # Avoid division by zero for total
    if total_word_count == 0:
        return None, AI_related_text_elements, None
    
    relative_keyword_count = total_keyword_count / total_word_count
    negativity_score = (negative_word_count - positive_word_count) / (negative_word_count + positive_word_count + 1) # change of neg. score has to be computed in main script
    
    return relative_keyword_count, AI_related_text_elements, negativity_score

def calc_quality(AI_related_text_elements, call_date, OpenAI_client=OpenAI_client_import, consolidated_papers=consolidated_papers_import, years_back=3):
    """
    Calculate semantic similarity of earnings call AI content by measuring alignment 
    with recent academic literature using cosine similarity.
    
    Args:
        AI_related_text_elements (list of str): AI-related text segments from call
        call_date (str): Call date in 'YYYY-MM-DD' format
        consolidated_papers (list of dict): Papers with 'publicationDate' and 
            'abstract_embedding' fields
        years_back (int, optional): Years before call_date to include papers. Default 3.
    
    Returns:
        float or None: Word-weighted average cosine similarity (-1 to 1, higher = 
            better quality). None if no papers found in time window.
    """
    
    # Check if we have any text elements to process
    if not AI_related_text_elements or len(AI_related_text_elements) == 0:
        return 0

    # 1.) GET AVERAGE EMBEDDING OF RELEVANT PAPERS
    
    # Convert reference_date string to datetime object
    ref_date = datetime.strptime(call_date, "%Y-%m-%d")
    
    # Calculate the cutoff date (N years before reference date)
    cutoff_date = ref_date - timedelta(days=365 * years_back)
    
    # Collect embeddings from papers in the time range
    embeddings_list = []
    
    for paper in consolidated_papers:
        pub_date_str = paper.get('publicationDate')
        
        # Skip papers without publication date
        if not pub_date_str:
            continue
        
        try:
            # Convert publication date to datetime
            pub_date = datetime.strptime(pub_date_str, "%Y-%m-%d")
            
            # Check if paper is within the time range
            if cutoff_date <= pub_date <= ref_date:
                # Check if paper has an embedding
                embedding = paper.get('abstract_embedding')
                if embedding is not None:
                    embeddings_list.append(embedding)
        except ValueError:
            # Skip papers with invalid date format
            continue
    
    # Check if we found any papers
    if len(embeddings_list) == 0:
        return None
    
    # Calculate average embedding using numpy
    average_paper_embedding = np.mean(embeddings_list, axis=0)
    
    
    # 2.) CALCULATE AVERAGE COSINE SIMILARITY, WEIGHTED BY WORDS
    
    # Function: Get embedding
    def get_embedding(text, OpenAI_client, model="text-embedding-3-small"):
        """
        Get embedding for a given text using OpenAI's embedding model.

        Args:
            text (str): The text to embed
            model (str): The embedding model to use

        Returns:
            list: The embedding vector
        """
        # Call the OpenAI API
        response = OpenAI_client.embeddings.create(
            input=text,
            model=model
        )

        # Extract the embedding vector
        embedding = response.data[0].embedding

        return embedding

    # Function: Calculate cosine similarity
    def calc_cosine_similarity(vector1, vector2):
        """
        Calculate cosine similarity between two vectors.

        Args:
            vector1 (list or numpy.ndarray): First vector
            vector2 (list or numpy.ndarray): Second vector

        Returns:
            float: Cosine similarity value between -1 and 1
        """
        # Convert to numpy arrays
        vec1 = np.array(vector1)
        vec2 = np.array(vector2)

        # Calculate dot product
        dot_product = np.dot(vec1, vec2)

        # Calculate magnitudes (norms)
        magnitude1 = np.linalg.norm(vec1)
        magnitude2 = np.linalg.norm(vec2)

        # Calculate cosine similarity
        similarity = dot_product / (magnitude1 * magnitude2)

        return similarity
    
    total_words = 0
    weighted_similarity_sum = 0
    
    for element in AI_related_text_elements:
        # Count words in this text element
        word_count = len(element.split())
        
        # Get embedding for this text element
        element_embedding = get_embedding(element, OpenAI_client)
        
        # Calculate cosine similarity
        cosine_similarity = calc_cosine_similarity(average_paper_embedding, element_embedding)
        
        # Add to weighted sum
        weighted_similarity_sum += cosine_similarity * word_count
        total_words += word_count
        
    avg_cosine_similarity = weighted_similarity_sum / total_words
    
    return avg_cosine_similarity

# Dependent variable
def calc_CAR(date, ticker, db, permno_dict=permno_dict_import, trading_days=trading_days_import, estimation_window=(-250, -15), CAR_time_windows=[1,3,15,30], latest_crsp_date='2024-12-31'):
    """
    Calculate Cumulative Abnormal Returns using Fama-French 5-factor model.
    Uses trading days instead of calendar days for all time windows.
    
    Args:
        date (str or datetime): Event date in format 'YYYY-MM-DD'
        ticker (str): Stock ticker symbol
        db: Database connection object with raw_sql method.
        permno_dict: Dictionary with ticker as key and PERMNO as value.
        trading_days: List of all trading days as strings in format 'YYYY-MM-DD'
        estimation_window (tuple): Estimation window relative to event date in TRADING days
                                   Default: (-250, -15) means 250 to 15 trading days before event
        CAR_time_windows (list): List of trading days after event for CAR calculation
    
    Returns:
        tuple: (CAR_data, crsp_fund)
            - CAR_data: List of dictionaries containing cumulative abnormal returns for each time window
            - crsp_fund: Dictionary containing fundamental data from estimation period
    """
    
    # Convert date to datetime if string
    if isinstance(date, str):
        event_date = pd.to_datetime(date)
    else:
        event_date = date
    
    #########################################################################################################################
    # 0.) GET TRADING DATES FROM PRE-LOADED LIST
    
    # Convert trading days list (strings) to datetime objects for comparison
    trading_dates = [pd.to_datetime(d) for d in trading_days]
    
    # Find the event date in trading dates (or closest trading date on or after event)
    event_trading_date = None
    for td in trading_dates:
        if td >= event_date:
            event_trading_date = td
            break
    
    # Raise error if no valid trading date found
    if event_trading_date is None:
        raise ValueError(f"No trading date found on or after {event_date}")
    
    # Get index position of event trading date in the list
    event_idx = trading_dates.index(event_trading_date)
    
    #########################################################################################################################
    # 1.) GET PERMNO FROM PRE-LOADED DICT
    
    # Extract PERMNO for the ticker
    permno = permno_dict.get(ticker)
    
    if permno is None:
        raise ValueError(f"No PERMNO found for ticker {ticker}")
    
    company_name = ticker  # Use ticker as company name since it's not in permno_dict
    
    #########################################################################################################################
    # 2.) ESTIMATE REGRESSION COEFFICIENTS
    
    # Calculate regression window dates based on TRADING days
    regression_start_idx = event_idx + estimation_window[0]  # estimation_window[0] is negative
    regression_end_idx = event_idx + estimation_window[1]    # estimation_window[1] is negative
    
    # Ensure indices are valid
    if regression_start_idx < 0:
        raise ValueError(f"Not enough historical trading data. Need {-estimation_window[0]} trading days before event.")
    if regression_end_idx < 0:
        regression_end_idx = 0
    
    regression_start_date = trading_dates[regression_start_idx]
    regression_end_date = trading_dates[regression_end_idx]
     
    # Fama-French 5 factor model query
    ff_query = """
    SELECT date, mktrf, smb, hml, rmw, cma, rf
    FROM ff.fivefactors_daily
    WHERE date >= %s AND date <= %s
    ORDER BY date
    """
    regression_ff_data = db.raw_sql(ff_query, params=(regression_start_date, regression_end_date))
    regression_ff_data['date'] = pd.to_datetime(regression_ff_data['date'])

    # CRSP stock data query using PERMNO only (now includes vol and shrout for additional calculations)
    crsp_query = """
    SELECT date, ret, vol, shrout
    FROM crsp.dsf
    WHERE permno = %s
    AND date >= %s AND date <= %s
    ORDER BY date
    """
    regression_stock_data = db.raw_sql(crsp_query, params=(permno, regression_start_date, regression_end_date))
    regression_stock_data['date'] = pd.to_datetime(regression_stock_data['date'])

    # Merge stock returns with Fama-French factors
    regression_merged_data = pd.merge(regression_stock_data, regression_ff_data, on='date', how='inner')

    # Convert all numeric columns to float (handle any string values)
    numeric_cols = ['ret', 'mktrf', 'smb', 'hml', 'rmw', 'cma', 'rf', 'vol', 'shrout']
    for col in numeric_cols:
        regression_merged_data[col] = pd.to_numeric(regression_merged_data[col], errors='coerce')

    # Handle missing data
    regression_merged_data = regression_merged_data.dropna(subset=numeric_cols)
    
    # Check if we have enough actual data after merging for the estimation window
    expected_days = estimation_window[1] - estimation_window[0]  # Both are negative, so this gives positive count
    if len(regression_merged_data) < expected_days:
        raise ValueError(
            f"Insufficient historical data: Only {len(regression_merged_data)} trading days available in estimation window, "
            f"but {expected_days} trading days required (from {estimation_window[0]} to {estimation_window[1]} days before event). "
            f"First available date in merged data: {regression_merged_data['date'].min()}, "
            f"Last date in estimation window: {regression_merged_data['date'].max()}"
        )
    
    # Calculate excess returns (dependent variable)
    regression_merged_data['excess_ret'] = regression_merged_data['ret'] - regression_merged_data['rf']

    # Prepare regression data
    y = regression_merged_data['excess_ret'].astype(float)
    X = regression_merged_data[['mktrf', 'smb', 'hml', 'rmw', 'cma']].astype(float)

    # Add constant for intercept (alpha)
    X = sm.add_constant(X)

    # Run OLS regression
    regression_model = sm.OLS(y, X)
    regression_raw_results = regression_model.fit()

    # Extract key results
    regression_results = {
        'ticker': ticker,
        'permno': permno,
        'company_name': company_name,
        'event_date': event_date,
        'event_trading_date': event_trading_date,
        'estimation_window': estimation_window,
        'n_observations': len(y),
        'alpha': regression_raw_results.params['const'],
        'beta_mkt': regression_raw_results.params['mktrf'],
        'beta_smb': regression_raw_results.params['smb'],
        'beta_hml': regression_raw_results.params['hml'],
        'beta_rmw': regression_raw_results.params['rmw'],
        'beta_cma': regression_raw_results.params['cma'],
        'r_squared': regression_raw_results.rsquared,
        'adj_r_squared': regression_raw_results.rsquared_adj,
        'regression_summary': regression_raw_results.summary(),
        'data': regression_merged_data
    }
    
    #########################################################################################################################
    # 3.) CALCULATE EXPECTED RETURNS BASED ON REGRESSION COEFFICIENTS
    
    # Calculate end date of CAR window based on TRADING days
    abnormal_returns_end_idx = event_idx + CAR_time_windows[-1]    
    abnormal_returns_end_date = trading_dates[abnormal_returns_end_idx]

    # Fama-French SQL query
    abnormal_returns_ff_data = db.raw_sql(ff_query, params=(event_trading_date, abnormal_returns_end_date))
    abnormal_returns_ff_data['date'] = pd.to_datetime(abnormal_returns_ff_data['date'])

    # CRSP SQL query using PERMNO only
    abnormal_returns_stock_data = db.raw_sql(crsp_query, params=(permno, event_trading_date, abnormal_returns_end_date))
    abnormal_returns_stock_data['date'] = pd.to_datetime(abnormal_returns_stock_data['date'])

    # Merge stock returns with Fama-French factors
    abnormal_returns_merged_data = pd.merge(abnormal_returns_stock_data, abnormal_returns_ff_data, on='date', how='inner')

    # Convert all numeric columns to float (handle any string values)
    for col in numeric_cols:
        abnormal_returns_merged_data[col] = pd.to_numeric(abnormal_returns_merged_data[col], errors='coerce')

    # Handle missing data
    abnormal_returns_merged_data = abnormal_returns_merged_data.dropna(subset=numeric_cols)

    # Check if we have enough actual data after merging
    if len(abnormal_returns_merged_data) < CAR_time_windows[-1]:
        raise ValueError(
            f"Insufficient data: Only {len(abnormal_returns_merged_data)} trading days available after event date, "
            f"but {CAR_time_windows[-1]} trading days required for longest CAR window. "
            f"Last available date in merged data: {abnormal_returns_merged_data['date'].max()}"
        )
    
    # Calculate expected returns
    abnormal_returns_merged_data['expected_excess_returns'] = (
        regression_results['alpha'] +
        regression_results['beta_mkt'] * abnormal_returns_merged_data['mktrf'] +
        regression_results['beta_smb'] * abnormal_returns_merged_data['smb'] +
        regression_results['beta_hml'] * abnormal_returns_merged_data['hml'] +
        regression_results['beta_rmw'] * abnormal_returns_merged_data['rmw'] +
        regression_results['beta_cma'] * abnormal_returns_merged_data['cma']
    )

    abnormal_returns_merged_data['expected_returns'] = abnormal_returns_merged_data['expected_excess_returns'] + abnormal_returns_merged_data['rf']
    
    #########################################################################################################################
    # 4.) CALCULATE CUMULATIVE ABNORMAL RETURNS
    
    # Calculate daily abnormal returns
    abnormal_returns_merged_data['daily_abnormal_returns'] = abnormal_returns_merged_data['ret'] - abnormal_returns_merged_data['expected_returns']
    
    # Calculate cumulative abnormal returns based on TRADING days
    CAR_data = []

    for trading_days_after in CAR_time_windows:
        CAR_cutoff_idx = event_idx + trading_days_after
        
        if CAR_cutoff_idx >= len(trading_dates):
            # Skip if we don't have enough data
            continue
            
        CAR_cutoff_date = trading_dates[CAR_cutoff_idx]
        time_window = f'CAR[0:{trading_days_after} trading days]'
        CAR = abnormal_returns_merged_data[abnormal_returns_merged_data['date'] <= CAR_cutoff_date]['daily_abnormal_returns'].sum()
        CAR_data.append({'Time window': time_window, 'CAR': CAR})
    
    #########################################################################################################################
    # 5.) CALCULATE ADDITIONAL CRSP FUNDAMENTALS FROM ESTIMATION PERIOD
    # (Separate from main CAR calculation - used to reduce API calls by retrieving alongside CAR data)
    
    # Calculate share turnover for each day in estimation period: vol / shrout / 1000
    # (shrout is in thousands, so we divide by 1000 to get the actual ratio)
    regression_merged_data['share_turnover'] = regression_merged_data['vol'] / regression_merged_data['shrout'] / 1000
    
    # Calculate average share turnover over estimation period
    avg_share_turnover = regression_merged_data['share_turnover'].mean()
    
    # Calculate volatility of returns over estimation period (standard deviation)
    returns_volatility = regression_merged_data['ret'].std()
    
    # Create dictionary with fundamental data
    crsp_fund = {
        'avg_share_turnover': avg_share_turnover,
        'returns_volatility': returns_volatility
    }
    
    return CAR_data, crsp_fund
