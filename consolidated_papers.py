from dotenv import load_dotenv
import os
import requests
import time
from datetime import datetime, timedelta
from openai import OpenAI
import re
import json
import pickle

# Import keywords
with open('data/keywords_list.pkl', 'rb') as f:
    keywords = pickle.load(f)

# Get API keys from environment variables
load_dotenv()
Semanticscholar_api_key = os.getenv('Semanticscholar_api_key')
OpenAI_api_key = os.getenv('OpenAI_api_key')

# Configuration
search_url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
field_of_study = "Computer Science"
top_n = 100  # Number of top papers to keep per year
wait_time = 1  # Seconds to wait between API requests
Semanticscholar_header = {"x-api-key": Semanticscholar_api_key}
end_date_str = "2024-10-31"  # Last date of the most recent time frame
years_back = 15  # How many years to go back

# Functions
def clean_abstract(text):
    """
    Clean abstract text.
    
    Args:
        text (str): The raw abstract text
        
    Returns:
        str: The cleaned abstract text
    """
    # 1. Remove newlines and replace with spaces
    text = text.replace('\n', ' ')
    
    # 2. Remove "ABSTRACT" (case-insensitive) at the beginning AND end
    text = re.sub(r'^\s*ABSTRACT\s*:?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*ABSTRACT\s*:?\s*$', '', text, flags=re.IGNORECASE)
    
    # 3. Clean up multiple spaces and trim
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

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

######################################################################################
# EXECUTION CODE
######################################################################################

# Initialize OpenAI client
OpenAI_client = OpenAI(api_key=OpenAI_api_key)

# Convert end_date string to datetime object
end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

# Dictionary to store overall_top_papers for each year
results_by_year = {}

# Loop through each year
for year_offset in range(years_back):
    # Calculate the time frame for this iteration
    current_end_date = end_date - timedelta(days=365 * year_offset)
    current_start_date = current_end_date - timedelta(days=365)
    
    # Format dates as strings
    start_date_formatted = current_start_date.strftime("%Y-%m-%d")
    end_date_formatted = current_end_date.strftime("%Y-%m-%d")
    
    # Get the year for naming the results list
    year_name = current_end_date.year
    list_name = f"overall_top_papers_{year_name}"
    
    print(f"\n{'#'*70}")
    print(f"# PROCESSING YEAR: {year_name}")
    print(f"# Time frame: {start_date_formatted} to {end_date_formatted}")
    print(f"{'#'*70}")
    
    # Initialize the overall list for this year
    overall_top_papers = []
    
    for idx, keyword in enumerate(keywords):
        # Wait between requests (except for the first one of each year)
        if idx > 0:
            print(f"\nWaiting {wait_time} seconds before next request...")
            time.sleep(wait_time)
        
        print(f"\n{'='*60}")
        print(f"Processing keyword: '{keyword}'")
        print(f"{'='*60}")
        
        query_params = {
            "query": keyword,
            "fieldsOfStudy": field_of_study,
            "publicationDateOrYear": f"{start_date_formatted}:{end_date_formatted}",
            "fields": "paperId,citationCount,abstract,title,publicationDate",
            "sort": "citationCount:desc"
            # Note: bulk endpoint doesn't support 'limit' parameter
        }
        
        response = requests.get(search_url, params=query_params, headers=Semanticscholar_header)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            continue
        
        response_data = response.json()
        all_papers = response_data.get('data', [])
        print(f"Retrieved {len(all_papers)} total papers for '{keyword}'")
        
        # Filter papers: keep only those with valid abstracts
        papers_with_abstracts = []
        for paper in all_papers:
            # Check if abstract exists and is not empty
            if paper.get('abstract') is not None and paper.get('abstract').strip() != '':
                # Create new dictionary with selected fields
                new_paper = {
                    'paperId': paper['paperId'],
                    'citationCount': paper['citationCount'],
                    'abstract': paper.get('abstract'),
                    'title': paper.get('title'),
                    'publicationDate': paper.get('publicationDate')
                }
                papers_with_abstracts.append(new_paper)
        
        print(f"Found {len(papers_with_abstracts)} papers with abstracts")
        
        # Sort by citation count (descending) and take top N
        papers_with_abstracts.sort(key=lambda x: x['citationCount'], reverse=True)
        current_papers = papers_with_abstracts[:top_n]
        
        # Add information about keyword to papers
        for paper in current_papers:
            paper['keyword'] = keyword
        
        print(f"Selected top {len(current_papers)} most cited papers with abstracts")
        
        # First keyword: initialize overall list
        if len(overall_top_papers) == 0:
            overall_top_papers = current_papers.copy()
            print(f"Initialized overall list with {len(overall_top_papers)} papers")
            continue # ends loop and goes to next keyword
        
        # For subsequent keywords: merge and keep top N
        # Find the minimum citation count in overall list (it's the last one since sorted)
        min_citations_overall = overall_top_papers[-1]['citationCount']
        
        # Process each paper from current keyword
        papers_added = 0
        for paper in current_papers:
            # Early exit optimization: if current paper has fewer citations than 
            # the minimum in overall list, skip it and all remaining papers
            if paper['citationCount'] <= min_citations_overall:
                break
            
            # Check if paper already exists in overall list
            paper_exists = False
            for p in overall_top_papers:
                if p['paperId'] == paper['paperId']:
                    paper_exists = True
                    break
            
            if not paper_exists:
                overall_top_papers.append(paper)
                papers_added += 1
        
        # Sort by citation count (descending) and keep only top N
        overall_top_papers.sort(key=lambda x: x['citationCount'], reverse=True)
        overall_top_papers = overall_top_papers[:top_n]
        
        # Update minimum for next iteration
        if len(overall_top_papers) > 0:
            min_citations_overall = overall_top_papers[-1]['citationCount']
        
        print(f"Added {papers_added} new papers")
        print(f"Overall list now contains {len(overall_top_papers)} papers")
        print(f"Citation range: {overall_top_papers[0]['citationCount']} to {overall_top_papers[-1]['citationCount']}")
    
    # OUTPUT after for-loop: list (overall_top_papers) with most cited papers across all keywords for one year
    
    # Clean abstracts
    for paper in overall_top_papers:
        original_abstract = paper['abstract']
        cleaned_abstract = clean_abstract(original_abstract)
        paper['abstract'] = cleaned_abstract
                  
    print("\nAbstracts cleaned")
    
    # Add embedding for abstracts of overall_top_papers
    for paper in overall_top_papers:
        abstract_embedding = get_embedding(paper['abstract'], OpenAI_client)
        paper['abstract_embedding'] = abstract_embedding
        
    print(f"\nAdded abstract embeddings to {list_name}")
    
    # Store the results for this year
    results_by_year[list_name] = overall_top_papers
    
    # Final results for this year
    print(f"\n{'='*60}")
    print(f"RESULTS FOR YEAR {year_name}")
    print(f"{'='*60}")
    print(f"Total papers in {list_name}: {len(overall_top_papers)}")
    if overall_top_papers:
        print(f"Highest citations: {overall_top_papers[0]['citationCount']}")
        print(f"Lowest citations: {overall_top_papers[-1]['citationCount']}")
        print(f"\nTop 5 most cited papers for {year_name}:")
        for i, paper in enumerate(overall_top_papers[:5], 1):
            abstract_preview = paper.get('abstract', 'No abstract available')
            if abstract_preview and len(abstract_preview) > 100:
                abstract_preview = abstract_preview[:100] + "..."
            print(f"  {i}. {paper.get('title', 'No title')}")
            print(f"     Paper ID: {paper['paperId']}, Citations: {paper['citationCount']}")
            print(f"     Publication Date: {paper.get('publicationDate', 'Unknown')}, Keyword: '{paper['keyword']}'")
            print(f"     Abstract: {abstract_preview}")
            print()

# Final summary across all years
print(f"\n{'#'*70}")
print(f"# FINAL SUMMARY - ALL YEARS")
print(f"{'#'*70}")
for list_name, papers in results_by_year.items():
    year = list_name.split('_')[-1]
    print(f"{list_name}: {len(papers)} papers (Year {year})")

print(f"\nYou can access individual year results using:")
print(f"results_by_year['overall_top_papers_2018']")
print(f"results_by_year['overall_top_papers_2017']")
print(f"etc.")

# Consolidate all papers from all years into one list
consolidated_papers = []
# Iterate through each year's results
for year_list_name, papers_list in results_by_year.items():
    for paper in papers_list:
        # Create a simplified dictionary with only the desired fields
        paper_to_append = {
            'publicationDate': paper.get('publicationDate'),
            'title': paper.get('title'),
            'abstract': paper.get('abstract'),
            'abstract_embedding': paper.get('abstract_embedding')
        }
        consolidated_papers.append(paper_to_append)
# Sort by publicationDate (newest first)
# Handle None values in publicationDate
consolidated_papers.sort(
    key=lambda x: x['publicationDate'] if x['publicationDate'] else '0000-00-00',
    reverse=True
)
# Display results
print(f"{'='*80}")
print(f"CONSOLIDATED PAPERS - ALL YEARS")
print(f"{'='*80}")
print(f"Total unique papers: {len(consolidated_papers)}\n")
# Show first 10 papers
print(f"Top 10 newest papers:")
print(f"{'-'*80}")
for i, paper in enumerate(consolidated_papers[:10], 1):
    abstract_preview = paper.get('abstract', 'No abstract available')
    if abstract_preview and len(abstract_preview) > 100:
        abstract_preview = abstract_preview[:100] + "..."
    
    # Get embedding preview (first 4 values)
    embedding = paper.get('abstract_embedding')
    if embedding:
        embedding_preview = embedding[:6]
        embedding_str = f"[{', '.join([f'{x:.4f}' for x in embedding_preview])}, ...]"
    else:
        embedding_str = "No embedding available"
    
    print(f"{i}. Date: {paper.get('publicationDate', 'Unknown')}")
    print(f"   Title: {paper.get('title', 'No title')}")
    print(f"   Abstract: {abstract_preview}")
    print(f"   Embedding: {embedding_str}")
    print()
print(f"\nThe consolidated list is stored in the 'consolidated_papers' variable")

# Save consolidated_papers
with open("data/consolidated_papers.json", "w") as f:
    json.dump(consolidated_papers, f, indent=2)