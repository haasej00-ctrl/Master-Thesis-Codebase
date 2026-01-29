import pickle
import pandas as pd
from main_functions import calc_quality

# Load the main dataframe
print("Loading main_dataframe.pkl...")
with open('data/main_dataframe.pkl', 'rb') as f:
    df = pickle.load(f)

print(f"Loaded dataframe with {len(df)} rows and {len(df.columns)} columns")
print(f"Columns: {df.columns.tolist()}")

# Initialize new column
df['AI_quality_all_texts'] = None

print("\nCalculating AI_quality_all_texts for each row...")
print("=" * 60)

# Loop over rows and calculate quality
total_rows = len(df)
iteration_count = 0

for idx, row in df.iterrows():
    iteration_count += 1
    
    # Print progress every 500 rows or on the last row
    if iteration_count % 500 == 0 or iteration_count == total_rows:
        print(f"PROGRESS: {iteration_count}/{total_rows} completed ({iteration_count/total_rows*100:.1f}%)")
    
    try:
        # Calculate quality using the full transcript
        # Convert call_date from Timestamp to string format
        call_date_str = row['call_date'].strftime('%Y-%m-%d')
        
        quality = calc_quality(
            AI_related_text_elements=row['transcript'],
            call_date=call_date_str
        )
        
        df.at[idx, 'AI_quality_all_texts'] = quality
        
    except Exception as e:
        print(f"  Error at iteration {iteration_count} (index {idx}): {e}")
        df.at[idx, 'AI_quality_all_texts'] = None

print("=" * 60)
print("\nCalculation complete!")

# Display summary statistics
print(f"\nSummary of AI_quality_all_texts:")
print(f"  Non-null values: {df['AI_quality_all_texts'].notna().sum()}")
print(f"  Null values: {df['AI_quality_all_texts'].isna().sum()}")
if df['AI_quality_all_texts'].notna().any():
    print(f"  Mean: {df['AI_quality_all_texts'].mean():.4f}")
    print(f"  Min: {df['AI_quality_all_texts'].min():.4f}")
    print(f"  Max: {df['AI_quality_all_texts'].max():.4f}")

# Save the extended dataframe
output_path = 'data/main_dataframe_extended.pkl'
df.to_pickle(output_path)
print(f"\nSaved extended dataframe to {output_path}")
print(f"Final dataframe: {len(df)} rows, {len(df.columns)} columns")