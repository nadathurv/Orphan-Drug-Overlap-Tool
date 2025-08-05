"""
Main pipeline for drug overlap analysis between CDSCO and FDA databases.

This script orchestrates the entire process:
1. Load data from both databases
2. Perform fuzzy matching
3. Format dates
4. Generate overlap report
"""

import pandas as pd
import sys
import os

# Add parent directory to path for imports when running from different locations
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_cdsco_data, load_fda_data
from fuzzy_matcher import DrugMatcher
from date_formatter import standardize_dates


def create_overlap_report(matches, output_file='output/overlap.csv'):
    """
    Create the final overlap report with required fields.
    
    Includes:
    - Drug names from both databases
    - Indications from both databases (for verification)
    - Approval dates in standardized format
    - Match confidence score
    """
    if not matches:
        print("No overlapping drugs found!")
        return None
    
    # Convert matches to DataFrame
    overlap_df = pd.DataFrame(matches)
    
    # Standardize dates
    overlap_df['CDSCO_Approval_Date_Formatted'] = standardize_dates(overlap_df['cdsco_approval_date'])
    overlap_df['FDA_Approval_Date_Formatted'] = standardize_dates(overlap_df['fda_approval_date'])
    
    # Select and rename columns for final output
    output_df = pd.DataFrame({
        'Drug_Name_CDSCO': overlap_df['cdsco_drug'],
        'Drug_Name_FDA': overlap_df['fda_drug'],
        'Indication_CDSCO': overlap_df['cdsco_indication'],
        'Indication_FDA': overlap_df['fda_indication'],
        'Approval_Date_CDSCO': overlap_df['CDSCO_Approval_Date_Formatted'],
        'Approval_Date_FDA': overlap_df['FDA_Approval_Date_Formatted'],
        'Match_Score': overlap_df['match_score'],
        'FDA_Generic_Name': overlap_df['fda_generic'],
        'FDA_Trade_Name': overlap_df['fda_trade']
    })
    
    # Sort by match score (highest first) to review best matches
    output_df = output_df.sort_values('Match_Score', ascending=False)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    output_df.to_csv(output_file, index=False, encoding='utf-8')
    
    return output_df


def analyze_matches(overlap_df):
    """
    Provide analysis of the matching results.
    
    Helps identify:
    - High confidence matches (>95% similarity)
    - Medium confidence matches (85-95%)
    - Potential false positives (check indications)
    """
    print("\n=== OVERLAP ANALYSIS ===")
    print(f"Total overlapping drugs found: {len(overlap_df)}")
    
    # Confidence breakdown
    high_confidence = len(overlap_df[overlap_df['Match_Score'] >= 95])
    medium_confidence = len(overlap_df[(overlap_df['Match_Score'] >= 85) & (overlap_df['Match_Score'] < 95)])
    
    print(f"\nConfidence breakdown:")
    print(f"  High confidence (≥95%): {high_confidence}")
    print(f"  Medium confidence (85-94%): {medium_confidence}")
    
    # Sample of matches
    print("\nTop 10 matches:")
    for idx, row in overlap_df.head(10).iterrows():
        print(f"  {row['Drug_Name_CDSCO']} <-> {row['Drug_Name_FDA']} (Score: {row['Match_Score']}%)")
    
    # Check for potential issues
    print("\nPotential issues to review:")
    
    # Very different indication lengths might indicate mismatch
    overlap_df['indication_length_diff'] = abs(
        overlap_df['Indication_CDSCO'].str.len() - 
        overlap_df['Indication_FDA'].str.len()
    )
    
    large_diff = overlap_df[overlap_df['indication_length_diff'] > 200]
    if len(large_diff) > 0:
        print(f"  - {len(large_diff)} matches have very different indication lengths (review for false positives)")
    
    # Missing dates
    missing_cdsco_dates = overlap_df[overlap_df['Approval_Date_CDSCO'] == ''].shape[0]
    missing_fda_dates = overlap_df[overlap_df['Approval_Date_FDA'] == ''].shape[0]
    
    if missing_cdsco_dates > 0:
        print(f"  - {missing_cdsco_dates} CDSCO drugs missing approval dates")
    if missing_fda_dates > 0:
        print(f"  - {missing_fda_dates} FDA drugs missing approval dates")


def run_pipeline(threshold=85):
    """
    Run the complete drug overlap analysis pipeline.
    
    Args:
        threshold: Minimum fuzzy matching score (default 85%)
    """
    print("=== DRUG OVERLAP ANALYSIS PIPELINE ===")
    print(f"Matching threshold: {threshold}%")
    
    # Step 1: Load data
    print("\nStep 1: Loading data...")
    try:
        # Adjust paths based on where script is run from
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cdsco_path = os.path.join(base_path, 'data', 'cdsco.csv')
        fda_path = os.path.join(base_path, 'data', 'FDA.csv')
        
        cdsco_df = load_cdsco_data(cdsco_path)
        fda_df = load_fda_data(fda_path)
        print(f"  Loaded {len(cdsco_df)} CDSCO drugs")
        print(f"  Loaded {len(fda_df)} FDA orphan drugs")
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return False
    
    # Step 2: Perform matching
    print("\nStep 2: Finding overlaps...")
    matcher = DrugMatcher(threshold=threshold)
    matches = matcher.find_overlaps(cdsco_df, fda_df)
    print(f"  Found {len(matches)} potential matches")
    
    if not matches:
        print("\nNo overlapping drugs found with current threshold.")
        print("Consider lowering the threshold or checking data formats.")
        return False
    
    # Step 3: Create report
    print("\nStep 3: Creating overlap report...")
    overlap_df = create_overlap_report(matches)
    
    if overlap_df is not None:
        print(f"  Report saved to: output/overlap.csv")
        
        # Step 4: Analyze results
        analyze_matches(overlap_df)
    
    return True


def main():
    """
    Main entry point for the pipeline.
    
    Can be run with custom threshold:
    python drug_overlap_pipeline.py 90
    """
    # Check if custom threshold provided
    threshold = 85
    if len(sys.argv) > 1:
        try:
            threshold = int(sys.argv[1])
            if threshold < 0 or threshold > 100:
                print("ERROR: Threshold must be between 0 and 100")
                sys.exit(1)
        except ValueError:
            print("ERROR: Threshold must be a number")
            sys.exit(1)
    
    # Run pipeline
    success = run_pipeline(threshold)
    
    if not success:
        sys.exit(1)
    
    print("\n=== PIPELINE COMPLETE ===")
    print("Review output/overlap.csv for results")
    print("\nTo adjust results:")
    print("  - Increase threshold for fewer, more accurate matches")
    print("  - Decrease threshold for more matches (may include false positives)")
    print(f"\nRerun with different threshold: python {sys.argv[0]} <threshold>")


if __name__ == "__main__":
    main()