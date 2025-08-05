"""
Data loading and preprocessing module for drug overlap analysis.

This module handles:
- Loading CSV files with proper encoding
- Basic data cleaning (trimming whitespace, handling nulls)
- Normalizing drug names for better matching
"""

import pandas as pd
import re


def load_cdsco_data(filepath='data/cdsco.csv'):
    """
    Load CDSCO data and perform basic preprocessing.
    
    The CDSCO dataset contains Indian drug approvals with columns:
    - Drug Name: Can contain single drugs or combinations
    - Indication: Medical use of the drug
    - Date of Approval: Various date formats
    """
    df = pd.read_csv(filepath, encoding='utf-8')
    
    # Clean drug names: strip whitespace and convert to lowercase for matching
    df['Drug Name'] = df['Drug Name'].astype(str).str.strip()
    # Apply advanced normalization for better matching accuracy
    df['Drug Name_normalized'] = df['Drug Name'].apply(normalize_drug_name)
    
    # Clean indication field
    df['Indication'] = df['Indication'].fillna('').astype(str).str.strip()
    
    # Keep original date as string for now - will be formatted later
    df['Date of Approval'] = df['Date of Approval'].astype(str).str.strip()
    
    return df


def load_fda_data(filepath='data/FDA.csv'):
    """
    Load FDA orphan drug data and perform basic preprocessing.
    
    FDA dataset contains orphan drug designations with key columns:
    - Generic Name: Official drug name
    - Trade Name: Brand/commercial name
    - Approved Labeled Indication: FDA-approved use
    - Marketing Approval Date: When drug was approved
    """
    df = pd.read_csv(filepath, encoding='utf-8')
    
    # Clean both generic and trade names
    df['Generic Name'] = df['Generic Name'].fillna('').astype(str).str.strip()
    df['Trade Name'] = df['Trade Name'].fillna('').astype(str).str.strip()
    
    # Create normalized versions for matching with advanced normalization
    df['Generic Name_normalized'] = df['Generic Name'].apply(normalize_drug_name)
    df['Trade Name_normalized'] = df['Trade Name'].apply(normalize_drug_name)
    
    # Clean indication - use the approved labeled indication
    df['Approved Labeled Indication'] = df['Approved Labeled Indication'].fillna('').astype(str).str.strip()
    
    # Keep date as string for consistent formatting later
    df['Marketing Approval Date'] = df['Marketing Approval Date'].fillna('').astype(str).str.strip()
    
    return df


def normalize_drug_name(drug_name):
    """
    Normalize a drug name for better matching accuracy.
    
    Handles:
    - Salt forms (hydrochloride, sulfate, etc.)
    - Parenthetical information
    - Common abbreviations
    - Extra whitespace
    """
    if not drug_name:
        return ''
    
    normalized = drug_name.lower().strip()
    
    # Remove content in parentheses but keep it available for secondary matching
    # e.g., "drug (salt form)" -> "drug"
    paren_content = re.findall(r'\([^)]+\)', normalized)
    normalized_no_paren = re.sub(r'\s*\([^)]+\)', '', normalized).strip()
    
    # Remove common salt forms and chemical suffixes
    # These often differ between databases but refer to the same drug
    salt_forms = [
        r'\s+hydrochloride\b', r'\s+hcl\b', r'\s+chloride\b',
        r'\s+sulfate\b', r'\s+sulphate\b', r'\s+acetate\b',
        r'\s+phosphate\b', r'\s+citrate\b', r'\s+maleate\b',
        r'\s+fumarate\b', r'\s+succinate\b', r'\s+tartrate\b',
        r'\s+mesylate\b', r'\s+besylate\b', r'\s+tosylate\b',
        r'\s+bromide\b', r'\s+iodide\b', r'\s+sodium\b',
        r'\s+potassium\b', r'\s+calcium\b', r'\s+dihydrate\b',
        r'\s+monohydrate\b', r'\s+anhydrous\b'
    ]
    
    for salt in salt_forms:
        normalized_no_paren = re.sub(salt, '', normalized_no_paren, flags=re.IGNORECASE)
    
    # Normalize common variations
    normalized_no_paren = re.sub(r'\s+', ' ', normalized_no_paren)  # Multiple spaces to single
    normalized_no_paren = normalized_no_paren.strip()
    
    return normalized_no_paren


def extract_active_ingredients(drug_name):
    """
    Extract individual active ingredients from combination drugs.
    
    Combination drugs in CDSCO often use separators like:
    - & (ampersand)
    - + (plus)
    - , (comma)
    - with
    - / (slash)
    
    This helps match individual components against FDA database.
    """
    # Define separators commonly used in drug combinations
    separators = [' & ', ' + ', ', ', ' with ', ' and ', '/']
    
    # Normalize the input
    drug_lower = drug_name.lower()
    
    # Split by various separators
    components = [drug_lower]
    for sep in separators:
        new_components = []
        for comp in components:
            new_components.extend(comp.split(sep))
        components = new_components
    
    # Clean each component
    components = [comp.strip() for comp in components if comp.strip()]
    
    # Normalize each component if it's a combination
    if len(components) > 1:
        normalized_components = []
        for comp in components:
            # Remove dosage forms and strengths
            comp = re.sub(r'\b\d+\s*mg\b', '', comp)
            comp = re.sub(r'\b\d+\s*g\b', '', comp)
            comp = re.sub(r'\b\d+\s*mcg\b', '', comp)
            comp = re.sub(r'\b\d+\s*ml\b', '', comp)
            comp = re.sub(r'\b\d+\s*%\b', '', comp)
            comp = re.sub(r'\btablet[s]?\b', '', comp)
            comp = re.sub(r'\bcapsule[s]?\b', '', comp)
            comp = re.sub(r'\binjection[s]?\b', '', comp)
            comp = re.sub(r'\bsolution[s]?\b', '', comp)
            
            # Apply drug name normalization to each component
            normalized = normalize_drug_name(comp.strip())
            if normalized:
                normalized_components.append(normalized)
        
        return normalized_components
    
    # For single drugs, return the normalized form
    return [normalize_drug_name(drug_lower)]


if __name__ == "__main__":
    # Test loading
    print("Testing data loading...")
    cdsco_df = load_cdsco_data()
    fda_df = load_fda_data()
    
    print(f"Loaded {len(cdsco_df)} CDSCO records")
    print(f"Loaded {len(fda_df)} FDA records")
    
    # Test combination drug extraction
    test_drugs = [
        "atenolol & chlorthalidone",
        "amylobarbitone & trifluoperazine dihydrochloride",
        "single drug name"
    ]
    
    print("\nTesting drug component extraction:")
    for drug in test_drugs:
        components = extract_active_ingredients(drug)
        print(f"{drug} -> {components}")