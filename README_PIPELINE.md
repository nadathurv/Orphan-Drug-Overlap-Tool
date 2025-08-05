# Drug Overlap Analysis Pipeline

A robust tool for finding overlapping drugs between drug databases using fuzzy matching algorithms with enhanced accuracy optimizations.

## Overview

This pipeline identifies drugs that appear in both the CDSCO (Indian drug regulatory) database and FDA orphan drug database. It uses fuzzy string matching to handle variations in drug naming conventions and can identify matches even when drugs are part of combination formulations.

## Features

- **Advanced Fuzzy Matching**: 
  - Multiple algorithm approach (token sort, token set, partial ratio)
  - Salt form normalization (handles HCl, sulfate, etc.)
  - Parenthetical content handling
- **Combination Drug Handling**: 
  - Extracts and matches individual components
  - Handles multiple separator types (&, +, /, with)
  - Smart component normalization
- **Dynamic Indication Verification**: 
  - Adaptive thresholds based on drug name match confidence
  - Key term extraction and matching
  - Stricter verification for borderline matches
- **Duplicate Prevention**: Each FDA drug matched only once
- **Date Standardization**: Converts various date formats to MM/DD/YYYY
- **Configurable Threshold**: Adjust matching sensitivity as needed

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python pipeline/run_analysis.py
```

### With Custom Threshold
```bash
python pipeline/run_analysis.py 90  # More strict matching (fewer results)
python pipeline/run_analysis.py 80  # Less strict matching (more results)
```

## Output

The pipeline generates `overlap.csv` with the following columns:
- Drug_Name_CDSCO: Drug name from Indian database
- Drug_Name_FDA: Matched drug name from FDA database
- Indication_CDSCO: Medical use from CDSCO
- Indication_FDA: Medical use from FDA
- Approval_Date_CDSCO: Indian approval date (MM/DD/YYYY)
- Approval_Date_FDA: FDA approval date (MM/DD/YYYY)
- Match_Score: Similarity score (0-100)
- FDA_Generic_Name: Generic name from FDA
- FDA_Trade_Name: Trade/brand name from FDA

## Pipeline Components

1. **data_loader.py**: Handles data loading and preprocessing
   - Normalizes drug names
   - Extracts components from combination drugs
   - Cleans indication fields

2. **fuzzy_matcher.py**: Core matching logic
   - Token-based similarity scoring
   - Handles both generic and trade name matching
   - Indication-based verification for ambiguous matches

3. **date_formatter.py**: Date standardization utilities
   - Handles multiple date formats
   - Converts to consistent MM/DD/YYYY format

4. **drug_overlap_pipeline.py**: Main orchestration
   - Coordinates all components
   - Generates analysis report
   - Creates final CSV output

## Adjusting Results

- **Too many false positives?** Increase threshold (e.g., 90 or 95)
- **Missing known matches?** Decrease threshold (e.g., 80 or 75)
- **Review indication differences** for matches with scores 85-95%

## Adding New Databases

To add a new database for comparison:

1. Create a loader function in `data_loader.py` following the existing pattern
2. Ensure it returns normalized drug names and relevant fields
3. Modify `drug_overlap_pipeline.py` to include the new database in matching

## Accuracy Improvements

The pipeline includes several enhancements focused on accuracy:

1. **Drug Name Normalization**
   - Removes salt forms (HCl, sulfate, mesylate, etc.) for better matching
   - Handles parenthetical information intelligently
   - Normalizes spacing and punctuation

2. **Multi-Algorithm Matching**
   - Uses best score from multiple fuzzy matching algorithms
   - Token sort ratio for word order variations
   - Token set ratio for subset matching
   - Partial ratio for substring matching

3. **Dynamic Indication Verification**
   - Score ≥95%: 50% indication threshold (lenient)
   - Score 90-94%: 65% indication threshold (moderate)
   - Score 85-89%: 70% indication threshold (strict)
   - Key medical term matching for bonus points

4. **Quality Controls**
   - Prevents duplicate FDA drug matches
   - Skips invalid entries (too short names)
   - Conservative handling of missing indications

## Performance

- Improved accuracy: ~181 high-quality matches (vs 390 with basic matching)
- Processing time: < 1 minute for current dataset sizes
- Better precision with fewer false positives
- Scalable to larger databases with minimal modifications