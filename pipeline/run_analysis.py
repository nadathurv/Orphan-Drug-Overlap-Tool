#!/usr/bin/env python3
"""
Master script to run the drug overlap analysis.

This is the main entry point for the pipeline.
Run with: python pipeline/run_analysis.py [threshold]

Example:
    python pipeline/run_analysis.py        # Uses default 85% threshold
    python pipeline/run_analysis.py 90     # Uses 90% threshold
    python pipeline/run_analysis.py 80     # Uses 80% threshold for more matches
"""

import sys
import os

# Add current directory to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drug_overlap_pipeline import main

if __name__ == "__main__":
    main()