"""
Date formatting utilities for standardizing approval dates.

Handles various date formats found in drug databases:
- FDA: YYYY-MM-DD format
- CDSCO: Multiple formats (D/M/YY, DD/MM/YYYY, etc.)
"""

import re
from datetime import datetime
import pandas as pd


def parse_date(date_string):
    """
    Parse a date string and return a datetime object.
    
    Handles multiple formats:
    - YYYY-MM-DD (FDA format)
    - D/M/YY or DD/MM/YY (CDSCO common formats)
    - DD/MM/YYYY
    - Various separators (/, -, .)
    """
    if not date_string or pd.isna(date_string) or str(date_string).strip() == '':
        return None
    
    date_str = str(date_string).strip()
    
    # Common date formats to try
    formats = [
        '%Y-%m-%d',      # 2017-09-05 (FDA format)
        '%d/%m/%y',      # 3/7/11 (CDSCO format)
        '%d/%m/%Y',      # 03/07/2011
        '%m/%d/%y',      # 7/3/11 (US format variant)
        '%m/%d/%Y',      # 07/03/2011
        '%d-%m-%Y',      # 03-07-2011
        '%d.%m.%Y',      # 03.07.2011
        '%Y/%m/%d',      # 2011/07/03
        '%d-%b-%y',      # 03-Jul-11
        '%d-%b-%Y',      # 03-Jul-2011
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # If no format worked, try some custom parsing
    # Handle single digit dates (common in CDSCO)
    if '/' in date_str:
        parts = date_str.split('/')
        if len(parts) == 3:
            try:
                # Assume D/M/YY format for CDSCO
                day = int(parts[0])
                month = int(parts[1])
                year = int(parts[2])
                
                # Handle 2-digit years
                if year < 100:
                    # Assume 70-99 is 1970-1999, 00-69 is 2000-2069
                    if year >= 70:
                        year += 1900
                    else:
                        year += 2000
                
                return datetime(year, month, day)
            except (ValueError, IndexError):
                pass
    
    return None


def format_date_output(date_obj):
    """
    Format a datetime object to MM/DD/YYYY or MM/YYYY format.
    
    If day is 1 and it seems like a placeholder, use MM/YYYY format.
    """
    if not date_obj:
        return ''
    
    # If day is 1, it might be a placeholder for unknown day
    # Check if multiple entries have day=1 for the same month/year
    if date_obj.day == 1:
        # For now, keep full date but this could be refined
        return date_obj.strftime('%m/%d/%Y')
    
    return date_obj.strftime('%m/%d/%Y')


def standardize_dates(dates_series):
    """
    Standardize a pandas Series of date strings.
    
    Returns a Series with standardized date strings.
    """
    standardized = []
    
    for date_str in dates_series:
        parsed = parse_date(date_str)
        formatted = format_date_output(parsed)
        standardized.append(formatted)
    
    return pd.Series(standardized)


if __name__ == "__main__":
    # Test date parsing
    test_dates = [
        '3/7/11',           # CDSCO format
        '2017-09-05',       # FDA format
        '1/1/70',           # Old CDSCO date
        '12/31/2020',       # Full year
        '',                 # Empty
        '15-Jan-21',        # Month name format
        '2021/06/15',       # Alternative format
    ]
    
    print("Testing date parsing and formatting:")
    for date_str in test_dates:
        parsed = parse_date(date_str)
        formatted = format_date_output(parsed)
        print(f"{date_str:15} -> {formatted}")
    
    # Test with pandas Series
    print("\nTesting with pandas Series:")
    dates_series = pd.Series(test_dates)
    standardized = standardize_dates(dates_series)
    for orig, std in zip(test_dates, standardized):
        print(f"{orig:15} -> {std}")