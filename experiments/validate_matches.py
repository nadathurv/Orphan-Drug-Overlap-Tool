#!/usr/bin/env python3
"""
Optimized Match Validation Tool for Drug Overlap Analysis

Purpose:
This tool validates matches from the drug overlap pipeline to identify potential
issues and calculate accuracy metrics. It focuses on identifying TRUE false positives
(completely wrong drug matches) while preserving matches where the same drug has
different indications or formulations between databases.

Key Principle:
A drug match with 100% name similarity is ALWAYS high confidence, regardless of:
- Different indications (this is valuable data we want to capture)
- Different formulations (oral vs injectable)
- Combination vs single ingredient (if core drug matches)

The goal is to find the SAME DRUG across databases and observe differences,
not to find drugs with identical uses.
"""

import pandas as pd
import os
import sys
from datetime import datetime

# Add pipeline directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pipeline'))


class OptimizedMatchValidator:
    """
    Validates drug matches focusing on name accuracy rather than indication similarity.
    
    The validator uses a tiered approach:
    1. Perfect matches (100% score) - Always high confidence
    2. Strong matches (95-99%) - High confidence unless known false pattern
    3. Good matches (90-94%) - Medium confidence, needs review
    4. Borderline matches (85-89%) - Low confidence, likely needs manual review
    """
    
    def __init__(self, overlap_path='output/overlap.csv'):
        """Initialize with overlap results"""
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.overlap_df = pd.read_csv(os.path.join(self.base_path, overlap_path))
        
        # Pre-compute match statistics for efficiency
        self._compute_statistics()
        
        # Known false positive patterns based on domain knowledge
        # These are drug pairs that commonly match due to substring similarity
        # but are actually different drugs
        self.false_positive_patterns = {
            # Format: (substring1, substring2): reason
            ('urea', 'hydroxyurea'): 'Different drugs despite name similarity',
            ('salicylic acid', 'aminosalicylic acid'): 'Different compounds',
            ('potassium', 'bismuth'): 'Matched on common ion, different drugs',
            ('ethanol', 'ethanolamine'): 'Ethanol vs ethanolamine oleate',
            ('cipro', 'ofloxacin'): 'Different fluoroquinolones',
            ('day', 'daybue'): 'Coincidental name match'
        }
    
    def _compute_statistics(self):
        """Pre-compute statistics for efficient analysis"""
        self.total_matches = len(self.overlap_df)
        self.score_distribution = {
            '100': len(self.overlap_df[self.overlap_df['Match_Score'] == 100]),
            '95-99': len(self.overlap_df[(self.overlap_df['Match_Score'] >= 95) & 
                                        (self.overlap_df['Match_Score'] < 100)]),
            '90-94': len(self.overlap_df[(self.overlap_df['Match_Score'] >= 90) & 
                                        (self.overlap_df['Match_Score'] < 95)]),
            '85-89': len(self.overlap_df[(self.overlap_df['Match_Score'] >= 85) & 
                                        (self.overlap_df['Match_Score'] < 90)])
        }
    
    def classify_match(self, row):
        """
        Classify a match based on score and known patterns.
        
        Returns dict with:
        - confidence: 'high', 'medium', 'low'
        - flags: List of potential issues
        - recommendation: 'accept', 'review', 'reject'
        """
        score = row['Match_Score']
        cdsco_name = row['Drug_Name_CDSCO'].lower()
        fda_name = row['Drug_Name_FDA'].lower()
        
        classification = {
            'confidence': 'high',
            'flags': [],
            'recommendation': 'accept',
            'reason': ''
        }
        
        # Rule 1: Perfect matches (100%) are ALWAYS high confidence
        # This is the key principle - same drug name = valid match
        if score == 100:
            classification['confidence'] = 'high'
            classification['recommendation'] = 'accept'
            classification['reason'] = 'Perfect name match'
            
            # Still flag interesting differences for analysis
            if self._has_indication_mismatch(row):
                classification['flags'].append('Different indications (expected)')
            if self._is_combination_to_single(row):
                classification['flags'].append('Combination matched to component')
                
            return classification
        
        # Rule 2: Check for known false positive patterns
        false_pattern = self._check_false_patterns(cdsco_name, fda_name)
        if false_pattern:
            classification['confidence'] = 'low'
            classification['recommendation'] = 'reject'
            classification['reason'] = false_pattern
            classification['flags'].append('Known false positive pattern')
            return classification
        
        # Rule 3: High scores (95-99) are generally reliable
        if score >= 95:
            classification['confidence'] = 'high'
            classification['recommendation'] = 'accept'
            classification['reason'] = 'Very high similarity score'
            
            # Minor differences acceptable at this level
            if self._has_salt_form_difference(cdsco_name, fda_name):
                classification['flags'].append('Different salt forms')
                
        # Rule 4: Good scores (90-94) need context
        elif score >= 90:
            classification['confidence'] = 'medium'
            classification['recommendation'] = 'review'
            classification['reason'] = 'Good similarity, needs verification'
            
            # Check for specific issues
            if self._is_likely_different_drug(row):
                classification['confidence'] = 'low'
                classification['flags'].append('Possibly different drugs')
                
        # Rule 5: Borderline scores (85-89) are suspicious
        else:
            classification['confidence'] = 'low'
            classification['recommendation'] = 'review'
            classification['reason'] = 'Borderline similarity score'
            
            # These need careful review
            if len(cdsco_name.split()) != len(fda_name.split()):
                classification['flags'].append('Different word counts')
        
        return classification
    
    def _check_false_patterns(self, name1, name2):
        """Check if names match any known false positive patterns"""
        for (pattern1, pattern2), reason in self.false_positive_patterns.items():
            if ((pattern1 in name1 and pattern2 in name2) or 
                (pattern2 in name1 and pattern1 in name2)):
                return reason
        return None
    
    def _has_indication_mismatch(self, row):
        """
        Check if indications are significantly different.
        Note: This is NOT a reason to reject a match, just information.
        """
        cdsco_ind = str(row.get('Indication_CDSCO', '')).lower()
        fda_ind = str(row.get('Indication_FDA', '')).lower()
        
        if not cdsco_ind or not fda_ind:
            return False
        
        # Simple check: do they share any key medical terms?
        medical_terms = ['cancer', 'diabetes', 'hypertension', 'infection', 
                        'epilepsy', 'arthritis', 'asthma']
        
        cdsco_terms = set(term for term in medical_terms if term in cdsco_ind)
        fda_terms = set(term for term in medical_terms if term in fda_ind)
        
        # Different if no overlap in medical terms
        return len(cdsco_terms) > 0 and len(fda_terms) > 0 and not cdsco_terms.intersection(fda_terms)
    
    def _is_combination_to_single(self, row):
        """Check if a combination drug matched to single ingredient"""
        cdsco_name = row['Drug_Name_CDSCO']
        return any(sep in cdsco_name for sep in [' + ', ' & ', ' with '])
    
    def _has_salt_form_difference(self, name1, name2):
        """Check if drugs differ only in salt form"""
        # Common salt forms
        salts = ['hydrochloride', 'hcl', 'sulfate', 'acetate', 'sodium', 
                'potassium', 'calcium', 'mesylate', 'citrate']
        
        # Remove salt forms and compare
        base1 = name1
        base2 = name2
        for salt in salts:
            base1 = base1.replace(salt, '').strip()
            base2 = base2.replace(salt, '').strip()
        
        # If bases are similar, it's just a salt difference
        return base1 == base2 and name1 != name2
    
    def _is_likely_different_drug(self, row):
        """
        Heuristic to detect if drugs might be different despite name similarity.
        Used for medium-confidence matches only.
        """
        # If indications are completely different therapeutic areas
        if self._has_indication_mismatch(row):
            # And it's not a perfect match
            if row['Match_Score'] < 95:
                return True
        
        return False
    
    def generate_validation_report(self):
        """
        Generate comprehensive validation report with proper classifications.
        
        The report prioritizes:
        1. Low confidence matches that need review
        2. Interesting findings (same drug, different uses)
        3. Summary statistics
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.base_path, 'experiments', 'results',
                                   f'validation_report_{timestamp}.md')
        
        # Classify all matches
        classifications = []
        indication_differences = []  # Track interesting indication differences
        
        for idx, row in self.overlap_df.iterrows():
            classification = self.classify_match(row)
            classification['index'] = idx
            classification['row'] = row
            classifications.append(classification)
            
            # Track perfect matches with different indications (interesting findings)
            if (classification['confidence'] == 'high' and 
                'Different indications (expected)' in classification['flags']):
                indication_differences.append(row)
        
        # Group by confidence
        high_conf = [c for c in classifications if c['confidence'] == 'high']
        medium_conf = [c for c in classifications if c['confidence'] == 'medium']
        low_conf = [c for c in classifications if c['confidence'] == 'low']
        
        # Write report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Drug Match Validation Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n")
            f.write(f"- Total matches analyzed: {self.total_matches}\n")
            f.write(f"- High confidence: {len(high_conf)} ({len(high_conf)/self.total_matches*100:.1f}%)\n")
            f.write(f"- Medium confidence: {len(medium_conf)} ({len(medium_conf)/self.total_matches*100:.1f}%)\n")
            f.write(f"- Low confidence: {len(low_conf)} ({len(low_conf)/self.total_matches*100:.1f}%)\n\n")
            
            # Score distribution
            f.write("## Score Distribution\n")
            for range_name, count in self.score_distribution.items():
                f.write(f"- Score {range_name}: {count} matches\n")
            f.write("\n")
            
            # Priority Review Section
            f.write("## Priority Review: Low Confidence Matches\n")
            f.write("These matches likely contain errors and need manual review:\n\n")
            
            for cls in low_conf:
                row = cls['row']
                f.write(f"### {cls['index'] + 1}. {row['Drug_Name_CDSCO']} ↔ {row['Drug_Name_FDA']}\n")
                f.write(f"- **Score**: {row['Match_Score']:.1f}\n")
                f.write(f"- **Reason**: {cls['reason']}\n")
                f.write(f"- **Flags**: {', '.join(cls['flags']) if cls['flags'] else 'None'}\n")
                cdsco_ind = str(row.get('Indication_CDSCO', 'N/A'))
                fda_ind = str(row.get('Indication_FDA', 'N/A'))
                f.write(f"- **CDSCO Indication**: {cdsco_ind[:150]}{'...' if len(cdsco_ind) > 150 else ''}\n")
                f.write(f"- **FDA Indication**: {fda_ind[:150]}{'...' if len(fda_ind) > 150 else ''}\n")
                f.write(f"- **Action**: [ ] Accept [ ] Reject [ ] Needs Investigation\n\n")
            
            # Interesting Findings Section
            if indication_differences:
                f.write("## Interesting Findings: Same Drug, Different Indications\n")
                f.write("These are valid matches showing how the same drug is used differently:\n\n")
                
                for row in indication_differences[:10]:  # Show first 10
                    f.write(f"### {row['Drug_Name_CDSCO']}\n")
                    cdsco_use = str(row.get('Indication_CDSCO', 'N/A'))
                    fda_use = str(row.get('Indication_FDA', 'N/A'))
                    f.write(f"- **CDSCO Use**: {cdsco_use[:200]}{'...' if len(cdsco_use) > 200 else ''}\n")
                    f.write(f"- **FDA Use**: {fda_use[:200]}{'...' if len(fda_use) > 200 else ''}\n\n")
                
                if len(indication_differences) > 10:
                    f.write(f"*...and {len(indication_differences) - 10} more examples*\n\n")
            
            # Medium confidence section (brief)
            f.write("## Medium Confidence Matches (Brief Summary)\n")
            f.write(f"Total: {len(medium_conf)} matches needing verification\n\n")
            
            # Show a few examples
            for cls in medium_conf[:5]:
                row = cls['row']
                f.write(f"- {row['Drug_Name_CDSCO']} ↔ {row['Drug_Name_FDA']} (Score: {row['Match_Score']:.1f})\n")
            
            if len(medium_conf) > 5:
                f.write(f"- *...and {len(medium_conf) - 5} more*\n")
        
        # Generate CSV for data analysis
        csv_data = []
        for cls in classifications:
            row = cls['row']
            csv_data.append({
                'cdsco_drug': row['Drug_Name_CDSCO'],
                'fda_drug': row['Drug_Name_FDA'],
                'match_score': row['Match_Score'],
                'confidence': cls['confidence'],
                'recommendation': cls['recommendation'],
                'reason': cls['reason'],
                'flags': '|'.join(cls['flags']) if cls['flags'] else '',
                'cdsco_indication': row.get('Indication_CDSCO', ''),
                'fda_indication': row.get('Indication_FDA', '')
            })
        
        csv_df = pd.DataFrame(csv_data)
        csv_path = report_path.replace('.md', '.csv')
        csv_df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"\nValidation complete!")
        print(f"Report saved to: {report_path}")
        print(f"CSV saved to: {csv_path}")
        
        # Return summary statistics
        return {
            'total': self.total_matches,
            'high_confidence': len(high_conf),
            'medium_confidence': len(medium_conf),
            'low_confidence': len(low_conf),
            'estimated_false_positives': len([c for c in low_conf if c['recommendation'] == 'reject']),
            'indication_differences': len(indication_differences)
        }


def main():
    """Run optimized validation analysis"""
    print("Starting optimized match validation...")
    
    validator = OptimizedMatchValidator()
    stats = validator.generate_validation_report()
    
    print("\n=== Validation Summary ===")
    print(f"Total matches: {stats['total']}")
    print(f"High confidence: {stats['high_confidence']} ({stats['high_confidence']/stats['total']*100:.1f}%)")
    print(f"Estimated false positives: {stats['estimated_false_positives']}")
    print(f"Same drug, different indications: {stats['indication_differences']}")
    
    print("\nNext steps:")
    print("1. Review low confidence matches in the report")
    print("2. Verify medium confidence matches if needed")
    print("3. Analyze indication differences for research insights")


if __name__ == "__main__":
    main()