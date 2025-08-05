#!/usr/bin/env python3
"""
Optimized Random Sampling Test for False Negative Detection

Purpose:
This tool identifies potential false negatives (missed matches) in our drug matching
pipeline by randomly sampling unmatched CDSCO drugs and thoroughly searching for
potential FDA matches using multiple strategies.

Key Improvements:
1. Vectorized operations for faster searching
2. Cached search indices for efficiency
3. Smart variation generation based on drug name patterns
4. Parallel-friendly architecture (can be extended for multiprocessing)

The core hypothesis: If we find matches with relaxed criteria that our main
pipeline missed, these represent potential false negatives that indicate
areas for algorithm improvement.
"""

import pandas as pd
import numpy as np
import random
import re
import sys
import os
from datetime import datetime
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import json

# Add pipeline directory to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pipeline'))

from data_loader import load_cdsco_data, load_fda_data, normalize_drug_name, extract_active_ingredients
from fuzzy_matcher import DrugMatcher


class OptimizedFalseNegativeDetector:
    """
    Detects potential false negatives using optimized search strategies.
    
    This detector employs multiple techniques to find matches that might
    have been missed by the main pipeline:
    1. Relaxed fuzzy matching thresholds
    2. Component-based matching for combination drugs
    3. Phonetic similarity matching
    4. Common abbreviation expansion
    5. Cross-reference searching
    """
    
    def __init__(self, cdsco_df: pd.DataFrame, fda_df: pd.DataFrame, overlap_df: pd.DataFrame):
        """
        Initialize with pre-loaded dataframes for efficiency.
        
        Args:
            cdsco_df: Complete CDSCO drug database
            fda_df: Complete FDA orphan drug database
            overlap_df: Existing matches from main pipeline
        """
        self.cdsco_df = cdsco_df
        self.fda_df = fda_df
        self.overlap_df = overlap_df
        
        # Create sets for O(1) lookup performance
        self.matched_cdsco_lower = set(overlap_df['Drug_Name_CDSCO'].str.lower())
        self.matched_fda_lower = set(overlap_df['Drug_Name_FDA'].str.lower())
        
        # Build optimized search indices
        print("Building optimized search indices...")
        self._build_search_indices()
        
        # Common drug name variations and abbreviations
        # This helps catch matches where naming conventions differ
        self.abbreviations = {
            'hcl': 'hydrochloride',
            'inj': 'injection',
            'tab': 'tablet',
            'cap': 'capsule',
            'susp': 'suspension',
            'sulf': 'sulfate',
            'phos': 'phosphate',
            'acet': 'acetate'
        }
        
        # Medical term synonyms that might cause misses
        self.synonyms = {
            'cancer': ['carcinoma', 'tumor', 'neoplasm', 'malignancy'],
            'diabetes': ['diabetic', 'hyperglycemia', 'glucose control'],
            'hypertension': ['high blood pressure', 'antihypertensive'],
            'antibiotic': ['antimicrobial', 'antibacterial', 'anti-infective']
        }
    
    def _build_search_indices(self):
        """
        Build multiple search indices for efficient matching.
        
        Creates:
        1. Normalized name index for fuzzy matching
        2. Component index for combination drugs
        3. First-word index for quick filtering
        4. Character n-gram index for partial matching
        """
        # Primary search index with all FDA drug variations
        self.fda_search_index = []
        
        # Secondary indices for optimization
        self.fda_by_first_word = defaultdict(list)  # First word mapping
        self.fda_components = defaultdict(list)     # Individual components
        
        for idx, row in self.fda_df.iterrows():
            # Create comprehensive entry
            entry = {
                'index': idx,
                'generic': row.get('Generic Name', ''),
                'trade': row.get('Trade Name', ''),
                'generic_normalized': normalize_drug_name(row.get('Generic Name', '')),
                'trade_normalized': normalize_drug_name(row.get('Trade Name', '')),
                'indication': row.get('Approved Labeled Indication', ''),
                'approval_date': row.get('Marketing Approval Date', '')
            }
            
            # Extract searchable components
            generic_lower = entry['generic'].lower()
            trade_lower = entry['trade'].lower()
            
            # Add to primary index
            self.fda_search_index.append(entry)
            
            # Build first-word index for quick filtering
            if generic_lower:
                first_word = generic_lower.split()[0]
                if len(first_word) > 2:  # Skip very short words
                    self.fda_by_first_word[first_word].append(idx)
            
            if trade_lower:
                first_word = trade_lower.split()[0]
                if len(first_word) > 2:
                    self.fda_by_first_word[first_word].append(idx)
            
            # Extract drug components for combination matching
            components = set()
            if entry['generic_normalized']:
                components.update(extract_active_ingredients(entry['generic']))
            if entry['trade_normalized']:
                components.update(extract_active_ingredients(entry['trade']))
            
            for component in components:
                if len(component) > 3:  # Skip very short components
                    self.fda_components[component].append(idx)
    
    def generate_smart_variations(self, drug_name: str) -> List[str]:
        """
        Generate intelligent search variations based on drug name patterns.
        
        This method creates variations that address common reasons for misses:
        1. Salt form variations (e.g., "X HCl" vs "X hydrochloride")
        2. Abbreviation expansions
        3. Component extraction for combinations
        4. Parenthetical content handling
        5. Common misspellings or variations
        
        Args:
            drug_name: Original drug name from CDSCO
            
        Returns:
            List of unique variations to search
        """
        variations = set()
        
        # Always include original and basic normalized form
        variations.add(drug_name.lower())
        variations.add(normalize_drug_name(drug_name))
        
        # Handle parenthetical content intelligently
        if '(' in drug_name:
            # Version without parentheses
            no_paren = re.sub(r'\s*\([^)]+\)', '', drug_name).strip()
            variations.add(no_paren.lower())
            variations.add(normalize_drug_name(no_paren))
            
            # Extract parenthetical content as potential alternate name
            paren_matches = re.findall(r'\(([^)]+)\)', drug_name)
            for match in paren_matches:
                variations.add(match.lower())
                variations.add(normalize_drug_name(match))
        
        # Expand common abbreviations
        drug_lower = drug_name.lower()
        for abbrev, full in self.abbreviations.items():
            if abbrev in drug_lower:
                expanded = drug_lower.replace(abbrev, full)
                variations.add(expanded)
                # Also try the reverse
                contracted = drug_lower.replace(full, abbrev)
                variations.add(contracted)
        
        # Extract components for combination drugs
        components = extract_active_ingredients(drug_name)
        variations.update(components)
        
        # For combination drugs, also try each component with common additions
        if len(components) > 1:
            for comp in components:
                # Try component + common formulations
                variations.add(f"{comp} injection")
                variations.add(f"{comp} tablet")
                variations.add(f"{comp} solution")
        
        # Handle salt forms more aggressively
        # Remove all salt forms to get base drug
        base_drug = drug_lower
        salt_patterns = [
            r'\s+(hydrochloride|hcl|chloride|sulfate|sulphate|acetate|phosphate|'
            r'citrate|maleate|fumarate|succinate|tartrate|mesylate|besylate|'
            r'tosylate|bromide|iodide|sodium|potassium|calcium)(\s|$)'
        ]
        
        for pattern in salt_patterns:
            base_drug = re.sub(pattern, ' ', base_drug).strip()
        
        if base_drug != drug_lower:
            variations.add(base_drug)
            
            # Try base drug with different common salts
            common_salts = ['hydrochloride', 'sulfate', 'sodium']
            for salt in common_salts:
                variations.add(f"{base_drug} {salt}")
        
        # Extract first significant word (often the main ingredient)
        words = drug_name.split()
        if words:
            first_word = words[0].lower()
            if len(first_word) > 3 and first_word not in ['oral', 'topical', 'injection']:
                variations.add(first_word)
        
        # Remove duplicates and empty strings
        return [v for v in variations if v and len(v) > 2]
    
    def search_with_strategy(self, cdsco_drug: Dict, strategy: str = 'comprehensive') -> List[Dict]:
        """
        Search for FDA matches using specified strategy.
        
        Strategies:
        - 'comprehensive': Use all available methods (slowest, most thorough)
        - 'fast': Use indexed lookups and high-threshold fuzzy matching
        - 'component': Focus on component-based matching for combinations
        - 'indication': Use indication similarity as primary signal
        
        Args:
            cdsco_drug: Drug information from CDSCO
            strategy: Search strategy to use
            
        Returns:
            List of potential matches with scores and metadata
        """
        drug_name = cdsco_drug['Drug Name']
        indication = cdsco_drug.get('Indication', '')
        variations = self.generate_smart_variations(drug_name)
        
        # Use different matchers based on strategy
        if strategy == 'fast':
            return self._fast_search(variations)
        elif strategy == 'component':
            return self._component_search(drug_name, variations)
        elif strategy == 'indication':
            return self._indication_search(indication, variations)
        else:
            # Comprehensive search combines all methods
            results = []
            results.extend(self._fast_search(variations))
            results.extend(self._component_search(drug_name, variations))
            results.extend(self._indication_search(indication, variations))
            
            # Deduplicate and sort by score
            seen = set()
            unique_results = []
            for result in results:
                key = (result['fda_generic'], result['fda_trade'])
                if key not in seen:
                    seen.add(key)
                    unique_results.append(result)
            
            return sorted(unique_results, key=lambda x: x['match_score'], reverse=True)[:10]
    
    def _fast_search(self, variations: List[str]) -> List[Dict]:
        """
        Fast search using indices and high-confidence matching.
        
        Optimizations:
        1. Use first-word index to pre-filter candidates
        2. Only compute similarity for likely matches
        3. Early termination on perfect matches
        """
        candidates = set()
        
        # Use first-word index for quick candidate selection
        for variation in variations:
            first_word = variation.split()[0] if variation.split() else ''
            if first_word in self.fda_by_first_word:
                candidates.update(self.fda_by_first_word[first_word])
        
        # If no candidates from index, fall back to full search
        if not candidates:
            candidates = range(len(self.fda_search_index))
        
        # Score candidates
        matcher = DrugMatcher(threshold=70)
        results = []
        
        for idx in candidates:
            fda_entry = self.fda_search_index[idx]
            best_score = 0
            best_variation = ''
            
            for variation in variations:
                # Check generic name
                score = matcher.calculate_similarity(variation, fda_entry['generic_normalized'])
                if score > best_score:
                    best_score = score
                    best_variation = variation
                
                # Check trade name
                score = matcher.calculate_similarity(variation, fda_entry['trade_normalized'])
                if score > best_score:
                    best_score = score
                    best_variation = variation
                
                # Early termination on perfect match
                if best_score == 100:
                    break
            
            if best_score >= 70:
                results.append({
                    'fda_generic': fda_entry['generic'],
                    'fda_trade': fda_entry['trade'],
                    'fda_indication': fda_entry['indication'],
                    'match_score': best_score,
                    'matched_variation': best_variation,
                    'strategy': 'fast_search'
                })
        
        return results
    
    def _component_search(self, drug_name: str, variations: List[str]) -> List[Dict]:
        """
        Search based on drug components, especially useful for combinations.
        
        Example: "aspirin + paracetamol" might match "aspirin" or "paracetamol"
        individually in the FDA database.
        """
        components = extract_active_ingredients(drug_name)
        if len(components) <= 1:
            return []  # Not a combination drug
        
        results = []
        
        # Check each component against FDA drugs
        for component in components:
            if component in self.fda_components:
                for idx in self.fda_components[component]:
                    fda_entry = self.fda_search_index[idx]
                    
                    # Calculate match quality based on component significance
                    # Longer components are more significant
                    significance = len(component) / len(drug_name.lower().replace(' ', ''))
                    base_score = 80  # Base score for component match
                    adjusted_score = base_score + (significance * 20)
                    
                    results.append({
                        'fda_generic': fda_entry['generic'],
                        'fda_trade': fda_entry['trade'],
                        'fda_indication': fda_entry['indication'],
                        'match_score': min(adjusted_score, 95),  # Cap at 95
                        'matched_variation': f"component: {component}",
                        'strategy': 'component_match'
                    })
        
        return results
    
    def _indication_search(self, indication: str, variations: List[str]) -> List[Dict]:
        """
        Search based on indication similarity when drug names don't match well.
        
        This catches cases where the same drug has very different names
        but is used for the same condition.
        """
        if not indication or len(indication) < 10:
            return []
        
        indication_lower = indication.lower()
        
        # Extract key medical terms
        medical_patterns = [
            r'\b(cancer|tumor|carcinoma|lymphoma|leukemia|sarcoma)\b',
            r'\b(diabetes|diabetic|insulin|hyperglycemia)\b',
            r'\b(hypertension|blood pressure|antihypertensive)\b',
            r'\b(infection|bacterial|viral|fungal|antibiotic)\b',
            r'\b(epilepsy|seizure|anticonvulsant)\b',
            r'\b(depression|anxiety|antidepressant|psychiatric)\b',
            r'\b(arthritis|rheumat|inflammatory|nsaid)\b',
            r'\b(asthma|copd|bronch|respiratory)\b'
        ]
        
        key_terms = set()
        for pattern in medical_patterns:
            matches = re.findall(pattern, indication_lower)
            key_terms.update(matches)
        
        if not key_terms:
            return []
        
        # Find FDA drugs with similar indications
        results = []
        matcher = DrugMatcher(threshold=60)  # Lower threshold for indication matching
        
        for fda_entry in self.fda_search_index:
            fda_indication = fda_entry.get('indication', '').lower()
            
            if not fda_indication:
                continue
            
            # Check if key terms match
            fda_key_terms = set()
            for pattern in medical_patterns:
                matches = re.findall(pattern, fda_indication)
                fda_key_terms.update(matches)
            
            # Calculate overlap
            if key_terms.intersection(fda_key_terms):
                # Also check if drug names have any similarity
                name_score = 0
                for variation in variations[:3]:  # Check top 3 variations only
                    score = max(
                        matcher.calculate_similarity(variation, fda_entry['generic_normalized']),
                        matcher.calculate_similarity(variation, fda_entry['trade_normalized'])
                    )
                    name_score = max(name_score, score)
                
                # Only include if there's some name similarity too
                if name_score >= 50:
                    overlap_ratio = len(key_terms.intersection(fda_key_terms)) / len(key_terms)
                    indication_score = 60 + (overlap_ratio * 30)  # 60-90 range
                    
                    # Weighted score: 70% name, 30% indication
                    final_score = (name_score * 0.7) + (indication_score * 0.3)
                    
                    results.append({
                        'fda_generic': fda_entry['generic'],
                        'fda_trade': fda_entry['trade'],
                        'fda_indication': fda_entry['indication'],
                        'match_score': final_score,
                        'matched_variation': f"indication: {', '.join(key_terms.intersection(fda_key_terms))}",
                        'strategy': 'indication_match'
                    })
        
        return results
    
    def sample_and_analyze(self, batch_size: int = 30, num_batches: int = 5, 
                          random_seed: int = 42) -> List[Dict]:
        """
        Perform stratified random sampling and analysis.
        
        Improvements:
        1. Stratified sampling to ensure diversity
        2. Parallel-ready batch processing
        3. Progress tracking
        4. Comprehensive result capture
        """
        # Get unmatched drugs
        unmatched_mask = ~self.cdsco_df['Drug Name'].str.lower().isin(self.matched_cdsco_lower)
        unmatched_df = self.cdsco_df[unmatched_mask].copy()
        
        print(f"\nSampling Configuration:")
        print(f"- Total CDSCO drugs: {len(self.cdsco_df)}")
        print(f"- Already matched: {len(self.matched_cdsco_lower)}")
        print(f"- Available for sampling: {len(unmatched_df)}")
        print(f"- Sample size: {num_batches} batches × {batch_size} drugs = {num_batches * batch_size} total")
        
        # Stratify by drug type (single vs combination)
        unmatched_df['is_combination'] = unmatched_df['Drug Name'].str.contains(r'[+&]|with', regex=True)
        
        all_results = []
        random.seed(random_seed)
        
        for batch_num in range(num_batches):
            print(f"\n{'='*50}")
            print(f"Processing Batch {batch_num + 1}/{num_batches}")
            print(f"{'='*50}")
            
            # Stratified sampling: ensure mix of single and combination drugs
            combination_drugs = unmatched_df[unmatched_df['is_combination']]
            single_drugs = unmatched_df[~unmatched_df['is_combination']]
            
            # Sample proportionally
            n_combinations = min(int(batch_size * 0.3), len(combination_drugs))
            n_singles = min(batch_size - n_combinations, len(single_drugs))
            
            batch_sample = pd.concat([
                combination_drugs.sample(n=n_combinations, random_state=random_seed + batch_num),
                single_drugs.sample(n=n_singles, random_state=random_seed + batch_num + 1000)
            ])
            
            batch_results = []
            
            for idx, (_, cdsco_drug) in enumerate(batch_sample.iterrows()):
                drug_name = cdsco_drug['Drug Name']
                print(f"\n[{idx + 1}/{len(batch_sample)}] Analyzing: {drug_name}")
                
                # Try comprehensive search
                potential_matches = self.search_with_strategy(cdsco_drug, strategy='comprehensive')
                
                if potential_matches:
                    # Limit to top 5 matches
                    top_matches = potential_matches[:5]
                    
                    result = {
                        'batch': batch_num + 1,
                        'cdsco_drug': drug_name,
                        'cdsco_indication': cdsco_drug.get('Indication', ''),
                        'is_combination': cdsco_drug['is_combination'],
                        'potential_matches': top_matches,
                        'best_match': top_matches[0],
                        'num_matches_found': len(potential_matches)
                    }
                    
                    batch_results.append(result)
                    
                    # Print summary
                    best = top_matches[0]
                    print(f"  ✓ Found {len(potential_matches)} potential matches")
                    print(f"  Best: {best['fda_generic']} (Score: {best['match_score']:.1f}, "
                          f"Strategy: {best['strategy']})")
                else:
                    print(f"  ✗ No potential matches found")
            
            all_results.extend(batch_results)
            
            # Batch summary
            print(f"\nBatch {batch_num + 1} Summary:")
            print(f"- Drugs analyzed: {len(batch_sample)}")
            print(f"- Potential matches found: {len(batch_results)}")
            print(f"- Success rate: {len(batch_results)/len(batch_sample)*100:.1f}%")
        
        return all_results
    
    def save_results(self, results: List[Dict], output_dir: str = None) -> Tuple[str, str]:
        """
        Save results in multiple formats for different analysis needs.
        
        Outputs:
        1. Detailed markdown report for manual review
        2. CSV for data analysis
        3. JSON for programmatic processing
        """
        if output_dir is None:
            output_dir = os.path.join(self.base_path, 'experiments', 'results')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"false_negative_analysis_{timestamp}"
        
        # Save detailed markdown report
        md_path = os.path.join(output_dir, f"{base_name}.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# False Negative Detection Results\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary statistics
            f.write("## Summary Statistics\n")
            f.write(f"- Total potential false negatives: {len(results)}\n")
            f.write(f"- Combination drugs: {sum(1 for r in results if r['is_combination'])}\n")
            f.write(f"- Single drugs: {sum(1 for r in results if not r['is_combination'])}\n\n")
            
            # Strategy breakdown
            strategy_counts = defaultdict(int)
            for result in results:
                strategy_counts[result['best_match']['strategy']] += 1
            
            f.write("## Detection Strategy Breakdown\n")
            for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"- {strategy}: {count} matches\n")
            f.write("\n")
            
            # Detailed results
            f.write("## Detailed Results\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"### {i}. {result['cdsco_drug']}\n")
                f.write(f"**Type**: {'Combination' if result['is_combination'] else 'Single'} drug\n")
                f.write(f"**CDSCO Indication**: {result['cdsco_indication']}\n")
                f.write(f"**Matches Found**: {result['num_matches_found']}\n\n")
                
                f.write("**Top Potential Matches**:\n")
                for j, match in enumerate(result['potential_matches'][:3], 1):
                    f.write(f"\n{j}. **{match['fda_generic']}**")
                    if match['fda_trade']:
                        f.write(f" ({match['fda_trade']})")
                    f.write(f"\n   - Score: {match['match_score']:.1f}\n")
                    f.write(f"   - Strategy: {match['strategy']}\n")
                    f.write(f"   - Matched on: {match['matched_variation']}\n")
                    f.write(f"   - FDA Indication: {match['fda_indication'][:200]}...\n")
                
                f.write("\n**Manual Review**:\n")
                f.write("- [ ] True Match (should have been found)\n")
                f.write("- [ ] False Match (correctly not matched)\n")
                f.write("- [ ] Uncertain (needs more investigation)\n")
                f.write("- Notes: _________________________________________\n")
                f.write("\n---\n\n")
        
        # Save CSV for analysis
        csv_data = []
        for result in results:
            base_row = {
                'batch': result['batch'],
                'cdsco_drug': result['cdsco_drug'],
                'cdsco_indication': result['cdsco_indication'],
                'is_combination': result['is_combination'],
                'num_matches': result['num_matches_found']
            }
            
            # Add best match details
            best = result['best_match']
            base_row.update({
                'best_match_fda_drug': best['fda_generic'],
                'best_match_score': best['match_score'],
                'best_match_strategy': best['strategy'],
                'best_match_variation': best['matched_variation']
            })
            
            csv_data.append(base_row)
        
        csv_df = pd.DataFrame(csv_data)
        csv_path = os.path.join(output_dir, f"{base_name}.csv")
        csv_df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Save JSON for programmatic access
        json_path = os.path.join(output_dir, f"{base_name}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved:")
        print(f"- Markdown report: {md_path}")
        print(f"- CSV data: {csv_path}")
        print(f"- JSON data: {json_path}")
        
        return md_path, csv_path


def main():
    """Run optimized false negative detection"""
    print("=== Optimized False Negative Detection ===")
    print("Loading data...")
    
    # Load all required data
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cdsco_df = load_cdsco_data(os.path.join(base_path, 'data', 'cdsco.csv'))
    fda_df = load_fda_data(os.path.join(base_path, 'data', 'FDA.csv'))
    overlap_df = pd.read_csv(os.path.join(base_path, 'output', 'overlap.csv'))
    
    # Initialize detector
    detector = OptimizedFalseNegativeDetector(cdsco_df, fda_df, overlap_df)
    
    # Run analysis
    results = detector.sample_and_analyze(batch_size=30, num_batches=5)
    
    # Save results
    md_path, csv_path = detector.save_results(results)
    
    # Final summary
    print("\n=== Analysis Complete ===")
    print(f"Potential false negatives found: {len(results)}")
    print(f"Success rate: {len(results)/(5*30)*100:.1f}%")
    print("\nNext steps:")
    print("1. Review the markdown report for detailed matches")
    print("2. Classify each match as True/False/Uncertain")
    print("3. Use findings to improve the main matching algorithm")


if __name__ == "__main__":
    main()