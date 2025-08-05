"""
Enhanced fuzzy matching module for drug name comparison.

This module implements sophisticated matching logic to handle:
- Salt/formulation differences (e.g., drug HCl vs drug sulfate)
- Combination drugs vs individual components
- Partial component matches in multi-drug combinations
- Species/source variations (e.g., human vs salmon calcitonin)
"""

from rapidfuzz import fuzz, process
import pandas as pd
from data_loader import extract_active_ingredients, normalize_drug_name
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MatchResult:
    """Structured result for drug matching with detailed metadata"""
    fda_drug: Dict
    score: float
    match_type: str  # 'exact', 'salt_variant', 'combination', 'partial', 'generic', 'trade'
    matched_components: List[str] = None
    total_components: int = 0
    component_coverage: float = 0.0
    confidence_reason: str = ""


class EnhancedDrugMatcher:
    """
    Advanced drug matching with support for complex pharmaceutical variations.
    
    Key improvements:
    1. Salt form intelligence - recognizes same drug with different salts
    2. Combination drug handling - matches fixed combos to individual drugs
    3. Partial match tracking - reports incomplete combination matches
    4. Species variation handling - matches human vs animal-derived drugs
    """
    
    def __init__(self, threshold=85):
        """
        Initialize matcher with similarity threshold and variation mappings.
        
        Args:
            threshold: Minimum similarity score (0-100) for a match
        """
        self.threshold = threshold
        self._normalization_cache = {}  # Cache normalized drug names
        
        # Salt form variations that represent the same active ingredient
        # These should not penalize match scores
        self.salt_equivalents = {
            'hydrochloride': ['hcl', 'chloride', 'hydrochlor'],
            'sulfate': ['sulphate', 'sulf'],
            'acetate': ['acet'],
            'phosphate': ['phos'],
            'citrate': ['citric acid salt'],
            'maleate': ['mal'],
            'sodium': ['na', 'disodium', 'monosodium'],
            'potassium': ['k', 'dipotassium'],
            'calcium': ['ca'],
        }
        
        # Species/source variations for biological drugs
        self.species_variations = {
            'human': ['human', 'recombinant human', 'rh', 'r-hu', 'hr'],
            'salmon': ['salmon', 'salcatonin', 'calcitonin salmon'],
            'porcine': ['porcine', 'pig', 'pork'],
            'bovine': ['bovine', 'cow', 'beef'],
            'synthetic': ['synthetic', 'artificial'],
        }
        
        # Common formulation terms to ignore when comparing
        self.formulation_terms = [
            'injection', 'tablet', 'capsule', 'solution', 'suspension',
            'cream', 'ointment', 'gel', 'syrup', 'oral', 'topical',
            'intravenous', 'iv', 'im', 'subcutaneous', 'sc'
        ]
    
    def normalize_for_salt_comparison(self, drug_name: str) -> str:
        """
        Normalize drug name specifically for salt form comparison.
        
        This creates a canonical form where different salt forms of the
        same drug will match with high similarity.
        
        Example:
        - "atenolol hydrochloride" -> "atenolol"
        - "atenolol HCl" -> "atenolol"
        - "atenolol sulfate" -> "atenolol"
        """
        # Use cache for performance
        if drug_name in self._normalization_cache:
            return self._normalization_cache[drug_name]
        
        normalized = drug_name.lower().strip()
        
        # Remove formulation terms
        for term in self.formulation_terms:
            escaped_term = re.escape(term)
            normalized = re.sub(rf'\b{escaped_term}\b', '', normalized)
        
        # Remove salt forms - combined pattern for efficiency
        salt_pattern = r'\s+(hydrochloride|hcl|chloride|sulfate|sulphate|acetate|acet|phosphate|phos|citrate|citric acid|maleate|mal|fumarate|succinate|tartrate|mesylate|besylate|tosylate|bromide|iodide|sodium|potassium|calcium|magnesium|monohydrate|dihydrate|trihydrate|anhydrous)'
        normalized = re.sub(salt_pattern, '', normalized, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        normalized = ' '.join(normalized.split()).strip()
        
        # Cache the result
        self._normalization_cache[drug_name] = normalized
        
        return normalized
    
    def extract_base_drug_name(self, drug_name: str) -> str:
        """
        Extract the base drug name without species/source modifiers.
        
        Example:
        - "calcitonin salmon" -> "calcitonin"
        - "insulin human" -> "insulin"
        """
        base_name = self.normalize_for_salt_comparison(drug_name)
        
        # Remove species modifiers
        for species_list in self.species_variations.values():
            for species in species_list:
                # Escape special regex characters
                escaped_species = re.escape(species)
                base_name = re.sub(rf'\b{escaped_species}\b', '', base_name, flags=re.IGNORECASE)
        
        return ' '.join(base_name.split()).strip()
    
    def calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity with multi-algorithm approach.
        
        Optimized for performance while maintaining accuracy.
        """
        if not str1 or not str2:
            return 0
        
        str1_lower = str1.lower().strip()
        str2_lower = str2.lower().strip()
        
        # First check: exact match
        if str1_lower == str2_lower:
            return 100
        
        # For performance, only do expensive normalizations if strings are somewhat similar
        quick_score = fuzz.ratio(str1_lower, str2_lower)
        if quick_score < 70:  # Unlikely to match
            return quick_score
        
        # Check if this might be a salt variant (only if quick score is promising)
        if quick_score >= 80:
            base1 = self.normalize_for_salt_comparison(str1)
            base2 = self.normalize_for_salt_comparison(str2)
            
            if base1 == base2 and base1:  # Same drug, different salt
                return 98  # Very high score but not perfect
        
        # Standard fuzzy matching
        scores = [
            quick_score,
            fuzz.token_sort_ratio(str1_lower, str2_lower),
            fuzz.token_set_ratio(str1_lower, str2_lower),
        ]
        
        return max(scores)
    
    def match_combination_drug_enhanced(self, cdsco_drug: str, fda_drugs_list: List[Dict]) -> Optional[MatchResult]:
        """
        Enhanced combination drug matching with support for:
        1. FDA combo matching individual CDSCO drugs
        2. CDSCO combo matching individual FDA drugs  
        3. Partial component matches with coverage tracking
        """
        # Extract components from CDSCO drug
        cdsco_components = extract_active_ingredients(cdsco_drug)
        
        # First, check if CDSCO drug (possibly combo) matches any FDA drug directly
        direct_match = self.match_single_drug(cdsco_drug, fda_drugs_list)
        if direct_match and direct_match.score >= self.threshold:
            return direct_match
        
        # If CDSCO is not a combination, check if it matches a component of FDA combos
        if len(cdsco_components) == 1:
            cdsco_base = self.normalize_for_salt_comparison(cdsco_components[0])
            
            for fda_drug in fda_drugs_list:
                # Check if FDA drug is a combination
                fda_generic_components = extract_active_ingredients(fda_drug.get('generic', ''))
                fda_trade_components = extract_active_ingredients(fda_drug.get('trade', ''))
                
                # Check generic name components
                if len(fda_generic_components) > 1:
                    for fda_comp in fda_generic_components:
                        fda_comp_base = self.normalize_for_salt_comparison(fda_comp)
                        if self.calculate_similarity(cdsco_base, fda_comp_base) >= 90:
                            return MatchResult(
                                fda_drug=fda_drug,
                                score=95,  # High score for component match
                                match_type='combination_component',
                                matched_components=[cdsco_drug],
                                total_components=len(fda_generic_components),
                                component_coverage=1.0 / len(fda_generic_components),
                                confidence_reason=f"Matches component of FDA combination: {fda_drug['generic']}"
                            )
        
        # If CDSCO is a combination, try matching components
        if len(cdsco_components) > 1:
            best_match = None
            best_coverage = 0
            
            for fda_drug in fda_drugs_list:
                matched_components = []
                
                # Try matching each CDSCO component against FDA drug
                for component in cdsco_components:
                    comp_score = max(
                        self.calculate_similarity(component, fda_drug.get('generic_normalized', '')),
                        self.calculate_similarity(component, fda_drug.get('trade_normalized', ''))
                    )
                    
                    if comp_score >= 85:
                        matched_components.append(component)
                
                # Calculate coverage
                coverage = len(matched_components) / len(cdsco_components)
                
                # Accept partial matches if coverage is significant
                if coverage >= 0.5 and coverage > best_coverage:  # At least 50% components match
                    score = 80 + (coverage * 15)  # Score 80-95 based on coverage
                    best_match = MatchResult(
                        fda_drug=fda_drug,
                        score=score,
                        match_type='partial_combination',
                        matched_components=matched_components,
                        total_components=len(cdsco_components),
                        component_coverage=coverage,
                        confidence_reason=f"Partial match: {len(matched_components)}/{len(cdsco_components)} components"
                    )
                    best_coverage = coverage
            
            if best_match:
                return best_match
        
        return None
    
    def match_single_drug(self, cdsco_drug: str, fda_drugs_list: List[Dict]) -> Optional[MatchResult]:
        """
        Enhanced single drug matching with salt form intelligence.
        """
        if not cdsco_drug:
            return None
        
        best_match = None
        best_score = 0
        match_type = None
        
        # Normalize for comparison
        cdsco_normalized = normalize_drug_name(cdsco_drug)
        cdsco_salt_normalized = self.normalize_for_salt_comparison(cdsco_drug)
        
        for fda_drug in fda_drugs_list:
            # Try multiple matching strategies
            
            # 1. Standard normalized matching
            generic_score = self.calculate_similarity(cdsco_normalized, fda_drug.get('generic_normalized', ''))
            trade_score = self.calculate_similarity(cdsco_normalized, fda_drug.get('trade_normalized', ''))
            
            # 2. Salt-normalized matching (for salt variants)
            fda_generic_salt_norm = self.normalize_for_salt_comparison(fda_drug.get('generic', ''))
            fda_trade_salt_norm = self.normalize_for_salt_comparison(fda_drug.get('trade', ''))
            
            salt_generic_score = self.calculate_similarity(cdsco_salt_normalized, fda_generic_salt_norm)
            salt_trade_score = self.calculate_similarity(cdsco_salt_normalized, fda_trade_salt_norm)
            
            # Take best score and track match type
            scores = [
                (generic_score, 'generic'),
                (trade_score, 'trade'),
                (salt_generic_score, 'salt_variant' if salt_generic_score > 90 else 'generic'),
                (salt_trade_score, 'salt_variant' if salt_trade_score > 90 else 'trade')
            ]
            
            max_score, score_type = max(scores, key=lambda x: x[0])
            
            if max_score > best_score:
                best_score = max_score
                match_type = score_type
                best_match = fda_drug
        
        if best_score >= self.threshold and best_match:
            confidence_reason = ""
            if match_type == 'salt_variant':
                confidence_reason = "Salt form variant of the same drug"
            elif best_score == 100:
                confidence_reason = "Exact name match"
            elif best_score >= 95:
                confidence_reason = "Very high similarity"
            
            return MatchResult(
                fda_drug=best_match,
                score=best_score,
                match_type=match_type,
                confidence_reason=confidence_reason
            )
        
        return None
    
    def find_overlaps(self, cdsco_df: pd.DataFrame, fda_df: pd.DataFrame) -> List[Dict]:
        """
        Find all overlapping drugs with enhanced matching logic.
        """
        # Prepare FDA data for efficient searching
        print("  Preparing FDA data index...")
        fda_drugs_list = []
        for _, row in fda_df.iterrows():
            fda_drugs_list.append({
                'generic_normalized': row.get('Generic Name_normalized', ''),
                'trade_normalized': row.get('Trade Name_normalized', ''),
                'generic': row.get('Generic Name', ''),
                'trade': row.get('Trade Name', ''),
                'indication': row.get('Approved Labeled Indication', ''),
                'approval_date': row.get('Marketing Approval Date', ''),
                'index': row.name
            })
        
        matches = []
        seen_fda_drugs = set()  # Prevent duplicate FDA matches
        
        print(f"  Processing {len(cdsco_df)} CDSCO drugs...")
        # Process each CDSCO drug
        for idx, cdsco_row in cdsco_df.iterrows():
            if idx % 500 == 0:
                print(f"    Progress: {idx}/{len(cdsco_df)} drugs processed...")
            
            cdsco_drug = cdsco_row.get('Drug Name', '')
            cdsco_indication = cdsco_row.get('Indication', '')
            
            # Skip very short drug names
            if len(cdsco_drug) < 3:
                continue
            
            # Try enhanced combination matching first
            match_result = self.match_combination_drug_enhanced(cdsco_drug, fda_drugs_list)
            
            # If no combination match, try single drug matching
            if not match_result:
                match_result = self.match_single_drug(cdsco_drug, fda_drugs_list)
            
            if match_result and match_result.score >= self.threshold:
                # Always verify by indication with dynamic thresholds
                if not self.verify_match_by_indication(cdsco_indication, 
                                                      match_result.fda_drug['indication'], 
                                                      match_result.score):
                    continue
                
                # Prevent duplicate FDA drugs
                fda_drug_key = (match_result.fda_drug['generic'], match_result.fda_drug['trade'])
                if fda_drug_key not in seen_fda_drugs:
                    seen_fda_drugs.add(fda_drug_key)
                    
                    # Convert MatchResult to dictionary for compatibility
                    match_dict = {
                        'cdsco_drug': cdsco_drug,
                        'cdsco_indication': cdsco_indication,
                        'cdsco_approval_date': cdsco_row.get('Date of Approval', ''),
                        'fda_drug': match_result.fda_drug['generic'] if match_result.match_type in ['generic', 'salt_variant'] else match_result.fda_drug['trade'],
                        'fda_generic': match_result.fda_drug['generic'],
                        'fda_trade': match_result.fda_drug['trade'],
                        'fda_indication': match_result.fda_drug['indication'],
                        'fda_approval_date': match_result.fda_drug['approval_date'],
                        'match_score': match_result.score,
                        'match_type': match_result.match_type,
                        'confidence_reason': match_result.confidence_reason
                    }
                    
                    # Add component match details if applicable
                    if match_result.matched_components:
                        match_dict['matched_components'] = ', '.join(match_result.matched_components)
                        match_dict['component_coverage'] = f"{match_result.component_coverage:.0%}"
                    
                    matches.append(match_dict)
        
        return matches
    
    def verify_match_by_indication(self, cdsco_indication: str, fda_indication: str, drug_match_score: float) -> bool:
        """
        Verify match by comparing indications with dynamic threshold.
        
        Updated logic: More lenient for high-confidence drug matches (salt variants, etc.)
        """
        if not cdsco_indication or not fda_indication:
            # Can't verify without indications - be conservative
            # Only allow if drug match is very strong
            return drug_match_score >= 95
        
        # Dynamic threshold based on drug name match confidence
        if drug_match_score >= 98:  # Salt variants, exact matches
            indication_threshold = 40  # Very lenient - different indications expected
        elif drug_match_score >= 95:
            indication_threshold = 50
        elif drug_match_score >= 90:
            indication_threshold = 65
        else:
            indication_threshold = 70
        
        # Calculate indication similarity
        similarity = self.calculate_similarity(cdsco_indication, fda_indication)
        
        # Extract medical terms for bonus scoring
        cdsco_terms = set(re.findall(r'\b[A-Z][a-z]+\b|\\b(?:cancer|tumor|syndrome|disease|disorder|infection)\\b', 
                                     cdsco_indication, re.IGNORECASE))
        fda_terms = set(re.findall(r'\\b[A-Z][a-z]+\\b|\\b(?:cancer|tumor|syndrome|disease|disorder|infection)\\b', 
                                   fda_indication, re.IGNORECASE))
        
        # Bonus for matching key terms
        if cdsco_terms and fda_terms:
            term_overlap = len(cdsco_terms.intersection(fda_terms)) / min(len(cdsco_terms), len(fda_terms))
            if term_overlap > 0.3:
                similarity = min(100, similarity + 10)
        
        return similarity >= indication_threshold


# For backward compatibility, create an alias
DrugMatcher = EnhancedDrugMatcher