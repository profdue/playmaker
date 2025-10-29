"""
Professional match data parsing with robust error handling, validation, 
and comprehensive real-world format support.

Enhanced with:
- Advanced validation and normalization
- Statistical quality metrics
- Context-aware parsing
- Fallback strategies
- Performance optimization
"""

import re
import logging
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import statistics

logger = logging.getLogger(__name__)

class DataQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INSUFFICIENT = "insufficient"

class ParsingStatus(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    INVALID = "invalid"

@dataclass
class ParsingResult:
    status: ParsingStatus
    data: Any
    confidence: float
    original_input: str
    parsed_elements: int
    total_elements: int
    warnings: List[str]
    normalization_applied: List[str]

class ProfessionalDataParser:
    """
    Institutional-grade data parser with advanced validation,
    statistical quality assessment, and comprehensive format support.
    """
    
    def __init__(self):
        # Comprehensive form mappings
        self.form_mappings = {
            'WIN': 'W', 'WON': 'W', 'VICTORY': 'W', 'V': 'W', '1': 'W',
            'DRAW': 'D', 'TIE': 'D', 'E': 'D', 'X': 'D', '0': 'D',
            'LOSS': 'L', 'LOST': 'L', 'DEFEAT': 'L', '2': 'L',
            'YES': 'W', 'NO': 'L', 'POSITIVE': 'W', 'NEGATIVE': 'L'
        }
        
        # Team name normalization database
        self.team_normalizations = {
            # Premier League
            'man united': 'Manchester United', 'man utd': 'Manchester United',
            'man city': 'Manchester City', 'mancity': 'Manchester City',
            'spurs': 'Tottenham Hotspur', 'tottenham': 'Tottenham Hotspur',
            'newcastle': 'Newcastle United', 
            'west ham': 'West Ham United', 'westham': 'West Ham United',
            'wolves': 'Wolverhampton Wanderers',
            'leicester': 'Leicester City',
            'brighton': 'Brighton & Hove Albion',
            'norwich': 'Norwich City',
            
            # Bundesliga
            'union berlin': 'Union Berlin',
            'borussia mgladbach': 'Borussia Mönchengladbach',
            'borussia m\'gladbach': 'Borussia Mönchengladbach',
            'bmg': 'Borussia Mönchengladbach',
            'bayern': 'Bayern Munich', 'bayern munich': 'Bayern Munich',
            'bvb': 'Borussia Dortmund', 'dortmund': 'Borussia Dortmund',
            
            # La Liga
            'real madrid': 'Real Madrid', 'rm': 'Real Madrid',
            'barcelona': 'FC Barcelona', 'barca': 'FC Barcelona',
            'atletico': 'Atletico Madrid', 'atm': 'Atletico Madrid',
            
            # Serie A
            'inter': 'Inter Milan', 'inter milan': 'Inter Milan',
            'ac milan': 'AC Milan', 'milan': 'AC Milan',
            'juve': 'Juventus', 'juventus': 'Juventus',
            
            # Common abbreviations
            'utd': 'United', 'cf': 'CF', 'fc': 'FC', 'afc': 'AFC'
        }
        
        # Injury keyword patterns
        self.injury_keywords = {
            'high_confidence': [
                'injur', 'suspended', 'doubtful', 'knock', 'strain', 'sprain',
                'fracture', 'ligament', 'muscle', 'hamstring', 'calf', 'thigh',
                'groin', 'ankle', 'knee', 'shoulder', 'concussion', 'illness'
            ],
            'medium_confidence': [
                'fatigue', 'rest', 'fitness', 'recovery', 'pain', 'discomfort',
                'problem', 'issue', 'absent', 'missing', 'unavailable'
            ]
        }
        
        # League-specific configurations
        self.league_configs = {
            'premier_league': {'max_teams': 20, 'points_win': 3, 'points_draw': 1},
            'la_liga': {'max_teams': 20, 'points_win': 3, 'points_draw': 1},
            'bundesliga': {'max_teams': 18, 'points_win': 3, 'points_draw': 1},
            'serie_a': {'max_teams': 20, 'points_win': 3, 'points_draw': 1},
            'ligue_1': {'max_teams': 20, 'points_win': 3, 'points_draw': 1},
            'champions_league': {'max_teams': 32, 'points_win': 3, 'points_draw': 1},
            'europa_league': {'max_teams': 48, 'points_win': 3, 'points_draw': 1}
        }
        
        # Performance optimization
        self._compiled_patterns = {
            'numbers': re.compile(r'([+-]?\d+\.?\d*)'),
            'score': re.compile(r'(\d+)[\-\:](\d+)'),
            'player_injury': re.compile(r'^([A-Z][A-Za-z\s\.\-\']+?)\s*[\(\[].*?[\)\]]?\s*[-\:]\s*(.+)$', re.IGNORECASE),
            'odds': re.compile(r'\b\d+\.?\d*\b')
        }

    def parse_form(self, raw_text: Union[str, List], max_matches: int = 10) -> ParsingResult:
        """
        Professional form parsing with statistical confidence scoring.
        
        Args:
            raw_text: Form data in various formats
            max_matches: Maximum number of matches to return
            
        Returns:
            ParsingResult with status, data, and confidence metrics
        """
        warnings = []
        normalization_applied = []
        
        if not raw_text:
            return ParsingResult(
                status=ParsingStatus.INVALID,
                data=[],
                confidence=0.0,
                original_input=str(raw_text),
                parsed_elements=0,
                total_elements=0,
                warnings=["Empty input provided"],
                normalization_applied=[]
            )
        
        try:
            results = []
            original_input = str(raw_text)
            
            # Handle list input
            if isinstance(raw_text, list):
                for i, item in enumerate(raw_text):
                    if not item:
                        warnings.append(f"Empty item at position {i}")
                        continue
                        
                    item_clean = str(item).strip().upper()
                    if item_clean in ['W', 'D', 'L']:
                        results.append(item_clean)
                    elif item_clean in self.form_mappings:
                        mapped = self.form_mappings[item_clean]
                        results.append(mapped)
                        normalization_applied.append(f"mapped_{item_clean}_to_{mapped}")
                    elif len(item_clean) == 1:
                        # Single character not in W/D/L
                        warnings.append(f"Unrecognized form character: {item_clean}")
            
            # Handle string input
            else:
                text_upper = str(raw_text).upper()
                
                # Replace common separators with spaces
                text_clean = re.sub(r'[,;/\|]', ' ', text_upper)
                normalization_applied.append("separator_normalization")
                
                # Split and process words
                words = text_clean.split()
                
                for word in words:
                    if word in ['W', 'D', 'L']:
                        results.append(word)
                    elif word in self.form_mappings:
                        mapped = self.form_mappings[word]
                        results.append(mapped)
                        normalization_applied.append(f"mapped_{word}_to_{mapped}")
                    elif len(word) == 1 and word in 'WDL1230':  # Extended character set
                        char_map = {'1': 'W', '2': 'L', '0': 'D'}
                        if word in char_map:
                            results.append(char_map[word])
                            normalization_applied.append(f"mapped_{word}_to_{char_map[word]}")
                
                # Handle concatenated form like "WWDLW"
                if not results and len(text_upper.replace(' ', '')) <= 15:
                    concatenated = text_upper.replace(' ', '')
                    for char in concatenated:
                        if char in 'WDL':
                            results.append(char)
                        elif char in '1':
                            results.append('W')
                        elif char in '0':
                            results.append('D')
                        elif char in '2':
                            results.append('L')
                    if results:
                        normalization_applied.append("concatenated_form_parsed")
            
            # Validate results
            valid_results = []
            for result in results:
                if result in ['W', 'D', 'L']:
                    valid_results.append(result)
                else:
                    warnings.append(f"Invalid form result filtered: {result}")
            
            # Calculate confidence score
            total_elements = len(valid_results)
            confidence = self._calculate_form_confidence(valid_results, original_input, warnings)
            
            # Limit results
            final_results = valid_results[:max_matches]
            
            status = ParsingStatus.SUCCESS if confidence > 0.7 else (
                ParsingStatus.PARTIAL if confidence > 0.3 else ParsingStatus.FAILED
            )
            
            logger.debug(f"Form parsing: {original_input} -> {final_results} (confidence: {confidence:.2f})")
            
            return ParsingResult(
                status=status,
                data=final_results,
                confidence=confidence,
                original_input=original_input,
                parsed_elements=len(final_results),
                total_elements=total_elements,
                warnings=warnings,
                normalization_applied=normalization_applied
            )
            
        except Exception as e:
            logger.error(f"Form parsing error: {e}")
            return ParsingResult(
                status=ParsingStatus.FAILED,
                data=[],
                confidence=0.0,
                original_input=str(raw_text),
                parsed_elements=0,
                total_elements=0,
                warnings=[f"Parsing error: {str(e)}"],
                normalization_applied=[]
            )

    def parse_standing(self, raw_text: Union[str, List], league_context: str = "premier_league") -> ParsingResult:
        """
        Professional standing parsing with league-aware validation.
        
        Args:
            raw_text: Standing data in various formats
            league_context: League context for validation
            
        Returns:
            ParsingResult with validated standing data
        """
        warnings = []
        normalization_applied = []
        
        if not raw_text:
            return ParsingResult(
                status=ParsingStatus.INVALID,
                data=None,
                confidence=0.0,
                original_input=str(raw_text),
                parsed_elements=0,
                total_elements=4,
                warnings=["Empty input provided"],
                normalization_applied=[]
            )
        
        try:
            league_config = self.league_configs.get(league_context, self.league_configs['premier_league'])
            max_teams = league_config['max_teams']
            
            # Handle list input
            if isinstance(raw_text, list) and len(raw_text) >= 4:
                position, points, played, goal_diff = self._validate_standing_components(
                    raw_text[0], raw_text[1], raw_text[2], raw_text[3],
                    max_teams, warnings, normalization_applied
                )
                
                if position is not None:
                    standing = [position, points, played, goal_diff]
                    confidence = self._calculate_standing_confidence(standing, league_config, warnings)
                    
                    return ParsingResult(
                        status=ParsingStatus.SUCCESS if confidence > 0.8 else ParsingStatus.PARTIAL,
                        data=standing,
                        confidence=confidence,
                        original_input=str(raw_text),
                        parsed_elements=4,
                        total_elements=4,
                        warnings=warnings,
                        normalization_applied=normalization_applied
                    )
            
            # Handle string input
            text_clean = str(raw_text).replace('[', '').replace(']', '').replace('(', '').replace(')', '')
            normalization_applied.append("bracket_cleaning")
            
            # Enhanced number extraction
            numbers = self._compiled_patterns['numbers'].findall(text_clean)
            
            if len(numbers) >= 4:
                position, points, played, goal_diff = self._validate_standing_components(
                    numbers[0], numbers[1], numbers[2], numbers[3],
                    max_teams, warnings, normalization_applied
                )
                
                if position is not None:
                    standing = [position, points, played, goal_diff]
                    confidence = self._calculate_standing_confidence(standing, league_config, warnings)
                    
                    logger.debug(f"Standing parsed: {raw_text} -> {standing} (confidence: {confidence:.2f})")
                    
                    return ParsingResult(
                        status=ParsingStatus.SUCCESS if confidence > 0.8 else ParsingStatus.PARTIAL,
                        data=standing,
                        confidence=confidence,
                        original_input=str(raw_text),
                        parsed_elements=4,
                        total_elements=4,
                        warnings=warnings,
                        normalization_applied=normalization_applied
                    )
            
            warnings.append(f"Insufficient standing data: {raw_text}")
            return ParsingResult(
                status=ParsingStatus.FAILED,
                data=None,
                confidence=0.0,
                original_input=str(raw_text),
                parsed_elements=0,
                total_elements=4,
                warnings=warnings,
                normalization_applied=normalization_applied
            )
            
        except Exception as e:
            logger.error(f"Standing parsing error: {e}")
            return ParsingResult(
                status=ParsingStatus.FAILED,
                data=None,
                confidence=0.0,
                original_input=str(raw_text),
                parsed_elements=0,
                total_elements=4,
                warnings=[f"Parsing error: {str(e)}"],
                normalization_applied=[]
            )

    def parse_h2h(self, raw_text: Union[str, List], max_matches: int = 20) -> ParsingResult:
        """
        Professional head-to-head parsing with score validation.
        
        Args:
            raw_text: H2H data in various formats
            max_matches: Maximum number of matches to return
            
        Returns:
            ParsingResult with validated match scores
        """
        warnings = []
        normalization_applied = []
        results = []
        
        if not raw_text:
            return ParsingResult(
                status=ParsingStatus.INVALID,
                data=[],
                confidence=0.0,
                original_input=str(raw_text),
                parsed_elements=0,
                total_elements=0,
                warnings=["Empty input provided"],
                normalization_applied=[]
            )
        
        try:
            original_input = str(raw_text)
            
            # Handle list input
            if isinstance(raw_text, list):
                for i, match in enumerate(raw_text):
                    if isinstance(match, list) and len(match) >= 2:
                        home_score, away_score, valid = self._validate_score(match[0], match[1])
                        if valid:
                            results.append([home_score, away_score])
                        else:
                            warnings.append(f"Invalid score in list position {i}: {match}")
                    elif isinstance(match, str):
                        score_match = self._compiled_patterns['score'].match(match.strip())
                        if score_match:
                            home_score, away_score, valid = self._validate_score(
                                score_match.group(1), score_match.group(2)
                            )
                            if valid:
                                results.append([home_score, away_score])
                            else:
                                warnings.append(f"Invalid score in string: {match}")
            
            # Handle string input
            else:
                text_clean = str(raw_text).replace(',', ' ').replace(';', ' ')
                matches = self._compiled_patterns['score'].findall(text_clean)
                
                for home_str, away_str in matches:
                    home_score, away_score, valid = self._validate_score(home_str, away_str)
                    if valid:
                        results.append([home_score, away_score])
                    else:
                        warnings.append(f"Invalid score: {home_str}-{away_str}")
            
            # Calculate confidence based on valid matches and realistic scores
            confidence = self._calculate_h2h_confidence(results, warnings)
            
            # Limit results
            final_results = results[:max_matches]
            
            status = ParsingStatus.SUCCESS if confidence > 0.8 else (
                ParsingStatus.PARTIAL if confidence > 0.5 else ParsingStatus.FAILED
            )
            
            logger.debug(f"H2H parsed: {len(final_results)} matches (confidence: {confidence:.2f})")
            
            return ParsingResult(
                status=status,
                data=final_results,
                confidence=confidence,
                original_input=original_input,
                parsed_elements=len(final_results),
                total_elements=len(results),
                warnings=warnings,
                normalization_applied=normalization_applied
            )
            
        except Exception as e:
            logger.error(f"H2H parsing error: {e}")
            return ParsingResult(
                status=ParsingStatus.FAILED,
                data=[],
                confidence=0.0,
                original_input=str(raw_text),
                parsed_elements=0,
                total_elements=0,
                warnings=[f"Parsing error: {str(e)}"],
                normalization_applied=[]
            )

    def parse_odds(self, raw_text: Union[str, List]) -> ParsingResult:
        """
        Professional odds parsing with market validation.
        
        Args:
            raw_text: Odds data in various formats
            
        Returns:
            ParsingResult with validated odds data
        """
        warnings = []
        normalization_applied = []
        
        if not raw_text:
            return ParsingResult(
                status=ParsingStatus.INVALID,
                data=None,
                confidence=0.0,
                original_input=str(raw_text),
                parsed_elements=0,
                total_elements=3,
                warnings=["Empty input provided"],
                normalization_applied=[]
            )
        
        try:
            # Handle list input
            if isinstance(raw_text, list) and len(raw_text) >= 3:
                odds = []
                valid_count = 0
                
                for i, odd in enumerate(raw_text[:3]):
                    try:
                        odd_float = float(odd)
                        if 1.0 <= odd_float <= 1000.0:  # Reasonable odds range
                            odds.append(round(odd_float, 3))
                            valid_count += 1
                        else:
                            warnings.append(f"Odds value out of range: {odd_float}")
                            odds.append(None)
                    except (ValueError, TypeError):
                        warnings.append(f"Invalid odds format: {odd}")
                        odds.append(None)
                
                if valid_count == 3:
                    confidence = self._calculate_odds_confidence(odds, warnings)
                    return ParsingResult(
                        status=ParsingStatus.SUCCESS,
                        data=odds,
                        confidence=confidence,
                        original_input=str(raw_text),
                        parsed_elements=3,
                        total_elements=3,
                        warnings=warnings,
                        normalization_applied=normalization_applied
                    )
            
            # Handle string input
            text_clean = str(raw_text).replace(',', ' ').replace(';', ' ')
            numbers = self._compiled_patterns['odds'].findall(text_clean)
            
            if len(numbers) >= 3:
                odds = []
                valid_count = 0
                
                for num in numbers[:3]:
                    try:
                        odd_float = float(num)
                        if 1.0 <= odd_float <= 1000.0:
                            odds.append(round(odd_float, 3))
                            valid_count += 1
                        else:
                            warnings.append(f"Odds value out of range: {odd_float}")
                    except ValueError:
                        warnings.append(f"Invalid odds number: {num}")
                
                if valid_count == 3:
                    confidence = self._calculate_odds_confidence(odds, warnings)
                    logger.debug(f"Odds parsed: {raw_text} -> {odds} (confidence: {confidence:.2f})")
                    
                    return ParsingResult(
                        status=ParsingStatus.SUCCESS,
                        data=odds,
                        confidence=confidence,
                        original_input=str(raw_text),
                        parsed_elements=3,
                        total_elements=3,
                        warnings=warnings,
                        normalization_applied=normalization_applied
                    )
            
            warnings.append(f"Insufficient valid odds data: {raw_text}")
            return ParsingResult(
                status=ParsingStatus.FAILED,
                data=None,
                confidence=0.0,
                original_input=str(raw_text),
                parsed_elements=0,
                total_elements=3,
                warnings=warnings,
                normalization_applied=normalization_applied
            )
            
        except Exception as e:
            logger.error(f"Odds parsing error: {e}")
            return ParsingResult(
                status=ParsingStatus.FAILED,
                data=None,
                confidence=0.0,
                original_input=str(raw_text),
                parsed_elements=0,
                total_elements=3,
                warnings=[f"Parsing error: {str(e)}"],
                normalization_applied=[]
            )

    def parse_injuries(self, raw_text: Union[str, List], max_injuries: int = 10) -> ParsingResult:
        """
        Professional injury parsing with confidence scoring.
        
        Args:
            raw_text: Injury data in various formats
            max_injuries: Maximum number of injuries to return
            
        Returns:
            ParsingResult with validated injury data
        """
        warnings = []
        normalization_applied = []
        injuries = []
        
        if not raw_text:
            return ParsingResult(
                status=ParsingStatus.INVALID,
                data=[],
                confidence=0.0,
                original_input=str(raw_text),
                parsed_elements=0,
                total_elements=0,
                warnings=["Empty input provided"],
                normalization_applied=[]
            )
        
        try:
            original_input = str(raw_text)
            
            # Handle list input
            if isinstance(raw_text, list):
                for item in raw_text:
                    if isinstance(item, str) and item.strip():
                        injury_data = self._parse_single_injury(item.strip())
                        if injury_data:
                            injuries.append(injury_data)
            
            # Handle string input
            else:
                text_clean = str(raw_text).strip()
                lines = [line.strip() for line in text_clean.split('\n') if line.strip()]
                
                for line in lines:
                    injury_data = self._parse_single_injury(line)
                    if injury_data:
                        injuries.append(injury_data)
            
            # Calculate confidence based on injury patterns
            confidence = self._calculate_injuries_confidence(injuries, warnings)
            
            # Limit results
            final_injuries = injuries[:max_injuries]
            
            status = ParsingStatus.SUCCESS if confidence > 0.7 else (
                ParsingStatus.PARTIAL if confidence > 0.4 else ParsingStatus.FAILED
            )
            
            logger.debug(f"Injuries parsed: {len(final_injuries)} found (confidence: {confidence:.2f})")
            
            return ParsingResult(
                status=status,
                data=final_injuries,
                confidence=confidence,
                original_input=original_input,
                parsed_elements=len(final_injuries),
                total_elements=len(injuries),
                warnings=warnings,
                normalization_applied=normalization_applied
            )
            
        except Exception as e:
            logger.error(f"Injury parsing error: {e}")
            return ParsingResult(
                status=ParsingStatus.FAILED,
                data=[],
                confidence=0.0,
                original_input=str(raw_text),
                parsed_elements=0,
                total_elements=0,
                warnings=[f"Parsing error: {str(e)}"],
                normalization_applied=[]
            )

    def _parse_single_injury(self, text: str) -> Optional[str]:
        """Parse a single injury entry with pattern matching."""
        # Try structured patterns first
        player_match = self._compiled_patterns['player_injury'].match(text)
        if player_match:
            return text.strip()
        
        # Check for injury keywords
        text_lower = text.lower()
        high_conf_keywords = [kw for kw in self.injury_keywords['high_confidence'] if kw in text_lower]
        med_conf_keywords = [kw for kw in self.injury_keywords['medium_confidence'] if kw in text_lower]
        
        if high_conf_keywords or med_conf_keywords:
            return text.strip()
        
        return None

    def _validate_standing_components(self, pos_input, pts_input, played_input, gd_input, 
                                   max_teams, warnings, normalization_applied):
        """Validate and convert standing components with error handling."""
        try:
            # Position validation
            position = int(float(str(pos_input)))
            if position < 1 or position > max_teams:
                warnings.append(f"Position {position} outside valid range 1-{max_teams}")
                return None, None, None, None
            
            # Points validation
            points = int(float(str(pts_input)))
            if points < 0:
                warnings.append(f"Points {points} cannot be negative")
                points = 0
            
            # Played validation
            played = int(float(str(played_input)))
            if played < 0:
                warnings.append(f"Played matches {played} cannot be negative")
                played = 0
            
            # Goal difference validation
            gd_text = str(gd_input)
            if gd_text.startswith('+'):
                goal_diff = int(gd_text[1:])
                normalization_applied.append("positive_gd_normalized")
            elif gd_text.startswith('-'):
                goal_diff = int(gd_text)
            else:
                goal_diff = int(gd_text)
            
            # Realistic goal difference check
            max_reasonable_gd = played * 10  # Very conservative upper bound
            if abs(goal_diff) > max_reasonable_gd:
                warnings.append(f"Goal difference {goal_diff} seems unrealistic for {played} matches")
            
            return position, points, played, goal_diff
            
        except (ValueError, TypeError) as e:
            warnings.append(f"Invalid standing component: {e}")
            return None, None, None, None

    def _validate_score(self, home_str: str, away_str: str) -> Tuple[int, int, bool]:
        """Validate a score pair for realism."""
        try:
            home_score = int(home_str)
            away_score = int(away_str)
            
            # Basic validation
            if home_score < 0 or away_score < 0:
                return 0, 0, False
            
            # Realistic score check (very conservative)
            if home_score > 20 or away_score > 20:
                return 0, 0, False
            
            return home_score, away_score, True
            
        except (ValueError, TypeError):
            return 0, 0, False

    def _calculate_form_confidence(self, results: List[str], original_input: str, warnings: List[str]) -> float:
        """Calculate confidence score for form parsing."""
        if not results:
            return 0.0
        
        base_confidence = 0.8
        penalty = 0.0
        
        # Penalize for warnings
        penalty += len(warnings) * 0.1
        
        # Penalize for very short forms
        if len(results) < 3:
            penalty += 0.2
        
        # Bonus for longer forms
        if len(results) >= 8:
            base_confidence += 0.1
        
        # Check form distribution (shouldn't be all wins/losses)
        win_ratio = results.count('W') / len(results)
        if win_ratio > 0.9 or win_ratio < 0.1:
            penalty += 0.1
        
        return max(0.0, min(1.0, base_confidence - penalty))

    def _calculate_standing_confidence(self, standing: List[int], league_config: Dict, warnings: List[str]) -> float:
        """Calculate confidence score for standing parsing."""
        if not standing:
            return 0.0
        
        base_confidence = 0.9
        penalty = 0.0
        
        # Penalize for warnings
        penalty += len(warnings) * 0.1
        
        # Validate position
        position, points, played, goal_diff = standing
        
        if position < 1 or position > league_config['max_teams']:
            penalty += 0.3
        
        # Validate points (maximum possible)
        max_possible_points = played * league_config['points_win']
        if points > max_possible_points:
            penalty += 0.2
        
        # Validate goal difference
        max_reasonable_gd = played * 5
        if abs(goal_diff) > max_reasonable_gd:
            penalty += 0.1
        
        return max(0.0, min(1.0, base_confidence - penalty))

    def _calculate_h2h_confidence(self, results: List[List[int]], warnings: List[str]) -> float:
        """Calculate confidence score for H2H parsing."""
        if not results:
            return 0.0
        
        base_confidence = 0.8
        penalty = 0.0
        
        # Penalize for warnings
        penalty += len(warnings) * 0.05
        
        # Bonus for more matches
        if len(results) >= 5:
            base_confidence += 0.1
        elif len(results) >= 10:
            base_confidence += 0.15
        
        # Check for realistic scores
        unrealistic_scores = 0
        for home, away in results:
            if home > 10 or away > 10:  # Very high scores
                unrealistic_scores += 1
        
        if unrealistic_scores > 0:
            penalty += (unrealistic_scores / len(results)) * 0.3
        
        return max(0.0, min(1.0, base_confidence - penalty))

    def _calculate_odds_confidence(self, odds: List[float], warnings: List[str]) -> float:
        """Calculate confidence score for odds parsing."""
        if not odds or len(odds) != 3:
            return 0.0
        
        base_confidence = 0.9
        penalty = 0.0
        
        # Penalize for warnings
        penalty += len(warnings) * 0.1
        
        # Check for reasonable odds relationships
        home_odds, draw_odds, away_odds = odds
        
        # Odds should generally be between 1.0 and 100.0
        if any(odd < 1.01 or odd > 100.0 for odd in odds):
            penalty += 0.2
        
        # Check implied probabilities (should sum to ~1.0-1.05 with overround)
        implied_prob = sum(1/odd for odd in odds)
        if implied_prob < 0.95 or implied_prob > 1.10:
            penalty += 0.1
        
        return max(0.0, min(1.0, base_confidence - penalty))

    def _calculate_injuries_confidence(self, injuries: List[str], warnings: List[str]) -> float:
        """Calculate confidence score for injury parsing."""
        if not injuries:
            return 0.0
        
        base_confidence = 0.7
        penalty = 0.0
        
        # Penalize for warnings
        penalty += len(warnings) * 0.05
        
        # Analyze injury patterns
        high_conf_count = 0
        for injury in injuries:
            injury_lower = injury.lower()
            if any(kw in injury_lower for kw in self.injury_keywords['high_confidence']):
                high_conf_count += 1
        
        # Bonus for high-confidence injuries
        high_conf_ratio = high_conf_count / len(injuries)
        base_confidence += high_conf_ratio * 0.2
        
        return max(0.0, min(1.0, base_confidence - penalty))

    def validate_team_name(self, team_name: str) -> Tuple[str, float]:
        """
        Validate and normalize team name with confidence scoring.
        
        Returns:
            Tuple of (normalized_name, confidence_score)
        """
        if not team_name or not isinstance(team_name, str):
            return "Unknown Team", 0.0
        
        original = team_name.strip()
        if not original:
            return "Unknown Team", 0.0
        
        # Basic cleaning
        cleaned = re.sub(r'[^\w\s\-\.\']', '', original)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        if not cleaned:
            return "Unknown Team", 0.0
        
        # Check for exact match in normalizations
        lower_cleaned = cleaned.lower()
        if lower_cleaned in self.team_normalizations:
            return self.team_normalizations[lower_cleaned], 0.9
        
        # Check for partial matches
        for common_name, normalized in self.team_normalizations.items():
            if common_name in lower_cleaned:
                return normalized, 0.7
        
        # Check if name looks reasonable
        words = cleaned.split()
        if len(words) >= 2 and any(len(word) >= 3 for word in words):
            confidence = 0.8
            # Penalize for very short names
            if len(cleaned) < 6:
                confidence = 0.6
            return cleaned, confidence
        
        return "Unknown Team", 0.3

# Simplified interface functions for backward compatibility
def parse_form(raw_text: Union[str, List]) -> List[str]:
    """Simple form parsing interface."""
    parser = ProfessionalDataParser()
    result = parser.parse_form(raw_text)
    return result.data if result.status != ParsingStatus.FAILED else []

def parse_standing(raw_text: Union[str, List]) -> Optional[List[int]]:
    """Simple standing parsing interface."""
    parser = ProfessionalDataParser()
    result = parser.parse_standing(raw_text)
    return result.data if result.status != ParsingStatus.FAILED else None

def parse_h2h(raw_text: Union[str, List]) -> List[List[int]]:
    """Simple H2H parsing interface."""
    parser = ProfessionalDataParser()
    result = parser.parse_h2h(raw_text)
    return result.data if result.status != ParsingStatus.FAILED else []

def parse_odds(raw_text: Union[str, List]) -> Optional[List[float]]:
    """Simple odds parsing interface."""
    parser = ProfessionalDataParser()
    result = parser.parse_odds(raw_text)
    return result.data if result.status != ParsingStatus.FAILED else None

def parse_injuries(raw_text: Union[str, List]) -> List[str]:
    """Simple injury parsing interface."""
    parser = ProfessionalDataParser()
    result = parser.parse_injuries(raw_text)
    return result.data if result.status != ParsingStatus.FAILED else []

def validate_team_name(team_name: str) -> str:
    """Simple team name validation interface."""
    parser = ProfessionalDataParser()
    normalized, confidence = parser.validate_team_name(team_name)
    return normalized

# Enhanced build_match_data function
def build_match_data(
    home_team: str,
    away_team: str,
    odds_1x2: Optional[Union[str, List]] = None,
    home_form: Optional[Union[str, List]] = None,
    away_form: Optional[Union[str, List]] = None,
    home_standing: Optional[Union[str, List]] = None,
    away_standing: Optional[Union[str, List]] = None,
    head_to_head: Optional[Union[str, List]] = None,
    home_injuries: Optional[Union[str, List]] = None,
    away_injuries: Optional[Union[str, List]] = None,
    league_type: str = "premier_league",
    match_importance: str = "Normal League",
    venue_context: str = "Normal"
) -> Dict[str, Any]:
    """
    Professional match data assembly with comprehensive validation and quality scoring.
    """
    parser = ProfessionalDataParser()
    
    # Validate required fields
    home_team_normalized, home_confidence = parser.validate_team_name(home_team)
    away_team_normalized, away_confidence = parser.validate_team_name(away_team)
    
    if home_team_normalized == "Unknown Team" or away_team_normalized == "Unknown Team":
        logger.error("Invalid team names provided")
        raise ValueError("Valid home and away team names are required")
    
    # Parse all data with professional error handling
    odds_result = parser.parse_odds(odds_1x2)
    home_form_result = parser.parse_form(home_form)
    away_form_result = parser.parse_form(away_form)
    home_standing_result = parser.parse_standing(home_standing, league_type)
    away_standing_result = parser.parse_standing(away_standing, league_type)
    h2h_result = parser.parse_h2h(head_to_head)
    home_injuries_result = parser.parse_injuries(home_injuries)
    away_injuries_result = parser.parse_injuries(away_injuries)
    
    # Compile all warnings
    all_warnings = []
    all_warnings.extend(odds_result.warnings)
    all_warnings.extend(home_form_result.warnings)
    all_warnings.extend(away_form_result.warnings)
    all_warnings.extend(home_standing_result.warnings)
    all_warnings.extend(away_standing_result.warnings)
    all_warnings.extend(h2h_result.warnings)
    all_warnings.extend(home_injuries_result.warnings)
    all_warnings.extend(away_injuries_result.warnings)
    
    # Calculate overall data quality score
    field_weights = {
        'odds': 0.25,
        'home_form': 0.10,
        'away_form': 0.10,
        'home_standing': 0.15,
        'away_standing': 0.15,
        'head_to_head': 0.15,
        'team_names': 0.10
    }
    
    quality_score = (
        odds_result.confidence * field_weights['odds'] +
        home_form_result.confidence * field_weights['home_form'] +
        away_form_result.confidence * field_weights['away_form'] +
        home_standing_result.confidence * field_weights['home_standing'] +
        away_standing_result.confidence * field_weights['away_standing'] +
        h2h_result.confidence * field_weights['head_to_head'] +
        min(home_confidence, away_confidence) * field_weights['team_names']
    ) * 100
    
    # Determine overall data quality
    if quality_score >= 85:
        data_quality = DataQuality.EXCELLENT
    elif quality_score >= 70:
        data_quality = DataQuality.GOOD
    elif quality_score >= 50:
        data_quality = DataQuality.FAIR
    elif quality_score >= 30:
        data_quality = DataQuality.POOR
    else:
        data_quality = DataQuality.INSUFFICIENT
    
    match_data = {
        "home_team": home_team_normalized,
        "away_team": away_team_normalized,
        "odds_1x2": odds_result.data,
        "home_form": home_form_result.data,
        "away_form": away_form_result.data,
        "home_standing": home_standing_result.data,
        "away_standing": away_standing_result.data,
        "head_to_head": h2h_result.data,
        "home_injuries": home_injuries_result.data,
        "away_injuries": away_injuries_result.data,
        "league_type": league_type,
        "match_importance": match_importance,
        "venue_context": venue_context,
        "data_quality": {
            "score": round(quality_score, 1),
            "level": data_quality.value,
            "warnings": all_warnings,
            "field_confidence": {
                "odds": round(odds_result.confidence, 3),
                "home_form": round(home_form_result.confidence, 3),
                "away_form": round(away_form_result.confidence, 3),
                "home_standing": round(home_standing_result.confidence, 3),
                "away_standing": round(away_standing_result.confidence, 3),
                "head_to_head": round(h2h_result.confidence, 3),
                "team_names": round(min(home_confidence, away_confidence), 3)
            }
        },
        "parsing_metadata": {
            "timestamp": datetime.now().isoformat(),
            "parser_version": "2.0.0",
            "fields_parsed": {
                "home_form": home_form_result.parsed_elements,
                "away_form": away_form_result.parsed_elements,
                "head_to_head": h2h_result.parsed_elements,
                "home_injuries": home_injuries_result.parsed_elements,
                "away_injuries": away_injuries_result.parsed_elements
            },
            "parsing_status": {
                "odds": odds_result.status.value,
                "home_form": home_form_result.status.value,
                "away_form": away_form_result.status.value,
                "home_standing": home_standing_result.status.value,
                "away_standing": away_standing_result.status.value,
                "head_to_head": h2h_result.status.value,
                "home_injuries": home_injuries_result.status.value,
                "away_injuries": away_injuries_result.status.value
            }
        }
    }
    
    logger.info(
        f"Match data built: {home_team_normalized} vs {away_team_normalized} | "
        f"Quality: {data_quality.value} ({quality_score:.1f}%) | "
        f"Form: {len(match_data['home_form'])}/{len(match_data['away_form'])} | "
        f"H2H: {len(match_data['head_to_head'])} | "
        f"Warnings: {len(all_warnings)}"
    )
    
    return match_data

# Quick parsing functions for simple use cases
def quick_parse_form(form_input: Union[str, List]) -> List[str]:
    """Quick form parsing for simple use cases."""
    return parse_form(form_input)

def quick_parse_standing(standing_input: Union[str, List]) -> Optional[List[int]]:
    """Quick standing parsing for simple use cases."""
    return parse_standing(standing_input)

# Compatibility functions for existing code
parse_team_form = parse_form
parse_league_standing = parse_standing
parse_head_to_head = parse_h2h
parse_betting_odds = parse_odds
parse_player_injuries = parse_injuries

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the enhanced professional parser
    test_cases = [
        {
            "home_team": "man utd",
            "away_team": "Liverpool", 
            "odds_1x2": "2.10, 3.40, 3.20",
            "home_form": "W D W L W",
            "away_form": ["W", "W", "D", "L", "W"],
            "home_standing": "3, 45, 20, +15",
            "away_standing": [1, 52, 20, 25],
            "head_to_head": "2-1, 1-1, 0-2, 3-0, 1-2",
            "home_injuries": "R. Martinez (D) - Hamstring Injury\nBruno Fernandes - Suspended",
            "away_injuries": ["Salah (F) - Knock", "Van Dijk (D) - Muscle fatigue"],
            "league_type": "premier_league",
            "match_importance": "High Derby",
            "venue_context": "Home Advantage"
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        try:
            print(f"\n=== Test Case {i+1} ===")
            match_data = build_match_data(**test_case)
            
            print(f"✓ Successfully parsed match data")
            print(f"  Teams: {match_data['home_team']} vs {match_data['away_team']}")
            print(f"  Quality: {match_data['data_quality']['level']} ({match_data['data_quality']['score']}%)")
            print(f"  Home Form: {match_data['home_form']} (confidence: {match_data['data_quality']['field_confidence']['home_form']})")
            print(f"  Standings: H{match_data['home_standing']} A{match_data['away_standing']}")
            print(f"  H2H Matches: {len(match_data['head_to_head'])}")
            print(f"  Odds: {match_data['odds_1x2']}")
            print(f"  Injuries: H{len(match_data['home_injuries'])} A{len(match_data['away_injuries'])}")
            
            if match_data['data_quality']['warnings']:
                print(f"  Warnings: {len(match_data['data_quality']['warnings'])}")
                for warning in match_data['data_quality']['warnings'][:3]:  # Show first 3
                    print(f"    - {warning}")
                    
        except Exception as e:
            print(f"✗ Error in test case {i+1}: {e}")
