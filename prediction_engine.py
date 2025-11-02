# prediction_engine.py
import numpy as np
from scipy.stats import poisson, skellam
from typing import Dict, Any, Tuple, List, Optional
import math
import pandas as pd
from dataclasses import dataclass
import logging
from datetime import datetime
import json
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MatchContext(Enum):
    OFFENSIVE_SHOWDOWN = "offensive_showdown"
    DEFENSIVE_BATTLE = "defensive_battle" 
    TACTICAL_STALEMATE = "tactical_stalemate"
    HOME_DOMINANCE = "home_dominance"
    AWAY_COUNTER = "away_counter"
    UNPREDICTABLE = "unpredictable"

@dataclass
class BettingSignal:
    """Data class for betting signals"""
    market: str
    model_prob: float
    book_prob: float
    edge: float
    confidence: str
    recommended_stake: float
    value_rating: str

@dataclass
class MonteCarloResults:
    """Data class for Monte Carlo simulation results"""
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    over_25_prob: float
    btts_prob: float
    exact_scores: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    probability_volatility: Dict[str, float]

class TeamTierCalibrator:
    """
    PROFESSIONAL MULTI-LEAGUE TIER-BASED CALIBRATION SYSTEM
    UPDATED WITH ACTUAL 2025/2026 SEASON DATA
    """
    
    def __init__(self):
        # Enhanced league-specific baselines
        self.league_baselines = {
            'premier_league': {'avg_goals': 2.8, 'home_advantage': 0.35, 'scoring_profile': 'high'},
            'la_liga': {'avg_goals': 2.6, 'home_advantage': 0.32, 'scoring_profile': 'medium'},
            'serie_a': {'avg_goals': 2.7, 'home_advantage': 0.38, 'scoring_profile': 'medium'},
            'bundesliga': {'avg_goals': 3.1, 'home_advantage': 0.28, 'scoring_profile': 'very_high'},
            'ligue_1': {'avg_goals': 2.5, 'home_advantage': 0.34, 'scoring_profile': 'low'},
            'liga_portugal': {'avg_goals': 2.6, 'home_advantage': 0.42, 'scoring_profile': 'medium'},
            'brasileirao': {'avg_goals': 2.4, 'home_advantage': 0.45, 'scoring_profile': 'medium'},
            'liga_mx': {'avg_goals': 2.7, 'home_advantage': 0.40, 'scoring_profile': 'high'},
            'eredivisie': {'avg_goals': 3.0, 'home_advantage': 0.30, 'scoring_profile': 'very_high'},
            'default': {'avg_goals': 2.7, 'home_advantage': 0.35, 'scoring_profile': 'medium'}
        }
        
        # Tier-based calibration
        self.tier_calibration = {
            'ELITE': {
                'baseline_xg_for': 2.1, 
                'baseline_xg_against': 0.8,
                'win_prob_range': (0.65, 0.85)
            },
            'STRONG': {
                'baseline_xg_for': 1.8, 
                'baseline_xg_against': 1.1,
                'win_prob_range': (0.45, 0.70)
            },
            'MEDIUM': {
                'baseline_xg_for': 1.5, 
                'baseline_xg_against': 1.4,
                'win_prob_range': (0.25, 0.55)
            },
            'WEAK': {
                'baseline_xg_for': 1.1, 
                'baseline_xg_against': 1.8,
                'win_prob_range': (0.10, 0.35)
            }
        }
        
        # Multi-league team databases - UPDATED WITH ACTUAL 2025/2026 DATA
        self.team_databases = self._initialize_multi_league_teams()
    
    def _initialize_multi_league_teams(self) -> Dict[str, Dict[str, str]]:
        """Initialize comprehensive multi-league team databases with ACTUAL 2025/2026 data"""
        return {
            'premier_league': {
                # ELITE (Title Contenders)
                'Arsenal': 'ELITE', 'Man City': 'ELITE', 
                # STRONG (European Places)
                'Liverpool': 'STRONG', 'Bournemouth': 'STRONG', 'Tottenham': 'STRONG',
                'Chelsea': 'STRONG', 'Sunderland': 'STRONG', 'Man Utd': 'STRONG',
                'Crystal Palace': 'STRONG', 'Brighton': 'STRONG', 'Aston Villa': 'STRONG',
                # MEDIUM (Mid-table)
                'Brentford': 'MEDIUM', 'Newcastle': 'MEDIUM', 'Fulham': 'MEDIUM',
                'Everton': 'MEDIUM', 'Leeds': 'MEDIUM', 'Burnley': 'MEDIUM',
                # WEAK (Relegation Battlers)
                'West Ham': 'WEAK', 'Forest': 'WEAK', 'Wolves': 'WEAK'
            },
            'serie_a': {
                # ELITE
                'Napoli': 'ELITE', 'Inter': 'ELITE', 'Roma': 'ELITE',
                # STRONG
                'Bologna': 'STRONG', 'Milan': 'STRONG', 'Juventus': 'STRONG',
                'Como': 'STRONG', 'Udinese': 'STRONG', 'Cremonese': 'STRONG',
                'Atalanta': 'STRONG', 'Sassuolo': 'STRONG', 'Torino': 'STRONG',
                # MEDIUM
                'Lazio': 'MEDIUM', 'Cagliari': 'MEDIUM', 'Lecce': 'MEDIUM',
                'Parma': 'MEDIUM', 'Pisa': 'MEDIUM', 'Verona': 'MEDIUM',
                # WEAK
                'Fiorentina': 'WEAK', 'Genoa': 'WEAK'
            },
            'la_liga': {
                # ELITE
                'Real Madrid': 'ELITE', 'Barcelona': 'ELITE', 
                # STRONG
                'Villarreal': 'STRONG', 'Atl. Madrid': 'STRONG', 'Espanyol': 'STRONG',
                'Getafe': 'STRONG', 'Real Betis': 'STRONG', 'Alavés': 'STRONG',
                'Elche': 'STRONG', 'Rayo Vallecano': 'STRONG', 'Athletic Club': 'STRONG',
                # MEDIUM
                'Celta': 'MEDIUM', 'Sevilla': 'MEDIUM', 'Real Sociedad': 'MEDIUM',
                'Osasuna': 'MEDIUM', 'Mallorca': 'MEDIUM', 'Levante': 'MEDIUM',
                'Valencia': 'MEDIUM', 'Real Oviedo': 'MEDIUM',
                # WEAK
                'Girona': 'WEAK'
            },
            'bundesliga': {
                # ELITE
                'Bayern': 'ELITE', 
                # STRONG
                'Leipzig': 'STRONG', 'Dortmund': 'STRONG', 'Stuttgart': 'STRONG',
                'Leverkusen': 'STRONG', 'Hoffenheim': 'STRONG', 'Köln': 'STRONG',
                'Frankfurt': 'STRONG', 'Bremen': 'STRONG', 'Union Berlin': 'STRONG',
                # MEDIUM
                'Freiburg': 'MEDIUM', 'Wolfsburg': 'MEDIUM', 'HSV': 'MEDIUM',
                'Augsburg': 'MEDIUM', 'St. Pauli': 'MEDIUM', 'M\'gladbach': 'MEDIUM',
                # WEAK
                'Mainz 05': 'WEAK', 'Heidenheim': 'WEAK'
            },
            'eredivisie': {
                # ELITE
                'Feyenoord': 'ELITE', 'PSV': 'ELITE',
                # STRONG
                'AZ': 'STRONG', 'Ajax': 'STRONG', 'Groningen': 'STRONG',
                'Utrecht': 'STRONG', 'Sparta': 'STRONG', 'NEC': 'STRONG',
                'Twente': 'STRONG', 'Heerenveen': 'STRONG', 'GA Eagles': 'STRONG',
                # MEDIUM
                'Fortuna': 'MEDIUM', 'NAC Breda': 'MEDIUM', 'Volendam': 'MEDIUM',
                'Excelsior': 'MEDIUM', 'Zwolle': 'MEDIUM', 'Telstar': 'MEDIUM',
                # WEAK
                'Heracles': 'WEAK'
            },
            'liga_portugal': {
                'Sporting CP': 'ELITE', 'Porto': 'ELITE', 'Benfica': 'ELITE',
                'Braga': 'STRONG', 
                'Vitoria Guimaraes': 'MEDIUM', 'Boavista': 'MEDIUM', 'Famalicao': 'MEDIUM',
                'Casa Pia': 'WEAK', 'Rio Ave': 'WEAK', 'Estoril': 'WEAK', 
            },
            'brasileirao': {
                'Flamengo': 'ELITE', 'Palmeiras': 'ELITE', 'Sao Paulo': 'ELITE',
                'Atletico Mineiro': 'STRONG', 'Gremio': 'STRONG', 'Fluminense': 'STRONG',
                'Corinthians': 'STRONG', 'Internacional': 'STRONG',
                'Fortaleza': 'MEDIUM', 'Cruzeiro': 'MEDIUM', 'Bahia': 'MEDIUM',
            },
            'liga_mx': {
                'America': 'ELITE', 'Monterrey': 'ELITE', 'Tigres': 'ELITE',
                'Cruz Azul': 'STRONG', 'Guadalajara': 'STRONG', 'Pumas': 'STRONG',
                'Santos Laguna': 'STRONG', 'Toluca': 'STRONG',
            }
        }
    
    def get_league_teams(self, league: str) -> List[str]:
        """Get all teams for a specific league"""
        return list(self.team_databases.get(league, {}).keys())
    
    def get_team_tier(self, team: str, league: str) -> str:
        """Get team tier with league context"""
        # Handle team name variations
        team_variations = {
            'Man City': 'Manchester City', 'Man Utd': 'Manchester United',
            'Atl. Madrid': 'Atletico Madrid', 'M\'gladbach': 'Borussia M\'gladbach',
            'GA Eagles': 'Go Ahead Eagles', 'Forest': 'Nottingham Forest'
        }
        
        actual_team = team_variations.get(team, team)
        league_teams = self.team_databases.get(league, {})
        return league_teams.get(actual_team, 'MEDIUM')
    
    def get_tier_baselines(self, team: str, league: str) -> Tuple[float, float]:
        """Get baseline xG for and against for a team with league adjustment"""
        tier = self.get_team_tier(team, league)
        baselines = self.tier_calibration.get(tier, self.tier_calibration['MEDIUM'])
        
        # Adjust for league scoring profile
        league_profile = self.league_baselines.get(league, self.league_baselines['default'])['scoring_profile']
        profile_multiplier = {
            'very_high': 1.15,
            'high': 1.08,
            'medium': 1.0,
            'low': 0.92
        }.get(league_profile, 1.0)
        
        xg_for = baselines['baseline_xg_for'] * profile_multiplier
        xg_against = baselines['baseline_xg_against'] * profile_multiplier
        
        return xg_for, xg_against
    
    def get_contextual_home_advantage(self, home_team: str, away_team: str, league: str) -> float:
        """Get contextual home advantage based on team tiers and league"""
        base_advantage = self.league_baselines.get(league, self.league_baselines['default'])['home_advantage']
        home_tier = self.get_team_tier(home_team, league)
        away_tier = self.get_team_tier(away_team, league)
        
        # Adjust home advantage based on tier matchup
        if home_tier == 'WEAK' and away_tier == 'ELITE':
            return base_advantage * 0.6  # Reduced advantage for weak teams vs elite
        elif home_tier == 'ELITE' and away_tier == 'WEAK':
            return base_advantage * 1.2  # Increased advantage for elite vs weak
        elif home_tier == 'WEAK' and away_tier == 'STRONG':
            return base_advantage * 0.8  # Slightly reduced
        
        return base_advantage
    
    def apply_tier_reality_check(self, home_xg: float, away_xg: float, home_team: str, away_team: str, league: str) -> Tuple[float, float]:
        """Apply tier-based reality checks to xG values with league context"""
        home_tier = self.get_team_tier(home_team, league)
        away_tier = self.get_team_tier(away_team, league)
        
        home_baseline, _ = self.get_tier_baselines(home_team, league)
        away_baseline, _ = self.get_tier_baselines(away_team, league)
        
        # Ensure xG values are within reasonable ranges for team tiers
        home_xg_min = home_baseline * 0.6
        home_xg_max = home_baseline * 1.4
        away_xg_min = away_baseline * 0.6
        away_xg_max = away_baseline * 1.4
        
        home_xg = max(home_xg_min, min(home_xg_max, home_xg))
        away_xg = max(away_xg_min, min(away_xg_max, away_xg))
        
        # Apply tier matchup logic
        if home_tier == 'ELITE' and away_tier == 'WEAK':
            home_xg = max(home_xg, away_xg + 0.8)  # Elite should have significant advantage
        elif home_tier == 'WEAK' and away_tier == 'ELITE':
            away_xg = max(away_xg, home_xg + 0.8)  # Elite should have significant advantage
        
        return round(home_xg, 3), round(away_xg, 3)
    
    def validate_probability_sanity(self, home_win: float, draw: float, away_win: float, home_team: str, away_team: str, league: str) -> bool:
        """Validate probabilities make sense given team tiers"""
        home_tier = self.get_team_tier(home_team, league)
        away_tier = self.get_team_tier(away_team, league)
        home_range = self.tier_calibration[home_tier]['win_prob_range']
        away_range = self.tier_calibration[away_tier]['win_prob_range']
        
        # Check if probabilities are within reasonable ranges
        if home_win < home_range[0] or home_win > home_range[1]:
            logger.warning(f"Home win probability {home_win:.3f} outside expected range {home_range} for {home_tier} team")
            return False
        
        if away_win < away_range[0] or away_win > away_range[1]:
            logger.warning(f"Away win probability {away_win:.3f} outside expected range {away_range} for {away_tier} team")
            return False
            
        return True

class PracticalFormAnalyzer:
    """Enhanced form analysis that considers opponent quality and trends"""
    
    def __init__(self, calibrator: TeamTierCalibrator):
        self.calibrator = calibrator
    
    def calculate_form_strength(self, form_data: List[str], league: str) -> float:
        """Calculate form strength from Last 5 data (W, D, L)"""
        if not form_data:
            return 1.0
            
        points_map = {'W': 3, 'D': 1, 'L': 0}
        total_points = 0
        
        for result in form_data:
            total_points += points_map.get(result, 0)
        
        avg_points = total_points / len(form_data)
        
        # Convert to multiplier (2.0 pts/game = 1.0 multiplier)
        form_multiplier = 0.7 + (avg_points / 2.0) * 0.6  # Range: 0.7 to 1.3
        
        return max(0.7, min(1.3, form_multiplier))
    
    def detect_form_trends(self, last_5_games: List[str]) -> str:
        """Simple trend detection without overcomplicating"""
        if len(last_5_games) < 3:
            return "STABLE"
            
        recent_3 = last_5_games[:3]
        previous_2 = last_5_games[3:5] if len(last_5_games) >= 5 else []
        
        points_map = {'W': 3, 'D': 1, 'L': 0}
        
        recent_points = sum(points_map.get(r, 0) for r in recent_3) / 3
        previous_points = sum(points_map.get(r, 0) for r in previous_2) / 2 if previous_2 else 2.0
        
        if recent_points > previous_points + 0.8:
            return "IMPROVING"
        elif recent_points < previous_points - 0.8:
            return "DECLINING" 
        else:
            return "STABLE"

class PracticalMotivationEngine:
    """Real-world motivation factors that matter"""
    
    def __init__(self, calibrator: TeamTierCalibrator):
        self.calibrator = calibrator
        self.major_derbies = self._initialize_derbies()
    
    def calculate_practical_motivation(self, team: str, opponent: str, league: str, 
                                    position: int, weeks_remaining: int) -> str:
        """Real-world motivation factors that matter"""
        
        # Derby matches
        if self._is_derby_match(team, opponent, league):
            return "VERY_HIGH"
        
        # End-of-season effects (last 8 weeks)
        if weeks_remaining <= 8:
            if position <= 4:  # Champions League
                return "VERY_HIGH"
            elif position <= 6:  # European places
                return "HIGH"
            elif position >= 17:  # Relegation zone
                return "VERY_HIGH"
            elif position >= 14:  # Relegation threatened
                return "HIGH"
            else:
                return "LOW"  # Mid-table, nothing to play for
        
        return "NORMAL"
    
    def _initialize_derbies(self) -> Dict[str, List[Tuple[str, str]]]:
        """Hardcode major derbies"""
        return {
            'premier_league': [
                ('Arsenal', 'Tottenham'), ('Man United', 'Man City'),
                ('Liverpool', 'Everton'), ('Chelsea', 'Tottenham'),
                ('Man United', 'Liverpool'), ('Arsenal', 'Chelsea')
            ],
            'la_liga': [
                ('Real Madrid', 'Barcelona'), ('Atletico Madrid', 'Real Madrid'),
                ('Barcelona', 'Espanyol'), ('Sevilla', 'Real Betis')
            ],
            'serie_a': [
                ('Inter', 'AC Milan'), ('Juventus', 'Inter'),
                ('Roma', 'Lazio'), ('Napoli', 'Roma')
            ],
            'bundesliga': [
                ('Bayern Munich', 'Borussia Dortmund'), ('Schalke', 'Borussia Dortmund'),
                ('Bayern Munich', 'Hamburg'), ('Borussia Dortmund', 'Schalke')
            ]
        }
    
    def _is_derby_match(self, home_team: str, away_team: str, league: str) -> bool:
        """Check if match is a major derby"""
        derbies = self.major_derbies.get(league, [])
        return (home_team, away_team) in derbies or (away_team, home_team) in derbies

class PracticalHomeAdvantage:
    """Enhanced home advantage with practical factors"""
    
    def __init__(self, calibrator: TeamTierCalibrator):
        self.calibrator = calibrator
    
    def get_enhanced_home_advantage(self, home_team: str, away_team: str, league: str, home_form: List[str]) -> float:
        """Better home advantage based on REAL patterns"""
        base_advantage = self.calibrator.get_contextual_home_advantage(home_team, away_team, league)
        
        # "Fortress" effect - teams strong at home
        fortress_bonus = self._calculate_fortress_effect(home_form)
        
        return base_advantage * fortress_bonus
    
    def _calculate_fortress_effect(self, home_form: List[str]) -> float:
        """Simple home strength calculation based on recent home form"""
        if not home_form:
            return 1.0
            
        home_wins = home_form.count('W')
        home_draws = home_form.count('D')
        home_points = home_wins * 3 + home_draws
        
        avg_points = home_points / len(home_form)
        
        if avg_points >= 2.0:  # Strong home form
            return 1.15
        elif avg_points <= 1.0:  # Poor home form
            return 0.9
        return 1.0

class PracticalInjuryEngine:
    """Better injury assessment using common knowledge"""
    
    def __init__(self, calibrator: TeamTierCalibrator):
        self.calibrator = calibrator
    
    def assess_injury_impact(self, team: str, league: str) -> float:
        """Simple injury impact based on team depth"""
        tier = self.calibrator.get_team_tier(team, league)
        
        # Elite teams have better squad depth
        depth_factor = {
            'ELITE': 0.95,    # Can cope better with injuries
            'STRONG': 0.90,
            'MEDIUM': 0.85, 
            'WEAK': 0.80      # Harder hit by injuries
        }.get(tier, 0.85)
        
        return depth_factor

class PracticalBTTSEngine:
    """Enhanced BTTS logic with practical adjustments"""
    
    def __init__(self, calibrator: TeamTierCalibrator):
        self.calibrator = calibrator
    
    def calculate_enhanced_btts(self, home_xg: float, away_xg: float, home_team: str, away_team: str, 
                              league: str, home_goals: int, away_goals: int, home_conceded: int, away_conceded: int) -> float:
        """Better BTTS using actual goal data"""
        
        # Base BTTS from independent probabilities
        home_score_prob = 1 - math.exp(-home_xg)
        away_score_prob = 1 - math.exp(-away_xg)
        base_btts = home_score_prob * away_score_prob
        
        # Recent goal trends (goals scored and conceded)
        home_attack_strength = home_goals / 10.0  # per game
        away_attack_strength = away_goals / 10.0
        home_defense_weakness = home_conceded / 10.0
        away_defense_weakness = away_conceded / 10.0
        
        attack_factor = (home_attack_strength + away_attack_strength) / 2.7  # Normalize to league average
        defense_factor = (home_defense_weakness + away_defense_weakness) / 2.7
        
        adjusted_btts = base_btts * attack_factor * defense_factor
        
        return max(0.15, min(0.85, adjusted_btts))

class SignalEngine:
    """
    REALISTIC PREDICTIVE ENGINE - With Enhanced Practical Features
    UPDATED FOR 2025/2026 SEASON
    """
    
    def __init__(self, match_data: Dict[str, Any], calibration_data: Optional[Dict] = None):
        self.data = self._validate_and_enhance_data(match_data)
        self.calibration_data = calibration_data or {}
        self.calibrator = TeamTierCalibrator()
        self.form_analyzer = PracticalFormAnalyzer(self.calibrator)
        self.motivation_engine = PracticalMotivationEngine(self.calibrator)
        self.home_advantage_engine = PracticalHomeAdvantage(self.calibrator)
        self.injury_engine = PracticalInjuryEngine(self.calibrator)
        self.btts_engine = PracticalBTTSEngine(self.calibrator)
        self._setup_realistic_parameters()
        self.match_context = MatchContext.UNPREDICTABLE
        
    def _validate_and_enhance_data(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced data validation with realistic features"""
        required_fields = ['home_team', 'away_team', 'league']
        
        for field in required_fields:
            if field not in match_data:
                raise ValueError(f"Missing required field: {field}")
        
        enhanced_data = match_data.copy()
        
        # Enhanced numeric validation
        predictive_fields = {
            'home_goals': (0, 30, 1.5),
            'away_goals': (0, 30, 1.5),
            'home_conceded': (0, 30, 1.5),
            'away_conceded': (0, 30, 1.5),
            'home_position': (1, 20, 10),
            'away_position': (1, 20, 10),
        }
        
        for field, (min_val, max_val, default) in predictive_fields.items():
            if field in enhanced_data:
                try:
                    value = float(enhanced_data[field])
                    if value < min_val or value > max_val:
                        logger.warning(f"Field {field} value {value} outside expected range")
                        enhanced_data[field] = max(min_val, min(value, max_val))
                    else:
                        enhanced_data[field] = value
                except (TypeError, ValueError):
                    enhanced_data[field] = default
            else:
                enhanced_data[field] = default
        
        # Handle form data (W, D, L)
        for form_field in ['home_form', 'away_form']:
            if form_field in enhanced_data:
                if isinstance(enhanced_data[form_field], list):
                    # Ensure all entries are valid
                    valid_form = []
                    for result in enhanced_data[form_field]:
                        if result in ['W', 'D', 'L']:
                            valid_form.append(result)
                    enhanced_data[form_field] = valid_form
                else:
                    enhanced_data[form_field] = []
        
        # Enhanced motivation handling
        if 'motivation' not in enhanced_data:
            enhanced_data['motivation'] = {'home': 'Normal', 'away': 'Normal'}
        
        # Calculate data quality score
        enhanced_data['data_quality_score'] = self._calculate_data_quality(enhanced_data)
        
        return enhanced_data

    def _calculate_data_quality(self, data: Dict) -> float:
        """Calculate data quality score"""
        score = 0
        max_score = 0
        
        # Basic match info
        if data.get('home_team') and data.get('away_team'):
            score += 20
        max_score += 20
        
        # Goals data
        if data.get('home_goals', 0) > 0:
            score += 15
        if data.get('away_goals', 0) > 0:
            score += 15
        max_score += 30
        
        # Recent form with sufficient data
        if len(data.get('home_form', [])) >= 3:
            score += 10
        if len(data.get('away_form', [])) >= 3:
            score += 10
        max_score += 20
        
        # League position data
        if data.get('home_position') and data.get('away_position'):
            score += 20
        max_score += 20
        
        return (score / max_score) * 100
    
    def _setup_realistic_parameters(self):
        """REALISTIC calibration parameters"""
        self.calibration_params = {
            'form_decay_rate': 0.85,
            'h2h_weight': 0.15,
            'injury_impact': 0.08,
            'motivation_impact': 0.10,
            'bivariate_correlation': 0.12,
            'defensive_impact_multiplier': 0.4,
        }
        
        if self.calibration_data:
            self.calibration_params.update(self.calibration_data)

    def _calculate_motivation_impact(self, motivation_level: str) -> float:
        """Motivation impact"""
        multipliers = {
            "Low": 0.90, "Normal": 1.0, "High": 1.08, "Very High": 1.12,
            "low": 0.90, "normal": 1.0, "high": 1.08, "very high": 1.12
        }
        return multipliers.get(motivation_level, 1.0)

    def _determine_match_context(self, home_xg: float, away_xg: float, home_team: str, away_team: str, league: str) -> MatchContext:
        """Realistic context determination with tier awareness"""
        home_tier = self.calibrator.get_team_tier(home_team, league)
        away_tier = self.calibrator.get_team_tier(away_team, league)
        
        total_xg = home_xg + away_xg
        xg_difference = home_xg - away_xg
        
        # Tier-aware context determination
        if home_tier == 'ELITE' and away_tier == 'WEAK' and xg_difference > 0.5:
            return MatchContext.HOME_DOMINANCE
        elif away_tier == 'ELITE' and home_tier == 'WEAK' and xg_difference < -0.5:
            return MatchContext.AWAY_COUNTER
        
        if total_xg < 2.2:
            return MatchContext.DEFENSIVE_BATTLE
        elif total_xg > 3.2:
            return MatchContext.OFFENSIVE_SHOWDOWN
        elif abs(xg_difference) < 0.3:
            return MatchContext.TACTICAL_STALEMATE
        
        return MatchContext.UNPREDICTABLE

    def _calculate_weighted_form_impact(self, form_data: List[str], team_tier: str) -> float:
        """Calculate form impact with tier-based weighting"""
        if not form_data:
            return 1.0
        
        form_strength = self.form_analyzer.calculate_form_strength(form_data, self.data['league'])
        
        # Tier-based form impact - weaker teams get more form volatility
        form_weight = {
            'ELITE': 0.3,    # 30% form, 70% baseline quality
            'STRONG': 0.4,   # 40% form, 60% baseline quality  
            'MEDIUM': 0.5,   # 50% form, 50% baseline quality
            'WEAK': 0.6      # 60% form, 40% baseline quality
        }.get(team_tier, 0.5)
        
        # Blend form impact with baseline (1.0)
        weighted_impact = (form_strength * form_weight) + (1.0 * (1 - form_weight))
        
        return max(0.7, min(1.3, weighted_impact))

    def _calculate_enhanced_btts_probability(self, home_xg: float, away_xg: float, home_team: str, away_team: str, league: str) -> float:
        """Enhanced BTTS probability with actual goal data"""
        
        home_goals = self.data.get('home_goals', 0)
        away_goals = self.data.get('away_goals', 0)
        home_conceded = self.data.get('home_conceded', 0)
        away_conceded = self.data.get('away_conceded', 0)
        
        return self.btts_engine.calculate_enhanced_btts(
            home_xg, away_xg, home_team, away_team, league, 
            home_goals, away_goals, home_conceded, away_conceded
        )

    def _calculate_enhanced_over_under(self, home_xg: float, away_xg: float, home_team: str, away_team: str, league: str) -> Dict[str, float]:
        """Enhanced Over/Under probabilities with tactical awareness"""
        
        total_xg = home_xg + away_xg
        
        # Base probabilities from Poisson
        over_25_base = 1 - poisson.cdf(2, total_xg)
        under_25_base = poisson.cdf(2, total_xg)
        
        # Use actual goal data for adjustments
        home_goals_avg = self.data.get('home_goals', 0) / 10.0
        away_goals_avg = self.data.get('away_goals', 0) / 10.0
        recent_scoring = (home_goals_avg + away_goals_avg) / 2.7  # Normalize to league average
        
        momentum_factor = min(1.3, max(0.7, recent_scoring))
        
        # Apply adjustments
        over_25_adj = over_25_base * momentum_factor
        under_25_adj = under_25_base / momentum_factor
        
        # Normalize to ensure they sum to ~1.0
        total = over_25_adj + under_25_adj
        if total > 0:
            over_25_adj /= total
            under_25_adj /= total
        
        return {
            'over_25': max(0.1, min(0.9, over_25_adj)),
            'under_25': max(0.1, min(0.9, under_25_adj))
        }

    def _calculate_realistic_xg(self) -> Tuple[float, float]:
        """Calculate REALISTIC predictive xG with 2025/2026 data"""
        league = self.data.get('league', 'premier_league')
        
        # Get tier-based baselines with league context
        home_baseline, home_def_baseline = self.calibrator.get_tier_baselines(self.data['home_team'], league)
        away_baseline, away_def_baseline = self.calibrator.get_tier_baselines(self.data['away_team'], league)
        
        # Use actual goal data from the provided tables
        home_goals_avg = self.data.get('home_goals', 0) / 10.0
        away_goals_avg = self.data.get('away_goals', 0) / 10.0
        
        # Blend recent performance with tier baseline
        home_attack = (home_goals_avg * 0.6) + (home_baseline * 0.4)
        away_attack = (away_goals_avg * 0.6) + (away_baseline * 0.4)
        
        home_conceded_avg = self.data.get('home_conceded', 0) / 10.0
        away_conceded_avg = self.data.get('away_conceded', 0) / 10.0
        
        # Blend defensive performance with tier baseline
        home_defense = (home_conceded_avg * 0.6) + (home_def_baseline * 0.4)
        away_defense = (away_conceded_avg * 0.6) + (away_def_baseline * 0.4)
        
        # Calculate xG using blended values
        league_avg = self.calibrator.league_baselines.get(league, self.calibrator.league_baselines['default'])['avg_goals']
        
        home_xg = home_attack * (1 - (away_defense / league_avg) * self.calibration_params['defensive_impact_multiplier'])
        away_xg = away_attack * (1 - (home_defense / league_avg) * self.calibration_params['defensive_impact_multiplier'])
        
        # Apply ENHANCED contextual home advantage
        home_form = self.data.get('home_form', [])
        home_advantage = self.home_advantage_engine.get_enhanced_home_advantage(
            self.data['home_team'], self.data['away_team'], league, home_form
        )
        home_xg *= (1 + home_advantage)
        
        # Apply weighted form impact
        home_tier = self.calibrator.get_team_tier(self.data['home_team'], league)
        away_tier = self.calibrator.get_team_tier(self.data['away_team'], league)
        
        home_form_impact = self._calculate_weighted_form_impact(self.data.get('home_form', []), home_tier)
        away_form_impact = self._calculate_weighted_form_impact(self.data.get('away_form', []), away_tier)
        
        home_xg *= home_form_impact
        away_xg *= away_form_impact
        
        # Apply ENHANCED motivation
        motivation = self.data.get('motivation', {})
        
        # Calculate practical motivation based on league position
        home_position = self.data.get('home_position', 10)
        away_position = self.data.get('away_position', 10)
        weeks_remaining = self.data.get('weeks_remaining', 28)  # Assuming ~10 games played
        
        home_motivation = self.motivation_engine.calculate_practical_motivation(
            self.data['home_team'], self.data['away_team'], league, home_position, weeks_remaining
        )
        away_motivation = self.motivation_engine.calculate_practical_motivation(
            self.data['away_team'], self.data['home_team'], league, away_position, weeks_remaining
        )
        
        home_motivation_multiplier = self._calculate_motivation_impact(home_motivation)
        away_motivation_multiplier = self._calculate_motivation_impact(away_motivation)
        
        # Apply injury impact
        home_injuries = self.injury_engine.assess_injury_impact(self.data['home_team'], league)
        away_injuries = self.injury_engine.assess_injury_impact(self.data['away_team'], league)
        
        home_xg *= home_motivation_multiplier * home_injuries
        away_xg *= away_motivation_multiplier * away_injuries
        
        # FINAL CRITICAL STEP: Apply tier reality check with league context
        home_xg, away_xg = self.calibrator.apply_tier_reality_check(
            home_xg, away_xg, self.data['home_team'], self.data['away_team'], league
        )
        
        logger.info(f"Calibrated xG - Home: {home_xg:.3f}, Away: {away_xg:.3f}")
        logger.info(f"Team Tiers - Home: {home_tier}, Away: {away_tier}")
        logger.info(f"League: {league}")
        
        return home_xg, away_xg

    def run_monte_carlo_simulation(self, home_xg: float, away_xg: float, iterations: int = 10000) -> MonteCarloResults:
        """Run Enhanced Monte Carlo simulation with better BTTS and Over/Under"""
        np.random.seed(42)
        
        if self.match_context == MatchContext.DEFENSIVE_BATTLE:
            lambda3_alpha = self.calibration_params['bivariate_correlation'] * 0.7
        elif self.match_context == MatchContext.OFFENSIVE_SHOWDOWN:
            lambda3_alpha = self.calibration_params['bivariate_correlation'] * 1.3
        else:
            lambda3_alpha = self.calibration_params['bivariate_correlation']
        
        lambda3 = lambda3_alpha * min(home_xg, away_xg)
        lambda1 = max(0.1, home_xg - lambda3)
        lambda2 = max(0.1, away_xg - lambda3)
        
        C = np.random.poisson(lambda3, iterations)
        A = np.random.poisson(lambda1, iterations)
        B = np.random.poisson(lambda2, iterations)
        
        home_goals_sim = A + C
        away_goals_sim = B + C
        
        # Use ENHANCED calculations for BTTS and Over/Under
        home_team = self.data['home_team']
        away_team = self.data['away_team']
        league = self.data.get('league', 'premier_league')
        
        # Enhanced BTTS probability
        btts_prob = self._calculate_enhanced_btts_probability(home_xg, away_xg, home_team, away_team, league)
        
        # Enhanced Over/Under probabilities
        over_under_probs = self._calculate_enhanced_over_under(home_xg, away_xg, home_team, away_team, league)
        
        # Match outcome probabilities from simulation
        home_wins = np.sum(home_goals_sim > away_goals_sim) / iterations
        draws = np.sum(home_goals_sim == away_goals_sim) / iterations
        away_wins = np.sum(home_goals_sim < away_goals_sim) / iterations
        
        # Use enhanced values for BTTS and Over/Under
        over_25 = over_under_probs['over_25']
        
        exact_scores = {}
        for i in range(6):
            for j in range(6):
                count = np.sum((home_goals_sim == i) & (away_goals_sim == j))
                prob = count / iterations
                if prob > 0.005:
                    exact_scores[f"{i}-{j}"] = round(prob * 100, 1)
        
        exact_scores = dict(sorted(exact_scores.items(), key=lambda x: x[1], reverse=True)[:8])
        
        def calculate_ci(probs, alpha=0.95):
            se = np.sqrt(probs * (1 - probs) / iterations)
            z_score = 1.96
            return (max(0, probs - z_score * se), min(1, probs + z_score * se))
        
        confidence_intervals = {
            'home_win': calculate_ci(home_wins),
            'draw': calculate_ci(draws),
            'away_win': calculate_ci(away_wins),
            'over_2.5': calculate_ci(over_25)
        }
        
        batch_size = 1000
        num_batches = iterations // batch_size
        
        batch_probs = {
            'home_win': [], 'draw': [], 'away_win': [], 'over_2.5': []
        }
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            batch_home_wins = np.sum(home_goals_sim[start_idx:end_idx] > away_goals_sim[start_idx:end_idx]) / batch_size
            batch_draws = np.sum(home_goals_sim[start_idx:end_idx] == away_goals_sim[start_idx:end_idx]) / batch_size
            batch_away_wins = np.sum(home_goals_sim[start_idx:end_idx] < away_goals_sim[start_idx:end_idx]) / batch_size
            batch_over_25 = np.sum((home_goals_sim[start_idx:end_idx] + away_goals_sim[start_idx:end_idx]) > 2.5) / batch_size
            
            batch_probs['home_win'].append(batch_home_wins)
            batch_probs['draw'].append(batch_draws)
            batch_probs['away_win'].append(batch_away_wins)
            batch_probs['over_2.5'].append(batch_over_25)
        
        probability_volatility = {
            market: float(np.std(probs))
            for market, probs in batch_probs.items()
        }
        
        return MonteCarloResults(
            home_win_prob=home_wins,
            draw_prob=draws,
            away_win_prob=away_wins,
            over_25_prob=over_25,
            btts_prob=btts_prob,
            exact_scores=exact_scores,
            confidence_intervals=confidence_intervals,
            probability_volatility=probability_volatility
        )

    # ... [Rest of the methods remain the same as previous implementation]
    # _calculate_confidence, _assess_risk, generate_predictions, etc.

# ... [ValueDetectionEngine and AdvancedFootballPredictor remain the same]

# =============================================================================
# TEST WITH ACTUAL 2025/2026 DATA
# =============================================================================

def test_2025_season():
    """Test with actual 2025/2026 season data"""
    
    match_data = {
        'home_team': 'Arsenal',
        'away_team': 'Wolves', 
        'league': 'premier_league',
        'home_goals': 18,    # From table: Goals for
        'away_goals': 7,     # From table: Goals for
        'home_conceded': 3,  # From table: Goals against
        'away_conceded': 22, # From table: Goals against
        'home_position': 1,  # League position
        'away_position': 20, # League position
        'home_form': ['W', 'W', 'W', 'W', 'W'],  # Last 5 from table
        'away_form': ['D', 'D', 'L', 'L', 'L'],  # Last 5 from table
        'weeks_remaining': 28,
        'market_odds': {
            '1x2 Home': 1.25,   # Arsenal heavy favorite
            '1x2 Draw': 6.50,   
            '1x2 Away': 12.00,  
            'Over 2.5 Goals': 1.70,
            'Under 2.5 Goals': 2.10,
            'BTTS Yes': 2.20,
            'BTTS No': 1.65
        }
    }
    
    predictor = AdvancedFootballPredictor(match_data)
    results = predictor.generate_comprehensive_analysis()
    
    print("🎯 2025/2026 SEASON PREDICTION RESULTS")
    print("=" * 50)
    print(f"Match: {results['match']}")
    print(f"League: {results.get('league', 'premier_league')}")
    print(f"Team Tiers: {results['team_tiers']['home']} vs {results['team_tiers']['away']}")
    print(f"Expected Goals: {results['expected_goals']['home']} - {results['expected_goals']['away']}")
    print(f"Match Context: {results['match_context']}")
    print(f"Confidence: {results['confidence_score']}%")
    print()
    
    print("📊 PROBABILITIES:")
    outcomes = results['probabilities']['match_outcomes']
    print(f"Home Win: {outcomes['home_win']}%")
    print(f"Draw: {outcomes['draw']}%") 
    print(f"Away Win: {outcomes['away_win']}%")
    print()
    
    print("⚽ GOALS ANALYSIS:")
    print(f"BTTS Yes: {results['probabilities']['both_teams_score']['yes']}%")
    print(f"Over 2.5: {results['probabilities']['over_under']['over_25']}%")
    print()
    
    print("💰 VALUE SIGNALS:")
    for signal in results['betting_signals']:
        print(f"- {signal['market']}: +{signal['edge']}% edge (Stake: {signal['recommended_stake']*100:.1f}%)")
    
    print()
    print("✅ SYSTEM VALIDATION:")
    print(f"Status: {results['system_validation']['status']}")
    print(f"Alignment: {results['system_validation']['alignment']}")

if __name__ == "__main__":
    test_2025_season()
