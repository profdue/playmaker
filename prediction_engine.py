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
    UPDATED WITH 2025/2026 SEASON STANDINGS
    """
    
    def __init__(self):
        # Enhanced league-specific baselines with 2025/2026 data
        self.league_baselines = {
            'premier_league': {'avg_goals': 2.93, 'home_win_rate': 0.51, 'scoring_profile': 'high'},
            'la_liga': {'avg_goals': 2.62, 'home_win_rate': 0.48, 'scoring_profile': 'medium'},
            'serie_a': {'avg_goals': 2.56, 'home_win_rate': 0.38, 'scoring_profile': 'low'},
            'bundesliga': {'avg_goals': 3.14, 'home_win_rate': 0.42, 'scoring_profile': 'very_high'},
            'ligue_1': {'avg_goals': 2.96, 'home_win_rate': 0.52, 'scoring_profile': 'high'},
            'liga_portugal': {'avg_goals': 2.57, 'home_win_rate': 0.36, 'scoring_profile': 'medium'},
            'brasileirao': {'avg_goals': 2.44, 'home_win_rate': 0.50, 'scoring_profile': 'low'},
            'liga_mx': {'avg_goals': 2.68, 'home_win_rate': 0.46, 'scoring_profile': 'high'},
            'eredivisie': {'avg_goals': 2.98, 'home_win_rate': 0.48, 'scoring_profile': 'very_high'},
            'default': {'avg_goals': 2.70, 'home_win_rate': 0.45, 'scoring_profile': 'medium'}
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
        
        # Multi-league team databases - UPDATED WITH 2025/2026 STANDINGS
        self.team_databases = self._initialize_multi_league_teams()
    
    def _initialize_multi_league_teams(self) -> Dict[str, Dict[str, str]]:
        """Initialize comprehensive multi-league team databases for 2025/2026"""
        return {
            'premier_league': {
                # Elite (Title Contenders)
                'Arsenal': 'ELITE', 'Man City': 'ELITE', 'Liverpool': 'ELITE',
                # Strong (European Places)
                'Bournemouth': 'STRONG', 'Tottenham': 'STRONG', 'Chelsea': 'STRONG',
                'Sunderland': 'STRONG', 'Man United': 'STRONG', 'Crystal Palace': 'STRONG',
                'Brighton': 'STRONG', 'Aston Villa': 'STRONG',
                # Medium (Mid-table)
                'Brentford': 'MEDIUM', 'Newcastle': 'MEDIUM', 'Fulham': 'MEDIUM',
                'Everton': 'MEDIUM', 'Leeds': 'MEDIUM', 'Burnley': 'MEDIUM',
                # Weak (Relegation Battlers)
                'West Ham': 'WEAK', 'Nottingham Forest': 'WEAK', 'Wolves': 'WEAK'
            },
            'bundesliga': {
                # Elite
                'Bayern': 'ELITE', 'Leipzig': 'ELITE', 'Dortmund': 'ELITE',
                # Strong
                'Stuttgart': 'STRONG', 'Leverkusen': 'STRONG', 'Hoffenheim': 'STRONG',
                'Köln': 'STRONG', 'Frankfurt': 'STRONG',
                # Medium
                'Bremen': 'MEDIUM', 'Union Berlin': 'MEDIUM', 'Freiburg': 'MEDIUM',
                'Wolfsburg': 'MEDIUM', 'HSV': 'MEDIUM', 'Augsburg': 'MEDIUM',
                # Weak
                'St. Pauli': 'WEAK', 'M\'gladbach': 'WEAK', 'Mainz 05': 'WEAK', 'Heidenheim': 'WEAK'
            },
            'serie_a': {
                # Elite
                'Napoli': 'ELITE', 'Inter': 'ELITE', 'Milan': 'ELITE', 'Roma': 'ELITE',
                # Strong
                'Bologna': 'STRONG', 'Juventus': 'STRONG', 'Como': 'STRONG', 'Udinese': 'STRONG',
                'Cremonese': 'STRONG',
                # Medium
                'Atalanta': 'MEDIUM', 'Sassuolo': 'MEDIUM', 'Torino': 'MEDIUM', 'Lazio': 'MEDIUM',
                'Cagliari': 'MEDIUM', 'Lecce': 'MEDIUM',
                # Weak
                'Parma': 'WEAK', 'Pisa': 'WEAK', 'Verona': 'WEAK', 'Fiorentina': 'WEAK', 'Genoa': 'WEAK'
            },
            'la_liga': {
                # Elite
                'Real Madrid': 'ELITE', 'Barcelona': 'ELITE', 'Villarreal': 'ELITE',
                # Strong
                'Atl. Madrid': 'STRONG', 'Real Betis': 'STRONG', 'Espanyol': 'STRONG',
                'Getafe': 'STRONG', 'Alavés': 'STRONG', 'Elche': 'STRONG',
                # Medium
                'Rayo Vallecano': 'MEDIUM', 'Athletic Club': 'MEDIUM', 'Celta': 'MEDIUM',
                'Sevilla': 'MEDIUM', 'Real Sociedad': 'MEDIUM', 'Osasuna': 'MEDIUM',
                # Weak
                'Levante': 'WEAK', 'Mallorca': 'WEAK', 'Valencia': 'WEAK', 'Real Oviedo': 'WEAK', 'Girona': 'WEAK'
            },
            'ligue_1': {
                # Elite
                'PSG': 'ELITE', 'Marseille': 'STRONG', 'Lens': 'STRONG',
                # Strong
                'Lille': 'STRONG', 'AS Monaco': 'STRONG', 'Lyon': 'STRONG',
                'Strasbourg': 'STRONG', 'Nice': 'STRONG', 'Toulouse': 'STRONG',
                # Medium
                'Rennes': 'MEDIUM', 'Paris': 'MEDIUM', 'Le Havre': 'MEDIUM',
                'Brest': 'MEDIUM', 'Angers': 'MEDIUM', 'Nantes': 'MEDIUM',
                # Weak
                'Lorient': 'WEAK', 'Metz': 'WEAK', 'Auxerre': 'WEAK'
            },
            'liga_portugal': {
                # Elite
                'Porto': 'ELITE', 'Sporting': 'ELITE', 'Benfica': 'ELITE',
                # Strong
                'Gil Vicente': 'STRONG', 'Famalicão': 'STRONG', 'Moreirense': 'STRONG',
                # Medium
                'Braga': 'MEDIUM', 'Santa Clara': 'MEDIUM', 'Nacional': 'MEDIUM',
                'Rio Ave': 'MEDIUM', 'Vitória': 'MEDIUM', 'Estoril': 'MEDIUM',
                # Weak
                'Estrela Amadora': 'WEAK', 'Alverca': 'WEAK', 'Arouca': 'WEAK', 'Casa Pia': 'WEAK', 'Tondela': 'WEAK', 'AVS': 'WEAK'
            },
            'eredivisie': {
                # Elite
                'Feyenoord': 'ELITE', 'PSV': 'ELITE', 'AZ': 'ELITE',
                # Strong
                'Ajax': 'STRONG', 'Groningen': 'STRONG', 'Utrecht': 'STRONG',
                'Sparta': 'STRONG', 'NEC': 'STRONG', 'Twente': 'STRONG',
                # Medium
                'Heerenveen': 'MEDIUM', 'GA Eagles': 'MEDIUM', 'Fortuna': 'MEDIUM',
                'NAC Breda': 'MEDIUM', 'Volendam': 'MEDIUM', 'Excelsior': 'MEDIUM',
                # Weak
                'Zwolle': 'WEAK', 'Telstar': 'WEAK', 'Heracles': 'WEAK'
            },
            'brasileirao': {
                # Elite
                'Flamengo': 'ELITE', 'Palmeiras': 'ELITE', 'Cruzeiro': 'ELITE',
                # Strong
                'Mirassol': 'STRONG', 'Bahia': 'STRONG', 'Botafogo': 'STRONG',
                'Fluminense': 'STRONG', 'Vasco': 'STRONG', 'Corinthians': 'STRONG',
                # Medium
                'São Paulo': 'MEDIUM', 'Grêmio': 'MEDIUM', 'Ceará': 'MEDIUM',
                'Atlético-MG': 'MEDIUM', 'RB Bragantino': 'MEDIUM', 'Internacional': 'MEDIUM',
                # Weak
                'Santos': 'WEAK', 'Vitória': 'WEAK', 'Fortaleza': 'WEAK', 'Juventude': 'WEAK', 'Sport': 'WEAK'
            }
        }
    
    def get_league_teams(self, league: str) -> List[str]:
        """Get all teams for a specific league"""
        return list(self.team_databases.get(league, {}).keys())
    
    def get_team_tier(self, team: str, league: str) -> str:
        """Get team tier with league context"""
        league_teams = self.team_databases.get(league, {})
        return league_teams.get(team, 'MEDIUM')
    
    def get_tier_baselines(self, team: str, league: str) -> Tuple[float, float]:
        """Get baseline xG for and against for a team with league adjustment"""
        tier = self.get_team_tier(team, league)
        baselines = self.tier_calibration.get(tier, self.tier_calibration['MEDIUM'])
        
        # Adjust for league scoring profile using 2025/2026 data
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
        base_advantage = self.league_baselines.get(league, self.league_baselines['default'])['home_win_rate']
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

class SignalEngine:
    """
    RESTORED ACCURATE PREDICTIVE ENGINE - With Golden BTTS/Over-Under Logic
    """
    
    def __init__(self, match_data: Dict[str, Any], calibration_data: Optional[Dict] = None):
        self.data = self._validate_and_enhance_data(match_data)
        self.calibration_data = calibration_data or {}
        self.calibrator = TeamTierCalibrator()
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
            'home_goals': (0, 20, 1.5),
            'away_goals': (0, 20, 1.5),
            'home_conceded': (0, 20, 1.5),
            'away_conceded': (0, 20, 1.5),
            'home_clean_sheets': (0, 6, 1.0),
            'away_clean_sheets': (0, 6, 1.0),
            'home_btts_rate': (0.0, 1.0, 0.5),
            'away_btts_rate': (0.0, 1.0, 0.5),
            'home_over_25_rate': (0.0, 1.0, 0.5),
            'away_over_25_rate': (0.0, 1.0, 0.5),
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
        
        # Ensure form data is properly formatted
        for form_field in ['home_form', 'away_form']:
            if form_field in enhanced_data:
                try:
                    if isinstance(enhanced_data[form_field], list):
                        enhanced_data[form_field] = [float(x) for x in enhanced_data[form_field]]
                    else:
                        enhanced_data[form_field] = []
                except (TypeError, ValueError):
                    logger.warning(f"Invalid form data in {form_field}, using default")
                    enhanced_data[form_field] = []
        
        # Enhanced motivation handling
        if 'motivation' in enhanced_data:
            motivation = enhanced_data['motivation']
            if isinstance(motivation, dict):
                cleaned_motivation = {}
                for key in ['home', 'away']:
                    if key in motivation:
                        mot_value = motivation[key]
                        if isinstance(mot_value, str):
                            cleaned_motivation[key] = mot_value
                        elif isinstance(mot_value, (int, float)):
                            if mot_value <= 0.8:
                                cleaned_motivation[key] = "Low"
                            elif mot_value <= 1.0:
                                cleaned_motivation[key] = "Normal"
                            elif mot_value <= 1.15:
                                cleaned_motivation[key] = "High"
                            else:
                                cleaned_motivation[key] = "Very High"
                        else:
                            cleaned_motivation[key] = "Normal"
                    else:
                        cleaned_motivation[key] = "Normal"
                enhanced_data['motivation'] = cleaned_motivation
            else:
                enhanced_data['motivation'] = {'home': 'Normal', 'away': 'Normal'}
        else:
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
        if len(data.get('home_form', [])) >= 4:
            score += 10
        if len(data.get('away_form', [])) >= 4:
            score += 10
        max_score += 20
        
        # H2H data
        h2h_data = data.get('h2h_data', {})
        if h2h_data.get('matches', 0) >= 3:
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

    def _calculate_weighted_form_impact(self, team: str, team_tier: str, league: str) -> float:
        """Calculate form impact with tier-based weighting"""
        form = self.data.get(f'{team}_form', [])
        if not form or len(form) == 0:
            return 1.0
        
        try:
            form_scores = [float(score) for score in form]
            avg_form = np.mean(form_scores)
            
            # Tier-based form impact - weaker teams get more form volatility
            form_weight = {
                'ELITE': 0.3,    # 30% form, 70% baseline quality
                'STRONG': 0.4,   # 40% form, 60% baseline quality  
                'MEDIUM': 0.5,   # 50% form, 50% baseline quality
                'WEAK': 0.6      # 60% form, 40% baseline quality
            }.get(team_tier, 0.5)
            
            form_ratio = avg_form / 3.0
            form_impact = 0.8 + (form_ratio * 0.4)
            
            # Blend form impact with baseline (1.0)
            weighted_impact = (form_impact * form_weight) + (1.0 * (1 - form_weight))
            
            return max(0.7, min(1.3, weighted_impact))
            
        except (TypeError, ValueError):
            return 1.0

    def _calculate_enhanced_btts_probability(self, home_xg: float, away_xg: float, home_team: str, away_team: str, league: str) -> float:
        """RESTORED GOLDEN BTTS LOGIC - Simple & Accurate"""
        
        # Base BTTS from independent probabilities (THE GOLDEN FORMULA)
        home_score_prob = 1 - math.exp(-home_xg)
        away_score_prob = 1 - math.exp(-away_xg)
        base_btts = home_score_prob * away_score_prob
        
        # Get historical BTTS rates
        home_btts_rate = self.data.get('home_btts_rate', 0.5)
        away_btts_rate = self.data.get('away_btts_rate', 0.5)
        historical_factor = (home_btts_rate + away_btts_rate) / 2
        
        # CRITICAL FIX: Blend base probability with historical data
        # 70% weight to mathematical probability, 30% to historical trend
        adjusted_btts = (base_btts * 0.7) + (historical_factor * 0.3)
        
        # Minimal tier-based adjustment (max ±10%)
        home_tier = self.calibrator.get_team_tier(home_team, league)
        away_tier = self.calibrator.get_team_tier(away_team, league)
        
        if home_tier == 'ELITE' and away_tier == 'ELITE':
            adjusted_btts *= 0.9   # Slight reduction for top defenses
        elif home_tier == 'WEAK' and away_tier == 'WEAK':
            adjusted_btts *= 1.1   # Slight increase for weak defenses
        
        return max(0.20, min(0.80, adjusted_btts))

    def _calculate_enhanced_over_under(self, home_xg: float, away_xg: float, home_team: str, away_team: str, league: str) -> Dict[str, float]:
        """RESTORED ACCURATE OVER/UNDER LOGIC - Monte Carlo Based"""
        
        total_xg = home_xg + away_xg
        
        # USE POISSON DIRECTLY (proven accurate)
        over_25 = 1 - poisson.cdf(2, total_xg)
        under_25 = poisson.cdf(2, total_xg)
        
        # Get historical Over rates
        home_over_rate = self.data.get('home_over_25_rate', 0.5)
        away_over_rate = self.data.get('away_over_25_rate', 0.5)
        historical_factor = (home_over_rate + away_over_rate) / 2
        
        # FIX: Blend mathematical probability with historical data
        over_25 = (over_25 * 0.8) + (historical_factor * 0.2)
        under_25 = (under_25 * 0.7) + ((1 - historical_factor) * 0.3)
        
        # Normalize
        total = over_25 + under_25
        if total > 0:
            over_25 /= total
            under_25 /= total
        
        return {
            'over_25': max(0.1, min(0.9, over_25)),
            'under_25': max(0.1, min(0.9, under_25))
        }

    def _calculate_realistic_xg(self) -> Tuple[float, float]:
        """Calculate REALISTIC predictive xG with multi-league tier calibration"""
        league = self.data.get('league', 'premier_league')
        
        # Get tier-based baselines with league context
        home_baseline, home_def_baseline = self.calibrator.get_tier_baselines(self.data['home_team'], league)
        away_baseline, away_def_baseline = self.calibrator.get_tier_baselines(self.data['away_team'], league)
        
        # Start from tier baselines instead of raw data
        home_goals_avg = self.data.get('home_goals', 0) / 6.0
        away_goals_avg = self.data.get('away_goals', 0) / 6.0
        
        # Blend recent performance with tier baseline
        home_attack = (home_goals_avg * 0.4) + (home_baseline * 0.6)
        away_attack = (away_goals_avg * 0.4) + (away_baseline * 0.6)
        
        home_conceded_avg = self.data.get('home_conceded', 0) / 6.0
        away_conceded_avg = self.data.get('away_conceded', 0) / 6.0
        
        # Blend defensive performance with tier baseline
        home_defense = (home_conceded_avg * 0.4) + (home_def_baseline * 0.6)
        away_defense = (away_conceded_avg * 0.4) + (away_def_baseline * 0.6)
        
        # Calculate xG using blended values
        league_avg = self.calibrator.league_baselines.get(league, self.calibrator.league_baselines['default'])['avg_goals']
        
        home_xg = home_attack * (1 - (away_defense / league_avg) * self.calibration_params['defensive_impact_multiplier'])
        away_xg = away_attack * (1 - (home_defense / league_avg) * self.calibration_params['defensive_impact_multiplier'])
        
        # Apply contextual home advantage with league context
        home_advantage = self.calibrator.get_contextual_home_advantage(
            self.data['home_team'], self.data['away_team'], league
        )
        home_xg *= (1 + home_advantage)
        
        # Apply weighted form impact
        home_tier = self.calibrator.get_team_tier(self.data['home_team'], league)
        away_tier = self.calibrator.get_team_tier(self.data['away_team'], league)
        
        home_form = self._calculate_weighted_form_impact('home', home_tier, league)
        away_form = self._calculate_weighted_form_impact('away', away_tier, league)
        
        home_xg *= home_form
        away_xg *= away_form
        
        # Apply motivation and injuries
        motivation = self.data.get('motivation', {})
        home_motivation = self._calculate_motivation_impact(motivation.get('home', 'Normal'))
        away_motivation = self._calculate_motivation_impact(motivation.get('away', 'Normal'))
        
        injuries = self.data.get('injuries', {})
        home_injuries = max(0.7, 1.0 - (float(injuries.get('home', 0)) * self.calibration_params['injury_impact']))
        away_injuries = max(0.7, 1.0 - (float(injuries.get('away', 0)) * self.calibration_params['injury_impact']))
        
        home_xg *= home_motivation * home_injuries
        away_xg *= away_motivation * away_injuries
        
        # Apply H2H adjustment
        h2h_data = self.data.get('h2h_data', {})
        if h2h_data and h2h_data.get('matches', 0) >= 3:
            home_xg, away_xg = self._apply_h2h_adjustment(home_xg, away_xg, h2h_data)
        
        # FINAL CRITICAL STEP: Apply tier reality check with league context
        home_xg, away_xg = self.calibrator.apply_tier_reality_check(
            home_xg, away_xg, self.data['home_team'], self.data['away_team'], league
        )
        
        logger.info(f"Calibrated xG - Home: {home_xg:.3f}, Away: {away_xg:.3f}")
        logger.info(f"Team Tiers - Home: {home_tier}, Away: {away_tier}")
        logger.info(f"League: {league}")
        
        return home_xg, away_xg
    
    def _apply_h2h_adjustment(self, home_xg: float, away_xg: float, h2h_data: Dict) -> Tuple[float, float]:
        """Apply H2H adjustment"""
        matches = h2h_data.get('matches', 0)
        home_goals = h2h_data.get('home_goals', 0)
        away_goals = h2h_data.get('away_goals', 0)
        
        if matches < 3:
            return home_xg, away_xg
        
        h2h_weight = min(0.25, matches * 0.06)
        h2h_home_avg = home_goals / matches
        h2h_away_avg = away_goals / matches
        
        adjusted_home_xg = (home_xg * (1 - h2h_weight)) + (h2h_home_avg * h2h_weight)
        adjusted_away_xg = (away_xg * (1 - h2h_weight)) + (h2h_away_avg * h2h_weight)
        
        return adjusted_home_xg, adjusted_away_xg

    def run_monte_carlo_simulation(self, home_xg: float, away_xg: float, iterations: int = 10000) -> MonteCarloResults:
        """Run Enhanced Monte Carlo simulation with RESTORED BTTS and Over/Under"""
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
        
        # Use RESTORED calculations for BTTS and Over/Under
        home_team = self.data['home_team']
        away_team = self.data['away_team']
        league = self.data.get('league', 'premier_league')
        
        # Enhanced BTTS probability
        btts_prob = self._calculate_enhanced_btts_probability(home_xg, away_xg, home_team, away_team, league)
        
        # Enhanced Over/Under probabilities
        over_under_probs = self._calculate_enhanced_over_under(home_xg, away_xg, home_team, away_team, league)
        
        # Match outcomes from simulation
        home_wins = np.sum(home_goals_sim > away_goals_sim) / iterations
        draws = np.sum(home_goals_sim == away_goals_sim) / iterations
        away_wins = np.sum(home_goals_sim < away_goals_sim) / iterations
        
        # For Over 2.5, use BOTH Monte Carlo AND enhanced logic
        mc_over_25 = np.sum((home_goals_sim + away_goals_sim) > 2.5) / iterations
        enhanced_over_25 = over_under_probs['over_25']
        
        # BLEND them for best accuracy (70% Monte Carlo, 30% enhanced)
        final_over_25 = (mc_over_25 * 0.7) + (enhanced_over_25 * 0.3)
        
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
            'over_2.5': calculate_ci(final_over_25)
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
            over_25_prob=final_over_25,
            btts_prob=btts_prob,
            exact_scores=exact_scores,
            confidence_intervals=confidence_intervals,
            probability_volatility=probability_volatility
        )

    def _calculate_confidence(self, mc_results: MonteCarloResults, home_team: str, away_team: str, league: str) -> int:
        """Calculate confidence score"""
        base_confidence = self.data['data_quality_score'] * 0.7
        
        if len(self.data.get('home_form', [])) >= 5:
            base_confidence += 8
        if len(self.data.get('away_form', [])) >= 5:
            base_confidence += 8
            
        h2h_data = self.data.get('h2h_data', {})
        if h2h_data.get('matches', 0) >= 5:
            base_confidence += 10
        
        avg_volatility = np.mean(list(mc_results.probability_volatility.values()))
        volatility_penalty = min(25, int(avg_volatility * 500))
        
        if self.match_context == MatchContext.UNPREDICTABLE:
            base_confidence -= 12
        
        confidence = base_confidence - volatility_penalty
        return max(20, min(90, int(confidence)))
    
    def _assess_risk(self, mc_results: MonteCarloResults, confidence: int, home_xg: float, away_xg: float, home_team: str, away_team: str, league: str) -> Dict[str, str]:
        """Realistic risk assessment with tier awareness"""
        home_win_prob = mc_results.home_win_prob
        home_tier = self.calibrator.get_team_tier(home_team, league)
        away_tier = self.calibrator.get_team_tier(away_team, league)
        
        # Tier-aware risk assessment
        if (home_tier == 'ELITE' and away_tier == 'WEAK' and
            home_win_prob > 0.75 and confidence > 75 and
            self.data['data_quality_score'] > 75):
            risk_level = "MEDIUM"
            explanation = "Strong home favorite with good data quality"
            recommendation = "CONSIDER SMALL STAKE"
        elif (home_win_prob > 0.65 and confidence > 70 and
              self.data['data_quality_score'] > 70):
            risk_level = "MEDIUM-HIGH" 
            explanation = "Reasonable probability but some uncertainty"
            recommendation = "TINY STAKE ONLY"
        else:
            risk_level = "HIGH"
            explanation = f"High uncertainty - {self.match_context.value.replace('_', ' ')}"
            recommendation = "AVOID OR MINIMAL STAKE"
        
        return {
            'risk_level': risk_level,
            'explanation': explanation,
            'recommendation': recommendation,
            'certainty': f"{home_win_prob*100:.1f}%",
            'home_advantage': f"{home_xg - away_xg:.2f} goals"
        }

    def generate_predictions(self, mc_iterations: int = 10000) -> Dict[str, Any]:
        """Generate ACCURATE football predictions with restored BTTS/Over-Under logic"""
        
        home_team = self.data['home_team']
        away_team = self.data['away_team']
        league = self.data.get('league', 'premier_league')
        
        home_xg, away_xg = self._calculate_realistic_xg()
        
        self.match_context = self._determine_match_context(home_xg, away_xg, home_team, away_team, league)
        
        mc_results = self.run_monte_carlo_simulation(home_xg, away_xg, mc_iterations)
        
        # CRITICAL: Validate probabilities with tier-based sanity check
        if not self.calibrator.validate_probability_sanity(
            mc_results.home_win_prob, mc_results.draw_prob, mc_results.away_win_prob,
            home_team, away_team, league
        ):
            logger.warning("Probability sanity check failed - applying tier-based corrections")
            
            home_tier = self.calibrator.get_team_tier(home_team, league)
            away_tier = self.calibrator.get_team_tier(away_team, league)
            
            # Apply tier-based probability corrections
            if home_tier == 'ELITE' and away_tier == 'WEAK':
                mc_results.home_win_prob = max(mc_results.home_win_prob, 0.65)
                mc_results.away_win_prob = min(mc_results.away_win_prob, 0.15)
            elif away_tier == 'ELITE' and home_tier == 'WEAK':
                mc_results.away_win_prob = max(mc_results.away_win_prob, 0.65)
                mc_results.home_win_prob = min(mc_results.home_win_prob, 0.15)
            
            # Normalize probabilities
            total = mc_results.home_win_prob + mc_results.draw_prob + mc_results.away_win_prob
            mc_results.home_win_prob /= total
            mc_results.draw_prob /= total  
            mc_results.away_win_prob /= total
        
        total_xg = home_xg + away_xg
        first_half_prob = 1 - poisson.pmf(0, total_xg * 0.46)
        second_half_prob = 1 - poisson.pmf(0, total_xg * 0.54)
        
        handicap_probs = {}
        handicaps = [-1.5, -1.0, -0.5, 0, 0.5]
        for handicap in handicaps:
            if handicap == 0:
                prob = 1 - skellam.cdf(0, home_xg, away_xg)
            elif handicap > 0:
                prob = 1 - skellam.cdf(-handicap, home_xg, away_xg)
            else:
                prob = 1 - skellam.cdf(abs(handicap), home_xg, away_xg)
            handicap_probs[f"handicap_{handicap}"] = round(prob * 100, 1)
        
        base_corners = 9.5
        attacking_bonus = (home_xg + away_xg - 2.7) * 0.8
        total_corners = max(6, min(14, base_corners + attacking_bonus))
        
        confidence_score = self._calculate_confidence(mc_results, home_team, away_team, league)
        risk_assessment = self._assess_risk(mc_results, confidence_score, home_xg, away_xg, home_team, away_team, league)
        
        predictions = {
            'match': f"{home_team} vs {away_team}",
            'league': league,
            'expected_goals': {'home': round(home_xg, 2), 'away': round(away_xg, 2)},
            'team_tiers': {
                'home': self.calibrator.get_team_tier(home_team, league),
                'away': self.calibrator.get_team_tier(away_team, league)
            },
            'match_context': self.match_context.value,
            'data_quality_score': round(self.data['data_quality_score'], 1),
            'probabilities': {
                'match_outcomes': {
                    'home_win': round(mc_results.home_win_prob * 100, 1),
                    'draw': round(mc_results.draw_prob * 100, 1),
                    'away_win': round(mc_results.away_win_prob * 100, 1)
                },
                'both_teams_score': {
                    'yes': round(mc_results.btts_prob * 100, 1),
                    'no': round((1 - mc_results.btts_prob) * 100, 1)
                },
                'over_under': {
                    'over_15': round((1 - poisson.cdf(1, total_xg)) * 100, 1),
                    'over_25': round(mc_results.over_25_prob * 100, 1),
                    'over_35': round((1 - poisson.cdf(3, total_xg)) * 100, 1),
                    'under_15': round(poisson.cdf(1, total_xg) * 100, 1),
                    'under_25': round((1 - mc_results.over_25_prob) * 100, 1),
                    'under_35': round(poisson.cdf(3, total_xg) * 100, 1)
                },
                'exact_scores': mc_results.exact_scores,
                'goal_timing': {
                    'first_half': round(first_half_prob * 100, 1),
                    'second_half': round(second_half_prob * 100, 1)
                }
            },
            'handicap_probabilities': handicap_probs,
            'corner_predictions': {
                'total': f"{int(total_corners)}-{int(total_corners + 1)}",
                'home': f"{int(total_corners * 0.55)}-{int(total_corners * 0.55 + 0.5)}",
                'away': f"{int(total_corners * 0.45)}-{int(total_corners * 0.45 + 0.5)}"
            },
            'timing_predictions': self._generate_timing_predictions(home_xg, away_xg),
            'summary': self._generate_summary(home_team, away_team, home_xg, away_xg, mc_results, league),
            'confidence_score': confidence_score,
            'risk_assessment': risk_assessment,
            'monte_carlo_results': {
                'confidence_intervals': mc_results.confidence_intervals,
                'probability_volatility': mc_results.probability_volatility
            }
        }
        
        return predictions

    def _generate_timing_predictions(self, home_xg: float, away_xg: float) -> Dict[str, str]:
        """Generate timing predictions"""
        total_xg = home_xg + away_xg
        
        if total_xg < 2.0:
            first_goal = "35+ minutes"
            late_goals = "UNLIKELY"
        elif total_xg < 2.8:
            first_goal = "25-35 minutes" 
            late_goals = "POSSIBLE"
        else:
            first_goal = "20-30 minutes"
            late_goals = "LIKELY"
        
        return {
            'first_goal': first_goal,
            'late_goals': late_goals,
            'most_action': "Second half" if total_xg > 2.5 else "Evenly distributed"
        }
    
    def _generate_summary(self, home_team: str, away_team: str, home_xg: float, 
                        away_xg: float, mc_results: MonteCarloResults, league: str) -> str:
        """Generate realistic football summary with tier awareness"""
        
        home_win_prob = mc_results.home_win_prob
        home_tier = self.calibrator.get_team_tier(home_team, league)
        away_tier = self.calibrator.get_team_tier(away_team, league)
        
        if home_tier == 'ELITE' and away_tier == 'WEAK' and home_win_prob > 0.70:
            return f"{home_team} are overwhelming favorites with {home_xg:.1f} expected goals. Their superior quality and home advantage should see them comfortably overcome {away_team}."
        
        elif away_tier == 'ELITE' and home_tier == 'WEAK' and home_win_prob < 0.25:
            return f"{away_team} demonstrate clear superiority with {away_xg:.1f} expected goals. Their quality should overcome {home_team}'s home advantage."
        
        elif self.match_context == MatchContext.HOME_DOMINANCE and home_win_prob > 0.60:
            return f"{home_team} hold clear advantage with {home_xg:.1f} expected goals. Home advantage and better metrics point towards a victory against {away_team}."
        
        elif self.match_context == MatchContext.UNPREDICTABLE:
            return f"Unpredictable encounter expected between {home_team} and {away_team}. Both teams show variable form, making this match difficult to forecast with confidence."
        
        elif self.match_context == MatchContext.DEFENSIVE_BATTLE:
            return f"Defensive battle anticipated between {home_team} and {away_team}. Low expected goals ({home_xg:.1f} - {away_xg:.1f}) suggest a tight, cagey affair."
        
        elif home_win_prob > 0.55 and home_xg > away_xg + 0.4:
            return f"{home_team} hold the advantage with better expected goal metrics. {away_team} will need a disciplined performance to get a result."
        
        else:
            return f"Competitive match expected with small margins likely deciding the outcome between {home_team} and {away_team}."


class ValueDetectionEngine:
    """
    PERFECTLY ALIGNED VALUE DETECTION ENGINE - MULTI-LEAGUE SUPPORT
    """
    
    def __init__(self):
        # Realistic thresholds
        self.value_thresholds = {
            'EXCEPTIONAL': 25.0,
            'HIGH': 15.0,  
            'GOOD': 8.0,
            'MODERATE': 4.0,
        }
        
        # CRITICAL: Minimum probabilities for value consideration
        self.min_probability_thresholds = {
            '1x2 Home': 20.0,
            '1x2 Draw': 25.0,  
            '1x2 Away': 15.0,
            'Over 2.5 Goals': 35.0,
            'Under 2.5 Goals': 35.0,
            'BTTS Yes': 40.0,
            'BTTS No': 40.0,
        }
        
        self.min_confidence = 60
        self.min_probability = 0.12
        self.max_stake = 0.03
        
        # Multi-league team strength tiers
        self.calibrator = TeamTierCalibrator()

    def calculate_implied_probabilities(self, market_odds: Dict[str, float]) -> Dict[str, float]:
        """Convert decimal odds to implied probabilities"""
        implied_probs = {}
        
        for market, odds in market_odds.items():
            if isinstance(odds, (int, float)) and odds > 1:
                implied_prob = 1.0 / odds
                
                if 0.03 <= implied_prob <= 0.85:
                    implied_probs[market] = implied_prob
                else:
                    logger.warning(f"Implied probability {implied_prob:.3f} for {market} outside reasonable range")
                    implied_probs[market] = 0.0
        
        return implied_probs
    
    def _get_primary_prediction(self, pure_probabilities: Dict) -> Dict:
        """Get the primary prediction from Signal Engine"""
        outcomes = pure_probabilities['probabilities']['match_outcomes']
        btts = pure_probabilities['probabilities']['both_teams_score']
        over_under = pure_probabilities['probabilities']['over_under']
        
        # Determine primary match outcome
        home_win = outcomes['home_win']
        draw = outcomes['draw']
        away_win = outcomes['away_win']
        
        primary_outcome = None
        if home_win >= draw and home_win >= away_win:
            primary_outcome = 'HOME'
        elif draw >= home_win and draw >= away_win:
            primary_outcome = 'DRAW'
        else:
            primary_outcome = 'AWAY'
        
        # Determine primary BTTS
        primary_btts = 'YES' if btts['yes'] > btts['no'] else 'NO'
        
        # Determine primary Over/Under
        primary_over_under = 'OVER' if over_under['over_25'] > over_under['under_25'] else 'UNDER'
        
        return {
            'outcome': primary_outcome,
            'btts': primary_btts,
            'over_under': primary_over_under,
            'match_context': pure_probabilities['match_context'],
            'team_tiers': pure_probabilities.get('team_tiers', {}),
            'league': pure_probabilities.get('league', 'premier_league')
        }
    
    def _is_contradictory_signal(self, signal: BettingSignal, primary_prediction: Dict) -> bool:
        """Check if a signal contradicts the primary prediction with tier awareness"""
        
        home_tier = primary_prediction.get('team_tiers', {}).get('home', 'MEDIUM')
        away_tier = primary_prediction.get('team_tiers', {}).get('away', 'MEDIUM')
        league = primary_prediction.get('league', 'premier_league')
        
        # NEVER allow contradictory outcomes in tier mismatches
        if (home_tier == 'ELITE' and away_tier == 'WEAK' and 
            primary_prediction['outcome'] == 'HOME'):
            
            if signal.market in ['1x2 Draw', '1x2 Away']:
                return True
        
        if (away_tier == 'ELITE' and home_tier == 'WEAK' and 
            primary_prediction['outcome'] == 'AWAY'):
            
            if signal.market in ['1x2 Draw', '1x2 Home']:
                return True
        
        # NEVER allow BTTS Yes when primary says No (and vice versa)
        if (signal.market == 'BTTS Yes' and primary_prediction['btts'] == 'NO') or \
           (signal.market == 'BTTS No' and primary_prediction['btts'] == 'YES'):
            return True
        
        # NEVER allow Over/Under contradictions when confidence is high
        if (signal.market == 'Over 2.5 Goals' and primary_prediction['over_under'] == 'UNDER' and 
            abs(primary_prediction.get('over_under_confidence', 0)) > 15):
            return True
            
        if (signal.market == 'Under 2.5 Goals' and primary_prediction['over_under'] == 'OVER' and 
            abs(primary_prediction.get('over_under_confidence', 0)) > 15):
            return True
        
        return False

    def detect_value_bets(self, pure_probabilities: Dict, market_odds: Dict) -> List[BettingSignal]:
        """PERFECTLY ALIGNED value bet detection - NO CONTRADICTIONS"""
        
        signals = []
        
        market_probs = self.calculate_implied_probabilities(market_odds)
        
        # Get the primary prediction from Signal Engine
        primary_prediction = self._get_primary_prediction(pure_probabilities)
        
        # Get probabilities
        home_pure = pure_probabilities['probabilities']['match_outcomes']['home_win'] / 100.0
        draw_pure = pure_probabilities['probabilities']['match_outcomes']['draw'] / 100.0  
        away_pure = pure_probabilities['probabilities']['match_outcomes']['away_win'] / 100.0
        
        # Normalize to ensure they sum to 1.0
        total = home_pure + draw_pure + away_pure
        if total > 0:
            home_pure /= total
            draw_pure /= total
            away_pure /= total
        
        # Market mapping
        probability_mapping = [
            ('1x2 Home', home_pure, '1x2 Home'),
            ('1x2 Draw', draw_pure, '1x2 Draw'), 
            ('1x2 Away', away_pure, '1x2 Away'),
            ('Over 2.5 Goals', pure_probabilities['probabilities']['over_under']['over_25'] / 100.0, 'Over 2.5 Goals'),
            ('Under 2.5 Goals', pure_probabilities['probabilities']['over_under']['under_25'] / 100.0, 'Under 2.5 Goals'),
            ('BTTS Yes', pure_probabilities['probabilities']['both_teams_score']['yes'] / 100.0, 'BTTS Yes'),
            ('BTTS No', pure_probabilities['probabilities']['both_teams_score']['no'] / 100.0, 'BTTS No')
        ]
        
        confidence_score = pure_probabilities.get('confidence_score', 0)
        if confidence_score < self.min_confidence:
            logger.info(f"Confidence score {confidence_score} below minimum {self.min_confidence}")
            return []
        
        home_team = pure_probabilities['match'].split(' vs ')[0]
        away_team = pure_probabilities['match'].split(' vs ')[1]
        league = pure_probabilities.get('league', 'premier_league')
        
        for market_name, pure_prob, market_key in probability_mapping:
            market_prob = market_probs.get(market_key, 0)
            
            # Skip if invalid probabilities
            if market_prob <= 0.02 or pure_prob <= 0 or pure_prob < self.min_probability:
                continue
                
            # Edge calculation
            edge = (pure_prob / market_prob) - 1.0
            edge_percentage = edge * 100
            
            # Market wisdom adjustments
            adjusted_edge = self._apply_market_wisdom(edge_percentage, market_prob, pure_prob)
            
            # Only consider positive edges above minimum threshold
            if adjusted_edge >= self.value_thresholds['MODERATE']:
                value_rating = self._get_value_rating(adjusted_edge)
                confidence = self._calculate_bet_confidence(pure_prob, adjusted_edge, confidence_score)
                stake = self._calculate_stake(pure_prob, market_prob, adjusted_edge, confidence_score)
                
                # Create signal
                signal = BettingSignal(
                    market=market_name,
                    model_prob=round(pure_prob * 100, 1),
                    book_prob=round(market_prob * 100, 1),
                    edge=round(adjusted_edge, 1),
                    confidence=confidence,
                    recommended_stake=stake,
                    value_rating=value_rating
                )
                
                # CRITICAL: REJECT CONTRADICTORY SIGNALS
                if self._is_contradictory_signal(signal, primary_prediction):
                    logger.info(f"REJECTED contradictory signal: {signal.market}")
                    continue
                
                # Apply minimum probability thresholds
                min_threshold = self.min_probability_thresholds.get(signal.market, 15.0)
                if signal.model_prob < min_threshold:
                    logger.info(f"Below probability threshold: {signal.market} ({signal.model_prob}% < {min_threshold}%)")
                    continue
                
                # Only include if stake is reasonable and value rating is at least MODERATE
                if (stake > 0.001 and stake <= self.max_stake and 
                    signal.value_rating in ["MODERATE", "GOOD", "HIGH", "EXCEPTIONAL"]):
                    signals.append(signal)
        
        # Sort by edge and return
        signals.sort(key=lambda x: x.edge, reverse=True)
        return signals
    
    def _apply_market_wisdom(self, edge: float, market_prob: float, pure_prob: float) -> float:
        """Market efficiency adjustments"""
        if market_prob < 0.12 or market_prob > 0.85:
            return edge * 0.6
        
        if market_prob < 0.20 or market_prob > 0.75:
            return edge * 0.75
        
        return edge
    
    def _get_value_rating(self, edge: float) -> str:
        """Get value rating based on edge percentage"""
        for rating, threshold in self.value_thresholds.items():
            if edge >= threshold:
                return rating
        return "LOW"
    
    def _calculate_bet_confidence(self, pure_prob: float, edge: float, model_confidence: int) -> str:
        """Calculate betting confidence"""
        if pure_prob > 0.65 and edge > 20 and model_confidence > 75:
            return "HIGH"
        elif pure_prob > 0.55 and edge > 12 and model_confidence > 70:
            return "MEDIUM-HIGH" 
        elif pure_prob > 0.45 and edge > 8 and model_confidence > 65:
            return "MEDIUM"
        elif pure_prob > 0.35 and edge > 5 and model_confidence > 60:
            return "LOW"
        else:
            return "SPECULATIVE"
    
    def _calculate_stake(self, pure_prob: float, market_prob: float, edge: float, 
                       confidence: int, kelly_fraction: float = 0.15) -> float:
        """Professional stake sizing"""
        if market_prob <= 0 or pure_prob <= market_prob:
            return 0.0
        
        decimal_odds = 1 / market_prob
        kelly_stake = (pure_prob * decimal_odds - 1) / (decimal_odds - 1)
        
        confidence_factor = max(0.4, confidence / 100)
        
        base_stake = kelly_stake * kelly_fraction * confidence_factor
        edge_cap = min(self.max_stake, edge / 1000)
        
        final_stake = min(base_stake, edge_cap)
        
        return max(0.001, min(self.max_stake, final_stake))


class AdvancedFootballPredictor:
    """
    PERFECTLY ALIGNED ORCHESTRATOR: Multi-League Support
    """
    
    def __init__(self, match_data: Dict[str, Any], calibration_data: Optional[Dict] = None):
        self.market_odds = match_data.get('market_odds', {})
        
        football_data = match_data.copy()
        if 'market_odds' in football_data:
            del football_data['market_odds']
        
        self.signal_engine = SignalEngine(football_data, calibration_data)
        self.value_engine = ValueDetectionEngine()
        self.prediction_history = []
    
    def generate_comprehensive_analysis(self, mc_iterations: int = 10000) -> Dict[str, Any]:
        """Generate comprehensive analysis with PERFECT alignment"""
        
        football_predictions = self.signal_engine.generate_predictions(mc_iterations)
        
        value_signals = self.value_engine.detect_value_bets(football_predictions, self.market_odds)
        
        comprehensive_result = football_predictions.copy()
        comprehensive_result['betting_signals'] = [signal.__dict__ for signal in value_signals]
        
        comprehensive_result['system_validation'] = self._validate_system_output(football_predictions, value_signals)
        
        self._store_prediction_history(football_predictions)
        
        return comprehensive_result
    
    def _validate_system_output(self, football_predictions: Dict, value_signals: List[BettingSignal]) -> Dict[str, str]:
        """Validate system output for perfect alignment"""
        
        issues = []
        
        # Get primary predictions
        outcomes = football_predictions['probabilities']['match_outcomes']
        btts = football_predictions['probabilities']['both_teams_score']
        over_under = football_predictions['probabilities']['over_under']
        team_tiers = football_predictions.get('team_tiers', {})
        league = football_predictions.get('league', 'premier_league')
        
        primary_outcome = max(outcomes, key=outcomes.get)
        primary_btts = 'yes' if btts['yes'] > btts['no'] else 'no'
        primary_over_under = 'over_25' if over_under['over_25'] > over_under['under_25'] else 'under_25'
        
        # Check for any contradictory value signals
        for signal in value_signals:
            signal_dict = signal.__dict__ if hasattr(signal, '__dict__') else signal
            
            # Check outcome contradictions with tier awareness
            home_tier = team_tiers.get('home', 'MEDIUM')
            away_tier = team_tiers.get('away', 'MEDIUM')
            
            if (signal_dict['market'] == '1x2 Draw' and primary_outcome == 'home_win' and 
                home_tier == 'ELITE' and away_tier == 'WEAK' and outcomes['home_win'] > 65):
                issues.append("Draw value bet contradicts strong home win prediction for elite vs weak")
            
            if (signal_dict['market'] == '1x2 Away' and primary_outcome == 'home_win' and 
                home_tier == 'ELITE' and away_tier == 'WEAK' and outcomes['home_win'] > 65):
                issues.append("Away win value bet contradicts strong home win prediction for elite vs weak")
            
            # Check BTTS contradictions
            if (signal_dict['market'] == 'BTTS Yes' and primary_btts == 'no' and 
                btts['no'] > 60):
                issues.append("BTTS Yes value bet contradicts BTTS No prediction")
            
            if (signal_dict['market'] == 'BTTS No' and primary_btts == 'yes' and 
                btts['yes'] > 60):
                issues.append("BTTS No value bet contradicts BTTS Yes prediction")
        
        if not issues:
            return {'status': 'VALID', 'issues': 'None', 'alignment': 'PERFECT'}
        else:
            return {'status': 'INVALID', 'issues': '; '.join(issues), 'alignment': 'CONTRADICTORY'}
    
    def _store_prediction_history(self, prediction: Dict):
        """Store prediction for historical tracking"""
        prediction_record = {
            'timestamp': datetime.now().isoformat(),
            'match': prediction['match'],
            'league': prediction.get('league', 'premier_league'),
            'expected_goals': prediction['expected_goals'],
            'team_tiers': prediction.get('team_tiers', {}),
            'probabilities': prediction['probabilities']['match_outcomes'],
            'match_context': prediction['match_context'],
            'confidence_score': prediction['confidence_score'],
            'data_quality': prediction['data_quality_score']
        }
        self.prediction_history.append(prediction_record)
        
        if len(self.prediction_history) > 100:
            self.prediction_history = self.prediction_history[-100:]
    
    def get_prediction_history(self) -> List[Dict]:
        """Get prediction history for analysis"""
        return self.prediction_history


# =============================================================================
# TEST THE RESTORED CALIBRATION
# =============================================================================

def test_restored_calibration():
    """Test the restored calibration system with updated teams"""
    
    match_data = {
        'home_team': 'West Ham',  # Weak team for 2025/2026
        'away_team': 'Arsenal',   # Elite team
        'league': 'premier_league',
        'home_goals': 8,    # Last 6 games
        'away_goals': 12,   # Last 6 games
        'home_conceded': 14,
        'away_conceded': 6,
        'home_clean_sheets': 1,    # Enhanced BTTS data
        'away_clean_sheets': 3,    # Enhanced BTTS data
        'home_btts_rate': 0.67,    # BTTS in last 6 games
        'away_btts_rate': 0.50,    # BTTS in last 6 games
        'home_over_25_rate': 0.83, # Over 2.5 in last 6 games
        'away_over_25_rate': 0.67, # Over 2.5 in last 6 games
        'home_form': [1, 0, 1, 0, 0, 1],  # Recent form points
        'away_form': [3, 3, 1, 3, 3, 3],  # Recent form points
        'h2h_data': {
            'matches': 5,
            'home_wins': 1,
            'away_wins': 3, 
            'draws': 1,
            'home_goals': 4,
            'away_goals': 9
        },
        'injuries': {
            'home': 1,  # Minor absences
            'away': 2   # Regular starters out
        },
        'motivation': {
            'home': 'High',    # Fighting relegation
            'away': 'Normal'   # Title challenge
        },
        'market_odds': {
            '1x2 Home': 6.50,   # ~15.4% implied
            '1x2 Draw': 4.50,   # ~22.2% implied  
            '1x2 Away': 1.50,   # ~66.7% implied
            'Over 2.5 Goals': 1.80,
            'Under 2.5 Goals': 2.00,
            'BTTS Yes': 1.90,
            'BTTS No': 1.90
        }
    }
    
    predictor = AdvancedFootballPredictor(match_data)
    results = predictor.generate_comprehensive_analysis()
    
    print("🎯 RESTORED CALIBRATED PREDICTION RESULTS")
    print("=" * 50)
    print(f"Match: {results['match']}")
    print(f"League: {results.get('league', 'premier_league')}")
    print(f"Team Tiers: {results['team_tiers']['home']} vs {results['team_tiers']['away']}")
    print(f"Expected Goals: {results['expected_goals']['home']} - {results['expected_goals']['away']}")
    print(f"Match Context: {results['match_context']}")
    print(f"Confidence: {results['confidence_score']}%")
    print()
    
    print("📊 RESTORED PROBABILITIES:")
    outcomes = results['probabilities']['match_outcomes']
    print(f"Home Win: {outcomes['home_win']}%")
    print(f"Draw: {outcomes['draw']}%") 
    print(f"Away Win: {outcomes['away_win']}%")
    print()
    
    print("⚽ RESTORED GOALS ANALYSIS:")
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
    test_restored_calibration()
