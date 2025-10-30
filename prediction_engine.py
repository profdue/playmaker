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
    UNKNOWN = "unknown"

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

class AdvancedPredictionEngine:
    """Enhanced Football Prediction Engine with Context-Aware Bivariate Poisson"""
    
    def __init__(self, match_data: Dict[str, Any], calibration_data: Optional[Dict] = None, mc_iterations: int = 10000):
        self.data = self._validate_and_clean_data(match_data)
        self.calibration_data = calibration_data or {}
        self.league_contexts = self._initialize_league_contexts()
        self.team_profiles = self._initialize_team_profiles()
        self.monte_carlo_iterations = mc_iterations
        self._setup_calibration_parameters()
        self.match_context = MatchContext.UNKNOWN
        
    def _validate_and_clean_data(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive data validation and cleaning"""
        required_fields = ['home_team', 'away_team', 'league']
        
        for field in required_fields:
            if field not in match_data:
                raise ValueError(f"Missing required field: {field}")
        
        cleaned_data = match_data.copy()
        
        # Validate and clean numeric fields
        numeric_fields = {
            'home_goals': (0, 100, 0),
            'away_goals': (0, 100, 0),
            'home_conceded': (0, 100, 0),
            'away_conceded': (0, 100, 0),
            'home_goals_home': (0, 100, None),
            'away_goals_away': (0, 100, None),
            'home_xg': (0, 10, None),
            'away_xg': (0, 10, None),
            'home_xg_against': (0, 10, None),
            'away_xg_against': (0, 10, None)
        }
        
        for field, (min_val, max_val, default) in numeric_fields.items():
            if field in cleaned_data:
                try:
                    value = float(cleaned_data[field])
                    if not (min_val <= value <= max_val):
                        logger.warning(f"Field {field} value {value} outside expected range, using default")
                        cleaned_data[field] = default if default is not None else min_val
                    else:
                        cleaned_data[field] = value
                except (TypeError, ValueError):
                    logger.warning(f"Invalid value for {field}, using default")
                    cleaned_data[field] = default if default is not None else min_val
        
        # Validate form data
        for side in ['home_form', 'away_form']:
            if side in cleaned_data:
                if not isinstance(cleaned_data[side], list):
                    cleaned_data[side] = []
                else:
                    # Convert form items to numeric, filter invalid
                    valid_form = []
                    for item in cleaned_data[side]:
                        try:
                            val = float(item)
                            if 0 <= val <= 3:
                                valid_form.append(val)
                        except (TypeError, ValueError):
                            continue
                    cleaned_data[side] = valid_form
        
        # Validate nested structures
        for nested in ['injuries', 'motivation', 'h2h_data']:
            if nested in cleaned_data and not isinstance(cleaned_data[nested], dict):
                cleaned_data[nested] = {}
        
        # Calculate data quality score
        cleaned_data['data_quality_score'] = self._calculate_data_quality(cleaned_data)
        
        return cleaned_data
    
    def _calculate_data_quality(self, data: Dict) -> float:
        """Calculate data quality score (0-100)"""
        score = 0
        max_score = 0
        
        # Basic match info
        if data.get('home_team') and data.get('away_team'):
            score += 20
        max_score += 20
        
        # Recent form
        if len(data.get('home_form', [])) >= 3:
            score += 15
        if len(data.get('away_form', [])) >= 3:
            score += 15
        max_score += 30
        
        # Goal data
        if data.get('home_goals', 0) > 0:
            score += 10
        if data.get('away_goals', 0) > 0:
            score += 10
        max_score += 20
        
        # Advanced metrics
        if data.get('home_xg') is not None:
            score += 10
        if data.get('away_xg') is not None:
            score += 10
        max_score += 20
        
        return (score / max_score) * 100
    
    def _initialize_league_contexts(self) -> Dict[str, Dict]:
        """Initialize league-specific parameters with calibrated values"""
        return {
            'premier_league': {
                'avg_goals': 2.8, 'avg_corners': 10.5, 'home_advantage': 1.12,
                'goal_timing_first_half': 0.47, 'goal_timing_second_half': 0.53,
                'attack_weight': 1.05, 'defense_weight': 0.95,
                'bivariate_correlation': 0.15, 'scoring_tendency': 'moderate_high'
            },
            'la_liga': {
                'avg_goals': 2.6, 'avg_corners': 9.8, 'home_advantage': 1.15,
                'goal_timing_first_half': 0.45, 'goal_timing_second_half': 0.55,
                'attack_weight': 1.02, 'defense_weight': 0.98,
                'bivariate_correlation': 0.12, 'scoring_tendency': 'moderate'
            },
            'serie_a': {
                'avg_goals': 2.4, 'avg_corners': 10.2, 'home_advantage': 1.08,
                'goal_timing_first_half': 0.46, 'goal_timing_second_half': 0.54,
                'attack_weight': 1.00, 'defense_weight': 1.05,  # Higher defense weight for Serie A
                'bivariate_correlation': 0.08, 'scoring_tendency': 'low'  # Lower correlation for defensive leagues
            },
            'bundesliga': {
                'avg_goals': 3.1, 'avg_corners': 9.5, 'home_advantage': 1.12,
                'goal_timing_first_half': 0.48, 'goal_timing_second_half': 0.52,
                'attack_weight': 1.08, 'defense_weight': 0.92,
                'bivariate_correlation': 0.18, 'scoring_tendency': 'high'
            },
            'ligue_1': {
                'avg_goals': 2.5, 'avg_corners': 9.2, 'home_advantage': 1.1,
                'goal_timing_first_half': 0.44, 'goal_timing_second_half': 0.56,
                'attack_weight': 1.01, 'defense_weight': 0.99,
                'bivariate_correlation': 0.11, 'scoring_tendency': 'moderate_low'
            },
            'default': {
                'avg_goals': 2.7, 'avg_corners': 10.0, 'home_advantage': 1.1,
                'goal_timing_first_half': 0.46, 'goal_timing_second_half': 0.54,
                'attack_weight': 1.04, 'defense_weight': 0.96,
                'bivariate_correlation': 0.12, 'scoring_tendency': 'moderate'
            }
        }
    
    def _initialize_team_profiles(self) -> Dict[str, Dict]:
        """Initialize known team profiles for context-aware predictions"""
        return {
            'Bologna': {'style': 'defensive', 'press_intensity': 'high', 'clean_sheet_freq': 0.45},
            'Torino': {'style': 'defensive', 'press_intensity': 'medium', 'clean_sheet_freq': 0.38},
            'Atalanta': {'style': 'attacking', 'press_intensity': 'high', 'clean_sheet_freq': 0.25},
            'Inter': {'style': 'balanced', 'press_intensity': 'high', 'clean_sheet_freq': 0.40},
            'default': {'style': 'balanced', 'press_intensity': 'medium', 'clean_sheet_freq': 0.30}
        }
    
    def _setup_calibration_parameters(self):
        """Setup calibration parameters from historical data"""
        self.calibration_params = {
            'home_attack_weight': 1.05,
            'away_attack_weight': 0.95,
            'defense_weight': 1.02,  # Increased defense weight
            'form_decay_rate': 0.9,
            'h2h_weight': 0.3,
            'injury_impact': 0.08,
            'motivation_impact': 0.12,
            'regression_strength': 0.25,
            'bivariate_lambda3_alpha': 0.12,
            'defensive_team_adjustment': 0.85,  # New: reduce xG for defensive teams
            'min_goals_threshold': 0.15,  # New: minimum goal probability
            'data_quality_threshold': 60.0  # New: minimum data quality for complex models
        }
        
        if self.calibration_data:
            self.calibration_params.update(self.calibration_data)

    def _determine_match_context(self, home_xg: float, away_xg: float, home_team: str, away_team: str) -> MatchContext:
        """Determine the match context based on team profiles and xG"""
        home_profile = self.team_profiles.get(home_team, self.team_profiles['default'])
        away_profile = self.team_profiles.get(away_team, self.team_profiles['default'])
        
        # Both teams defensive
        if (home_profile['style'] == 'defensive' and away_profile['style'] == 'defensive'):
            return MatchContext.DEFENSIVE_BATTLE
        
        # High xG differential
        elif home_xg > away_xg + 1.0:
            return MatchContext.HOME_DOMINANCE
        elif away_xg > home_xg + 0.8:
            return MatchContext.AWAY_COUNTER
        
        # Both teams attacking
        elif (home_profile['style'] == 'attacking' and away_profile['style'] == 'attacking'):
            return MatchContext.OFFENSIVE_SHOWDOWN
        
        # Close defensive match
        elif home_xg + away_xg < 2.0 and abs(home_xg - away_xg) < 0.5:
            return MatchContext.TACTICAL_STALEMATE
        
        return MatchContext.UNKNOWN

    def bivariate_poisson_pmf_matrix(self, lambda1: float, lambda2: float, lambda3: float, max_goals: int = 8):
        """Compute joint PMF matrix for Bivariate Poisson"""
        exp_term = math.exp(-(lambda1 + lambda2 + lambda3))
        P = np.zeros((max_goals+1, max_goals+1), dtype=float)
        fact = [math.factorial(n) for n in range(max_goals+1)]

        for i in range(max_goals+1):
            for j in range(max_goals+1):
                s = 0.0
                k_max = min(i, j)
                for k in range(k_max + 1):
                    a = i - k
                    b = j - k
                    term = ( (lambda1 ** a) / fact[a] ) * ( (lambda2 ** b) / fact[b] ) * ( (lambda3 ** k) / fact[k] )
                    s += term
                P[i, j] = exp_term * s
        
        P /= P.sum()
        return P
    
    def get_markets_from_joint_pmf(self, P: np.ndarray):
        """Extract market probabilities from joint PMF matrix"""
        max_goals = P.shape[0] - 1
        exact_scores = {}
        
        home_win_prob = 0.0
        draw_prob = 0.0
        away_win_prob = 0.0
        
        for i in range(max_goals+1):
            for j in range(max_goals+1):
                prob = P[i, j]
                if i > j:
                    home_win_prob += prob
                elif i == j:
                    draw_prob += prob
                else:
                    away_win_prob += prob
                
                if prob > 0.001:  # Only include probabilities > 0.1%
                    exact_scores[f"{i}-{j}"] = round(prob * 100, 1)

        # Sort exact scores by probability
        exact_scores = dict(sorted(exact_scores.items(), key=lambda x: x[1], reverse=True)[:10])

        # Over/under probabilities
        total_prob = {}
        total_goals_matrix = np.add.outer(np.arange(max_goals+1), np.arange(max_goals+1))
        
        for line in [1.5, 2.5, 3.5]:
            over_mask = total_goals_matrix > line
            under_mask = total_goals_matrix < line
            total_prob[f"over_{str(line).replace('.', '')}"] = round(P[over_mask].sum() * 100, 1)
            total_prob[f"under_{str(line).replace('.', '')}"] = round(P[under_mask].sum() * 100, 1)

        # Both teams to score
        btts_mask = (np.arange(max_goals+1) > 0)[:, None] & (np.arange(max_goals+1) > 0)
        btts_prob = round(P[btts_mask].sum() * 100, 1)
        btts_no_prob = round(100 - btts_prob, 1)

        return {
            'match_outcomes': {
                'home_win': round(home_win_prob * 100, 1),
                'draw': round(draw_prob * 100, 1),
                'away_win': round(away_win_prob * 100, 1)
            },
            'exact_scores': exact_scores,
            'over_under': total_prob,
            'both_teams_score': {
                'yes': btts_prob,
                'no': btts_no_prob
            }
        }
    
    def generate_advanced_predictions(self) -> Dict[str, Any]:
        """Generate comprehensive predictions with context-aware modeling"""
        
        home_team = self.data['home_team']
        away_team = self.data['away_team']
        league = self.data.get('league', 'default')
        market_odds = self.data.get('market_odds', {})
        
        # Calculate context-aware expected goals
        home_xg, away_xg = self._calculate_context_aware_xg()
        
        # Determine match context
        self.match_context = self._determine_match_context(home_xg, away_xg, home_team, away_team)
        
        # Use appropriate model based on context and data quality
        if self.data['data_quality_score'] < self.calibration_params['data_quality_threshold']:
            probabilities = self._calculate_simple_probabilities(home_xg, away_xg)
        else:
            probabilities = self._calculate_bivariate_probabilities(home_xg, away_xg, league)
        
        # Run Monte Carlo simulation
        mc_results = self._run_bivariate_monte_carlo_simulation(home_xg, away_xg, league)
        
        # Calculate handicap probabilities
        handicap_probs = self._calculate_handicap_probabilities(home_xg, away_xg)
        
        # Generate betting signals
        betting_signals = self._generate_betting_signals(probabilities, market_odds, mc_results)
        
        # Calculate advanced metrics
        corner_predictions = self._calculate_corner_predictions(home_xg, away_xg, league)
        timing_predictions = self._calculate_enhanced_timing_predictions(home_xg, away_xg)
        
        # Risk and confidence assessment
        confidence_score = self._calculate_advanced_confidence(mc_results)
        risk_assessment = self._assess_prediction_risk(probabilities, confidence_score, mc_results)
        
        # Validate output consistency
        self._validate_output_consistency(probabilities, home_xg, away_xg, risk_assessment)
        
        return {
            'match': f"{home_team} vs {away_team}",
            'expected_goals': {'home': round(home_xg, 2), 'away': round(away_xg, 2)},
            'match_context': self.match_context.value,
            'data_quality_score': round(self.data['data_quality_score'], 1),
            'probabilities': probabilities,
            'handicap_probabilities': handicap_probs,
            'monte_carlo_results': {
                'confidence_intervals': mc_results.confidence_intervals,
                'probability_volatility': mc_results.probability_volatility
            },
            'betting_signals': betting_signals,
            'corner_predictions': corner_predictions,
            'timing_predictions': timing_predictions,
            'summary': self._generate_context_aware_summary(home_team, away_team, probabilities, home_xg, away_xg),
            'confidence_score': confidence_score,
            'risk_assessment': risk_assessment,
            'model_metrics': self._calculate_model_metrics(probabilities, mc_results),
            'bivariate_parameters': {
                'lambda1': round(home_xg - (0.12 * min(home_xg, away_xg)), 3),
                'lambda2': round(away_xg - (0.12 * min(home_xg, away_xg)), 3),
                'lambda3': round(0.12 * min(home_xg, away_xg), 3)
            }
        }
    
    def _calculate_context_aware_xg(self) -> Tuple[float, float]:
        """Calculate context-aware expected goals with team profiling"""
        league = self.data.get('league', 'default')
        league_params = self.league_contexts.get(league, self.league_contexts['default'])
        
        # Extract validated data
        home_goals = self.data.get('home_goals', 0)
        away_goals = self.data.get('away_goals', 0)
        home_conceded = self.data.get('home_conceded', 0)
        away_conceded = self.data.get('away_conceded', 0)
        
        home_goals_home = self.data.get('home_goals_home', home_goals)
        away_goals_away = self.data.get('away_goals_away', away_goals)
        
        home_form = self.data.get('home_form', [])
        away_form = self.data.get('away_form', [])
        
        injuries = self.data.get('injuries', {})
        motivation = self.data.get('motivation', {})
        h2h_data = self.data.get('h2h_data', {})
        
        # Get team profiles
        home_profile = self.team_profiles.get(self.data['home_team'], self.team_profiles['default'])
        away_profile = self.team_profiles.get(self.data['away_team'], self.team_profiles['default'])
        
        # Bayesian prior (league average adjusted for team style)
        prior_goals = league_params['avg_goals'] / 2
        prior_weight = 6
        
        # Calculate observed metrics
        home_attack_obs = max(0.3, home_goals / max(1, 6.0))
        away_attack_obs = max(0.2, away_goals / max(1, 6.0))
        home_defense_obs = max(0.3, home_conceded / max(1, 6.0))
        away_defense_obs = max(0.4, away_conceded / max(1, 6.0))
        
        # Home/away specific observations
        home_attack_home_obs = max(0.3, home_goals_home / max(1, 3.0))
        away_attack_away_obs = max(0.2, away_goals_away / max(1, 3.0))
        
        # Bayesian posterior estimates
        home_attack = (prior_goals * prior_weight + home_attack_obs * 6) / (prior_weight + 6)
        away_attack = (prior_goals * prior_weight + away_attack_obs * 6) / (prior_weight + 6)
        home_defense = (prior_goals * prior_weight + home_defense_obs * 6) / (prior_weight + 6)
        away_defense = (prior_goals * prior_weight + away_defense_obs * 6) / (prior_weight + 6)
        
        home_attack_home = (prior_goals * prior_weight + home_attack_home_obs * 3) / (prior_weight + 3)
        away_attack_away = (prior_goals * prior_weight + away_attack_away_obs * 3) / (prior_weight + 3)
        
        # Apply team style adjustments
        if home_profile['style'] == 'defensive':
            home_attack *= 0.9
            home_defense *= 1.1
        elif home_profile['style'] == 'attacking':
            home_attack *= 1.1
            home_defense *= 0.9
            
        if away_profile['style'] == 'defensive':
            away_attack *= 0.9
            away_defense *= 1.1
        elif away_profile['style'] == 'attacking':
            away_attack *= 1.1
            away_defense *= 0.9
        
        # Apply calibration weights
        home_attack *= self.calibration_params['home_attack_weight']
        away_attack *= self.calibration_params['away_attack_weight']
        home_defense *= self.calibration_params['defense_weight']
        away_defense *= self.calibration_params['defense_weight']
        
        # Form factors with decay
        home_form_factor = self._calculate_decaying_form_factor(home_form)
        away_form_factor = self._calculate_decaying_form_factor(away_form)
        
        # Injury and motivation adjustments
        home_injury_factor = max(0.7, 1.0 - (injuries.get('home', 0) * self.calibration_params['injury_impact']))
        away_injury_factor = max(0.7, 1.0 - (injuries.get('away', 0) * self.calibration_params['injury_impact']))
        
        home_motivation_factor = 1.0 + (motivation.get('home', 1.0) - 1.0) * self.calibration_params['motivation_impact']
        away_motivation_factor = 1.0 + (motivation.get('away', 1.0) - 1.0) * self.calibration_params['motivation_impact']
        
        # Calculate base xG
        base_home_xg = (home_attack_home * away_defense * league_params['home_advantage'] * 
                       home_form_factor * home_injury_factor * home_motivation_factor)
        base_away_xg = (away_attack_away * home_defense * 
                       away_form_factor * away_injury_factor * away_motivation_factor)
        
        # Apply defensive context adjustment
        if self.match_context == MatchContext.DEFENSIVE_BATTLE:
            base_home_xg *= 0.8
            base_away_xg *= 0.8
        elif self.match_context == MatchContext.TACTICAL_STALEMATE:
            base_home_xg *= 0.9
            base_away_xg *= 0.9
        
        # Apply H2H adjustment if available
        if h2h_data:
            base_home_xg, base_away_xg = self._apply_bayesian_h2h_adjustment(
                base_home_xg, base_away_xg, h2h_data
            )
        
        # Final regression to mean
        home_xg = (base_home_xg + league_params['avg_goals'] * self.calibration_params['regression_strength']) / (1 + self.calibration_params['regression_strength'])
        away_xg = (base_away_xg + league_params['avg_goals'] * self.calibration_params['regression_strength']) / (1 + self.calibration_params['regression_strength'])
        
        # Ensure reasonable bounds with context awareness
        home_xg = max(self.calibration_params['min_goals_threshold'], min(4.0, home_xg))
        away_xg = max(self.calibration_params['min_goals_threshold'], min(3.5, away_xg))
        
        return round(home_xg, 3), round(away_xg, 3)
    
    def _calculate_simple_probabilities(self, home_xg: float, away_xg: float) -> Dict[str, Any]:
        """Calculate probabilities using simple Poisson when data quality is low"""
        # Simple Poisson for match outcomes
        home_win_prob = 0.0
        draw_prob = 0.0
        away_win_prob = 0.0
        
        for i in range(8):
            for j in range(8):
                prob = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
                if i > j:
                    home_win_prob += prob
                elif i == j:
                    draw_prob += prob
                else:
                    away_win_prob += prob
        
        # Simple BTTS calculation
        btts_prob = (1 - poisson.pmf(0, home_xg)) * (1 - poisson.pmf(0, away_xg))
        
        return {
            'match_outcomes': {
                'home_win': round(home_win_prob * 100, 1),
                'draw': round(draw_prob * 100, 1),
                'away_win': round(away_win_prob * 100, 1)
            },
            'both_teams_score': {
                'yes': round(btts_prob * 100, 1),
                'no': round((1 - btts_prob) * 100, 1)
            },
            'goal_timing': {
                'first_half': round((1 - math.exp(-(home_xg + away_xg) * 0.46)) * 100, 1),
                'second_half': round((1 - math.exp(-(home_xg + away_xg) * 0.54)) * 100, 1)
            }
        }
    
    def _calculate_bivariate_probabilities(self, home_xg: float, away_xg: float, league: str) -> Dict[str, Any]:
        """Calculate probabilities using bivariate Poisson distribution"""
        league_params = self.league_contexts.get(league, self.league_contexts['default'])
        
        # Adjust correlation based on match context
        if self.match_context == MatchContext.DEFENSIVE_BATTLE:
            lambda3_alpha = league_params['bivariate_correlation'] * 0.6
        elif self.match_context == MatchContext.OFFENSIVE_SHOWDOWN:
            lambda3_alpha = league_params['bivariate_correlation'] * 1.2
        else:
            lambda3_alpha = league_params['bivariate_correlation']
        
        lambda3 = lambda3_alpha * min(home_xg, away_xg)
        lambda1 = max(0.1, home_xg - lambda3)
        lambda2 = max(0.1, away_xg - lambda3)
        
        P = self.bivariate_poisson_pmf_matrix(lambda1, lambda2, lambda3, max_goals=8)
        markets = self.get_markets_from_joint_pmf(P)
        markets['goal_timing'] = self._calculate_goal_timing_probabilities(home_xg, away_xg, league)
        
        return markets
    
    def _calculate_goal_timing_probabilities(self, home_xg: float, away_xg: float, league: str) -> Dict[str, float]:
        """Calculate goal timing probabilities"""
        league_params = self.league_contexts.get(league, self.league_contexts['default'])
        
        first_half_xg_home = home_xg * league_params['goal_timing_first_half']
        first_half_xg_away = away_xg * league_params['goal_timing_first_half']
        prob_no_goals_first = poisson.pmf(0, first_half_xg_home) * poisson.pmf(0, first_half_xg_away)
        
        second_half_xg_home = home_xg * league_params['goal_timing_second_half']
        second_half_xg_away = away_xg * league_params['goal_timing_second_half']
        prob_no_goals_second = poisson.pmf(0, second_half_xg_home) * poisson.pmf(0, second_half_xg_away)
        
        return {
            'first_half': round((1 - prob_no_goals_first) * 100, 1),
            'second_half': round((1 - prob_no_goals_second) * 100, 1)
        }
    
    def _run_bivariate_monte_carlo_simulation(self, home_xg: float, away_xg: float, league: str) -> MonteCarloResults:
        """Run Monte Carlo simulation using bivariate Poisson structure"""
        np.random.seed(42)
        
        league_params = self.league_contexts.get(league, self.league_contexts['default'])
        lambda3_alpha = league_params['bivariate_correlation']
        lambda3 = lambda3_alpha * min(home_xg, away_xg)
        lambda1 = max(0.1, home_xg - lambda3)
        lambda2 = max(0.1, away_xg - lambda3)
        
        C = np.random.poisson(lambda3, self.monte_carlo_iterations)
        A = np.random.poisson(lambda1, self.monte_carlo_iterations)
        B = np.random.poisson(lambda2, self.monte_carlo_iterations)
        
        home_goals_sim = A + C
        away_goals_sim = B + C
        
        home_wins = np.sum(home_goals_sim > away_goals_sim) / self.monte_carlo_iterations
        draws = np.sum(home_goals_sim == away_goals_sim) / self.monte_carlo_iterations
        away_wins = np.sum(home_goals_sim < away_goals_sim) / self.monte_carlo_iterations
        
        total_goals = home_goals_sim + away_goals_sim
        over_25 = np.sum(total_goals > 2.5) / self.monte_carlo_iterations
        btts = np.sum((home_goals_sim > 0) & (away_goals_sim > 0)) / self.monte_carlo_iterations
        
        exact_scores = {}
        for i in range(6):  # Reduced range for more realistic scores
            for j in range(6):
                count = np.sum((home_goals_sim == i) & (away_goals_sim == j))
                prob = count / self.monte_carlo_iterations
                if prob > 0.005:  # Only include probabilities > 0.5%
                    exact_scores[f"{i}-{j}"] = round(prob * 100, 1)
        
        # Sort by probability
        exact_scores = dict(sorted(exact_scores.items(), key=lambda x: x[1], reverse=True)[:8])
        
        def calculate_ci(probs, alpha=0.95):
            se = np.sqrt(probs * (1 - probs) / self.monte_carlo_iterations)
            z_score = 1.96
            return (max(0, probs - z_score * se), min(1, probs + z_score * se))
        
        confidence_intervals = {
            'home_win': calculate_ci(home_wins),
            'draw': calculate_ci(draws),
            'away_win': calculate_ci(away_wins),
            'over_2.5': calculate_ci(over_25)
        }
        
        # Calculate probability volatility
        batch_size = 1000
        num_batches = self.monte_carlo_iterations // batch_size
        
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
            btts_prob=btts,
            exact_scores=exact_scores,
            confidence_intervals=confidence_intervals,
            probability_volatility=probability_volatility
        )
    
    def _calculate_decaying_form_factor(self, form: List[float]) -> float:
        """Calculate form factor with exponential decay"""
        if not form or len(form) == 0:
            return 1.0
        
        try:
            form_scores = [float(score) for score in form]
        except:
            return 1.0
        
        weights = [self.calibration_params['form_decay_rate'] ** i for i in range(len(form_scores))]
        weights = [w / sum(weights) for w in weights]
        
        total_points = sum(score * weight for score, weight in zip(form_scores, reversed(weights)))
        max_possible = sum(3 * weight for weight in weights)
        
        form_ratio = total_points / max_possible if max_possible > 0 else 0.5
        return 0.8 + (form_ratio * 0.4)
    
    def _apply_bayesian_h2h_adjustment(self, home_xg: float, away_xg: float, h2h_data: Dict) -> Tuple[float, float]:
        """Apply Bayesian H2H adjustment"""
        matches = h2h_data.get('matches', 0)
        home_goals = h2h_data.get('home_goals', 0)
        away_goals = h2h_data.get('away_goals', 0)
        
        if matches < 2:
            return home_xg, away_xg
        
        h2h_weight = min(0.4, matches * 0.1)
        h2h_home_avg = home_goals / matches
        h2h_away_avg = away_goals / matches
        
        adjusted_home_xg = (home_xg * (1 - h2h_weight)) + (h2h_home_avg * h2h_weight)
        adjusted_away_xg = (away_xg * (1 - h2h_weight)) + (h2h_away_avg * h2h_weight)
        
        return adjusted_home_xg, adjusted_away_xg
    
    def _calculate_handicap_probabilities(self, home_xg: float, away_xg: float) -> Dict[str, float]:
        """Calculate Asian Handicap probabilities using Skellam distribution"""
        handicap_probs = {}
        
        handicaps = [-2.5, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5]
        
        for handicap in handicaps:
            if handicap == 0:
                prob = 1 - skellam.cdf(0, home_xg, away_xg)
            elif handicap > 0:
                prob = 1 - skellam.cdf(handicap - 1, home_xg, away_xg)
            else:
                prob = skellam.cdf(abs(handicap), home_xg, away_xg)
            
            handicap_probs[f"handicap_{handicap}"] = round(prob * 100, 1)
        
        return handicap_probs
    
    def _generate_betting_signals(self, probabilities: Dict, market_odds: Dict, 
                                mc_results: MonteCarloResults) -> List[Dict]:
        """Generate actionable betting signals with value detection"""
        signals = []
        market_probs = self._convert_odds_to_probabilities(market_odds)
        
        markets_to_check = [
            ('home_win', '1x2 Home'),
            ('draw', '1x2 Draw'), 
            ('away_win', '1x2 Away'),
            ('over_2.5', 'Over 2.5 Goals'),
            ('both_teams_score', 'BTTS Yes')
        ]
        
        for model_key, market_name in markets_to_check:
            if model_key in probabilities.get('match_outcomes', {}):
                model_prob = probabilities['match_outcomes'][model_key] / 100
            elif model_key == 'both_teams_score':
                model_prob = probabilities.get('both_teams_score', {}).get('yes', 0) / 100
            elif model_key == 'over_2.5':
                model_prob = probabilities.get('over_under', {}).get('over_25', 0) / 100
            else:
                model_prob = probabilities.get(model_key, 0) / 100
            
            book_prob = market_probs.get(market_name, 0)
            
            if book_prob > 0 and model_prob > 0:
                edge = (model_prob - book_prob) * 100
                
                if abs(edge) > 2:
                    confidence = self._calculate_bet_confidence(model_prob, mc_results.probability_volatility.get(model_key, 0))
                    stake = self._calculate_kelly_stake(model_prob, book_prob)
                    value_rating = self._get_value_rating(edge)
                    
                    signals.append(BettingSignal(
                        market=market_name,
                        model_prob=round(model_prob * 100, 1),
                        book_prob=round(book_prob * 100, 1),
                        edge=round(edge, 1),
                        confidence=confidence,
                        recommended_stake=stake,
                        value_rating=value_rating
                    ).__dict__)
        
        signals.sort(key=lambda x: x['edge'], reverse=True)
        return signals
    
    def _convert_odds_to_probabilities(self, market_odds: Dict) -> Dict[str, float]:
        """Convert decimal odds to probabilities"""
        probs = {}
        
        for market, odds in market_odds.items():
            if isinstance(odds, (int, float)) and odds > 1:
                probs[market] = 1 / odds
            elif isinstance(odds, dict):
                total_implied = sum(1 / o for o in odds.values() if isinstance(o, (int, float)) and o > 1)
                if total_implied > 0:
                    for outcome, odd in odds.items():
                        if isinstance(odd, (int, float)) and odd > 1:
                            probs[f"{market} {outcome}"] = (1 / odd) / total_implied
        
        return probs
    
    def _calculate_bet_confidence(self, model_prob: float, volatility: float) -> str:
        """Calculate betting confidence based on probability and volatility"""
        if volatility < 0.02 and model_prob > 0.6:
            return "HIGH"
        elif volatility < 0.04 and model_prob > 0.55:
            return "MEDIUM"
        elif volatility < 0.06:
            return "LOW"
        else:
            return "SPECULATIVE"
    
    def _calculate_kelly_stake(self, model_prob: float, book_prob: float, kelly_fraction: float = 0.25) -> float:
        """Calculate Kelly criterion stake"""
        if book_prob <= 0 or model_prob <= book_prob:
            return 0.0
        
        decimal_odds = 1 / book_prob
        kelly_stake = (model_prob * decimal_odds - 1) / (decimal_odds - 1)
        return max(0.0, min(0.05, kelly_stake * kelly_fraction))
    
    def _get_value_rating(self, edge: float) -> str:
        """Get value rating based on edge percentage"""
        if edge > 10:
            return "EXCEPTIONAL"
        elif edge > 7:
            return "HIGH"
        elif edge > 5:
            return "GOOD"
        elif edge > 3:
            return "MODERATE"
        else:
            return "LOW"
    
    def _calculate_corner_predictions(self, home_xg: float, away_xg: float, league: str) -> Dict[str, Any]:
        """Calculate realistic corner predictions"""
        league_params = self.league_contexts.get(league, self.league_contexts['default'])
        
        base_corners = league_params['avg_corners']
        attacking_bonus = (home_xg + away_xg - league_params['avg_goals']) * 1.2
        
        total_corners = base_corners + attacking_bonus
        total_corners = max(6, min(14, total_corners))  # More realistic bounds
        
        home_corners = total_corners * 0.55
        away_corners = total_corners * 0.45
        
        return {
            'total': f"{int(total_corners)}-{int(total_corners + 1)}",
            'home': f"{int(home_corners)}-{int(home_corners + 0.5)}",
            'away': f"{int(away_corners)}-{int(away_corners + 0.5)}",
            'over_9.5': 'YES' if total_corners > 9.5 else 'NO'
        }
    
    def _calculate_enhanced_timing_predictions(self, home_xg: float, away_xg: float) -> Dict[str, Any]:
        """Calculate enhanced goal timing predictions"""
        total_xg = home_xg + away_xg
        
        if total_xg < 1.5:
            first_goal = "35+ minutes"
            late_goals = "UNLIKELY"
        elif total_xg < 2.5:
            first_goal = "25-35 minutes"
            late_goals = "POSSIBLE"
        else:
            first_goal = "15-30 minutes"
            late_goals = "LIKELY"
        
        return {
            'first_goal': first_goal,
            'late_goals': late_goals,
            'most_action': "Last 20 minutes of each half" if total_xg > 2.0 else "Scattered throughout"
        }
    
    def _calculate_advanced_confidence(self, mc_results: MonteCarloResults) -> int:
        """Calculate advanced confidence score using Monte Carlo results"""
        base_confidence = self.data['data_quality_score'] * 0.5  # Base from data quality
        
        # Add bonuses for good data
        if len(self.data.get('home_form', [])) >= 4:
            base_confidence += 10
        if len(self.data.get('away_form', [])) >= 4:
            base_confidence += 10
            
        h2h_data = self.data.get('h2h_data', {})
        if h2h_data.get('matches', 0) >= 3:
            base_confidence += 15
        
        # Penalty for high volatility
        avg_volatility = np.mean(list(mc_results.probability_volatility.values()))
        volatility_penalty = min(30, int(avg_volatility * 800))
        
        confidence = base_confidence - volatility_penalty
        return max(10, min(95, int(confidence)))
    
    def _assess_prediction_risk(self, probabilities: Dict, confidence: int, 
                              mc_results: MonteCarloResults) -> Dict[str, str]:
        """Enhanced risk assessment with Monte Carlo uncertainty"""
        outcomes = probabilities['match_outcomes']
        highest_prob = max(outcomes.values())
        
        probs = np.array([outcomes['home_win'], outcomes['draw'], outcomes['away_win']]) / 100
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(3)
        uncertainty_ratio = entropy / max_entropy
        
        # Context-aware risk assessment
        if self.match_context == MatchContext.DEFENSIVE_BATTLE:
            risk_adjustment = 0.9
        elif self.match_context == MatchContext.OFFENSIVE_SHOWDOWN:
            risk_adjustment = 1.1
        else:
            risk_adjustment = 1.0
            
        adjusted_uncertainty = uncertainty_ratio * risk_adjustment
        
        if highest_prob > 70 and confidence > 80 and adjusted_uncertainty < 0.7:
            risk_level = "LOW"
            explanation = "Strong favorite with low uncertainty"
        elif highest_prob > 55 and confidence > 65 and adjusted_uncertainty < 0.85:
            risk_level = "MEDIUM"
            explanation = "Moderate favorite with acceptable uncertainty"
        else:
            risk_level = "HIGH"
            explanation = f"High uncertainty (entropy: {uncertainty_ratio:.2f})"
        
        return {
            'risk_level': risk_level,
            'explanation': explanation,
            'certainty': f"{highest_prob}%",
            'uncertainty': round(uncertainty_ratio, 2)
        }
    
    def _calculate_model_metrics(self, probabilities: Dict, mc_results: MonteCarloResults) -> Dict[str, float]:
        """Calculate model performance metrics"""
        probs = np.array([probabilities['match_outcomes'][k] for k in ['home_win', 'draw', 'away_win']]) / 100
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        avg_volatility = np.mean(list(mc_results.probability_volatility.values()))
        avg_ci_width = np.mean([
            ci[1] - ci[0] for ci in mc_results.confidence_intervals.values()
        ])
        
        return {
            'shannon_entropy': round(entropy, 3),
            'avg_probability_volatility': round(avg_volatility, 4),
            'avg_confidence_interval_width': round(avg_ci_width, 4),
            'monte_carlo_iterations': self.monte_carlo_iterations,
            'match_context': self.match_context.value
        }
    
    def _generate_context_aware_summary(self, home_team: str, away_team: str, probabilities: Dict,
                                      home_xg: float, away_xg: float) -> str:
        """Generate context-aware match summary"""
        outcomes = probabilities['match_outcomes']
        
        if self.match_context == MatchContext.DEFENSIVE_BATTLE:
            return f"Defensive stalemate expected between {home_team} and {away_team}. Low-scoring affair likely with both teams prioritizing defensive solidity. Set-pieces and individual moments could prove decisive."
        
        elif self.match_context == MatchContext.TACTICAL_STALEMATE:
            return f"Tactical battle anticipated with minimal separation between {home_team} and {away_team}. Both teams well-organized defensively, suggesting a cagey encounter decided by fine margins."
        
        elif self.match_context == MatchContext.OFFENSIVE_SHOWDOWN:
            return f"Open, attacking football expected as {home_team} and {away_team} both favor offensive approaches. High-scoring encounter likely with both teams creating numerous chances."
        
        elif outcomes['home_win'] > 65 and home_xg > away_xg + 0.8:
            return f"{home_team} demonstrate clear superiority with {home_xg:.1f} expected goals. Strong home advantage and attacking efficiency suggest comfortable victory."
        
        elif outcomes['home_win'] > 55 and home_xg > away_xg + 0.4:
            return f"{home_team} hold measurable advantage with better expected goal metrics. {away_team} will need exceptional defensive discipline to contain home threat."
        
        elif outcomes['away_win'] > 50 and away_xg > home_xg:
            return f"{away_team} pose significant threat with competitive expected goals. This could challenge {home_team}'s home defensive record."
        
        elif outcomes['draw'] > 35 and abs(home_xg - away_xg) < 0.3:
            return f"Evenly balanced encounter with minimal separation in expected goals. Set-piece efficiency and individual quality likely decisive."
        
        else:
            return f"Competitive match with both teams creating opportunities. Small margins expected to determine outcome in what promises tactical engagement."
    
    def _validate_output_consistency(self, probabilities: Dict, home_xg: float, away_xg: float, 
                                   risk_assessment: Dict) -> None:
        """Validate output consistency and log warnings for inconsistencies"""
        outcomes = probabilities['match_outcomes']
        exact_scores = probabilities.get('exact_scores', {})
        
        # Check if high xG contradicts low-scoring predictions
        total_xg = home_xg + away_xg
        low_scoring_prob = sum(prob for score, prob in exact_scores.items() 
                             if int(score.split('-')[0]) + int(score.split('-')[1]) <= 1)
        
        if total_xg > 2.5 and low_scoring_prob > 40:
            logger.warning(f"High total xG ({total_xg:.1f}) contradicts low-scoring predictions ({low_scoring_prob}% probability)")
        
        # Check if match outcomes align with exact scores
        implied_home_win = sum(prob for score, prob in exact_scores.items() 
                             if int(score.split('-')[0]) > int(score.split('-')[1]))
        actual_home_win = outcomes['home_win']
        
        if abs(implied_home_win - actual_home_win) > 15:
            logger.warning(f"Home win probability mismatch: {actual_home_win}% vs {implied_home_win}% from exact scores")
        
        # Validate risk assessment aligns with probabilities
        highest_prob = max(outcomes.values())
        if risk_assessment['risk_level'] == 'LOW' and highest_prob < 60:
            logger.warning("Low risk assessment contradicts moderate highest probability")

# Example usage with improved data
if __name__ == "__main__":
    calibration_data = {
        'home_attack_weight': 1.05,
        'away_attack_weight': 0.95,
        'defense_weight': 1.02,
        'form_decay_rate': 0.85,
        'h2h_weight': 0.25,
        'injury_impact': 0.08,
        'motivation_impact': 0.12,
        'regression_strength': 0.2,
        'bivariate_lambda3_alpha': 0.12,
        'defensive_team_adjustment': 0.85,
        'min_goals_threshold': 0.15,
        'data_quality_threshold': 60.0
    }
    
    # Realistic data for Bologna vs Torino (defensive battle)
    match_data = {
        'home_team': 'Bologna',
        'away_team': 'Torino', 
        'league': 'serie_a',
        'home_goals': 12,
        'away_goals': 8,
        'home_conceded': 8,
        'away_conceded': 10,
        'home_goals_home': 7,
        'away_goals_away': 4,
        'home_form': [3, 1, 3, 0, 1, 3],  # Realistic form points
        'away_form': [0, 1, 1, 3, 0, 1],
        'h2h_data': {
            'matches': 4,
            'home_wins': 2,
            'away_wins': 0, 
            'draws': 2,
            'home_goals': 5,
            'away_goals': 3
        },
        'injuries': {'home': 1, 'away': 2},
        'motivation': {'home': 1.1, 'away': 1.0},
        'market_odds': {
            '1x2 Home': 2.10,
            '1x2 Draw': 3.10,
            '1x2 Away': 3.80,
            'Over 2.5 Goals': 2.30,
            'BTTS Yes': 1.95
        }
    }
    
    engine = AdvancedPredictionEngine(match_data, calibration_data)
    predictions = engine.generate_advanced_predictions()
    
    print("Enhanced Context-Aware Prediction Results:")
    print(json.dumps(predictions, indent=2, default=str))
