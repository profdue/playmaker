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
    """Data class for betting signals - SEPARATE from prediction logic"""
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

class SignalEngine:
    """
    PURE PREDICTIVE ENGINE - No market influence
    Only football data → probabilities
    """
    
    def __init__(self, match_data: Dict[str, Any], calibration_data: Optional[Dict] = None):
        self.data = self._validate_and_clean_data(match_data)
        self.calibration_data = calibration_data or {}
        self.league_contexts = self._initialize_league_contexts()
        self.team_profiles = self._initialize_team_profiles()
        self._setup_calibration_parameters()
        self.match_context = MatchContext.UNKNOWN
        
    def _validate_and_clean_data(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Data validation - NO MARKET DATA PROCESSING HERE"""
        required_fields = ['home_team', 'away_team']
        
        for field in required_fields:
            if field not in match_data:
                raise ValueError(f"Missing required field: {field}")
        
        cleaned_data = match_data.copy()
        
        # Remove any market data that might accidentally be passed
        if 'market_odds' in cleaned_data:
            del cleaned_data['market_odds']
        
        # Standard numeric validation
        numeric_fields = {
            'home_goals': (0, 20, 1.5),
            'away_goals': (0, 20, 1.5),
            'home_conceded': (0, 20, 1.5),
            'away_conceded': (0, 20, 1.5),
        }
        
        for field, (min_val, max_val, default) in numeric_fields.items():
            if field in cleaned_data:
                try:
                    value = float(cleaned_data[field])
                    if value < min_val or value > max_val:
                        logger.warning(f"Field {field} value {value} outside expected range")
                        cleaned_data[field] = max(min_val, min(value, max_val))
                except (TypeError, ValueError):
                    cleaned_data[field] = default
        
        # Calculate data quality score
        cleaned_data['data_quality_score'] = self._calculate_data_quality(cleaned_data)
        
        return cleaned_data
    
    def _calculate_data_quality(self, data: Dict) -> float:
        """Calculate data quality score (0-100) - PURE FOOTBALL METRICS ONLY"""
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
        
        # H2H data
        h2h_data = data.get('h2h_data', {})
        if h2h_data.get('matches', 0) >= 2:
            score += 20
        max_score += 20
        
        return (score / max_score) * 100
    
    def _initialize_league_contexts(self) -> Dict[str, Dict]:
        """League-specific parameters - NO MARKET INFLUENCE"""
        return {
            'premier_league': {'avg_goals': 1.42, 'home_advantage': 1.12},
            'la_liga': {'avg_goals': 1.30, 'home_advantage': 1.15},
            'serie_a': {'avg_goals': 1.35, 'home_advantage': 1.12},
            'bundesliga': {'avg_goals': 1.55, 'home_advantage': 1.10},
            'ligue_1': {'avg_goals': 1.28, 'home_advantage': 1.13},
            'default': {'avg_goals': 1.35, 'home_advantage': 1.12}
        }
    
    def _initialize_team_profiles(self) -> Dict[str, Dict]:
        """Team playing styles - PURE FOOTBALL CHARACTERISTICS"""
        # Base profiles for known teams, others will be calculated dynamically
        base_profiles = {
            'Bologna': {'style': 'defensive', 'press_intensity': 'high', 'clean_sheet_freq': 0.45},
            'Torino': {'style': 'defensive', 'press_intensity': 'medium', 'clean_sheet_freq': 0.38},
            'Atalanta': {'style': 'attacking', 'press_intensity': 'high', 'clean_sheet_freq': 0.25},
            'Inter': {'style': 'balanced', 'press_intensity': 'high', 'clean_sheet_freq': 0.40},
            'PSG': {'style': 'attacking', 'press_intensity': 'high', 'clean_sheet_freq': 0.35},
            'Nice': {'style': 'balanced', 'press_intensity': 'medium', 'clean_sheet_freq': 0.30},
            'Augsburg': {'style': 'defensive', 'press_intensity': 'medium', 'clean_sheet_freq': 0.20},
            'Borussia Dortmund': {'style': 'attacking', 'press_intensity': 'high', 'clean_sheet_freq': 0.40},
            'Getafe': {'style': 'defensive', 'press_intensity': 'medium', 'clean_sheet_freq': 0.25},
            'Girona': {'style': 'attacking', 'press_intensity': 'high', 'clean_sheet_freq': 0.15},
            'default': {'style': 'balanced', 'press_intensity': 'medium', 'clean_sheet_freq': 0.30}
        }
        return base_profiles
    
    def _calculate_dynamic_team_profile(self, team_data: Dict, is_home: bool) -> Dict:
        """Calculate team profile dynamically based on performance data"""
        goals_for = team_data.get('goals_scored', team_data.get('goals', 0))
        goals_against = team_data.get('goals_conceded', 0)
        matches_played = 6  # Default to 6 games for form
        
        if matches_played == 0:
            return self.team_profiles['default']
        
        attack_ratio = goals_for / matches_played
        defense_ratio = goals_against / matches_played
        
        # Determine style based on ratios
        if attack_ratio < 1.0 and defense_ratio < 1.0:
            style = 'defensive'
        elif attack_ratio > 1.5 and defense_ratio > 1.2:
            style = 'attacking'
        else:
            style = 'balanced'
        
        # Estimate clean sheet frequency
        clean_sheet_freq = max(0.1, min(0.6, (matches_played - (goals_against / 2)) / matches_played))
        
        return {
            'style': style,
            'press_intensity': 'medium',  # Default, could be enhanced
            'clean_sheet_freq': clean_sheet_freq
        }
    
    def _setup_calibration_parameters(self):
        """Calibration based on historical football data only"""
        self.calibration_params = {
            'home_advantage': 1.12,
            'form_decay_rate': 0.9,
            'h2h_weight': 0.3,
            'injury_impact': 0.08,
            'motivation_impact': 0.12,
            'bivariate_correlation': 0.12,
            'data_quality_threshold': 50.0
        }
        
        if self.calibration_data:
            self.calibration_params.update(self.calibration_data)

    def _calculate_motivation_impact(self, motivation_level: str, match_context: str) -> float:
        """Context-aware motivation impact calculation"""
        base_multipliers = {
            "Low": 0.7, "Normal": 1.0, "High": 1.2, "Very High": 1.4
        }
        
        # Adjust based on match context
        context_adjustments = {
            'DEFENSIVE_BATTLE': 0.9,    # Motivation matters less in defensive games
            'OFFENSIVE_SHOWDOWN': 1.1,  # Motivation matters more in open games
            'HOME_DOMINANCE': 0.95,     # Home advantage reduces motivation need
            'AWAY_COUNTER': 1.15,       # Away teams need more motivation
            'TACTICAL_STALEMATE': 1.05, # Motivation can break stalemates
            'UNKNOWN': 1.0
        }
        
        adjustment = context_adjustments.get(match_context, 1.0)
        return base_multipliers.get(motivation_level, 1.0) * adjustment

    def _determine_match_context(self, home_xg: float, away_xg: float, home_team: str, away_team: str) -> MatchContext:
        """Determine match context - PURE FOOTBALL ANALYSIS"""
        home_profile = self.team_profiles.get(home_team, self.team_profiles['default'])
        away_profile = self.team_profiles.get(away_team, self.team_profiles['default'])
        
        total_xg = home_xg + away_xg
        xg_difference = abs(home_xg - away_xg)
        
        # Defensive battle detection
        if (home_profile['style'] == 'defensive' and away_profile['style'] == 'defensive' 
            and total_xg < 2.2):
            return MatchContext.DEFENSIVE_BATTLE
        
        # Home dominance
        elif home_xg > away_xg + 1.0:
            return MatchContext.HOME_DOMINANCE
        
        # Away counter
        elif away_xg > home_xg + 0.8:
            return MatchContext.AWAY_COUNTER
        
        # Offensive showdown
        elif (home_profile['style'] == 'attacking' and away_profile['style'] == 'attacking'
              and total_xg > 3.0):
            return MatchContext.OFFENSIVE_SHOWDOWN
        
        # Tactical stalemate
        elif total_xg < 2.5 and xg_difference < 0.5:
            return MatchContext.TACTICAL_STALEMATE
        
        # Enhanced unknown context handling
        if total_xg > 3.0:
            return MatchContext.OFFENSIVE_SHOWDOWN
        elif xg_difference > 0.7:
            return MatchContext.HOME_DOMINANCE if home_xg > away_xg else MatchContext.AWAY_COUNTER
        else:
            return MatchContext.TACTICAL_STALEMATE

    def _apply_realistic_xg_bounds(self, home_xg: float, away_xg: float, league: str) -> Tuple[float, float]:
        """Apply realistic bounds to xG based on league and team styles"""
        league_bounds = {
            'premier_league': (0.2, 3.5),
            'la_liga': (0.15, 3.2),
            'serie_a': (0.1, 3.0),
            'bundesliga': (0.25, 4.0),
            'ligue_1': (0.15, 3.3),
            'default': (0.1, 3.5)
        }
        
        min_xg, max_xg = league_bounds.get(league, (0.1, 3.5))
        
        # Apply team-style specific adjustments
        home_profile = self.team_profiles.get(self.data['home_team'], self.team_profiles['default'])
        away_profile = self.team_profiles.get(self.data['away_team'], self.team_profiles['default'])
        
        if home_profile['style'] == 'defensive':
            max_xg = min(max_xg, 2.8)
        if away_profile['style'] == 'defensive':
            max_xg = min(max_xg, 2.5)
        
        return max(min_xg, min(max_xg, home_xg)), max(min_xg, min(max_xg, away_xg))

    def _calculate_normalized_xg(self) -> Tuple[float, float]:
        """Calculate expected goals with Dixon-Coles normalization - PURE FOOTBALL DATA"""
        league = self.data.get('league', 'serie_a')
        league_params = self.league_contexts.get(league, self.league_contexts['default'])
        league_avg_goals = league_params['avg_goals']
        
        # Extract raw football data
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
        
        # Get team profiles for context
        home_profile = self.team_profiles.get(self.data['home_team'], self.team_profiles['default'])
        away_profile = self.team_profiles.get(self.data['away_team'], self.team_profiles['default'])
        
        # Convert to per-game averages
        home_attack_pg = home_goals / 6.0 if home_goals > 0 else league_avg_goals
        away_attack_pg = away_goals / 6.0 if away_goals > 0 else league_avg_goals
        home_defense_pg = home_conceded / 6.0 if home_conceded > 0 else league_avg_goals
        away_defense_pg = away_conceded / 6.0 if away_conceded > 0 else league_avg_goals
        
        # Home/away specific averages
        home_attack_home_pg = home_goals_home / 3.0 if home_goals_home > 0 else home_attack_pg
        away_attack_away_pg = away_goals_away / 3.0 if away_goals_away > 0 else away_attack_pg
        
        # Dixon-Coles normalization
        home_attack_strength = home_attack_home_pg / league_avg_goals
        home_defense_strength = home_defense_pg / league_avg_goals
        away_attack_strength = away_attack_away_pg / league_avg_goals
        away_defense_strength = away_defense_pg / league_avg_goals
        
        # Calculate base expected goals
        base_home_xg = league_avg_goals * home_attack_strength * away_defense_strength
        base_away_xg = league_avg_goals * away_attack_strength * home_defense_strength
        
        # Apply home advantage
        base_home_xg *= self.calibration_params['home_advantage']
        
        # Form factors
        home_form_factor = self._calculate_decaying_form_factor(home_form)
        away_form_factor = self._calculate_decaying_form_factor(away_form)
        
        base_home_xg *= home_form_factor
        base_away_xg *= away_form_factor
        
        # Enhanced motivation adjustments
        home_motivation_level = motivation.get('home', 'Normal')
        away_motivation_level = motivation.get('away', 'Normal')
        
        # Get current match context for motivation adjustment
        temp_context = self._determine_match_context(base_home_xg, base_away_xg, 
                                                   self.data['home_team'], self.data['away_team'])
        
        home_motivation_factor = self._calculate_motivation_impact(home_motivation_level, temp_context.value)
        away_motivation_factor = self._calculate_motivation_impact(away_motivation_level, temp_context.value)
        
        # Injury adjustments
        home_injury_factor = max(0.7, 1.0 - (injuries.get('home', 0) * self.calibration_params['injury_impact']))
        away_injury_factor = max(0.7, 1.0 - (injuries.get('away', 0) * self.calibration_params['injury_impact']))
        
        base_home_xg *= home_injury_factor * home_motivation_factor
        base_away_xg *= away_injury_factor * away_motivation_factor
        
        # Apply H2H adjustment if available
        if h2h_data and h2h_data.get('matches', 0) >= 2:
            base_home_xg, base_away_xg = self._apply_bayesian_h2h_adjustment(
                base_home_xg, base_away_xg, h2h_data
            )
        
        # Apply team style adjustments
        if home_profile['style'] == 'defensive':
            base_home_xg *= 0.85
            base_away_xg *= 0.80
        elif home_profile['style'] == 'attacking':
            base_home_xg *= 1.15
            base_away_xg *= 1.10
            
        if away_profile['style'] == 'defensive':
            base_away_xg *= 0.85
            base_home_xg *= 0.80
        elif away_profile['style'] == 'attacking':
            base_away_xg *= 1.15
            base_home_xg *= 1.10
        
        # Apply realistic bounds
        home_xg, away_xg = self._apply_realistic_xg_bounds(base_home_xg, base_away_xg, league)
        
        logger.info(f"PURE xG Calculation - Home: {home_xg:.3f}, Away: {away_xg:.3f}")
        logger.info(f"Match context factors applied - Home style: {home_profile['style']}, Away style: {away_profile['style']}")
        
        return round(home_xg, 3), round(away_xg, 3)
    
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

    def run_monte_carlo_simulation(self, home_xg: float, away_xg: float, iterations: int = 10000) -> MonteCarloResults:
        """Run Monte Carlo simulation - PURE PROBABILISTIC MODEL"""
        np.random.seed(42)
        
        # Adjust correlation based on match context
        if self.match_context == MatchContext.DEFENSIVE_BATTLE:
            lambda3_alpha = self.calibration_params['bivariate_correlation'] * 0.6
        elif self.match_context == MatchContext.OFFENSIVE_SHOWDOWN:
            lambda3_alpha = self.calibration_params['bivariate_correlation'] * 1.2
        else:
            lambda3_alpha = self.calibration_params['bivariate_correlation']
        
        lambda3 = lambda3_alpha * min(home_xg, away_xg)
        lambda1 = max(0.1, home_xg - lambda3)
        lambda2 = max(0.1, away_xg - lambda3)
        
        # Bivariate Poisson simulation
        C = np.random.poisson(lambda3, iterations)
        A = np.random.poisson(lambda1, iterations)
        B = np.random.poisson(lambda2, iterations)
        
        home_goals_sim = A + C
        away_goals_sim = B + C
        
        # Calculate probabilities
        home_wins = np.sum(home_goals_sim > away_goals_sim) / iterations
        draws = np.sum(home_goals_sim == away_goals_sim) / iterations
        away_wins = np.sum(home_goals_sim < away_goals_sim) / iterations
        
        total_goals = home_goals_sim + away_goals_sim
        over_25 = np.sum(total_goals > 2.5) / iterations
        btts = np.sum((home_goals_sim > 0) & (away_goals_sim > 0)) / iterations
        
        # Exact scores
        exact_scores = {}
        for i in range(6):
            for j in range(6):
                count = np.sum((home_goals_sim == i) & (away_goals_sim == j))
                prob = count / iterations
                if prob > 0.005:
                    exact_scores[f"{i}-{j}"] = round(prob * 100, 1)
        
        exact_scores = dict(sorted(exact_scores.items(), key=lambda x: x[1], reverse=True)[:8])
        
        # Confidence intervals
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
        
        # Probability volatility
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
            btts_prob=btts,
            exact_scores=exact_scores,
            confidence_intervals=confidence_intervals,
            probability_volatility=probability_volatility
        )

    def generate_predictions(self, mc_iterations: int = 10000) -> Dict[str, Any]:
        """Generate PURE football predictions - NO MARKET INFLUENCE"""
        
        home_team = self.data['home_team']
        away_team = self.data['away_team']
        
        # Calculate context-aware expected goals
        home_xg, away_xg = self._calculate_normalized_xg()
        
        # Determine match context
        self.match_context = self._determine_match_context(home_xg, away_xg, home_team, away_team)
        
        # Run Monte Carlo simulation
        mc_results = self.run_monte_carlo_simulation(home_xg, away_xg, mc_iterations)
        
        # Calculate additional probabilities
        total_xg = home_xg + away_xg
        first_half_prob = 1 - poisson.pmf(0, total_xg * 0.46)
        second_half_prob = 1 - poisson.pmf(0, total_xg * 0.54)
        
        # Calculate handicap probabilities
        handicap_probs = {}
        handicaps = [-0.5, 0, 0.5, 1.0]
        for handicap in handicaps:
            if handicap == 0:
                prob = 1 - skellam.cdf(0, home_xg, away_xg)
            elif handicap > 0:
                prob = 1 - skellam.cdf(-handicap, home_xg, away_xg)
            else:
                prob = 1 - skellam.cdf(abs(handicap), home_xg, away_xg)
            handicap_probs[f"handicap_{handicap}"] = round(prob * 100, 1)
        
        # Corner predictions
        base_corners = 10.0
        attacking_bonus = (home_xg + away_xg - 2.7) * 1.2
        total_corners = max(6, min(14, base_corners + attacking_bonus))
        
        # Risk assessment
        confidence_score = self._calculate_confidence_score(mc_results)
        risk_assessment = self._assess_prediction_risk(mc_results, confidence_score)
        
        return {
            'match': f"{home_team} vs {away_team}",
            'expected_goals': {'home': round(home_xg, 2), 'away': round(away_xg, 2)},
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
            'summary': self._generate_football_summary(home_team, away_team, home_xg, away_xg, mc_results),
            'confidence_score': confidence_score,
            'risk_assessment': risk_assessment,
            'monte_carlo_results': {
                'confidence_intervals': mc_results.confidence_intervals,
                'probability_volatility': mc_results.probability_volatility
            }
        }
    
    def _calculate_confidence_score(self, mc_results: MonteCarloResults) -> int:
        """Calculate confidence score based on data quality and simulation stability"""
        base_confidence = self.data['data_quality_score'] * 0.5
        
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
    
    def _assess_prediction_risk(self, mc_results: MonteCarloResults, confidence: int) -> Dict[str, str]:
        """Assess prediction risk - PURE FOOTBALL UNCERTAINTY"""
        home_win_prob = mc_results.home_win_prob
        draw_prob = mc_results.draw_prob
        away_win_prob = mc_results.away_win_prob
        
        highest_prob = max(home_win_prob, draw_prob, away_win_prob)
        
        # Calculate entropy for uncertainty measurement
        probs = np.array([home_win_prob, draw_prob, away_win_prob])
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
        
        if highest_prob > 0.7 and confidence > 80 and adjusted_uncertainty < 0.7:
            risk_level = "LOW"
            explanation = "Strong favorite with low uncertainty"
        elif highest_prob > 0.55 and confidence > 65 and adjusted_uncertainty < 0.85:
            risk_level = "MEDIUM"
            explanation = "Moderate favorite with acceptable uncertainty"
        else:
            risk_level = "HIGH"
            explanation = f"High uncertainty (entropy: {uncertainty_ratio:.2f})"
        
        return {
            'risk_level': risk_level,
            'explanation': explanation,
            'certainty': f"{highest_prob*100:.1f}%",
            'uncertainty': round(uncertainty_ratio, 2)
        }
    
    def _generate_timing_predictions(self, home_xg: float, away_xg: float) -> Dict[str, str]:
        """Generate timing predictions based on xG"""
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
    
    def _generate_football_summary(self, home_team: str, away_team: str, home_xg: float, 
                                 away_xg: float, mc_results: MonteCarloResults) -> str:
        """Generate pure football summary - NO MARKET TALK"""
        
        if self.match_context == MatchContext.DEFENSIVE_BATTLE:
            return f"Defensive stalemate expected between {home_team} and {away_team}. Both teams prioritize defensive solidity, suggesting a low-scoring affair where set-pieces and individual moments could prove decisive."
        
        elif self.match_context == MatchContext.TACTICAL_STALEMATE:
            return f"Tactical battle anticipated with minimal separation between {home_team} and {away_team}. Both teams well-organized defensively, suggesting a cagey encounter decided by fine margins."
        
        elif self.match_context == MatchContext.OFFENSIVE_SHOWDOWN:
            return f"Goals expected in this offensive showdown between {home_team} and {away_team}. Both teams favor attacking football, promising an open, end-to-end encounter with multiple scoring opportunities."
        
        elif self.match_context == MatchContext.HOME_DOMINANCE:
            return f"{home_team} demonstrate clear superiority with {home_xg:.1f} expected goals. Strong home advantage and defensive organization suggest controlled victory."
        
        elif self.match_context == MatchContext.AWAY_COUNTER:
            return f"{away_team} hold the tactical advantage with better expected goal metrics. Their counter-attacking threat could prove decisive against {home_team}'s defense."
        
        home_win_prob = mc_results.home_win_prob
        
        if home_win_prob > 0.65 and home_xg > away_xg + 0.8:
            return f"{home_team} demonstrate clear superiority with {home_xg:.1f} expected goals. Strong home advantage and defensive organization suggest controlled victory."
        
        elif home_win_prob > 0.55:
            return f"{home_team} hold measurable advantage with better expected goal metrics. {away_team} will need exceptional defensive discipline to contain home threat."
        
        else:
            return f"Competitive match with both teams creating opportunities. Small margins expected to determine outcome in what promises to be a tactical engagement."


class ValueDetectionEngine:
    """
    SEPARATE VALUE DETECTION ENGINE
    Only compares pure probabilities to market odds
    """
    
    def __init__(self):
        self.value_thresholds = {
            'EXCEPTIONAL': 10.0,
            'HIGH': 7.0,
            'GOOD': 5.0,
            'MODERATE': 3.0,
            'LOW': 0.0
        }
    
    def calculate_implied_probabilities(self, market_odds: Dict[str, float]) -> Dict[str, float]:
        """Convert decimal odds to implied probabilities"""
        implied_probs = {}
        
        for market, odds in market_odds.items():
            if isinstance(odds, (int, float)) and odds > 1:
                implied_probs[market] = 1 / odds
        
        return implied_probs
    
    def detect_value_bets(self, pure_probabilities: Dict, market_odds: Dict) -> List[BettingSignal]:
        """Detect value bets by comparing pure probabilities to market odds"""
        signals = []
        
        # Convert market odds to probabilities
        market_probs = self.calculate_implied_probabilities(market_odds)
        
        # Map between pure probability keys and market names
        probability_mapping = [
            ('probabilities.match_outcomes.home_win', '1x2 Home'),
            ('probabilities.match_outcomes.draw', '1x2 Draw'),
            ('probabilities.match_outcomes.away_win', '1x2 Away'),
            ('probabilities.over_under.over_25', 'Over 2.5 Goals'),
            ('probabilities.over_under.under_25', 'Under 2.5 Goals'),
            ('probabilities.both_teams_score.yes', 'BTTS Yes'),
            ('probabilities.both_teams_score.no', 'BTTS No')
        ]
        
        for prob_path, market_name in probability_mapping:
            # Extract pure probability
            pure_prob = self._get_nested_value(pure_probabilities, prob_path) / 100.0
            
            # Get market probability
            market_prob = market_probs.get(market_name, 0)
            
            if market_prob > 0 and pure_prob > 0:
                edge = (pure_prob - market_prob) * 100
                
                # Enhanced edge calculation with market confidence weighting
                adjusted_edge = self._apply_market_confidence_weighting(edge, market_prob)
                
                if adjusted_edge > self.value_thresholds['MODERATE']:
                    value_rating = self._get_value_rating(adjusted_edge)
                    confidence = self._calculate_bet_confidence(pure_prob, adjusted_edge)
                    stake = self._calculate_kelly_stake(pure_prob, market_prob)
                    
                    signals.append(BettingSignal(
                        market=market_name,
                        model_prob=round(pure_prob * 100, 1),
                        book_prob=round(market_prob * 100, 1),
                        edge=round(adjusted_edge, 1),
                        confidence=confidence,
                        recommended_stake=stake,
                        value_rating=value_rating
                    ))
        
        # Sort by edge descending
        signals.sort(key=lambda x: x.edge, reverse=True)
        return signals
    
    def _apply_market_confidence_weighting(self, edge: float, market_prob: float) -> float:
        """Adjust edge based on market confidence and efficiency"""
        # Markets with extreme probabilities are often less efficient
        if market_prob < 0.1 or market_prob > 0.9:
            return edge * 0.8  # Reduce edge for extreme probabilities
        
        # Medium-range probabilities are typically more efficient markets
        if 0.3 < market_prob < 0.7:
            return edge * 1.1  # Slight boost for efficient markets
        
        return edge
    
    def _get_nested_value(self, data: Dict, path: str):
        """Get nested value from dictionary using dot notation"""
        keys = path.split('.')
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return 0
        return current
    
    def _get_value_rating(self, edge: float) -> str:
        """Get value rating based on edge percentage"""
        for rating, threshold in self.value_thresholds.items():
            if edge >= threshold:
                return rating
        return "LOW"
    
    def _calculate_bet_confidence(self, pure_prob: float, edge: float) -> str:
        """Calculate betting confidence"""
        if pure_prob > 0.6 and edge > 8:
            return "HIGH"
        elif pure_prob > 0.55 and edge > 5:
            return "MEDIUM"
        elif pure_prob > 0.5 and edge > 3:
            return "LOW"
        else:
            return "SPECULATIVE"
    
    def _calculate_kelly_stake(self, pure_prob: float, market_prob: float, kelly_fraction: float = 0.25) -> float:
        """Calculate Kelly criterion stake"""
        if market_prob <= 0 or pure_prob <= market_prob:
            return 0.0
        
        decimal_odds = 1 / market_prob
        kelly_stake = (pure_prob * decimal_odds - 1) / (decimal_odds - 1)
        return max(0.0, min(0.05, kelly_stake * kelly_fraction))


# Main orchestrator class
class AdvancedFootballPredictor:
    """
    ORCHESTRATOR: Coordinates pure prediction engine with value detection
    Maintains strict separation of concerns
    """
    
    def __init__(self, match_data: Dict[str, Any], calibration_data: Optional[Dict] = None):
        # Extract market data for separate processing
        self.market_odds = match_data.get('market_odds', {})
        
        # Create pure football data (remove market influence)
        football_data = match_data.copy()
        if 'market_odds' in football_data:
            del football_data['market_odds']
        
        # Initialize engines
        self.signal_engine = SignalEngine(football_data, calibration_data)
        self.value_engine = ValueDetectionEngine()
        self.prediction_history = []
    
    def generate_comprehensive_analysis(self, mc_iterations: int = 10000) -> Dict[str, Any]:
        """Generate comprehensive analysis with strict separation"""
        
        # Step 1: Generate PURE football predictions
        football_predictions = self.signal_engine.generate_predictions(mc_iterations)
        
        # Step 2: SEPARATELY detect value bets
        value_signals = self.value_engine.detect_value_bets(football_predictions, self.market_odds)
        
        # Step 3: Combine results (no feedback between systems)
        comprehensive_result = football_predictions.copy()
        comprehensive_result['betting_signals'] = [signal.__dict__ for signal in value_signals]
        
        # Add bias monitoring
        comprehensive_result['bias_monitoring'] = self._calculate_bias_metrics(football_predictions)
        
        # Store prediction for historical tracking
        self._store_prediction_history(football_predictions)
        
        return comprehensive_result
    
    def _calculate_bias_metrics(self, football_predictions: Dict) -> Dict[str, float]:
        """Calculate bias monitoring metrics"""
        return {
            'pure_probability_entropy': self._calculate_entropy(football_predictions['probabilities']['match_outcomes']),
            'data_quality_score': football_predictions['data_quality_score'],
            'match_context': football_predictions['match_context'],
            'total_predictions_tracked': len(self.prediction_history)
        }
    
    def _calculate_entropy(self, outcomes: Dict[str, float]) -> float:
        """Calculate Shannon entropy of outcome probabilities"""
        probs = np.array([v / 100 for v in outcomes.values()])
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return round(entropy, 3)
    
    def _store_prediction_history(self, prediction: Dict):
        """Store prediction for historical tracking and learning"""
        prediction_record = {
            'timestamp': datetime.now().isoformat(),
            'match': prediction['match'],
            'expected_goals': prediction['expected_goals'],
            'probabilities': prediction['probabilities']['match_outcomes'],
            'match_context': prediction['match_context'],
            'confidence_score': prediction['confidence_score'],
            'data_quality': prediction['data_quality_score']
        }
        self.prediction_history.append(prediction_record)
        
        # Keep only last 100 predictions to manage memory
        if len(self.prediction_history) > 100:
            self.prediction_history = self.prediction_history[-100:]
    
    def get_prediction_history(self) -> List[Dict]:
        """Get prediction history for analysis"""
        return self.prediction_history


# Example usage demonstrating the separation
if __name__ == "__main__":
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
        'home_form': [3, 1, 3, 0, 1, 3],
        'away_form': [0, 1, 1, 3, 0, 1],
        'h2h_data': {
            'matches': 6,
            'home_wins': 4,
            'away_wins': 1, 
            'draws': 1,
            'home_goals': 9,
            'away_goals': 4
        },
        'injuries': {'home': 1, 'away': 2},
        'motivation': {'home': 'High', 'away': 'Normal'},
        'market_odds': {
            '1x2 Home': 2.10,
            '1x2 Draw': 3.10,
            '1x2 Away': 3.80,
            'Over 2.5 Goals': 2.30,
            'BTTS Yes': 1.95
        }
    }
    
    # Use the orchestrator
    predictor = AdvancedFootballPredictor(match_data)
    results = predictor.generate_comprehensive_analysis()
    
    print("COMPREHENSIVE ANALYSIS WITH SEPARATION OF CONCERNS:")
    print(f"Match: {results['match']}")
    print(f"Expected Goals: Home {results['expected_goals']['home']:.2f} - Away {results['expected_goals']['away']:.2f}")
    print(f"Match Context: {results['match_context']}")
    print(f"Pure Probabilities: {results['probabilities']['match_outcomes']}")
    print(f"Betting Signals: {len(results['betting_signals'])} value bets detected")
    
    for signal in results['betting_signals']:
        print(f"  {signal['market']}: +{signal['edge']}% edge")
