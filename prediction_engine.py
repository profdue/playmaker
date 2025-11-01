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

class SignalEngine:
    """
    REALISTIC PREDICTIVE ENGINE - Balanced football probabilities
    """
    
    def __init__(self, match_data: Dict[str, Any], calibration_data: Optional[Dict] = None):
        self.data = self._validate_and_enhance_data(match_data)
        self.calibration_data = calibration_data or {}
        self.league_contexts = self._initialize_league_contexts()
        self.team_profiles = self._initialize_team_profiles()
        self._setup_balanced_parameters()
        self.match_context = MatchContext.UNPREDICTABLE
        
    def _validate_and_enhance_data(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced data validation with balanced features"""
        required_fields = ['home_team', 'away_team']
        
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
            'home_xg': (0, 5, 1.3),
            'away_xg': (0, 5, 1.1),
            'home_shots': (0, 30, 12),
            'away_shots': (0, 30, 10),
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
        
        # Calculate performance trends
        enhanced_data = self._calculate_performance_trends(enhanced_data)
        
        return enhanced_data
    
    def _calculate_data_quality(self, data: Dict) -> float:
        """Calculate data quality score"""
        score = 0
        max_score = 0
        
        # Basic match info
        if data.get('home_team') and data.get('away_team'):
            score += 15
        max_score += 15
        
        # Advanced metrics
        if data.get('home_xg', 0) > 0:
            score += 10
        if data.get('away_xg', 0) > 0:
            score += 10
        max_score += 20
        
        # Shot data
        if data.get('home_shots', 0) > 0:
            score += 5
        if data.get('away_shots', 0) > 0:
            score += 5
        max_score += 10
        
        # Recent form with sufficient data
        if len(data.get('home_form', [])) >= 4:
            score += 10
        if len(data.get('away_form', [])) >= 4:
            score += 10
        max_score += 20
        
        # H2H data
        h2h_data = data.get('h2h_data', {})
        if h2h_data.get('matches', 0) >= 3:
            score += 15
        max_score += 15
        
        # Motivation and injuries
        if data.get('motivation', {}).get('home') != 'Normal':
            score += 5
        if data.get('motivation', {}).get('away') != 'Normal':
            score += 5
        max_score += 10
        
        return (score / max_score) * 100
    
    def _calculate_performance_trends(self, data: Dict) -> Dict:
        """Calculate performance trends"""
        for team in ['home', 'away']:
            form = data.get(f'{team}_form', [])
            if len(form) >= 4:
                recent_games = min(3, len(form))
                earlier_games = min(3, len(form) - recent_games)
                
                if earlier_games > 0:
                    recent_avg = np.mean(form[:recent_games])
                    earlier_avg = np.mean(form[recent_games:recent_games + earlier_games])
                    trend = (recent_avg - earlier_avg) / max(earlier_avg, 0.1)
                    data[f'{team}_trend'] = trend
                else:
                    data[f'{team}_trend'] = 0.0
                
                data[f'{team}_consistency'] = np.std(form) / max(np.mean(form), 0.1) if len(form) > 1 else 1.0
            else:
                data[f'{team}_trend'] = 0.0
                data[f'{team}_consistency'] = 1.0
        
        return data
    
    def _initialize_league_contexts(self) -> Dict[str, Dict]:
        """League-specific parameters"""
        return {
            'premier_league': {'avg_goals': 2.8, 'home_advantage': 1.15, 'goal_variance': 1.15},
            'la_liga': {'avg_goals': 2.6, 'home_advantage': 1.18, 'goal_variance': 1.10},
            'serie_a': {'avg_goals': 2.7, 'home_advantage': 1.20, 'goal_variance': 1.05},
            'bundesliga': {'avg_goals': 3.1, 'home_advantage': 1.12, 'goal_variance': 1.20},
            'ligue_1': {'avg_goals': 2.5, 'home_advantage': 1.18, 'goal_variance': 1.08},
            'liga_portugal': {'avg_goals': 2.6, 'home_advantage': 1.22, 'goal_variance': 1.10},
            'default': {'avg_goals': 2.7, 'home_advantage': 1.18, 'goal_variance': 1.12}
        }
    
    def _initialize_team_profiles(self) -> Dict[str, Dict]:
        """Team playing styles"""
        base_profiles = {
            'Sporting CP': {'style': 'attacking', 'press_intensity': 'high', 'clean_sheet_freq': 0.35},
            'FC Alverca': {'style': 'defensive', 'press_intensity': 'medium', 'clean_sheet_freq': 0.25},
            'Bologna': {'style': 'balanced', 'press_intensity': 'high', 'clean_sheet_freq': 0.45},
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
    
    def _setup_balanced_parameters(self):
        """BALANCED calibration parameters"""
        self.calibration_params = {
            'home_advantage': 1.20,  # Balanced home advantage
            'form_decay_rate': 0.85,
            'h2h_weight': 0.15,
            'injury_impact': 0.06,
            'motivation_impact': 0.08,
            'bivariate_correlation': 0.10,
            'data_quality_threshold': 50.0,
            'trend_weight': 0.12,
            'consistency_weight': 0.10,
            'regression_factor': 0.85,
            'surprise_factor': 1.02,
        }
        
        if self.calibration_data:
            self.calibration_params.update(self.calibration_data)

    def _calculate_motivation_impact(self, motivation_level: str, match_context: str) -> float:
        """Context-aware motivation impact"""
        base_multipliers = {
            "Low": 0.90, "Normal": 1.0, "High": 1.08, "Very High": 1.12,
            "low": 0.90, "normal": 1.0, "high": 1.08, "very high": 1.12
        }
        
        motivation_multiplier = base_multipliers.get(motivation_level, 1.0)
        
        context_adjustments = {
            'defensive_battle': 0.98,
            'offensive_showdown': 1.05, 
            'home_dominance': 1.03,
            'away_counter': 1.05,
            'tactical_stalemate': 1.0,
            'unpredictable': 0.95
        }
        
        adjustment = context_adjustments.get(match_context, 1.0)
        return motivation_multiplier * adjustment

    def _determine_match_context(self, home_xg: float, away_xg: float, home_team: str, away_team: str) -> MatchContext:
        """Balanced context determination"""
        home_profile = self.team_profiles.get(home_team, self.team_profiles['default'])
        away_profile = self.team_profiles.get(away_team, self.team_profiles['default'])
        
        total_xg = home_xg + away_xg
        xg_difference = home_xg - away_xg
        
        # Home dominance detection
        if xg_difference > 0.6:
            return MatchContext.HOME_DOMINANCE
        elif xg_difference < -0.4:
            return MatchContext.AWAY_COUNTER
        
        # Check for unpredictability
        home_consistency = self.data.get('home_consistency', 1.0)
        away_consistency = self.data.get('away_consistency', 1.0)
        
        if home_consistency > 0.8 or away_consistency > 0.8:
            return MatchContext.UNPREDICTABLE
        
        # Standard context detection
        if (home_profile['style'] == 'defensive' and away_profile['style'] == 'defensive' 
            and total_xg < 2.4):
            return MatchContext.DEFENSIVE_BATTLE
        elif (home_profile['style'] == 'attacking' and away_profile['style'] == 'attacking'
              and total_xg > 3.0):
            return MatchContext.OFFENSIVE_SHOWDOWN
        elif total_xg < 2.7 and abs(xg_difference) < 0.3:
            return MatchContext.TACTICAL_STALEMATE
        
        return MatchContext.UNPREDICTABLE

    def _apply_football_sanity_checks(self, home_xg: float, away_xg: float) -> Tuple[float, float]:
        """Apply football reality checks to xG outputs"""
        # Get team profiles
        home_profile = self.team_profiles.get(self.data['home_team'], self.team_profiles['default'])
        away_profile = self.team_profiles.get(self.data['away_team'], self.team_profiles['default'])
        
        # Ensure realistic home advantage
        min_home_advantage = 0.3
        if home_xg - away_xg < min_home_advantage:
            adjustment = (min_home_advantage - (home_xg - away_xg)) / 2
            home_xg += adjustment
            away_xg = max(0.4, away_xg - adjustment)
        
        # Ensure realistic bounds
        home_xg = max(0.5, min(3.2, home_xg))
        away_xg = max(0.4, min(2.2, away_xg))
        
        logger.info(f"Football sanity check: Home xG: {home_xg:.2f}, Away xG: {away_xg:.2f}")
        return home_xg, away_xg

    def _calculate_predictive_xg(self) -> Tuple[float, float]:
        """Calculate realistic predictive xG"""
        league = self.data.get('league', 'liga_portugal')
        league_params = self.league_contexts.get(league, self.league_contexts['default'])
        league_avg_goals = league_params['avg_goals'] / 2
        
        # Base xG from goals
        home_base_xg = self.data.get('home_xg', self.data.get('home_goals', 0) / 5.0)
        away_base_xg = self.data.get('away_xg', self.data.get('away_goals', 0) / 5.0)
        
        # Defensive strength
        home_xg_against = self.data.get('home_xg_against', self.data.get('home_conceded', 0) / 5.0)
        away_xg_against = self.data.get('away_xg_against', self.data.get('away_conceded', 0) / 5.0)
        
        home_defensive_strength = 1 - (home_xg_against / league_avg_goals)
        away_defensive_strength = 1 - (away_xg_against / league_avg_goals)
        
        # Home advantage application
        home_xg = home_base_xg * (1 - away_defensive_strength * 0.3)
        away_xg = away_base_xg * (1 - home_defensive_strength * 0.3)
        
        home_xg *= league_params['home_advantage']
        
        # Form factors
        home_form_factor = self._calculate_form_factor('home')
        away_form_factor = self._calculate_form_factor('away')
        
        home_xg *= home_form_factor
        away_xg *= away_form_factor
        
        # Trends
        home_trend = self.data.get('home_trend', 0.0)
        away_trend = self.data.get('away_trend', 0.0)
        
        home_xg *= (1 + home_trend * self.calibration_params['trend_weight'])
        away_xg *= (1 + away_trend * self.calibration_params['trend_weight'])
        
        # Consistency
        home_consistency = self.data.get('home_consistency', 1.0)
        away_consistency = self.data.get('away_consistency', 1.0)
        
        home_xg *= (1 - (home_consistency - 0.7) * self.calibration_params['consistency_weight'])
        away_xg *= (1 - (away_consistency - 0.7) * self.calibration_params['consistency_weight'])
        
        # Regression to mean
        home_xg = (home_xg * self.calibration_params['regression_factor'] + 
                  league_avg_goals * (1 - self.calibration_params['regression_factor']))
        away_xg = (away_xg * self.calibration_params['regression_factor'] + 
                  league_avg_goals * (1 - self.calibration_params['regression_factor']))
        
        # Motivation and injuries
        motivation = self.data.get('motivation', {})
        home_motivation_level = motivation.get('home', 'Normal')
        away_motivation_level = motivation.get('away', 'Normal')
        
        temp_context = self._determine_match_context(home_xg, away_xg, 
                                                   self.data['home_team'], self.data['away_team'])
        
        home_motivation_factor = self._calculate_motivation_impact(home_motivation_level, temp_context.value)
        away_motivation_factor = self._calculate_motivation_impact(away_motivation_level, temp_context.value)
        
        injuries = self.data.get('injuries', {})
        home_injuries = float(injuries.get('home', 0))
        away_injuries = float(injuries.get('away', 0))
        
        home_injury_factor = max(0.8, 1.0 - (home_injuries * self.calibration_params['injury_impact']))
        away_injury_factor = max(0.8, 1.0 - (away_injuries * self.calibration_params['injury_impact']))
        
        home_xg *= home_injury_factor * home_motivation_factor
        away_xg *= away_injury_factor * away_motivation_factor
        
        # H2H adjustment
        h2h_data = self.data.get('h2h_data', {})
        if h2h_data and h2h_data.get('matches', 0) >= 3:
            home_xg, away_xg = self._apply_h2h_adjustment(home_xg, away_xg, h2h_data)
        
        # Team style adjustments
        home_profile = self.team_profiles.get(self.data['home_team'], self.team_profiles['default'])
        away_profile = self.team_profiles.get(self.data['away_team'], self.team_profiles['default'])
        
        if home_profile['style'] == 'defensive':
            home_xg *= 0.90
            away_xg *= 0.85
        elif home_profile['style'] == 'attacking':
            home_xg *= 1.10
            away_xg *= 1.05
            
        if away_profile['style'] == 'defensive':
            away_xg *= 0.90
            home_xg *= 0.85
        elif away_profile['style'] == 'attacking':
            away_xg *= 1.10
            home_xg *= 1.05
        
        # Apply football sanity checks
        home_xg, away_xg = self._apply_football_sanity_checks(home_xg, away_xg)
        
        logger.info(f"Predictive xG - Home: {home_xg:.3f}, Away: {away_xg:.3f}")
        
        return round(home_xg, 3), round(away_xg, 3)
    
    def _calculate_form_factor(self, team: str) -> float:
        """Calculate form factor"""
        form = self.data.get(f'{team}_form', [])
        if not form or len(form) == 0:
            return 1.0
        
        try:
            form_scores = [float(score) for score in form]
        except (TypeError, ValueError):
            return 1.0
        
        weights = [self.calibration_params['form_decay_rate'] ** i for i in range(len(form_scores))]
        weights = [w / sum(weights) for w in weights]
        
        total_points = sum(score * weight for score, weight in zip(form_scores, reversed(weights)))
        max_possible = sum(3 * weight for weight in weights)
        
        form_ratio = total_points / max_possible if max_possible > 0 else 0.5
        
        return 0.8 + (form_ratio * 0.4)  # 0.8 to 1.2 range
    
    def _apply_h2h_adjustment(self, home_xg: float, away_xg: float, h2h_data: Dict) -> Tuple[float, float]:
        """Apply H2H adjustment"""
        matches = h2h_data.get('matches', 0)
        home_goals = h2h_data.get('home_goals', 0)
        away_goals = h2h_data.get('away_goals', 0)
        
        if matches < 3:
            return home_xg, away_xg
        
        h2h_weight = min(0.3, matches * 0.08)
        h2h_home_avg = home_goals / matches
        h2h_away_avg = away_goals / matches
        
        adjusted_home_xg = (home_xg * (1 - h2h_weight)) + (h2h_home_avg * h2h_weight)
        adjusted_away_xg = (away_xg * (1 - h2h_weight)) + (h2h_away_avg * h2h_weight)
        
        return adjusted_home_xg, adjusted_away_xg

    def run_monte_carlo_simulation(self, home_xg: float, away_xg: float, iterations: int = 10000) -> MonteCarloResults:
        """Run Monte Carlo simulation"""
        np.random.seed(42)
        
        if self.match_context == MatchContext.DEFENSIVE_BATTLE:
            lambda3_alpha = self.calibration_params['bivariate_correlation'] * 0.7
        elif self.match_context == MatchContext.OFFENSIVE_SHOWDOWN:
            lambda3_alpha = self.calibration_params['bivariate_correlation'] * 1.2
        elif self.match_context == MatchContext.UNPREDICTABLE:
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
        
        home_wins = np.sum(home_goals_sim > away_goals_sim) / iterations
        draws = np.sum(home_goals_sim == away_goals_sim) / iterations
        away_wins = np.sum(home_goals_sim < away_goals_sim) / iterations
        
        total_goals = home_goals_sim + away_goals_sim
        over_25 = np.sum(total_goals > 2.5) / iterations
        btts = np.sum((home_goals_sim > 0) & (away_goals_sim > 0)) / iterations
        
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
            btts_prob=btts,
            exact_scores=exact_scores,
            confidence_intervals=confidence_intervals,
            probability_volatility=probability_volatility
        )

    def _validate_probability_sanity(self, home_win: float, draw: float, away_win: float) -> bool:
        """Validate that probabilities make football sense"""
        total = home_win + draw + away_win
        
        # Check normalization
        if not 0.99 <= total <= 1.01:
            logger.error(f"Probabilities not normalized: {total}")
            return False
        
        # Basic football sanity - home should usually be favorite
        if home_win < draw or home_win < away_win:
            logger.warning(f"Home not favorite: Home={home_win}, Draw={draw}, Away={away_win}")
            return False
            
        return True

    def generate_predictions(self, mc_iterations: int = 10000) -> Dict[str, Any]:
        """Generate football predictions with sanity checks"""
        
        home_team = self.data['home_team']
        away_team = self.data['away_team']
        
        home_xg, away_xg = self._calculate_predictive_xg()
        
        self.match_context = self._determine_match_context(home_xg, away_xg, home_team, away_team)
        
        mc_results = self.run_monte_carlo_simulation(home_xg, away_xg, mc_iterations)
        
        # Validate probability sanity
        if not self._validate_probability_sanity(mc_results.home_win_prob, mc_results.draw_prob, mc_results.away_win_prob):
            logger.warning("Probability sanity check failed - applying corrections")
            # Apply emergency corrections
            total = mc_results.home_win_prob + mc_results.draw_prob + mc_results.away_win_prob
            mc_results.home_win_prob = max(mc_results.home_win_prob, 0.45)  # Ensure home favorite
            mc_results.home_win_prob /= total
            mc_results.draw_prob /= total  
            mc_results.away_win_prob /= total
        
        total_xg = home_xg + away_xg
        first_half_prob = 1 - poisson.pmf(0, total_xg * 0.46)
        second_half_prob = 1 - poisson.pmf(0, total_xg * 0.54)
        
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
        
        base_corners = 10.0
        attacking_bonus = (home_xg + away_xg - 2.7) * 1.0
        total_corners = max(6, min(14, base_corners + attacking_bonus))
        
        confidence_score = self._calculate_confidence(mc_results)
        risk_assessment = self._assess_risk(mc_results, confidence_score, home_xg, away_xg)
        
        predictions = {
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
            'summary': self._generate_summary(home_team, away_team, home_xg, away_xg, mc_results),
            'confidence_score': confidence_score,
            'risk_assessment': risk_assessment,
            'predictive_insights': self._generate_insights(),
            'monte_carlo_results': {
                'confidence_intervals': mc_results.confidence_intervals,
                'probability_volatility': mc_results.probability_volatility
            }
        }
        
        return predictions
    
    def _calculate_confidence(self, mc_results: MonteCarloResults) -> int:
        """Calculate confidence score"""
        base_confidence = self.data['data_quality_score'] * 0.6
        
        if self.data.get('home_xg', 0) > 0:
            base_confidence += 5
        if self.data.get('away_xg', 0) > 0:
            base_confidence += 5
            
        home_consistency = self.data.get('home_consistency', 1.0)
        away_consistency = self.data.get('away_consistency', 1.0)
        
        if home_consistency < 0.8:
            base_confidence += 8
        if away_consistency < 0.8:
            base_confidence += 8
        
        avg_volatility = np.mean(list(mc_results.probability_volatility.values()))
        volatility_penalty = min(25, int(avg_volatility * 600))
        
        if self.match_context == MatchContext.UNPREDICTABLE:
            base_confidence -= 15
        
        confidence = base_confidence - volatility_penalty
        return max(15, min(90, int(confidence)))
    
    def _assess_risk(self, mc_results: MonteCarloResults, confidence: int, home_xg: float, away_xg: float) -> Dict[str, str]:
        """Balanced risk assessment"""
        home_win_prob = mc_results.home_win_prob
        
        if (home_win_prob > 0.60 and confidence > 70 and 
            self.data['data_quality_score'] > 75 and
            home_xg - away_xg > 0.8):
            risk_level = "MEDIUM"
            explanation = "Good statistical advantage with reasonable confidence"
            recommendation = "CONSIDER SMALL STAKE"
        elif (home_win_prob > 0.55 and confidence > 65 and
              self.data['data_quality_score'] > 65):
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

    def _generate_insights(self) -> Dict[str, str]:
        """Generate predictive insights"""
        insights = {}
        
        home_trend = self.data.get('home_trend', 0.0)
        away_trend = self.data.get('away_trend', 0.0)
        
        if home_trend > 0.12:
            insights['home_trend'] = f"Home team showing strong improvement"
        elif home_trend > 0.06:
            insights['home_trend'] = f"Home team showing positive trend"
        elif home_trend < -0.10:
            insights['home_trend'] = f"Home team showing concerning decline"
            
        if away_trend > 0.12:
            insights['away_trend'] = f"Away team showing strong improvement"
        elif away_trend > 0.06:
            insights['away_trend'] = f"Away team showing positive trend" 
        elif away_trend < -0.10:
            insights['away_trend'] = f"Away team showing concerning decline"
        
        home_consistency = self.data.get('home_consistency', 1.0)
        away_consistency = self.data.get('away_consistency', 1.0)
        
        if home_consistency > 0.8:
            insights['home_consistency'] = "Home team inconsistent - higher risk"
        elif home_consistency > 0.7:
            insights['home_consistency'] = "Home team somewhat inconsistent"
            
        if away_consistency > 0.8:
            insights['away_consistency'] = "Away team inconsistent - higher risk"
        elif away_consistency > 0.7:
            insights['away_consistency'] = "Away team somewhat inconsistent"
        
        if self.match_context == MatchContext.UNPREDICTABLE:
            insights['context'] = "High unpredictability - teams show inconsistent form"
        elif self.match_context == MatchContext.DEFENSIVE_BATTLE:
            insights['context'] = "Defensive battle expected - low scoring likely"
        elif self.match_context == MatchContext.HOME_DOMINANCE:
            insights['context'] = "Home dominance expected"
        
        return insights

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
            'most_action': "Second half more likely" if total_xg > 2.5 else "Evenly distributed"
        }
    
    def _generate_summary(self, home_team: str, away_team: str, home_xg: float, 
                        away_xg: float, mc_results: MonteCarloResults) -> str:
        """Generate football summary"""
        
        home_win_prob = mc_results.home_win_prob
        
        if self.match_context == MatchContext.HOME_DOMINANCE and home_win_prob > 0.60:
            return f"{home_team} show clear superiority with {home_xg:.1f} expected goals. Home advantage and better metrics suggest comfortable victory against {away_team}."
        
        elif self.match_context == MatchContext.UNPREDICTABLE:
            return f"High unpredictability expected between {home_team} and {away_team}. Both teams show inconsistent recent performances, making this match difficult to forecast."
        
        elif self.match_context == MatchContext.DEFENSIVE_BATTLE:
            return f"Defensive stalemate anticipated between {home_team} and {away_team}. Both teams prioritize defensive solidity with low expected goals ({home_xg:.1f} - {away_xg:.1f}), suggesting a cagey affair."
        
        elif home_win_prob > 0.55 and home_xg > away_xg + 0.5:
            return f"{home_team} hold advantage with better expected goal metrics. {away_team} will need good performance to contain the home threat."
        
        else:
            return f"Competitive match expected. Small margins likely to determine outcome in what promises to be a closely-fought encounter."


class ValueDetectionEngine:
    """
    REALISTIC VALUE DETECTION ENGINE - Balanced edge detection
    """
    
    def __init__(self):
        # Realistic thresholds for professional betting
        self.value_thresholds = {
            'EXCEPTIONAL': 25.0,   # 25%+ edge - rare
            'HIGH': 15.0,          # 15%+ edge  
            'GOOD': 8.0,           # 8%+ edge
            'MODERATE': 4.0,       # 4%+ edge
        }
        self.min_confidence = 60    # Reasonable minimum confidence
        self.min_probability = 0.10  # Reasonable minimum probability
        self.max_stake = 0.03       # Conservative maximum stake
        
    def calculate_implied_probabilities(self, market_odds: Dict[str, float]) -> Dict[str, float]:
        """Convert decimal odds to implied probabilities"""
        implied_probs = {}
        
        for market, odds in market_odds.items():
            if isinstance(odds, (int, float)) and odds > 1:
                implied_prob = 1.0 / odds
                
                # Reasonable sanity check
                if 0.03 <= implied_prob <= 0.85:
                    implied_probs[market] = implied_prob
                else:
                    logger.warning(f"Implied probability {implied_prob:.3f} for {market} outside reasonable range")
                    implied_probs[market] = 0.0
        
        return implied_probs
    
    def _validate_probability_sanity(self, pure_probabilities: Dict) -> bool:
        """Validate that pure probabilities make football sense"""
        try:
            home_win = pure_probabilities['probabilities']['match_outcomes']['home_win'] / 100.0
            draw = pure_probabilities['probabilities']['match_outcomes']['draw'] / 100.0
            away_win = pure_probabilities['probabilities']['match_outcomes']['away_win'] / 100.0
            
            total = home_win + draw + away_win
            
            # Check normalization
            if not 0.98 <= total <= 1.02:
                logger.error(f"Pure probabilities not normalized: {total}")
                return False
            
            # Check for football reality
            if home_win < 0.35:  # Home probability too low
                logger.warning(f"Home probability unrealistically low: {home_win}")
                return False
                
            return True
            
        except (KeyError, TypeError) as e:
            logger.error(f"Error validating probabilities: {e}")
            return False
    
    def detect_value_bets(self, pure_probabilities: Dict, market_odds: Dict) -> List[BettingSignal]:
        """Realistic value bet detection with sanity checks"""
        
        # Validate pure probabilities first
        if not self._validate_probability_sanity(pure_probabilities):
            logger.warning("Pure probabilities failed sanity check - returning no signals")
            return []
        
        signals = []
        
        market_probs = self.calculate_implied_probabilities(market_odds)
        
        # Get and normalize probabilities
        home_pure = pure_probabilities['probabilities']['match_outcomes']['home_win'] / 100.0
        draw_pure = pure_probabilities['probabilities']['match_outcomes']['draw'] / 100.0  
        away_pure = pure_probabilities['probabilities']['match_outcomes']['away_win'] / 100.0
        
        # Normalize to ensure they sum to 1.0
        total = home_pure + draw_pure + away_pure
        if total > 0:
            home_pure /= total
            draw_pure /= total
            away_pure /= total
        
        # Correct market mapping
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
                
                # Only add if stake is meaningful
                if stake > 0.001 and stake <= self.max_stake:
                    signals.append(BettingSignal(
                        market=market_name,
                        model_prob=round(pure_prob * 100, 1),
                        book_prob=round(market_prob * 100, 1),
                        edge=round(adjusted_edge, 1),
                        confidence=confidence,
                        recommended_stake=stake,
                        value_rating=value_rating
                    ))
        
        # Enforce mutual exclusivity for 1X2 markets
        signals = self._enforce_mutual_exclusivity(signals)
        
        # Sort by edge and return
        signals.sort(key=lambda x: x.edge, reverse=True)
        return signals
    
    def _enforce_mutual_exclusivity(self, signals: List[BettingSignal]) -> List[BettingSignal]:
        """Ensure only one 1X2 bet is recommended"""
        one_x_two_signals = [s for s in signals if s.market in ['1x2 Home', '1x2 Draw', '1x2 Away']]
        other_signals = [s for s in signals if s.market not in ['1x2 Home', '1x2 Draw', '1x2 Away']]
        
        if len(one_x_two_signals) > 1:
            # Keep only the 1X2 signal with highest edge AND reasonable probability
            valid_signals = [s for s in one_x_two_signals if s.model_prob >= 12]  # Minimum 12% probability
            if valid_signals:
                best_signal = max(valid_signals, key=lambda x: x.edge)
                return [best_signal] + other_signals
            else:
                return other_signals
        
        return signals
    
    def _apply_market_wisdom(self, edge: float, market_prob: float, pure_prob: float) -> float:
        """Market efficiency adjustments"""
        # Reduce edge for extreme probabilities
        if market_prob < 0.12 or market_prob > 0.85:
            return edge * 0.6
        
        # Moderate reduction for medium probabilities
        if market_prob < 0.20 or market_prob > 0.75:
            return edge * 0.75
        
        # Small reduction for normal ranges
        if market_prob < 0.30 or market_prob > 0.65:
            return edge * 0.85
        
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
        
        # Kelly criterion
        decimal_odds = 1 / market_prob
        kelly_stake = (pure_prob * decimal_odds - 1) / (decimal_odds - 1)
        
        # Confidence adjustment
        confidence_factor = max(0.4, confidence / 100)
        
        # Edge-based cap
        base_stake = kelly_stake * kelly_fraction * confidence_factor
        edge_cap = min(self.max_stake, edge / 1000)
        
        final_stake = min(base_stake, edge_cap)
        
        return max(0.001, min(self.max_stake, final_stake))


class AdvancedFootballPredictor:
    """
    BALANCED ORCHESTRATOR: Professional integration with validation
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
        """Generate comprehensive analysis with validation"""
        
        football_predictions = self.signal_engine.generate_predictions(mc_iterations)
        
        value_signals = self.value_engine.detect_value_bets(football_predictions, self.market_odds)
        
        comprehensive_result = football_predictions.copy()
        comprehensive_result['betting_signals'] = [signal.__dict__ for signal in value_signals]
        
        comprehensive_result['bias_monitoring'] = self._calculate_bias_metrics(football_predictions)
        
        self._store_prediction_history(football_predictions)
        
        return comprehensive_result
    
    def _calculate_bias_metrics(self, football_predictions: Dict) -> Dict[str, float]:
        """Calculate bias monitoring metrics"""
        return {
            'pure_probability_entropy': self._calculate_entropy(football_predictions['probabilities']['match_outcomes']),
            'data_quality_score': football_predictions['data_quality_score'],
            'match_context': football_predictions['match_context'],
            'total_predictions_tracked': len(self.prediction_history),
            'home_win_probability': football_predictions['probabilities']['match_outcomes']['home_win']
        }
    
    def _calculate_entropy(self, outcomes: Dict[str, float]) -> float:
        """Calculate Shannon entropy of outcome probabilities"""
        probs = np.array([v / 100 for v in outcomes.values()])
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return round(entropy, 3)
    
    def _store_prediction_history(self, prediction: Dict):
        """Store prediction for historical tracking"""
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
        
        if len(self.prediction_history) > 100:
            self.prediction_history = self.prediction_history[-100:]
    
    def get_prediction_history(self) -> List[Dict]:
        """Get prediction history for analysis"""
        return self.prediction_history
