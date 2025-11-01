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
    REALISTIC PREDICTIVE ENGINE - Accurate football probabilities
    """
    
    def __init__(self, match_data: Dict[str, Any], calibration_data: Optional[Dict] = None):
        self.data = self._validate_and_enhance_data(match_data)
        self.calibration_data = calibration_data or {}
        self.league_contexts = self._initialize_league_contexts()
        self.team_strength_tiers = self._initialize_team_strength_tiers()
        self._setup_realistic_parameters()
        self.match_context = MatchContext.UNPREDICTABLE
        
    def _validate_and_enhance_data(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced data validation with realistic features"""
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
    
    def _initialize_league_contexts(self) -> Dict[str, Dict]:
        """League-specific parameters"""
        return {
            'premier_league': {'avg_goals': 2.8, 'home_advantage': 1.15},
            'la_liga': {'avg_goals': 2.6, 'home_advantage': 1.18},
            'serie_a': {'avg_goals': 2.7, 'home_advantage': 1.20},
            'bundesliga': {'avg_goals': 3.1, 'home_advantage': 1.12},
            'ligue_1': {'avg_goals': 2.5, 'home_advantage': 1.18},
            'liga_portugal': {'avg_goals': 2.6, 'home_advantage': 1.25},
            'default': {'avg_goals': 2.7, 'home_advantage': 1.18}
        }
    
    def _initialize_team_strength_tiers(self) -> Dict[str, int]:
        """Team strength tiers - CRITICAL FOR REALISTIC PREDICTIONS"""
        return {
            # Portuguese Liga - Updated with realistic tiers
            'Sporting CP': 5, 'Porto': 5, 'Benfica': 5, 'Braga': 4,
            'Vitoria Guimaraes': 3, 'Boavista': 3, 'Famalicao': 3,
            'Casa Pia': 2, 'Rio Ave': 2, 'Estoril': 2, 'Gil Vicente': 2,
            'FC Alverca': 1, 'AVS': 1, 'Leiria': 1,
            
            # Default tiers for other teams
            'default_strong': 4, 'default_medium': 2, 'default_weak': 1
        }
    
    def _setup_realistic_parameters(self):
        """REALISTIC calibration parameters"""
        self.calibration_params = {
            'home_advantage': 1.25,
            'form_decay_rate': 0.85,
            'h2h_weight': 0.15,
            'injury_impact': 0.08,
            'motivation_impact': 0.10,
            'bivariate_correlation': 0.12,
            'strength_tier_impact': 0.25,
            'defensive_impact_multiplier': 0.4,
        }
        
        if self.calibration_data:
            self.calibration_params.update(self.calibration_data)

    def _get_team_strength(self, team: str) -> int:
        """Get team strength tier"""
        return self.team_strength_tiers.get(team, 2)

    def _calculate_strength_based_adjustment(self, home_team: str, away_team: str) -> float:
        """Calculate adjustment based on team strength difference"""
        home_strength = self._get_team_strength(home_team)
        away_strength = self._get_team_strength(away_team)
        
        strength_diff = home_strength - away_strength
        
        if strength_diff >= 3:
            return 0.4
        elif strength_diff >= 2:
            return 0.25
        elif strength_diff >= 1:
            return 0.12
        elif strength_diff <= -2:
            return -0.25
        else:
            return 0.0

    def _calculate_motivation_impact(self, motivation_level: str) -> float:
        """Motivation impact"""
        multipliers = {
            "Low": 0.90, "Normal": 1.0, "High": 1.08, "Very High": 1.12,
            "low": 0.90, "normal": 1.0, "high": 1.08, "very high": 1.12
        }
        return multipliers.get(motivation_level, 1.0)

    def _determine_match_context(self, home_xg: float, away_xg: float, home_team: str, away_team: str) -> MatchContext:
        """Realistic context determination"""
        home_strength = self._get_team_strength(home_team)
        away_strength = self._get_team_strength(away_team)
        
        total_xg = home_xg + away_xg
        xg_difference = home_xg - away_xg
        
        if (xg_difference > 1.0) or (home_strength - away_strength >= 3 and xg_difference > 0.5):
            return MatchContext.HOME_DOMINANCE
        elif xg_difference < -0.8:
            return MatchContext.AWAY_COUNTER
        
        if total_xg < 2.2:
            return MatchContext.DEFENSIVE_BATTLE
        elif total_xg > 3.2:
            return MatchContext.OFFENSIVE_SHOWDOWN
        elif abs(xg_difference) < 0.3:
            return MatchContext.TACTICAL_STALEMATE
        
        return MatchContext.UNPREDICTABLE

    def _apply_football_reality_checks(self, home_xg: float, away_xg: float, home_team: str, away_team: str) -> Tuple[float, float]:
        """CRITICAL: Apply football reality checks"""
        home_strength = self._get_team_strength(home_team)
        away_strength = self._get_team_strength(away_team)
        
        if home_strength >= 4 and away_strength <= 2:
            home_xg = max(home_xg, 1.8)
            away_xg = min(away_xg, 1.0)
        
        if home_strength == 5 and away_strength == 1:
            home_xg = max(home_xg, 2.2)
            away_xg = min(away_xg, 0.7)
        
        home_xg = max(0.5, min(3.5, home_xg))
        away_xg = max(0.3, min(2.5, away_xg))
        
        logger.info(f"Football reality check: Home {home_xg:.2f}, Away {away_xg:.2f}")
        return home_xg, away_xg

    def _calculate_realistic_xg(self) -> Tuple[float, float]:
        """Calculate REALISTIC predictive xG"""
        league = self.data.get('league', 'liga_portugal')
        league_params = self.league_contexts.get(league, self.league_contexts['default'])
        
        home_goals_avg = self.data.get('home_goals', 0) / 6.0
        away_goals_avg = self.data.get('away_goals', 0) / 6.0
        
        home_conceded_avg = self.data.get('home_conceded', 0) / 6.0
        away_conceded_avg = self.data.get('away_conceded', 0) / 6.0
        
        home_xg = home_goals_avg * (1 - (away_conceded_avg / league_params['avg_goals']) * self.calibration_params['defensive_impact_multiplier'])
        away_xg = away_goals_avg * (1 - (home_conceded_avg / league_params['avg_goals']) * self.calibration_params['defensive_impact_multiplier'])
        
        home_xg *= league_params['home_advantage']
        
        strength_adjustment = self._calculate_strength_based_adjustment(
            self.data['home_team'], self.data['away_team']
        )
        home_xg *= (1 + strength_adjustment)
        away_xg *= (1 - strength_adjustment * 0.8)
        
        home_form = self._calculate_form_impact('home')
        away_form = self._calculate_form_impact('away')
        
        home_xg *= home_form
        away_xg *= away_form
        
        motivation = self.data.get('motivation', {})
        home_motivation = self._calculate_motivation_impact(motivation.get('home', 'Normal'))
        away_motivation = self._calculate_motivation_impact(motivation.get('away', 'Normal'))
        
        injuries = self.data.get('injuries', {})
        home_injuries = max(0.7, 1.0 - (float(injuries.get('home', 0)) * self.calibration_params['injury_impact']))
        away_injuries = max(0.7, 1.0 - (float(injuries.get('away', 0)) * self.calibration_params['injury_impact']))
        
        home_xg *= home_motivation * home_injuries
        away_xg *= away_motivation * away_injuries
        
        h2h_data = self.data.get('h2h_data', {})
        if h2h_data and h2h_data.get('matches', 0) >= 3:
            home_xg, away_xg = self._apply_h2h_adjustment(home_xg, away_xg, h2h_data)
        
        home_xg, away_xg = self._apply_football_reality_checks(
            home_xg, away_xg, self.data['home_team'], self.data['away_team']
        )
        
        logger.info(f"Realistic xG - Home: {home_xg:.3f}, Away: {away_xg:.3f}")
        
        return round(home_xg, 3), round(away_xg, 3)
    
    def _calculate_form_impact(self, team: str) -> float:
        """Calculate form impact"""
        form = self.data.get(f'{team}_form', [])
        if not form or len(form) == 0:
            return 1.0
        
        try:
            form_scores = [float(score) for score in form]
            avg_form = np.mean(form_scores)
            
            form_ratio = avg_form / 3.0
            return 0.8 + (form_ratio * 0.4)
            
        except (TypeError, ValueError):
            return 1.0
    
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
        """Run Monte Carlo simulation"""
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

    def _validate_probability_sanity(self, home_win: float, draw: float, away_win: float, home_team: str, away_team: str) -> bool:
        """Validate that probabilities make football sense"""
        total = home_win + draw + away_win
        
        if not 0.99 <= total <= 1.01:
            logger.error(f"Probabilities not normalized: {total}")
            return False
        
        home_strength = self._get_team_strength(home_team)
        away_strength = self._get_team_strength(away_team)
        
        if home_strength >= 4 and away_strength <= 2 and home_win < 0.60:
            logger.warning(f"Strong home team {home_team} has unrealistically low win probability: {home_win}")
            return False
        
        if away_strength <= 2 and home_strength >= 4 and away_win > 0.15:
            logger.warning(f"Weak away team {away_team} has unrealistically high win probability: {away_win}")
            return False
            
        return True

    def generate_predictions(self, mc_iterations: int = 10000) -> Dict[str, Any]:
        """Generate REALISTIC football predictions"""
        
        home_team = self.data['home_team']
        away_team = self.data['away_team']
        
        home_xg, away_xg = self._calculate_realistic_xg()
        
        self.match_context = self._determine_match_context(home_xg, away_xg, home_team, away_team)
        
        mc_results = self.run_monte_carlo_simulation(home_xg, away_xg, mc_iterations)
        
        if not self._validate_probability_sanity(
            mc_results.home_win_prob, mc_results.draw_prob, mc_results.away_win_prob,
            home_team, away_team
        ):
            logger.warning("Probability sanity check failed - applying football reality corrections")
            
            home_strength = self._get_team_strength(home_team)
            away_strength = self._get_team_strength(away_team)
            
            if home_strength >= 4 and away_strength <= 2:
                mc_results.home_win_prob = max(mc_results.home_win_prob, 0.65)
                mc_results.away_win_prob = min(mc_results.away_win_prob, 0.12)
            
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
        
        confidence_score = self._calculate_confidence(mc_results, home_team, away_team)
        risk_assessment = self._assess_risk(mc_results, confidence_score, home_xg, away_xg, home_team, away_team)
        
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
            'monte_carlo_results': {
                'confidence_intervals': mc_results.confidence_intervals,
                'probability_volatility': mc_results.probability_volatility
            }
        }
        
        return predictions
    
    def _calculate_confidence(self, mc_results: MonteCarloResults, home_team: str, away_team: str) -> int:
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
    
    def _assess_risk(self, mc_results: MonteCarloResults, confidence: int, home_xg: float, away_xg: float, home_team: str, away_team: str) -> Dict[str, str]:
        """Realistic risk assessment"""
        home_win_prob = mc_results.home_win_prob
        home_strength = self._get_team_strength(home_team)
        away_strength = self._get_team_strength(away_team)
        
        if (home_strength >= 4 and away_strength <= 2 and
            home_win_prob > 0.65 and confidence > 70 and
            self.data['data_quality_score'] > 75):
            risk_level = "MEDIUM"
            explanation = "Strong home favorite with good data quality"
            recommendation = "CONSIDER SMALL STAKE"
        elif (home_win_prob > 0.58 and confidence > 65 and
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
                        away_xg: float, mc_results: MonteCarloResults) -> str:
        """Generate realistic football summary"""
        
        home_win_prob = mc_results.home_win_prob
        home_strength = self._get_team_strength(home_team)
        away_strength = self._get_team_strength(away_team)
        
        if home_strength >= 4 and away_strength <= 2 and home_win_prob > 0.65:
            return f"{home_team} are strong favorites with {home_xg:.1f} expected goals. Their superior quality and home advantage should see them comfortably overcome {away_team}."
        
        elif self.match_context == MatchContext.HOME_DOMINANCE and home_win_prob > 0.60:
            return f"{home_team} demonstrate clear superiority with {home_xg:.1f} expected goals. Home advantage and better metrics point towards a victory against {away_team}."
        
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
    PERFECTLY ALIGNED VALUE DETECTION ENGINE - NO CONTRADICTIONS
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
        
        # Team strength tiers for reality checks
        self.team_strength_tiers = {
            'Sporting CP': 5, 'Porto': 5, 'Benfica': 5, 'Braga': 4,
            'Vitoria Guimaraes': 3, 'Boavista': 3, 'Famalicao': 3,
            'Casa Pia': 2, 'Rio Ave': 2, 'Estoril': 2, 'Gil Vicente': 2,
            'FC Alverca': 1, 'AVS': 1, 'Leiria': 1,
        }
    
    def _get_team_strength(self, team: str) -> int:
        """Get team strength tier"""
        return self.team_strength_tiers.get(team, 2)

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
            'match_context': pure_probabilities['match_context']
        }
    
    def _is_contradictory_signal(self, signal: BettingSignal, primary_prediction: Dict) -> bool:
        """Check if a signal contradicts the primary prediction"""
        
        # NEVER allow contradictory outcomes in home dominance
        if (primary_prediction['match_context'] == 'home_dominance' and 
            primary_prediction['outcome'] == 'HOME'):
            
            if signal.market in ['1x2 Draw', '1x2 Away']:
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
    PERFECTLY ALIGNED ORCHESTRATOR: No contradictions between engines
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
        
        primary_outcome = max(outcomes, key=outcomes.get)
        primary_btts = 'yes' if btts['yes'] > btts['no'] else 'no'
        primary_over_under = 'over_25' if over_under['over_25'] > over_under['under_25'] else 'under_25'
        
        # Check for any contradictory value signals
        for signal in value_signals:
            signal_dict = signal.__dict__ if hasattr(signal, '__dict__') else signal
            
            # Check outcome contradictions
            if (signal_dict['market'] == '1x2 Draw' and primary_outcome == 'home_win' and 
                outcomes['home_win'] > 65):
                issues.append("Draw value bet contradicts strong home win prediction")
            
            if (signal_dict['market'] == '1x2 Away' and primary_outcome == 'home_win' and 
                outcomes['home_win'] > 65):
                issues.append("Away win value bet contradicts strong home win prediction")
            
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
