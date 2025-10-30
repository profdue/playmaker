import numpy as np
from scipy.stats import poisson, skellam
from typing import Dict, Any, Tuple, List, Optional
import math
import pandas as pd
from dataclasses import dataclass
import logging
from datetime import datetime
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    """Enhanced Football Prediction Engine with Bivariate Poisson & Monte Carlo Simulation"""
    
    def __init__(self, match_data: Dict[str, Any], calibration_data: Optional[Dict] = None):
        self.data = match_data
        self.calibration_data = calibration_data or {}
        self.league_contexts = self._initialize_league_contexts()
        self.monte_carlo_iterations = 10000
        self._setup_calibration_parameters()
        
    def _initialize_league_contexts(self) -> Dict[str, Dict]:
        """Initialize league-specific parameters with calibrated values"""
        return {
            'premier_league': {
                'avg_goals': 2.8, 'avg_corners': 10.5, 'home_advantage': 1.12,
                'goal_timing_first_half': 0.47, 'goal_timing_second_half': 0.53,
                'attack_weight': 1.05, 'defense_weight': 0.95,
                'bivariate_correlation': 0.15  # Higher correlation in open leagues
            },
            'la_liga': {
                'avg_goals': 2.6, 'avg_corners': 9.8, 'home_advantage': 1.15,
                'goal_timing_first_half': 0.45, 'goal_timing_second_half': 0.55,
                'attack_weight': 1.02, 'defense_weight': 0.98,
                'bivariate_correlation': 0.12
            },
            'serie_a': {
                'avg_goals': 2.7, 'avg_corners': 10.2, 'home_advantage': 1.08,
                'goal_timing_first_half': 0.46, 'goal_timing_second_half': 0.54,
                'attack_weight': 1.03, 'defense_weight': 0.97,
                'bivariate_correlation': 0.10  # Lower correlation in tactical leagues
            },
            'bundesliga': {
                'avg_goals': 3.1, 'avg_corners': 9.5, 'home_advantage': 1.12,
                'goal_timing_first_half': 0.48, 'goal_timing_second_half': 0.52,
                'attack_weight': 1.08, 'defense_weight': 0.92,
                'bivariate_correlation': 0.18  # Higher correlation in high-scoring leagues
            },
            'ligue_1': {
                'avg_goals': 2.5, 'avg_corners': 9.2, 'home_advantage': 1.1,
                'goal_timing_first_half': 0.44, 'goal_timing_second_half': 0.56,
                'attack_weight': 1.01, 'defense_weight': 0.99,
                'bivariate_correlation': 0.11
            },
            'default': {
                'avg_goals': 2.7, 'avg_corners': 10.0, 'home_advantage': 1.1,
                'goal_timing_first_half': 0.46, 'goal_timing_second_half': 0.54,
                'attack_weight': 1.04, 'defense_weight': 0.96,
                'bivariate_correlation': 0.12
            }
        }
    
    def _setup_calibration_parameters(self):
        """Setup calibration parameters from historical data"""
        # Default calibration values
        self.calibration_params = {
            'home_attack_weight': 1.05,
            'away_attack_weight': 0.95,
            'defense_weight': 0.92,
            'form_decay_rate': 0.9,
            'h2h_weight': 0.3,
            'injury_impact': 0.08,
            'motivation_impact': 0.12,
            'regression_strength': 0.25,
            'bivariate_lambda3_alpha': 0.12  # Heuristic for shared component
        }
        
        # Update with provided calibration data
        if self.calibration_data:
            self.calibration_params.update(self.calibration_data)
    
    def bivariate_poisson_pmf_matrix(self, lambda1: float, lambda2: float, lambda3: float, max_goals: int = 8):
        """
        Compute joint PMF matrix for Bivariate Poisson:
          X = A + C
          Y = B + C
        where A ~ Pois(lambda1), B ~ Pois(lambda2), C ~ Pois(lambda3) independent.
        Joint PMF:
          P(X=i, Y=j) = sum_{k=0}^{min(i,j)} e^{-(λ1+λ2+λ3)} * λ1^(i-k)/((i-k)!) * λ2^(j-k)/((j-k)!) * λ3^k/(k!)
        Returns matrix shape (max_goals+1, max_goals+1)
        """
        exp_term = math.exp(-(lambda1 + lambda2 + lambda3))
        P = np.zeros((max_goals+1, max_goals+1), dtype=float)

        # Precompute factorials for speed
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
        
        # Normalize (numerical rounding may cause tiny error)
        P /= P.sum()
        return P
    
    def get_markets_from_joint_pmf(self, P: np.ndarray):
        """
        Given joint PMF matrix P (i rows, j cols), compute:
          - exact score probabilities dict,
          - match outcome probs,
          - over/under (1.5,2.5,3.5),
          - btts
        """
        max_goals = P.shape[0] - 1
        exact_scores = {}
        
        # Match outcomes
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
                
                # Store exact scores above threshold
                if prob > 0.001:  # 0.1% threshold
                    exact_scores[f"{i}-{j}"] = round(prob * 100, 1)

        # Over/under probabilities
        total_prob = {}
        total_goals_matrix = np.add.outer(np.arange(max_goals+1), np.arange(max_goals+1))
        
        for line in [1.5, 2.5, 3.5]:
            over_mask = total_goals_matrix > line
            total_prob[f"over_{str(line).replace('.', '')}"] = round(P[over_mask].sum() * 100, 1)

        # Both teams to score
        btts_mask = (np.arange(max_goals+1) > 0)[:, None] & (np.arange(max_goals+1) > 0)
        btts_prob = round(P[btts_mask].sum() * 100, 1)

        return {
            'match_outcomes': {
                'home_win': round(home_win_prob * 100, 1),
                'draw': round(draw_prob * 100, 1),
                'away_win': round(away_win_prob * 100, 1)
            },
            'exact_scores': exact_scores,
            'over_under': total_prob,
            'both_teams_score': btts_prob
        }
    
    def generate_advanced_predictions(self) -> Dict[str, Any]:
        """Generate comprehensive predictions with bivariate Poisson"""
        
        # Extract and validate data
        home_team = self.data.get('home_team', 'Home Team')
        away_team = self.data.get('away_team', 'Away Team')
        league = self.data.get('league', 'default')
        
        # Get market odds if available
        market_odds = self.data.get('market_odds', {})
        
        # Calculate Bayesian-enhanced expected goals
        home_xg, away_xg = self._calculate_bayesian_xg()
        
        # Calculate bivariate Poisson probabilities
        bivariate_results = self._calculate_bivariate_probabilities(home_xg, away_xg, league)
        
        # Run Monte Carlo simulation for robust probabilities (using bivariate structure)
        mc_results = self._run_bivariate_monte_carlo_simulation(home_xg, away_xg, league)
        
        # Calculate Skellam-based handicap probabilities
        handicap_probs = self._calculate_handicap_probabilities(home_xg, away_xg)
        
        # Use bivariate results for main probabilities
        probabilities = bivariate_results
        
        # Generate betting signals with value detection
        betting_signals = self._generate_betting_signals(probabilities, market_odds, mc_results)
        
        # Calculate advanced metrics
        corner_predictions = self._calculate_corner_predictions(home_xg, away_xg, league)
        timing_predictions = self._calculate_enhanced_timing_predictions(home_xg, away_xg)
        
        # Risk and confidence assessment
        confidence_score = self._calculate_advanced_confidence(mc_results)
        risk_assessment = self._assess_prediction_risk(probabilities, confidence_score, mc_results)
        
        return {
            'match': f"{home_team} vs {away_team}",
            'expected_goals': {'home': round(home_xg, 2), 'away': round(away_xg, 2)},
            'probabilities': probabilities,
            'handicap_probabilities': handicap_probs,
            'monte_carlo_results': {
                'confidence_intervals': mc_results.confidence_intervals,
                'probability_volatility': mc_results.probability_volatility
            },
            'betting_signals': betting_signals,
            'corner_predictions': corner_predictions,
            'timing_predictions': timing_predictions,
            'summary': self._generate_quantitative_summary(home_team, away_team, probabilities, home_xg, away_xg),
            'confidence_score': confidence_score,
            'risk_assessment': risk_assessment,
            'model_metrics': self._calculate_model_metrics(probabilities, mc_results),
            'bivariate_parameters': {
                'lambda1': round(home_xg - (0.12 * min(home_xg, away_xg)), 3),
                'lambda2': round(away_xg - (0.12 * min(home_xg, away_xg)), 3),
                'lambda3': round(0.12 * min(home_xg, away_xg), 3)
            }
        }
    
    def _calculate_bivariate_probabilities(self, home_xg: float, away_xg: float, league: str) -> Dict[str, Any]:
        """Calculate probabilities using bivariate Poisson distribution"""
        
        # Get league-specific correlation parameter
        league_params = self.league_contexts.get(league, self.league_contexts['default'])
        lambda3_alpha = self.calibration_params.get('bivariate_lambda3_alpha', 0.12)
        
        # Calculate lambda3 using heuristic: λ3 = α * min(λ1+λ3, λ2+λ3)
        lambda3 = lambda3_alpha * min(home_xg, away_xg)
        
        # Ensure non-negative lambdas
        lambda1 = max(0.1, home_xg - lambda3)
        lambda2 = max(0.1, away_xg - lambda3)
        
        # Compute joint PMF matrix
        P = self.bivariate_poisson_pmf_matrix(lambda1, lambda2, lambda3, max_goals=8)
        
        # Extract market probabilities
        markets = self.get_markets_from_joint_pmf(P)
        
        # Add goal timing probabilities (keep existing method for now)
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
        np.random.seed(42)  # For reproducible results
        
        # Get correlation parameter
        lambda3_alpha = self.calibration_params.get('bivariate_lambda3_alpha', 0.12)
        lambda3 = lambda3_alpha * min(home_xg, away_xg)
        lambda1 = max(0.1, home_xg - lambda3)
        lambda2 = max(0.1, away_xg - lambda3)
        
        # Sample from bivariate Poisson structure
        C = np.random.poisson(lambda3, self.monte_carlo_iterations)  # Shared component
        A = np.random.poisson(lambda1, self.monte_carlo_iterations)  # Home-specific
        B = np.random.poisson(lambda2, self.monte_carlo_iterations)  # Away-specific
        
        home_goals_sim = A + C
        away_goals_sim = B + C
        
        # Calculate outcome probabilities
        home_wins = np.sum(home_goals_sim > away_goals_sim) / self.monte_carlo_iterations
        draws = np.sum(home_goals_sim == away_goals_sim) / self.monte_carlo_iterations
        away_wins = np.sum(home_goals_sim < away_goals_sim) / self.monte_carlo_iterations
        
        # Calculate market probabilities
        total_goals = home_goals_sim + away_goals_sim
        over_25 = np.sum(total_goals > 2.5) / self.monte_carlo_iterations
        btts = np.sum((home_goals_sim > 0) & (away_goals_sim > 0)) / self.monte_carlo_iterations
        
        # Exact score probabilities (from simulation)
        exact_scores = {}
        for i in range(5):  # Home goals
            for j in range(5):  # Away goals
                count = np.sum((home_goals_sim == i) & (away_goals_sim == j))
                prob = count / self.monte_carlo_iterations
                if prob > 0.01:  # Only include probabilities > 1%
                    exact_scores[f"{i}-{j}"] = round(prob * 100, 1)
        
        # Calculate confidence intervals (95%)
        def calculate_ci(probs, alpha=0.95):
            se = np.sqrt(probs * (1 - probs) / self.monte_carlo_iterations)
            z_score = 1.96  # for 95% CI
            return (probs - z_score * se, probs + z_score * se)
        
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
    
    def _calculate_bayesian_xg(self) -> Tuple[float, float]:
        """Calculate Bayesian-enhanced expected goals"""
        league = self.data.get('league', 'default')
        league_params = self.league_contexts.get(league, self.league_contexts['default'])
        
        # Extract data
        home_goals = self.data.get('home_goals', 0)
        away_goals = self.data.get('away_goals', 0)
        home_conceded = self.data.get('home_conceded', 0)
        away_conceded = self.data.get('away_conceded', 0)
        
        home_goals_home = self.data.get('home_goals_home', home_goals)
        away_goals_away = self.data.get('away_goals_away', away_goals)
        
        home_form = self.data.get('home_form', [])
        away_form = self.data.get('away_form', [])
        
        injuries = self.data.get('injuries', {'home': 0, 'away': 0})
        motivation = self.data.get('motivation', {'home': 1.0, 'away': 1.0})
        h2h_data = self.data.get('h2h_data', {})
        home_avg_stats = self.data.get('home_avg_stats', {})
        away_avg_stats = self.data.get('away_avg_stats', {})
        
        # Bayesian prior (league average)
        prior_goals = league_params['avg_goals'] / 2  # per team
        prior_weight = 6  # Equivalent to 6 matches of data
        
        # Calculate observed attack/defense strength with Bayesian updating
        home_attack_obs = max(0.3, home_goals / 6.0)
        away_attack_obs = max(0.2, away_goals / 6.0)
        home_defense_obs = max(0.3, home_conceded / 6.0)
        away_defense_obs = max(0.4, away_conceded / 6.0)
        
        # Home/away specific observations
        home_attack_home_obs = max(0.3, home_goals_home / 3.0)
        away_attack_away_obs = max(0.2, away_goals_away / 3.0)
        
        # Bayesian posterior estimates
        home_attack = (prior_goals * prior_weight + home_attack_obs * 6) / (prior_weight + 6)
        away_attack = (prior_goals * prior_weight + away_attack_obs * 6) / (prior_weight + 6)
        home_defense = (prior_goals * prior_weight + home_defense_obs * 6) / (prior_weight + 6)
        away_defense = (prior_goals * prior_weight + away_defense_obs * 6) / (prior_weight + 6)
        
        home_attack_home = (prior_goals * prior_weight + home_attack_home_obs * 3) / (prior_weight + 3)
        away_attack_away = (prior_goals * prior_weight + away_attack_away_obs * 3) / (prior_weight + 3)
        
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
        
        # Apply H2H adjustment if available
        if h2h_data:
            base_home_xg, base_away_xg = self._apply_bayesian_h2h_adjustment(
                base_home_xg, base_away_xg, h2h_data
            )
        
        # Final regression to mean
        home_xg = (base_home_xg + league_params['avg_goals'] * self.calibration_params['regression_strength']) / (1 + self.calibration_params['regression_strength'])
        away_xg = (base_away_xg + league_params['avg_goals'] * self.calibration_params['regression_strength']) / (1 + self.calibration_params['regression_strength'])
        
        # Ensure reasonable bounds
        home_xg = max(0.1, min(4.0, home_xg))
        away_xg = max(0.1, min(3.5, away_xg))
        
        return round(home_xg, 3), round(away_xg, 3)
    
    def _calculate_decaying_form_factor(self, form: List[float]) -> float:
        """Calculate form factor with exponential decay for recent matches"""
        if not form or len(form) == 0:
            return 1.0
        
        # Exponential decay weights (most recent match has highest weight)
        weights = [self.calibration_params['form_decay_rate'] ** i for i in range(len(form))]
        weights = [w / sum(weights) for w in weights]  # Normalize
        
        total_points = sum(score * weight for score, weight in zip(form, reversed(weights)))
        max_possible = sum(3 * weight for weight in weights)
        
        form_ratio = total_points / max_possible if max_possible > 0 else 0.5
        
        # Convert to multiplier (0.8 to 1.2 range)
        return 0.8 + (form_ratio * 0.4)
    
    def _apply_bayesian_h2h_adjustment(self, home_xg: float, away_xg: float, h2h_data: Dict) -> Tuple[float, float]:
        """Apply Bayesian H2H adjustment"""
        matches = h2h_data.get('matches', 0)
        home_wins = h2h_data.get('home_wins', 0)
        away_wins = h2h_data.get('away_wins', 0)
        draws = h2h_data.get('draws', 0)
        home_goals = h2h_data.get('home_goals', 0)
        away_goals = h2h_data.get('away_goals', 0)
        
        if matches < 2:  # Not enough H2H data
            return home_xg, away_xg
        
        # Calculate H2H averages with Bayesian shrinkage
        h2h_weight = min(0.4, matches * 0.1)  # Cap at 40% influence
        
        h2h_home_avg = home_goals / matches
        h2h_away_avg = away_goals / matches
        
        # Blend with current estimates
        adjusted_home_xg = (home_xg * (1 - h2h_weight)) + (h2h_home_avg * h2h_weight)
        adjusted_away_xg = (away_xg * (1 - h2h_weight)) + (h2h_away_avg * h2h_weight)
        
        return adjusted_home_xg, adjusted_away_xg
    
    def _calculate_handicap_probabilities(self, home_xg: float, away_xg: float) -> Dict[str, float]:
        """Calculate Asian Handicap probabilities using Skellam distribution"""
        # Skellam distribution for goal difference
        goal_diffs = range(-5, 6)  # From -5 to +5 goal difference
        
        handicap_probs = {}
        
        # Calculate probabilities for common handicaps
        handicaps = [-2.5, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5]
        
        for handicap in handicaps:
            if handicap == 0:
                # Draw no bet (home wins)
                prob = 1 - skellam.cdf(0, home_xg, away_xg)
            elif handicap > 0:
                # Home team gives handicap
                prob = 1 - skellam.cdf(handicap - 1, home_xg, away_xg)
            else:
                # Away team gives handicap
                prob = skellam.cdf(abs(handicap), home_xg, away_xg)
            
            handicap_probs[f"handicap_{handicap}"] = round(prob * 100, 1)
        
        return handicap_probs
    
    def _generate_betting_signals(self, probabilities: Dict, market_odds: Dict, 
                                mc_results: MonteCarloResults) -> List[Dict]:
        """Generate actionable betting signals with value detection"""
        signals = []
        
        # Convert market odds to probabilities (if available)
        market_probs = self._convert_odds_to_probabilities(market_odds)
        
        # Check main markets for value
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
            else:
                model_prob = probabilities.get(model_key, 0) / 100
            
            book_prob = market_probs.get(market_name, 0)
            
            if book_prob > 0:  # Only if we have market odds
                edge = (model_prob - book_prob) * 100
                
                if abs(edge) > 2:  # Minimum edge threshold
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
        
        # Sort by edge descending
        signals.sort(key=lambda x: x['edge'], reverse=True)
        return signals
    
    def _convert_odds_to_probabilities(self, market_odds: Dict) -> Dict[str, float]:
        """Convert decimal odds to probabilities"""
        probs = {}
        
        for market, odds in market_odds.items():
            if isinstance(odds, (int, float)) and odds > 1:
                # Adjust for overround (bookmaker margin)
                probs[market] = 1 / odds
            elif isinstance(odds, dict):
                # Handle multiple outcomes (e.g., 1x2 market)
                total_implied = sum(1 / o for o in odds.values() if o > 1)
                if total_implied > 0:
                    for outcome, odd in odds.items():
                        if odd > 1:
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
        """Calculate Kelly criterion stake (fraction of bankroll)"""
        if book_prob <= 0 or model_prob <= book_prob:
            return 0.0
        
        decimal_odds = 1 / book_prob
        kelly_stake = (model_prob * decimal_odds - 1) / (decimal_odds - 1)
        
        # Apply fractional Kelly and cap at 5%
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
        attacking_bonus = (home_xg + away_xg - league_params['avg_goals']) * 1.5
        
        total_corners = base_corners + attacking_bonus
        total_corners = max(4, min(16, total_corners))
        
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
        
        # Use Monte Carlo for more robust timing predictions
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
        base_confidence = 0
        
        # Data completeness factors
        if self.data.get('home_goals', 0) > 0: base_confidence += 10
        if self.data.get('away_goals', 0) > 0: base_confidence += 10
        if self.data.get('home_form'): base_confidence += 10
        if self.data.get('away_form'): base_confidence += 10
        
        # Historical data
        h2h_data = self.data.get('h2h_data', {})
        if h2h_data.get('matches', 0) >= 3: base_confidence += 20
        
        # Monte Carlo volatility adjustment
        avg_volatility = np.mean(list(mc_results.probability_volatility.values()))
        volatility_penalty = min(30, int(avg_volatility * 1000))
        
        confidence = base_confidence - volatility_penalty
        
        return max(10, min(95, confidence))
    
    def _assess_prediction_risk(self, probabilities: Dict, confidence: int, 
                              mc_results: MonteCarloResults) -> Dict[str, str]:
        """Enhanced risk assessment with Monte Carlo uncertainty"""
        outcomes = probabilities['match_outcomes']
        highest_prob = max(outcomes.values())
        
        # Calculate probability entropy (uncertainty measure)
        probs = np.array([outcomes['home_win'], outcomes['draw'], outcomes['away_win']]) / 100
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(3)  # Maximum entropy for 3 outcomes
        
        uncertainty_ratio = entropy / max_entropy
        
        if highest_prob > 70 and confidence > 80 and uncertainty_ratio < 0.7:
            risk_level = "LOW"
            explanation = "Strong favorite with low uncertainty"
        elif highest_prob > 55 and confidence > 65 and uncertainty_ratio < 0.85:
            risk_level = "MEDIUM"
            explanation = "Moderate favorite with acceptable uncertainty"
        else:
            risk_level = "HIGH"
            explanation = f"High uncertainty (entropy: {uncertainty_ratio:.2f})"
        
        return {
            'risk_level': risk_level,
            'explanation': explanation,
            'certainty': f"{highest_prob}%",
            'uncertainty_index': round(uncertainty_ratio, 2)
        }
    
    def _calculate_model_metrics(self, probabilities: Dict, mc_results: MonteCarloResults) -> Dict[str, float]:
        """Calculate model performance metrics"""
        # Shannon entropy of match outcomes
        probs = np.array([probabilities['match_outcomes'][k] for k in ['home_win', 'draw', 'away_win']]) / 100
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        # Average probability volatility
        avg_volatility = np.mean(list(mc_results.probability_volatility.values()))
        
        # Confidence interval width (average)
        avg_ci_width = np.mean([
            ci[1] - ci[0] for ci in mc_results.confidence_intervals.values()
        ])
        
        return {
            'shannon_entropy': round(entropy, 3),
            'avg_probability_volatility': round(avg_volatility, 4),
            'avg_confidence_interval_width': round(avg_ci_width, 4),
            'monte_carlo_iterations': self.monte_carlo_iterations
        }
    
    def _generate_quantitative_summary(self, home_team: str, away_team: str, probabilities: Dict,
                                     home_xg: float, away_xg: float) -> str:
        """Generate quantitative match summary"""
        outcomes = probabilities['match_outcomes']
        
        if outcomes['home_win'] > 65 and home_xg > away_xg + 0.8:
            return f"{home_team} demonstrate clear superiority with {home_xg:.1f} expected goals. Strong home advantage and attacking efficiency suggest comfortable victory."
        elif outcomes['home_win'] > 55 and home_xg > away_xg + 0.4:
            return f"{home_team} hold measurable advantage with better expected goal metrics. {away_team} will need exceptional defensive discipline to contain home threat."
        elif outcomes['away_win'] > 50 and away_xg > home_xg:
            return f"{away_team} pose significant threat with competitive expected goals. This could challenge {home_team}'s home defensive record."
        elif outcomes['draw'] > 35 and abs(home_xg - away_xg) < 0.3:
            return f"Evenly balanced encounter with minimal separation in expected goals. Set-piece efficiency and individual quality likely decisive."
        else:
            return f"Competitive match with both teams creating opportunities. Small margins expected to determine outcome in what promises tactical engagement."

# Example usage
if __name__ == "__main__":
    # Example calibration data
    calibration_data = {
        'home_attack_weight': 1.05,
        'away_attack_weight': 0.95,
        'defense_weight': 0.92,
        'form_decay_rate': 0.85,
        'h2h_weight': 0.25,
        'injury_impact': 0.08,
        'motivation_impact': 0.12,
        'regression_strength': 0.2,
        'bivariate_lambda3_alpha': 0.12
    }
    
    # Example match data
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
        'home_form': [3, 3, 3, 1, 0, 3],
        'away_form': [0, 3, 1, 0, 3, 1],
        'h2h_data': {
            'matches': 4,
            'home_wins': 3,
            'away_wins': 0,
            'draws': 1,
            'home_goals': 8,
            'away_goals': 2
        },
        'injuries': {'home': 0, 'away': 1},
        'motivation': {'home': 1.15, 'away': 1.0},
        'market_odds': {
            '1x2 Home': 1.85,
            '1x2 Draw': 3.40,
            '1x2 Away': 4.50,
            'Over 2.5 Goals': 2.10,
            'BTTS Yes': 1.95
        }
    }
    
    # Initialize engine and generate predictions
    engine = AdvancedPredictionEngine(match_data, calibration_data)
    predictions = engine.generate_advanced_predictions()
    
    print("Enhanced Bivariate Poisson Prediction Results:")
    print(json.dumps(predictions, indent=2, default=str))
