import numpy as np
from scipy.stats import poisson, skellam
from typing import Dict, Any, Tuple, List, Optional
import math
import pandas as pd
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import json
from enum import Enum
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

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
    market: str
    model_prob: float
    book_prob: float
    edge: float
    confidence: str
    recommended_stake: float
    value_rating: str

@dataclass
class MonteCarloResults:
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    over_25_prob: float
    btts_prob: float
    exact_scores: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]

class TruePredictiveEngine:
    """
    TRULY PREDICTIVE ENGINE - Focuses on future performance, not past results
    """
    
    def __init__(self, match_data: Dict[str, Any], historical_performance: Optional[Dict] = None):
        self.data = self._validate_and_enhance_data(match_data)
        self.historical_performance = historical_performance or {}
        self.team_quality_models = {}
        self.league_models = self._initialize_league_models()
        self._setup_predictive_parameters()
        
    def _validate_and_enhance_data(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced data validation with predictive features"""
        required_fields = ['home_team', 'away_team', 'league']
        
        for field in required_fields:
            if field not in match_data:
                raise ValueError(f"Missing required field: {field}")
        
        enhanced_data = match_data.copy()
        
        # Remove market data
        if 'market_odds' in enhanced_data:
            del enhanced_data['market_odds']
        
        # Enhanced numeric validation with predictive features
        predictive_fields = {
            'home_goals': (0, 20, 1.2),
            'away_goals': (0, 20, 1.2),
            'home_xg': (0, 5, 1.3),
            'away_xg': (0, 5, 1.1),
            'home_xg_against': (0, 5, 1.3),
            'away_xg_against': (0, 5, 1.1),
            'home_shots': (0, 30, 12),
            'away_shots': (0, 30, 10),
            'home_shots_on_target': (0, 15, 4),
            'away_shots_on_target': (0, 15, 3),
        }
        
        for field, (min_val, max_val, default) in predictive_fields.items():
            if field in enhanced_data:
                try:
                    value = float(enhanced_data[field])
                    enhanced_data[field] = max(min_val, min(value, max_val))
                except (TypeError, ValueError):
                    enhanced_data[field] = default
            else:
                enhanced_data[field] = default
        
        # Calculate PREDICTIVE metrics (not descriptive)
        enhanced_data = self._calculate_predictive_metrics(enhanced_data)
        
        return enhanced_data
    
    def _calculate_predictive_metrics(self, data: Dict) -> Dict:
        """Calculate metrics that actually predict future performance"""
        
        # Process form data to extract predictive signals
        home_form = data.get('home_form', [])
        away_form = data.get('away_form', [])
        
        # Convert form to performance indicators
        if home_form:
            data['home_form_trend'] = self._calculate_performance_trend(home_form)
            data['home_consistency'] = self._calculate_consistency(home_form)
            data['home_recent_momentum'] = self._calculate_momentum(home_form)
        
        if away_form:
            data['away_form_trend'] = self._calculate_performance_trend(away_form)
            data['away_consistency'] = self._calculate_consistency(away_form)
            data['away_recent_momentum'] = self._calculate_momentum(away_form)
        
        # Calculate efficiency metrics
        data['home_shot_efficiency'] = data.get('home_goals', 0) / max(1, data.get('home_shots', 1))
        data['away_shot_efficiency'] = data.get('away_goals', 0) / max(1, data.get('away_shots', 1))
        
        # Calculate defensive solidity
        data['home_defensive_efficiency'] = 1 - (data.get('home_xg_against', 1.3) / max(1, data.get('home_conceded', 1)))
        data['away_defensive_efficiency'] = 1 - (data.get('away_xg_against', 1.1) / max(1, data.get('away_conceded', 1)))
        
        # Calculate predictive data quality score
        data['predictive_data_score'] = self._calculate_predictive_data_quality(data)
        
        return data
    
    def _calculate_performance_trend(self, form: List[float]) -> float:
        """Calculate if team is improving or declining"""
        if len(form) < 3:
            return 0.0
        
        recent = form[:3]  # Last 3 games
        earlier = form[3:6] if len(form) >= 6 else form[3:]
        
        if not earlier:
            return 0.0
            
        recent_avg = np.mean(recent)
        earlier_avg = np.mean(earlier)
        
        return (recent_avg - earlier_avg) / max(earlier_avg, 0.1)
    
    def _calculate_consistency(self, form: List[float]) -> float:
        """Calculate performance consistency (lower = more consistent)"""
        if len(form) < 3:
            return 1.0
        return np.std(form) / max(np.mean(form), 0.1)
    
    def _calculate_momentum(self, form: List[float]) -> float:
        """Calculate recent momentum with exponential weighting"""
        if not form:
            return 0.5
            
        weights = [0.5, 0.3, 0.2]  # Recent games matter more
        weighted_form = form[:3]  # Last 3 games
        if len(weighted_form) < len(weights):
            weights = weights[:len(weighted_form)]
            weights = [w/sum(weights) for w in weights]
        
        return sum(score * weight for score, weight in zip(weighted_form, weights)) / 3.0
    
    def _calculate_predictive_data_quality(self, data: Dict) -> float:
        """Score data quality based on predictive value"""
        score = 0
        max_score = 0
        
        # Advanced metrics availability
        if data.get('home_xg', 0) > 0:
            score += 20
        if data.get('away_xg', 0) > 0:
            score += 20
        max_score += 40
        
        # Form analysis quality
        if len(data.get('home_form', [])) >= 5:
            score += 15
        if len(data.get('away_form', [])) >= 5:
            score += 15
        max_score += 30
        
        # Shot data availability
        if data.get('home_shots', 0) > 0:
            score += 10
        if data.get('away_shots', 0) > 0:
            score += 10
        max_score += 20
        
        # H2H data
        h2h_data = data.get('h2h_data', {})
        if h2h_data.get('matches', 0) >= 3:
            score += 10
        max_score += 10
        
        return (score / max_score) * 100
    
    def _initialize_league_models(self) -> Dict[str, Dict]:
        """League-specific predictive parameters"""
        return {
            'premier_league': {
                'avg_goals': 2.8, 'home_advantage': 1.08,
                'goal_variance': 1.15, 'draw_frequency': 0.26
            },
            'la_liga': {
                'avg_goals': 2.6, 'home_advantage': 1.10,
                'goal_variance': 1.10, 'draw_frequency': 0.28
            },
            'serie_a': {
                'avg_goals': 2.7, 'home_advantage': 1.12,
                'goal_variance': 1.05, 'draw_frequency': 0.27
            },
            'bundesliga': {
                'avg_goals': 3.1, 'home_advantage': 1.06,
                'goal_variance': 1.20, 'draw_frequency': 0.24
            },
            'ligue_1': {
                'avg_goals': 2.5, 'home_advantage': 1.11,
                'goal_variance': 1.08, 'draw_frequency': 0.29
            },
            'default': {
                'avg_goals': 2.7, 'home_advantage': 1.10,
                'goal_variance': 1.12, 'draw_frequency': 0.27
            }
        }
    
    def _setup_predictive_parameters(self):
        """Parameters focused on prediction, not description"""
        self.predictive_params = {
            'form_trend_weight': 0.15,
            'consistency_weight': 0.10,
            'momentum_weight': 0.12,
            'efficiency_weight': 0.08,
            'h2h_weight': 0.10,
            'injury_impact': 0.06,
            'motivation_impact': 0.08,
            'regression_to_mean': 0.85,  # Teams regress to mean performance
            'surprise_factor': 1.02,     # Account for unexpected outcomes
        }
    
    def _calculate_predictive_xg(self) -> Tuple[float, float]:
        """Calculate xG based on PREDICTIVE factors, not just past results"""
        league = self.data.get('league', 'serie_a')
        league_params = self.league_models.get(league, self.league_models['default'])
        
        # Base xG from underlying performance (predictive)
        home_base_xg = self.data.get('home_xg', 1.3)
        away_base_xg = self.data.get('away_xg', 1.1)
        
        # Adjust for defensive quality (predictive)
        home_defensive_quality = 1 - (self.data.get('home_xg_against', 1.3) / league_params['avg_goals'])
        away_defensive_quality = 1 - (self.data.get('away_xg_against', 1.1) / league_params['avg_goals'])
        
        # Apply defensive adjustments
        home_xg = home_base_xg * (1 - away_defensive_quality * 0.3)
        away_xg = away_base_xg * (1 - home_defensive_quality * 0.3)
        
        # Apply form trends (predictive)
        home_trend = self.data.get('home_form_trend', 0.0)
        away_trend = self.data.get('away_form_trend', 0.0)
        
        home_xg *= (1 + home_trend * self.predictive_params['form_trend_weight'])
        away_xg *= (1 + away_trend * self.predictive_params['form_trend_weight'])
        
        # Apply momentum (predictive)
        home_momentum = self.data.get('home_recent_momentum', 0.5)
        away_momentum = self.data.get('away_recent_momentum', 0.5)
        
        momentum_factor_home = 0.8 + (home_momentum * 0.4)
        momentum_factor_away = 0.8 + (away_momentum * 0.4)
        
        home_xg *= momentum_factor_home
        away_xg *= momentum_factor_away
        
        # Apply home advantage
        home_xg *= league_params['home_advantage']
        
        # Apply regression to mean (CRITICAL for prediction)
        home_xg = (home_xg * self.predictive_params['regression_to_mean'] + 
                  league_params['avg_goals'] * (1 - self.predictive_params['regression_to_mean']))
        away_xg = (away_xg * self.predictive_params['regression_to_mean'] + 
                  league_params['avg_goals'] * (1 - self.predictive_params['regression_to_mean']))
        
        # Apply surprise factor (football is unpredictable)
        home_xg *= self.predictive_params['surprise_factor']
        away_xg *= self.predictive_params['surprise_factor']
        
        # Realistic bounds
        home_xg = max(0.1, min(3.5, home_xg))
        away_xg = max(0.1, min(3.0, away_xg))
        
        logger.info(f"PREDICTIVE xG - Home: {home_xg:.3f}, Away: {away_xg:.3f}")
        logger.info(f"Trend factors - Home: {home_trend:.3f}, Away: {away_trend:.3f}")
        
        return round(home_xg, 3), round(away_xg, 3)
    
    def _determine_predictive_context(self, home_xg: float, away_xg: float) -> MatchContext:
        """Determine match context based on predictive factors"""
        total_xg = home_xg + away_xg
        xg_difference = home_xg - away_xg
        
        home_consistency = self.data.get('home_consistency', 1.0)
        away_consistency = self.data.get('away_consistency', 1.0)
        
        # High unpredictability
        if home_consistency > 0.8 or away_consistency > 0.8:
            return MatchContext.UNPREDICTABLE
        
        # Defensive battle
        if total_xg < 2.3 and abs(xg_difference) < 0.5:
            return MatchContext.DEFENSIVE_BATTLE
        
        # Home dominance
        if xg_difference > 0.8:
            return MatchContext.HOME_DOMINANCE
        
        # Away counter
        if xg_difference < -0.6:
            return MatchContext.AWAY_COUNTER
        
        # Offensive showdown
        if total_xg > 3.2:
            return MatchContext.OFFENSIVE_SHOWDOWN
        
        return MatchContext.TACTICAL_STALEMATE
    
    def run_predictive_simulation(self, home_xg: float, away_xg: float, iterations: int = 10000) -> MonteCarloResults:
        """Run simulation with predictive adjustments"""
        np.random.seed(42)
        
        # Adjust for match context
        context = self._determine_predictive_context(home_xg, away_xg)
        
        if context == MatchContext.DEFENSIVE_BATTLE:
            # Lower variance for defensive games
            home_goals_sim = np.random.poisson(home_xg * 0.9, iterations)
            away_goals_sim = np.random.poisson(away_xg * 0.9, iterations)
        elif context == MatchContext.OFFENSIVE_SHOWDOWN:
            # Higher variance for offensive games
            home_goals_sim = np.random.poisson(home_xg * 1.1, iterations)
            away_goals_sim = np.random.poisson(away_xg * 1.1, iterations)
        else:
            home_goals_sim = np.random.poisson(home_xg, iterations)
            away_goals_sim = np.random.poisson(away_xg, iterations)
        
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
        
        return MonteCarloResults(
            home_win_prob=home_wins,
            draw_prob=draws,
            away_win_prob=away_wins,
            over_25_prob=over_25,
            btts_prob=btts,
            exact_scores=exact_scores,
            confidence_intervals=confidence_intervals
        )
    
    def generate_predictions(self, mc_iterations: int = 10000) -> Dict[str, Any]:
        """Generate TRULY PREDICTIVE football forecasts"""
        
        home_team = self.data['home_team']
        away_team = self.data['away_team']
        
        # Calculate predictive expected goals
        home_xg, away_xg = self._calculate_predictive_xg()
        
        # Determine match context
        match_context = self._determine_predictive_context(home_xg, away_xg)
        
        # Run simulation
        mc_results = self.run_predictive_simulation(home_xg, away_xg, mc_iterations)
        
        # Calculate confidence based on predictive factors
        confidence_score = self._calculate_predictive_confidence(mc_results)
        
        # Risk assessment
        risk_assessment = self._assess_predictive_risk(mc_results, confidence_score)
        
        return {
            'match': f"{home_team} vs {away_team}",
            'expected_goals': {'home': round(home_xg, 2), 'away': round(away_xg, 2)},
            'match_context': match_context.value,
            'predictive_data_score': round(self.data['predictive_data_score'], 1),
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
                    'over_25': round(mc_results.over_25_prob * 100, 1),
                    'under_25': round((1 - mc_results.over_25_prob) * 100, 1)
                },
                'exact_scores': mc_results.exact_scores
            },
            'predictive_insights': self._generate_predictive_insights(home_xg, away_xg, match_context),
            'confidence_score': confidence_score,
            'risk_assessment': risk_assessment,
            'monte_carlo_results': {
                'confidence_intervals': mc_results.confidence_intervals
            }
        }
    
    def _calculate_predictive_confidence(self, mc_results: MonteCarloResults) -> int:
        """Calculate confidence based on predictive data quality"""
        base_confidence = self.data['predictive_data_score'] * 0.6
        
        # Bonus for good predictive features
        if self.data.get('home_xg', 0) > 0 and self.data.get('away_xg', 0) > 0:
            base_confidence += 20
        
        # Bonus for consistent teams
        home_consistency = self.data.get('home_consistency', 1.0)
        away_consistency = self.data.get('away_consistency', 1.0)
        
        if home_consistency < 0.7:
            base_confidence += 10
        if away_consistency < 0.7:
            base_confidence += 10
        
        # Penalty for high uncertainty in outcomes
        entropy = -sum([mc_results.home_win_prob * np.log(mc_results.home_win_prob + 1e-10),
                       mc_results.draw_prob * np.log(mc_results.draw_prob + 1e-10),
                       mc_results.away_win_prob * np.log(mc_results.away_win_prob + 1e-10)])
        
        uncertainty_penalty = min(30, entropy * 25)
        
        confidence = base_confidence - uncertainty_penalty
        return max(10, min(95, int(confidence)))
    
    def _assess_predictive_risk(self, mc_results: MonteCarloResults, confidence: int) -> Dict[str, str]:
        """Assess risk based on predictive uncertainty"""
        max_prob = max(mc_results.home_win_prob, mc_results.draw_prob, mc_results.away_win_prob)
        
        if max_prob > 0.65 and confidence > 75:
            risk_level = "LOW"
            explanation = "Clear favorite with strong predictive signals"
        elif max_prob > 0.55 and confidence > 60:
            risk_level = "MEDIUM"
            explanation = "Moderate favorite with reasonable predictability"
        else:
            risk_level = "HIGH"
            explanation = "High uncertainty - consider avoiding or small stakes"
        
        return {
            'risk_level': risk_level,
            'explanation': explanation,
            'recommendation': "BET" if risk_level in ["LOW", "MEDIUM"] else "AVOID"
        }
    
    def _generate_predictive_insights(self, home_xg: float, away_xg: float, context: MatchContext) -> Dict[str, str]:
        """Generate insights about what's likely to happen"""
        
        insights = {}
        
        if context == MatchContext.UNPREDICTABLE:
            insights['main'] = "High unpredictability detected - teams show inconsistent recent performances"
            insights['recommendation'] = "Avoid large stakes, consider alternative markets"
        elif context == MatchContext.DEFENSIVE_BATTLE:
            insights['main'] = "Defensive battle expected - low scoring likely"
            insights['recommendation'] = "Consider Under 2.5 goals or BTTS No"
        elif context == MatchContext.HOME_DOMINANCE:
            insights['main'] = "Home dominance predicted - strong home advantage expected"
            insights['recommendation'] = "Home win or handicap markets offer value"
        else:
            insights['main'] = "Competitive match expected - fine margins will decide"
            insights['recommendation'] = "Look for value in draw or narrow victory markets"
        
        # Trend-based insights
        home_trend = self.data.get('home_form_trend', 0)
        away_trend = self.data.get('away_form_trend', 0)
        
        if home_trend > 0.1:
            insights['home_trend'] = f"Home team showing improvement trend (+{home_trend:.2f})"
        if away_trend > 0.1:
            insights['away_trend'] = f"Away team showing improvement trend (+{away_trend:.2f})"
        
        return insights


class SmartValueEngine:
    """
    SMART VALUE DETECTION - Only bets when genuine edge exists
    """
    
    def __init__(self):
        # MUCH stricter thresholds
        self.value_thresholds = {
            'EXCEPTIONAL': 12.0,  # Only clear mispricings
            'HIGH': 8.0,          # Strong edge
            'GOOD': 5.0,          # Solid edge
            'MODERATE': 3.0,      # Minimum edge
        }
        self.min_confidence = 65   # Only confident predictions
        
    def detect_smart_value(self, predictions: Dict, market_odds: Dict) -> List[BettingSignal]:
        """Only detect value when it's genuine"""
        signals = []
        
        market_probs = self._calculate_implied_probabilities(market_odds)
        
        # Market mapping
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
            pure_prob = self._get_nested_value(predictions, prob_path) / 100.0
            market_prob = market_probs.get(market_name, 0)
            
            if market_prob > 0 and pure_prob > 0:
                edge = (pure_prob - market_prob) * 100
                
                # Apply market wisdom adjustment
                adjusted_edge = self._apply_market_wisdom(edge, market_prob, pure_prob)
                
                # Check confidence requirement
                if (adjusted_edge >= self.value_thresholds['MODERATE'] and 
                    predictions.get('confidence_score', 0) >= self.min_confidence):
                    
                    value_rating = self._get_value_rating(adjusted_edge)
                    stake = self._calculate_smart_stake(pure_prob, market_prob, predictions['confidence_score'])
                    
                    signals.append(BettingSignal(
                        market=market_name,
                        model_prob=round(pure_prob * 100, 1),
                        book_prob=round(market_prob * 100, 1),
                        edge=round(adjusted_edge, 1),
                        confidence=self._get_confidence_level(predictions['confidence_score']),
                        recommended_stake=stake,
                        value_rating=value_rating
                    ))
        
        return sorted(signals, key=lambda x: x.edge, reverse=True)
    
    def _apply_market_wisdom(self, edge: float, market_prob: float, pure_prob: float) -> float:
        """Respect market efficiency - reduce edges in efficient markets"""
        # Markets are most efficient around 30-70% probabilities
        if 0.3 < market_prob < 0.7:
            return edge * 0.8  # Reduce edge in efficient ranges
        
        # Be cautious with extreme probabilities
        if market_prob < 0.15 or market_prob > 0.85:
            return edge * 0.7  # Markets are often right about extremes
        
        return edge
    
    def _calculate_smart_stake(self, pure_prob: float, market_prob: float, confidence: int) -> float:
        """Very conservative staking"""
        if market_prob <= 0 or pure_prob <= market_prob:
            return 0.0
        
        decimal_odds = 1 / market_prob
        kelly_stake = (pure_prob * decimal_odds - 1) / (decimal_odds - 1)
        
        # Ultra-conservative: max 2% stake, reduced by confidence
        confidence_factor = confidence / 100
        max_stake = 0.02 * confidence_factor
        
        return max(0.0, min(max_stake, kelly_stake * 0.1))  # 1/10th Kelly
    
    def _get_confidence_level(self, confidence_score: int) -> str:
        if confidence_score >= 80:
            return "HIGH"
        elif confidence_score >= 65:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _calculate_implied_probabilities(self, market_odds: Dict[str, float]) -> Dict[str, float]:
        implied_probs = {}
        for market, odds in market_odds.items():
            if isinstance(odds, (int, float)) and odds > 1:
                implied_probs[market] = 1 / odds
        return implied_probs
    
    def _get_nested_value(self, data: Dict, path: str):
        keys = path.split('.')
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return 0
        return current
    
    def _get_value_rating(self, edge: float) -> str:
        for rating, threshold in self.value_thresholds.items():
            if edge >= threshold:
                return rating
        return "LOW"


class PredictiveFootballEngine:
    """
    MAIN ORCHESTRATOR - Combines predictive engine with smart value detection
    """
    
    def __init__(self, match_data: Dict[str, Any], historical_data: Optional[Dict] = None):
        self.market_odds = match_data.get('market_odds', {})
        
        # Create predictive football data
        football_data = match_data.copy()
        if 'market_odds' in football_data:
            del football_data['market_odds']
        
        self.predictive_engine = TruePredictiveEngine(football_data, historical_data)
        self.value_engine = SmartValueEngine()
        self.prediction_history = []
    
    def generate_predictive_analysis(self, mc_iterations: int = 10000) -> Dict[str, Any]:
        """Generate truly predictive analysis"""
        
        # Generate predictive forecasts
        football_predictions = self.predictive_engine.generate_predictions(mc_iterations)
        
        # Smart value detection
        value_signals = self.value_engine.detect_smart_value(football_predictions, self.market_odds)
        
        # Combine results
        result = football_predictions.copy()
        result['betting_signals'] = [signal.__dict__ for signal in value_signals]
        result['analysis_timestamp'] = datetime.now().isoformat()
        
        # Store for learning
        self._store_prediction(football_predictions)
        
        return result
    
    def _store_prediction(self, prediction: Dict):
        """Store prediction for future learning"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'match': prediction['match'],
            'expected_goals': prediction['expected_goals'],
            'probabilities': prediction['probabilities']['match_outcomes'],
            'context': prediction['match_context'],
            'confidence': prediction['confidence_score']
        }
        self.prediction_history.append(record)
        
        if len(self.prediction_history) > 50:
            self.prediction_history = self.prediction_history[-50:]
