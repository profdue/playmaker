import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam
from typing import Dict, Any, Tuple, List, Optional
import math
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TeamMetrics:
    """True predictive metrics - not descriptive outcomes"""
    non_penalty_xg: float = 0.0
    xg_against_per_shot: float = 0.0
    progressive_passes: float = 0.0
    press_intensity: float = 0.0
    defensive_actions: float = 0.0
    set_piece_xg: float = 0.0
    possession_quality: float = 0.0
    consistency_score: float = 0.0

@dataclass
class PredictiveFeatures:
    """Features that actually predict future performance"""
    home_attack_strength: float
    home_defense_quality: float 
    away_attack_strength: float
    away_defense_quality: float
    style_mismatch: float
    momentum_differential: float
    pressure_handling: float
    sustainability_score: float

@dataclass
class BettingSignal:
    market: str
    model_prob: float
    book_prob: float 
    edge: float
    confidence: str
    recommended_stake: float
    value_rating: str
    predictive_power: float

class TruePredictiveEngine:
    """
    TRUE PREDICTIVE ENGINE - Not descriptive, not correlational
    Focuses on sustainable performance metrics and market inefficiencies
    """
    
    def __init__(self, match_data: Dict[str, Any], historical_data: Optional[pd.DataFrame] = None):
        self.match_data = self._validate_predictive_data(match_data)
        self.historical_data = historical_data
        self.team_metrics = {}
        self.market_biases = self._initialize_market_biases()
        self.performance_tracker = PerformanceTracker()
        
        # Bayesian priors - updated with evidence
        self.team_priors = self._initialize_bayesian_priors()
        
    def _validate_predictive_data(self, match_data: Dict) -> Dict:
        """Ensure we have PREDICTIVE features, not descriptive outcomes"""
        required_predictive_features = [
            'home_team', 'away_team', 'home_xg', 'away_xg', 
            'home_xg_against', 'away_xg_against', 'home_possession_quality',
            'away_possession_quality', 'home_press_intensity', 'away_press_intensity'
        ]
        
        for feature in required_predictive_features:
            if feature not in match_data:
                raise ValueError(f"Missing predictive feature: {feature}")
        
        # Convert all to predictive metrics
        validated_data = match_data.copy()
        
        # Ensure we're using expected goals, not actual goals
        if 'home_goals' in validated_data:
            del validated_data['home_goals']  # Remove descriptive data
        if 'away_goals' in validated_data:
            del validated_data['away_goals']
            
        return validated_data
    
    def _initialize_market_biases(self) -> Dict:
        """Known market inefficiencies to exploit"""
        return {
            'recency_bias': 0.15,        # Overweight recent results
            'big_team_bias': 0.12,       # Overvalue famous teams
            'goal_bias': 0.18,           # Overvalue goals vs xG
            'home_bias': 0.08,           # Overvalue home advantage
            'public_sentiment_bias': 0.10 # Follow media narratives
        }
    
    def _initialize_bayesian_priors(self) -> Dict:
        """Initial beliefs about team quality - updated with evidence"""
        return {
            'attack_strength_prior': 1.0,
            'defense_quality_prior': 1.0,
            'consistency_prior': 0.5,
            'home_advantage_prior': 1.12
        }
    
    def calculate_sustainable_performance(self, team: str, side: str) -> TeamMetrics:
        """Calculate TRUE performance metrics that predict future results"""
        prefix = f"{side}_"
        
        # Base metrics from current match data
        base_xg = self.match_data.get(f"{prefix}xg", 1.3)
        xg_against = self.match_data.get(f"{prefix}xg_against", 1.3)
        possession = self.match_data.get(f"{prefix}possession_quality", 0.5)
        press_intensity = self.match_data.get(f"{prefix}press_intensity", 15.0)
        
        # Calculate predictive metrics
        metrics = TeamMetrics(
            non_penalty_xg=max(0.1, base_xg * 0.85),  # Remove penalty luck
            xg_against_per_shot=max(0.05, xg_against / 12.0),  # Defensive quality
            progressive_passes=possession * 25.0,  # Game control proxy
            press_intensity=press_intensity,
            defensive_actions=20.0 - (xg_against * 8.0),  # Higher = better defense
            set_piece_xg=base_xg * 0.15,  # Set piece threat
            possession_quality=possession,
            consistency_score=0.7  # Base consistency
        )
        
        # Apply Bayesian updating if historical data available
        if self.historical_data is not None:
            metrics = self._apply_bayesian_updating(team, metrics)
            
        return metrics
    
    def _apply_bayesian_updating(self, team: str, current_metrics: TeamMetrics) -> TeamMetrics:
        """Update beliefs with historical evidence"""
        if self.historical_data is None:
            return current_metrics
            
        # Get team's historical performance
        team_history = self.historical_data[
            (self.historical_data['team'] == team) & 
            (self.historical_data['date'] > datetime.now() - timedelta(days=90))
        ]
        
        if len(team_history) < 5:
            return current_metrics
            
        # Calculate historical averages
        hist_xg = team_history['xg'].mean()
        hist_xg_against = team_history['xg_against'].mean()
        hist_consistency = 1.0 - (team_history['xg'].std() / team_history['xg'].mean())
        
        # Bayesian update: weighted average of prior and evidence
        alpha = 0.3  # Learning rate
        updated_metrics = TeamMetrics(
            non_penalty_xg=(current_metrics.non_penalty_xg * (1-alpha)) + (hist_xg * alpha),
            xg_against_per_shot=(current_metrics.xg_against_per_shot * (1-alpha)) + (hist_xg_against/12 * alpha),
            progressive_passes=current_metrics.progressive_passes,  # Keep current
            press_intensity=current_metrics.press_intensity,  # Keep current
            defensive_actions=current_metrics.defensive_actions,  # Keep current
            set_piece_xg=current_metrics.set_piece_xg,  # Keep current
            possession_quality=current_metrics.possession_quality,  # Keep current
            consistency_score=(current_metrics.consistency_score * (1-alpha)) + (hist_consistency * alpha)
        )
        
        return updated_metrics
    
    def extract_predictive_features(self) -> PredictiveFeatures:
        """Extract features that actually predict match outcomes"""
        home_metrics = self.calculate_sustainable_performance(
            self.match_data['home_team'], 'home'
        )
        away_metrics = self.calculate_sustainable_performance(
            self.match_data['away_team'], 'away'  
        )
        
        # True predictive features
        features = PredictiveFeatures(
            home_attack_strength=home_metrics.non_penalty_xg,
            home_defense_quality=1.0 / (home_metrics.xg_against_per_shot + 0.001),
            away_attack_strength=away_metrics.non_penalty_xg,
            away_defense_quality=1.0 / (away_metrics.xg_against_per_shot + 0.001),
            style_mismatch=self._calculate_style_mismatch(home_metrics, away_metrics),
            momentum_differential=self._calculate_momentum_differential(home_metrics, away_metrics),
            pressure_handling=self._calculate_pressure_handling(home_metrics, away_metrics),
            sustainability_score=self._calculate_sustainability(home_metrics, away_metrics)
        )
        
        return features
    
    def _calculate_style_mismatch(self, home: TeamMetrics, away: TeamMetrics) -> float:
        """How well do team styles match up?"""
        press_differential = abs(home.press_intensity - away.press_intensity)
        possession_differential = abs(home.possession_quality - away.possession_quality)
        
        # High mismatch favors the more intense presser
        style_advantage = 0.0
        if home.press_intensity > away.press_intensity + 5:
            style_advantage = 0.15
        elif away.press_intensity > home.press_intensity + 5:
            style_advantage = -0.15
            
        return style_advantage
    
    def _calculate_momentum_differential(self, home: TeamMetrics, away: TeamMetrics) -> float:
        """Which team has sustainable momentum?"""
        home_momentum = home.consistency_score * home.non_penalty_xg
        away_momentum = away.consistency_score * away.non_penalty_xg
        
        return (home_momentum - away_momentum) / max(home_momentum + away_momentum, 0.1)
    
    def _calculate_pressure_handling(self, home: TeamMetrics, away: TeamMetrics) -> float:
        """How well do teams handle pressure situations?"""
        home_pressure = home.defensive_actions * home.consistency_score
        away_pressure = away.defensive_actions * away.consistency_score
        
        return home_pressure - away_pressure
    
    def _calculate_sustainability(self, home: TeamMetrics, away: TeamMetrics) -> float:
        """How sustainable are current performance levels?"""
        home_sustainability = (home.consistency_score * home.non_penalty_xg * 
                             (1.0 / (home.xg_against_per_shot + 0.001)))
        away_sustainability = (away.consistency_score * away.non_penalty_xg * 
                             (1.0 / (away.xg_against_per_shot + 0.001)))
        
        return home_sustainability - away_sustainability
    
    def predict_match_outcomes(self) -> Dict[str, float]:
        """Generate TRUE probabilistic predictions"""
        features = self.extract_predictive_features()
        
        # Base probabilities from team strengths
        home_attack_advantage = features.home_attack_strength / (
            features.home_attack_strength + features.away_attack_strength + 0.001)
        home_defense_advantage = features.home_defense_quality / (
            features.home_defense_quality + features.away_defense_quality + 0.001)
        
        # Combined advantage score
        home_advantage = (home_attack_advantage * 0.6 + home_defense_advantage * 0.4)
        
        # Apply style and momentum adjustments
        home_advantage += features.style_mismatch * 0.3
        home_advantage += features.momentum_differential * 0.4
        home_advantage += features.pressure_handling * 0.0005
        home_advantage += features.sustainability_score * 0.0002
        
        # Apply home advantage
        home_advantage *= 1.12
        
        # Convert to probabilities with proper normalization
        home_win_prob = self._sigmoid_normalize(home_advantage, 0.45, 0.75)
        away_win_prob = self._sigmoid_normalize(1 - home_advantage, 0.15, 0.55)
        draw_prob = 1.0 - home_win_prob - away_win_prob
        
        # Ensure valid probabilities
        home_win_prob = max(0.15, min(0.85, home_win_prob))
        away_win_prob = max(0.10, min(0.70, away_win_prob))
        draw_prob = max(0.10, min(0.35, draw_prob))
        
        # Normalize to sum to 1
        total = home_win_prob + draw_prob + away_win_prob
        home_win_prob /= total
        draw_prob /= total
        away_win_prob /= total
        
        return {
            'home_win': home_win_prob,
            'draw': draw_prob, 
            'away_win': away_win_prob,
            'predictive_confidence': features.sustainability_score,
            'market_mispricing': self._calculate_market_mispricing(home_win_prob)
        }
    
    def _sigmoid_normalize(self, x: float, min_val: float, max_val: float) -> float:
        """Normalize to reasonable probability range"""
        scaled = (x - 0.5) * 6.0  # Increase sensitivity
        sigmoid = 1.0 / (1.0 + math.exp(-scaled))
        return min_val + (sigmoid * (max_val - min_val))
    
    def _calculate_market_mispricing(self, model_prob: float) -> float:
        """Estimate where market is likely wrong"""
        # Base mispricing on known biases
        base_mispricing = 0.0
        
        # Add recency bias correction
        if self.match_data.get('home_recent_wins', 0) > 3:
            base_mispricing += self.market_biases['recency_bias']
            
        # Add big team bias correction
        big_teams = ['Barcelona', 'Real Madrid', 'Bayern', 'PSG', 'Man City', 'Liverpool']
        if self.match_data['home_team'] in big_teams:
            base_mispricing += self.market_biases['big_team_bias']
            
        return base_mispricing
    
    def run_predictive_simulation(self, iterations: int = 5000) -> Dict[str, Any]:
        """Run simulation based on TRUE predictive features"""
        predictions = self.predict_match_outcomes()
        
        # Use predictive probabilities for simulation
        home_win_prob = predictions['home_win']
        away_win_prob = predictions['away_win'] 
        draw_prob = predictions['draw']
        
        # Simulate match outcomes
        outcomes = np.random.choice(['H', 'D', 'A'], iterations, 
                                  p=[home_win_prob, draw_prob, away_win_prob])
        
        home_wins = np.sum(outcomes == 'H') / iterations
        draws = np.sum(outcomes == 'D') / iterations
        away_wins = np.sum(outcomes == 'A') / iterations
        
        # Calculate additional markets
        features = self.extract_predictive_features()
        total_goals_expectation = (features.home_attack_strength + 
                                 features.away_attack_strength) * 0.9
        
        over_25_prob = 1 - poisson.cdf(2.5, total_goals_expectation)
        btts_prob = 1 - (
            (1 - features.home_attack_strength/3) * 
            (1 - features.away_attack_strength/3)
        )
        
        return {
            'probabilities': {
                'home_win': round(home_wins * 100, 1),
                'draw': round(draws * 100, 1),
                'away_win': round(away_wins * 100, 1),
                'over_25': round(over_25_prob * 100, 1),
                'btts_yes': round(btts_prob * 100, 1)
            },
            'predictive_metrics': {
                'sustainability_score': round(features.sustainability_score, 3),
                'market_mispricing': round(predictions['market_mispricing'] * 100, 1),
                'predictive_confidence': round(predictions['predictive_confidence'], 3)
            },
            'expected_goals': {
                'home': round(features.home_attack_strength, 2),
                'away': round(features.away_attack_strength, 2)
            }
        }

class PredictiveValueEngine:
    """Value detection based on TRUE predictive power"""
    
    def __init__(self):
        self.value_thresholds = {
            'EXCEPTIONAL': 12.0,  # Much stricter - genuine edges only
            'HIGH': 8.0,
            'GOOD': 5.0, 
            'MODERATE': 3.0,
            'LOW': 0.0
        }
        
    def detect_predictive_value(self, predictive_results: Dict, market_odds: Dict) -> List[BettingSignal]:
        """Find value based on predictive power, not just probability differences"""
        signals = []
        
        model_probs = predictive_results['probabilities']
        predictive_metrics = predictive_results['predictive_metrics']
        
        # Convert market odds to probabilities
        market_probs = self._calculate_implied_probabilities(market_odds)
        
        # Check each market for predictive value
        markets_to_check = [
            ('home_win', '1x2 Home'),
            ('away_win', '1x2 Away'), 
            ('draw', '1x2 Draw'),
            ('over_25', 'Over 2.5 Goals'),
            ('btts_yes', 'BTTS Yes')
        ]
        
        for model_key, market_name in markets_to_check:
            model_prob = model_probs[model_key] / 100.0
            market_prob = market_probs.get(market_name, 0)
            
            if market_prob > 0 and model_prob > 0:
                # Calculate edge with predictive power weighting
                raw_edge = (model_prob - market_prob) * 100
                predictive_power = predictive_metrics['predictive_confidence']
                market_mispricing = predictive_metrics['market_mispricing'] / 100.0
                
                # Enhanced edge calculation
                adjusted_edge = raw_edge * (1 + predictive_power) * (1 + market_mispricing)
                
                if adjusted_edge > self.value_thresholds['MODERATE']:
                    value_rating = self._get_value_rating(adjusted_edge)
                    confidence = self._calculate_predictive_confidence(
                        model_prob, adjusted_edge, predictive_power
                    )
                    stake = self._calculate_predictive_stake(
                        model_prob, market_prob, predictive_power
                    )
                    
                    signals.append(BettingSignal(
                        market=market_name,
                        model_prob=round(model_prob * 100, 1),
                        book_prob=round(market_prob * 100, 1),
                        edge=round(adjusted_edge, 1),
                        confidence=confidence,
                        recommended_stake=stake,
                        value_rating=value_rating,
                        predictive_power=round(predictive_power, 3)
                    ))
        
        return sorted(signals, key=lambda x: x.edge, reverse=True)
    
    def _calculate_implied_probabilities(self, market_odds: Dict) -> Dict[str, float]:
        """Convert odds to probabilities with margin removal"""
        implied_probs = {}
        
        for market, odds in market_odds.items():
            if isinstance(odds, (int, float)) and odds > 1:
                # Simple margin removal
                implied_probs[market] = (1 / odds) * 0.95  # Adjust for typical margin
        
        return implied_probs
    
    def _get_value_rating(self, edge: float) -> str:
        for rating, threshold in self.value_thresholds.items():
            if edge >= threshold:
                return rating
        return "LOW"
    
    def _calculate_predictive_confidence(self, model_prob: float, edge: float, predictive_power: float) -> str:
        """Confidence based on predictive power"""
        base_confidence = "SPECULATIVE"
        
        if predictive_power > 0.7 and edge > 8:
            base_confidence = "HIGH"
        elif predictive_power > 0.5 and edge > 5:
            base_confidence = "MEDIUM"
        elif predictive_power > 0.3 and edge > 3:
            base_confidence = "LOW"
            
        return base_confidence
    
    def _calculate_predictive_stake(self, model_prob: float, market_prob: float, predictive_power: float) -> float:
        """Stake sizing based on predictive power"""
        kelly_stake = (model_prob * (1/market_prob) - 1) / ((1/market_prob) - 1)
        predictive_adjustment = min(1.0, predictive_power * 1.5)  # Scale by predictive power
        
        return max(0.0, min(0.03, kelly_stake * 0.25 * predictive_adjustment))  # Very conservative

class PerformanceTracker:
    """Track actual performance to improve predictions"""
    
    def __init__(self):
        self.bet_history = []
        self.accuracy_by_market = {}
        
    def add_prediction(self, prediction: Dict, actual_result: str):
        """Track prediction accuracy"""
        self.bet_history.append({
            'timestamp': datetime.now(),
            'prediction': prediction,
            'actual_result': actual_result,
            'correct': self._was_prediction_correct(prediction, actual_result)
        })
        
    def _was_prediction_correct(self, prediction: Dict, actual_result: str) -> bool:
        """Check if the highest probability outcome occurred"""
        probs = prediction['probabilities']
        predicted_outcome = max(probs, key=probs.get)
        
        # Map actual result to prediction format
        result_map = {'H': 'home_win', 'D': 'draw', 'A': 'away_win'}
        return result_map.get(actual_result, '') == predicted_outcome
    
    def get_performance_metrics(self) -> Dict:
        """Calculate actual performance"""
        if len(self.bet_history) < 10:
            return {"status": "INSUFFICIENT_DATA"}
            
        correct_predictions = sum(1 for bet in self.bet_history if bet['correct'])
        total_predictions = len(self.bet_history)
        accuracy = correct_predictions / total_predictions
        
        return {
            'total_predictions': total_predictions,
            'accuracy': round(accuracy * 100, 1),
            'expected_accuracy': 55.0,  # Target
            'performance_gap': round((accuracy - 0.55) * 100, 1)
        }

# Main orchestrator
class TruePredictiveFootballEngine:
    """Complete predictive football engine"""
    
    def __init__(self, match_data: Dict, historical_data: Optional[pd.DataFrame] = None):
        self.match_data = match_data
        self.historical_data = historical_data
        self.predictive_engine = TruePredictiveEngine(match_data, historical_data)
        self.value_engine = PredictiveValueEngine()
        self.performance_tracker = PerformanceTracker()
        
    def generate_predictive_analysis(self) -> Dict[str, Any]:
        """Generate complete predictive analysis"""
        # Generate true predictive probabilities
        predictive_results = self.predictive_engine.run_predictive_simulation()
        
        # Detect value bets based on predictive power
        market_odds = self.match_data.get('market_odds', {})
        value_signals = self.value_engine.detect_predictive_value(predictive_results, market_odds)
        
        # Compile comprehensive results
        analysis = {
            'match': f"{self.match_data['home_team']} vs {self.match_data['away_team']}",
            'predictive_results': predictive_results,
            'value_signals': [signal.__dict__ for signal in value_signals],
            'analysis_timestamp': datetime.now().isoformat(),
            'predictive_power_rating': self._calculate_overall_predictive_power(predictive_results)
        }
        
        return analysis
    
    def _calculate_overall_predictive_power(self, results: Dict) -> str:
        """Overall rating of predictive power for this match"""
        metrics = results['predictive_metrics']
        sustainability = metrics['sustainability_score']
        confidence = metrics['predictive_confidence']
        
        if sustainability > 0.8 and confidence > 0.7:
            return "HIGH_PREDICTIVE_POWER"
        elif sustainability > 0.6 and confidence > 0.5:
            return "MEDIUM_PREDICTIVE_POWER"
        else:
            return "LOW_PREDICTIVE_POWER"

# Example usage with realistic predictive data
if __name__ == "__main__":
    # REAL predictive data - not just goals and results
    predictive_match_data = {
        'home_team': 'Bologna',
        'away_team': 'Torino',
        'home_xg': 1.8,           # Expected goals, not actual
        'away_xg': 1.2,           # Expected goals, not actual  
        'home_xg_against': 0.9,   # Defensive quality
        'away_xg_against': 1.1,   # Defensive quality
        'home_possession_quality': 0.58,  # Game control
        'away_possession_quality': 0.52,  # Game control
        'home_press_intensity': 18.5,     # Pressing effectiveness
        'away_press_intensity': 16.2,     # Pressing effectiveness
        'home_recent_wins': 2,            # For bias detection
        'away_recent_wins': 1,            # For bias detection
        'market_odds': {
            '1x2 Home': 2.30,
            '1x2 Draw': 3.10,
            '1x2 Away': 3.40,
            'Over 2.5 Goals': 2.10,
            'BTTS Yes': 1.95
        }
    }
    
    engine = TruePredictiveFootballEngine(predictive_match_data)
    results = engine.generate_predictive_analysis()
    
    print("TRUE PREDICTIVE ANALYSIS:")
    print(f"Match: {results['match']}")
    print(f"Predictive Power: {results['predictive_power_rating']}")
    print(f"Expected Goals: Home {results['predictive_results']['expected_goals']['home']} - Away {results['predictive_results']['expected_goals']['away']}")
    print(f"Probabilities: {results['predictive_results']['probabilities']}")
    print(f"Value Signals: {len(results['value_signals'])} detected")
    
    for signal in results['value_signals']:
        print(f"  {signal['market']}: +{signal['edge']}% edge (Predictive Power: {signal['predictive_power']})")
