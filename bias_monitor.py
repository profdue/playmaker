# bias_monitor.py
"""Institutional-grade bias monitoring and performance calibration with advanced analytics"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
import os
from scipy.optimize import curve_fit
import warnings
from collections import deque
import math

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class BiasType(Enum):
    OVERCONFIDENCE = "overconfidence"
    UNDERCONFIDENCE = "underconfidence" 
    HOME_BIAS = "home_bias"
    FAVORITE_BIAS = "favorite_bias"
    UNDERDOG_BIAS = "underdog_bias"
    DRAW_AVERSION = "draw_aversion"
    PROBABILITY_CLUSTERING = "probability_clustering"
    RECENCY_BIAS = "recency_bias"
    MARKET_INEFFICIENCY = "market_inefficiency"

class CalibrationStatus(Enum):
    OPTIMAL = "optimal"
    ACCEPTABLE = "acceptable"
    DEGRADING = "degrading"
    POOR = "poor"
    CRITICAL = "critical"

class StatisticalSignificance(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSIGNIFICANT = "insignificant"

@dataclass
class BiasDetection:
    bias_type: BiasType
    severity: float  # 0-1 scale
    evidence: str
    impact_on_accuracy: float
    recommendation: str
    confidence_interval: Tuple[float, float]
    statistical_significance: StatisticalSignificance
    sample_size: int

@dataclass
class CalibrationMetrics:
    brier_score: float
    brier_score_ci: Tuple[float, float]
    calibration_error: float
    calibration_error_ci: Tuple[float, float]
    confidence_reliability: float
    probability_resolution: float
    status: CalibrationStatus
    log_score: float
    sharpness: float

@dataclass
class PerformanceAnalysis:
    accuracy: float
    accuracy_ci: Tuple[float, float]
    expected_accuracy: float
    calibration_gap: float
    value_betting_performance: float
    recent_trend: str
    trend_strength: float
    market_efficiency: float
    kelly_criterion_performance: float

@dataclass
class BayesianUpdate:
    prior_mean: float
    prior_std: float
    posterior_mean: float
    posterior_std: float
    update_strength: float

class InstitutionalBiasMonitor:
    """
    Enhanced institutional-grade bias monitoring with advanced analytics
    """
    
    def __init__(self, log_file: str = "professional_bias_log.json"):
        self.log_file = log_file
        self.calibration_file = "model_calibration.json"
        
        # Enhanced professional monitoring configuration
        self.config = {
            'min_samples_analysis': 50,
            'recent_performance_window': 100,
            'calibration_update_frequency': 50,
            'confidence_level': 0.95,
            'trend_analysis_window': 30,
            
            # Enhanced thresholds with confidence intervals
            'brier_score_thresholds': {
                'excellent': (0.12, 0.15),
                'good': (0.15, 0.20),
                'acceptable': (0.20, 0.25),
                'poor': (0.25, 0.30),
                'critical': (0.30, 1.0)
            },
            
            'bias_severity_thresholds': {
                'low': 0.05,
                'medium': 0.10,
                'high': 0.15,
                'critical': 0.20
            },
            
            'statistical_significance_thresholds': {
                'high': 0.01,
                'medium': 0.05,
                'low': 0.10
            },
            
            # Bayesian prior parameters
            'bayesian_priors': {
                'accuracy': {'mean': 0.55, 'std': 0.1},
                'home_bias': {'mean': 0.0, 'std': 0.05},
                'confidence_calibration': {'mean': 1.0, 'std': 0.1}
            },
            
            # Market efficiency parameters
            'market_efficiency_threshold': 0.02,  # 2% edge for inefficiency
            'minimum_odds_comparison_samples': 20
        }
        
        # Initialize calibration state with Bayesian priors
        self.calibration_state = self._load_calibration_state()
        
        # Advanced analytics state
        self.market_odds_data = {}
        self.weather_context_data = {}
        self.performance_trends = deque(maxlen=200)
        
        logger.info("Initialized Enhanced InstitutionalBiasMonitor")

    def _load_calibration_state(self) -> Dict[str, Any]:
        """Load existing calibration state or initialize with Bayesian priors"""
        try:
            if os.path.exists(self.calibration_file):
                with open(self.calibration_file, 'r') as f:
                    state = json.load(f)
                    # Ensure new fields exist
                    state.setdefault('bayesian_posteriors', {})
                    state.setdefault('market_efficiency_metrics', {})
                    state.setdefault('advanced_metrics', {})
                    return state
        except Exception as e:
            logger.warning(f"Could not load calibration state: {e}")
        
        # Enhanced default calibration state with Bayesian priors
        return {
            'historical_accuracy': 0.55,
            'calibration_factors': {
                'confidence_adjustment': 1.0,
                'home_bias_correction': 0.0,
                'favorite_adjustment': 1.0,
                'draw_adjustment': 1.0,
                'probability_smoothing': 0.0
            },
            'bayesian_posteriors': {
                'accuracy': self.config['bayesian_priors']['accuracy'],
                'home_bias': self.config['bayesian_priors']['home_bias'],
                'confidence_calibration': self.config['bayesian_priors']['confidence_calibration']
            },
            'market_efficiency_metrics': {
                'avg_value_edge': 0.0,
                'efficient_market_percentage': 0.0,
                'profitable_market_percentage': 0.0
            },
            'pattern_success_rates': {},
            'performance_history': [],
            'advanced_metrics': {
                'volatility_index': 0.0,
                'stability_score': 0.0,
                'predictability_index': 0.0
            },
            'last_calibration_update': datetime.now().isoformat(),
            'version': '3.0.0'
        }

    def _save_calibration_state(self):
        """Save calibration state to file"""
        try:
            self.calibration_state['last_calibration_update'] = datetime.now().isoformat()
            with open(self.calibration_file, 'w') as f:
                json.dump(self.calibration_state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving calibration state: {e}")

    def log_prediction_with_result(self, match_id: str, predictions: Dict, 
                                 actual_result: Dict, confidence: float,
                                 market_odds: Optional[Dict] = None,
                                 context_data: Optional[Dict] = None):
        """
        Enhanced prediction logging with market odds and context data
        """
        try:
            # Determine actual outcome
            home_goals = actual_result.get('home_goals', 0)
            away_goals = actual_result.get('away_goals', 0)
            
            if home_goals > away_goals:
                actual_outcome = 'home'
            elif away_goals > home_goals:
                actual_outcome = 'away'
            else:
                actual_outcome = 'draw'
            
            # Get predicted probabilities
            pred_1x2 = predictions['predictions']['1X2']
            home_prob = pred_1x2['Home Win']
            draw_prob = pred_1x2['Draw']
            away_prob = pred_1x2['Away Win']
            
            # Determine predicted outcome
            if home_prob >= draw_prob and home_prob >= away_prob:
                predicted_outcome = 'home'
                predicted_confidence = home_prob
            elif away_prob >= home_prob and away_prob >= draw_prob:
                predicted_outcome = 'away'
                predicted_confidence = away_prob
            else:
                predicted_outcome = 'draw'
                predicted_confidence = draw_prob
            
            is_correct = predicted_outcome == actual_outcome
            
            # Calculate enhanced metrics
            if actual_outcome == 'home':
                actual_prob_vector = [1.0, 0.0, 0.0]
            elif actual_outcome == 'away':
                actual_prob_vector = [0.0, 0.0, 1.0]
            else:
                actual_prob_vector = [0.0, 1.0, 0.0]
            
            predicted_prob_vector = [home_prob/100, draw_prob/100, away_prob/100]
            
            # Core metrics
            brier_score = sum((p - a) ** 2 for p, a in zip(predicted_prob_vector, actual_prob_vector))
            log_score = -np.log(predicted_prob_vector[['home', 'draw', 'away'].index(actual_outcome)] + 1e-10)
            
            # Market efficiency analysis
            market_efficiency = self._calculate_market_efficiency(predicted_prob_vector, market_odds, actual_outcome)
            
            # Value betting metrics
            kelly_performance = self._calculate_kelly_performance(predicted_prob_vector, market_odds, actual_outcome)
            
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'match_id': match_id,
                'predictions': {
                    'home_prob': home_prob,
                    'draw_prob': draw_prob,
                    'away_prob': away_prob,
                    'predicted_outcome': predicted_outcome,
                    'predicted_confidence': predicted_confidence
                },
                'actual_result': {
                    'outcome': actual_outcome,
                    'home_goals': home_goals,
                    'away_goals': away_goals
                },
                'performance': {
                    'is_correct': is_correct,
                    'brier_score': brier_score,
                    'log_score': log_score,
                    'model_confidence': confidence,
                    'confidence_gap': (predicted_confidence/100) - (1.0 if is_correct else 0.0),
                    'market_efficiency': market_efficiency,
                    'kelly_performance': kelly_performance
                },
                'metadata': {
                    'probability_entropy': self._calculate_entropy([home_prob, draw_prob, away_prob]),
                    'max_probability': max(home_prob, draw_prob, away_prob),
                    'favorite_defined': max(home_prob, away_prob) > 45,
                    'probability_clustering': self._detect_probability_clustering(home_prob, draw_prob, away_prob),
                    'market_odds': market_odds,
                    'context_data': context_data
                }
            }
            
            # Append to log file
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            # Update performance history with enhanced metrics
            self._update_performance_history(log_entry)
            
            # Update market efficiency metrics
            if market_odds:
                self._update_market_efficiency_metrics(log_entry)
            
            # Check if calibration update is needed
            if len(self.calibration_state['performance_history']) % self.config['calibration_update_frequency'] == 0:
                self._update_calibration_factors()
                self._update_bayesian_posteriors()
            
            logger.debug(f"Logged prediction: {match_id}, Correct: {is_correct}, Brier: {brier_score:.3f}")
            
        except Exception as e:
            logger.error(f"Error logging prediction: {e}")

    def _calculate_entropy(self, probabilities: List[float]) -> float:
        """Calculate entropy of probability distribution"""
        try:
            probs = np.array(probabilities) / 100.0
            probs = probs[probs > 0]  # Remove zeros
            return -np.sum(probs * np.log(probs))
        except Exception:
            return 0.0

    def _detect_probability_clustering(self, home_prob: float, draw_prob: float, away_prob: float) -> bool:
        """Detect if probabilities are clustering around common values"""
        common_values = [10, 15, 20, 25, 30, 33, 40, 50, 60, 67, 70, 75, 80, 85, 90]
        for prob in [home_prob, draw_prob, away_prob]:
            if any(abs(prob - common) <= 2 for common in common_values):
                return True
        return False

    def _calculate_market_efficiency(self, predicted_probs: List[float], market_odds: Optional[Dict], actual_outcome: str) -> float:
        """Calculate market efficiency metric"""
        if not market_odds:
            return 0.0
        
        try:
            # Convert odds to implied probabilities
            if 'home' in market_odds and 'draw' in market_odds and 'away' in market_odds:
                home_implied = 1 / market_odds['home'] if market_odds['home'] > 0 else 0
                draw_implied = 1 / market_odds['draw'] if market_odds['draw'] > 0 else 0
                away_implied = 1 / market_odds['away'] if market_odds['away'] > 0 else 0
                
                # Normalize
                total = home_implied + draw_implied + away_implied
                if total > 0:
                    home_implied /= total
                    draw_implied /= total
                    away_implied /= total
                    
                    # Calculate value edge
                    actual_idx = ['home', 'draw', 'away'].index(actual_outcome)
                    model_edge = predicted_probs[actual_idx] - [home_implied, draw_implied, away_implied][actual_idx]
                    return max(-1.0, min(1.0, model_edge))
            
            return 0.0
        except Exception:
            return 0.0

    def _calculate_kelly_performance(self, predicted_probs: List[float], market_odds: Optional[Dict], actual_outcome: str) -> float:
        """Calculate Kelly criterion performance"""
        if not market_odds:
            return 0.0
        
        try:
            outcome_idx = ['home', 'draw', 'away'].index(actual_outcome)
            model_prob = predicted_probs[outcome_idx]
            
            # Get odds for the actual outcome
            odds_key = list(market_odds.keys())[outcome_idx]
            decimal_odds = market_odds[odds_key]
            
            if decimal_odds > 0:
                # Kelly fraction
                kelly_fraction = (decimal_odds * model_prob - 1) / (decimal_odds - 1)
                kelly_fraction = max(0, min(0.25, kelly_fraction))  # Cap at 25%
                
                # Calculate return
                if actual_outcome == list(market_odds.keys())[outcome_idx]:
                    return kelly_fraction * (decimal_odds - 1)
                else:
                    return -kelly_fraction
            
            return 0.0
        except Exception:
            return 0.0

    def _update_performance_history(self, log_entry: Dict):
        """Update performance history with enhanced metrics"""
        try:
            performance_record = {
                'timestamp': log_entry['timestamp'],
                'is_correct': log_entry['performance']['is_correct'],
                'brier_score': log_entry['performance']['brier_score'],
                'log_score': log_entry['performance']['log_score'],
                'confidence': log_entry['performance']['model_confidence'],
                'confidence_gap': log_entry['performance']['confidence_gap'],
                'max_probability': log_entry['metadata']['max_probability'],
                'market_efficiency': log_entry['performance']['market_efficiency'],
                'kelly_performance': log_entry['performance']['kelly_performance'],
                'entropy': log_entry['metadata']['probability_entropy']
            }
            
            self.calibration_state['performance_history'].append(performance_record)
            self.performance_trends.append(performance_record)
            
            # Keep only recent history
            if len(self.calibration_state['performance_history']) > 500:
                self.calibration_state['performance_history'] = self.calibration_state['performance_history'][-500:]
                
        except Exception as e:
            logger.error(f"Error updating performance history: {e}")

    def _update_market_efficiency_metrics(self, log_entry: Dict):
        """Update market efficiency metrics"""
        try:
            efficiency = log_entry['performance']['market_efficiency']
            kelly_perf = log_entry['performance']['kelly_performance']
            
            # Update running averages
            current_avg = self.calibration_state['market_efficiency_metrics'].get('avg_value_edge', 0.0)
            count = self.calibration_state['market_efficiency_metrics'].get('sample_count', 0)
            
            new_avg = (current_avg * count + efficiency) / (count + 1)
            
            self.calibration_state['market_efficiency_metrics']['avg_value_edge'] = new_avg
            self.calibration_state['market_efficiency_metrics']['sample_count'] = count + 1
            
            # Update efficiency percentages
            if efficiency > self.config['market_efficiency_threshold']:
                efficient_count = self.calibration_state['market_efficiency_metrics'].get('efficient_count', 0) + 1
                self.calibration_state['market_efficiency_metrics']['efficient_count'] = efficient_count
                self.calibration_state['market_efficiency_metrics']['efficient_market_percentage'] = efficient_count / (count + 1)
            
            if kelly_perf > 0:
                profitable_count = self.calibration_state['market_efficiency_metrics'].get('profitable_count', 0) + 1
                self.calibration_state['market_efficiency_metrics']['profitable_count'] = profitable_count
                self.calibration_state['market_efficiency_metrics']['profitable_market_percentage'] = profitable_count / (count + 1)
                
        except Exception as e:
            logger.error(f"Error updating market efficiency metrics: {e}")

    def _update_calibration_factors(self):
        """Enhanced calibration factors update with volatility analysis"""
        try:
            if len(self.calibration_state['performance_history']) < self.config['min_samples_analysis']:
                return
            
            recent_performance = self.calibration_state['performance_history'][-self.config['recent_performance_window']:]
            
            # Calculate current accuracy with CI
            accuracies = [p['is_correct'] for p in recent_performance]
            accuracy = np.mean(accuracies)
            accuracy_ci = self._calculate_confidence_interval(accuracies)
            
            # Calculate confidence calibration
            avg_confidence = np.mean([p['confidence'] for p in recent_performance]) / 100.0
            confidence_calibration_ratio = accuracy / avg_confidence if avg_confidence > 0 else 1.0
            
            # Calculate enhanced biases
            home_bias = self._calculate_home_bias()
            favorite_bias = self._calculate_favorite_bias()
            draw_bias = self._calculate_draw_bias()
            
            # Update calibration factors with smoothing
            self.calibration_state['historical_accuracy'] = accuracy
            
            # Apply Bayesian smoothing to calibration factors
            current_conf_adj = self.calibration_state['calibration_factors']['confidence_adjustment']
            new_conf_adj = max(0.7, min(1.3, confidence_calibration_ratio))
            self.calibration_state['calibration_factors']['confidence_adjustment'] = 0.7 * current_conf_adj + 0.3 * new_conf_adj
            
            current_home_corr = self.calibration_state['calibration_factors']['home_bias_correction']
            new_home_corr = max(-0.15, min(0.15, home_bias))
            self.calibration_state['calibration_factors']['home_bias_correction'] = 0.8 * current_home_corr + 0.2 * new_home_corr
            
            # Update advanced metrics
            self._update_advanced_metrics(recent_performance)
            
            self._save_calibration_state()
            
            logger.info(f"Calibration updated: Accuracy={accuracy:.3f}±{accuracy_ci[1]-accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Error updating calibration factors: {e}")

    def _update_bayesian_posteriors(self):
        """Update Bayesian posterior distributions"""
        try:
            if len(self.calibration_state['performance_history']) < 10:
                return
            
            recent_performance = self.calibration_state['performance_history'][-50:]
            
            # Update accuracy posterior
            accuracies = [p['is_correct'] for p in recent_performance]
            accuracy_update = self._bayesian_update(
                self.calibration_state['bayesian_posteriors']['accuracy'],
                accuracies
            )
            self.calibration_state['bayesian_posteriors']['accuracy'] = accuracy_update
            
            # Update home bias posterior
            home_bias_data = self._calculate_home_bias_series()
            if home_bias_data:
                home_bias_update = self._bayesian_update(
                    self.calibration_state['bayesian_posteriors']['home_bias'],
                    home_bias_data
                )
                self.calibration_state['bayesian_posteriors']['home_bias'] = home_bias_update
            
        except Exception as e:
            logger.error(f"Error updating Bayesian posteriors: {e}")

    def _bayesian_update(self, prior: Dict, data: List[float]) -> Dict[str, float]:
        """Perform Bayesian update for normal distribution"""
        try:
            prior_mean = prior['mean']
            prior_std = prior['std']
            data_mean = np.mean(data)
            data_std = np.std(data)
            n = len(data)
            
            if n == 0 or data_std == 0:
                return prior
            
            # Bayesian update for normal-normal conjugate
            posterior_variance = 1 / (1/prior_std**2 + n/data_std**2)
            posterior_mean = posterior_variance * (prior_mean/prior_std**2 + n*data_mean/data_std**2)
            posterior_std = math.sqrt(posterior_variance)
            
            return {
                'mean': float(posterior_mean),
                'std': float(posterior_std)
            }
        except Exception:
            return prior

    def _calculate_confidence_interval(self, data: List[float], confidence: float = None) -> Tuple[float, float]:
        """Calculate confidence interval for data"""
        if confidence is None:
            confidence = self.config['confidence_level']
        
        try:
            if len(data) < 2:
                return (np.mean(data), np.mean(data))
            
            return stats.t.interval(confidence, len(data)-1, loc=np.mean(data), scale=stats.sem(data))
        except Exception:
            return (np.mean(data), np.mean(data))

    def _calculate_home_bias(self) -> float:
        """Calculate home team prediction bias with enhanced analysis"""
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < self.config['min_samples_analysis']:
                return 0.0
            
            home_probs = []
            home_correct = []
            
            for line in lines[-self.config['recent_performance_window']:]:
                data = json.loads(line.strip())
                home_probs.append(data['predictions']['home_prob'])
                home_correct.append(1.0 if data['actual_result']['outcome'] == 'home' else 0.0)
            
            avg_predicted_home = np.mean(home_probs) / 100.0
            actual_home_win_rate = np.mean(home_correct)
            
            # Statistical significance test
            if len(home_probs) >= 30:
                t_stat, p_value = stats.ttest_1samp(home_probs, actual_home_win_rate * 100)
            else:
                p_value = 0.5
            
            home_bias = avg_predicted_home - actual_home_win_rate
            
            # Adjust for statistical significance
            if p_value > 0.1:  # Not statistically significant
                home_bias *= 0.5
            
            return home_bias
            
        except Exception as e:
            logger.error(f"Error calculating home bias: {e}")
            return 0.0

    def _calculate_home_bias_series(self) -> List[float]:
        """Calculate home bias time series for Bayesian analysis"""
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 10:
                return []
            
            home_biases = []
            window_size = 20
            
            for i in range(0, len(lines) - window_size + 1, 5):
                window_lines = lines[i:i + window_size]
                home_probs = []
                home_correct = []
                
                for line in window_lines:
                    data = json.loads(line.strip())
                    home_probs.append(data['predictions']['home_prob'])
                    home_correct.append(1.0 if data['actual_result']['outcome'] == 'home' else 0.0)
                
                if home_probs:
                    avg_pred = np.mean(home_probs) / 100.0
                    avg_actual = np.mean(home_correct)
                    home_biases.append(avg_pred - avg_actual)
            
            return home_biases
            
        except Exception as e:
            logger.error(f"Error calculating home bias series: {e}")
            return []

    def _calculate_favorite_bias(self) -> float:
        """Calculate favorite team prediction bias"""
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            
            favorite_predictions = []
            for line in lines[-self.config['recent_performance_window']:]:
                data = json.loads(line.strip())
                if data['metadata']['favorite_defined']:
                    favorite_predictions.append(data)
            
            if len(favorite_predictions) < 10:
                return 0.0
            
            favorite_accuracies = []
            expected_accuracies = []
            
            for pred in favorite_predictions:
                is_favorite_win = (
                    (pred['predictions']['home_prob'] == pred['metadata']['max_probability'] and pred['actual_result']['outcome'] == 'home') or
                    (pred['predictions']['away_prob'] == pred['metadata']['max_probability'] and pred['actual_result']['outcome'] == 'away')
                )
                favorite_accuracies.append(1.0 if is_favorite_win else 0.0)
                expected_accuracies.append(pred['metadata']['max_probability'] / 100.0)
            
            avg_expected = np.mean(expected_accuracies)
            avg_actual = np.mean(favorite_accuracies)
            
            return avg_expected - avg_actual
            
        except Exception as e:
            logger.error(f"Error calculating favorite bias: {e}")
            return 0.0

    def _calculate_draw_bias(self) -> float:
        """Calculate draw prediction bias"""
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            
            draw_probs = []
            actual_draws = []
            
            for line in lines[-self.config['recent_performance_window']:]:
                data = json.loads(line.strip())
                draw_probs.append(data['predictions']['draw_prob'])
                actual_draws.append(1.0 if data['actual_result']['outcome'] == 'draw' else 0.0)
            
            avg_predicted_draw = np.mean(draw_probs) / 100.0
            avg_actual_draw = np.mean(actual_draws)
            
            return avg_predicted_draw - avg_actual_draw
            
        except Exception as e:
            logger.error(f"Error calculating draw bias: {e}")
            return 0.0

    def _update_advanced_metrics(self, recent_performance: List[Dict]):
        """Update advanced performance metrics"""
        try:
            # Volatility index (standard deviation of accuracy)
            accuracies = [p['is_correct'] for p in recent_performance]
            if len(accuracies) >= 10:
                volatility = np.std(accuracies)
                self.calibration_state['advanced_metrics']['volatility_index'] = volatility
            
            # Stability score (inverse of volatility)
            self.calibration_state['advanced_metrics']['stability_score'] = max(0, 1 - volatility * 2)
            
            # Predictability index (R-squared of performance trend)
            if len(accuracies) >= 20:
                x = np.arange(len(accuracies))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, accuracies)
                self.calibration_state['advanced_metrics']['predictability_index'] = r_value ** 2
            
        except Exception as e:
            logger.error(f"Error updating advanced metrics: {e}")

    def analyze_comprehensive_bias(self) -> Dict[str, Any]:
        """
        Enhanced comprehensive bias analysis with advanced metrics
        """
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < self.config['min_samples_analysis']:
                return {
                    "status": "INSUFFICIENT_DATA",
                    "samples": len(lines),
                    "required_samples": self.config['min_samples_analysis'],
                    "message": f"Need {self.config['min_samples_analysis']} samples, have {len(lines)}"
                }
            
            # Parse all data
            predictions_data = [json.loads(line.strip()) for line in lines]
            
            # Calculate enhanced metrics
            calibration_metrics = self._calculate_enhanced_calibration_metrics(predictions_data)
            bias_detections = self._detect_enhanced_biases(predictions_data)
            performance_analysis = self._analyze_enhanced_performance(predictions_data)
            market_efficiency = self._analyze_market_efficiency(predictions_data)
            
            return {
                "status": "COMPLETE",
                "samples": len(predictions_data),
                "analysis_period": self._get_analysis_period(predictions_data),
                "calibration_metrics": calibration_metrics,
                "detected_biases": bias_detections,
                "performance_analysis": performance_analysis,
                "market_efficiency_analysis": market_efficiency,
                "bayesian_posteriors": self.calibration_state['bayesian_posteriors'],
                "advanced_metrics": self.calibration_state['advanced_metrics'],
                "recommendations": self._generate_enhanced_recommendations(calibration_metrics, bias_detections, market_efficiency),
                "calibration_factors": self.calibration_state['calibration_factors'],
                "statistical_reliability": self._assess_statistical_reliability(predictions_data)
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive bias analysis: {e}")
            return {
                "status": "ERROR",
                "error": str(e),
                "samples": 0
            }

    def _calculate_enhanced_calibration_metrics(self, predictions_data: List[Dict]) -> CalibrationMetrics:
        """Calculate enhanced calibration metrics with confidence intervals"""
        try:
            # Enhanced confidence binning with dynamic ranges
            confidence_bins = self._create_dynamic_confidence_bins(predictions_data)
            calibration_errors = []
            brier_scores = []
            log_scores = []
            
            for conf_min, conf_max in confidence_bins:
                bin_predictions = [
                    p for p in predictions_data 
                    if conf_min <= p['predictions']['predicted_confidence'] <= conf_max
                ]
                
                if len(bin_predictions) >= 5:
                    expected_accuracy = (conf_min + conf_max) / 2 / 100.0
                    actual_accuracy = np.mean([p['performance']['is_correct'] for p in bin_predictions])
                    calibration_error = abs(expected_accuracy - actual_accuracy)
                    calibration_errors.append(calibration_error)
            
            # Calculate enhanced scores
            brier_scores = [p['performance']['brier_score'] for p in predictions_data]
            log_scores = [p['performance']['log_score'] for p in predictions_data]
            
            avg_brier_score = np.mean(brier_scores) if brier_scores else 0.25
            brier_ci = self._calculate_confidence_interval(brier_scores)
            
            avg_log_score = np.mean(log_scores) if log_scores else -np.log(0.33)  # Default for random
            
            # Calculate confidence reliability with CI
            avg_confidence = np.mean([p['predictions']['predicted_confidence'] for p in predictions_data]) / 100.0
            actual_accuracy = np.mean([p['performance']['is_correct'] for p in predictions_data])
            accuracy_ci = self._calculate_confidence_interval([p['performance']['is_correct'] for p in predictions_data])
            confidence_reliability = 1.0 - abs(avg_confidence - actual_accuracy)
            
            # Calculate sharpness (how concentrated predictions are)
            max_probs = [p['metadata']['max_probability'] for p in predictions_data]
            sharpness = np.mean(max_probs) / 100.0 if max_probs else 0.5
            
            # Determine calibration status with enhanced thresholds
            status = self._determine_calibration_status(avg_brier_score, np.mean(calibration_errors) if calibration_errors else 0.1)
            
            return CalibrationMetrics(
                brier_score=round(avg_brier_score, 4),
                brier_score_ci=tuple(round(x, 4) for x in brier_ci),
                calibration_error=round(np.mean(calibration_errors) if calibration_errors else 0.1, 4),
                calibration_error_ci=tuple(round(x, 4) for x in self._calculate_confidence_interval(calibration_errors) if calibration_errors else (0.1, 0.1)),
                confidence_reliability=round(confidence_reliability, 4),
                probability_resolution=round(1.0 - avg_brier_score, 4),
                status=status,
                log_score=round(avg_log_score, 4),
                sharpness=round(sharpness, 4)
            )
            
        except Exception as e:
            logger.error(f"Error calculating enhanced calibration metrics: {e}")
            return CalibrationMetrics(0.25, (0.2, 0.3), 0.1, (0.05, 0.15), 0.5, 0.5, CalibrationStatus.CRITICAL, -np.log(0.33), 0.5)

    def _create_dynamic_confidence_bins(self, predictions_data: List[Dict]) -> List[Tuple[float, float]]:
        """Create dynamic confidence bins based on data distribution"""
        confidences = [p['predictions']['predicted_confidence'] for p in predictions_data]
        
        if len(confidences) < 50:
            return [(0, 40), (40, 60), (60, 75), (75, 100)]
        
        # Use quantile-based binning for better distribution
        quantiles = [0, 0.25, 0.5, 0.75, 1.0]
        bins = []
        q_values = np.quantile(confidences, quantiles)
        
        for i in range(len(q_values)-1):
            bins.append((q_values[i], q_values[i+1]))
        
        return bins

    def _determine_calibration_status(self, brier_score: float, calibration_error: float) -> CalibrationStatus:
        """Determine calibration status with enhanced logic"""
        if brier_score <= self.config['brier_score_thresholds']['excellent'][1]:
            return CalibrationStatus.OPTIMAL
        elif brier_score <= self.config['brier_score_thresholds']['good'][1]:
            return CalibrationStatus.ACCEPTABLE
        elif brier_score <= self.config['brier_score_thresholds']['acceptable'][1]:
            return CalibrationStatus.DEGRADING
        elif brier_score <= self.config['brier_score_thresholds']['poor'][1]:
            return CalibrationStatus.POOR
        else:
            return CalibrationStatus.CRITICAL

    def _detect_enhanced_biases(self, predictions_data: List[Dict]) -> List[BiasDetection]:
        """Detect biases with enhanced statistical significance testing"""
        biases = []
        
        try:
            # 1. Home Team Bias with significance testing
            home_bias = self._detect_home_bias_enhanced(predictions_data)
            if home_bias:
                biases.append(home_bias)
            
            # 2. Over/Under Confidence Bias
            confidence_bias = self._detect_confidence_bias_enhanced(predictions_data)
            if confidence_bias:
                biases.append(confidence_bias)
            
            # 3. Favorite Bias
            favorite_bias = self._detect_favorite_bias_enhanced(predictions_data)
            if favorite_bias:
                biases.append(favorite_bias)
            
            # 4. Draw Aversion Bias
            draw_bias = self._detect_draw_bias_enhanced(predictions_data)
            if draw_bias:
                biases.append(draw_bias)
            
            # 5. Probability Clustering
            clustering_bias = self._detect_probability_clustering_bias(predictions_data)
            if clustering_bias:
                biases.append(clustering_bias)
            
            # 6. Recency Bias
            recency_bias = self._detect_recency_bias(predictions_data)
            if recency_bias:
                biases.append(recency_bias)
            
        except Exception as e:
            logger.error(f"Error detecting enhanced biases: {e}")
        
        return biases

    def _detect_home_bias_enhanced(self, predictions_data: List[Dict]) -> Optional[BiasDetection]:
        """Enhanced home bias detection with statistical testing"""
        try:
            home_probs = [p['predictions']['home_prob'] for p in predictions_data]
            actual_home_wins = [1.0 if p['actual_result']['outcome'] == 'home' else 0.0 for p in predictions_data]
            
            avg_predicted_home = np.mean(home_probs) / 100.0
            avg_actual_home = np.mean(actual_home_wins)
            
            home_bias = avg_predicted_home - avg_actual_home
            
            # Statistical significance test
            t_stat, p_value = stats.ttest_1samp(home_probs, avg_actual_home * 100)
            
            # Calculate confidence interval
            bias_ci = self._calculate_confidence_interval([home_probs[i]/100 - actual_home_wins[i] for i in range(len(home_probs))])
            
            significance = self._determine_statistical_significance(p_value)
            
            if abs(home_bias) > self.config['bias_severity_thresholds']['low'] and significance != StatisticalSignificance.INSIGNIFICANT:
                severity = min(1.0, abs(home_bias) / 0.2)
                direction = "over" if home_bias > 0 else "under"
                
                return BiasDetection(
                    bias_type=BiasType.HOME_BIAS,
                    severity=severity,
                    evidence=f"Home win prediction {direction} by {abs(home_bias):.1%} (predicted: {avg_predicted_home:.1%}, actual: {avg_actual_home:.1%}, p={p_value:.3f})",
                    impact_on_accuracy=abs(home_bias) * 0.5,
                    recommendation=f"Apply {direction} correction of {abs(home_bias):.1%} to home win probabilities",
                    confidence_interval=tuple(round(x, 4) for x in bias_ci),
                    statistical_significance=significance,
                    sample_size=len(home_probs)
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting enhanced home bias: {e}")
            return None

    def _detect_confidence_bias_enhanced(self, predictions_data: List[Dict]) -> Optional[BiasDetection]:
        """Enhanced confidence bias detection"""
        try:
            confidences = [p['predictions']['predicted_confidence'] for p in predictions_data]
            accuracies = [1.0 if p['performance']['is_correct'] else 0.0 for p in predictions_data]
            
            avg_confidence = np.mean(confidences) / 100.0
            avg_accuracy = np.mean(accuracies)
            
            confidence_gap = avg_confidence - avg_accuracy
            
            # Statistical test
            if len(confidences) >= 30:
                confidences_scaled = [c/100 for c in confidences]
                t_stat, p_value = stats.ttest_rel(confidences_scaled, accuracies)
            else:
                p_value = 0.5
            
            significance = self._determine_statistical_significance(p_value)
            
            if abs(confidence_gap) > self.config['bias_severity_thresholds']['low'] and significance != StatisticalSignificance.INSIGNIFICANT:
                severity = min(1.0, abs(confidence_gap) / 0.15)
                bias_type = BiasType.OVERCONFIDENCE if confidence_gap > 0 else BiasType.UNDERCONFIDENCE
                
                return BiasDetection(
                    bias_type=bias_type,
                    severity=severity,
                    evidence=f"Model is {bias_type.value} by {abs(confidence_gap):.1%} (confidence: {avg_confidence:.1%}, accuracy: {avg_accuracy:.1%}, p={p_value:.3f})",
                    impact_on_accuracy=abs(confidence_gap) * 0.3,
                    recommendation=f"Adjust confidence scores by factor of {avg_accuracy/max(avg_confidence, 0.01):.2f}",
                    confidence_interval=self._calculate_confidence_interval([confidences[i]/100 - accuracies[i] for i in range(len(confidences))]),
                    statistical_significance=significance,
                    sample_size=len(confidences)
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting enhanced confidence bias: {e}")
            return None

    def _detect_favorite_bias_enhanced(self, predictions_data: List[Dict]) -> Optional[BiasDetection]:
        """Enhanced favorite bias detection"""
        try:
            favorite_predictions = [
                p for p in predictions_data 
                if p['metadata']['favorite_defined'] and p['metadata']['max_probability'] > 45
            ]
            
            if len(favorite_predictions) < 10:
                return None
            
            favorite_accuracies = []
            expected_accuracies = []
            
            for pred in favorite_predictions:
                is_favorite_win = (
                    (pred['predictions']['home_prob'] == pred['metadata']['max_probability'] and pred['actual_result']['outcome'] == 'home') or
                    (pred['predictions']['away_prob'] == pred['metadata']['max_probability'] and pred['actual_result']['outcome'] == 'away')
                )
                favorite_accuracies.append(1.0 if is_favorite_win else 0.0)
                expected_accuracies.append(pred['metadata']['max_probability'] / 100.0)
            
            avg_expected_accuracy = np.mean(expected_accuracies)
            avg_actual_accuracy = np.mean(favorite_accuracies)
            
            favorite_bias = avg_expected_accuracy - avg_actual_accuracy
            
            # Statistical test
            if len(favorite_predictions) >= 20:
                t_stat, p_value = stats.ttest_1samp(expected_accuracies, avg_actual_accuracy)
            else:
                p_value = 0.5
            
            significance = self._determine_statistical_significance(p_value)
            
            if abs(favorite_bias) > self.config['bias_severity_thresholds']['low'] and significance != StatisticalSignificance.INSIGNIFICANT:
                severity = min(1.0, abs(favorite_bias) / 0.2)
                bias_type = BiasType.UNDERDOG_BIAS if favorite_bias > 0 else BiasType.FAVORITE_BIAS
                
                return BiasDetection(
                    bias_type=bias_type,
                    severity=severity,
                    evidence=f"Favorite prediction error: {abs(favorite_bias):.1%} (expected: {avg_expected_accuracy:.1%}, actual: {avg_actual_accuracy:.1%}, p={p_value:.3f})",
                    impact_on_accuracy=abs(favorite_bias) * 0.4,
                    recommendation="Review favorite identification logic and implement dynamic probability calibration",
                    confidence_interval=self._calculate_confidence_interval([expected_accuracies[i] - favorite_accuracies[i] for i in range(len(expected_accuracies))]),
                    statistical_significance=significance,
                    sample_size=len(favorite_predictions)
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting enhanced favorite bias: {e}")
            return None

    def _detect_draw_bias_enhanced(self, predictions_data: List[Dict]) -> Optional[BiasDetection]:
        """Enhanced draw bias detection"""
        try:
            draw_probs = [p['predictions']['draw_prob'] for p in predictions_data]
            actual_draws = [1.0 if p['actual_result']['outcome'] == 'draw' else 0.0 for p in predictions_data]
            
            avg_predicted_draw = np.mean(draw_probs) / 100.0
            avg_actual_draw = np.mean(actual_draws)
            
            draw_bias = avg_predicted_draw - avg_actual_draw
            
            # Statistical test
            if len(draw_probs) >= 30:
                t_stat, p_value = stats.ttest_1samp(draw_probs, avg_actual_draw * 100)
            else:
                p_value = 0.5
            
            significance = self._determine_statistical_significance(p_value)
            
            if abs(draw_bias) > self.config['bias_severity_thresholds']['medium'] and significance != StatisticalSignificance.INSIGNIFICANT:
                severity = min(1.0, abs(draw_bias) / 0.25)
                
                return BiasDetection(
                    bias_type=BiasType.DRAW_AVERSION,
                    severity=severity,
                    evidence=f"Draw prediction bias: {abs(draw_bias):.1%} (predicted: {avg_predicted_draw:.1%}, actual: {avg_actual_draw:.1%}, p={p_value:.3f})",
                    impact_on_accuracy=abs(draw_bias) * 0.6,
                    recommendation="Implement draw probability smoothing and context-aware draw prediction",
                    confidence_interval=self._calculate_confidence_interval([draw_probs[i]/100 - actual_draws[i] for i in range(len(draw_probs))]),
                    statistical_significance=significance,
                    sample_size=len(draw_probs)
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting enhanced draw bias: {e}")
            return None

    def _detect_probability_clustering_bias(self, predictions_data: List[Dict]) -> Optional[BiasDetection]:
        """Detect probability clustering around common values"""
        try:
            all_probs = []
            for p in predictions_data:
                all_probs.extend([p['predictions']['home_prob'], p['predictions']['draw_prob'], p['predictions']['away_prob']])
            
            # Check for clustering around common values
            common_values = [10, 15, 20, 25, 30, 33, 40, 50, 60, 67, 70, 75, 80, 85, 90]
            clustered_count = 0
            
            for prob in all_probs:
                if any(abs(prob - common) <= 2 for common in common_values):
                    clustered_count += 1
            
            clustering_ratio = clustered_count / len(all_probs)
            
            if clustering_ratio > 0.3:  # More than 30% clustered
                severity = min(1.0, (clustering_ratio - 0.3) / 0.4)
                
                return BiasDetection(
                    bias_type=BiasType.PROBABILITY_CLUSTERING,
                    severity=severity,
                    evidence=f"Probability clustering detected: {clustering_ratio:.1%} of probabilities near common values",
                    impact_on_accuracy=severity * 0.2,
                    recommendation="Implement probability smoothing and avoid rounding to common values",
                    confidence_interval=(clustering_ratio - 0.05, clustering_ratio + 0.05),
                    statistical_significance=StatisticalSignificance.MEDIUM,
                    sample_size=len(all_probs)
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting probability clustering: {e}")
            return None

    def _detect_recency_bias(self, predictions_data: List[Dict]) -> Optional[BiasDetection]:
        """Detect recency bias (overweighting recent events)"""
        try:
            if len(predictions_data) < 40:
                return None
            
            # Split into early and late periods
            split_point = len(predictions_data) // 2
            early_data = predictions_data[:split_point]
            late_data = predictions_data[split_point:]
            
            # Calculate accuracy in each period
            early_accuracy = np.mean([1.0 if p['performance']['is_correct'] else 0.0 for p in early_data])
            late_accuracy = np.mean([1.0 if p['performance']['is_correct'] else 0.0 for p in late_data])
            
            # Test for significant difference
            early_accuracies = [1.0 if p['performance']['is_correct'] else 0.0 for p in early_data]
            late_accuracies = [1.0 if p['performance']['is_correct'] else 0.0 for p in late_data]
            
            t_stat, p_value = stats.ttest_ind(early_accuracies, late_accuracies)
            
            significance = self._determine_statistical_significance(p_value)
            
            accuracy_change = late_accuracy - early_accuracy
            
            if abs(accuracy_change) > 0.1 and significance != StatisticalSignificance.INSIGNIFICANT:
                severity = min(1.0, abs(accuracy_change) / 0.2)
                direction = "improvement" if accuracy_change > 0 else "deterioration"
                
                return BiasDetection(
                    bias_type=BiasType.RECENCY_BIAS,
                    severity=severity,
                    evidence=f"Significant performance {direction} detected (early: {early_accuracy:.1%}, late: {late_accuracy:.1%}, p={p_value:.3f})",
                    impact_on_accuracy=abs(accuracy_change) * 0.3,
                    recommendation="Investigate model stability and implement rolling calibration",
                    confidence_interval=self._calculate_confidence_interval([late_accuracy - early_accuracy]),
                    statistical_significance=significance,
                    sample_size=len(predictions_data)
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting recency bias: {e}")
            return None

    def _determine_statistical_significance(self, p_value: float) -> StatisticalSignificance:
        """Determine statistical significance level from p-value"""
        if p_value < self.config['statistical_significance_thresholds']['high']:
            return StatisticalSignificance.HIGH
        elif p_value < self.config['statistical_significance_thresholds']['medium']:
            return StatisticalSignificance.MEDIUM
        elif p_value < self.config['statistical_significance_thresholds']['low']:
            return StatisticalSignificance.LOW
        else:
            return StatisticalSignificance.INSIGNIFICANT

    def _analyze_enhanced_performance(self, predictions_data: List[Dict]) -> PerformanceAnalysis:
        """Enhanced performance analysis with trend detection"""
        try:
            accuracies = [1.0 if p['performance']['is_correct'] else 0.0 for p in predictions_data]
            confidences = [p['predictions']['predicted_confidence'] for p in predictions_data]
            
            accuracy = np.mean(accuracies)
            accuracy_ci = self._calculate_confidence_interval(accuracies)
            expected_accuracy = np.mean(confidences) / 100.0
            calibration_gap = expected_accuracy - accuracy
            
            # Enhanced trend analysis
            trend_strength = self._calculate_trend_strength(accuracies)
            recent_trend = self._determine_recent_trend(accuracies, trend_strength)
            
            # Market efficiency analysis
            market_efficiency = np.mean([p['performance']['market_efficiency'] for p in predictions_data])
            
            # Kelly performance
            kelly_performance = np.mean([p['performance']['kelly_performance'] for p in predictions_data])
            
            # Value betting performance
            value_performance = max(0.0, accuracy - 0.52)  # Above random + margin
            
            return PerformanceAnalysis(
                accuracy=round(accuracy, 4),
                accuracy_ci=tuple(round(x, 4) for x in accuracy_ci),
                expected_accuracy=round(expected_accuracy, 4),
                calibration_gap=round(calibration_gap, 4),
                value_betting_performance=round(value_performance, 4),
                recent_trend=recent_trend,
                trend_strength=round(trend_strength, 4),
                market_efficiency=round(market_efficiency, 4),
                kelly_criterion_performance=round(kelly_performance, 4)
            )
            
        except Exception as e:
            logger.error(f"Error analyzing enhanced performance: {e}")
            return PerformanceAnalysis(0.5, (0.4, 0.6), 0.5, 0.0, 0.0, "unknown", 0.0, 0.0, 0.0)

    def _calculate_trend_strength(self, accuracies: List[float]) -> float:
        """Calculate the strength of performance trend"""
        if len(accuracies) < 10:
            return 0.0
        
        try:
            x = np.arange(len(accuracies))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, accuracies)
            return abs(r_value)  # Use absolute value for strength
        except Exception:
            return 0.0

    def _determine_recent_trend(self, accuracies: List[float], trend_strength: float) -> str:
        """Determine recent performance trend"""
        if len(accuracies) < 10:
            return "insufficient_data"
        
        # Use last 20% of data for recent trend
        recent_window = max(10, len(accuracies) // 5)
        recent_accuracies = accuracies[-recent_window:]
        earlier_accuracies = accuracies[-2*recent_window:-recent_window] if len(accuracies) >= 2*recent_window else accuracies[:recent_window]
        
        if not earlier_accuracies:
            return "stable"
        
        recent_avg = np.mean(recent_accuracies)
        earlier_avg = np.mean(earlier_accuracies)
        
        if trend_strength < 0.3:
            return "stable"
        elif recent_avg > earlier_avg + 0.05:
            return "strong_improving"
        elif recent_avg > earlier_avg + 0.02:
            return "improving"
        elif recent_avg < earlier_avg - 0.05:
            return "strong_degrading"
        elif recent_avg < earlier_avg - 0.02:
            return "degrading"
        else:
            return "stable"

    def _analyze_market_efficiency(self, predictions_data: List[Dict]) -> Dict[str, Any]:
        """Analyze market efficiency and value betting opportunities"""
        try:
            market_efficiencies = [p['performance']['market_efficiency'] for p in predictions_data if 'market_efficiency' in p['performance']]
            kelly_performances = [p['performance']['kelly_performance'] for p in predictions_data if 'kelly_performance' in p['performance']]
            
            if not market_efficiencies:
                return {
                    "avg_value_edge": 0.0,
                    "efficient_market_percentage": 0.0,
                    "profitable_market_percentage": 0.0,
                    "avg_kelly_performance": 0.0,
                    "sample_size": 0
                }
            
            avg_value_edge = np.mean(market_efficiencies)
            efficient_markets = len([e for e in market_efficiencies if e > self.config['market_efficiency_threshold']])
            efficient_percentage = efficient_markets / len(market_efficiencies)
            
            profitable_markets = len([k for k in kelly_performances if k > 0])
            profitable_percentage = profitable_markets / len(kelly_performances) if kelly_performances else 0.0
            
            avg_kelly_performance = np.mean(kelly_performances) if kelly_performances else 0.0
            
            return {
                "avg_value_edge": round(avg_value_edge, 4),
                "efficient_market_percentage": round(efficient_percentage, 4),
                "profitable_market_percentage": round(profitable_percentage, 4),
                "avg_kelly_performance": round(avg_kelly_performance, 4),
                "sample_size": len(market_efficiencies)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market efficiency: {e}")
            return {
                "avg_value_edge": 0.0,
                "efficient_market_percentage": 0.0,
                "profitable_market_percentage": 0.0,
                "avg_kelly_performance": 0.0,
                "sample_size": 0
            }

    def _assess_statistical_reliability(self, predictions_data: List[Dict]) -> Dict[str, Any]:
        """Assess statistical reliability of the analysis"""
        try:
            sample_size = len(predictions_data)
            
            # Power analysis
            if sample_size < 30:
                power = "low"
            elif sample_size < 100:
                power = "medium"
            else:
                power = "high"
            
            # Confidence level assessment
            confidence_level = self.config['confidence_level']
            
            # Data quality assessment
            accuracies = [1.0 if p['performance']['is_correct'] else 0.0 for p in predictions_data]
            data_volatility = np.std(accuracies) if accuracies else 0.0
            
            if data_volatility < 0.1:
                stability = "high"
            elif data_volatility < 0.2:
                stability = "medium"
            else:
                stability = "low"
            
            return {
                "sample_size": sample_size,
                "statistical_power": power,
                "confidence_level": confidence_level,
                "data_stability": stability,
                "data_volatility": round(data_volatility, 4),
                "recommended_sample_size": max(100, sample_size * 2)  # For higher reliability
            }
            
        except Exception as e:
            logger.error(f"Error assessing statistical reliability: {e}")
            return {
                "sample_size": len(predictions_data),
                "statistical_power": "unknown",
                "confidence_level": self.config['confidence_level'],
                "data_stability": "unknown",
                "data_volatility": 0.0,
                "recommended_sample_size": 100
            }

    def _get_analysis_period(self, predictions_data: List[Dict]) -> str:
        """Get analysis period description"""
        if not predictions_data:
            return "No data"
        
        timestamps = [datetime.fromisoformat(p['timestamp'].replace('Z', '+00:00')) for p in predictions_data]
        start_date = min(timestamps).strftime('%Y-%m-%d')
        end_date = max(timestamps).strftime('%Y-%m-%d')
        days_span = (max(timestamps) - min(timestamps)).days
        
        return f"{start_date} to {end_date} ({len(predictions_data)} matches over {days_span} days)"

    def _generate_enhanced_recommendations(self, calibration: CalibrationMetrics, 
                                         biases: List[BiasDetection],
                                         market_efficiency: Dict[str, Any]) -> List[str]:
        """Generate enhanced professional recommendations"""
        recommendations = []
        
        # Calibration-based recommendations
        if calibration.status == CalibrationStatus.CRITICAL:
            recommendations.append("🚨 CRITICAL: Immediate model recalibration required - significant performance degradation detected")
        elif calibration.status == CalibrationStatus.POOR:
            recommendations.append("⚠️ HIGH PRIORITY: Model calibration is poor - review feature engineering and training data")
        elif calibration.status == CalibrationStatus.DEGRADING:
            recommendations.append("🔔 MEDIUM PRIORITY: Model calibration is degrading - implement rolling calibration")
        
        if calibration.brier_score > self.config['brier_score_thresholds']['acceptable'][1]:
            recommendations.append("🎯 Focus on improving probability calibration through ensemble methods or Bayesian smoothing")
        
        # Bias-based recommendations with priority
        critical_biases = [b for b in biases if b.severity > self.config['bias_severity_thresholds']['high'] and b.statistical_significance in [StatisticalSignificance.HIGH, StatisticalSignificance.MEDIUM]]
        significant_biases = [b for b in biases if b.severity > self.config['bias_severity_thresholds']['medium'] and b.statistical_significance != StatisticalSignificance.INSIGNIFICANT]
        
        for bias in critical_biases:
            recommendations.append(f"🔴 CRITICAL BIAS: {bias.recommendation} (p < {self.config['statistical_significance_thresholds']['medium']})")
        
        for bias in significant_biases:
            recommendations.append(f"🟡 SIGNIFICANT BIAS: {bias.recommendation}")
        
        # Market efficiency recommendations
        if market_efficiency['avg_value_edge'] > 0.05:
            recommendations.append(f"💰 STRONG VALUE: Model shows {market_efficiency['avg_value_edge']:.1%} average edge - consider increasing bet sizing")
        elif market_efficiency['avg_value_edge'] > 0.02:
            recommendations.append(f"💡 MODERATE VALUE: Model shows {market_efficiency['avg_value_edge']:.1%} average edge - maintain current strategy")
        elif market_efficiency['avg_value_edge'] < -0.02:
            recommendations.append("🔻 NEGATIVE EDGE: Model underperforming market - review prediction methodology")
        
        if market_efficiency['profitable_market_percentage'] > 0.6:
            recommendations.append(f"📈 EXCELLENT PROFITABILITY: {market_efficiency['profitable_market_percentage']:.1%} of predictions profitable")
        
        # Bayesian insights
        accuracy_posterior = self.calibration_state['bayesian_posteriors']['accuracy']
        if accuracy_posterior['std'] < 0.05:
            recommendations.append(f"📊 HIGH CONFIDENCE: Model accuracy estimated at {accuracy_posterior['mean']:.1%} ± {accuracy_posterior['std']:.1%}")
        
        if not recommendations:
            recommendations.append("✅ OPTIMAL: Model performance is within acceptable ranges across all metrics - continue monitoring")
        
        # Add advanced metrics insights
        if self.calibration_state['advanced_metrics']['volatility_index'] > 0.15:
            recommendations.append("⚡ HIGH VOLATILITY: Model performance is unstable - consider implementing volatility smoothing")
        
        if self.calibration_state['advanced_metrics']['predictability_index'] > 0.7:
            recommendations.append("🎯 HIGH PREDICTABILITY: Model shows consistent performance patterns - suitable for automated trading")
        
        return recommendations

    def get_calibration_feedback(self) -> Dict[str, Any]:
        """Get enhanced calibration feedback for prediction engine"""
        return {
            'calibration_factors': self.calibration_state['calibration_factors'],
            'historical_accuracy': self.calibration_state['historical_accuracy'],
            'bayesian_posteriors': self.calibration_state['bayesian_posteriors'],
            'market_efficiency': self.calibration_state['market_efficiency_metrics'],
            'advanced_metrics': self.calibration_state['advanced_metrics'],
            'last_update': self.calibration_state['last_calibration_update'],
            'performance_summary': {
                'total_samples': len(self.calibration_state['performance_history']),
                'recent_accuracy': self._get_recent_accuracy(),
                'recent_accuracy_ci': self._get_recent_accuracy_ci(),
                'stability_score': self.calibration_state['advanced_metrics']['stability_score']
            }
        }

    def _get_recent_accuracy(self) -> float:
        """Get recent accuracy for quick assessment"""
        try:
            recent_performance = self.calibration_state['performance_history'][-self.config['recent_performance_window']:]
            if not recent_performance:
                return 0.5
            return np.mean([p['is_correct'] for p in recent_performance])
        except Exception:
            return 0.5

    def _get_recent_accuracy_ci(self) -> Tuple[float, float]:
        """Get recent accuracy confidence interval"""
        try:
            recent_performance = self.calibration_state['performance_history'][-self.config['recent_performance_window']:]
            if not recent_performance:
                return (0.5, 0.5)
            accuracies = [p['is_correct'] for p in recent_performance]
            return self._calculate_confidence_interval(accuracies)
        except Exception:
            return (0.5, 0.5)

    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard"""
        analysis = self.analyze_comprehensive_bias()
        
        dashboard = {
            'overview': {
                'status': analysis['status'],
                'samples': analysis['samples'],
                'analysis_period': analysis['analysis_period'],
                'overall_health': self._calculate_overall_health(analysis)
            },
            'key_metrics': {
                'accuracy': analysis['performance_analysis'].accuracy,
                'brier_score': analysis['calibration_metrics'].brier_score,
                'calibration_status': analysis['calibration_metrics'].status.value,
                'market_efficiency': analysis['market_efficiency_analysis']['avg_value_edge'],
                'trend': analysis['performance_analysis'].recent_trend
            },
            'alerts': {
                'critical_biases': len([b for b in analysis['detected_biases'] if b.severity > 0.7]),
                'performance_issues': 1 if analysis['calibration_metrics'].status in [CalibrationStatus.POOR, CalibrationStatus.CRITICAL] else 0,
                'statistical_reliability': analysis['statistical_reliability']['statistical_power']
            },
            'recommendations': analysis['recommendations'][:5]  # Top 5 recommendations
        }
        
        return dashboard

    def _calculate_overall_health(self, analysis: Dict[str, Any]) -> str:
        """Calculate overall model health score"""
        try:
            score = 0.0
            max_score = 0.0
            
            # Calibration health (30%)
            cal_status = analysis['calibration_metrics'].status
            if cal_status == CalibrationStatus.OPTIMAL:
                score += 30
            elif cal_status == CalibrationStatus.ACCEPTABLE:
                score += 25
            elif cal_status == CalibrationStatus.DEGRADING:
                score += 15
            elif cal_status == CalibrationStatus.POOR:
                score += 5
            max_score += 30
            
            # Accuracy health (30%)
            accuracy = analysis['performance_analysis'].accuracy
            if accuracy > 0.60:
                score += 30
            elif accuracy > 0.55:
                score += 25
            elif accuracy > 0.52:
                score += 15
            elif accuracy > 0.50:
                score += 10
            else:
                score += 5
            max_score += 30
            
            # Bias health (20%)
            bias_penalty = sum(b.severity * 10 for b in analysis['detected_biases'] if b.statistical_significance != StatisticalSignificance.INSIGNIFICANT)
            score += max(0, 20 - bias_penalty)
            max_score += 20
            
            # Market efficiency health (20%)
            market_eff = analysis['market_efficiency_analysis']['avg_value_edge']
            if market_eff > 0.03:
                score += 20
            elif market_eff > 0.01:
                score += 15
            elif market_eff > -0.01:
                score += 10
            else:
                score += 5
            max_score += 20
            
            health_percentage = (score / max_score) * 100
            
            if health_percentage >= 80:
                return "excellent"
            elif health_percentage >= 65:
                return "good"
            elif health_percentage >= 50:
                return "fair"
            else:
                return "poor"
                
        except Exception:
            return "unknown"

# Enhanced compatibility wrapper
class BiasMonitor:
    """Enhanced compatibility wrapper for existing code"""
    
    def __init__(self, log_file: str = "bias_log.json"):
        self.professional_monitor = InstitutionalBiasMonitor(log_file)
    
    def log_prediction(self, match_id: str, predictions: Dict, actual_result: str = None, market_odds: Dict = None):
        """Enhanced compatibility method"""
        if actual_result and isinstance(actual_result, str):
            try:
                if ':' in actual_result:
                    home_goals, away_goals = map(int, actual_result.split(':'))
                    actual_result_dict = {
                        'home_goals': home_goals,
                        'away_goals': away_goals
                    }
                    self.professional_monitor.log_prediction_with_result(
                        match_id, predictions, actual_result_dict, 
                        predictions.get('confidence_score', 50),
                        market_odds
                    )
            except Exception as e:
                logger.warning(f"Could not parse actual result: {e}")
    
    def analyze_bias(self, min_samples: int = 50) -> Dict[str, float]:
        """Enhanced compatibility method"""
        analysis = self.professional_monitor.analyze_comprehensive_bias()
        
        if analysis['status'] != 'COMPLETE':
            return {"status": analysis['status'], "samples": analysis['samples']}
        
        return {
            "accuracy": analysis['performance_analysis'].accuracy,
            "brier_score": analysis['calibration_metrics'].brier_score,
            "calibration_status": analysis['calibration_metrics'].status.value,
            "market_efficiency": analysis['market_efficiency_analysis']['avg_value_edge'],
            "detected_biases": len(analysis['detected_biases']),
            "samples": analysis['samples'],
            "health_score": self.professional_monitor._calculate_overall_health(analysis),
            "status": "Enhanced analysis available - use analyze_comprehensive_bias() for full details"
        }

# Utility functions
def create_bias_monitor(log_file: str = "professional_bias_log.json") -> InstitutionalBiasMonitor:
    """Create an enhanced professional bias monitor instance"""
    return InstitutionalBiasMonitor(log_file)

def calculate_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Utility function to calculate confidence interval"""
    if len(data) < 2:
        return (np.mean(data), np.mean(data))
    
    try:
        return stats.t.interval(confidence, len(data)-1, loc=np.mean(data), scale=stats.sem(data))
    except Exception:
        return (np.mean(data), np.mean(data))

# Example usage and testing
if __name__ == "__main__":
    # Example of using the enhanced monitor
    monitor = create_bias_monitor()
    
    # Example prediction log
    example_prediction = {
        'predictions': {
            '1X2': {'Home Win': 45.2, 'Draw': 28.1, 'Away Win': 26.7}
        },
        'confidence_score': 70.0
    }
    
    example_result = {
        'home_goals': 2,
        'away_goals': 1
    }
    
    example_odds = {
        'home': 2.10,
        'draw': 3.40,
        'away': 3.60
    }
    
    # Log a prediction
    monitor.log_prediction_with_result(
        "MATCH_001",
        example_prediction,
        example_result,
        70.0,
        example_odds
    )
    
    # Get analysis
    analysis = monitor.analyze_comprehensive_bias()
    print("Enhanced Bias Analysis:", json.dumps(analysis, indent=2, default=str))
    
    # Get dashboard
    dashboard = monitor.get_performance_dashboard()
    print("Performance Dashboard:", json.dumps(dashboard, indent=2))
