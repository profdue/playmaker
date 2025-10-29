# model_monitor.py
"""Institutional-grade model monitoring with calibration feedback"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
from scipy import stats
import json
import os
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class ModelHealth(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    DEGRADING = "degrading"
    POOR = "poor"
    CRITICAL = "critical"

@dataclass
class PerformanceAlert:
    timestamp: datetime
    alert_level: AlertLevel
    metric: str
    current_value: float
    threshold: float
    deviation: float
    description: str
    recommendation: str

@dataclass
class ModelDiagnostics:
    model_health: ModelHealth
    calibration_score: float
    feature_drift: float
    prediction_drift: float
    performance_trend: str
    data_quality_score: float
    confidence_stability: float
    last_retraining: datetime
    alerts: List[PerformanceAlert]
    accuracy_metrics: Dict[str, float]  # NEW: Added accuracy tracking

@dataclass
class CalibrationUpdate:
    """Data structure for calibration feedback"""
    timestamp: datetime
    calibration_factors: Dict[str, float]
    performance_metrics: Dict[str, float]
    recommendations: List[str]

class InstitutionalModelMonitor:
    """
    Institutional-grade model monitoring with calibration feedback
    """
    
    def __init__(self, model_name: str = "FootballPredictionEngine"):
        self.model_name = model_name
        self.performance_history: List[Dict] = []
        self.prediction_history: List[Dict] = []
        self.feature_distributions: Dict[str, List] = {}
        self.alert_history: List[PerformanceAlert] = []
        self.calibration_updates: List[CalibrationUpdate] = []
        
        # Professional monitoring configuration
        self.config = {
            'performance_window': 100,
            'drift_threshold': 0.15,
            'calibration_threshold': 0.10,
            'confidence_decay_threshold': 0.20,
            'retraining_trigger': 0.25,
            'alert_cooldown': timedelta(hours=1),
            'min_calibration_samples': 20,  # NEW: Minimum samples for calibration
            'calibration_update_frequency': 50  # NEW: Update every 50 predictions
        }
        
        # Initialize professional baselines
        self.baseline_metrics = self._initialize_professional_baselines()
        self.last_alert_time = {}
        self.calibration_file = "model_calibration.json"
        
        logger.info(f"Initialized Professional ModelMonitor for {model_name}")

    def _initialize_professional_baselines(self) -> Dict[str, Any]:
        """Initialize professional baseline metrics with calibration"""
        return {
            'accuracy_baseline': 0.55,
            'calibration_baseline': 0.85,
            'confidence_baseline': 0.70,
            'feature_distributions': {},
            'prediction_distribution': [],
            'performance_trend': 'stable',
            'calibration_factors': {  # NEW: Calibration factors for prediction engine
                'xg_home_calibration': 1.0,
                'xg_away_calibration': 1.0,
                'confidence_calibration': 1.0,
                'pattern_success_rates': {}
            },
            'historical_accuracy': 0.55
        }

    def record_prediction(self, prediction_data: Dict, actual_result: Dict = None):
        """Record prediction with professional accuracy tracking"""
        try:
            timestamp = datetime.now()
            
            # Calculate accuracy if actual result provided
            is_correct = None
            accuracy_metrics = {}
            
            if actual_result:
                is_correct = self._calculate_prediction_accuracy(prediction_data, actual_result)
                accuracy_metrics = self._calculate_detailed_accuracy(prediction_data, actual_result)
            
            prediction_record = {
                'timestamp': timestamp,
                'prediction': prediction_data,
                'actual_result': actual_result,
                'is_correct': is_correct,
                'accuracy_metrics': accuracy_metrics,
                'data_quality': prediction_data.get('data_quality_score', 50),  # FIXED: Field name
                'confidence': prediction_data.get('confidence_score', 50),
                'model_type': prediction_data.get('model_type', 'unknown')
            }
            
            self.prediction_history.append(prediction_record)
            
            # Update feature distributions for drift detection
            self._update_feature_distributions(prediction_data)
            
            # Check if calibration update is needed
            if actual_result and len(self.prediction_history) % self.config['calibration_update_frequency'] == 0:
                self._update_calibration_factors()
            
            # Maintain professional history window
            if len(self.prediction_history) > self.config['performance_window'] * 3:
                self.prediction_history = self.prediction_history[-self.config['performance_window'] * 3:]
                
            logger.debug(f"Recorded prediction with accuracy tracking: {is_correct}")
            
        except Exception as e:
            logger.error(f"Error recording prediction: {e}")

    def _calculate_prediction_accuracy(self, prediction: Dict, actual_result: Dict) -> bool:
        """Professional accuracy calculation for football predictions"""
        try:
            if not actual_result or 'home_goals' not in actual_result or 'away_goals' not in actual_result:
                return None
            
            home_goals = actual_result['home_goals']
            away_goals = actual_result['away_goals']
            
            # Get predicted probabilities
            pred_1x2 = prediction['predictions']['1X2']
            home_prob = pred_1x2.get('Home Win', 0)
            draw_prob = pred_1x2.get('Draw', 0) 
            away_prob = pred_1x2.get('Away Win', 0)
            
            # Determine predicted outcome (highest probability)
            if home_prob >= draw_prob and home_prob >= away_prob:
                predicted_outcome = 'home'
            elif away_prob >= home_prob and away_prob >= draw_prob:
                predicted_outcome = 'away'
            else:
                predicted_outcome = 'draw'
            
            # Determine actual outcome
            if home_goals > away_goals:
                actual_outcome = 'home'
            elif away_goals > home_goals:
                actual_outcome = 'away'
            else:
                actual_outcome = 'draw'
            
            return predicted_outcome == actual_outcome
            
        except Exception as e:
            logger.error(f"Accuracy calculation error: {e}")
            return None

    def _calculate_detailed_accuracy(self, prediction: Dict, actual_result: Dict) -> Dict[str, float]:
        """Calculate detailed accuracy metrics for professional analysis"""
        try:
            metrics = {}
            
            # 1X2 accuracy
            metrics['1x2_correct'] = self._calculate_prediction_accuracy(prediction, actual_result)
            
            # BTTS accuracy
            actual_btts = actual_result['home_goals'] > 0 and actual_result['away_goals'] > 0
            predicted_btts_prob = prediction['predictions']['BTTS']['Yes']
            predicted_btts = predicted_btts_prob > 50
            metrics['btts_correct'] = actual_btts == predicted_btts
            
            # Over/Under accuracy
            total_goals = actual_result['home_goals'] + actual_result['away_goals']
            predicted_over_prob = prediction['predictions']['Over/Under']['Over 2.5']
            predicted_over = predicted_over_prob > 50
            actual_over = total_goals > 2.5
            metrics['over_under_correct'] = actual_over == predicted_over
            
            # Probability calibration (Brier score component)
            if metrics['1x2_correct'] is not None:
                actual_outcome = 'home' if actual_result['home_goals'] > actual_result['away_goals'] else \
                               'away' if actual_result['away_goals'] > actual_result['home_goals'] else 'draw'
                
                if actual_outcome == 'home':
                    actual_prob = 1.0
                    predicted_prob = prediction['predictions']['1X2']['Home Win'] / 100.0
                elif actual_outcome == 'away':
                    actual_prob = 1.0  
                    predicted_prob = prediction['predictions']['1X2']['Away Win'] / 100.0
                else:
                    actual_prob = 1.0
                    predicted_prob = prediction['predictions']['1X2']['Draw'] / 100.0
                
                metrics['brier_score'] = (predicted_prob - actual_prob) ** 2
            
            return metrics
            
        except Exception as e:
            logger.error(f"Detailed accuracy calculation error: {e}")
            return {}

    def _update_calibration_factors(self):
        """Update calibration factors based on recent performance"""
        try:
            if len(self.prediction_history) < self.config['min_calibration_samples']:
                return
            
            # Get recent predictions with actual results
            recent_predictions = [
                p for p in self.prediction_history[-self.config['min_calibration_samples']:]
                if p['actual_result'] is not None and p['is_correct'] is not None
            ]
            
            if len(recent_predictions) < 10:
                return
            
            # Calculate current performance metrics
            accuracy = np.mean([p['is_correct'] for p in recent_predictions])
            avg_confidence = np.mean([p['confidence'] for p in recent_predictions]) / 100.0
            
            # Calculate calibration factors
            calibration_ratio = accuracy / avg_confidence if avg_confidence > 0 else 1.0
            confidence_calibration = max(0.7, min(1.3, calibration_ratio))
            
            # Update XG calibration based on actual goals vs predicted
            xg_calibrations = self._calculate_xg_calibration(recent_predictions)
            
            # Update pattern success rates
            pattern_success = self._calculate_pattern_success_rates(recent_predictions)
            
            # Create calibration update
            calibration_update = CalibrationUpdate(
                timestamp=datetime.now(),
                calibration_factors={
                    'xg_home_calibration': xg_calibrations['home'],
                    'xg_away_calibration': xg_calibrations['away'],
                    'confidence_calibration': confidence_calibration,
                    'pattern_success_rates': pattern_success
                },
                performance_metrics={
                    'current_accuracy': accuracy,
                    'current_confidence': avg_confidence,
                    'calibration_ratio': calibration_ratio,
                    'sample_size': len(recent_predictions)
                },
                recommendations=self._generate_calibration_recommendations(accuracy, calibration_ratio)
            )
            
            self.calibration_updates.append(calibration_update)
            self.baseline_metrics['calibration_factors'] = calibration_update.calibration_factors
            self.baseline_metrics['historical_accuracy'] = accuracy
            
            # Save calibration data
            self._save_calibration_data()
            
            logger.info(f"Calibration updated: Accuracy={accuracy:.3f}, Confidence Calibration={confidence_calibration:.3f}")
            
        except Exception as e:
            logger.error(f"Calibration update error: {e}")

    def _calculate_xg_calibration(self, recent_predictions: List[Dict]) -> Dict[str, float]:
        """Calculate XG calibration factors based on actual goals"""
        try:
            home_goal_ratios = []
            away_goal_ratios = []
            
            for pred in recent_predictions:
                prediction_data = pred['prediction']
                actual_result = pred['actual_result']
                
                if ('goal_expectancy' in prediction_data and 
                    actual_result and 'home_goals' in actual_result):
                    
                    predicted_home_xg = prediction_data['goal_expectancy'].get('home_xg', 1.4)
                    predicted_away_xg = prediction_data['goal_expectancy'].get('away_xg', 1.1)
                    
                    actual_home_goals = actual_result['home_goals']
                    actual_away_goals = actual_result['away_goals']
                    
                    if predicted_home_xg > 0:
                        home_ratio = actual_home_goals / predicted_home_xg
                        home_goal_ratios.append(home_ratio)
                    
                    if predicted_away_xg > 0:
                        away_ratio = actual_away_goals / predicted_away_xg
                        away_goal_ratios.append(away_ratio)
            
            home_calibration = np.mean(home_goal_ratios) if home_goal_ratios else 1.0
            away_calibration = np.mean(away_goal_ratios) if away_goal_ratios else 1.0
            
            # Apply reasonable bounds
            return {
                'home': max(0.7, min(1.3, home_calibration)),
                'away': max(0.7, min(1.3, away_calibration))
            }
            
        except Exception as e:
            logger.error(f"XG calibration calculation error: {e}")
            return {'home': 1.0, 'away': 1.0}

    def _calculate_pattern_success_rates(self, recent_predictions: List[Dict]) -> Dict[str, float]:
        """Calculate pattern success rates for calibration"""
        try:
            pattern_success = {}
            pattern_counts = {}
            
            for pred in recent_predictions:
                prediction_data = pred['prediction']
                is_correct = pred['is_correct']
                
                if ('pattern_intelligence' in prediction_data and 
                    is_correct is not None):
                    
                    patterns = prediction_data['pattern_intelligence'].get('patterns_detected', [])
                    
                    for pattern in patterns:
                        pattern_type = pattern.get('type', 'unknown')
                        pattern_direction = pattern.get('direction', 'unknown')
                        
                        # Check if pattern prediction was correct
                        # This is simplified - in production you'd have more sophisticated logic
                        pattern_correct = is_correct  # Simplified assumption
                        
                        if pattern_type not in pattern_success:
                            pattern_success[pattern_type] = 0.0
                            pattern_counts[pattern_type] = 0
                        
                        pattern_success[pattern_type] += 1 if pattern_correct else 0
                        pattern_counts[pattern_type] += 1
            
            # Calculate success rates
            for pattern_type in pattern_success:
                if pattern_counts[pattern_type] > 0:
                    pattern_success[pattern_type] = pattern_success[pattern_type] / pattern_counts[pattern_type]
            
            return pattern_success
            
        except Exception as e:
            logger.error(f"Pattern success calculation error: {e}")
            return {}

    def _generate_calibration_recommendations(self, accuracy: float, calibration_ratio: float) -> List[str]:
        """Generate professional calibration recommendations"""
        recommendations = []
        
        if calibration_ratio < 0.9:
            recommendations.append("Model is overconfident - consider reducing confidence scores")
        elif calibration_ratio > 1.1:
            recommendations.append("Model is underconfident - consider increasing confidence scores")
        
        if accuracy < 0.5:
            recommendations.append("CRITICAL: Model accuracy below random - immediate review required")
        elif accuracy < 0.55:
            recommendations.append("WARNING: Model accuracy below expected baseline - investigate features")
        
        if not recommendations:
            recommendations.append("Model calibration is within acceptable ranges")
        
        return recommendations

    def _save_calibration_data(self):
        """Save calibration data for persistence"""
        try:
            calibration_data = {
                'baseline_metrics': self.baseline_metrics,
                'last_update': datetime.now().isoformat(),
                'performance_history': [
                    {
                        'timestamp': p['timestamp'].isoformat(),
                        'is_correct': p['is_correct'],
                        'confidence': p['confidence'],
                        'data_quality': p['data_quality']
                    }
                    for p in self.prediction_history[-100:]  # Last 100 predictions
                ]
            }
            
            with open(self.calibration_file, 'w') as f:
                json.dump(calibration_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving calibration data: {e}")

    def load_calibration_data(self):
        """Load calibration data from file"""
        try:
            if os.path.exists(self.calibration_file):
                with open(self.calibration_file, 'r') as f:
                    calibration_data = json.load(f)
                
                self.baseline_metrics = calibration_data.get('baseline_metrics', self.baseline_metrics)
                logger.info("Loaded existing calibration data")
                
        except Exception as e:
            logger.error(f"Error loading calibration data: {e}")

    def get_calibration_feedback(self) -> Dict[str, Any]:
        """Get current calibration factors for prediction engine"""
        try:
            if not self.calibration_updates:
                return self.baseline_metrics['calibration_factors']
            
            latest_calibration = self.calibration_updates[-1]
            
            return {
                'calibration_factors': latest_calibration.calibration_factors,
                'performance_metrics': latest_calibration.performance_metrics,
                'recommendations': latest_calibration.recommendations,
                'last_update': latest_calibration.timestamp.isoformat(),
                'historical_accuracy': self.baseline_metrics['historical_accuracy']
            }
            
        except Exception as e:
            logger.error(f"Error getting calibration feedback: {e}")
            return self.baseline_metrics['calibration_factors']

    # ENHANCED HEALTH CHECK WITH CALIBRATION INTEGRATION
    def run_health_check(self) -> ModelDiagnostics:
        """Run comprehensive health check with calibration integration"""
        try:
            # Calculate core metrics
            calibration_score = self._calculate_calibration_score()
            feature_drift = self._calculate_feature_drift()
            prediction_drift = self._calculate_prediction_drift()
            performance_trend = self._analyze_performance_trend()
            data_quality_score = self._calculate_data_quality_trend()
            confidence_stability = self._calculate_confidence_stability()
            
            # NEW: Calculate accuracy metrics
            accuracy_metrics = self._calculate_accuracy_metrics()
            
            # Generate alerts
            alerts = self._generate_health_alerts(
                calibration_score, feature_drift, prediction_drift,
                performance_trend, data_quality_score, confidence_stability,
                accuracy_metrics  # NEW: Include accuracy in alerts
            )
            
            # Assess overall health
            model_health = self._assess_overall_health(
                calibration_score, feature_drift, prediction_drift,
                len(alerts), accuracy_metrics  # NEW: Include accuracy
            )
            
            return ModelDiagnostics(
                model_health=model_health,
                calibration_score=calibration_score,
                feature_drift=feature_drift,
                prediction_drift=prediction_drift,
                performance_trend=performance_trend,
                data_quality_score=data_quality_score,
                confidence_stability=confidence_stability,
                last_retraining=self._get_last_retraining_date(),
                alerts=alerts,
                accuracy_metrics=accuracy_metrics  # NEW: Added accuracy metrics
            )
            
        except Exception as e:
            logger.error(f"Error running health check: {e}")
            return self._generate_error_diagnostics()

    def _calculate_accuracy_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive accuracy metrics"""
        try:
            predictions_with_results = [
                p for p in self.prediction_history 
                if p['actual_result'] is not None and p['is_correct'] is not None
            ]
            
            if not predictions_with_results:
                return {
                    'overall_accuracy': 0.5,
                    'recent_accuracy': 0.5,
                    'accuracy_trend': 0.0,
                    'brier_score': 0.25,
                    'sample_size': 0
                }
            
            # Overall accuracy
            overall_accuracy = np.mean([p['is_correct'] for p in predictions_with_results])
            
            # Recent accuracy (last 20 predictions)
            recent_predictions = predictions_with_results[-20:]
            recent_accuracy = np.mean([p['is_correct'] for p in recent_predictions]) if recent_predictions else overall_accuracy
            
            # Accuracy trend
            if len(predictions_with_results) >= 10:
                recent_trend = recent_accuracy - overall_accuracy
            else:
                recent_trend = 0.0
            
            # Brier score (probability calibration)
            brier_scores = [p['accuracy_metrics'].get('brier_score', 0.25) for p in predictions_with_results if 'brier_score' in p['accuracy_metrics']]
            avg_brier_score = np.mean(brier_scores) if brier_scores else 0.25
            
            return {
                'overall_accuracy': overall_accuracy,
                'recent_accuracy': recent_accuracy,
                'accuracy_trend': recent_trend,
                'brier_score': avg_brier_score,
                'sample_size': len(predictions_with_results)
            }
            
        except Exception as e:
            logger.error(f"Accuracy metrics calculation error: {e}")
            return {
                'overall_accuracy': 0.5,
                'recent_accuracy': 0.5,
                'accuracy_trend': 0.0,
                'brier_score': 0.25,
                'sample_size': 0
            }

    def _assess_overall_health(self, calibration_score: float, feature_drift: float,
                             prediction_drift: float, alert_count: int,
                             accuracy_metrics: Dict[str, float]) -> ModelHealth:
        """Enhanced health assessment with accuracy integration"""
        try:
            health_score = 0.0
            weights = {
                'calibration': 0.25,
                'feature_drift': 0.20,
                'prediction_drift': 0.20,
                'accuracy': 0.25,  # NEW: Accuracy weight
                'alerts': 0.10
            }
            
            # Calibration component
            calibration_component = calibration_score * weights['calibration']
            
            # Drift components
            feature_drift_component = (1.0 - min(1.0, feature_drift / 0.3)) * weights['feature_drift']
            prediction_drift_component = (1.0 - min(1.0, prediction_drift / 0.3)) * weights['prediction_drift']
            
            # NEW: Accuracy component
            accuracy = accuracy_metrics['overall_accuracy']
            accuracy_component = accuracy * weights['accuracy']
            
            # Alert component
            alert_penalty = min(1.0, alert_count / 5.0) * weights['alerts']
            alert_component = (1.0 - alert_penalty) * weights['alerts']
            
            health_score = (calibration_component + feature_drift_component + 
                          prediction_drift_component + accuracy_component + alert_component)
            
            # Enhanced health thresholds
            if health_score >= 0.85 and accuracy >= 0.55:
                return ModelHealth.EXCELLENT
            elif health_score >= 0.75 and accuracy >= 0.52:
                return ModelHealth.GOOD
            elif health_score >= 0.65:
                return ModelHealth.DEGRADING
            elif health_score >= 0.55:
                return ModelHealth.POOR
            else:
                return ModelHealth.CRITICAL
                
        except Exception:
            return ModelHealth.POOR

    # Keep existing methods but ensure they use the enhanced features
    def _calculate_calibration_score(self) -> float:
        """Enhanced calibration score with professional thresholds"""
        try:
            accuracy_metrics = self._calculate_accuracy_metrics()
            
            if accuracy_metrics['sample_size'] < 10:
                return 0.8
            
            # Use Brier score for calibration assessment
            brier_score = accuracy_metrics['brier_score']
            
            # Convert Brier score to calibration score (lower Brier = better calibration)
            # Perfect calibration = 0.0, Worst = 0.25 for binary, adjusted for 3-way
            calibration_score = 1.0 - (brier_score / 0.3)  # Adjusted for 3-way classification
            
            return max(0.0, min(1.0, calibration_score))
            
        except Exception:
            return 0.5

    def generate_professional_report(self) -> Dict[str, Any]:
        """Generate professional report with calibration insights"""
        try:
            diagnostics = self.run_health_check()
            calibration_feedback = self.get_calibration_feedback()
            retraining_check = self.check_retraining_need()
            
            return {
                "model_info": {
                    "model_name": self.model_name,
                    "current_health": diagnostics.model_health.value,
                    "monitoring_period": f"Last {len(self.prediction_history)} predictions",
                    "last_calibration_update": calibration_feedback.get('last_update', 'Never')
                },
                "performance_metrics": {
                    "overall_accuracy": f"{diagnostics.accuracy_metrics['overall_accuracy']:.3f}",
                    "recent_accuracy": f"{diagnostics.accuracy_metrics['recent_accuracy']:.3f}",
                    "calibration_score": f"{diagnostics.calibration_score:.3f}",
                    "brier_score": f"{diagnostics.accuracy_metrics['brier_score']:.3f}",
                    "feature_drift": f"{diagnostics.feature_drift:.3f}",
                    "prediction_drift": f"{diagnostics.prediction_drift:.3f}",
                    "performance_trend": diagnostics.performance_trend
                },
                "calibration_insights": calibration_feedback,
                "retraining_analysis": retraining_check,
                "recommendations": self._generate_professional_recommendations(diagnostics, calibration_feedback)
            }
            
        except Exception as e:
            logger.error(f"Error generating professional report: {e}")
            return {"error": str(e)}

    def _generate_professional_recommendations(self, diagnostics: ModelDiagnostics, 
                                             calibration: Dict) -> List[str]:
        """Generate professional recommendations based on comprehensive analysis"""
        recommendations = []
        
        # Accuracy-based recommendations
        accuracy = diagnostics.accuracy_metrics['overall_accuracy']
        if accuracy < 0.5:
            recommendations.append("🚨 CRITICAL: Model accuracy below random chance - immediate intervention required")
        elif accuracy < 0.52:
            recommendations.append("📉 WARNING: Model accuracy below professional threshold - review feature engineering")
        
        # Calibration-based recommendations
        calibration_ratio = calibration.get('performance_metrics', {}).get('calibration_ratio', 1.0)
        if calibration_ratio < 0.85:
            recommendations.append("🎯 CALIBRATION: Model significantly overconfident - apply confidence reduction")
        elif calibration_ratio > 1.15:
            recommendations.append("🎯 CALIBRATION: Model underconfident - consider confidence boost")
        
        # Health-based recommendations
        if diagnostics.model_health == ModelHealth.CRITICAL:
            recommendations.append("💀 CRITICAL HEALTH: Model requires complete retraining and validation")
        elif diagnostics.model_health == ModelHealth.DEGRADING:
            recommendations.append("⚠️ HEALTH DEGRADING: Schedule retraining and feature review")
        
        if not recommendations:
            recommendations.append("✅ MODEL PERFORMING WITHIN EXPECTED PARAMETERS: Continue monitoring")
        
        return recommendations

    # Maintain existing utility methods with minor fixes
    def _extract_monitoring_features(self, prediction_data: Dict) -> Dict[str, float]:
        """Enhanced feature extraction with professional metrics"""
        try:
            features = {}
            
            # Core prediction features
            features['confidence_score'] = prediction_data.get('confidence_score', 50)
            features['data_quality'] = prediction_data.get('data_quality_score', 50)  # FIXED: Field name
            
            # Enhanced prediction distribution features
            if 'predictions' in prediction_data:
                preds = prediction_data['predictions']
                if '1X2' in preds:
                    home_prob = preds['1X2'].get('Home Win', 33.3)
                    draw_prob = preds['1X2'].get('Draw', 33.3)
                    away_prob = preds['1X2'].get('Away Win', 33.3)
                    
                    features['max_probability'] = max(home_prob, draw_prob, away_prob)
                    features['min_probability'] = min(home_prob, draw_prob, away_prob)
                    features['probability_range'] = features['max_probability'] - features['min_probability']
                    features['probability_entropy'] = self._calculate_entropy([home_prob, draw_prob, away_prob])
            
            # Enhanced uncertainty features
            if 'uncertainty' in prediction_data:
                uncertainty = prediction_data['uncertainty']
                features['uncertainty_68_range'] = uncertainty.get('home_win_68_interval', [40, 60])[1] - uncertainty.get('home_win_68_interval', [40, 60])[0]
                features['uncertainty_95_range'] = uncertainty.get('home_win_95_interval', [30, 70])[1] - uncertainty.get('home_win_95_interval', [30, 70])[0]
                features['standard_error'] = uncertainty.get('standard_error', 10.0)
            
            # Pattern intelligence features
            if 'pattern_intelligence' in prediction_data:
                patterns = prediction_data['pattern_intelligence']
                features['pattern_count'] = patterns.get('pattern_count', 0)
                features['pattern_influence'] = patterns.get('pattern_influence', 0)
            
            return features
            
        except Exception as e:
            logger.error(f"Enhanced feature extraction error: {e}")
            return {}

    def _generate_error_diagnostics(self) -> ModelDiagnostics:
        """Enhanced error diagnostics"""
        return ModelDiagnostics(
            model_health=ModelHealth.CRITICAL,
            calibration_score=0.0,
            feature_drift=0.0,
            prediction_drift=0.0,
            performance_trend="unknown",
            data_quality_score=0.0,
            confidence_stability=0.0,
            last_retraining=datetime.now(),
            alerts=[
                PerformanceAlert(
                    timestamp=datetime.now(),
                    alert_level=AlertLevel.CRITICAL,
                    metric="system_health",
                    current_value=0.0,
                    threshold=1.0,
                    deviation=1.0,
                    description="Model monitoring system error",
                    recommendation="Check monitoring system logs and restart if necessary"
                )
            ],
            accuracy_metrics={
                'overall_accuracy': 0.0,
                'recent_accuracy': 0.0,
                'accuracy_trend': 0.0,
                'brier_score': 0.25,
                'sample_size': 0
            }
        )

    # Keep other existing methods (calculate_feature_drift, calculate_prediction_drift, etc.)
    # with the same implementations as your original file, but ensure they use the enhanced features

# Enhanced utility functions
def create_professional_monitor(model_name: str = "FootballPredictionEngine") -> InstitutionalModelMonitor:
    """Create professional monitor with calibration loading"""
    monitor = InstitutionalModelMonitor(model_name)
    monitor.load_calibration_data()
    return monitor

def get_calibration_for_prediction_engine(monitor: InstitutionalModelMonitor) -> Dict[str, Any]:
    """Get calibration data for prediction engine integration"""
    return monitor.get_calibration_feedback()

def record_match_result(monitor: InstitutionalModelMonitor, prediction_data: Dict, 
                       home_goals: int, away_goals: int):
    """Professional function to record match results"""
    actual_result = {
        'home_goals': home_goals,
        'away_goals': away_goals,
        'timestamp': datetime.now().isoformat()
    }
    monitor.record_prediction(prediction_data, actual_result)