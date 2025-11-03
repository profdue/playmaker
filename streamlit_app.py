# performance_tracker.py - NEW PERFORMANCE TRACKING MODULE
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PredictionRecord:
    timestamp: datetime
    match: str
    league: str
    predictions: Dict[str, Any]
    actual_result: Dict[str, Any]
    accuracy_metrics: Dict[str, float]

class PerformanceTracker:
    """Comprehensive performance tracking system"""
    
    def __init__(self, data_file: str = "performance_data.json"):
        self.data_file = data_file
        self.records: List[PredictionRecord] = []
        self.load_data()
    
    def load_data(self):
        """Load performance data from file"""
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
                for record in data:
                    self.records.append(PredictionRecord(
                        timestamp=datetime.fromisoformat(record['timestamp']),
                        match=record['match'],
                        league=record['league'],
                        predictions=record['predictions'],
                        actual_result=record['actual_result'],
                        accuracy_metrics=record['accuracy_metrics']
                    ))
            logger.info(f"Loaded {len(self.records)} performance records")
        except FileNotFoundError:
            logger.info("No existing performance data found")
        except Exception as e:
            logger.error(f"Error loading performance data: {e}")
    
    def save_data(self):
        """Save performance data to file"""
        try:
            data = []
            for record in self.records:
                data.append({
                    'timestamp': record.timestamp.isoformat(),
                    'match': record.match,
                    'league': record.league,
                    'predictions': record.predictions,
                    'actual_result': record.actual_result,
                    'accuracy_metrics': record.accuracy_metrics
                })
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.records)} performance records")
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
    
    def add_prediction(self, prediction: Dict[str, Any], actual_result: Dict[str, Any]):
        """Add a new prediction with actual result"""
        record = PredictionRecord(
            timestamp=datetime.now(),
            match=prediction.get('match', 'Unknown'),
            league=prediction.get('league', 'Unknown'),
            predictions=prediction,
            actual_result=actual_result,
            accuracy_metrics=self._calculate_accuracy_metrics(prediction, actual_result)
        )
        
        self.records.append(record)
        self.save_data()
    
    def _calculate_accuracy_metrics(self, prediction: Dict, actual: Dict) -> Dict[str, float]:
        """Calculate accuracy metrics for a prediction"""
        metrics = {}
        
        # 1X2 Accuracy
        pred_outcomes = prediction.get('probabilities', {}).get('match_outcomes', {})
        actual_outcome = actual.get('outcome')
        
        if pred_outcomes and actual_outcome:
            predicted_winner = max(pred_outcomes, key=pred_outcomes.get)
            metrics['1x2_accuracy'] = 1.0 if predicted_winner == actual_outcome else 0.0
        
        # BTTS Accuracy
        pred_btts = prediction.get('probabilities', {}).get('both_teams_score', {})
        actual_btts = actual.get('both_teams_score')
        
        if pred_btts and actual_btts is not None:
            pred_btts_yes = pred_btts.get('yes', 0) > pred_btts.get('no', 0)
            metrics['btts_accuracy'] = 1.0 if pred_btts_yes == actual_btts else 0.0
        
        # Over/Under Accuracy
        pred_ou = prediction.get('probabilities', {}).get('over_under', {})
        actual_goals = actual.get('total_goals', 0)
        
        if pred_ou and actual_goals is not None:
            pred_over = pred_ou.get('over_25', 0) > pred_ou.get('under_25', 0)
            actual_over = actual_goals > 2.5
            metrics['over_under_accuracy'] = 1.0 if pred_over == actual_over else 0.0
        
        # Confidence Calibration
        confidence = prediction.get('confidence_score', 0) / 100.0
        if '1x2_accuracy' in metrics:
            metrics['confidence_calibration'] = abs(confidence - metrics['1x2_accuracy'])
        
        return metrics
    
    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get performance summary for recent period"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_records = [r for r in self.records if r.timestamp >= cutoff_date]
        
        if not recent_records:
            return {'status': 'No data available'}
        
        summary = {
            'period_days': days,
            'total_matches': len(recent_records),
            'accuracy_metrics': {},
            'league_breakdown': {},
            'risk_performance': {},
            'trends': {}
        }
        
        # Calculate accuracy metrics
        for metric in ['1x2_accuracy', 'btts_accuracy', 'over_under_accuracy']:
            values = [r.accuracy_metrics.get(metric, 0) for r in recent_records if metric in r.accuracy_metrics]
            if values:
                summary['accuracy_metrics'][metric] = {
                    'accuracy': np.mean(values) * 100,
                    'total': len(values),
                    'successful': sum(values)
                }
        
        # League breakdown
        leagues = set(r.league for r in recent_records)
        for league in leagues:
            league_records = [r for r in recent_records if r.league == league]
            if league_records:
                accuracies = [r.accuracy_metrics.get('1x2_accuracy', 0) for r in league_records if '1x2_accuracy' in r.accuracy_metrics]
                if accuracies:
                    summary['league_breakdown'][league] = {
                        'matches': len(league_records),
                        'accuracy': np.mean(accuracies) * 100
                    }
        
        # Risk performance
        for record in recent_records:
            risk_level = record.predictions.get('risk_assessment', {}).get('risk_level', 'UNKNOWN')
            accuracy = record.accuracy_metrics.get('1x2_accuracy', 0)
            
            if risk_level not in summary['risk_performance']:
                summary['risk_performance'][risk_level] = {'total': 0, 'successful': 0}
            
            summary['risk_performance'][risk_level]['total'] += 1
            summary['risk_performance'][risk_level]['successful'] += accuracy
        
        # Calculate trends
        if len(recent_records) >= 10:
            recent_accuracies = [r.accuracy_metrics.get('1x2_accuracy', 0) for r in recent_records[-10:]]
            if recent_accuracies:
                summary['trends']['recent_10_accuracy'] = np.mean(recent_accuracies) * 100
                summary['trends']['trend_direction'] = 'improving' if len(recent_accuracies) > 1 and recent_accuracies[-1] > recent_accuracies[0] else 'declining'
        
        return summary
    
    def get_model_health(self) -> Dict[str, Any]:
        """Get comprehensive model health assessment"""
        summary = self.get_performance_summary(days=30)
        
        if summary.get('status') == 'No data available':
            return {'status': 'INSUFFICIENT_DATA'}
        
        health = {
            'status': 'HEALTHY',
            'confidence': 'HIGH',
            'recommendations': [],
            'alerts': []
        }
        
        # Check overall accuracy
        overall_accuracy = summary['accuracy_metrics'].get('1x2_accuracy', {}).get('accuracy', 0)
        if overall_accuracy < 45:
            health['status'] = 'NEEDS_ATTENTION'
            health['alerts'].append(f"Low overall accuracy: {overall_accuracy:.1f}%")
        elif overall_accuracy > 55:
            health['confidence'] = 'VERY_HIGH'
        
        # Check confidence calibration
        calibration_errors = []
        for record in self.records[-20:]:  # Last 20 records
            if 'confidence_calibration' in record.accuracy_metrics:
                calibration_errors.append(record.accuracy_metrics['confidence_calibration'])
        
        if calibration_errors and np.mean(calibration_errors) > 0.2:
            health['alerts'].append("Poor confidence calibration detected")
            health['recommendations'].append("Review confidence scoring algorithm")
        
        # Check for biases
        league_accuracies = []
        for league, data in summary['league_breakdown'].items():
            if data['matches'] >= 5:
                league_accuracies.append(data['accuracy'])
        
        if league_accuracies and max(league_accuracies) - min(league_accuracies) > 20:
            health['alerts'].append("Significant league performance variation detected")
            health['recommendations'].append("Review league-specific calibration")
        
        return health

# Example usage
if __name__ == "__main__":
    tracker = PerformanceTracker()
    
    # Example: Add a prediction
    sample_prediction = {
        'match': 'Team A vs Team B',
        'league': 'premier_league',
        'probabilities': {
            'match_outcomes': {'home_win': 60, 'draw': 25, 'away_win': 15},
            'both_teams_score': {'yes': 70, 'no': 30},
            'over_under': {'over_25': 65, 'under_25': 35}
        },
        'confidence_score': 60,
        'risk_assessment': {'risk_level': 'MEDIUM'}
    }
    
    sample_actual = {
        'outcome': 'home_win',
        'both_teams_score': True,
        'total_goals': 3
    }
    
    tracker.add_prediction(sample_prediction, sample_actual)
    
    # Get performance summary
    summary = tracker.get_performance_summary()
    print("Performance Summary:", json.dumps(summary, indent=2))
    
    # Get model health
    health = tracker.get_model_health()
    print("Model Health:", json.dumps(health, indent=2))
