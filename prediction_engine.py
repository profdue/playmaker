# prediction_engine.py - IMPROVED VERSION
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

class MatchNarrative:
    """SIMPLIFIED BUT SMARTER MATCH NARRATIVE"""
    
    def __init__(self):
        self.dominance = "balanced"
        self.expected_tempo = "medium"
        self.defensive_stability = "mixed"
        self.primary_pattern = None
        self.key_factors = []
        
    def to_dict(self):
        return {
            'dominance': self.dominance,
            'expected_tempo': self.expected_tempo,
            'defensive_stability': self.defensive_stability,
            'primary_pattern': self.primary_pattern,
            'key_factors': self.key_factors
        }

@dataclass
class BettingSignal:
    market: str
    model_prob: float
    book_prob: float
    edge: float
    confidence: str
    recommended_stake: float
    value_rating: str
    aligns_with_primary: bool

@dataclass
class MonteCarloResults:
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    over_25_prob: float
    btts_prob: float
    exact_scores: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]

@dataclass
class IntelligenceMetrics:
    narrative_coherence: float
    prediction_alignment: str
    data_quality_score: float
    certainty_score: float
    risk_level: str
    football_iq_score: float

class DynamicTierCalibrator:
    """DYNAMIC TIER SYSTEM - Real-time team assessment"""
    
    def __init__(self):
        self.base_tiers = self._initialize_base_tiers()
        self.performance_cache = {}
        
    def _initialize_base_tiers(self):
        """Base tiers as starting point"""
        return {
            'premier_league': {
                'Arsenal': 'ELITE', 'Man City': 'ELITE', 'Liverpool': 'ELITE',
                'Tottenham': 'STRONG', 'Aston Villa': 'STRONG', 'Newcastle': 'STRONG',
                'West Ham': 'MEDIUM', 'Brighton': 'MEDIUM', 'Wolves': 'MEDIUM',
                'Chelsea': 'STRONG', 'Man United': 'STRONG', 'Crystal Palace': 'MEDIUM',
                'Fulham': 'MEDIUM', 'Bournemouth': 'MEDIUM', 'Brentford': 'MEDIUM',
                'Everton': 'MEDIUM', 'Nottingham Forest': 'MEDIUM', 'Luton': 'WEAK',
                'Burnley': 'WEAK', 'Sheffield United': 'WEAK'
            },
            'la_liga': {
                'Real Madrid': 'ELITE', 'Barcelona': 'ELITE', 'Atletico Madrid': 'ELITE',
                'Athletic Bilbao': 'STRONG', 'Real Sociedad': 'STRONG', 'Sevilla': 'STRONG',
                'Real Betis': 'MEDIUM', 'Getafe': 'MEDIUM', 'Osasuna': 'MEDIUM',
                'Valencia': 'MEDIUM', 'Villarreal': 'MEDIUM', 'Girona': 'STRONG',
                'Las Palmas': 'MEDIUM', 'Rayo Vallecano': 'MEDIUM', 'Mallorca': 'MEDIUM',
                'Alaves': 'WEAK', 'Celta Vigo': 'WEAK', 'Cadiz': 'WEAK',
                'Granada': 'WEAK', 'Almeria': 'WEAK'
            },
            'serie_a': {
                'Inter': 'ELITE', 'Juventus': 'ELITE', 'AC Milan': 'ELITE',
                'Napoli': 'STRONG', 'Atalanta': 'STRONG', 'Roma': 'STRONG',
                'Lazio': 'STRONG', 'Fiorentina': 'MEDIUM', 'Bologna': 'MEDIUM',
                'Monza': 'MEDIUM', 'Torino': 'MEDIUM', 'Genoa': 'MEDIUM',
                'Lecce': 'MEDIUM', 'Sassuolo': 'MEDIUM', 'Frosinone': 'WEAK',
                'Udinese': 'WEAK', 'Verona': 'WEAK', 'Empoli': 'WEAK',
                'Cagliari': 'WEAK', 'Salernitana': 'WEAK'
            }
        }
    
    def calculate_dynamic_tier(self, team: str, league: str, form_data: List[float], 
                             recent_goals: float, recent_conceded: float) -> str:
        """Calculate real-time tier based on performance"""
        base_tier = self.base_tiers.get(league, {}).get(team, 'MEDIUM')
        
        if not form_data:
            return base_tier
            
        # Calculate form score (recent matches weighted higher)
        form_score = self._calculate_form_score(form_data)
        
        # Calculate performance metrics
        goals_per_game = recent_goals / max(1, len(form_data))
        conceded_per_game = recent_conceded / max(1, len(form_data))
        
        # Performance adjustment
        performance_adjustment = self._calculate_performance_adjustment(
            form_score, goals_per_game, conceded_per_game, base_tier
        )
        
        return self._apply_tier_adjustment(base_tier, performance_adjustment)
    
    def _calculate_form_score(self, form_data: List[float]) -> float:
        """Weight recent form more heavily"""
        if not form_data:
            return 1.0
            
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Recent matches weighted higher
        available_weights = weights[:len(form_data)]
        
        # Normalize weights
        total_weight = sum(available_weights)
        normalized_weights = [w / total_weight for w in available_weights]
        
        # Calculate weighted form (points per game)
        max_points = 3.0  # Maximum points per game
        weighted_form = sum(score * weight for score, weight in zip(form_data, normalized_weights))
        
        return weighted_form / max_points
    
    def _calculate_performance_adjustment(self, form_score: float, goals_per_game: float, 
                                       conceded_per_game: float, base_tier: str) -> str:
        """Calculate performance-based tier adjustment"""
        
        # Expected performance by tier
        tier_expectations = {
            'ELITE': {'min_form': 0.7, 'min_goals': 1.8, 'max_conceded': 1.0},
            'STRONG': {'min_form': 0.6, 'min_goals': 1.5, 'max_conceded': 1.2},
            'MEDIUM': {'min_form': 0.4, 'min_goals': 1.0, 'max_conceded': 1.5},
            'WEAK': {'min_form': 0.3, 'min_goals': 0.8, 'max_conceded': 1.8}
        }
        
        expectations = tier_expectations.get(base_tier, tier_expectations['MEDIUM'])
        
        # Check if team exceeds expectations
        if (form_score > expectations['min_form'] + 0.15 and 
            goals_per_game > expectations['min_goals'] + 0.3 and
            conceded_per_game < expectations['max_conceded'] - 0.2):
            return "UPGRADE"
        
        # Check if team underperforms
        elif (form_score < expectations['min_form'] - 0.15 or 
              goals_per_game < expectations['min_goals'] - 0.3 or
              conceded_per_game > expectations['max_conceded'] + 0.2):
            return "DOWNGRADE"
        
        return "MAINTAIN"
    
    def _apply_tier_adjustment(self, base_tier: str, adjustment: str) -> str:
        """Apply tier adjustment"""
        tier_order = ['WEAK', 'MEDIUM', 'STRONG', 'ELITE']
        
        if adjustment == "UPGRADE":
            current_index = tier_order.index(base_tier)
            if current_index < len(tier_order) - 1:
                return tier_order[current_index + 1]
        
        elif adjustment == "DOWNGRADE":
            current_index = tier_order.index(base_tier)
            if current_index > 0:
                return tier_order[current_index - 1]
        
        return base_tier

class ContextAwareXGEngine:
    """CONTEXT-AWARE XG ENGINE - Smart adjustments"""
    
    def __init__(self):
        self.context_factors = {
            'derby_multiplier': 1.15,
            'relegation_battle': 1.10,
            'european_qualification': 1.08,
            'midtable_nothing': 0.95,
            'new_manager_bounce': 1.05
        }
        
        self.league_baselines = {
            'premier_league': {'avg_goals': 2.8, 'home_advantage': 0.35},
            'la_liga': {'avg_goals': 2.6, 'home_advantage': 0.32},
            'serie_a': {'avg_goals': 2.7, 'home_advantage': 0.38},
            'bundesliga': {'avg_goals': 3.1, 'home_advantage': 0.28},
            'ligue_1': {'avg_goals': 2.5, 'home_advantage': 0.34}
        }
    
    def calculate_contextual_xg(self, base_home_xg: float, base_away_xg: float, 
                              match_context: Dict, league: str) -> Tuple[float, float]:
        """Apply context-aware adjustments"""
        home_context_multiplier = 1.0
        away_context_multiplier = 1.0
        
        # Motivational context
        home_motivation = match_context.get('home_motivation', 'Normal')
        away_motivation = match_context.get('away_motivation', 'Normal')
        
        motivation_factors = {'Low': 0.9, 'Normal': 1.0, 'High': 1.08, 'Very High': 1.12}
        home_context_multiplier *= motivation_factors.get(home_motivation, 1.0)
        away_context_multiplier *= motivation_factors.get(away_motivation, 1.0)
        
        # Special match circumstances
        if match_context.get('is_derby', False):
            home_context_multiplier *= 1.08
            away_context_multiplier *= 1.08
        
        if match_context.get('relegation_battle', False):
            home_context_multiplier *= 1.05
            away_context_multiplier *= 1.05
        
        # Apply home advantage from league baselines
        league_baseline = self.league_baselines.get(league, self.league_baselines['premier_league'])
        home_advantage = league_baseline['home_advantage']
        home_context_multiplier *= (1 + home_advantage)
        
        # Calculate final xG
        home_xg = base_home_xg * home_context_multiplier
        away_xg = base_away_xg * away_context_multiplier
        
        return max(0.2, min(4.0, home_xg)), max(0.2, min(4.0, away_xg))

class EliteStakeCalculator:
    """ELITE STAKE SIZING USING KELLY CRITERION"""
    
    def __init__(self, max_stake=0.03, bankroll_fraction=0.02):
        self.max_stake = max_stake
        self.bankroll_fraction = bankroll_fraction
    
    def kelly_stake(self, model_prob: float, market_odds: float, confidence: str) -> float:
        """Enhanced Kelly Criterion with confidence adjustment"""
        if market_odds <= 1:
            return 0
        
        # Convert percentage to decimal
        model_prob_decimal = model_prob / 100.0
        implied_prob = 1.0 / market_odds
        
        # Basic Kelly formula
        kelly_fraction = (model_prob_decimal * market_odds - 1) / (market_odds - 1)
        
        # Confidence adjustment
        confidence_multiplier = {
            'HIGH': 0.25,      # 1/4 Kelly - conservative
            'MEDIUM': 0.15,    # 1/6 Kelly - more conservative
            'LOW': 0.08,       # 1/12 Kelly - very conservative
            'SPECULATIVE': 0.04 # 1/25 Kelly - extremely conservative
        }.get(confidence, 0.10)
        
        # Apply fractional Kelly with bankroll management
        stake_fraction = max(0, kelly_fraction * confidence_multiplier * self.bankroll_fraction)
        
        # Apply maximum stake limit
        final_stake = min(self.max_stake, stake_fraction)
        
        return max(0.005, final_stake)  # Minimum 0.5% stake

class PerformanceMonitor:
    """REAL-TIME PERFORMANCE MONITORING"""
    
    def __init__(self):
        self.prediction_history = []
        self.accuracy_metrics = {
            '1x2_accuracy': [],
            'btts_accuracy': [],
            'over_under_accuracy': []
        }
    
    def track_prediction(self, prediction: Dict, actual_result: Dict):
        """Track prediction for accuracy analysis"""
        match_record = {
            'timestamp': datetime.now(),
            'prediction': prediction,
            'actual_result': actual_result,
            'accuracy_metrics': self._calculate_accuracy_metrics(prediction, actual_result)
        }
        
        self.prediction_history.append(match_record)
        
        # Keep only last 100 predictions
        if len(self.prediction_history) > 100:
            self.prediction_history = self.prediction_history[-100:]
    
    def _calculate_accuracy_metrics(self, prediction: Dict, actual: Dict) -> Dict:
        """Calculate accuracy metrics for a single prediction"""
        pred_outcomes = prediction.get('probabilities', {}).get('match_outcomes', {})
        actual_outcome = actual.get('outcome')
        
        # 1X2 Accuracy
        predicted_winner = max(pred_outcomes, key=pred_outcomes.get)
        correct_1x2 = 1 if predicted_winner == actual_outcome else 0
        
        # BTTS Accuracy
        pred_btts = prediction.get('probabilities', {}).get('both_teams_score', {})
        pred_btts_yes = pred_btts.get('yes', 0) > pred_btts.get('no', 0)
        actual_btts = actual.get('both_teams_score', False)
        correct_btts = 1 if pred_btts_yes == actual_btts else 0
        
        # Over/Under Accuracy
        pred_ou = prediction.get('probabilities', {}).get('over_under', {})
        pred_over = pred_ou.get('over_25', 0) > pred_ou.get('under_25', 0)
        actual_over = actual.get('total_goals', 0) > 2.5
        correct_ou = 1 if pred_over == actual_over else 0
        
        return {
            '1x2_accuracy': correct_1x2,
            'btts_accuracy': correct_btts,
            'over_under_accuracy': correct_ou
        }
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        if not self.prediction_history:
            return {'status': 'No data available'}
        
        recent_predictions = self.prediction_history[-20:]  # Last 20 matches
        
        accuracies = {}
        for metric in ['1x2_accuracy', 'btts_accuracy', 'over_under_accuracy']:
            values = [p['accuracy_metrics'][metric] for p in recent_predictions if metric in p['accuracy_metrics']]
            if values:
                accuracies[metric] = sum(values) / len(values) * 100
        
        return {
            'recent_accuracy': accuracies,
            'total_predictions': len(self.prediction_history),
            'recent_sample': len(recent_predictions),
            'health_status': self._assess_health_status(accuracies)
        }
    
    def _assess_health_status(self, accuracies: Dict) -> str:
        """Assess model health based on accuracy metrics"""
        if not accuracies:
            return "UNKNOWN"
        
        avg_accuracy = sum(accuracies.values()) / len(accuracies)
        
        if avg_accuracy > 60:
            return "HEALTHY"
        elif avg_accuracy > 50:
            return "NEEDS_ATTENTION"
        else:
            return "PROBLEMATIC"

class SmartFootballPredictor:
    """SMARTER FOOTBALL PREDICTOR - Improved Version"""
    
    def __init__(self, match_data: Dict[str, Any]):
        self.data = self._validate_and_enhance_data(match_data)
        self.tier_calibrator = DynamicTierCalibrator()
        self.context_engine = ContextAwareXGEngine()
        self.stake_calculator = EliteStakeCalculator()
        self.performance_monitor = PerformanceMonitor()
        self.narrative = MatchNarrative()
        
    def _validate_and_enhance_data(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced data validation"""
        enhanced_data = match_data.copy()
        
        # Ensure required fields
        required_fields = ['home_team', 'away_team', 'league']
        for field in required_fields:
            if field not in enhanced_data:
                enhanced_data[field] = 'Unknown'
        
        # Enhanced form data processing
        for form_field in ['home_form', 'away_form']:
            if form_field in enhanced_data and enhanced_data[form_field]:
                try:
                    if isinstance(enhanced_data[form_field], list):
                        form_data = enhanced_data[form_field]
                        enhanced_data[form_field] = form_data
                    else:
                        enhanced_data[form_field] = []
                except (TypeError, ValueError):
                    enhanced_data[form_field] = []
            else:
                enhanced_data[form_field] = []
        
        # Enhanced context data
        if 'match_context' not in enhanced_data:
            enhanced_data['match_context'] = {
                'home_motivation': 'Normal',
                'away_motivation': 'Normal',
                'is_derby': False,
                'relegation_battle': False
            }
            
        return enhanced_data

    def _calculate_base_xg(self) -> Tuple[float, float]:
        """Calculate base xG without context adjustments"""
        league = self.data.get('league', 'premier_league')
        
        # Calculate averages per game
        home_goals_avg = self.data.get('home_goals', 8) / max(1, len(self.data.get('home_form', [1.5])))
        away_goals_avg = self.data.get('away_goals', 8) / max(1, len(self.data.get('away_form', [1.5])))
        home_conceded_avg = self.data.get('home_conceded', 8) / max(1, len(self.data.get('home_form', [1.5])))
        away_conceded_avg = self.data.get('away_conceded', 8) / max(1, len(self.data.get('away_form', [1.5])))
        
        # Home/away specific adjustments
        home_goals_home_avg = self.data.get('home_goals_home', 4) / 3.0
        away_goals_away_avg = self.data.get('away_goals_away', 4) / 3.0
        
        # Blend overall and home/away specific performance
        home_attack = (home_goals_avg * 0.7) + (home_goals_home_avg * 0.3)
        away_attack = (away_goals_avg * 0.7) + (away_goals_away_avg * 0.3)
        
        # Defensive adjustment
        league_baseline = self.context_engine.league_baselines.get(league, {'avg_goals': 2.7})
        home_xg = home_attack * (1 - (away_conceded_avg / (league_baseline['avg_goals'] + 0.5)) * 0.3)
        away_xg = away_attack * (1 - (home_conceded_avg / (league_baseline['avg_goals'] + 0.5)) * 0.3)
        
        return home_xg, away_xg

    def _determine_match_narrative(self, home_xg: float, away_xg: float) -> MatchNarrative:
        """Simplified but smarter narrative determination"""
        narrative = MatchNarrative()
        total_xg = home_xg + away_xg
        xg_difference = home_xg - away_xg
        
        # Determine dominance
        if xg_difference > 0.8:
            narrative.dominance = "home_dominance"
            narrative.primary_pattern = "home_dominance"
            narrative.key_factors.append("Home team expected to dominate")
        elif xg_difference < -0.8:
            narrative.dominance = "away_dominance"
            narrative.primary_pattern = "away_counter"
            narrative.key_factors.append("Away team has significant advantage")
        else:
            narrative.dominance = "balanced"
            narrative.primary_pattern = "balanced"
            narrative.key_factors.append("Evenly matched contest")
        
        # Determine tempo based on total xG
        if total_xg > 3.2:
            narrative.expected_tempo = "high"
            narrative.key_factors.append("High-scoring game expected")
        elif total_xg < 2.0:
            narrative.expected_tempo = "low"
            narrative.key_factors.append("Low-scoring affair likely")
        else:
            narrative.expected_tempo = "medium"
        
        # Defensive stability
        home_defense = self.data.get('home_conceded', 0) / max(1, len(self.data.get('home_form', [1.5])))
        away_defense = self.data.get('away_conceded', 0) / max(1, len(self.data.get('away_form', [1.5])))
        avg_defense = (home_defense + away_defense) / 2
        
        if avg_defense < 0.8:
            narrative.defensive_stability = "solid"
            narrative.key_factors.append("Both teams defensively solid")
        elif avg_defense > 1.5:
            narrative.defensive_stability = "leaky"
            narrative.key_factors.append("Defensive vulnerabilities present")
        else:
            narrative.defensive_stability = "mixed"
            
        return narrative

    def _run_monte_carlo(self, home_xg: float, away_xg: float, iterations: int = 10000) -> MonteCarloResults:
        """Run Monte Carlo simulation"""
        np.random.seed(42)  # For reproducible results
        
        # Generate Poisson distributions
        home_goals_sim = np.random.poisson(home_xg, iterations)
        away_goals_sim = np.random.poisson(away_xg, iterations)
        
        # Calculate probabilities
        home_wins = np.sum(home_goals_sim > away_goals_sim) / iterations
        draws = np.sum(home_goals_sim == away_goals_sim) / iterations
        away_wins = np.sum(home_goals_sim < away_goals_sim) / iterations
        
        # Calculate exact score probabilities
        exact_scores = {}
        for i in range(6):  # 0-5 goals
            for j in range(6):
                count = np.sum((home_goals_sim == i) & (away_goals_sim == j))
                prob = count / iterations
                if prob > 0.005:  # Only include probabilities > 0.5%
                    exact_scores[f"{i}-{j}"] = round(prob * 100, 1)
        
        # Get top 6 most likely scores
        exact_scores = dict(sorted(exact_scores.items(), key=lambda x: x[1], reverse=True)[:6])
        
        # Calculate BTTS and Over 2.5 probabilities
        btts_count = np.sum((home_goals_sim > 0) & (away_goals_sim > 0)) / iterations
        over_25_count = np.sum(home_goals_sim + away_goals_sim > 2.5) / iterations
        
        return MonteCarloResults(
            home_win_prob=home_wins, 
            draw_prob=draws, 
            away_win_prob=away_wins,
            over_25_prob=over_25_count, 
            btts_prob=btts_count, 
            exact_scores=exact_scores,
            confidence_intervals={
                'home_win': (home_wins - 0.02, home_wins + 0.02),
                'draw': (draws - 0.02, draws + 0.02),
                'away_win': (away_wins - 0.02, away_wins + 0.02)
            }
        )

    def generate_predictions(self, mc_iterations: int = 10000) -> Dict[str, Any]:
        """Generate comprehensive predictions"""
        # Calculate dynamic tiers
        home_tier = self.tier_calibrator.calculate_dynamic_tier(
            self.data['home_team'], self.data['league'],
            self.data.get('home_form', []), self.data.get('home_goals', 0), self.data.get('home_conceded', 0)
        )
        away_tier = self.tier_calibrator.calculate_dynamic_tier(
            self.data['away_team'], self.data['league'],
            self.data.get('away_form', []), self.data.get('away_goals', 0), self.data.get('away_conceded', 0)
        )
        
        # Calculate base xG
        base_home_xg, base_away_xg = self._calculate_base_xg()
        
        # Apply context adjustments
        home_xg, away_xg = self.context_engine.calculate_contextual_xg(
            base_home_xg, base_away_xg, 
            self.data.get('match_context', {}), self.data['league']
        )
        
        # Determine match narrative
        self.narrative = self._determine_match_narrative(home_xg, away_xg)
        
        # Run Monte Carlo simulation
        mc_results = self._run_monte_carlo(home_xg, away_xg, mc_iterations)
        
        # Calculate additional probabilities
        goal_timing = self._calculate_goal_timing(home_xg + away_xg)
        
        # Calculate certainty and risk
        certainty = max(mc_results.home_win_prob, mc_results.away_win_prob, mc_results.draw_prob)
        risk_level = self._calculate_risk_level(certainty, home_xg, away_xg)
        
        # Generate intelligent summary
        summary = self._generate_intelligent_summary(home_xg, away_xg, mc_results)
        
        return {
            'match': f"{self.data['home_team']} vs {self.data['away_team']}",
            'league': self.data['league'],
            'expected_goals': {'home': round(home_xg, 2), 'away': round(away_xg, 2)},
            'team_tiers': {'home': home_tier, 'away': away_tier},
            'match_context': self.narrative.primary_pattern,
            'confidence_score': round(certainty * 100, 1),
            'match_narrative': self.narrative.to_dict(),
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
                'goal_timing': goal_timing,
                'exact_scores': mc_results.exact_scores
            },
            'risk_assessment': {
                'risk_level': risk_level,
                'explanation': self._get_risk_explanation(risk_level),
                'recommendation': self._get_risk_recommendation(risk_level),
                'certainty': f"{certainty * 100:.1f}%"
            },
            'summary': summary,
            'key_factors': self.narrative.key_factors
        }

    def _calculate_goal_timing(self, total_xg: float) -> Dict[str, float]:
        """Calculate goal timing probabilities"""
        first_half = 1 - poisson.pmf(0, total_xg * 0.46)
        second_half = 1 - poisson.pmf(0, total_xg * 0.54)
        
        return {
            'first_half': round(first_half * 100, 1),
            'second_half': round(second_half * 100, 1)
        }

    def _calculate_risk_level(self, certainty: float, home_xg: float, away_xg: float) -> str:
        """Calculate risk level based on certainty and xG patterns"""
        xg_difference = abs(home_xg - away_xg)
        
        if certainty > 0.7 and xg_difference > 1.0:
            return "LOW"
        elif certainty > 0.5 and xg_difference > 0.5:
            return "MEDIUM"
        elif certainty > 0.4:
            return "HIGH"
        else:
            return "VERY_HIGH"

    def _generate_intelligent_summary(self, home_xg: float, away_xg: float, 
                                   mc_results: MonteCarloResults) -> str:
        """Generate data-driven match summary"""
        home_team = self.data['home_team']
        away_team = self.data['away_team']
        
        home_win_prob = mc_results.home_win_prob * 100
        away_win_prob = mc_results.away_win_prob * 100
        draw_prob = mc_results.draw_prob * 100
        
        if home_win_prob > 60:
            return f"{home_team} are strong favorites to win this match. Their attacking quality should overwhelm {away_team}'s defense, with multiple goals likely."
        elif away_win_prob > 60:
            return f"{away_team} are expected to control this match. {home_team} will need to be defensively disciplined to contain the away side's threat."
        elif draw_prob > 40:
            return f"A closely contested match is anticipated between {home_team} and {away_team}. Neither side has a clear advantage, making a draw the most likely outcome."
        elif home_win_prob > away_win_prob:
            return f"{home_team} have a slight edge in this encounter. Home advantage could be the deciding factor in what promises to be a competitive match."
        else:
            return f"A competitive match expected between {home_team} and {away_team}. Both teams will seek to establish control in what promises to be a closely-fought encounter."

    def _get_risk_explanation(self, risk_level: str) -> str:
        explanations = {
            'LOW': "Clear favorite with strong predictive signals",
            'MEDIUM': "Reasonable prediction alignment with some uncertainties",
            'HIGH': "Multiple uncertainties with conflicting signals",
            'VERY_HIGH': "High unpredictability with limited clear patterns"
        }
        return explanations.get(risk_level, "Risk assessment unavailable")

    def _get_risk_recommendation(self, risk_level: str) -> str:
        recommendations = {
            'LOW': "CONSIDER CONFIDENT STAKE",
            'MEDIUM': "SMALL TO MEDIUM STAKE", 
            'HIGH': "MINIMAL STAKE ONLY",
            'VERY_HIGH': "AVOID OR TINY STAKE"
        }
        return recommendations.get(risk_level, "N/A")

class ValueDetectionEngine:
    """IMPROVED VALUE DETECTION ENGINE"""
    
    def __init__(self):
        self.value_thresholds = {
            'EXCEPTIONAL': 25.0, 'HIGH': 15.0, 'GOOD': 8.0, 'MODERATE': 4.0,
        }
        self.stake_calculator = EliteStakeCalculator()

    def detect_value_bets(self, pure_probabilities: Dict, market_odds: Dict, 
                         primary_predictions: Dict) -> List[BettingSignal]:
        """Detect value bets with alignment checking"""
        signals = []
        
        # Extract probabilities
        outcomes = pure_probabilities.get('probabilities', {}).get('match_outcomes', {})
        home_pure = outcomes.get('home_win', 33.3) / 100.0
        draw_pure = outcomes.get('draw', 33.3) / 100.0  
        away_pure = outcomes.get('away_win', 33.3) / 100.0
        
        # Get other probabilities
        over_under = pure_probabilities.get('probabilities', {}).get('over_under', {})
        btts = pure_probabilities.get('probabilities', {}).get('both_teams_score', {})
        
        over_25_pure = over_under.get('over_25', 50) / 100.0
        under_25_pure = over_under.get('under_25', 50) / 100.0
        btts_yes_pure = btts.get('yes', 50) / 100.0
        btts_no_pure = btts.get('no', 50) / 100.0
        
        # Define probability mapping
        probability_mapping = [
            ('1x2 Home', home_pure, '1x2 Home'),
            ('1x2 Draw', draw_pure, '1x2 Draw'), 
            ('1x2 Away', away_pure, '1x2 Away'),
            ('Over 2.5 Goals', over_25_pure, 'Over 2.5 Goals'),
            ('Under 2.5 Goals', under_25_pure, 'Under 2.5 Goals'),
            ('BTTS Yes', btts_yes_pure, 'BTTS Yes'),
            ('BTTS No', btts_no_pure, 'BTTS No')
        ]
        
        # Get primary predictions for alignment checking
        primary_outcome = max(
            primary_predictions.get('match_outcomes', {}), 
            key=primary_predictions.get('match_outcomes', {}).get
        ) if primary_predictions.get('match_outcomes') else 'unknown'
        
        primary_btts = 'yes' if primary_predictions.get('both_teams_score', {}).get('yes', 0) > 50 else 'no'
        primary_ou = 'over_25' if primary_predictions.get('over_under', {}).get('over_25', 0) > 50 else 'under_25'
        
        for market_name, pure_prob, market_key in probability_mapping:
            market_odd = market_odds.get(market_key, 0)
            if market_odd <= 1:
                continue
                
            market_prob = 1.0 / market_odd
            edge = (pure_prob / market_prob) - 1.0
            edge_percentage = edge * 100
            
            if edge_percentage >= 4.0:  # Minimum edge threshold
                value_rating = self._get_value_rating(edge_percentage)
                
                # Determine confidence for stake sizing
                confidence = self._determine_confidence(pure_prob, edge_percentage)
                
                # Check alignment with primary predictions
                aligns_with_primary = self._check_alignment(
                    market_name, primary_outcome, primary_btts, primary_ou
                )
                
                # Use Kelly criterion for stake sizing
                stake = self.stake_calculator.kelly_stake(
                    pure_prob * 100, market_odd, confidence
                )
                
                signal = BettingSignal(
                    market=market_name, 
                    model_prob=round(pure_prob * 100, 1),
                    book_prob=round(market_prob * 100, 1), 
                    edge=round(edge_percentage, 1),
                    confidence=confidence, 
                    recommended_stake=stake, 
                    value_rating=value_rating,
                    aligns_with_primary=aligns_with_primary
                )
                signals.append(signal)
        
        # Sort by edge (highest first)
        signals.sort(key=lambda x: x.edge, reverse=True)
        return signals
    
    def _get_value_rating(self, edge: float) -> str:
        for rating, threshold in self.value_thresholds.items():
            if edge >= threshold:
                return rating
        return "LOW"
    
    def _determine_confidence(self, pure_prob: float, edge: float) -> str:
        """Determine confidence level for stake sizing"""
        if pure_prob > 0.7 and edge > 15:
            return "HIGH"
        elif pure_prob > 0.6 and edge > 10:
            return "MEDIUM"
        elif pure_prob > 0.5 and edge > 6:
            return "LOW"
        else:
            return "SPECULATIVE"
    
    def _check_alignment(self, market: str, primary_outcome: str, 
                        primary_btts: str, primary_ou: str) -> bool:
        """Check if value bet aligns with primary predictions"""
        if market == '1x2 Home' and primary_outcome == 'home_win':
            return True
        elif market == '1x2 Away' and primary_outcome == 'away_win':
            return True
        elif market == '1x2 Draw' and primary_outcome == 'draw':
            return True
        elif market == 'BTTS Yes' and primary_btts == 'yes':
            return True
        elif market == 'BTTS No' and primary_btts == 'no':
            return True
        elif market == 'Over 2.5 Goals' and primary_ou == 'over_25':
            return True
        elif market == 'Under 2.5 Goals' and primary_ou == 'under_25':
            return True
        else:
            return False

class AdvancedFootballPredictor:
    """MAIN PREDICTOR CLASS - Improved Version"""
    
    def __init__(self, match_data: Dict[str, Any]):
        self.market_odds = match_data.get('market_odds', {})
        football_data = match_data.copy()
        if 'market_odds' in football_data:
            del football_data['market_odds']
        
        self.football_predictor = SmartFootballPredictor(football_data)
        self.value_engine = ValueDetectionEngine()
        self.performance_monitor = PerformanceMonitor()

    def generate_comprehensive_analysis(self, mc_iterations: int = 10000) -> Dict[str, Any]:
        """Generate comprehensive analysis with performance tracking"""
        football_predictions = self.football_predictor.generate_predictions(mc_iterations)
        value_signals = self.value_engine.detect_value_bets(
            football_predictions, self.market_odds, 
            football_predictions['probabilities']
        )
        
        # Enhanced system validation
        alignment_status = self._validate_system_alignment(football_predictions, value_signals)
        
        comprehensive_result = football_predictions.copy()
        comprehensive_result['betting_signals'] = [signal.__dict__ for signal in value_signals]
        comprehensive_result['system_validation'] = {
            'status': 'VALID', 
            'alignment': alignment_status,
            'engine_sync': 'OPTIMAL'
        }
        
        # Performance tracking
        self.performance_monitor.track_prediction(
            comprehensive_result, 
            {'outcome': 'pending', 'both_teams_score': False, 'total_goals': 0}
        )
        
        return comprehensive_result
    
    def _validate_system_alignment(self, football_predictions: Dict, value_signals: List[BettingSignal]) -> str:
        """Validate system alignment"""
        if not value_signals:
            return "PERFECT"
            
        # Count aligned vs contradictory signals
        aligned_count = sum(1 for signal in value_signals if signal.aligns_with_primary)
        total_count = len(value_signals)
        
        alignment_ratio = aligned_count / total_count
        
        if alignment_ratio >= 0.8:
            return "PERFECT"
        elif alignment_ratio >= 0.6:
            return "GOOD"
        elif alignment_ratio >= 0.4:
            return "PARTIAL"
        else:
            return "CONTRADICTORY"

# TEST FUNCTION
def test_improved_predictor():
    """Test the improved predictor"""
    match_data = {
        'home_team': 'Crystal Palace', 'away_team': 'Brentford', 'league': 'premier_league',
        'home_goals': 8, 'away_goals': 12, 'home_conceded': 8, 'away_conceded': 6,
        'home_goals_home': 5, 'away_goals_away': 7,
        'home_form': [3, 0, 3, 0, 0, 3], 'away_form': [3, 3, 1, 3, 3, 3],
        'h2h_data': {'matches': 6, 'home_wins': 3, 'away_wins': 2, 'draws': 1, 'home_goals': 10, 'away_goals': 9},
        'match_context': {
            'home_motivation': 'High', 
            'away_motivation': 'Normal',
            'is_derby': False,
            'relegation_battle': True
        },
        'market_odds': {
            '1x2 Home': 2.70, '1x2 Draw': 3.75, '1x2 Away': 2.38,
            'Over 2.5 Goals': 1.44, 'Under 2.5 Goals': 2.75,
            'BTTS Yes': 1.40, 'BTTS No': 2.75
        }
    }
    
    predictor = AdvancedFootballPredictor(match_data)
    results = predictor.generate_comprehensive_analysis()
    
    print("🧠 IMPROVED FOOTBALL PREDICTOR RESULTS")
    print("=" * 60)
    print(f"Match: {results['match']}")
    print(f"Dynamic Tiers: {results['team_tiers']['home']} vs {results['team_tiers']['away']}")
    print(f"Expected Goals: {results['expected_goals']['home']:.2f} - {results['expected_goals']['away']:.2f}")
    print(f"Confidence: {results['confidence_score']}%")
    print(f"Risk Level: {results['risk_assessment']['risk_level']}")
    print()
    
    print("📊 PROBABILITIES:")
    outcomes = results['probabilities']['match_outcomes']
    print(f"Home Win: {outcomes['home_win']}% | Draw: {outcomes['draw']}% | Away Win: {outcomes['away_win']}%")
    print()
    
    print("🎯 KEY FACTORS:")
    for factor in results.get('key_factors', []):
        print(f"  • {factor}")
    print()
    
    print("💰 VALUE BETS:")
    if results['betting_signals']:
        for signal in results['betting_signals']:
            alignment = "✅" if signal['aligns_with_primary'] else "⚠️"
            print(f"  {alignment} {signal['market']}: {signal['edge']}% edge | Stake: {signal['recommended_stake']*100:.1f}%")
    else:
        print("  No value bets detected - market is efficient")
    print()
    
    print("📝 SUMMARY:")
    print(results['summary'])

if __name__ == "__main__":
    test_improved_predictor()
