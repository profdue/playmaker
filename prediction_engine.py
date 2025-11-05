# prediction_engine.py - ENHANCED OUTCOME-BASED PROFESSIONAL BETTING GRADE
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
import warnings
warnings.filterwarnings('ignore')

# Set up professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MatchContext(Enum):
    OFFENSIVE_SHOWDOWN = "offensive_showdown"
    DEFENSIVE_BATTLE = "defensive_battle" 
    HOME_DOMINANCE = "home_dominance"
    AWAY_COUNTER = "away_counter"
    TACTICAL_STALEMATE = "tactical_stalemate"
    BALANCED = "balanced"

class MatchNarrative:
    """ENHANCED OUTCOME-BASED MATCH NARRATIVE - Professional Football Intelligence"""
    
    def __init__(self):
        self.dominance = "balanced"
        self.style_conflict = "neutral"
        self.expected_tempo = "medium"
        self.expected_openness = 0.5
        self.defensive_stability = "mixed"
        self.primary_pattern = None
        self.quality_gap = "even"
        self.expected_outcome = "balanced"  # NEW: Direct outcome expectation
        self.betting_priority = []  # NEW: Recommended bet types
        
    def to_dict(self):
        return {
            'dominance': self.dominance,
            'style_conflict': self.style_conflict,
            'expected_tempo': self.expected_tempo,
            'expected_openness': self.expected_openness,
            'defensive_stability': self.defensive_stability,
            'primary_pattern': self.primary_pattern,
            'quality_gap': self.quality_gap,
            'expected_outcome': self.expected_outcome,  # NEW
            'betting_priority': self.betting_priority   # NEW
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
    explanation: List[str]
    alignment: str
    confidence_reasoning: List[str]
    context_alignment: str  # NEW: How well it aligns with match context

@dataclass
class MonteCarloResults:
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    over_25_prob: float
    btts_prob: float
    exact_scores: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    probability_volatility: Dict[str, float]

@dataclass
class IntelligenceMetrics:
    narrative_coherence: float
    prediction_alignment: str
    data_quality_score: float
    certainty_score: float
    market_edge_score: float
    risk_level: str
    football_iq_score: float
    calibration_status: str
    context_confidence: float  # NEW: Confidence in context assessment

class EliteFeatureEngine:
    """ENHANCED Feature Engineering with Outcome-Based Calibration"""
    
    def __init__(self):
        self.feature_metadata = {}
        
    def create_match_features(self, home_data: Dict, away_data: Dict, context: Dict, home_tier: str, away_tier: str) -> Dict[str, float]:
        """Generate professional feature set with outcome-based calibration"""
        
        features = {}
        
        # CRITICAL: Apply tier-based adjustments FIRST
        tier_adjustments = self._calculate_tier_adjustments(home_tier, away_tier, context.get('league', 'premier_league'))
        
        # Base team metrics with tier adjustments
        features.update({
            'home_xg_for': home_data.get('xg_home', 1.5) * tier_adjustments['home_attack_multiplier'],
            'away_xg_for': away_data.get('xg_away', 1.2) * tier_adjustments['away_attack_multiplier'],
            'home_xg_against': home_data.get('xga_home', 1.3) * tier_adjustments['home_defense_multiplier'],
            'away_xg_against': away_data.get('xga_away', 1.4) * tier_adjustments['away_defense_multiplier'],
        })
        
        # Form metrics with professional weighting
        features.update({
            'home_form_attack': self._professional_form_weight(home_data.get('xg_last_5', [1.5]*5), is_home=True),
            'away_form_attack': self._professional_form_weight(away_data.get('xg_last_5', [1.2]*5), is_home=False),
            'home_form_defense': self._professional_form_weight(home_data.get('xga_last_5', [1.3]*5), is_home=True),
            'away_form_defense': self._professional_form_weight(away_data.get('xga_last_5', [1.4]*5), is_home=False),
        })
        
        # ENHANCED: Professional interaction features
        features.update({
            'home_attack_vs_away_defense': features['home_xg_for'] * (1 - (features['away_xg_against'] / 3.0)),
            'away_attack_vs_home_defense': features['away_xg_for'] * (1 - (features['home_xg_against'] / 3.0)),
            'total_xg_expected': features['home_xg_for'] + features['away_xg_for'],
            'xg_difference': features['home_xg_for'] - features['away_xg_for'],
            'quality_gap_metric': tier_adjustments['quality_gap_score'],
            'home_dominance_potential': features['home_xg_for'] * (2 - features['away_xg_against']),
            'away_counter_potential': features['away_xg_for'] * (2 - features['home_xg_against']),  # NEW
        })
        
        # Contextual modifiers with professional impact assessment
        features.update(self._professional_contextual_features(context, tier_adjustments))
        
        return features
    
    def _calculate_tier_adjustments(self, home_tier: str, away_tier: str, league: str) -> Dict[str, float]:
        """CRITICAL: Professional tier-based adjustments for betting"""
        
        tier_strength = {
            'ELITE': 1.4, 'STRONG': 1.15, 'MEDIUM': 1.0, 'WEAK': 0.8
        }
        
        home_strength = tier_strength.get(home_tier, 1.0)
        away_strength = tier_strength.get(away_tier, 1.0)
        
        strength_ratio = home_strength / away_strength
        quality_gap_score = min(2.0, strength_ratio)
        
        # ENHANCED: League-specific confidence multipliers
        league_confidence_multipliers = {
            'premier_league': 1.0, 'la_liga': 0.95, 'serie_a': 1.15, 
            'bundesliga': 0.9, 'ligue_1': 1.05, 'liga_portugal': 1.1,
            'brasileirao': 0.95, 'liga_mx': 1.0, 'eredivisie': 0.9,
            'championship': 1.08
        }
        
        impact_multiplier = league_confidence_multipliers.get(league, 1.0)
        quality_gap_score *= impact_multiplier
        
        # ENHANCED: More conservative away counter adjustments
        if strength_ratio < 0.8:  # Away team stronger
            home_attack_multiplier = 0.90
            away_attack_multiplier = 1.15
            home_defense_multiplier = 0.85
            away_defense_multiplier = 1.10
        elif strength_ratio > 1.2:  # Home team stronger
            home_attack_multiplier = 1.15
            away_attack_multiplier = 0.85
            home_defense_multiplier = 1.10
            away_defense_multiplier = 0.90
        else:  # Even matchup
            home_attack_multiplier = 1.05
            away_attack_multiplier = 0.95
            home_defense_multiplier = 1.05
            away_defense_multiplier = 0.95
        
        return {
            'home_attack_multiplier': home_attack_multiplier,
            'away_attack_multiplier': away_attack_multiplier,
            'home_defense_multiplier': home_defense_multiplier,
            'away_defense_multiplier': away_defense_multiplier,
            'quality_gap_score': quality_gap_score,
            'strength_ratio': strength_ratio
        }
    
    def _professional_form_weight(self, form_data: List[float], is_home: bool) -> float:
        """Professional form weighting with home/away differentiation"""
        if not form_data:
            return 1.5 if is_home else 1.2
            
        weights = np.array([0.85**i for i in range(len(form_data)-1, -1, -1)])
        weighted_avg = np.average(form_data, weights=weights)
        
        if is_home:
            return weighted_avg * 1.05
        else:
            return weighted_avg * 0.98
    
    def _professional_contextual_features(self, context: Dict, tier_adjustments: Dict) -> Dict[str, float]:
        """Professional contextual factor assessment"""
        features = {}
        
        injury_impact_map = {1: 0.02, 2: 0.06, 3: 0.12, 4: 0.20, 5: 0.30}
        home_injury_impact = injury_impact_map.get(context.get('home_injuries', 1), 0.05)
        away_injury_impact = injury_impact_map.get(context.get('away_injuries', 1), 0.05)
        
        motivation_map = {'Low': 0.88, 'Normal': 1.0, 'High': 1.08, 'Very High': 1.12}
        home_motivation = motivation_map.get(context.get('home_motivation', 'Normal'), 1.0)
        away_motivation = motivation_map.get(context.get('away_motivation', 'Normal'), 1.0)
        
        quality_gap = tier_adjustments.get('quality_gap_score', 1.0)
        motivation_amplifier = min(1.5, 1.0 + (quality_gap - 1.0) * 0.5)
        
        features.update({
            'home_injury_factor': 1.0 - home_injury_impact,
            'away_injury_factor': 1.0 - away_injury_impact,
            'home_motivation_factor': home_motivation * motivation_amplifier,
            'away_motivation_factor': away_motivation,
            'match_importance': context.get('match_importance', 0.5),
            'quality_gap_amplifier': motivation_amplifier,
        })
        
        return features

class ProfessionalMatchSimulator:
    """ENHANCED Monte Carlo Simulation with Realistic Dependencies"""
    
    def __init__(self, n_simulations: int = 25000):
        self.n_simulations = n_simulations
        
    def simulate_match_dixon_coles(self, home_xg: float, away_xg: float, correlation: float = 0.2):
        """Enhanced Dixon-Coles implementation"""
        goal_sum = home_xg + away_xg
        dynamic_correlation = correlation * min(1.0, goal_sum / 3.0)
        
        lambda1 = max(0.15, home_xg - dynamic_correlation * min(home_xg, away_xg))
        lambda2 = max(0.15, away_xg - dynamic_correlation * min(home_xg, away_xg))
        lambda3 = dynamic_correlation * min(home_xg, away_xg)
        
        C = np.random.poisson(lambda3, self.n_simulations)
        A = np.random.poisson(lambda1, self.n_simulations)
        B = np.random.poisson(lambda2, self.n_simulations)
        
        home_goals = A + C
        away_goals = B + C
        
        return home_goals, away_goals
    
    def get_market_probabilities(self, home_goals: np.array, away_goals: np.array) -> Dict[str, float]:
        """Enhanced market probability calculation"""
        
        btts_yes = np.mean((home_goals > 0) & (away_goals > 0))
        btts_no = 1 - btts_yes
        
        total_goals = home_goals + away_goals
        over_15 = np.mean(total_goals > 1.5)
        over_25 = np.mean(total_goals > 2.5)
        over_35 = np.mean(total_goals > 3.5)
        
        score_counts = {}
        for h, a in zip(home_goals[:5000], away_goals[:5000]):
            score = f"{h}-{a}"
            score_counts[score] = score_counts.get(score, 0) + 1
        
        exact_scores = {score: count/5000 
                       for score, count in sorted(score_counts.items(), 
                       key=lambda x: x[1], reverse=True)[:8]}
        
        return {
            'btts_yes': btts_yes,
            'btts_no': btts_no,
            'over_15': over_15,
            'over_25': over_25,
            'over_35': over_35,
            'under_25': 1 - over_25,
            'exact_scores': exact_scores,
        }

class ProfessionalLeagueCalibrator:
    """ENHANCED League-Specific Calibration"""
    
    def __init__(self):
        self.league_profiles = {
            'premier_league': {
                'goal_intensity': 'high', 'defensive_variance': 'medium', 
                'calibration_factor': 1.02, 'home_advantage': 0.38,
                'btts_baseline': 0.52, 'over_25_baseline': 0.51, 'tier_impact': 1.0,
                'confidence_multiplier': 1.0
            },
            'la_liga': {
                'goal_intensity': 'medium', 'defensive_variance': 'low', 
                'calibration_factor': 0.98, 'home_advantage': 0.32,
                'btts_baseline': 0.48, 'over_25_baseline': 0.47, 'tier_impact': 0.95,
                'confidence_multiplier': 0.95
            },
            'serie_a': {
                'goal_intensity': 'low', 'defensive_variance': 'low', 
                'calibration_factor': 0.94, 'home_advantage': 0.42,
                'btts_baseline': 0.45, 'over_25_baseline': 0.44, 'tier_impact': 1.15,
                'confidence_multiplier': 1.15
            },
            'bundesliga': {
                'goal_intensity': 'very_high', 'defensive_variance': 'high', 
                'calibration_factor': 1.08, 'home_advantage': 0.28,
                'btts_baseline': 0.55, 'over_25_baseline': 0.58, 'tier_impact': 0.9,
                'confidence_multiplier': 0.9
            },
            'ligue_1': {
                'goal_intensity': 'low', 'defensive_variance': 'medium', 
                'calibration_factor': 0.95, 'home_advantage': 0.34,
                'btts_baseline': 0.47, 'over_25_baseline': 0.46, 'tier_impact': 1.05,
                'confidence_multiplier': 1.05
            },
            'liga_portugal': {
                'goal_intensity': 'medium', 'defensive_variance': 'medium', 
                'calibration_factor': 0.97, 'home_advantage': 0.42,
                'btts_baseline': 0.49, 'over_25_baseline': 0.48, 'tier_impact': 1.1,
                'confidence_multiplier': 1.1
            },
            'brasileirao': {
                'goal_intensity': 'high', 'defensive_variance': 'high', 
                'calibration_factor': 1.02, 'home_advantage': 0.45,
                'btts_baseline': 0.51, 'over_25_baseline': 0.52, 'tier_impact': 0.95,
                'confidence_multiplier': 0.95
            },
            'liga_mx': {
                'goal_intensity': 'medium', 'defensive_variance': 'high', 
                'calibration_factor': 1.01, 'home_advantage': 0.40,
                'btts_baseline': 0.50, 'over_25_baseline': 0.49, 'tier_impact': 1.0,
                'confidence_multiplier': 1.0
            },
            'eredivisie': {
                'goal_intensity': 'high', 'defensive_variance': 'high', 
                'calibration_factor': 1.03, 'home_advantage': 0.30,
                'btts_baseline': 0.54, 'over_25_baseline': 0.56, 'tier_impact': 0.9,
                'confidence_multiplier': 0.9
            },
            'championship': {
                'goal_intensity': 'medium_high', 'defensive_variance': 'high', 
                'calibration_factor': 1.02, 'home_advantage': 0.40,
                'btts_baseline': 0.51, 'over_25_baseline': 0.49, 'tier_impact': 1.08,
                'confidence_multiplier': 1.08
            }
        }
    
    def calibrate_probability(self, raw_prob: float, league: str, market_type: str) -> float:
        """Enhanced probability calibration"""
        profile = self.league_profiles.get(league, self.league_profiles['premier_league'])
        base_calibrated = raw_prob * profile['calibration_factor']
        
        if market_type == 'over_25':
            if profile['goal_intensity'] == 'very_high':
                base_calibrated *= 1.08
            elif profile['goal_intensity'] == 'high':
                base_calibrated *= 1.04
            elif profile['goal_intensity'] == 'medium_high':
                base_calibrated *= 1.02
            elif profile['goal_intensity'] == 'low':
                base_calibrated *= 0.94
            elif profile['goal_intensity'] == 'very_low':
                base_calibrated *= 0.88
                
        elif market_type == 'btts_yes':
            if profile['defensive_variance'] == 'low':
                base_calibrated *= 0.92
            elif profile['defensive_variance'] == 'high':
                base_calibrated *= 1.06
                
        return np.clip(base_calibrated, 0.025, 0.975)
    
    def get_league_confidence_multiplier(self, league: str) -> float:
        """Get league-specific confidence adjustment"""
        profile = self.league_profiles.get(league, self.league_profiles['premier_league'])
        return profile.get('confidence_multiplier', 1.0)

class ProfessionalPredictionExplainer:
    """ENHANCED Explanation Engine with Outcome-Based Context"""
    
    def __init__(self):
        self.feature_descriptions = {
            'home_attack_vs_away_defense': 'Home attacking strength against away defense',
            'away_attack_vs_home_defense': 'Away attacking strength against home defense',
            'total_xg_expected': 'Expected total goals in match',
            'home_form_attack': 'Recent home attacking form',
            'away_form_attack': 'Recent away attacking form',
            'xg_difference': 'Difference in expected goals between teams',
            'quality_gap_metric': 'Quality difference between teams',
            'home_dominance_potential': 'Home team dominance potential'
        }
    
    def generate_outcome_explanations(self, context: str, probabilities: Dict, home_tier: str, away_tier: str) -> List[str]:
        """ENHANCED: Generate outcome-focused explanations for betting"""
        explanations = {
            'home_dominance': [
                f"🏠 HOME DOMINANCE CONTEXT: Expect comfortable home victory ({probabilities.get('home_win', 0):.1%} probability)",
                f"Superior {home_tier} quality and home advantage should lead to controlled performance",
                "Primary betting focus: Home Win, Home -1 Handicap, Under 3.5 goals"
            ],
            'away_counter': [
                f"✈️ AWAY COUNTER CONTEXT: Strong away win/upset potential ({probabilities.get('away_win', 0):.1%} probability)", 
                f"{away_tier} visitors' quality advantage can overcome home field disadvantage",
                "Primary betting focus: Away Win, Double Chance Away/Draw, BTTS Yes"
            ],
            'offensive_showdown': [
                f"🔥 OFFENSIVE SHOWDOWN CONTEXT: Expect high-scoring game (Over 2.5: {probabilities.get('over_25', 0):.1%})",
                "Both teams' attacking approach and defensive vulnerabilities suggest multiple goals",
                "Primary betting focus: Over 2.5 goals, BTTS Yes, Both Teams to Score & Over 2.5"
            ],
            'defensive_battle': [
                f"🛡️ DEFENSIVE BATTLE CONTEXT: Anticipate low-scoring affair (Under 2.5: {probabilities.get('under_25', 0):.1%})",
                "Organized defenses and cautious tactical approach should limit goal-scoring opportunities",
                "Primary betting focus: Under 2.5 goals, BTTS No, Under 1.5 goals"
            ],
            'tactical_stalemate': [
                f"⚔️ TACTICAL STALEMATE CONTEXT: High draw probability expected ({probabilities.get('draw', 0):.1%})",
                "Evenly matched teams with organized approaches likely to cancel each other out",
                "Primary betting focus: Draw, Under 2.5 goals, 0-0 or 1-1 Correct Score"
            ],
            'balanced': [
                "⚖️ BALANCED CONTEXT: No strong outcome bias detected",
                "Match could swing either way based on key moments and individual quality",
                "Focus on value bets with positive expected value across all markets"
            ]
        }
        return explanations.get(context, ["Context analysis in progress..."])
    
    def generate_explanation(self, features: Dict, probabilities: Dict, narrative: Dict, home_tier: str, away_tier: str) -> Dict[str, List[str]]:
        """Generate enhanced explanations with outcome focus"""
        
        explanations = {}
        
        home_attack = features.get('home_xg_for', 1.5)
        away_attack = features.get('away_xg_for', 1.2)
        home_defense = features.get('home_xg_against', 1.3)
        away_defense = features.get('away_xg_against', 1.4)
        total_xg = features.get('total_xg_expected', 2.7)
        quality_gap = features.get('quality_gap_metric', 1.0)
        
        btts_prob = probabilities.get('btts_yes', 0.5)
        
        # ENHANCED: Outcome-based BTTS explanations
        context = narrative.get('expected_outcome', 'balanced')
        if context == 'offensive_showdown':
            explanations['btts'] = [
                f"🔥 OFFENSIVE CONTEXT: High BTTS probability ({btts_prob:.1%}) expected",
                "Both teams possess strong attacking capabilities and defensive vulnerabilities",
                f"Home attack: {home_attack:.1f}xG vs Away defense: {away_defense:.1f}xGA"
            ]
        elif context == 'defensive_battle':
            explanations['btts'] = [
                f"🛡️ DEFENSIVE CONTEXT: Low BTTS probability ({btts_prob:.1%}) expected",
                "Organized defensive structures from both teams should limit scoring chances",
                f"Defensive solidity suggests clean sheet potential for one or both teams"
            ]
        else:
            if btts_prob > 0.65:
                explanations['btts'] = [
                    f"Strong attacking capabilities from both teams (Home: {home_attack:.1f}, Away: {away_attack:.1f} xG)",
                    f"Defensive vulnerabilities suggest high BTTS probability ({btts_prob:.1%})"
                ]
            elif btts_prob < 0.35:
                explanations['btts'] = [
                    f"Defensive organization likely to limit scoring opportunities",
                    f"One team may struggle to find the net (BTTS: {btts_prob:.1%})"
                ]
            else:
                explanations['btts'] = [
                    f"Balanced attacking and defensive capabilities",
                    f"Moderate BTTS probability ({btts_prob:.1%}) reflects evenly matched teams"
                ]
        
        over_prob = probabilities.get('over_25', 0.5)
        # ENHANCED: Context-based over/under explanations
        if context == 'offensive_showdown':
            explanations['over_under'] = [
                f"🔥 OFFENSIVE CONTEXT: High goal expectation (Total xG: {total_xg:.2f})",
                f"Multiple goals expected from open attacking play (Over 2.5: {over_prob:.1%})",
                "Both teams likely to contribute to goal tally"
            ]
        elif context == 'defensive_battle':
            explanations['over_under'] = [
                f"🛡️ DEFENSIVE CONTEXT: Low goal expectation (Total xG: {total_xg:.2f})",
                f"Cautious approach should limit total goals (Under 2.5: {1-over_prob:.1%})",
                "Set-pieces and individual moments may prove decisive"
            ]
        elif context == 'tactical_stalemate':
            explanations['over_under'] = [
                f"⚔️ TACTICAL CONTEXT: Moderate goal expectation",
                f"Organized defenses may limit scoring (Under 2.5: {1-over_prob:.1%})",
                "Game management could keep total goals low"
            ]
        else:
            if over_prob > 0.65:
                explanations['over_under'] = [
                    f"High expected goal volume (Total xG: {total_xg:.2f})",
                    f"Attacking styles suggest multiple goals (Over 2.5: {over_prob:.1%})"
                ]
            elif over_prob < 0.35:
                explanations['over_under'] = [
                    f"Defensive organization likely to limit scoring",
                    f"Expected total goals: {total_xg:.2f}"
                ]
            else:
                explanations['over_under'] = [
                    f"Average expected goal volume (Total xG: {total_xg:.2f})",
                    f"Game could go either way in terms of total goals"
                ]
            
        style_conflict = narrative.get('style_conflict', 'balanced')
        quality_gap_level = narrative.get('quality_gap', 'even')
        
        if quality_gap_level in ['significant', 'extreme']:
            explanations['quality'] = [f"Significant quality gap between {home_tier} and {away_tier} teams"]
            
        if style_conflict == "attacking_vs_attacking":
            explanations['style'] = ["Open game expected with both teams prioritizing attack"]
        elif style_conflict == "attacking_vs_defensive":
            explanations['style'] = ["Tactical battle between attacking initiative and defensive organization"]
        else:
            explanations['style'] = ["Balanced tactical approach from both teams"]
            
        # ENHANCED: Add context-specific explanations
        context_explanations = self.generate_outcome_explanations(context, probabilities, home_tier, away_tier)
        explanations['context'] = context_explanations
            
        return explanations

class ProfessionalTeamTierCalibrator:
    """ENHANCED Team Tier Calibration"""
    
    def __init__(self):
        self.league_baselines = {
            'premier_league': {'avg_goals': 2.8, 'home_advantage': 0.38, 'btts_rate': 0.52},
            'la_liga': {'avg_goals': 2.6, 'home_advantage': 0.32, 'btts_rate': 0.48},
            'serie_a': {'avg_goals': 2.7, 'home_advantage': 0.42, 'btts_rate': 0.45},
            'bundesliga': {'avg_goals': 3.1, 'home_advantage': 0.28, 'btts_rate': 0.55},
            'ligue_1': {'avg_goals': 2.5, 'home_advantage': 0.34, 'btts_rate': 0.47},
            'liga_portugal': {'avg_goals': 2.6, 'home_advantage': 0.42, 'btts_rate': 0.49},
            'brasileirao': {'avg_goals': 2.4, 'home_advantage': 0.45, 'btts_rate': 0.51},
            'liga_mx': {'avg_goals': 2.7, 'home_advantage': 0.40, 'btts_rate': 0.50},
            'eredivisie': {'avg_goals': 3.0, 'home_advantage': 0.30, 'btts_rate': 0.54},
            'championship': {'avg_goals': 2.6, 'home_advantage': 0.40, 'btts_rate': 0.51},
        }
        
        self.team_databases = {
            'premier_league': {
                'Arsenal': 'ELITE', 'Manchester City': 'ELITE', 'Liverpool': 'ELITE',
                'Sunderland': 'MEDIUM', 'Bournemouth': 'MEDIUM', 'Tottenham Hotspur': 'STRONG',
                'Chelsea': 'STRONG', 'Manchester United': 'STRONG', 'Crystal Palace': 'MEDIUM',
                'Brighton & Hove Albion': 'MEDIUM', 'Aston Villa': 'STRONG', 'Brentford': 'MEDIUM',
                'Newcastle United': 'STRONG', 'Everton': 'MEDIUM', 'Fulham': 'MEDIUM',
                'Leeds United': 'MEDIUM', 'Burnley': 'WEAK', 'West Ham United': 'STRONG',
                'Nottingham Forest': 'MEDIUM', 'Wolverhampton': 'MEDIUM'
            },
            'la_liga': {
                'Real Madrid': 'ELITE', 'Barcelona': 'ELITE', 'Villarreal': 'STRONG',
                'Atlético Madrid': 'ELITE', 'Real Betis': 'MEDIUM', 'Espanyol': 'MEDIUM',
                'Getafe': 'MEDIUM', 'Deportivo Alavés': 'WEAK', 'Elche': 'WEAK',
                'Rayo Vallecano': 'MEDIUM', 'Athletic Club': 'STRONG', 'Celta Vigo': 'MEDIUM',
                'Sevilla': 'STRONG', 'Real Sociedad': 'STRONG', 'Osasuna': 'MEDIUM',
                'Levante UD': 'WEAK', 'Mallorca': 'MEDIUM', 'Valencia': 'MEDIUM',
                'Real Oviedo': 'WEAK', 'Girona FC': 'MEDIUM'
            },
            'serie_a': {
                'Napoli': 'ELITE', 'Inter': 'ELITE', 'Milan': 'ELITE',
                'Roma': 'STRONG', 'Bologna': 'MEDIUM', 'Juventus': 'ELITE',
                'Como': 'WEAK', 'Lazio': 'STRONG', 'Udinese': 'MEDIUM',
                'Cremonese': 'WEAK', 'Atalanta': 'STRONG', 'Sassuolo': 'MEDIUM',
                'Torino': 'MEDIUM', 'Cagliari': 'WEAK', 'Lecce': 'WEAK',
                'Parma': 'MEDIUM', 'Pisa': 'WEAK', 'Genoa': 'MEDIUM',
                'Hellas Verona': 'WEAK', 'Fiorentina': 'MEDIUM'
            },
            'bundesliga': {
                'FC Bayern München': 'ELITE', 'RB Leipzig': 'ELITE', 'Borussia Dortmund': 'ELITE',
                'VfB Stuttgart': 'STRONG', 'Bayer 04 Leverkusen': 'STRONG', 'TSG Hoffenheim': 'MEDIUM',
                'FC Köln': 'MEDIUM', 'Eintracht Frankfurt': 'STRONG', 'SV Werder Bremen': 'MEDIUM',
                'FC Union Berlin': 'MEDIUM', 'SC Freiburg': 'STRONG', 'VfL Wolfsburg': 'MEDIUM',
                'Hamburger SV': 'MEDIUM', 'FC Augsburg': 'MEDIUM', 'FC St. Pauli': 'MEDIUM',
                'Borussia M\'gladbach': 'MEDIUM', 'FSV Mainz 05': 'MEDIUM', 'FC Heidenheim': 'WEAK'
            },
            'ligue_1': {
                'Paris Saint-Germain': 'ELITE', 'Olympique de Marseille': 'STRONG', 'RC Lens': 'STRONG',
                'Lille': 'STRONG', 'AS Monaco': 'STRONG', 'Olympique Lyonnais': 'STRONG',
                'RC Strasbourg': 'MEDIUM', 'Nice': 'MEDIUM', 'Toulouse': 'MEDIUM',
                'Stade Rennais': 'STRONG', 'Paris FC': 'MEDIUM', 'Le Havre': 'MEDIUM',
                'Stade Brestois': 'MEDIUM', 'Angers': 'WEAK', 'Nantes': 'MEDIUM',
                'Lorient': 'MEDIUM', 'Metz': 'WEAK', 'Auxerre': 'WEAK'
            },
            'liga_portugal': {
                'Benfica': 'ELITE', 'Porto': 'ELITE', 'Sporting CP': 'ELITE',
                'Braga': 'STRONG', 'Vitoria Guimaraes': 'STRONG', 'Famalicao': 'MEDIUM',
                'Casa Pia': 'MEDIUM', 'Rio Ave': 'MEDIUM', 'Estoril': 'MEDIUM',
                'Gil Vicente': 'MEDIUM', 'Arouca': 'MEDIUM', 'Chaves': 'MEDIUM',
                'Portimonense': 'MEDIUM', 'Boavista': 'MEDIUM', 'Vizela': 'WEAK',
                'Estrela Amadora': 'WEAK', 'Farense': 'WEAK'
            },
            'brasileirao': {
                'Flamengo': 'ELITE', 'Palmeiras': 'ELITE', 'Sao Paulo': 'ELITE',
                'Atletico Mineiro': 'STRONG', 'Gremio': 'STRONG', 'Fluminense': 'STRONG',
                'Botafogo': 'STRONG', 'Corinthians': 'STRONG', 'Internacional': 'STRONG',
                'Fortaleza': 'MEDIUM', 'Cruzeiro': 'MEDIUM', 'Bahia': 'MEDIUM',
                'Vasco da Gama': 'MEDIUM', 'Bragantino': 'MEDIUM', 'Athletico Paranaense': 'MEDIUM',
                'Santos': 'MEDIUM', 'Cuiaba': 'WEAK', 'Goias': 'WEAK',
                'Coritiba': 'WEAK', 'America MG': 'WEAK'
            },
            'liga_mx': {
                'Cruz Azul': 'ELITE', 'CD Toluca': 'STRONG', 'Club América': 'ELITE',
                'Tigres UANL': 'ELITE', 'CF Monterrey': 'ELITE', 'CD Guadalajara': 'STRONG',
                'FC Juárez': 'MEDIUM', 'CF Pachuca': 'MEDIUM', 'Club Tijuana': 'MEDIUM',
                'Pumas UNAM': 'STRONG', 'Santos Laguna': 'MEDIUM', 'Atlas FC': 'MEDIUM',
                'Querétaro FC': 'MEDIUM', 'Atlético San Luis': 'MEDIUM', 'Club Necaxa': 'MEDIUM',
                'Mazatlán FC': 'WEAK', 'Club León': 'MEDIUM', 'Club Puebla': 'WEAK'
            },
            'eredivisie': {
                'Ajax': 'ELITE', 'PSV': 'ELITE', 'Feyenoord': 'ELITE',
                'AZ Alkmaar': 'STRONG', 'Twente': 'STRONG', 'Sparta Rotterdam': 'MEDIUM',
                'Heerenveen': 'MEDIUM', 'NEC Nijmegen': 'MEDIUM', 'Utrecht': 'MEDIUM',
                'Go Ahead Eagles': 'MEDIUM', 'Fortuna Sittard': 'MEDIUM', 'Heracles': 'MEDIUM',
                'Almere City': 'MEDIUM', 'Excelsior': 'WEAK', 'RKC Waalwijk': 'WEAK',
                'Volendam': 'WEAK', 'Vitesse': 'WEAK'
            },
            'championship': {
                'Leicester City': 'STRONG', 'Southampton': 'STRONG', 'Leeds United': 'STRONG',
                'West Brom': 'STRONG', 'Norwich City': 'STRONG', 'Middlesbrough': 'MEDIUM',
                'Stoke City': 'MEDIUM', 'Watford': 'MEDIUM', 'Swansea City': 'MEDIUM',
                'Coventry City': 'MEDIUM', 'Hull City': 'MEDIUM', 'Queens Park Rangers': 'MEDIUM',
                'Blackburn Rovers': 'MEDIUM', 'Millwall': 'WEAK', 'Bristol City': 'WEAK',
                'Preston North End': 'WEAK', 'Birmingham City': 'WEAK', 'Sheffield Wednesday': 'WEAK',
                'Wrexham': 'WEAK', 'Oxford United': 'WEAK', 'Derby County': 'WEAK',
                'Portsmouth': 'WEAK', 'Charlton Athletic': 'WEAK', 'Ipswich Town': 'WEAK',
                'Sheffield United': 'STRONG'
            }
        }
    
    def get_league_teams(self, league: str) -> List[str]:
        """Get teams for a league"""
        return list(self.team_databases.get(league, {}).keys())
    
    def get_team_tier(self, team: str, league: str) -> str:
        """Get professional team tier assessment"""
        league_teams = self.team_databases.get(league, {})
        return league_teams.get(team, 'MEDIUM')

class ProfessionalGoalModel:
    """ENHANCED Goal Prediction Model"""
    
    def __init__(self):
        self.feature_engine = EliteFeatureEngine()
        self.simulator = ProfessionalMatchSimulator()
        self.calibrator = ProfessionalLeagueCalibrator()
        self.explainer = ProfessionalPredictionExplainer()
        self.tier_calibrator = ProfessionalTeamTierCalibrator()
        
    def calculate_team_strength(self, team_data: Dict, is_home: bool = True) -> Dict[str, float]:
        """Enhanced team strength calculation"""
        prior_games = 8
        prior_strength = 1.5

        xg_data = team_data.get('xg_last_6', [1.5] * 6)
        xga_data = team_data.get('xga_last_6', [1.3] * 6)
        
        weights = np.array([0.8**i for i in range(5, -1, -1)])
        recent_xg = np.average(xg_data, weights=weights)
        recent_xga = np.average(xga_data, weights=weights)
        
        games_played = len(team_data.get('xg_season', [])) or 6
        shrinkage_factor = min(0.7, games_played / (games_played + prior_games))
        
        strength_attack = recent_xg * shrinkage_factor + prior_strength * (1 - shrinkage_factor)
        strength_defense = recent_xga * shrinkage_factor + prior_strength * (1 - shrinkage_factor)
        
        return {
            'attack': strength_attack,
            'defense': strength_defense,
            'sample_size': games_played,
            'form_confidence': shrinkage_factor
        }

class ApexProfessionalEngine:
    """APEX ENHANCED ENGINE - Outcome-Based Money-Grade Predictions"""
    
    def __init__(self, match_data: Dict[str, Any]):
        self.data = self._professional_data_validation(match_data)
        self.calibrator = ProfessionalTeamTierCalibrator()
        self.goal_model = ProfessionalGoalModel()
        self.narrative = MatchNarrative()
        self.intelligence = IntelligenceMetrics(
            narrative_coherence=0.0, prediction_alignment="LOW", 
            data_quality_score=0.0, certainty_score=0.0,
            market_edge_score=0.0, risk_level="HIGH", football_iq_score=0.0,
            calibration_status="PENDING", context_confidence=0.0  # NEW
        )
        self._setup_professional_parameters()
        
    def _professional_data_validation(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Professional data validation and enhancement"""
        enhanced_data = match_data.copy()
        
        required_fields = ['home_team', 'away_team', 'league']
        for field in required_fields:
            if field not in enhanced_data:
                enhanced_data[field] = 'Unknown'
                logger.warning(f"Missing required field: {field}")
        
        predictive_fields = {
            'home_goals': (0, 30, 8), 'away_goals': (0, 30, 8),
            'home_conceded': (0, 30, 8), 'away_conceded': (0, 30, 8),
            'home_goals_home': (0, 15, 4), 'away_goals_away': (0, 15, 4),
        }
        
        for field, (min_val, max_val, default) in predictive_fields.items():
            if field in enhanced_data:
                try:
                    value = float(enhanced_data[field])
                    enhanced_data[field] = max(min_val, min(value, max_val))
                except (TypeError, ValueError):
                    enhanced_data[field] = default
                    logger.warning(f"Invalid value for {field}, using default: {default}")
            else:
                enhanced_data[field] = default
        
        for form_field in ['home_form', 'away_form']:
            if form_field in enhanced_data and enhanced_data[form_field]:
                try:
                    if isinstance(enhanced_data[form_field], list):
                        form_data = enhanced_data[form_field]
                        if len(form_data) < 6:
                            avg_form = np.mean(form_data) if form_data else 1.5
                            form_data.extend([avg_form] * (6 - len(form_data)))
                        enhanced_data[form_field] = form_data[:6]
                    else:
                        enhanced_data[form_field] = [1.5] * 6
                except (TypeError, ValueError):
                    enhanced_data[form_field] = [1.5] * 6
            else:
                enhanced_data[form_field] = [1.5] * 6
        
        if 'h2h_data' not in enhanced_data:
            enhanced_data['h2h_data'] = {
                'matches': 0, 'home_wins': 0, 'away_wins': 0, 'draws': 0,
                'home_goals': 0, 'away_goals': 0
            }
            
        if 'motivation' not in enhanced_data:
            enhanced_data['motivation'] = {'home': 'Normal', 'away': 'Normal'}
            
        if 'injuries' not in enhanced_data:
            enhanced_data['injuries'] = {'home': 1, 'away': 1}
            
        logger.info(f"Data validation complete for {enhanced_data['home_team']} vs {enhanced_data['away_team']}")
        return enhanced_data

    def _setup_professional_parameters(self):
        """Professional parameter setup"""
        self.calibration_params = {
            'form_decay_rate': 0.80,
            'h2h_weight': 0.18,
            'injury_impact': 0.10,
            'motivation_impact': 0.12,
            'defensive_impact_multiplier': 0.45,
            'tier_impact_base': 0.15,
        }

    def _calculate_professional_xg(self) -> Tuple[float, float]:
        """ENHANCED xG calculation with outcome-based calibration"""
        league = self.data.get('league', 'premier_league')
        league_baseline = self.calibrator.league_baselines.get(league, {'avg_goals': 2.7, 'home_advantage': 0.35})
        
        home_tier = self.calibrator.get_team_tier(self.data['home_team'], league)
        away_tier = self.calibrator.get_team_tier(self.data['away_team'], league)
        
        home_team_data = {
            'xg_home': self.data.get('home_goals_home', 4) / 3.0,
            'xga_home': self.data.get('home_conceded', 8) / 6.0,
            'xg_last_6': self.data.get('home_form', [1.5]*6),
            'xga_last_6': [self.data.get('home_conceded', 8)/6.0] * 6
        }
        
        away_team_data = {
            'xg_away': self.data.get('away_goals_away', 4) / 3.0,
            'xga_away': self.data.get('away_conceded', 8) / 6.0,
            'xg_last_6': self.data.get('away_form', [1.5]*6),
            'xga_last_6': [self.data.get('away_conceded', 8)/6.0] * 6
        }
        
        home_strength = self.goal_model.calculate_team_strength(home_team_data, is_home=True)
        away_strength = self.goal_model.calculate_team_strength(away_team_data, is_home=False)
        
        home_xg = home_strength['attack'] * (1 - (away_strength['defense'] / league_baseline['avg_goals']) * 0.35)
        away_xg = away_strength['attack'] * (1 - (home_strength['defense'] / league_baseline['avg_goals']) * 0.35)
        
        home_advantage = league_baseline['home_advantage']
        home_xg *= (1 + home_advantage)
        
        context = {
            'home_injuries': self.data.get('injuries', {}).get('home', 1),
            'away_injuries': self.data.get('injuries', {}).get('away', 1),
            'home_motivation': self.data.get('motivation', {}).get('home', 'Normal'),
            'away_motivation': self.data.get('motivation', {}).get('away', 'Normal'),
            'match_importance': 0.5,
            'league': league
        }
        
        features = self.goal_model.feature_engine.create_match_features(
            home_team_data, away_team_data, context, home_tier, away_tier
        )
        
        home_xg *= features.get('home_injury_factor', 1.0) * features.get('home_motivation_factor', 1.0)
        away_xg *= features.get('away_injury_factor', 1.0) * features.get('away_motivation_factor', 1.0)
        
        h2h_data = self.data.get('h2h_data', {})
        if h2h_data and h2h_data.get('matches', 0) >= 2:
            home_xg, away_xg = self._apply_enhanced_h2h_adjustment(home_xg, away_xg, h2h_data, home_tier, away_tier)
        
        # ENHANCED: More conservative away counter adjustments
        xg_difference = home_xg - away_xg
        if home_tier in ['WEAK', 'MEDIUM'] and away_tier in ['STRONG', 'ELITE'] and xg_difference < -0.2:
            # Away team significantly stronger AND clear xG edge - apply conservative boost
            away_xg *= 1.08
            home_xg *= 0.95
            logger.info(f"Applied ENHANCED away counter boost: {self.data['away_team']} xG increased by 8%")
        
        home_xg = max(0.25, min(3.5, home_xg))
        away_xg = max(0.25, min(3.0, away_xg))
        
        logger.info(f"Enhanced xG calculated: {self.data['home_team']} {home_xg:.2f} - {self.data['away_team']} {away_xg:.2f}")
        return round(home_xg, 3), round(away_xg, 3)

    def _apply_enhanced_h2h_adjustment(self, home_xg: float, away_xg: float, h2h_data: Dict, home_tier: str, away_tier: str) -> Tuple[float, float]:
        """Enhanced H2H adjustment"""
        matches = h2h_data.get('matches', 0)
        if matches < 2:
            return home_xg, away_xg
        
        h2h_weight = min(0.30, matches * 0.08)
        
        h2h_home_avg = h2h_data.get('home_goals', 0) / matches
        h2h_away_avg = h2h_data.get('away_goals', 0) / matches
        
        if h2h_home_avg > 0 or h2h_away_avg > 0:
            tier_strength = {'ELITE': 1.4, 'STRONG': 1.15, 'MEDIUM': 1.0, 'WEAK': 0.8}
            home_tier_strength = tier_strength.get(home_tier, 1.0)
            away_tier_strength = tier_strength.get(away_tier, 1.0)
            
            # ENHANCED: More conservative H2H weighting
            if h2h_home_avg > h2h_away_avg + 0.5:  # Clear home dominance
                h2h_weight = min(0.30, h2h_weight * 1.1)
            elif h2h_away_avg > h2h_home_avg + 0.5:  # Clear away dominance
                h2h_weight = min(0.30, h2h_weight * 1.1)
            
            adjusted_home_xg = (home_xg * (1 - h2h_weight)) + (h2h_home_avg * h2h_weight * home_tier_strength)
            adjusted_away_xg = (away_xg * (1 - h2h_weight)) + (h2h_away_avg * h2h_weight * away_tier_strength)
            
            return adjusted_home_xg, adjusted_away_xg
        
        return home_xg, away_xg

    def _determine_outcome_based_context(self, home_xg: float, away_xg: float, narrative: MatchNarrative) -> str:
        """ENHANCED: Outcome-based context determination"""
        total_xg = home_xg + away_xg
        xg_difference = home_xg - away_xg
        
        # Get defensive metrics
        home_defense = self.data.get('home_conceded', 8) / 6.0
        away_defense = self.data.get('away_conceded', 8) / 6.0
        avg_defense = (home_defense + away_defense) / 2
        
        # ENHANCED: Outcome-based context determination
        if xg_difference >= 0.35 and narrative.quality_gap in ['significant', 'extreme']:
            return "home_dominance"  # Expect comfortable home win
        
        elif xg_difference <= -0.35 and narrative.quality_gap in ['significant', 'extreme']:
            return "away_counter"  # Expect away win/upset
        
        elif total_xg > 3.2 and avg_defense > 1.3:
            return "offensive_showdown"  # Expect high-scoring game
        
        elif total_xg < 2.2 and avg_defense < 0.9:
            return "defensive_battle"  # Expect low-scoring game
        
        elif abs(xg_difference) < 0.2 and total_xg < 2.8:
            return "tactical_stalemate"  # Expect draw
        
        else:
            return "balanced"

    def _determine_professional_narrative(self, home_xg: float, away_xg: float) -> MatchNarrative:
        """ENHANCED match narrative determination with outcome-based context"""
        narrative = MatchNarrative()
        total_xg = home_xg + away_xg
        xg_difference = home_xg - away_xg
        
        league = self.data.get('league', 'premier_league')
        home_tier = self.calibrator.get_team_tier(self.data['home_team'], league)
        away_tier = self.calibrator.get_team_tier(self.data['away_team'], league)
        
        tier_strength = {'ELITE': 4, 'STRONG': 3, 'MEDIUM': 2, 'WEAK': 1}
        home_strength = tier_strength.get(home_tier, 2)
        away_strength = tier_strength.get(away_tier, 2)
        
        strength_difference = home_strength - away_strength
        
        # Calculate form gap (recent form difference)
        home_form = np.mean(self.data.get('home_form', [1.5]*6))
        away_form = np.mean(self.data.get('away_form', [1.5]*6))
        form_gap = home_form - away_form
        
        # ENHANCED: Improved context detection with dynamic thresholds
        if xg_difference >= 0.35 and form_gap >= 0.25:
            narrative.quality_gap = "significant"
            narrative.dominance = "home"
            narrative.primary_pattern = "home_dominance"
        elif xg_difference <= -0.35 and form_gap <= -0.25:
            narrative.quality_gap = "significant"
            narrative.dominance = "away"
            narrative.primary_pattern = "away_counter"
        elif abs(xg_difference) < 0.15 and abs(form_gap) < 0.15:
            # Near parity - use tempo/defense to refine
            home_defense = self.data.get('home_conceded', 8) / 6.0
            away_defense = self.data.get('away_conceded', 8) / 6.0
            avg_defense = (home_defense + away_defense) / 2
            
            if total_xg > 3.0 and avg_defense > 1.2:
                narrative.primary_pattern = "offensive_showdown"
                narrative.style_conflict = "attacking_vs_attacking"
                narrative.expected_tempo = "high"
            elif total_xg < 2.0 and avg_defense < 0.8:
                narrative.primary_pattern = "defensive_battle"
                narrative.style_conflict = "defensive_vs_defensive"
                narrative.expected_tempo = "low"
            else:
                narrative.primary_pattern = "balanced"
                narrative.style_conflict = "balanced"
        else:
            # Transitional area between biases
            if total_xg < 2.5:
                narrative.primary_pattern = "tactical_stalemate"
                narrative.expected_tempo = "medium"
            else:
                narrative.primary_pattern = "balanced"
                narrative.expected_tempo = "medium"
        
        # Set quality gap based on strength difference
        if abs(strength_difference) >= 2:
            narrative.quality_gap = "extreme"
        elif abs(strength_difference) >= 1:
            narrative.quality_gap = "significant"
        else:
            narrative.quality_gap = "even"
        
        # Set defensive stability
        home_defense = self.data.get('home_conceded', 8) / 6.0
        away_defense = self.data.get('away_conceded', 8) / 6.0
        avg_conceded = (home_defense + away_defense) / 2
        
        if avg_conceded < 0.8:
            narrative.defensive_stability = "solid"
        elif avg_conceded > 1.5:
            narrative.defensive_stability = "leaky" 
        else:
            narrative.defensive_stability = "mixed"
        
        # ENHANCED: Determine outcome-based context
        narrative.expected_outcome = self._determine_outcome_based_context(home_xg, away_xg, narrative)
        
        # ENHANCED: Set betting priorities based on context
        betting_priorities = {
            'home_dominance': ['1x2 Home', 'Home -1 Handicap', 'Under 3.5 Goals'],
            'away_counter': ['1x2 Away', 'Double Chance Away/Draw', 'BTTS Yes'],
            'offensive_showdown': ['Over 2.5 Goals', 'BTTS Yes', 'Both Teams to Score & Over 2.5'],
            'defensive_battle': ['Under 2.5 Goals', 'BTTS No', 'Under 1.5 Goals'],
            'tactical_stalemate': ['1x2 Draw', 'Under 2.5 Goals', 'Correct Score 0-0 or 1-1'],
            'balanced': ['Value Bets Only', 'Small Stakes', 'Multiple Markets']
        }
        narrative.betting_priority = betting_priorities.get(narrative.expected_outcome, [])
            
        return narrative

    def _run_professional_monte_carlo(self, home_xg: float, away_xg: float, iterations: int = 25000) -> MonteCarloResults:
        """Enhanced Monte Carlo simulation"""
        home_goals, away_goals = self.goal_model.simulator.simulate_match_dixon_coles(home_xg, away_xg)
        market_probs = self.goal_model.simulator.get_market_probabilities(home_goals, away_goals)
        
        home_wins = np.mean(home_goals > away_goals)
        draws = np.mean(home_goals == away_goals)
        away_wins = np.mean(home_goals < away_goals)
        
        return MonteCarloResults(
            home_win_prob=home_wins, 
            draw_prob=draws, 
            away_win_prob=away_wins,
            over_25_prob=market_probs['over_25'], 
            btts_prob=market_probs['btts_yes'], 
            exact_scores=market_probs['exact_scores'],
            confidence_intervals={},
            probability_volatility={}
        )

    def _calculate_professional_data_quality(self) -> float:
        """Enhanced data quality assessment"""
        score = 0
        max_score = 100
        
        if self.data.get('home_team') and self.data.get('away_team') and self.data.get('home_team') != 'Unknown' and self.data.get('away_team') != 'Unknown':
            score += 20
        
        home_goals = self.data.get('home_goals', 0)
        away_goals = self.data.get('away_goals', 0)
        if home_goals > 0 and away_goals > 0:
            score += 25
        elif home_goals > 0 or away_goals > 0:
            score += 15
        
        home_form = self.data.get('home_form', [])
        away_form = self.data.get('away_form', [])
        if len(home_form) >= 5 and len(away_form) >= 5:
            score += 25
        elif len(home_form) >= 3 or len(away_form) >= 3:
            score += 15
        
        h2h_data = self.data.get('h2h_data', {})
        if h2h_data.get('matches', 0) >= 4:
            score += 20
        elif h2h_data.get('matches', 0) >= 2:
            score += 10
        
        if self.data.get('motivation') and self.data.get('injuries'):
            score += 10
        
        return min(100, score)

    def _assess_professional_coherence(self, predictions: Dict) -> Tuple[float, str]:
        """Enhanced prediction coherence assessment"""
        btts_yes = predictions.get('btts_yes', 0.5)
        over_25 = predictions.get('over_25', 0.5)
        home_win = predictions.get('home_win', 0.33)
        
        coherence_score = 0.0
        
        # ENHANCED: More nuanced contradiction detection
        if btts_yes > 0.7 and over_25 < 0.4:
            coherence_score -= 0.2
        elif btts_yes < 0.3 and over_25 > 0.7:
            coherence_score -= 0.2
        else:
            coherence_score += 0.3
            
        if home_win > 0.65 and over_25 < 0.35:
            coherence_score -= 0.15
        elif home_win < 0.25 and over_25 > 0.65:
            coherence_score -= 0.15
        else:
            coherence_score += 0.2
            
        narrative = self.narrative.to_dict()
        if narrative.get('expected_outcome') == 'home_dominance' and home_win < 0.45:
            coherence_score -= 0.1
        elif narrative.get('expected_outcome') == 'away_counter' and home_win > 0.55:
            coherence_score -= 0.1
        else:
            coherence_score += 0.2
            
        if coherence_score >= 0.5:
            alignment = "HIGH"
        elif coherence_score >= 0.2:
            alignment = "MEDIUM"
        else:
            alignment = "LOW"
            
        return max(0.0, min(1.0, 0.5 + coherence_score)), alignment

    def _calculate_professional_risk(self, certainty: float, data_quality: float, 
                                   market_edge: float, alignment: str) -> str:
        """Enhanced risk assessment"""
        base_risk = (1 - certainty) * 0.4 + (1 - data_quality/100) * 0.3 + (1 - market_edge) * 0.3
        
        alignment_penalty = {
            "HIGH": 0.0, "MEDIUM": 0.08, "LOW": 0.15
        }.get(alignment, 0.10)
        
        total_risk = base_risk + alignment_penalty
        
        if total_risk < 0.25:
            return "LOW"
        elif total_risk < 0.45:
            return "MEDIUM"
        elif total_risk < 0.65:
            return "HIGH"
        else:
            return "VERY_HIGH"
    
    def _generate_professional_summary(self, narrative: MatchNarrative, predictions: Dict, 
                                    home_team: str, away_team: str, home_tier: str, away_tier: str) -> str:
        """Enhanced match summary with outcome focus"""
        home_win = predictions.get('home_win', 0.33)
        btts_yes = predictions.get('btts_yes', 0.5)
        over_25 = predictions.get('over_25', 0.5)
        
        quality_gap = narrative.quality_gap
        expected_outcome = narrative.expected_outcome
        
        # ENHANCED: Outcome-focused summaries
        if expected_outcome == "away_counter":
            if home_win < 0.4:
                return f"🎯 AWAY COUNTER CONTEXT: {away_team} ({away_tier}) are favored despite being away at {home_team} ({home_tier}). The significant quality gap suggests the visitors should secure a positive result, with an away win being the most probable outcome. Expect {away_team} to control the game and capitalize on counter-attacking opportunities."
            else:
                return f"🎯 AWAY COUNTER CONTEXT: {away_team} ({away_tier}) possess clear tactical advantages at {home_team} ({home_tier}). Despite home advantage, the visitors' superior quality makes them dangerous underdogs with excellent chances of avoiding defeat. Look for {away_team} to dominate possession and create quality chances."
                
        elif expected_outcome == "home_dominance":
            if home_win > 0.65:
                return f"🎯 HOME DOMINANCE CONTEXT: {home_team} ({home_tier}) are expected to dominate this match against {away_team} ({away_tier}). Home advantage combined with superior quality should result in a comfortable victory for the hosts. Expect {home_team} to control possession, create numerous chances, and secure a convincing win."
            else:
                return f"🎯 HOME DOMINANCE CONTEXT: {home_team} should control proceedings against {away_team}, though organized defensive resistance may limit the margin of victory. A patient, possession-based performance should yield a home win, with {home_team} likely to break down the opposition through sustained pressure."
                
        elif expected_outcome == "offensive_showdown":
            if btts_yes > 0.65:
                return f"🎯 OFFENSIVE SHOWDOWN CONTEXT: An entertaining, open contest awaits as two attack-minded teams face off. Both {home_team} and {away_team} have demonstrated strong attacking capabilities and defensive vulnerabilities, suggesting goals at both ends in what could be a high-scoring affair. Expect end-to-end action with multiple scoring opportunities."
            else:
                return f"🎯 OFFENSIVE SHOWDOWN CONTEXT: Despite both teams' attacking intentions, this could become a tactical battle where chances are limited. The offensive quality may cancel out, leading to a tighter encounter than expected, though the potential for goals remains if either defense falters."
                
        elif expected_outcome == "defensive_battle":
            return f"🎯 DEFENSIVE BATTLE CONTEXT: A tight, tactical encounter expected between {home_team} and {away_team}. Both teams prioritize defensive organization and discipline, suggesting a low-scoring affair where set-pieces and individual quality could prove decisive. Expect limited clear-cut chances and a game decided by small margins."
                
        elif expected_outcome == "tactical_stalemate":
            return f"🎯 TACTICAL STALEMATE CONTEXT: A closely-fought tactical battle expected between {home_team} and {away_team}. Both teams will seek control through organization rather than outright attack, with small margins likely deciding the outcome. The evenly matched nature suggests a draw is the most probable result, with both teams canceling each other out."
                
        else:
            return f"🎯 BALANCED CONTEXT: A competitive match expected between {home_team} and {away_team}, with both teams having reasonable chances. The outcome will likely be decided by key moments, individual quality, and tactical adjustments in what promises to be an evenly-matched encounter where either team could emerge victorious."
    
    def _get_professional_risk_explanation(self, risk_level: str) -> str:
        explanations = {
            'LOW': "High prediction coherence with strong data support and clear match patterns. Enhanced professional confidence level with outcome-based context alignment.",
            'MEDIUM': "Reasonable prediction alignment with minor data uncertainties. Professional assessment with good confidence and clear context direction.",
            'HIGH': "Several uncertainties present requiring professional attention. Enhanced context detection provides guidance despite data limitations.",
            'VERY_HIGH': "Significant unpredictability with data limitations. Outcome-based context offers directional guidance but professional discretion strongly recommended."
        }
        return explanations.get(risk_level, "Enhanced risk assessment unavailable")
    
    def _get_professional_risk_recommendation(self, risk_level: str) -> str:
        recommendations = {
            'LOW': "PROFESSIONAL CONFIDENT STAKE - Strong context alignment",
            'MEDIUM': "PROFESSIONAL STANDARD STAKE - Good context direction", 
            'HIGH': "PROFESSIONAL CAUTIOUS STAKE - Context provides guidance",
            'VERY_HIGH': "PROFESSIONAL MINIMAL STAKE - Context directional only"
        }
        return recommendations.get(risk_level, "PROFESSIONAL ASSESSMENT REQUIRED")
    
    def _get_professional_intelligence_breakdown(self) -> str:
        return (f"Enhanced IQ: {self.intelligence.football_iq_score:.1f}/100 | "
                f"Coherence: {self.intelligence.narrative_coherence:.1%} | "
                f"Alignment: {self.intelligence.prediction_alignment} | "
                f"Risk: {self.intelligence.risk_level} | "
                f"Context Confidence: {self.intelligence.context_confidence:.1%}")
    
    def _risk_to_penalty(self, risk_level: str) -> float:
        return {'LOW': 0.02, 'MEDIUM': 0.12, 'HIGH': 0.30, 'VERY_HIGH': 0.60}.get(risk_level, 0.15)

    def _calculate_form_stability_bonus(self, home_form: List[float], away_form: List[float]) -> float:
        """Calculate form stability bonus for confidence enhancement"""
        if not home_form or not away_form:
            return 0.0
        
        # Calculate form consistency (lower variance = more stable)
        home_variance = np.var(home_form) if len(home_form) > 1 else 0
        away_variance = np.var(away_form) if len(away_form) > 1 else 0
        
        avg_variance = (home_variance + away_variance) / 2
        
        # Convert variance to stability score (0-1)
        max_expected_variance = 2.0
        stability_score = max(0, 1 - (avg_variance / max_expected_variance))
        
        return stability_score * 4

    def _calculate_context_confidence(self, narrative: MatchNarrative, home_xg: float, away_xg: float) -> float:
        """ENHANCED: Calculate confidence in context assessment"""
        total_xg = home_xg + away_xg
        xg_difference = home_xg - away_xg
        
        confidence_factors = []
        
        # Quality gap confidence
        if narrative.quality_gap == 'extreme':
            confidence_factors.append(0.9)
        elif narrative.quality_gap == 'significant':
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # xG difference confidence
        if abs(xg_difference) > 0.5:
            confidence_factors.append(0.8)
        elif abs(xg_difference) > 0.3:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.4)
        
        # Total xG confidence for offensive/defensive contexts
        if narrative.expected_outcome == 'offensive_showdown' and total_xg > 3.5:
            confidence_factors.append(0.8)
        elif narrative.expected_outcome == 'defensive_battle' and total_xg < 2.0:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)
        
        return np.mean(confidence_factors)

    def generate_professional_predictions(self, mc_iterations: int = 25000) -> Dict[str, Any]:
        """Generate enhanced professional-grade predictions with outcome focus"""
        logger.info(f"Starting enhanced prediction for {self.data['home_team']} vs {self.data['away_team']}")
        
        home_xg, away_xg = self._calculate_professional_xg()
        self.narrative = self._determine_professional_narrative(home_xg, away_xg)
        
        mc_results = self._run_professional_monte_carlo(home_xg, away_xg, mc_iterations)
        
        league = self.data.get('league', 'premier_league')
        calibrated_btts = self.goal_model.calibrator.calibrate_probability(mc_results.btts_prob, league, 'btts_yes')
        calibrated_over = self.goal_model.calibrator.calibrate_probability(mc_results.over_25_prob, league, 'over_25')
        
        home_tier = self.calibrator.get_team_tier(self.data['home_team'], league)
        away_tier = self.calibrator.get_team_tier(self.data['away_team'], league)
        
        home_team_data = {
            'xg_home': home_xg, 'xga_home': self.data.get('home_conceded', 8)/6.0, 
            'xg_last_5': self.data.get('home_form', [1.5]*5)
        }
        away_team_data = {
            'xg_away': away_xg, 'xga_away': self.data.get('away_conceded', 8)/6.0,
            'xg_last_5': self.data.get('away_form', [1.5]*5)
        }
        
        features = self.goal_model.feature_engine.create_match_features(
            home_team_data, away_team_data, self.data, home_tier, away_tier
        )
        
        probabilities = {'btts_yes': calibrated_btts, 'over_25': calibrated_over}
        explanations = self.goal_model.explainer.generate_explanation(
            features, probabilities, self.narrative.to_dict(), home_tier, away_tier
        )
        
        prediction_set = {
            'home_win': mc_results.home_win_prob,
            'btts_yes': calibrated_btts,
            'over_25': calibrated_over
        }
        
        coherence, alignment = self._assess_professional_coherence(prediction_set)
        certainty = max(mc_results.home_win_prob, mc_results.away_win_prob, mc_results.draw_prob)
        data_quality = self._calculate_professional_data_quality()
        market_edge = 0.45 + (data_quality/100 * 0.3) + (coherence * 0.25)
        
        risk_level = self._calculate_professional_risk(certainty, data_quality, market_edge, alignment)
        
        # ENHANCED: Apply form stability bonus to Football IQ
        form_stability = self._calculate_form_stability_bonus(
            self.data.get('home_form', []), 
            self.data.get('away_form', [])
        )
        stability_bonus = form_stability
        
        # ENHANCED: Calculate context confidence
        context_confidence = self._calculate_context_confidence(self.narrative, home_xg, away_xg)
        
        football_iq_score = (coherence * 30 + (data_quality/100) * 25 + 
                           (1 - self._risk_to_penalty(risk_level)) * 20 + 
                           certainty * 15 + stability_bonus + context_confidence * 10)
        
        self.intelligence = IntelligenceMetrics(
            narrative_coherence=coherence, 
            prediction_alignment=alignment,
            data_quality_score=data_quality, 
            certainty_score=certainty,
            market_edge_score=market_edge, 
            risk_level=risk_level,
            football_iq_score=football_iq_score,
            calibration_status="ENHANCED",
            context_confidence=context_confidence  # NEW
        )
        
        summary = self._generate_professional_summary(
            self.narrative, prediction_set, 
            self.data['home_team'], self.data['away_team'],
            home_tier, away_tier
        )
        
        match_context = self.narrative.expected_outcome or "balanced"
        
        logger.info(f"Enhanced prediction complete: {self.data['home_team']} {home_xg:.2f}xG - {self.data['away_team']} {away_xg:.2f}xG | Context: {match_context}")
        
        return {
            'match': f"{self.data['home_team']} vs {self.data['away_team']}",
            'league': league,
            'expected_goals': {'home': round(home_xg, 2), 'away': round(away_xg, 2)},
            'team_tiers': {'home': home_tier, 'away': away_tier},
            'match_context': match_context,
            'confidence_score': round(certainty * 100, 1),
            'data_quality_score': round(data_quality, 1),
            'match_narrative': self.narrative.to_dict(),
            'apex_intelligence': {
                'narrative_coherence': round(coherence * 100, 1),
                'prediction_alignment': alignment,
                'football_iq_score': round(football_iq_score, 1),
                'data_quality': round(data_quality, 1),
                'certainty': round(certainty * 100, 1),
                'calibration_status': 'ENHANCED',
                'form_stability_bonus': round(stability_bonus, 1),
                'context_confidence': round(context_confidence * 100, 1)  # NEW
            },
            'probabilities': {
                'match_outcomes': {
                    'home_win': round(mc_results.home_win_prob * 100, 1),
                    'draw': round(mc_results.draw_prob * 100, 1),
                    'away_win': round(mc_results.away_win_prob * 100, 1)
                },
                'both_teams_score': {
                    'yes': round(calibrated_btts * 100, 1),
                    'no': round((1 - calibrated_btts) * 100, 1)
                },
                'over_under': {
                    'over_25': round(calibrated_over * 100, 1),
                    'under_25': round((1 - calibrated_over) * 100, 1)
                },
                'exact_scores': mc_results.exact_scores
            },
            'explanations': explanations,
            'risk_assessment': {
                'risk_level': risk_level,
                'explanation': self._get_professional_risk_explanation(risk_level),
                'recommendation': self._get_professional_risk_recommendation(risk_level),
                'certainty': f"{certainty * 100:.1f}%",
            },
            'summary': summary,
            'intelligence_breakdown': self._get_professional_intelligence_breakdown(),
            'monte_carlo_results': {
                'home_win_prob': mc_results.home_win_prob,
                'draw_prob': mc_results.draw_prob,
                'away_win_prob': mc_results.away_win_prob,
            },
            'betting_context': {  # NEW: Enhanced betting context
                'primary_context': match_context,
                'recommended_markets': self.narrative.betting_priority,
                'context_confidence': round(context_confidence * 100, 1),
                'expected_outcome': self.narrative.expected_outcome
            }
        }

class ProfessionalBettingEngine:
    """ENHANCED Betting Decision Engine with Context Alignment"""
    
    def __init__(self, bankroll: float = 1000, kelly_fraction: float = 0.2):
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.value_thresholds = {
            'EXCEPTIONAL': 12.0, 'HIGH': 8.0, 'GOOD': 5.0, 'MODERATE': 2.5,
        }
        
    def calculate_expected_value(self, model_prob: float, market_odds: float) -> Dict[str, float]:
        """Enhanced expected value calculation"""
        if market_odds <= 1:
            return {'edge_percentage': 0, 'expected_value': 0, 'implied_probability': 0}
            
        implied_prob = 1.0 / market_odds
        edge = model_prob - implied_prob
        ev = (model_prob * (market_odds - 1)) - ((1 - model_prob) * 1)
        
        return {
            'edge_percentage': edge * 100,
            'expected_value': ev,
            'implied_probability': implied_prob
        }
    
    def professional_kelly_stake(self, model_prob: float, market_odds: float, confidence: str, context_alignment: str) -> float:
        """ENHANCED Kelly Criterion with context alignment"""
        if market_odds <= 1:
            return 0
            
        q = 1 - model_prob
        b = market_odds - 1
        kelly = (model_prob * (b + 1) - 1) / b
        
        if kelly <= 0:
            return 0
            
        confidence_multiplier = {
            'HIGH': 1.0,
            'MEDIUM': 0.75,
            'LOW': 0.5,
            'SPECULATIVE': 0.25
        }.get(confidence, 0.4)
        
        # ENHANCED: Context alignment multiplier
        context_multiplier = {
            'perfect': 1.2,
            'strong': 1.1,
            'moderate': 1.0,
            'weak': 0.8,
            'contradictory': 0.5
        }.get(context_alignment, 1.0)
        
        stake = max(0, kelly * self.kelly_fraction * confidence_multiplier * context_multiplier * self.bankroll)
        
        max_stake = 0.035 * self.bankroll
        min_stake = 0.005 * self.bankroll
        
        return min(max(stake, min_stake), max_stake)
    
    def _get_professional_value_rating(self, edge: float) -> str:
        """Enhanced value rating"""
        for rating, threshold in self.value_thresholds.items():
            if edge >= threshold:
                return rating
        return "LOW"
    
    def _assign_professional_confidence(self, probability: float, edge: float, data_quality: float, 
                                     league: str, form_stability: float = 0.5) -> str:
        """ENHANCED: Professional confidence assignment with context awareness"""
        
        # Get league-specific confidence multiplier
        league_calibrator = ProfessionalLeagueCalibrator()
        league_multiplier = league_calibrator.get_league_confidence_multiplier(league)
        
        # ENHANCED: More conservative confidence thresholds
        adjusted_prob_threshold_high = 0.68 * league_multiplier
        adjusted_prob_threshold_medium = 0.58 * league_multiplier
        adjusted_prob_threshold_low = 0.52 * league_multiplier
        
        adjusted_edge_threshold_high = 8.0 * league_multiplier
        adjusted_edge_threshold_medium = 5.0 * league_multiplier
        adjusted_edge_threshold_low = 2.5 * league_multiplier
        
        # Apply form stability bonus
        form_bonus = form_stability * 0.06
        effective_probability = probability + form_bonus
        
        # ENHANCED: More conservative confidence assessment
        confidence_factors = []
        
        # HIGH CONFIDENCE: Strong probability + significant edge + excellent data
        if (effective_probability > adjusted_prob_threshold_high and 
            edge > adjusted_edge_threshold_high and 
            data_quality > 80):
            confidence = 'HIGH'
            confidence_factors.append(f"Strong probability ({effective_probability:.1%})")
            confidence_factors.append(f"Significant edge ({edge:.1f}%)")
            confidence_factors.append("Excellent data quality")
            
        # MEDIUM CONFIDENCE: Good probability + solid edge + good data  
        elif (effective_probability > adjusted_prob_threshold_medium and 
              edge > adjusted_edge_threshold_medium and 
              data_quality > 70):
            confidence = 'MEDIUM'
            confidence_factors.append(f"Good probability ({effective_probability:.1%})")
            confidence_factors.append(f"Solid edge ({edge:.1f}%)")
            confidence_factors.append("Good data quality")
            
        # LOW CONFIDENCE: Reasonable probability + moderate edge + adequate data
        elif (effective_probability > adjusted_prob_threshold_low and 
              edge > adjusted_edge_threshold_low and 
              data_quality > 60):
            confidence = 'LOW'
            confidence_factors.append(f"Reasonable probability ({effective_probability:.1%})")
            confidence_factors.append(f"Moderate edge ({edge:.1f}%)")
            confidence_factors.append("Adequate data quality")
            
        # SPECULATIVE: Below thresholds
        else:
            confidence = 'SPECULATIVE'
            if effective_probability <= adjusted_prob_threshold_low:
                confidence_factors.append(f"Low probability ({effective_probability:.1%})")
            if edge <= adjusted_edge_threshold_low:
                confidence_factors.append(f"Small edge ({edge:.1f}%)")
            if data_quality <= 60:
                confidence_factors.append("Limited data quality")
        
        # Apply form stability consideration
        if form_stability > 0.75:
            confidence_factors.append("Excellent form stability")
            if confidence == 'MEDIUM':
                confidence = 'HIGH'
            elif confidence == 'LOW':
                confidence = 'MEDIUM'
            elif confidence == 'SPECULATIVE':
                confidence = 'LOW'
        elif form_stability < 0.25:
            confidence_factors.append("Unstable recent form")
            if confidence == 'HIGH':
                confidence = 'MEDIUM'
            elif confidence == 'MEDIUM':
                confidence = 'LOW'
        
        return confidence

    def _assess_context_alignment(self, market: str, betting_context: Dict) -> str:
        """ENHANCED: Assess how well a bet aligns with match context"""
        primary_context = betting_context.get('primary_context', 'balanced')
        recommended_markets = betting_context.get('recommended_markets', [])
        context_confidence = betting_context.get('context_confidence', 50)
        
        # Perfect alignment: Market is in recommended list for strong context
        if market in recommended_markets and context_confidence > 70:
            return 'perfect'
        
        # Strong alignment: Market aligns with context theme
        context_themes = {
            'home_dominance': ['1x2 Home', 'Home -1', 'Home Win to Nil'],
            'away_counter': ['1x2 Away', 'Away Win', 'Away/Draw'],
            'offensive_showdown': ['Over 2.5', 'BTTS Yes', 'Over 3.5'],
            'defensive_battle': ['Under 2.5', 'BTTS No', 'Under 1.5'],
            'tactical_stalemate': ['Draw', 'Under 2.5', 'Correct Score 0-0']
        }
        
        theme_markets = context_themes.get(primary_context, [])
        if any(theme_market in market for theme_market in theme_markets):
            return 'strong'
        
        # Moderate alignment: Neutral markets in balanced context
        if primary_context == 'balanced':
            return 'moderate'
        
        # Weak alignment: Contradicts context but has value
        contradictory_markets = {
            'home_dominance': ['1x2 Away', 'Away Win'],
            'away_counter': ['1x2 Home', 'Home Win'],
            'offensive_showdown': ['Under 2.5', 'BTTS No'],
            'defensive_battle': ['Over 2.5', 'BTTS Yes'],
            'tactical_stalemate': ['Home Win', 'Away Win']
        }
        
        if any(contradictory_market in market for contradictory_market in contradictory_markets.get(primary_context, [])):
            return 'contradictory'
        
        return 'weak'

    def _detect_signal_contradictions(self, market_name: str, primary_outcome: str, 
                                    primary_btts: str, primary_over_under: str,
                                    probability: float, betting_context: Dict) -> Tuple[bool, List[str]]:
        """Enhanced contradiction detection with context awareness"""
        contradictions = []
        is_contradiction = False
        
        primary_context = betting_context.get('primary_context', 'balanced')
        
        # ENHANCED: Context-aware contradiction detection
        if probability > 0.62:
            context_contradictions = {
                'home_dominance': ['1x2 Away', '1x2 Draw'],
                'away_counter': ['1x2 Home', '1x2 Draw'],
                'offensive_showdown': ['Under 2.5', 'BTTS No'],
                'defensive_battle': ['Over 2.5', 'BTTS Yes'],
                'tactical_stalemate': ['1x2 Home', '1x2 Away']
            }
            
            if any(contradictory in market_name for contradictory in context_contradictions.get(primary_context, [])):
                contradictions.append(f"Contradicts {primary_context.replace('_', ' ')} context")
                is_contradiction = True
        
        return is_contradiction, contradictions

    def detect_professional_value_bets(self, pure_probabilities: Dict, market_odds: Dict, 
                                    explanations: Dict, data_quality: float) -> List[BettingSignal]:
        """ENHANCED: Professional value bet detection with context alignment"""
        signals = []
        
        outcomes = pure_probabilities.get('probabilities', {}).get('match_outcomes', {})
        home_pure = outcomes.get('home_win', 33.3) / 100.0
        draw_pure = outcomes.get('draw', 33.3) / 100.0  
        away_pure = outcomes.get('away_win', 33.3) / 100.0
        
        total = home_pure + draw_pure + away_pure
        if total > 0:
            home_pure /= total
            draw_pure /= total
            away_pure /= total
        
        over_under = pure_probabilities.get('probabilities', {}).get('over_under', {})
        btts = pure_probabilities.get('probabilities', {}).get('both_teams_score', {})
        
        over_25_pure = over_under.get('over_25', 50) / 100.0
        under_25_pure = over_under.get('under_25', 50) / 100.0
        btts_yes_pure = btts.get('yes', 50) / 100.0
        btts_no_pure = btts.get('no', 50) / 100.0
        
        # Get enhanced context information
        form_stability = pure_probabilities.get('apex_intelligence', {}).get('form_stability_bonus', 0) / 4.0
        betting_context = pure_probabilities.get('betting_context', {})
        league = pure_probabilities.get('league', 'premier_league')
        
        probability_mapping = [
            ('1x2 Home', home_pure, '1x2 Home'),
            ('1x2 Draw', draw_pure, '1x2 Draw'), 
            ('1x2 Away', away_pure, '1x2 Away'),
            ('Over 2.5 Goals', over_25_pure, 'Over 2.5 Goals'),
            ('Under 2.5 Goals', under_25_pure, 'Under 2.5 Goals'),
            ('BTTS Yes', btts_yes_pure, 'BTTS Yes'),
            ('BTTS No', btts_no_pure, 'BTTS No')
        ]
        
        primary_outcome = max(outcomes, key=outcomes.get) if outcomes else 'unknown'
        primary_btts = 'yes' if btts.get('yes', 0) > btts.get('no', 0) else 'no'
        primary_over_under = 'over_25' if over_under.get('over_25', 0) > over_under.get('under_25', 0) else 'under_25'
        
        for market_name, pure_prob, market_key in probability_mapping:
            market_odd = market_odds.get(market_key, 0)
            if market_odd <= 1:
                continue
                
            ev_data = self.calculate_expected_value(pure_prob, market_odd)
            edge_percentage = ev_data['edge_percentage']
            
            # ENHANCED: Conservative edge threshold
            if edge_percentage >= 3.0:
                # Assess context alignment
                context_alignment = self._assess_context_alignment(market_name, betting_context)
                
                # Check for signal contradictions
                has_contradiction, contradiction_reasons = self._detect_signal_contradictions(
                    market_name, primary_outcome, primary_btts, primary_over_under, pure_prob, betting_context
                )
                
                # ENHANCED: Apply improved confidence assignment
                base_confidence = self._assign_professional_confidence(
                    pure_prob, edge_percentage, data_quality, league, form_stability
                )
                
                # ENHANCED: Context-aware confidence adjustment
                if context_alignment == 'perfect':
                    if base_confidence == 'MEDIUM':
                        confidence = 'HIGH'
                    elif base_confidence == 'LOW':
                        confidence = 'MEDIUM'
                    else:
                        confidence = base_confidence
                    contradiction_reasons.append("Perfect context alignment boosts confidence")
                elif context_alignment == 'contradictory':
                    if base_confidence == 'HIGH':
                        confidence = 'MEDIUM'
                    elif base_confidence == 'MEDIUM':
                        confidence = 'LOW'
                    else:
                        confidence = 'SPECULATIVE'
                    contradiction_reasons.append("Context contradiction reduces confidence")
                else:
                    confidence = base_confidence
                
                # Additional contradiction penalties
                if has_contradiction:
                    if confidence == 'HIGH':
                        confidence = 'MEDIUM'
                    elif confidence == 'MEDIUM':
                        confidence = 'LOW'
                    contradiction_reasons.append("Signal contradiction further reduces confidence")
                
                value_rating = self._get_professional_value_rating(edge_percentage)
                stake = self.professional_kelly_stake(pure_prob, market_odd, confidence, context_alignment)
                
                alignment = self._assess_professional_alignment(market_name, primary_outcome, primary_btts, primary_over_under)
                
                market_explanations = []
                if 'BTTS' in market_name:
                    market_explanations = explanations.get('btts', [])
                elif 'Over' in market_name or 'Under' in market_name:
                    market_explanations = explanations.get('over_under', [])
                elif '1x2' in market_name:
                    market_explanations = explanations.get('quality', []) + explanations.get('style', [])
                
                # Add context explanations
                context_explanations = explanations.get('context', [])
                if context_explanations:
                    market_explanations.extend(context_explanations[:1])
                
                # Combine with contradiction reasons
                all_explanations = market_explanations + contradiction_reasons
                
                # Add context alignment note
                context_note = f"Context alignment: {context_alignment}"
                if context_alignment == 'perfect':
                    context_note += " ✅"
                elif context_alignment == 'contradictory':
                    context_note += " ⚠️"
                all_explanations.append(context_note)
                
                signal = BettingSignal(
                    market=market_name, 
                    model_prob=round(pure_prob * 100, 1),
                    book_prob=round(ev_data['implied_probability'] * 100, 1), 
                    edge=round(edge_percentage, 1),
                    confidence=confidence, 
                    recommended_stake=stake, 
                    value_rating=value_rating,
                    explanation=all_explanations,
                    alignment=alignment,
                    confidence_reasoning=[f"Confidence: {confidence}", f"Context: {context_alignment}"],
                    context_alignment=context_alignment  # NEW
                )
                signals.append(signal)
        
        signals.sort(key=lambda x: (x.edge, self._confidence_weight(x.confidence), self._context_weight(x.context_alignment)), reverse=True)
        return signals
    
    def _assess_professional_alignment(self, market: str, primary_outcome: str, 
                                     primary_btts: str, primary_over_under: str) -> str:
        """Enhanced alignment assessment"""
        if '1x2 Home' in market and primary_outcome == 'home_win':
            return 'aligns_with_primary'
        elif '1x2 Away' in market and primary_outcome == 'away_win':
            return 'aligns_with_primary'
        elif '1x2 Draw' in market and primary_outcome == 'draw':
            return 'aligns_with_primary'
        elif 'BTTS Yes' in market and primary_btts == 'yes':
            return 'aligns_with_primary'
        elif 'BTTS No' in market and primary_btts == 'no':
            return 'aligns_with_primary'
        elif 'Over' in market and primary_over_under == 'over_25':
            return 'aligns_with_primary'
        elif 'Under' in market and primary_over_under == 'under_25':
            return 'aligns_with_primary'
        else:
            return 'contradicts'
    
    def _confidence_weight(self, confidence: str) -> int:
        """Enhanced confidence weighting"""
        weights = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'SPECULATIVE': 0}
        return weights.get(confidence, 0)
    
    def _context_weight(self, context_alignment: str) -> int:
        """ENHANCED: Context alignment weighting"""
        weights = {'perfect': 4, 'strong': 3, 'moderate': 2, 'weak': 1, 'contradictory': 0}
        return weights.get(context_alignment, 0)

class AdvancedFootballPredictor:
    """MAIN ENHANCED PREDICTOR - Outcome-Based Money-Grade Analysis"""
    
    def __init__(self, match_data: Dict[str, Any]):
        self.market_odds = match_data.get('market_odds', {})
        football_data = match_data.copy()
        if 'market_odds' in football_data:
            del football_data['market_odds']
        
        self.apex_engine = ApexProfessionalEngine(football_data)
        self.betting_engine = ProfessionalBettingEngine(
            bankroll=match_data.get('bankroll', 1000),
            kelly_fraction=match_data.get('kelly_fraction', 0.2)
        )
        self.predictions = None

    def generate_comprehensive_analysis(self, mc_iterations: int = 25000) -> Dict[str, Any]:
        """Generate enhanced professional-grade analysis"""
        football_predictions = self.apex_engine.generate_professional_predictions(mc_iterations)
        
        explanations = football_predictions.get('explanations', {})
        data_quality = football_predictions.get('data_quality_score', 0)
        value_signals = self.betting_engine.detect_professional_value_bets(
            football_predictions, self.market_odds, explanations, data_quality
        )
        
        alignment_status = "PERFECT" if self._validate_enhanced_alignment(football_predictions, value_signals) else "PARTIAL"
        
        professional_result = football_predictions.copy()
        professional_result['betting_signals'] = [signal.__dict__ for signal in value_signals]
        professional_result['system_validation'] = {
            'status': 'ENHANCED', 
            'alignment': alignment_status,
            'engine_sync': 'ELITE',
            'model_version': '2.4.0_enhanced',  # UPDATED: Enhanced version
            'calibration_level': 'MONEY_GRADE'
        }
        
        self.predictions = professional_result
        return professional_result
    
    def _validate_enhanced_alignment(self, football_predictions: Dict, value_signals: List[BettingSignal]) -> bool:
        """Enhanced alignment validation with context awareness"""
        if not value_signals:
            return True
            
        primary_outcome = max(
            football_predictions['probabilities']['match_outcomes'], 
            key=football_predictions['probabilities']['match_outcomes'].get
        )
        
        betting_context = football_predictions.get('betting_context', {})
        primary_context = betting_context.get('primary_context', 'balanced')
        
        # Check for major contradictions
        for signal in value_signals:
            market = signal.market
            signal_dict = signal.__dict__ if hasattr(signal, '__dict__') else signal
            
            # Major context contradictions
            if (primary_context == 'home_dominance' and '1x2 Away' in market and 
                football_predictions['probabilities']['match_outcomes']['home_win'] > 65):
                return False
            elif (primary_context == 'away_counter' and '1x2 Home' in market and 
                  football_predictions['probabilities']['match_outcomes']['away_win'] > 65):
                return False
            elif (primary_context == 'offensive_showdown' and 'Under 2.5' in market and 
                  football_predictions['probabilities']['over_under']['over_25'] > 70):
                return False
            elif (primary_context == 'defensive_battle' and 'Over 2.5' in market and 
                  football_predictions['probabilities']['over_under']['under_25'] > 70):
                return False
                
        return True

# ENHANCED TEST FUNCTION
def test_enhanced_predictor():
    """Test the enhanced professional predictor"""
    match_data = {
        'home_team': 'Derby County', 'away_team': 'Hull City', 'league': 'championship',
        'home_goals': 9, 'away_goals': 8, 'home_conceded': 7, 'away_conceded': 9,
        'home_goals_home': 5, 'away_goals_away': 4,
        'home_form': [3, 1, 0, 3, 1, 1], 'away_form': [1, 3, 1, 0, 3, 1],
        'h2h_data': {'matches': 4, 'home_wins': 2, 'away_wins': 1, 'draws': 1, 'home_goals': 6, 'away_goals': 4},
        'motivation': {'home': 'Normal', 'away': 'Normal'},
        'injuries': {'home': 2, 'away': 2},
        'market_odds': {
            '1x2 Home': 2.30, '1x2 Draw': 3.20, '1x2 Away': 3.10,
            'Over 2.5 Goals': 2.10, 'Under 2.5 Goals': 1.72,
            'BTTS Yes': 1.85, 'BTTS No': 1.95
        },
        'bankroll': 1000,
        'kelly_fraction': 0.2
    }
    
    predictor = AdvancedFootballPredictor(match_data)
    results = predictor.generate_comprehensive_analysis()
    
    print("🎯 ENHANCED PROFESSIONAL FOOTBALL PREDICTION RESULTS")
    print("=" * 70)
    print(f"Match: {results['match']}")
    print(f"League: {results['league'].upper()}")
    print(f"Enhanced IQ: {results['apex_intelligence']['football_iq_score']}/100")
    print(f"Context Confidence: {results['apex_intelligence'].get('context_confidence', 0)}%")
    print(f"Form Stability Bonus: +{results['apex_intelligence'].get('form_stability_bonus', 0)}")
    print(f"Calibration: {results['apex_intelligence']['calibration_status']}")
    print(f"Risk Level: {results['risk_assessment']['risk_level']}")
    print()
    
    print("📊 ENHANCED PROFESSIONAL PROBABILITIES:")
    outcomes = results['probabilities']['match_outcomes']
    print(f"Home Win: {outcomes['home_win']}% | Draw: {outcomes['draw']}% | Away Win: {outcomes['away_win']}%")
    print(f"BTTS Yes: {results['probabilities']['both_teams_score']['yes']}%")
    print(f"Over 2.5: {results['probabilities']['over_under']['over_25']}%")
    print()
    
    print("🎯 ENHANCED PROFESSIONAL CONTEXT:")
    betting_context = results.get('betting_context', {})
    print(f"Primary Context: {betting_context.get('primary_context', 'N/A')}")
    print(f"Expected Outcome: {betting_context.get('expected_outcome', 'N/A')}")
    print(f"Recommended Markets: {', '.join(betting_context.get('recommended_markets', []))}")
    print()
    
    print("📝 ENHANCED PROFESSIONAL SUMMARY:")
    print(results['summary'])
    print()
    
    print("💰 ENHANCED PROFESSIONAL VALUE BETS:")
    for signal in results.get('betting_signals', []):
        alignment_emoji = "✅" if signal['alignment'] == 'aligns_with_primary' else "⚠️"
        context_emoji = "🎯" if signal['context_alignment'] in ['perfect', 'strong'] else "⚖️"
        contradiction_note = " ⚠️CONTRADICTION" if "contradicts" in signal['alignment'] else ""
        print(f"- {alignment_emoji}{context_emoji} {signal['market']}: {signal['edge']}% edge | Stake: ${signal['recommended_stake']:.2f} | Confidence: {signal['confidence']} | Context: {signal['context_alignment']}{contradiction_note}")

if __name__ == "__main__":
    test_enhanced_predictor()
