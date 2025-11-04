# prediction_engine.py - PROFESSIONAL BETTING GRADE (COMPLETE FIXED VERSION)
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
    UNPREDICTABLE = "unpredictable"

class MatchNarrative:
    """PROFESSIONAL MATCH NARRATIVE - Elite Football Intelligence"""
    
    def __init__(self):
        self.dominance = "balanced"
        self.style_conflict = "neutral"
        self.expected_tempo = "medium"
        self.expected_openness = 0.5
        self.defensive_stability = "mixed"
        self.primary_pattern = None
        self.quality_gap = "even"  # even, moderate, significant, extreme
        
    def to_dict(self):
        return {
            'dominance': self.dominance,
            'style_conflict': self.style_conflict,
            'expected_tempo': self.expected_tempo,
            'expected_openness': self.expected_openness,
            'defensive_stability': self.defensive_stability,
            'primary_pattern': self.primary_pattern,
            'quality_gap': self.quality_gap
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
    alignment: str  # aligns_with_primary, contradicts, neutral

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

class EliteFeatureEngine:
    """ELITE Feature Engineering with Professional Calibration"""
    
    def __init__(self):
        self.feature_metadata = {}
        
    def create_match_features(self, home_data: Dict, away_data: Dict, context: Dict, home_tier: str, away_tier: str) -> Dict[str, float]:
        """Generate professional feature set with elite calibration"""
        
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
        
        # ELITE: Professional interaction features
        features.update({
            # Attack vs Defense matchups with tier awareness
            'home_attack_vs_away_defense': features['home_xg_for'] * (1 - (features['away_xg_against'] / 3.0)),
            'away_attack_vs_home_defense': features['away_xg_for'] * (1 - (features['home_xg_against'] / 3.0)),
            
            # Goal expectation correlations
            'total_xg_expected': features['home_xg_for'] + features['away_xg_for'],
            'xg_difference': features['home_xg_for'] - features['away_xg_for'],
            
            # Professional quality metrics
            'quality_gap_metric': tier_adjustments['quality_gap_score'],
            'home_dominance_potential': features['home_xg_for'] * (2 - features['away_xg_against']),
        })
        
        # Contextual modifiers with professional impact assessment
        features.update(self._professional_contextual_features(context, tier_adjustments))
        
        return features
    
    def _calculate_tier_adjustments(self, home_tier: str, away_tier: str, league: str) -> Dict[str, float]:
        """CRITICAL: Professional tier-based adjustments for betting"""
        
        # Tier strength mapping (professional assessment)
        tier_strength = {
            'ELITE': 1.4,    # Top European level
            'STRONG': 1.15,  # European contenders
            'MEDIUM': 1.0,   # Mid-table
            'WEAK': 0.8      # Relegation battlers
        }
        
        home_strength = tier_strength.get(home_tier, 1.0)
        away_strength = tier_strength.get(away_tier, 1.0)
        
        # Quality gap assessment
        strength_ratio = home_strength / away_strength
        quality_gap_score = min(2.0, strength_ratio)
        
        # League-specific tier impact modifiers
        league_tier_impact = {
            'premier_league': 1.0,    # Balanced tier impact
            'la_liga': 0.95,          # Slightly less tier impact
            'serie_a': 1.15,          # HIGH tier impact - critical fix!
            'bundesliga': 0.9,        # Less tier impact (more competitive)
            'ligue_1': 1.05,          # Moderate tier impact
            'liga_portugal': 1.1,     # Good tier impact
            'brasileirao': 0.95,      # Balanced
            'liga_mx': 1.0,           # Balanced
            'eredivisie': 0.9         # Less tier impact
        }
        
        impact_multiplier = league_tier_impact.get(league, 1.0)
        quality_gap_score *= impact_multiplier
        
        # Calculate multipliers with professional bounds
        if strength_ratio > 1.2:  # Home significantly stronger
            home_attack_multiplier = 1.0 + (min(0.3, (strength_ratio - 1.2) * 0.5))
            away_attack_multiplier = 1.0 - (min(0.25, (strength_ratio - 1.2) * 0.4))
            home_defense_multiplier = 1.0 + (min(0.2, (strength_ratio - 1.2) * 0.3))
            away_defense_multiplier = 1.0 - (min(0.3, (strength_ratio - 1.2) * 0.5))
        elif strength_ratio < 0.8:  # Away significantly stronger
            home_attack_multiplier = 1.0 - (min(0.25, (1.2 - strength_ratio) * 0.4))
            away_attack_multiplier = 1.0 + (min(0.3, (1.2 - strength_ratio) * 0.5))
            home_defense_multiplier = 1.0 - (min(0.3, (1.2 - strength_ratio) * 0.5))
            away_defense_multiplier = 1.0 + (min(0.2, (1.2 - strength_ratio) * 0.3))
        else:  # Relatively even
            home_attack_multiplier = 1.05  # Slight home advantage
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
            
        # More sophisticated weighting for professional use
        weights = np.array([0.85**i for i in range(len(form_data)-1, -1, -1)])
        weighted_avg = np.average(form_data, weights=weights)
        
        # Home teams get slight form boost, away teams slight reduction
        if is_home:
            return weighted_avg * 1.05
        else:
            return weighted_avg * 0.98
    
    def _professional_contextual_features(self, context: Dict, tier_adjustments: Dict) -> Dict[str, float]:
        """Professional contextual factor assessment"""
        features = {}
        
        # Professional injury impact assessment
        injury_impact_map = {
            1: 0.02,  # Rotation player
            2: 0.06,  # Regular starter  
            3: 0.12,  # Key player
            4: 0.20,  # Star player
            5: 0.30   # Multiple key players
        }
        
        home_injury_impact = injury_impact_map.get(context.get('home_injuries', 1), 0.05)
        away_injury_impact = injury_impact_map.get(context.get('away_injuries', 1), 0.05)
        
        # Motivation factors with professional calibration
        motivation_map = {'Low': 0.88, 'Normal': 1.0, 'High': 1.08, 'Very High': 1.12}
        home_motivation = motivation_map.get(context.get('home_motivation', 'Normal'), 1.0)
        away_motivation = motivation_map.get(context.get('away_motivation', 'Normal'), 1.0)
        
        # Quality gap amplifies motivation impact
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
    """PROFESSIONAL Monte Carlo Simulation with Realistic Dependencies"""
    
    def __init__(self, n_simulations: int = 25000):  # Increased for professional accuracy
        self.n_simulations = n_simulations
        
    def simulate_match_dixon_coles(self, home_xg: float, away_xg: float, correlation: float = 0.2):
        """Professional Dixon-Coles implementation"""
        # Professional correlation adjustment based on goal expectation
        goal_sum = home_xg + away_xg
        dynamic_correlation = correlation * min(1.0, goal_sum / 3.0)
        
        # Professional parameter adjustment
        lambda1 = max(0.15, home_xg - dynamic_correlation * min(home_xg, away_xg))
        lambda2 = max(0.15, away_xg - dynamic_correlation * min(home_xg, away_xg))
        lambda3 = dynamic_correlation * min(home_xg, away_xg)
        
        # Generate professional correlated Poisson
        C = np.random.poisson(lambda3, self.n_simulations)
        A = np.random.poisson(lambda1, self.n_simulations)
        B = np.random.poisson(lambda2, self.n_simulations)
        
        home_goals = A + C
        away_goals = B + C
        
        return home_goals, away_goals
    
    def get_market_probabilities(self, home_goals: np.array, away_goals: np.array) -> Dict[str, float]:
        """Professional market probability calculation"""
        
        # BTTS probabilities
        btts_yes = np.mean((home_goals > 0) & (away_goals > 0))
        btts_no = 1 - btts_yes
        
        # Over/Under probabilities
        total_goals = home_goals + away_goals
        over_15 = np.mean(total_goals > 1.5)
        over_25 = np.mean(total_goals > 2.5)
        over_35 = np.mean(total_goals > 3.5)
        
        # Professional exact score calculation
        score_counts = {}
        for h, a in zip(home_goals[:5000], away_goals[:5000]):  # Larger sample for professional accuracy
            score = f"{h}-{a}"
            score_counts[score] = score_counts.get(score, 0) + 1
        
        exact_scores = {score: count/5000 
                       for score, count in sorted(score_counts.items(), 
                       key=lambda x: x[1], reverse=True)[:8]}  # More scores for professional analysis
        
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
    """PROFESSIONAL League-Specific Calibration"""
    
    def __init__(self):
        # COMPREHENSIVE professional league profiles
        self.league_profiles = {
            'premier_league': {
                'goal_intensity': 'high', 
                'defensive_variance': 'medium', 
                'calibration_factor': 1.02,
                'home_advantage': 0.38,
                'btts_baseline': 0.52,
                'over_25_baseline': 0.51,
                'tier_impact': 1.0
            },
            'la_liga': {
                'goal_intensity': 'medium', 
                'defensive_variance': 'low', 
                'calibration_factor': 0.98,
                'home_advantage': 0.32,
                'btts_baseline': 0.48,
                'over_25_baseline': 0.47,
                'tier_impact': 0.95
            },
            'serie_a': {  # CRITICAL FIX - Serie A specific adjustments
                'goal_intensity': 'low', 
                'defensive_variance': 'low', 
                'calibration_factor': 0.94,
                'home_advantage': 0.42,  # HIGH home advantage
                'btts_baseline': 0.45,
                'over_25_baseline': 0.44,
                'tier_impact': 1.15  # HIGH tier impact
            },
            'bundesliga': {
                'goal_intensity': 'very_high', 
                'defensive_variance': 'high', 
                'calibration_factor': 1.08,
                'home_advantage': 0.28,
                'btts_baseline': 0.55,
                'over_25_baseline': 0.58,
                'tier_impact': 0.9
            },
            'ligue_1': {
                'goal_intensity': 'low', 
                'defensive_variance': 'medium', 
                'calibration_factor': 0.95,
                'home_advantage': 0.34,
                'btts_baseline': 0.47,
                'over_25_baseline': 0.46,
                'tier_impact': 1.05
            },
            'liga_portugal': {
                'goal_intensity': 'medium', 
                'defensive_variance': 'medium', 
                'calibration_factor': 0.97,
                'home_advantage': 0.42,
                'btts_baseline': 0.49,
                'over_25_baseline': 0.48,
                'tier_impact': 1.1
            },
            'brasileirao': {
                'goal_intensity': 'high', 
                'defensive_variance': 'high', 
                'calibration_factor': 1.02,
                'home_advantage': 0.45,
                'btts_baseline': 0.51,
                'over_25_baseline': 0.52,
                'tier_impact': 0.95
            },
            'liga_mx': {
                'goal_intensity': 'medium', 
                'defensive_variance': 'high', 
                'calibration_factor': 1.01,
                'home_advantage': 0.40,
                'btts_baseline': 0.50,
                'over_25_baseline': 0.49,
                'tier_impact': 1.0
            },
            'eredivisie': {
                'goal_intensity': 'high', 
                'defensive_variance': 'high', 
                'calibration_factor': 1.03,
                'home_advantage': 0.30,
                'btts_baseline': 0.54,
                'over_25_baseline': 0.56,
                'tier_impact': 0.9
            }
        }
    
    def calibrate_probability(self, raw_prob: float, league: str, market_type: str) -> float:
        """Professional probability calibration"""
        profile = self.league_profiles.get(league, self.league_profiles['premier_league'])
        base_calibrated = raw_prob * profile['calibration_factor']
        
        # Professional market-specific adjustments
        if market_type == 'over_25':
            if profile['goal_intensity'] == 'very_high':
                base_calibrated *= 1.08
            elif profile['goal_intensity'] == 'high':
                base_calibrated *= 1.04
            elif profile['goal_intensity'] == 'low':
                base_calibrated *= 0.94
            elif profile['goal_intensity'] == 'very_low':
                base_calibrated *= 0.88
                
        elif market_type == 'btts_yes':
            if profile['defensive_variance'] == 'low':
                base_calibrated *= 0.92
            elif profile['defensive_variance'] == 'high':
                base_calibrated *= 1.06
                
        # Professional bounds
        return np.clip(base_calibrated, 0.025, 0.975)

class ProfessionalPredictionExplainer:
    """PROFESSIONAL Explanation Engine"""
    
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
    
    def generate_explanation(self, features: Dict, probabilities: Dict, narrative: Dict, home_tier: str, away_tier: str) -> Dict[str, List[str]]:
        """Generate professional explanations"""
        
        explanations = {}
        
        # Extract professional metrics
        home_attack = features.get('home_xg_for', 1.5)
        away_attack = features.get('away_xg_for', 1.2)
        home_defense = features.get('home_xg_against', 1.3)
        away_defense = features.get('away_xg_against', 1.4)
        total_xg = features.get('total_xg_expected', 2.7)
        quality_gap = features.get('quality_gap_metric', 1.0)
        
        # Professional BTTS explanation
        btts_prob = probabilities.get('btts_yes', 0.5)
        if btts_prob > 0.65:
            explanations['btts'] = [
                f"Both teams demonstrate strong attacking capabilities (Home: {home_attack:.1f}, Away: {away_attack:.1f} xG)",
                f"Defensive vulnerabilities evident (Home concedes: {home_defense:.1f}, Away concedes: {away_defense:.1f} xG)"
            ]
        elif btts_prob < 0.35:
            if quality_gap > 1.3:
                explanations['btts'] = [
                    f"Significant quality gap favors {home_tier} home team over {away_tier} away side",
                    f"Defensive organization likely to limit scoring opportunities for weaker team"
                ]
            else:
                explanations['btts'] = [
                    f"Strong defensive organization from one or both teams",
                    f"Limited attacking threat reduces BTTS probability"
                ]
        else:
            explanations['btts'] = [
                f"Balanced attacking and defensive capabilities",
                f"Moderate chance for both teams to score"
            ]
        
        # Professional Over/Under explanation
        over_prob = probabilities.get('over_25', 0.5)
        if over_prob > 0.65:
            explanations['over_under'] = [
                f"High expected goal volume (Total xG: {total_xg:.2f})",
                f"Attacking styles and defensive vulnerabilities suggest multiple goals"
            ]
        elif over_prob < 0.35:
            if quality_gap > 1.4:
                explanations['over_under'] = [
                    f"One-sided contest expected with {home_tier} team dominating",
                    f"Game management may limit total goals despite quality gap"
                ]
            else:
                explanations['over_under'] = [
                    f"Defensive organization likely to limit scoring opportunities",
                    f"Tactical approach suggests lower-scoring affair"
                ]
        else:
            explanations['over_under'] = [
                f"Average expected goal volume (Total xG: {total_xg:.2f})",
                f"Game could go either way in terms of total goals"
            ]
            
        # Professional narrative-based explanations
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
            
        return explanations

class ProfessionalTeamTierCalibrator:
    """PROFESSIONAL Team Tier Calibration"""
    
    def __init__(self):
        # Professional league baselines
        self.league_baselines = {
            'premier_league': {'avg_goals': 2.8, 'home_advantage': 0.38, 'btts_rate': 0.52},
            'la_liga': {'avg_goals': 2.6, 'home_advantage': 0.32, 'btts_rate': 0.48},
            'serie_a': {'avg_goals': 2.7, 'home_advantage': 0.42, 'btts_rate': 0.45},  # Fixed
            'bundesliga': {'avg_goals': 3.1, 'home_advantage': 0.28, 'btts_rate': 0.55},
            'ligue_1': {'avg_goals': 2.5, 'home_advantage': 0.34, 'btts_rate': 0.47},
            'liga_portugal': {'avg_goals': 2.6, 'home_advantage': 0.42, 'btts_rate': 0.49},
            'brasileirao': {'avg_goals': 2.4, 'home_advantage': 0.45, 'btts_rate': 0.51},
            'liga_mx': {'avg_goals': 2.7, 'home_advantage': 0.40, 'btts_rate': 0.50},
            'eredivisie': {'avg_goals': 3.0, 'home_advantage': 0.30, 'btts_rate': 0.54},
        }
        
        # PROFESSIONAL TEAM DATABASES - Comprehensive and accurate
        self.team_databases = {
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
            'serie_a': {  # CRITICAL FIX - Accurate Serie A tiers
                'Inter': 'ELITE', 'Juventus': 'ELITE', 'AC Milan': 'ELITE',
                'Napoli': 'STRONG', 'Atalanta': 'STRONG', 'Roma': 'STRONG',
                'Lazio': 'STRONG', 'Fiorentina': 'MEDIUM', 'Bologna': 'MEDIUM',
                'Monza': 'MEDIUM', 'Torino': 'MEDIUM', 'Genoa': 'MEDIUM',
                'Lecce': 'MEDIUM', 'Sassuolo': 'MEDIUM', 'Frosinone': 'WEAK',
                'Udinese': 'WEAK', 'Verona': 'WEAK', 'Empoli': 'WEAK',
                'Cagliari': 'WEAK', 'Salernitana': 'WEAK'
            },
            'bundesliga': {
                'Bayern Munich': 'ELITE', 'Bayer Leverkusen': 'ELITE', 'Borussia Dortmund': 'ELITE',
                'RB Leipzig': 'STRONG', 'Eintracht Frankfurt': 'STRONG', 'Wolfsburg': 'STRONG',
                'Freiburg': 'STRONG', 'Hoffenheim': 'STRONG', 'Augsburg': 'MEDIUM',
                'Stuttgart': 'STRONG', 'Borussia Mönchengladbach': 'MEDIUM', 'Werder Bremen': 'MEDIUM',
                'Heidenheim': 'MEDIUM', 'Union Berlin': 'MEDIUM', 'Bochum': 'WEAK',
                'Mainz': 'WEAK', 'Köln': 'WEAK', 'Darmstadt': 'WEAK'
            },
            'ligue_1': {
                'PSG': 'ELITE', 'Monaco': 'STRONG', 'Marseille': 'STRONG',
                'Lille': 'STRONG', 'Lyon': 'STRONG', 'Rennes': 'STRONG',
                'Nice': 'STRONG', 'Lens': 'STRONG', 'Reims': 'MEDIUM',
                'Montpellier': 'MEDIUM', 'Toulouse': 'MEDIUM', 'Strasbourg': 'MEDIUM',
                'Nantes': 'MEDIUM', 'Le Havre': 'MEDIUM', 'Brest': 'MEDIUM',
                'Metz': 'WEAK', 'Lorient': 'WEAK', 'Clermont': 'WEAK'
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
                'America': 'ELITE', 'Monterrey': 'ELITE', 'Tigres': 'ELITE',
                'Cruz Azul': 'STRONG', 'Guadalajara': 'STRONG', 'Pumas': 'STRONG',
                'Toluca': 'STRONG', 'Santos Laguna': 'MEDIUM', 'Pachuca': 'MEDIUM',
                'Leon': 'MEDIUM', 'Juarez': 'MEDIUM', 'Mazatlan': 'MEDIUM',
                'Necaxa': 'MEDIUM', 'Queretaro': 'MEDIUM', 'Atlas': 'MEDIUM',
                'Tijuana': 'WEAK', 'Puebla': 'WEAK', 'San Luis': 'WEAK'
            },
            'eredivisie': {
                'Ajax': 'ELITE', 'PSV': 'ELITE', 'Feyenoord': 'ELITE',
                'AZ Alkmaar': 'STRONG', 'Twente': 'STRONG', 'Sparta Rotterdam': 'MEDIUM',
                'Heerenveen': 'MEDIUM', 'NEC Nijmegen': 'MEDIUM', 'Utrecht': 'MEDIUM',
                'Go Ahead Eagles': 'MEDIUM', 'Fortuna Sittard': 'MEDIUM', 'Heracles': 'MEDIUM',
                'Almere City': 'MEDIUM', 'Excelsior': 'WEAK', 'RKC Waalwijk': 'WEAK',
                'Volendam': 'WEAK', 'Vitesse': 'WEAK'
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
    """PROFESSIONAL Goal Prediction Model"""
    
    def __init__(self):
        self.feature_engine = EliteFeatureEngine()
        self.simulator = ProfessionalMatchSimulator()
        self.calibrator = ProfessionalLeagueCalibrator()
        self.explainer = ProfessionalPredictionExplainer()
        self.tier_calibrator = ProfessionalTeamTierCalibrator()
        
    def calculate_team_strength(self, team_data: Dict, is_home: bool = True) -> Dict[str, float]:
        """Professional team strength calculation"""
        prior_games = 8  # Reduced for more responsiveness
        prior_strength = 1.5  # League average
        
        # Professional form analysis
        xg_data = team_data.get('xg_last_6', [1.5] * 6)
        xga_data = team_data.get('xga_last_6', [1.3] * 6)
        
        # Professional weighting with recent bias
        weights = np.array([0.8**i for i in range(5, -1, -1)])  # More recent bias
        recent_xg = np.average(xg_data, weights=weights)
        recent_xga = np.average(xga_data, weights=weights)
        
        # Professional Bayesian shrinkage
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
    """APEX PROFESSIONAL ENGINE - Money-Grade Predictions"""
    
    def __init__(self, match_data: Dict[str, Any]):
        self.data = self._professional_data_validation(match_data)
        self.calibrator = ProfessionalTeamTierCalibrator()
        self.goal_model = ProfessionalGoalModel()
        self.narrative = MatchNarrative()
        self.intelligence = IntelligenceMetrics(
            narrative_coherence=0.0, prediction_alignment="LOW", 
            data_quality_score=0.0, certainty_score=0.0,
            market_edge_score=0.0, risk_level="HIGH", football_iq_score=0.0,
            calibration_status="PENDING"
        )
        self._setup_professional_parameters()
        
    def _professional_data_validation(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Professional data validation and enhancement"""
        enhanced_data = match_data.copy()
        
        # Professional field validation
        required_fields = ['home_team', 'away_team', 'league']
        for field in required_fields:
            if field not in enhanced_data:
                enhanced_data[field] = 'Unknown'
                logger.warning(f"Missing required field: {field}")
        
        # Professional data quality assessment
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
        
        # Professional form data processing
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
        
        # Professional default values
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
            'form_decay_rate': 0.80,  # More responsive
            'h2h_weight': 0.18,       # Increased H2H importance
            'injury_impact': 0.10,    # Realistic injury impact
            'motivation_impact': 0.12, # Professional motivation assessment
            'defensive_impact_multiplier': 0.45,
            'tier_impact_base': 0.15, # Base tier impact
        }

    def _calculate_professional_xg(self) -> Tuple[float, float]:
        """PROFESSIONAL xG calculation with elite calibration"""
        league = self.data.get('league', 'premier_league')
        league_baseline = self.calibrator.league_baselines.get(league, {'avg_goals': 2.7, 'home_advantage': 0.35})
        
        # Get professional team tiers
        home_tier = self.calibrator.get_team_tier(self.data['home_team'], league)
        away_tier = self.calibrator.get_team_tier(self.data['away_team'], league)
        
        # Prepare professional team data
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
        
        # Calculate professional team strengths
        home_strength = self.goal_model.calculate_team_strength(home_team_data, is_home=True)
        away_strength = self.goal_model.calculate_team_strength(away_team_data, is_home=False)
        
        # CRITICAL: Professional base xG calculation
        home_xg = home_strength['attack'] * (1 - (away_strength['defense'] / league_baseline['avg_goals']) * 0.35)
        away_xg = away_strength['attack'] * (1 - (home_strength['defense'] / league_baseline['avg_goals']) * 0.35)
        
        # PROFESSIONAL: Apply league-specific home advantage
        home_advantage = league_baseline['home_advantage']
        home_xg *= (1 + home_advantage)
        
        # PROFESSIONAL: Apply contextual factors with tier awareness
        context = {
            'home_injuries': self.data.get('injuries', {}).get('home', 1),
            'away_injuries': self.data.get('injuries', {}).get('away', 1),
            'home_motivation': self.data.get('motivation', {}).get('home', 'Normal'),
            'away_motivation': self.data.get('motivation', {}).get('away', 'Normal'),
            'match_importance': 0.5,
            'league': league
        }
        
        # Use professional feature engine with tier information
        features = self.goal_model.feature_engine.create_match_features(
            home_team_data, away_team_data, context, home_tier, away_tier
        )
        
        # Apply professional feature adjustments
        home_xg *= features.get('home_injury_factor', 1.0) * features.get('home_motivation_factor', 1.0)
        away_xg *= features.get('away_injury_factor', 1.0) * features.get('away_motivation_factor', 1.0)
        
        # CRITICAL: Professional H2H adjustment
        h2h_data = self.data.get('h2h_data', {})
        if h2h_data and h2h_data.get('matches', 0) >= 2:
            home_xg, away_xg = self._apply_professional_h2h_adjustment(home_xg, away_xg, h2h_data, home_tier, away_tier)
        
        # PROFESSIONAL: Final calibration and bounds
        home_xg = max(0.25, min(3.5, home_xg))
        away_xg = max(0.25, min(3.0, away_xg))
        
        logger.info(f"Professional xG calculated: {self.data['home_team']} {home_xg:.2f} - {self.data['away_team']} {away_xg:.2f}")
        return round(home_xg, 3), round(away_xg, 3)

    def _apply_professional_h2h_adjustment(self, home_xg: float, away_xg: float, h2h_data: Dict, home_tier: str, away_tier: str) -> Tuple[float, float]:
        """Professional H2H adjustment"""
        matches = h2h_data.get('matches', 0)
        if matches < 2:
            return home_xg, away_xg
        
        # Professional H2H weight calculation
        h2h_weight = min(0.30, matches * 0.08)  # Increased maximum weight
        
        h2h_home_avg = h2h_data.get('home_goals', 0) / matches
        h2h_away_avg = h2h_data.get('away_goals', 0) / matches
        
        if h2h_home_avg > 0 or h2h_away_avg > 0:
            # Professional adjustment considering team tiers
            tier_strength = {'ELITE': 1.4, 'STRONG': 1.15, 'MEDIUM': 1.0, 'WEAK': 0.8}
            home_tier_strength = tier_strength.get(home_tier, 1.0)
            away_tier_strength = tier_strength.get(away_tier, 1.0)
            
            # More sophisticated H2H adjustment
            adjusted_home_xg = (home_xg * (1 - h2h_weight)) + (h2h_home_avg * h2h_weight * home_tier_strength)
            adjusted_away_xg = (away_xg * (1 - h2h_weight)) + (h2h_away_avg * h2h_weight * away_tier_strength)
            
            return adjusted_home_xg, adjusted_away_xg
        
        return home_xg, away_xg

    def _determine_professional_narrative(self, home_xg: float, away_xg: float) -> MatchNarrative:
        """Professional match narrative determination"""
        narrative = MatchNarrative()
        total_xg = home_xg + away_xg
        xg_difference = home_xg - away_xg
        
        # Get professional team tiers
        league = self.data.get('league', 'premier_league')
        home_tier = self.calibrator.get_team_tier(self.data['home_team'], league)
        away_tier = self.calibrator.get_team_tier(self.data['away_team'], league)
        
        # PROFESSIONAL: Quality gap assessment
        tier_strength = {'ELITE': 4, 'STRONG': 3, 'MEDIUM': 2, 'WEAK': 1}
        home_strength = tier_strength.get(home_tier, 2)
        away_strength = tier_strength.get(away_tier, 2)
        
        strength_difference = home_strength - away_strength
        
        if strength_difference >= 2:
            narrative.quality_gap = "extreme"
            narrative.dominance = "home"
            narrative.primary_pattern = "home_dominance"
        elif strength_difference >= 1:
            narrative.quality_gap = "significant"
            narrative.dominance = "home"
            narrative.primary_pattern = "home_dominance"
        elif strength_difference <= -2:
            narrative.quality_gap = "extreme"
            narrative.dominance = "away"
            narrative.primary_pattern = "away_counter"
        elif strength_difference <= -1:
            narrative.quality_gap = "significant" 
            narrative.dominance = "away"
            narrative.primary_pattern = "away_counter"
        else:
            narrative.quality_gap = "even"
            # Use xG difference for balanced teams
            if xg_difference > 0.6:
                narrative.dominance = "home"
                narrative.primary_pattern = "home_dominance"
            elif xg_difference < -0.6:
                narrative.dominance = "away"
                narrative.primary_pattern = "away_counter"
            else:
                narrative.dominance = "balanced"
                narrative.primary_pattern = "tactical_stalemate"
        
        # Professional style assessment
        home_attack = self.data.get('home_goals', 0) / 6.0
        away_attack = self.data.get('away_goals', 0) / 6.0
        home_defense = self.data.get('home_conceded', 0) / 6.0
        away_defense = self.data.get('away_conceded', 0) / 6.0
        
        # Adjust for league context
        league_profile = self.goal_model.calibrator.league_profiles.get(league, {})
        goal_intensity = league_profile.get('goal_intensity', 'medium')
        
        if goal_intensity == 'low':
            attack_threshold = 1.6
            defense_threshold = 1.0
        elif goal_intensity == 'high':
            attack_threshold = 2.0
            defense_threshold = 1.4
        else:
            attack_threshold = 1.8
            defense_threshold = 1.2
        
        if home_attack > attack_threshold and away_attack > attack_threshold:
            narrative.style_conflict = "attacking_vs_attacking"
            narrative.expected_openness = 0.8
            narrative.expected_tempo = "high"
        elif home_attack > attack_threshold and away_defense < defense_threshold:
            narrative.style_conflict = "attacking_vs_defensive"
            narrative.expected_openness = 0.6
            narrative.expected_tempo = "medium"
        elif away_attack > attack_threshold and home_defense < defense_threshold:
            narrative.style_conflict = "defensive_vs_attacking" 
            narrative.expected_openness = 0.6
            narrative.expected_tempo = "medium"
        else:
            narrative.style_conflict = "balanced"
            narrative.expected_openness = 0.5
            narrative.expected_tempo = "medium"
            
        # Professional defensive stability assessment
        avg_conceded = (home_defense + away_defense) / 2
        if avg_conceded < 0.8:
            narrative.defensive_stability = "solid"
        elif avg_conceded > 1.5:
            narrative.defensive_stability = "leaky" 
        else:
            narrative.defensive_stability = "mixed"
            
        return narrative

    def _run_professional_monte_carlo(self, home_xg: float, away_xg: float, iterations: int = 25000) -> MonteCarloResults:
        """Professional Monte Carlo simulation"""
        home_goals, away_goals = self.goal_model.simulator.simulate_match_dixon_coles(home_xg, away_xg)
        market_probs = self.goal_model.simulator.get_market_probabilities(home_goals, away_goals)
        
        # Professional outcome probabilities
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
        """Professional data quality assessment"""
        score = 0
        max_score = 100
        
        # Team information
        if self.data.get('home_team') and self.data.get('away_team') and self.data.get('home_team') != 'Unknown' and self.data.get('away_team') != 'Unknown':
            score += 20
        
        # Goal data quality
        home_goals = self.data.get('home_goals', 0)
        away_goals = self.data.get('away_goals', 0)
        if home_goals > 0 and away_goals > 0:
            score += 25
        elif home_goals > 0 or away_goals > 0:
            score += 15
        
        # Form data quality
        home_form = self.data.get('home_form', [])
        away_form = self.data.get('away_form', [])
        if len(home_form) >= 5 and len(away_form) >= 5:
            score += 25
        elif len(home_form) >= 3 or len(away_form) >= 3:
            score += 15
        
        # H2H data quality
        h2h_data = self.data.get('h2h_data', {})
        if h2h_data.get('matches', 0) >= 4:
            score += 20
        elif h2h_data.get('matches', 0) >= 2:
            score += 10
        
        # Contextual data quality
        if self.data.get('motivation') and self.data.get('injuries'):
            score += 10
        
        return min(100, score)

    def _assess_professional_coherence(self, predictions: Dict) -> Tuple[float, str]:
        """Professional prediction coherence assessment"""
        btts_yes = predictions.get('btts_yes', 0.5)
        over_25 = predictions.get('over_25', 0.5)
        home_win = predictions.get('home_win', 0.33)
        
        coherence_score = 0.0
        
        # Professional coherence rules
        if btts_yes > 0.7 and over_25 < 0.4:
            coherence_score -= 0.4  # Strong contradiction
        elif btts_yes < 0.3 and over_25 > 0.7:
            coherence_score -= 0.4
        else:
            coherence_score += 0.3
            
        if home_win > 0.65 and over_25 < 0.35:
            coherence_score -= 0.3
        elif home_win < 0.25 and over_25 > 0.65:
            coherence_score -= 0.3
        else:
            coherence_score += 0.2
            
        # Narrative alignment
        narrative = self.narrative.to_dict()
        if narrative.get('primary_pattern') == 'home_dominance' and home_win < 0.45:
            coherence_score -= 0.2
        elif narrative.get('primary_pattern') == 'away_counter' and home_win > 0.55:
            coherence_score -= 0.2
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
        """Professional risk assessment"""
        base_risk = (1 - certainty) * 0.4 + (1 - data_quality/100) * 0.3 + (1 - market_edge) * 0.3
        
        alignment_penalty = {
            "HIGH": 0.0, "MEDIUM": 0.15, "LOW": 0.3
        }.get(alignment, 0.2)
        
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
        """Professional match summary"""
        home_win = predictions.get('home_win', 0.33)
        btts_yes = predictions.get('btts_yes', 0.5)
        over_25 = predictions.get('over_25', 0.5)
        
        quality_gap = narrative.quality_gap
        
        if quality_gap == "extreme":
            if home_win > 0.6:
                return f"{home_team} ({home_tier}) are overwhelming favorites against {away_team} ({away_tier}). The significant quality gap suggests comprehensive home dominance, with the hosts expected to control proceedings and secure a comfortable victory."
            else:
                return f"{away_team} ({away_tier}) possess clear superiority over {home_team} ({home_tier}). Despite home advantage, the visitors' quality should prevail in what appears to be a mismatch on paper."
                
        elif quality_gap == "significant":
            if home_win > 0.55:
                return f"{home_team} ({home_tier}) hold a clear advantage over {away_team} ({away_tier}). Home advantage combined with superior quality should see the hosts control this encounter, though the visitors may offer periods of resistance."
            else:
                return f"{away_team} ({away_tier}) are favored despite traveling to {home_team} ({home_tier}). The visitors' quality edge may overcome home advantage, suggesting an away win or hard-fought draw is the most likely outcome."
                
        elif narrative.primary_pattern == "home_dominance":
            if over_25 > 0.6:
                return f"{home_team} are expected to dominate possession and create numerous chances against {away_team}. The home side's attacking quality combined with the visitors' defensive vulnerabilities points toward a convincing home victory with multiple goals."
            else:
                return f"{home_team} should control this match against {away_team}, but may face organized defensive resistance. A patient, probing performance could yield a narrow victory rather than a goal-filled rout."
                
        elif narrative.style_conflict == "attacking_vs_attacking":
            if btts_yes > 0.65:
                return f"An entertaining, open contest awaits as two attack-minded teams face off. Both {home_team} and {away_team} have shown defensive frailties, suggesting goals at both ends in what could be a high-scoring affair decided by attacking quality."
            else:
                return f"Despite both teams' attacking intentions, this could become a tactical battle where chances are limited. The offensive quality on display may cancel out, leading to a tighter encounter than the attacking reputations suggest."
                
        else:
            return f"A competitive match expected between {home_team} and {away_team}, with small margins likely deciding the outcome. Both teams will seek to establish control in what promises to be a closely-fought encounter where tactical discipline could prove decisive."
    
    def _get_professional_risk_explanation(self, risk_level: str) -> str:
        explanations = {
            'LOW': "High prediction coherence with strong data support and clear match patterns. Professional confidence level.",
            'MEDIUM': "Reasonable prediction alignment with some data uncertainties. Professional assessment with standard confidence.",
            'HIGH': "Multiple uncertainties present with some conflicting signals. Professional caution advised.",
            'VERY_HIGH': "Significant unpredictability with limited data quality. Professional discretion strongly recommended."
        }
        return explanations.get(risk_level, "Professional risk assessment unavailable")
    
    def _get_professional_risk_recommendation(self, risk_level: str) -> str:
        recommendations = {
            'LOW': "PROFESSIONAL CONFIDENT STAKE",
            'MEDIUM': "PROFESSIONAL STANDARD STAKE", 
            'HIGH': "PROFESSIONAL CAUTIOUS STAKE",
            'VERY_HIGH': "PROFESSIONAL MINIMAL STAKE"
        }
        return recommendations.get(risk_level, "PROFESSIONAL ASSESSMENT REQUIRED")
    
    def _get_professional_intelligence_breakdown(self) -> str:
        return (f"Professional IQ: {self.intelligence.football_iq_score:.1f}/100 | "
                f"Coherence: {self.intelligence.narrative_coherence:.1%} | "
                f"Alignment: {self.intelligence.prediction_alignment} | "
                f"Risk: {self.intelligence.risk_level} | "
                f"Calibration: {self.intelligence.calibration_status}")
    
    def _risk_to_penalty(self, risk_level: str) -> float:
        return {'LOW': 0.05, 'MEDIUM': 0.2, 'HIGH': 0.5, 'VERY_HIGH': 0.8}.get(risk_level, 0.3)

    def generate_professional_predictions(self, mc_iterations: int = 25000) -> Dict[str, Any]:
        """Generate professional-grade predictions"""
        logger.info(f"Starting professional prediction for {self.data['home_team']} vs {self.data['away_team']}")
        
        # Calculate professional metrics
        home_xg, away_xg = self._calculate_professional_xg()
        self.narrative = self._determine_professional_narrative(home_xg, away_xg)
        
        # Run professional Monte Carlo simulation
        mc_results = self._run_professional_monte_carlo(home_xg, away_xg, mc_iterations)
        
        # Apply professional league-specific calibration
        league = self.data.get('league', 'premier_league')
        calibrated_btts = self.goal_model.calibrator.calibrate_probability(mc_results.btts_prob, league, 'btts_yes')
        calibrated_over = self.goal_model.calibrator.calibrate_probability(mc_results.over_25_prob, league, 'over_25')
        
        # Get professional team tiers
        home_tier = self.calibrator.get_team_tier(self.data['home_team'], league)
        away_tier = self.calibrator.get_team_tier(self.data['away_team'], league)
        
        # Generate professional explanations
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
        
        # Professional prediction set for coherence assessment
        prediction_set = {
            'home_win': mc_results.home_win_prob,
            'btts_yes': calibrated_btts,
            'over_25': calibrated_over
        }
        
        # Calculate professional intelligence metrics
        coherence, alignment = self._assess_professional_coherence(prediction_set)
        certainty = max(mc_results.home_win_prob, mc_results.away_win_prob, mc_results.draw_prob)
        data_quality = self._calculate_professional_data_quality()
        market_edge = 0.4 + (data_quality/100 * 0.3) + (coherence * 0.3)
        
        risk_level = self._calculate_professional_risk(certainty, data_quality, market_edge, alignment)
        
        # Professional Football IQ score
        football_iq_score = (coherence * 35 + (data_quality/100) * 30 + 
                           (1 - self._risk_to_penalty(risk_level)) * 25 + certainty * 10)
        
        self.intelligence = IntelligenceMetrics(
            narrative_coherence=coherence, 
            prediction_alignment=alignment,
            data_quality_score=data_quality, 
            certainty_score=certainty,
            market_edge_score=market_edge, 
            risk_level=risk_level,
            football_iq_score=football_iq_score,
            calibration_status="PROFESSIONAL"
        )
        
        # Professional summary
        summary = self._generate_professional_summary(
            self.narrative, prediction_set, 
            self.data['home_team'], self.data['away_team'],
            home_tier, away_tier
        )
        
        # Determine professional match context
        match_context = self.narrative.primary_pattern or "balanced"
        
        logger.info(f"Professional prediction complete: {self.data['home_team']} {home_xg:.2f}xG - {self.data['away_team']} {away_xg:.2f}xG")
        
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
                'calibration_status': 'PROFESSIONAL'
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
            }
        }

class ProfessionalBettingEngine:
    """PROFESSIONAL Betting Decision Engine"""
    
    def __init__(self, bankroll: float = 1000, kelly_fraction: float = 0.2):  # More conservative
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.value_thresholds = {
            'EXCEPTIONAL': 20.0, 'HIGH': 12.0, 'GOOD': 7.0, 'MODERATE': 4.0,  # More realistic
        }
        
    def calculate_expected_value(self, model_prob: float, market_odds: float) -> Dict[str, float]:
        """Professional expected value calculation"""
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
    
    def professional_kelly_stake(self, model_prob: float, market_odds: float, confidence: str) -> float:
        """Professional Kelly Criterion with confidence adjustment"""
        if market_odds <= 1:
            return 0
            
        q = 1 - model_prob
        b = market_odds - 1
        kelly = (model_prob * (b + 1) - 1) / b
        
        if kelly <= 0:
            return 0
            
        # Professional confidence adjustments
        confidence_multiplier = {
            'HIGH': 1.0,
            'MEDIUM': 0.7, 
            'LOW': 0.4,
            'SPECULATIVE': 0.2
        }.get(confidence, 0.5)
        
        # Apply fractional Kelly with professional limits
        stake = max(0, kelly * self.kelly_fraction * confidence_multiplier * self.bankroll)
        
        # Professional stake limits
        max_stake = 0.03 * self.bankroll  # Max 3% of bankroll - more conservative
        min_stake = 0.005 * self.bankroll  # Min 0.5% of bankroll
        
        return min(max(stake, min_stake), max_stake)
    
    def _get_professional_value_rating(self, edge: float) -> str:
        """Professional value rating"""
        for rating, threshold in self.value_thresholds.items():
            if edge >= threshold:
                return rating
        return "LOW"
    
    def _assign_professional_confidence(self, probability: float, edge: float, data_quality: float) -> str:
        """Professional confidence assignment"""
        if probability > 0.70 and edge > 12 and data_quality > 80:
            return 'HIGH'
        elif probability > 0.60 and edge > 8 and data_quality > 70:
            return 'MEDIUM'
        elif probability > 0.55 and edge > 5 and data_quality > 60:
            return 'LOW'
        else:
            return 'SPECULATIVE'

    def detect_professional_value_bets(self, pure_probabilities: Dict, market_odds: Dict, 
                                    explanations: Dict, data_quality: float) -> List[BettingSignal]:
        """Professional value bet detection"""
        signals = []
        
        # Professional probability extraction
        outcomes = pure_probabilities.get('probabilities', {}).get('match_outcomes', {})
        home_pure = outcomes.get('home_win', 33.3) / 100.0
        draw_pure = outcomes.get('draw', 33.3) / 100.0  
        away_pure = outcomes.get('away_win', 33.3) / 100.0
        
        # Professional normalization
        total = home_pure + draw_pure + away_pure
        if total > 0:
            home_pure /= total
            draw_pure /= total
            away_pure /= total
        
        # Get other probabilities professionally
        over_under = pure_probabilities.get('probabilities', {}).get('over_under', {})
        btts = pure_probabilities.get('probabilities', {}).get('both_teams_score', {})
        
        over_25_pure = over_under.get('over_25', 50) / 100.0
        under_25_pure = over_under.get('under_25', 50) / 100.0
        btts_yes_pure = btts.get('yes', 50) / 100.0
        btts_no_pure = btts.get('no', 50) / 100.0
        
        # Professional probability mapping
        probability_mapping = [
            ('1x2 Home', home_pure, '1x2 Home'),
            ('1x2 Draw', draw_pure, '1x2 Draw'), 
            ('1x2 Away', away_pure, '1x2 Away'),
            ('Over 2.5 Goals', over_25_pure, 'Over 2.5 Goals'),
            ('Under 2.5 Goals', under_25_pure, 'Under 2.5 Goals'),
            ('BTTS Yes', btts_yes_pure, 'BTTS Yes'),
            ('BTTS No', btts_no_pure, 'BTTS No')
        ]
        
        # Get primary predictions for alignment
        primary_outcome = max(outcomes, key=outcomes.get) if outcomes else 'unknown'
        primary_btts = 'yes' if btts.get('yes', 0) > btts.get('no', 0) else 'no'
        primary_over_under = 'over_25' if over_under.get('over_25', 0) > over_under.get('under_25', 0) else 'under_25'
        
        for market_name, pure_prob, market_key in probability_mapping:
            market_odd = market_odds.get(market_key, 0)
            if market_odd <= 1:
                continue
                
            ev_data = self.calculate_expected_value(pure_prob, market_odd)
            edge_percentage = ev_data['edge_percentage']
            
            # Professional edge threshold
            if edge_percentage >= 4.0:
                value_rating = self._get_professional_value_rating(edge_percentage)
                confidence = self._assign_professional_confidence(pure_prob, edge_percentage, data_quality)
                stake = self.professional_kelly_stake(pure_prob, market_odd, confidence)
                
                # Professional alignment assessment
                alignment = self._assess_professional_alignment(market_name, primary_outcome, primary_btts, primary_over_under)
                
                # Professional explanations
                market_explanations = []
                if 'BTTS' in market_name:
                    market_explanations = explanations.get('btts', [])
                elif 'Over' in market_name or 'Under' in market_name:
                    market_explanations = explanations.get('over_under', [])
                elif '1x2' in market_name:
                    market_explanations = explanations.get('quality', []) + explanations.get('style', [])
                
                signal = BettingSignal(
                    market=market_name, 
                    model_prob=round(pure_prob * 100, 1),
                    book_prob=round(ev_data['implied_probability'] * 100, 1), 
                    edge=round(edge_percentage, 1),
                    confidence=confidence, 
                    recommended_stake=stake, 
                    value_rating=value_rating,
                    explanation=market_explanations,
                    alignment=alignment
                )
                signals.append(signal)
        
        # Professional sorting by edge and confidence
        signals.sort(key=lambda x: (x.edge, self._confidence_weight(x.confidence)), reverse=True)
        return signals
    
    def _assess_professional_alignment(self, market: str, primary_outcome: str, 
                                     primary_btts: str, primary_over_under: str) -> str:
        """Professional alignment assessment"""
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
        """Professional confidence weighting"""
        weights = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'SPECULATIVE': 0}
        return weights.get(confidence, 0)

class AdvancedFootballPredictor:
    """MAIN PROFESSIONAL PREDICTOR - Money-Grade Analysis"""
    
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
        """Generate professional-grade analysis"""
        football_predictions = self.apex_engine.generate_professional_predictions(mc_iterations)
        
        # Professional value detection
        explanations = football_predictions.get('explanations', {})
        data_quality = football_predictions.get('data_quality_score', 0)
        value_signals = self.betting_engine.detect_professional_value_bets(
            football_predictions, self.market_odds, explanations, data_quality
        )
        
        # Professional system validation
        alignment_status = "PERFECT" if self._validate_professional_alignment(football_predictions, value_signals) else "PARTIAL"
        
        professional_result = football_predictions.copy()
        professional_result['betting_signals'] = [signal.__dict__ for signal in value_signals]
        professional_result['system_validation'] = {
            'status': 'PROFESSIONAL', 
            'alignment': alignment_status,
            'engine_sync': 'ELITE',
            'model_version': '2.0.0_professional',
            'calibration_level': 'MONEY_GRADE'
        }
        
        self.predictions = professional_result
        return professional_result
    
    def _validate_professional_alignment(self, football_predictions: Dict, value_signals: List[BettingSignal]) -> bool:
        """Professional alignment validation"""
        if not value_signals:
            return True
            
        # Check for major contradictions
        primary_outcome = max(
            football_predictions['probabilities']['match_outcomes'], 
            key=football_predictions['probabilities']['match_outcomes'].get
        )
        
        for signal in value_signals:
            market = signal.market
            signal_dict = signal.__dict__ if hasattr(signal, '__dict__') else signal
            
            # Major contradiction check
            if (market in ['1x2 Away', '1x2 Draw'] and primary_outcome == 'home_win' and 
                football_predictions['probabilities']['match_outcomes']['home_win'] > 65):
                return False
            elif (market in ['1x2 Home', '1x2 Draw'] and primary_outcome == 'away_win' and 
                  football_predictions['probabilities']['match_outcomes']['away_win'] > 65):
                return False
                
        return True

# PROFESSIONAL TEST FUNCTION
def test_professional_predictor():
    """Professional test function"""
    match_data = {
        'home_team': 'Lazio', 'away_team': 'Cagliari', 'league': 'serie_a',
        'home_goals': 7, 'away_goals': 6, 'home_conceded': 4, 'away_conceded': 10,
        'home_goals_home': 4, 'away_goals_away': 5,
        'home_form': [3, 3, 0, 3, 1, 3], 'away_form': [0, 1, 0, 1, 0, 0],
        'h2h_data': {'matches': 6, 'home_wins': 5, 'away_wins': 0, 'draws': 1, 'home_goals': 13, 'away_goals': 5},
        'motivation': {'home': 'High', 'away': 'Normal'},
        'injuries': {'home': 4, 'away': 3},
        'market_odds': {
            '1x2 Home': 1.62, '1x2 Draw': 3.60, '1x2 Away': 6.00,
            'Over 2.5 Goals': 2.10, 'Under 2.5 Goals': 1.73,
            'BTTS Yes': 2.05, 'BTTS No': 1.70
        },
        'bankroll': 1000,
        'kelly_fraction': 0.2
    }
    
    predictor = AdvancedFootballPredictor(match_data)
    results = predictor.generate_comprehensive_analysis()
    
    print("🎯 PROFESSIONAL FOOTBALL PREDICTION RESULTS")
    print("=" * 70)
    print(f"Match: {results['match']}")
    print(f"League: {results['league'].upper()}")
    print(f"Football IQ: {results['apex_intelligence']['football_iq_score']}/100")
    print(f"Calibration: {results['apex_intelligence']['calibration_status']}")
    print(f"Risk Level: {results['risk_assessment']['risk_level']}")
    print()
    
    print("📊 PROFESSIONAL PROBABILITIES:")
    outcomes = results['probabilities']['match_outcomes']
    print(f"Home Win: {outcomes['home_win']}% | Draw: {outcomes['draw']}% | Away Win: {outcomes['away_win']}%")
    print(f"BTTS Yes: {results['probabilities']['both_teams_score']['yes']}%")
    print(f"Over 2.5: {results['probabilities']['over_under']['over_25']}%")
    print()
    
    print("🎯 PROFESSIONAL NARRATIVE:")
    narrative = results['match_narrative']
    print(f"Quality Gap: {narrative['quality_gap']} | Pattern: {narrative['primary_pattern']}")
    print(f"Style: {narrative['style_conflict']} | Defense: {narrative['defensive_stability']}")
    print()
    
    print("📝 PROFESSIONAL SUMMARY:")
    print(results['summary'])
    print()
    
    print("💰 PROFESSIONAL VALUE BETS:")
    for signal in results.get('betting_signals', []):
        alignment_emoji = "✅" if signal['alignment'] == 'aligns_with_primary' else "⚠️"
        print(f"- {alignment_emoji} {signal['market']}: {signal['edge']}% edge | Stake: ${signal['recommended_stake']:.2f} | Confidence: {signal['confidence']}")

if __name__ == "__main__":
    test_professional_predictor()
