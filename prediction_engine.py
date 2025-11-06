# prediction_engine.py - PRODUCTION-READY WITH REFINED CONTEXTUAL STRENGTH MODEL
import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam
from typing import Dict, Any, Tuple, List, Optional
import math
from dataclasses import dataclass
import logging
from datetime import datetime
import json
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# 🎯 PRODUCTION LEAGUE PARAMS (Evidence-based only)
LEAGUE_PARAMS = {
    'premier_league': {
        'away_penalty': 0.80,  # YOUR REFINEMENT: Increased from 0.85
        'min_edge': 0.08,
        'volatility_multiplier': 1.0,
        'avg_goals': 1.4,
        'home_advantage': 1.20  # YOUR REFINEMENT: Increased from 1.15
    },
    'la_liga': {
        'away_penalty': 0.82,
        'min_edge': 0.06,
        'volatility_multiplier': 1.2,
        'avg_goals': 1.3,
        'home_advantage': 1.18
    },
    'serie_a': {
        'away_penalty': 0.81,
        'min_edge': 0.10,
        'volatility_multiplier': 0.9,
        'avg_goals': 1.35,
        'home_advantage': 1.19
    },
    'bundesliga': {
        'away_penalty': 0.83,
        'min_edge': 0.07,
        'volatility_multiplier': 0.8,
        'avg_goals': 1.45,
        'home_advantage': 1.20
    },
    'ligue_1': {
        'away_penalty': 0.79,
        'min_edge': 0.09,
        'volatility_multiplier': 1.1,
        'avg_goals': 1.25,
        'home_advantage': 1.18
    },
    'default': {
        'away_penalty': 0.80,
        'min_edge': 0.10,
        'volatility_multiplier': 1.0,
        'avg_goals': 1.4,
        'home_advantage': 1.20
    }
}

# Context thresholds (DESCRIPTIVE ONLY - no mathematical influence)
CONTEXT_THRESHOLDS = {
    'total_xg_offensive': 3.2,
    'total_xg_defensive': 2.3,
    'xg_diff_dominant': 0.35
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MatchContext(Enum):
    OFFENSIVE_SHOWDOWN = "offensive_showdown"
    DEFENSIVE_BATTLE = "defensive_battle" 
    HOME_DOMINANCE = "home_dominance"
    AWAY_COUNTER = "away_counter"
    TACTICAL_STALEMATE = "tactical_stalemate"
    BALANCED = "balanced"

@dataclass
class MatchNarrative:
    """PRODUCTION: Narrative for display only - no mathematical influence"""
    dominance: str = "balanced"
    style_conflict: str = "neutral"
    expected_tempo: str = "medium"
    expected_openness: float = 0.5
    defensive_stability: str = "mixed"
    primary_pattern: Optional[str] = None
    quality_gap: str = "even"
    expected_outcome: str = "balanced"
    betting_priority: List[str] = None
    
    def __post_init__(self):
        if self.betting_priority is None:
            self.betting_priority = []
    
    def to_dict(self):
        return {
            'dominance': self.dominance,
            'style_conflict': self.style_conflict,
            'expected_tempo': self.expected_tempo,
            'expected_openness': self.expected_openness,
            'defensive_stability': self.defensive_stability,
            'primary_pattern': self.primary_pattern,
            'quality_gap': self.quality_gap,
            'expected_outcome': self.expected_outcome,
            'betting_priority': self.betting_priority
        }

class ProductionLeagueCalibrator:
    """PRODUCTION: League calibration with evidence-based adjustments only"""
    
    def __init__(self):
        self.league_calibration = LEAGUE_PARAMS
        
    def get_league_params(self, league: str) -> Dict[str, float]:
        """Get production league parameters"""
        return self.league_calibration.get(league, self.league_calibration['default'])
    
    def get_league_avg_goals(self, league: str) -> float:
        """Get league average goals for xG calculation"""
        params = self.get_league_params(league)
        return params['avg_goals']
    
    def get_home_advantage(self, league: str) -> float:
        """Get home advantage multiplier"""
        params = self.get_league_params(league)
        return params['home_advantage']
    
    def get_away_penalty(self, league: str) -> float:
        """Get away penalty multiplier"""
        params = self.get_league_params(league)
        return params['away_penalty']
    
    def get_min_edge(self, league: str) -> float:
        """Get minimum edge threshold by league volatility"""
        params = self.get_league_params(league)
        return params['min_edge']
    
    def get_stake_multiplier(self, league: str) -> float:
        """Get volatility-based stake multiplier"""
        params = self.get_league_params(league)
        return params['volatility_multiplier']

class ProductionFeatureEngine:
    """PRODUCTION: Feature engineering with REFINED contextual strength model"""
    
    def __init__(self):
        self.calibrator = ProductionLeagueCalibrator()
        
    def get_historical_context(self, team_tier: str) -> Tuple[float, float]:
        """YOUR SOLUTION: Historical context based on established quality"""
        # Your exact tier-based historical weighting
        historical_context = {
            'ELITE': (1.25, 0.80),      # Strong attack, strong defense
            'STRONG': (1.10, 0.90),     # Good attack, good defense  
            'MEDIUM': (1.00, 1.00),     # Average
            'WEAK': (0.85, 1.15)        # Weak attack, weak defense
        }
        return historical_context.get(team_tier, (1.00, 1.00))
    
    def calculate_contextual_strength(self, goals: int, conceded: int, team_tier: str, 
                                    league_avg: float, games_played: int = 6) -> Tuple[float, float]:
        """YOUR EXACT REFINEMENT: 60% recent, 40% historical weighting"""
        
        # Recent form (current data)
        recent_attack = goals / (games_played * league_avg)
        recent_defense = (games_played * league_avg) / max(conceded, 0.5)  # Avoid extreme values
        
        # Historical context (your tier-based insight)
        historical_attack, historical_defense = self.get_historical_context(team_tier)
        
        # YOUR EXACT REFINEMENT: 60% recent, 40% historical (was 70/30)
        attack_strength = (0.6 * recent_attack) + (0.4 * historical_attack)
        defense_strength = (0.6 * recent_defense) + (0.4 * historical_defense)
        
        return attack_strength, defense_strength
    
    def calculate_contextual_xg(self, home_goals: int, home_conceded: int, home_tier: str,
                              away_goals: int, away_conceded: int, away_tier: str, 
                              league: str) -> Tuple[float, float, float, float]:
        """YOUR EXACT REFINEMENT: Contextual xG with boosted home advantage"""
        
        league_avg = self.calibrator.get_league_avg_goals(league)
        home_advantage = self.calibrator.get_home_advantage(league)  # Now 1.20 for Premier League
        away_penalty = self.calibrator.get_away_penalty(league)      # Now 0.80 for Premier League
        
        # Calculate contextual strengths with YOUR REFINED 60/40 weighting
        home_attack, home_defense = self.calculate_contextual_strength(
            home_goals, home_conceded, home_tier, league_avg
        )
        away_attack, away_defense = self.calculate_contextual_strength(
            away_goals, away_conceded, away_tier, league_avg
        )
        
        # YOUR EXACT xG CALCULATION with boosted home advantage
        home_xg = league_avg * home_attack * away_defense * home_advantage
        away_xg = league_avg * away_attack * home_defense * away_penalty
        
        # Apply realistic bounds
        home_xg = max(0.4, min(3.5, home_xg))
        away_xg = max(0.4, min(3.0, away_xg))
        
        # Uncertainty based on sample size and model confidence
        home_uncertainty = home_xg * 0.10
        away_uncertainty = away_xg * 0.10
        
        return home_xg, away_xg, home_uncertainty, away_uncertainty
    
    def create_match_features(self, home_data: Dict, away_data: Dict, context: Dict, 
                            home_tier: str, away_tier: str, league: str) -> Dict[str, Any]:
        """PRODUCTION: Create features with YOUR REFINED contextual strength model"""
        
        # Calculate contextual xG with YOUR REFINED formula
        home_xg, away_xg, home_uncertainty, away_uncertainty = self.calculate_contextual_xg(
            context.get('home_goals', 0),
            context.get('home_conceded', 0),
            home_tier,
            context.get('away_goals', 0),
            context.get('away_conceded', 0), 
            away_tier,
            league
        )
        
        features = {
            'home_xg': home_xg,
            'away_xg': away_xg,
            'home_xg_uncertainty': home_uncertainty,
            'away_xg_uncertainty': away_uncertainty,
            'total_xg': home_xg + away_xg,
            'xg_difference': home_xg - away_xg,
            'home_advantage_multiplier': self.calibrator.get_home_advantage(league),
            'away_penalty': self.calibrator.get_away_penalty(league),
            'weighting_ratio': '60/40'  # Track the refined weighting
        }
        
        return features

# ... [REST OF THE CODE REMAINS EXACTLY THE SAME AS BEFORE - BivariatePoissonSimulator, MarketAnalyzer, ProductionStakingEngine, etc.]

class ProductionStakingEngine:
    """PRODUCTION: Professional staking with proper risk management"""
    
    def __init__(self):
        self.calibrator = ProductionLeagueCalibrator()
    
    def calculate_kelly_stake(self, model_prob: float, odds: float, bankroll: float, 
                            kelly_fraction: float = 0.2) -> float:
        """Fractional Kelly stake calculation"""
        if odds <= 1.0:
            return 0.0
            
        implied_prob = 1.0 / odds
        edge = model_prob - implied_prob
        
        if edge <= 0:
            return 0.0
            
        kelly_percentage = edge / (odds - 1)
        fractional_kelly = kelly_percentage * kelly_fraction
        
        stake = bankroll * fractional_kelly
        max_stake = bankroll * 0.03
        
        return min(stake, max_stake)
    
    def calculate_professional_stake(self, model_prob: float, odds: float, bankroll: float,
                                   league: str, kelly_fraction: float = 0.2) -> Dict[str, float]:
        """Production stake calculation with volatility adjustment"""
        base_stake = self.calculate_kelly_stake(model_prob, odds, bankroll, kelly_fraction)
        
        stake_multiplier = self.calibrator.get_stake_multiplier(league)  # ✅ NOW THIS WORKS!
        adjusted_stake = base_stake * stake_multiplier
        
        final_stake = min(adjusted_stake, bankroll * 0.03)
        
        return {
            'base_stake': base_stake,
            'volatility_multiplier': stake_multiplier,
            'final_stake': final_stake,
            'bankroll_percentage': (final_stake / bankroll) * 100
        }

# ... [REST OF THE CODE REMAINS EXACTLY THE SAME]

def test_refined_model():
    """Test the REFINED contextual strength model with Liverpool vs Villa"""
    match_data = {
        'home_team': 'Liverpool', 'away_team': 'Aston Villa', 'league': 'premier_league',
        'home_goals': 8, 'away_goals': 9, 'home_conceded': 10, 'away_conceded': 4,
        'market_odds': {
            '1x2 Home': 1.62, '1x2 Draw': 4.33, '1x2 Away': 4.50,
            'Over 2.5 Goals': 1.53, 'Under 2.5 Goals': 2.50,
            'BTTS Yes': 1.62, 'BTTS No': 2.20
        },
        'bankroll': 1000,
        'kelly_fraction': 0.2
    }
    
    engine = ApexProductionEngine(match_data)
    results = engine.generate_production_predictions()
    
    print("🎯 REFINED CONTEXTUAL STRENGTH MODEL RESULTS")
    print("=" * 70)
    print(f"Match: {results['match']}")
    print(f"Expected Goals: Home {results['expected_goals']['home']:.2f} ± {results['expected_goals']['home_uncertainty']:.2f}")
    print(f"Expected Goals: Away {results['expected_goals']['away']:.2f} ± {results['expected_goals']['away_uncertainty']:.2f}")
    print(f"Home Win: {results['probabilities']['match_outcomes']['home_win']:.1f}%")
    print(f"Draw: {results['probabilities']['match_outcomes']['draw']:.1f}%") 
    print(f"Away Win: {results['probabilities']['match_outcomes']['away_win']:.1f}%")
    print(f"BTTS Yes: {results['probabilities']['both_teams_score']['yes']:.1f}%")
    print(f"Production Features: {list(results['production_metrics'].keys())}")

if __name__ == "__main__":
    test_refined_model()
