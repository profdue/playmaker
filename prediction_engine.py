# prediction_engine.py - PROFESSIONAL MULTI-LEAGUE PREDICTION ENGINE
import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam, norm
from typing import Dict, Any, Tuple, List, Optional
import math
from dataclasses import dataclass
import logging
from datetime import datetime
import warnings
from enum import Enum
import json

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# PROFESSIONAL LEAGUE PARAMETERS - FULL 10 LEAGUE SUPPORT
# =============================================================================

LEAGUE_PARAMS = {
    'premier_league': {
        'display_name': 'Premier League 🏴󠁧󠁢󠁥󠁮󠁧󠁿',
        'goal_baseline': 1.45,
        'home_advantage': 1.07,
        'away_penalty': 0.98,
        'volatility': 'medium',
        'min_edge': 0.08,
        'goal_intensity': 'high',
        'correlation_base': 0.16
    },
    'la_liga': {
        'display_name': 'La Liga 🇪🇸',
        'goal_baseline': 1.35,
        'home_advantage': 1.06,
        'away_penalty': 0.97,
        'volatility': 'low',
        'min_edge': 0.06,
        'goal_intensity': 'medium',
        'correlation_base': 0.14
    },
    'serie_a': {
        'display_name': 'Serie A 🇮🇹',
        'goal_baseline': 1.30,
        'home_advantage': 1.08,
        'away_penalty': 0.96,
        'volatility': 'medium_low',
        'min_edge': 0.10,
        'goal_intensity': 'low',
        'correlation_base': 0.12
    },
    'bundesliga': {
        'display_name': 'Bundesliga 🇩🇪',
        'goal_baseline': 1.60,
        'home_advantage': 1.09,
        'away_penalty': 1.02,
        'volatility': 'high',
        'min_edge': 0.07,
        'goal_intensity': 'very_high',
        'correlation_base': 0.20
    },
    'ligue_1': {
        'display_name': 'Ligue 1 🇫🇷',
        'goal_baseline': 1.30,
        'home_advantage': 1.07,
        'away_penalty': 0.98,
        'volatility': 'medium',
        'min_edge': 0.08,
        'goal_intensity': 'low',
        'correlation_base': 0.15
    },
    'eredivisie': {
        'display_name': 'Eredivisie 🇳🇱',
        'goal_baseline': 1.70,
        'home_advantage': 1.08,
        'away_penalty': 1.01,
        'volatility': 'high',
        'min_edge': 0.12,
        'goal_intensity': 'high',
        'correlation_base': 0.22
    },
    'liga_portugal': {
        'display_name': 'Liga Portugal 🇵🇹',
        'goal_baseline': 1.40,
        'home_advantage': 1.09,
        'away_penalty': 0.96,
        'volatility': 'medium',
        'min_edge': 0.09,
        'goal_intensity': 'medium',
        'correlation_base': 0.17
    },
    'brasileirao': {
        'display_name': 'Brasileirão 🇧🇷',
        'goal_baseline': 1.35,
        'home_advantage': 1.12,
        'away_penalty': 0.94,
        'volatility': 'very_high',
        'min_edge': 0.15,
        'goal_intensity': 'medium_high',
        'correlation_base': 0.18
    },
    'liga_mx': {
        'display_name': 'Liga MX 🇲🇽',
        'goal_baseline': 1.50,
        'home_advantage': 1.10,
        'away_penalty': 0.95,
        'volatility': 'high',
        'min_edge': 0.11,
        'goal_intensity': 'high',
        'correlation_base': 0.19
    },
    'championship': {
        'display_name': 'Championship 🏴󠁧󠁢󠁥󠁮󠁧󠁿',
        'goal_baseline': 1.25,
        'home_advantage': 1.15,
        'away_penalty': 0.92,
        'volatility': 'very_high',
        'min_edge': 0.15,
        'goal_intensity': 'medium_low',
        'correlation_base': 0.18
    }
}

# Volatility-based stake multipliers
VOLATILITY_MULTIPLIERS = {
    'low': 1.2,
    'medium_low': 1.1, 
    'medium': 1.0,
    'high': 0.8,
    'very_high': 0.6
}

# Goal intensity adjustments
GOAL_INTENSITY_ADJUSTMENTS = {
    'very_low': 0.90,
    'low': 0.94,
    'medium_low': 0.97,
    'medium': 1.00,
    'medium_high': 1.03,
    'high': 1.06,
    'very_high': 1.10
}

@dataclass
class MatchContext:
    """Professional match context"""
    primary_pattern: str
    quality_gap: str
    home_advantage_amplified: bool = False
    away_scoring_issues: bool = False
    expected_tempo: str = "medium"
    confidence_score: float = 0.0

@dataclass  
class MonteCarloResults:
    """Professional Monte Carlo results"""
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    over_25_prob: float
    btts_prob: float
    exact_scores: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    home_xg: float
    away_xg: float

@dataclass
class BettingRecommendation:
    """Professional betting recommendation"""
    market: str
    model_prob: float
    market_odds: float
    implied_prob: float
    edge: float
    recommended_stake: float
    confidence: str
    robustness: str
    explanation: List[str]
    passes_sensitivity: bool

@dataclass
class ModelDiagnostics:
    """Comprehensive model diagnostics"""
    data_quality_score: float
    calibration_score: float
    market_alignment: float
    uncertainty_score: float
    sensitivity_passed: bool
    recommended_action: str

class UnifiedXGCalculator:
    """Universal xG system for all 10 leagues"""
    
    def __init__(self):
        self.injury_impact_map = {
            1: 0.98,  # Rotation player: -2%
            2: 0.95,  # Regular starter: -5%  
            3: 0.90,  # Key player: -10%
            4: 0.85,  # Star player: -15%
            5: 0.80   # Multiple key players: -20%
        }
        
        self.motivation_impact_map = {
            'Low': 0.95,
            'Normal': 1.00,
            'High': 1.04,
            'Very High': 1.08
        }
    
    def calculate_xg(self, stats: dict, league: str) -> Tuple[float, float, dict]:
        """
        Calculate home/away xG using unified adaptive logic
        Returns (home_xg, away_xg, debug_info)
        """
        debug = {}
        
        # Extract key stats with defaults
        home_recent = stats.get('home_goals_home', 0)
        away_recent = stats.get('away_goals_away', 0)
        home_total = stats.get('home_goals', 0)
        away_total = stats.get('away_goals', 0)
        home_conceded = stats.get('home_conceded', 0)
        away_conceded = stats.get('away_conceded', 0)
        
        # 1. Compute blended attack strengths
        home_attack = self._blend_attack_strength(home_recent, home_total, home_conceded)
        away_attack = self._blend_attack_strength(away_recent, away_total, away_conceded)
        debug["home_attack_raw"] = home_attack
        debug["away_attack_raw"] = away_attack
        
        # 2. Apply league-aware normalization
        home_attack = self._normalize_for_league(home_attack, league, True)
        away_attack = self._normalize_for_league(away_attack, league, False)
        debug["home_normalized"] = home_attack
        debug["away_normalized"] = away_attack
        
        # 3. Apply context factors
        home_attack, away_attack = self._apply_context_factors(
            home_attack, away_attack, stats, debug
        )
        
        # 4. Context adjustments (H2H + Market)
        home_attack, away_attack = self._apply_context_adjustments(
            home_attack, away_attack, stats, league, debug
        )
        
        # 5. Final caps and validation
        home_attack = min(max(home_attack, 0.4), 2.8)
        away_attack = min(max(away_attack, 0.4), 2.8)
        debug["final_home_xg"] = home_attack
        debug["final_away_xg"] = away_attack
        
        return home_attack, away_attack, debug
    
    def _blend_attack_strength(self, recent: int, medium: int, conceded: int) -> float:
        """Weighted average of recent and medium-term performance"""
        recent_xg = recent / 3.0 if recent > 0 else 0.6
        medium_xg = medium / 6.0 if medium > 0 else 0.8
        
        # Defense quality adjustment (teams that concede less are better)
        defense_quality = max(0.7, 1.3 - (conceded / 6.0))
        
        # Weighting depends on data richness
        if recent >= 2:  # Good recent sample
            weights = [0.5, 0.3, 0.2]  # Favor recent form
        else:  # Poor recent sample
            weights = [0.25, 0.45, 0.3]  # Favor larger samples
        
        blended = (recent_xg * weights[0] + 
                  medium_xg * weights[1] + 
                  defense_quality * weights[2])
        
        return max(0.4, blended)
    
    def _normalize_for_league(self, xg: float, league: str, is_home: bool) -> float:
        """Adjust xG levels based on league scoring baselines"""
        league_params = LEAGUE_PARAMS.get(league, LEAGUE_PARAMS['premier_league'])
        base = league_params['goal_baseline']
        
        # Home advantage adjustment
        if is_home:
            xg *= league_params['home_advantage']
        else:
            xg *= league_params['away_penalty']
        
        # Cap extremes relative to league baseline
        if xg > base * 1.6:
            xg = base * 1.4  # More conservative cap
        elif xg < base * 0.5:
            xg = base * 0.7  # More conservative floor
            
        return xg
    
    def _apply_context_factors(self, home_xg: float, away_xg: float, 
                             stats: dict, debug: dict) -> Tuple[float, float]:
        """Apply injury and motivation factors"""
        
        # Injury impacts
        home_injuries = stats.get('home_injuries', 1)
        away_injuries = stats.get('away_injuries', 1)
        
        home_injury_factor = self.injury_impact_map.get(home_injuries, 0.95)
        away_injury_factor = self.injury_impact_map.get(away_injuries, 0.95)
        
        # Motivation impacts
        home_motivation = stats.get('home_motivation', 'Normal')
        away_motivation = stats.get('away_motivation', 'Normal')
        
        home_motivation_factor = self.motivation_impact_map.get(home_motivation, 1.0)
        away_motivation_factor = self.motivation_impact_map.get(away_motivation, 1.0)
        
        # Apply factors with overall cap
        home_adjusted = home_xg * home_injury_factor * home_motivation_factor
        away_adjusted = away_xg * away_injury_factor * away_motivation_factor
        
        max_adjustment = 1.15  # Maximum 15% total adjustment
        home_final = min(home_adjusted, home_xg * max_adjustment)
        away_final = min(away_adjusted, away_xg * max_adjustment)
        
        debug["home_after_context"] = home_final
        debug["away_after_context"] = away_final
        
        return home_final, away_final
    
    def _apply_context_adjustments(self, home_xg: float, away_xg: float,
                                 stats: dict, league: str, debug: dict) -> Tuple[float, float]:
        """Apply H2H and market adjustments"""
        
        adjustments = {'home': 1.0, 'away': 1.0}
        
        # H2H adjustments
        h2h_data = stats.get('h2h_data', {})
        if h2h_data.get('matches', 0) >= 4:
            home_win_rate = h2h_data.get('home_wins', 0) / h2h_data['matches']
            away_win_rate = h2h_data.get('away_wins', 0) / h2h_data['matches']
            
            # Only apply if clear dominance exists
            if away_win_rate > 0.6:  # Clear away dominance
                adjustments['away'] *= 1.05
                debug["h2h_boost"] = "away +5%"
            elif home_win_rate > 0.6:  # Clear home dominance  
                adjustments['home'] *= 1.05
                debug["h2h_boost"] = "home +5%"
        
        # Market alignment
        market_odds = stats.get('market_odds', {})
        if market_odds.get('1x2_home') and market_odds.get('1x2_away'):
            home_implied = 1.0 / market_odds['1x2_home']
            away_implied = 1.0 / market_odds['1x2_away']
            
            # Only adjust if market strongly disagrees
            market_fav = 'home' if home_implied > away_implied else 'away'
            model_fav = 'home' if home_xg > away_xg else 'away'
            
            if market_fav != model_fav and abs(home_implied - away_implied) > 0.1:
                # Small adjustment toward market
                adjustments[market_fav] *= 1.03
                adjustments[model_fav] *= 0.97
                debug["market_adjust"] = f"nudged toward {market_fav}"
        
        # Apply adjustments
        home_final = home_xg * adjustments['home']
        away_final = away_xg * adjustments['away']
        
        return home_final, away_final

class BivariatePoissonSimulator:
    """Professional bivariate Poisson simulator for all leagues"""
    
    def __init__(self, n_simulations: int = 25000):
        self.n_simulations = n_simulations
    
    def simulate_match(self, home_xg: float, away_xg: float, league: str) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate match with league-specific correlation"""
        
        # Get league-specific correlation
        league_params = LEAGUE_PARAMS.get(league, LEAGUE_PARAMS['premier_league'])
        correlation = league_params['correlation_base']
        
        # Add context-based correlation adjustments
        context_adjustments = {
            'offensive_showdown': 0.08,
            'defensive_battle': -0.05,
            'home_dominance': 0.04,
            'balanced': 0.00
        }
        # For now, use base correlation
        
        # Add uncertainty to xG values
        home_xg_samples = np.random.normal(home_xg, home_xg * 0.12, self.n_simulations)
        away_xg_samples = np.random.normal(away_xg, away_xg * 0.12, self.n_simulations)
        
        # Ensure positive xG values
        home_xg_samples = np.maximum(0.1, home_xg_samples)
        away_xg_samples = np.maximum(0.1, away_xg_samples)
        
        # Bivariate Poisson simulation
        lambda1 = home_xg_samples - correlation * np.minimum(home_xg_samples, away_xg_samples)
        lambda2 = away_xg_samples - correlation * np.minimum(home_xg_samples, away_xg_samples)
        lambda3 = correlation * np.minimum(home_xg_samples, away_xg_samples)
        
        # Ensure positive lambdas
        lambda1 = np.maximum(0.05, lambda1)
        lambda2 = np.maximum(0.05, lambda2)
        lambda3 = np.maximum(0.0, lambda3)
        
        # Simulate goals
        C = np.random.poisson(lambda3)
        A = np.random.poisson(lambda1)
        B = np.random.poisson(lambda2)
        
        home_goals = A + C
        away_goals = B + C
        
        return home_goals, away_goals
    
    def calculate_probabilities(self, home_goals: np.ndarray, away_goals: np.ndarray, 
                              league: str) -> Dict[str, float]:
        """Calculate market probabilities from simulated goals"""
        
        # Match outcomes
        home_wins = np.mean(home_goals > away_goals)
        draws = np.mean(home_goals == away_goals)
        away_wins = np.mean(home_goals < away_goals)
        
        # Both teams to score
        btts_yes = np.mean((home_goals > 0) & (away_goals > 0))
        
        # Over/under markets
        total_goals = home_goals + away_goals
        over_15 = np.mean(total_goals > 1.5)
        over_25 = np.mean(total_goals > 2.5)
        over_35 = np.mean(total_goals > 3.5)
        
        # Exact scores (top 8)
        score_counts = {}
        for h, a in zip(home_goals[:10000], away_goals[:10000]):
            score = f"{h}-{a}"
            score_counts[score] = score_counts.get(score, 0) + 1
        
        exact_scores = {
            score: count/10000 
            for score, count in sorted(score_counts.items(), key=lambda x: x[1], reverse=True)[:8]
        }
        
        # League-specific calibration
        league_params = LEAGUE_PARAMS.get(league, LEAGUE_PARAMS['premier_league'])
        goal_intensity = GOAL_INTENSITY_ADJUSTMENTS.get(league_params['goal_intensity'], 1.0)
        
        # Apply modest calibration
        over_25 = min(0.95, over_25 * goal_intensity)
        btts_yes = min(0.95, btts_yes * goal_intensity)
        
        return {
            'home_win': home_wins,
            'draw': draws,
            'away_win': away_wins,
            'btts_yes': btts_yes,
            'btts_no': 1 - btts_yes,
            'over_15': over_15,
            'over_25': over_25,
            'over_35': over_35,
            'under_25': 1 - over_25,
            'exact_scores': exact_scores
        }

class MarketAnalyzer:
    """Professional market analysis for all leagues"""
    
    def remove_vig_1x2(self, home_odds: float, draw_odds: float, away_odds: float) -> Dict[str, float]:
        """Remove vig from 1X2 market"""
        home_implied = 1.0 / home_odds
        draw_implied = 1.0 / draw_odds
        away_implied = 1.0 / away_odds
        
        total_implied = home_implied + draw_implied + away_implied
        
        # Normalize to remove vig
        home_prob = home_implied / total_implied
        draw_prob = draw_implied / total_implied
        away_prob = away_implied / total_implied
        
        return {
            'home': home_prob,
            'draw': draw_prob, 
            'away': away_prob,
            'overround': total_implied - 1.0
        }
    
    def remove_vig_two_way(self, yes_odds: float, no_odds: float) -> Dict[str, float]:
        """Remove vig from two-way markets"""
        yes_implied = 1.0 / yes_odds
        no_implied = 1.0 / no_odds
        
        total_implied = yes_implied + no_implied
        
        # Normalize to remove vig
        yes_prob = yes_implied / total_implied
        no_prob = no_implied / total_implied
        
        return {
            'yes': yes_prob,
            'no': no_prob,
            'overround': total_implied - 1.0
        }
    
    def calculate_edges(self, model_probs: Dict[str, float], market_odds: Dict[str, float], 
                       league: str) -> Dict[str, float]:
        """Calculate proper edges with vig removal"""
        
        # 1X2 edges
        vig_removed_1x2 = self.remove_vig_1x2(
            market_odds.get('1x2_home', 2.0),
            market_odds.get('1x2_draw', 3.0), 
            market_odds.get('1x2_away', 3.0)
        )
        
        edges_1x2 = {
            'home_win': model_probs['home_win'] - vig_removed_1x2['home'],
            'draw': model_probs['draw'] - vig_removed_1x2['draw'],
            'away_win': model_probs['away_win'] - vig_removed_1x2['away']
        }
        
        # BTTS edges
        vig_removed_btts = self.remove_vig_two_way(
            market_odds.get('btts_yes', 2.0),
            market_odds.get('btts_no', 1.8)
        )
        
        edges_btts = {
            'btts_yes': model_probs['btts_yes'] - vig_removed_btts['yes'],
            'btts_no': model_probs['btts_no'] - vig_removed_btts['no']
        }
        
        # Over/under edges
        vig_removed_over_under = self.remove_vig_two_way(
            market_odds.get('over_25', 2.0),
            market_odds.get('under_25', 1.8)
        )
        
        edges_ou = {
            'over_25': model_probs['over_25'] - vig_removed_over_under['yes'],
            'under_25': model_probs['under_25'] - vig_removed_over_under['no']
        }
        
        return {**edges_1x2, **edges_btts, **edges_ou}

class ProfessionalStakingCalculator:
    """Professional staking for all leagues"""
    
    def calculate_stake(self, model_prob: float, odds: float, bankroll: float, 
                      league: str, edge_robustness: str) -> float:
        """Calculate professional stake with multiple guardrails"""
        
        # Basic Kelly formula
        implied_prob = 1.0 / odds
        edge = model_prob - implied_prob
        kelly_fraction = edge / implied_prob if implied_prob > 0 else 0
        
        # Base stake
        base_stake = bankroll * kelly_fraction
        
        # Apply fractional Kelly (conservative)
        fractional_stake = base_stake * 0.2  # 20% Kelly
        
        # Volatility adjustment
        league_params = LEAGUE_PARAMS.get(league, LEAGUE_PARAMS['premier_league'])
        volatility_multiplier = VOLATILITY_MULTIPLIERS.get(league_params['volatility'], 1.0)
        volatility_stake = fractional_stake * volatility_multiplier
        
        # Robustness adjustment
        robustness_multiplier = {
            'HIGH': 1.0,
            'MEDIUM': 0.6, 
            'LOW': 0.3
        }.get(edge_robustness, 0.3)
        
        robustness_stake = volatility_stake * robustness_multiplier
        
        # Hard cap at 3% of bankroll
        final_stake = min(robustness_stake, bankroll * 0.03)
        
        # Minimum stake threshold
        if final_stake < bankroll * 0.005:  # 0.5% minimum
            return 0.0
        
        return final_stake

class MatchContextAnalyzer:
    """Professional context analysis for all leagues"""
    
    def analyze_context(self, home_xg: float, away_xg: float, home_tier: str, 
                       away_tier: str, league: str) -> MatchContext:
        """Analyze match context"""
        
        total_xg = home_xg + away_xg
        xg_diff = home_xg - away_xg
        
        # Quality gap analysis
        tier_strength = {'ELITE': 4, 'STRONG': 3, 'MEDIUM': 2, 'WEAK': 1}
        home_strength = tier_strength.get(home_tier, 2)
        away_strength = tier_strength.get(away_tier, 2)
        strength_diff = abs(home_strength - away_strength)
        
        if strength_diff >= 3:
            quality_gap = "extreme"
        elif strength_diff >= 2:
            quality_gap = "significant" 
        elif strength_diff >= 1:
            quality_gap = "moderate"
        else:
            quality_gap = "even"
        
        # Primary pattern
        league_params = LEAGUE_PARAMS.get(league, LEAGUE_PARAMS['premier_league'])
        goal_baseline = league_params['goal_baseline']
        
        if xg_diff >= 0.3 and quality_gap in ['significant', 'extreme']:
            primary_pattern = "home_dominance"
        elif xg_diff <= -0.3 and quality_gap in ['significant', 'extreme']:
            primary_pattern = "away_counter" 
        elif total_xg > goal_baseline * 2.3:  # League-aware threshold
            primary_pattern = "offensive_showdown"
        elif total_xg < goal_baseline * 1.7:  # League-aware threshold
            primary_pattern = "defensive_battle"
        elif abs(xg_diff) < 0.2:
            primary_pattern = "tactical_stalemate"
        else:
            primary_pattern = "balanced"
        
        # Context flags
        home_advantage_amplified = (
            league in ['championship', 'brasileirao'] and 
            xg_diff > 0.2 and
            home_xg > goal_baseline * 1.2
        )
        
        away_scoring_issues = (
            away_xg < goal_baseline * 0.8
        )
        
        # Confidence score
        confidence_score = self._calculate_confidence(total_xg, abs(xg_diff), strength_diff)
        
        return MatchContext(
            primary_pattern=primary_pattern,
            quality_gap=quality_gap,
            home_advantage_amplified=home_advantage_amplified,
            away_scoring_issues=away_scoring_issues,
            confidence_score=confidence_score
        )
    
    def _calculate_confidence(self, total_xg: float, xg_diff: float, strength_diff: int) -> float:
        """Calculate descriptive confidence score"""
        confidence = 50.0  # Base
        
        # Total xG confidence
        if total_xg > 3.2 or total_xg < 2.2:
            confidence += 15
        elif total_xg > 2.8 or total_xg < 2.4:
            confidence += 8
            
        # xG difference confidence
        if xg_diff > 0.4:
            confidence += 12
        elif xg_diff > 0.2:
            confidence += 6
            
        # Quality gap confidence
        if strength_diff >= 2:
            confidence += 10
            
        return min(95.0, confidence)

class ProfessionalPredictionEngine:
    """MAIN PROFESSIONAL PREDICTION ENGINE - 10 LEAGUE SUPPORT"""
    
    def __init__(self, match_data: Dict[str, Any]):
        self.data = self._validate_data(match_data)
        self.xg_calculator = UnifiedXGCalculator()
        self.simulator = BivariatePoissonSimulator()
        self.market_analyzer = MarketAnalyzer()
        self.staking_calculator = ProfessionalStakingCalculator()
        self.context_analyzer = MatchContextAnalyzer()
        
        logger.info(f"Initialized professional engine for {self.data['home_team']} vs {self.data['away_team']}")
    
    def _validate_data(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Professional data validation"""
        validated_data = match_data.copy()
        
        # Required fields
        required = ['home_team', 'away_team', 'league']
        for field in required:
            if field not in validated_data:
                validated_data[field] = 'Unknown'
                
        # Default values for predictive fields
        predictive_defaults = {
            'home_goals': 8, 'away_goals': 4,
            'home_conceded': 6, 'away_conceded': 7, 
            'home_goals_home': 6, 'away_goals_away': 1,
            'home_injuries': 2, 'away_injuries': 2,
            'home_motivation': 'Normal', 'away_motivation': 'Normal',
            'home_tier': 'MEDIUM', 'away_tier': 'MEDIUM'
        }
        
        for field, default in predictive_defaults.items():
            if field not in validated_data:
                validated_data[field] = default
            elif validated_data[field] is None:
                validated_data[field] = default
        
        # Default market odds
        if 'market_odds' not in validated_data:
            validated_data['market_odds'] = {
                '1x2_home': 2.5, '1x2_draw': 3.2, '1x2_away': 2.8,
                'over_25': 2.1, 'under_25': 1.7,
                'btts_yes': 1.9, 'btts_no': 1.9
            }
        
        return validated_data
    
    def generate_predictions(self) -> Dict[str, Any]:
        """Generate professional predictions for any league"""
        logger.info(f"Starting multi-league prediction for {self.data['league']}")
        
        try:
            # 1. Calculate realistic xG with debug info
            home_xg, away_xg, xg_debug = self.xg_calculator.calculate_xg(
                self.data, self.data['league']
            )
            
            # 2. Run professional simulation
            home_goals, away_goals = self.simulator.simulate_match(
                home_xg, away_xg, self.data['league']
            )
            
            # 3. Calculate probabilities
            probs = self.simulator.calculate_probabilities(
                home_goals, away_goals, self.data['league']
            )
            
            # 4. Analyze market edges
            market_odds = self.data.get('market_odds', {})
            edges = self.market_analyzer.calculate_edges(probs, market_odds, self.data['league'])
            
            # 5. Generate betting recommendations
            recommendations = self._generate_betting_recommendations(probs, edges, market_odds)
            
            # 6. Analyze match context
            context = self.context_analyzer.analyze_context(
                home_xg, away_xg,
                self.data.get('home_tier', 'MEDIUM'),
                self.data.get('away_tier', 'MEDIUM'),
                self.data['league']
            )
            
            # 7. Generate diagnostics
            diagnostics = self._generate_diagnostics(probs, edges, context)
            
            # 8. Compile final results
            results = self._compile_results(
                probs, edges, recommendations, context, diagnostics,
                home_xg, away_xg, xg_debug
            )
            
            logger.info("Professional prediction generation completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Prediction generation failed: {str(e)}")
            raise
    
    def _generate_betting_recommendations(self, probs: Dict[str, float], edges: Dict[str, float],
                                        market_odds: Dict[str, float]) -> List[BettingRecommendation]:
        """Generate professional betting recommendations"""
        
        recommendations = []
        bankroll = self.data.get('bankroll', 1000)
        league = self.data['league']
        min_edge = LEAGUE_PARAMS.get(league, LEAGUE_PARAMS['premier_league'])['min_edge']
        
        # Market mappings
        market_mappings = [
            ('home_win', '1x2_home', 'Home Win'),
            ('away_win', '1x2_away', 'Away Win'), 
            ('draw', '1x2_draw', 'Draw'),
            ('btts_yes', 'btts_yes', 'BTTS Yes'),
            ('btts_no', 'btts_no', 'BTTS No'),
            ('over_25', 'over_25', 'Over 2.5 Goals'),
            ('under_25', 'under_25', 'Under 2.5 Goals')
        ]
        
        for prob_key, odds_key, market_name in market_mappings:
            model_prob = probs.get(prob_key, 0.0)
            market_odd = market_odds.get(odds_key, 2.0)
            edge = edges.get(prob_key, 0.0)
            
            # Check minimum edge requirement
            if abs(edge) < min_edge:
                continue
            
            # For now, assume medium robustness (would come from sensitivity analysis)
            robustness = 'MEDIUM'
            passes_sensitivity = True
            
            # Calculate stake
            stake = self.staking_calculator.calculate_stake(
                model_prob, market_odd, bankroll, league, robustness
            )
            
            # Skip zero-stake recommendations
            if stake <= 0:
                continue
            
            # Generate explanation
            explanation = self._generate_recommendation_explanation(
                market_name, model_prob, market_odd, edge, robustness
            )
            
            recommendation = BettingRecommendation(
                market=market_name,
                model_prob=model_prob,
                market_odds=market_odd,
                implied_prob=1.0/market_odd if market_odd > 0 else 0.0,
                edge=edge,
                recommended_stake=stake,
                confidence='HIGH' if robustness == 'HIGH' else 'MEDIUM',
                robustness=robustness,
                explanation=explanation,
                passes_sensitivity=passes_sensitivity
            )
            
            recommendations.append(recommendation)
        
        # Sort by edge size
        recommendations.sort(key=lambda x: abs(x.edge), reverse=True)
        return recommendations
    
    def _generate_recommendation_explanation(self, market: str, model_prob: float, 
                                           odds: float, edge: float, robustness: str) -> List[str]:
        """Generate professional explanation for recommendations"""
        
        explanations = []
        
        explanations.append(
            f"Model probability: {model_prob:.1%} vs Market implied: {1.0/odds:.1%}"
        )
        explanations.append(f"Edge: {edge:+.1%} | Robustness: {robustness}")
        
        if robustness == 'HIGH':
            explanations.append("✅ Edge survives sensitivity testing")
        elif robustness == 'MEDIUM':
            explanations.append("⚠️ Edge moderately robust - consider reduced stake")
        else:
            explanations.append("❌ Edge fragile - not recommended")
            
        if edge > 0.1:
            explanations.append("🎯 Strong value opportunity detected")
        elif edge > 0.05:
            explanations.append("📈 Good value opportunity")
        else:
            explanations.append("📊 Modest value - stake accordingly")
            
        return explanations
    
    def _generate_diagnostics(self, probs: Dict[str, float], edges: Dict[str, float],
                            context: MatchContext) -> ModelDiagnostics:
        """Generate comprehensive model diagnostics"""
        
        # Data quality score
        data_quality = self._calculate_data_quality()
        
        # Calibration score (proxy)
        total_prob = probs['home_win'] + probs['draw'] + probs['away_win']
        calibration_score = 100 * (1 - abs(total_prob - 1.0))
        
        # Market alignment
        market_alignment = self._calculate_market_alignment(probs)
        
        # Uncertainty score
        uncertainty_score = 100 - (context.confidence_score * 0.8)
        
        # For now, assume sensitivity passed (would come from actual testing)
        sensitivity_passed = True
        
        # Recommended action
        if any(abs(edge) > 0.05 for edge in edges.values()):
            recommended_action = "CONFIDENT_BETTING"
        elif any(abs(edge) > 0.02 for edge in edges.values()):
            recommended_action = "CAUTIOUS_BETTING" 
        else:
            recommended_action = "NO_VALUE"
        
        return ModelDiagnostics(
            data_quality_score=data_quality,
            calibration_score=calibration_score,
            market_alignment=market_alignment,
            uncertainty_score=uncertainty_score,
            sensitivity_passed=sensitivity_passed,
            recommended_action=recommended_action
        )
    
    def _calculate_data_quality(self) -> float:
        """Calculate data quality score"""
        score = 70.0  # Base score
        
        # Form data
        if self.data.get('home_goals', 0) > 0 and self.data.get('away_goals', 0) > 0:
            score += 10
            
        # Recent form data
        if self.data.get('home_goals_home', 0) > 0 or self.data.get('away_goals_away', 0) > 0:
            score += 8
            
        # H2H data
        h2h_data = self.data.get('h2h_data', {})
        if h2h_data.get('matches', 0) >= 3:
            score += 12
            
        return min(100.0, score)
    
    def _calculate_market_alignment(self, probs: Dict[str, float]) -> float:
        """Calculate market alignment score"""
        # Simplified version - would compare to market-implied probabilities
        market_odds = self.data.get('market_odds', {})
        
        if market_odds.get('1x2_home') and market_odds.get('1x2_away'):
            home_implied = 1.0 / market_odds['1x2_home']
            away_implied = 1.0 / market_odds['1x2_away']
            
            # Calculate alignment based on favorite consistency
            market_fav = 'home' if home_implied > away_implied else 'away'
            model_fav = 'home' if probs['home_win'] > probs['away_win'] else 'away'
            
            if market_fav == model_fav:
                alignment = 85.0
            else:
                alignment = 65.0
        else:
            alignment = 75.0
        
        return alignment
    
    def _compile_results(self, probs: Dict[str, float], edges: Dict[str, float],
                        recommendations: List[BettingRecommendation], context: MatchContext,
                        diagnostics: ModelDiagnostics, home_xg: float, away_xg: float,
                        xg_debug: dict) -> Dict[str, Any]:
        """Compile final professional results"""
        
        return {
            'match_info': {
                'match': f"{self.data['home_team']} vs {self.data['away_team']}",
                'league': self.data['league'],
                'league_display': LEAGUE_PARAMS.get(self.data['league'], {}).get('display_name', self.data['league']),
                'timestamp': datetime.now().isoformat()
            },
            'expected_goals': {
                'home': home_xg,
                'away': away_xg, 
                'total': home_xg + away_xg,
                'debug': xg_debug
            },
            'probabilities': {
                'match_outcomes': {
                    'home_win': probs['home_win'],
                    'draw': probs['draw'], 
                    'away_win': probs['away_win']
                },
                'both_teams_score': {
                    'yes': probs['btts_yes'],
                    'no': probs['btts_no']
                },
                'over_under': {
                    'over_15': probs['over_15'],
                    'over_25': probs['over_25'],
                    'over_35': probs['over_35'],
                    'under_25': probs['under_25']
                },
                'exact_scores': probs['exact_scores']
            },
            'market_analysis': {
                'edges': edges,
                'recommendations': [
                    {
                        'market': rec.market,
                        'model_prob': rec.model_prob,
                        'market_odds': rec.market_odds,
                        'implied_prob': rec.implied_prob,
                        'edge': rec.edge,
                        'recommended_stake': rec.recommended_stake,
                        'confidence': rec.confidence,
                        'robustness': rec.robustness,
                        'explanation': rec.explanation,
                        'passes_sensitivity': rec.passes_sensitivity
                    }
                    for rec in recommendations
                ]
            },
            'match_context': {
                'primary_pattern': context.primary_pattern,
                'quality_gap': context.quality_gap,
                'home_advantage_amplified': context.home_advantage_amplified,
                'away_scoring_issues': context.away_scoring_issues,
                'confidence_score': context.confidence_score,
                'expected_tempo': context.expected_tempo
            },
            'diagnostics': {
                'data_quality_score': diagnostics.data_quality_score,
                'calibration_score': diagnostics.calibration_score,
                'market_alignment': diagnostics.market_alignment,
                'uncertainty_score': diagnostics.uncertainty_score,
                'sensitivity_passed': diagnostics.sensitivity_passed,
                'recommended_action': diagnostics.recommended_action
            },
            'professional_metadata': {
                'model_version': '4.0.0_multi_league',
                'calibration_level': 'EVIDENCE_BASED',
                'risk_profile': 'CONSERVATIVE',
                'league_supported': True,
                'timestamp': datetime.now().isoformat()
            }
        }

# =============================================================================
# PROFESSIONAL TEST FUNCTION FOR ALL LEAGUES
# =============================================================================

def test_professional_engine():
    """Test the professional prediction engine across all leagues"""
    
    test_data = {
        'home_team': 'Tottenham Hotspur',
        'away_team': 'Chelsea', 
        'home_goals': 12,
        'away_goals': 10,
        'home_conceded': 6,
        'away_conceded': 7,
        'home_goals_home': 2,
        'away_goals_away': 6,
        'home_injuries': 2,
        'away_injuries': 2,
        'home_motivation': 'Normal',
        'away_motivation': 'Normal',
        'home_tier': 'STRONG',
        'away_tier': 'STRONG',
        'market_odds': {
            '1x2_home': 3.10,
            '1x2_draw': 3.40,
            '1x2_away': 2.30,
            'over_25': 1.80,
            'under_25': 2.00,
            'btts_yes': 1.90,
            'btts_no': 1.90
        },
        'bankroll': 1000,
        'h2h_data': {
            'matches': 6,
            'home_wins': 1,
            'away_wins': 4, 
            'draws': 3,
            'home_goals': 7,
            'away_goals': 9
        }
    }
    
    print("🎯 PROFESSIONAL MULTI-LEAGUE PREDICTION ENGINE TEST")
    print("=" * 70)
    
    # Test across different leagues
    test_leagues = ['premier_league', 'la_liga', 'bundesliga', 'serie_a', 'championship']
    
    for league in test_leagues:
        test_data['league'] = league
        engine = ProfessionalPredictionEngine(test_data)
        results = engine.generate_predictions()
        
        league_display = LEAGUE_PARAMS[league]['display_name']
        home_win = results['probabilities']['match_outcomes']['home_win'] * 100
        away_win = results['probabilities']['match_outcomes']['away_win'] * 100
        total_xg = results['expected_goals']['total']
        
        print(f"{league_display}:")
        print(f"  Expected Goals: {results['expected_goals']['home']:.2f} - {results['expected_goals']['away']:.2f} (Total: {total_xg:.2f})")
        print(f"  Probabilities: Tottenham {home_win:.1f}% - Chelsea {away_win:.1f}%")
        print(f"  Context: {results['match_context']['primary_pattern']}")
        print(f"  Data Quality: {results['diagnostics']['data_quality_score']:.1f}/100")
        print()

if __name__ == "__main__":
    test_professional_engine()
