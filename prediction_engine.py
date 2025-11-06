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

# 🎯 PRODUCTION LEAGUE PARAMS (Your refined values)
LEAGUE_PARAMS = {
    'premier_league': {
        'away_penalty': 0.80,  # Your refinement: stronger away penalty
        'min_edge': 0.08,
        'volatility_multiplier': 1.0,
        'avg_goals': 1.4,
        'home_advantage': 1.20  # Your refinement: boosted home advantage
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
        'home_advantage': 1.21
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

# Context thresholds
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
    """PRODUCTION: Narrative for display only"""
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
    """PRODUCTION: League calibration with refined adjustments"""
    
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
        """Get refined home advantage multiplier"""
        params = self.get_league_params(league)
        return params['home_advantage']
    
    def get_away_penalty(self, league: str) -> float:
        """Get refined away penalty multiplier"""
        params = self.get_league_params(league)
        return params['away_penalty']
    
    def get_min_edge(self, league: str) -> float:
        """Get minimum edge threshold by league volatility"""
        params = self.get_league_params(league)
        return params['min_edge']

class ProductionFeatureEngine:
    """PRODUCTION: Feature engineering with REFINED contextual strength model"""
    
    def __init__(self):
        self.calibrator = ProductionLeagueCalibrator()
        
    def get_historical_context(self, team_tier: str) -> Tuple[float, float]:
        """YOUR REFINED SOLUTION: Historical context with stronger elite weighting"""
        historical_context = {
            'ELITE': (1.30, 0.75),      # STRONGER elite weighting
            'STRONG': (1.12, 0.88),     # Adjusted strong tier  
            'MEDIUM': (1.00, 1.00),     # Average
            'WEAK': (0.82, 1.20)        # Weaker teams
        }
        return historical_context.get(team_tier, (1.00, 1.00))
    
    def calculate_contextual_strength(self, goals: int, conceded: int, team_tier: str, 
                                    league_avg: float, games_played: int = 6) -> Tuple[float, float]:
        """YOUR EXACT REFINEMENT: 60% recent, 40% historical weighting"""
        
        # Recent form
        recent_attack = goals / (games_played * league_avg)
        recent_defense = (games_played * league_avg) / max(conceded, 0.5)
        
        # Historical context
        historical_attack, historical_defense = self.get_historical_context(team_tier)
        
        # YOUR EXACT REFINEMENT: 60% recent, 40% historical
        attack_strength = (0.6 * recent_attack) + (0.4 * historical_attack)
        defense_strength = (0.6 * recent_defense) + (0.4 * historical_defense)
        
        return attack_strength, defense_strength
    
    def calculate_contextual_xg(self, home_goals: int, home_conceded: int, home_tier: str,
                              away_goals: int, away_conceded: int, away_tier: str, 
                              league: str) -> Tuple[float, float, float, float]:
        """YOUR REFINED SOLUTION: Contextual xG with boosted home advantage"""
        
        league_avg = self.calibrator.get_league_avg_goals(league)
        home_advantage = self.calibrator.get_home_advantage(league)  # Your 1.20 refinement
        away_penalty = self.calibrator.get_away_penalty(league)      # Your 0.80 refinement
        
        # Calculate contextual strengths with refined 60/40 weighting
        home_attack, home_defense = self.calculate_contextual_strength(
            home_goals, home_conceded, home_tier, league_avg
        )
        away_attack, away_defense = self.calculate_contextual_strength(
            away_goals, away_conceded, away_tier, league_avg
        )
        
        # xG calculation with YOUR boosted home advantage
        home_xg = league_avg * home_attack * away_defense * home_advantage
        away_xg = league_avg * away_attack * home_defense * away_penalty
        
        # Apply realistic bounds
        home_xg = max(0.4, min(3.8, home_xg))
        away_xg = max(0.3, min(2.8, away_xg))
        
        # Reduced uncertainty with refined model
        home_uncertainty = home_xg * 0.08
        away_uncertainty = away_xg * 0.08
        
        return home_xg, away_xg, home_uncertainty, away_uncertainty
    
    def create_match_features(self, home_data: Dict, away_data: Dict, context: Dict, 
                            home_tier: str, away_tier: str, league: str) -> Dict[str, Any]:
        """PRODUCTION: Create features with REFINED contextual strength model"""
        
        # Calculate contextual xG with YOUR refined formula
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
            'historical_weighting': '60/40'  # Track the refined weighting
        }
        
        return features

class BivariatePoissonSimulator:
    """PRODUCTION: Bivariate Poisson simulation for goal correlation"""
    
    def __init__(self, n_simulations: int = 25000):
        self.n_simulations = n_simulations
        
    def simulate_match(self, home_xg: float, away_xg: float, correlation: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
        """Bivariate Poisson simulation with correlation"""
        home_goals = np.random.poisson(home_xg, self.n_simulations)
        away_goals = np.random.poisson(away_xg, self.n_simulations)
        
        if correlation > 0:
            correlated_count = np.random.binomial(
                np.minimum(home_goals, away_goals),
                correlation
            )
            home_goals = home_goals - correlated_count + np.random.poisson(correlation * np.minimum(home_xg, away_xg), self.n_simulations)
            away_goals = away_goals - correlated_count + np.random.poisson(correlation * np.minimum(home_xg, away_xg), self.n_simulations)
        
        return np.maximum(0, home_goals), np.maximum(0, away_goals)
    
    def get_market_probabilities(self, home_goals: np.ndarray, away_goals: np.ndarray) -> Dict[str, float]:
        """Calculate market probabilities from simulations"""
        home_wins = np.mean(home_goals > away_goals)
        draws = np.mean(home_goals == away_goals)
        away_wins = np.mean(home_goals < away_goals)
        
        btts_yes = np.mean((home_goals > 0) & (away_goals > 0))
        
        total_goals = home_goals + away_goals
        over_25 = np.mean(total_goals > 2.5)
        
        # Exact scores (top 8)
        score_counts = {}
        for h, a in zip(home_goals[:10000], away_goals[:10000]):
            score = f"{h}-{a}"
            score_counts[score] = score_counts.get(score, 0) + 1
        
        exact_scores = {
            score: count / 10000 
            for score, count in sorted(score_counts.items(), 
            key=lambda x: x[1], reverse=True)[:8]
        }
        
        return {
            'home_win': home_wins,
            'draw': draws,
            'away_win': away_wins,
            'btts_yes': btts_yes,
            'over_25': over_25,
            'under_25': 1 - over_25,
            'exact_scores': exact_scores
        }

class MarketAnalyzer:
    """PRODUCTION: Proper market analysis with vig removal"""
    
    def __init__(self):
        pass
    
    def remove_vig_1x2(self, home_odds: float, draw_odds: float, away_odds: float) -> Dict[str, float]:
        """Proper vig removal for 1X2 markets"""
        home_implied = 1.0 / home_odds
        draw_implied = 1.0 / draw_odds
        away_implied = 1.0 / away_odds
        
        total_implied = home_implied + draw_implied + away_implied
        
        home_true = home_implied / total_implied
        draw_true = draw_implied / total_implied
        away_true = away_implied / total_implied
        
        return {'home': home_true, 'draw': draw_true, 'away': away_true}
    
    def remove_vig_two_way(self, yes_odds: float, no_odds: float) -> Dict[str, float]:
        """Proper vig removal for two-way markets"""
        yes_implied = 1.0 / yes_odds
        no_implied = 1.0 / no_odds
        
        total_implied = yes_implied + no_implied
        
        yes_true = yes_implied / total_implied
        no_true = no_implied / total_implied
        
        return {'yes': yes_true, 'no': no_true}
    
    def calculate_edges(self, model_probs: Dict[str, float], market_odds: Dict[str, float]) -> Dict[str, float]:
        """Calculate proper edges with vig removal"""
        edges = {}
        
        if all(k in market_odds for k in ['1x2 Home', '1x2 Draw', '1x2 Away']):
            true_probs = self.remove_vig_1x2(
                market_odds['1x2 Home'],
                market_odds['1x2 Draw'], 
                market_odds['1x2 Away']
            )
            
            edges['home_win'] = model_probs['home_win'] - true_probs['home']
            edges['draw'] = model_probs['draw'] - true_probs['draw']
            edges['away_win'] = model_probs['away_win'] - true_probs['away']
        
        if all(k in market_odds for k in ['BTTS Yes', 'BTTS No']):
            true_probs = self.remove_vig_two_way(
                market_odds['BTTS Yes'],
                market_odds['BTTS No']
            )
            edges['btts_yes'] = model_probs['btts_yes'] - true_probs['yes']
            edges['btts_no'] = (1 - model_probs['btts_yes']) - true_probs['no']
        
        if all(k in market_odds for k in ['Over 2.5 Goals', 'Under 2.5 Goals']):
            true_probs = self.remove_vig_two_way(
                market_odds['Over 2.5 Goals'],
                market_odds['Under 2.5 Goals']
            )
            edges['over_25'] = model_probs['over_25'] - true_probs['yes']
            edges['under_25'] = model_probs['under_25'] - true_probs['no']
        
        return edges

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
        
        stake_multiplier = self.calibrator.get_stake_multiplier(league)
        adjusted_stake = base_stake * stake_multiplier
        
        final_stake = min(adjusted_stake, bankroll * 0.03)
        
        return {
            'base_stake': base_stake,
            'volatility_multiplier': stake_multiplier,
            'final_stake': final_stake,
            'bankroll_percentage': (final_stake / bankroll) * 100
        }

class ProductionPredictionExplainer:
    """PRODUCTION: Explanation engine"""
    
    def __init__(self):
        pass
    
    def generate_context_explanation(self, context: str, home_team: str, away_team: str, 
                                  home_xg: float, away_xg: float, home_tier: str, away_tier: str) -> List[str]:
        """Generate descriptive context explanations"""
        explanations = {
            'home_dominance': [
                f"🏠 **Home Dominance**: {home_team} ({home_tier}) shows strong home advantage with {home_xg:.1f} expected goals",
                f"Superior attacking threat expected against {away_team}'s defense",
                f"Focus: Home win markets, potential clean sheet"
            ],
            'away_counter': [
                f"✈️ **Away Counter**: {away_team} ({away_tier}) quality may overcome home disadvantage",
                f"Strong away performance expected with {away_xg:.1f} xG against {home_team}",
                f"Focus: Away win, double chance away/draw"
            ],
            'offensive_showdown': [
                f"🔥 **Offensive Showdown**: High-scoring game expected ({home_xg + away_xg:.1f} total xG)",
                f"Both teams have strong attacking capabilities - goals likely at both ends",
                f"Focus: Over 2.5 goals, BTTS yes, goals markets"
            ],
            'defensive_battle': [
                f"🛡️ **Defensive Battle**: Organized defenses from both sides ({home_xg + away_xg:.1f} total xG)",
                f"Lower scoring game expected with limited clear chances",
                f"Focus: Under 2.5 goals, BTTS no, low correct scores"
            ],
            'tactical_stalemate': [
                f"⚔️ **Tactical Stalemate**: Evenly matched teams ({abs(home_xg - away_xg):.1f} xG difference)",
                f"Game likely to be decided by fine margins or set pieces", 
                f"Focus: Draw, under 2.5, 1-1/0-0 correct score"
            ],
            'balanced': [
                f"⚖️ **Balanced Match**: No clear dominance pattern detected",
                f"Both teams show competitive expected goals ({home_xg:.1f} vs {away_xg:.1f})",
                f"Focus: Value bets across all markets"
            ]
        }
        
        return explanations.get(context, ["Match analysis in progress..."])
    
    def generate_risk_assessment(self, certainty: float) -> Dict[str, Any]:
        """Generate risk assessment"""
        if certainty > 0.45:
            risk_level = "LOW"
            explanation = "High model certainty with strong context alignment"
        elif certainty > 0.35:
            risk_level = "MEDIUM" 
            explanation = "Reasonable certainty with good context support"
        else:
            risk_level = "HIGH"
            explanation = "Lower certainty or weak context alignment"
        
        return {
            'risk_level': risk_level,
            'explanation': explanation,
            'recommendation': f"{risk_level} risk - adjust stakes accordingly",
            'certainty': f"{certainty * 100:.1f}%"
        }

# Team database for contextual strength
class EnhancedTeamTierCalibrator:
    """Team database for refined contextual strength model"""
    
    def __init__(self):
        self.team_databases = {
            'premier_league': {
                'Arsenal': 'ELITE', 'Manchester City': 'ELITE', 'Liverpool': 'ELITE',
                'Tottenham Hotspur': 'STRONG', 'Chelsea': 'STRONG', 'Manchester United': 'STRONG',
                'Newcastle United': 'STRONG', 'Aston Villa': 'STRONG', 'Brighton & Hove Albion': 'MEDIUM',
                'West Ham United': 'MEDIUM', 'Crystal Palace': 'MEDIUM', 'Wolverhampton': 'MEDIUM',
                'Fulham': 'MEDIUM', 'Brentford': 'MEDIUM', 'Everton': 'MEDIUM',
                'Nottingham Forest': 'MEDIUM', 'Luton Town': 'WEAK', 'Burnley': 'WEAK', 'Sheffield United': 'WEAK'
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
                'Cardiff City': 'MEDIUM', 'Sunderland': 'MEDIUM'
            }
        }
    
    def get_team_tier(self, team: str, league: str) -> str:
        """Get team tier for refined contextual strength model"""
        league_teams = self.team_databases.get(league, {})
        return league_teams.get(team, 'MEDIUM')
    
    def get_all_teams_for_league(self, league: str) -> List[str]:
        """Get all teams for a given league"""
        return list(self.team_databases.get(league, {}).keys())

class ApexProductionEngine:
    """PRODUCTION-READY PREDICTION ENGINE WITH REFINED CONTEXTUAL STRENGTH MODEL"""
    
    def __init__(self, match_data: Dict[str, Any]):
        self.data = self._production_data_validation(match_data)
        self.calibrator = ProductionLeagueCalibrator()
        self.feature_engine = ProductionFeatureEngine()
        self.simulator = BivariatePoissonSimulator()
        self.market_analyzer = MarketAnalyzer()
        self.staking_engine = ProductionStakingEngine()
        self.explainer = ProductionPredictionExplainer()
        self.tier_calibrator = EnhancedTeamTierCalibrator()
        
    def _production_data_validation(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """PRODUCTION: Robust data validation"""
        validated_data = match_data.copy()
        
        required_fields = ['home_team', 'away_team', 'league']
        for field in required_fields:
            if field not in validated_data:
                validated_data[field] = 'Unknown'
        
        predictive_fields = {
            'home_goals': (0, 30, 8), 'away_goals': (0, 30, 4),
            'home_conceded': (0, 30, 6), 'away_conceded': (0, 30, 7),
        }
        
        for field, (min_val, max_val, default) in predictive_fields.items():
            if field in validated_data:
                try:
                    value = float(validated_data[field])
                    validated_data[field] = max(min_val, min(value, max_val))
                except (TypeError, ValueError):
                    validated_data[field] = default
            else:
                validated_data[field] = default
        
        if 'market_odds' not in validated_data:
            validated_data['market_odds'] = {
                '1x2 Home': 2.50, '1x2 Draw': 2.95, '1x2 Away': 2.85,
                'Over 2.5 Goals': 2.63, 'Under 2.5 Goals': 1.50,
                'BTTS Yes': 2.10, 'BTTS No': 1.67
            }
        
        validated_data['bankroll'] = validated_data.get('bankroll', 1000)
        validated_data['kelly_fraction'] = validated_data.get('kelly_fraction', 0.2)
        
        logger.info(f"Production data validation complete for {validated_data['home_team']} vs {validated_data['away_team']}")
        return validated_data

    def _calculate_contextual_xg(self) -> Tuple[float, float, float, float]:
        """YOUR REFINED SOLUTION: Calculate xG with refined contextual strength model"""
        league = self.data.get('league', 'premier_league')
        
        # Get team tiers for contextual strength
        home_tier = self.tier_calibrator.get_team_tier(self.data['home_team'], league)
        away_tier = self.tier_calibrator.get_team_tier(self.data['away_team'], league)
        
        # Create feature context
        feature_context = {
            'home_goals': self.data.get('home_goals', 0),
            'home_conceded': self.data.get('home_conceded', 0),
            'away_goals': self.data.get('away_goals', 0),
            'away_conceded': self.data.get('away_conceded', 0),
        }
        
        home_team_data = {'name': self.data['home_team']}
        away_team_data = {'name': self.data['away_team']}
        
        features = self.feature_engine.create_match_features(
            home_team_data, away_team_data, feature_context, home_tier, away_tier, league
        )
        
        return (features['home_xg'], features['away_xg'], 
                features['home_xg_uncertainty'], features['away_xg_uncertainty'])

    def _run_production_simulation(self, home_xg: float, away_xg: float, 
                                 home_uncertainty: float, away_uncertainty: float) -> Dict[str, float]:
        """PRODUCTION: Run simulation with uncertainty propagation"""
        home_xg_samples = np.random.normal(home_xg, home_uncertainty, 5)
        away_xg_samples = np.random.normal(away_xg, away_uncertainty, 5)
        
        all_results = []
        
        for h_xg, a_xg in zip(home_xg_samples, away_xg_samples):
            home_goals, away_goals = self.simulator.simulate_match(
                max(0.1, h_xg), max(0.1, a_xg)
            )
            
            results = self.simulator.get_market_probabilities(home_goals, away_goals)
            all_results.append(results)
        
        final_results = {}
        for key in all_results[0].keys():
            if key == 'exact_scores':
                score_aggregate = {}
                for result in all_results:
                    for score, prob in result[key].items():
                        score_aggregate[score] = score_aggregate.get(score, 0) + prob
                
                total = sum(score_aggregate.values())
                final_results[key] = {
                    score: prob/total 
                    for score, prob in sorted(score_aggregate.items(), 
                    key=lambda x: x[1], reverse=True)[:8]
                }
            else:
                final_results[key] = np.mean([r[key] for r in all_results])
        
        return final_results

    def _determine_descriptive_context(self, home_xg: float, away_xg: float) -> str:
        """PRODUCTION: Determine context for display only"""
        total_xg = home_xg + away_xg
        xg_diff = home_xg - away_xg
        
        if total_xg > CONTEXT_THRESHOLDS['total_xg_offensive']:
            return "offensive_showdown"
        elif total_xg < CONTEXT_THRESHOLDS['total_xg_defensive']:
            return "defensive_battle"
        elif xg_diff > CONTEXT_THRESHOLDS['xg_diff_dominant']:
            return "home_dominance"
        elif xg_diff < -CONTEXT_THRESHOLDS['xg_diff_dominant']:
            return "away_counter"
        elif abs(xg_diff) < 0.2:
            return "tactical_stalemate"
        else:
            return "balanced"

    def generate_production_predictions(self) -> Dict[str, Any]:
        """PRODUCTION: Generate professional predictions with YOUR refined model"""
        logger.info(f"Starting refined production prediction for {self.data['home_team']} vs {self.data['away_team']}")
        
        # Calculate xG with YOUR refined contextual strength model
        home_xg, away_xg, home_uncertainty, away_uncertainty = self._calculate_contextual_xg()
        
        # Run production simulation
        simulation_results = self._run_production_simulation(
            home_xg, away_xg, home_uncertainty, away_uncertainty
        )
        
        # Get team tiers for display
        league = self.data.get('league', 'premier_league')
        home_tier = self.tier_calibrator.get_team_tier(self.data['home_team'], league)
        away_tier = self.tier_calibrator.get_team_tier(self.data['away_team'], league)
        
        # Determine descriptive context
        context = self._determine_descriptive_context(home_xg, away_xg)
        
        # Calculate market edges
        market_edges = self.market_analyzer.calculate_edges(
            simulation_results, self.data['market_odds']
        )
        
        # Generate explanations
        explanations = self.explainer.generate_context_explanation(
            context, self.data['home_team'], self.data['away_team'], 
            home_xg, away_xg, home_tier, away_tier
        )
        
        # Risk assessment
        certainty = max(simulation_results['home_win'], simulation_results['draw'], simulation_results['away_win'])
        risk_assessment = self.explainer.generate_risk_assessment(certainty)
        
        # Calculate stakes for value opportunities
        betting_opportunities = []
        bankroll = self.data['bankroll']
        kelly_fraction = self.data['kelly_fraction']
        
        for market, edge in market_edges.items():
            if edge > self.calibrator.get_min_edge(league):
                if 'home_win' in market:
                    odds = self.data['market_odds']['1x2 Home']
                elif 'away_win' in market:
                    odds = self.data['market_odds']['1x2 Away']
                elif 'btts_yes' in market:
                    odds = self.data['market_odds']['BTTS Yes']
                elif 'over_25' in market:
                    odds = self.data['market_odds']['Over 2.5 Goals']
                else:
                    continue
                
                stake_info = self.staking_engine.calculate_professional_stake(
                    simulation_results[market.replace('_edge', '')], 
                    odds, bankroll, league, kelly_fraction
                )
                
                betting_opportunities.append({
                    'market': market,
                    'edge': edge,
                    'model_prob': simulation_results[market.replace('_edge', '')],
                    'odds': odds,
                    'stake': stake_info['final_stake'],
                    'bankroll_percentage': stake_info['bankroll_percentage']
                })
        
        # Final production output
        return {
            'match': f"{self.data['home_team']} vs {self.data['away_team']}",
            'league': league,
            'expected_goals': {
                'home': home_xg,
                'away': away_xg,
                'home_uncertainty': home_uncertainty,
                'away_uncertainty': away_uncertainty,
                'total': home_xg + away_xg
            },
            'team_tiers': {'home': home_tier, 'away': away_tier},
            'match_context': context,
            'confidence_score': certainty * 100,
            'production_metrics': {
                'refined_contextual_model': True,
                '60_40_historical_weighting': True,
                'boosted_home_advantage': True,
                'stronger_away_penalty': True,
                'vig_properly_removed': True
            },
            'probabilities': {
                'match_outcomes': {
                    'home_win': simulation_results['home_win'] * 100,
                    'draw': simulation_results['draw'] * 100,
                    'away_win': simulation_results['away_win'] * 100
                },
                'both_teams_score': {
                    'yes': simulation_results['btts_yes'] * 100,
                    'no': (1 - simulation_results['btts_yes']) * 100
                },
                'over_under': {
                    'over_25': simulation_results['over_25'] * 100,
                    'under_25': simulation_results['under_25'] * 100
                },
                'exact_scores': simulation_results['exact_scores']
            },
            'market_analysis': {
                'edges': market_edges,
                'min_edge_threshold': self.calibrator.get_min_edge(league) * 100,
                'value_opportunities': len(betting_opportunities)
            },
            'betting_recommendations': betting_opportunities,
            'explanations': explanations,
            'risk_assessment': risk_assessment,
            'production_summary': f"REFINED contextual strength analysis complete. Using 60/40 recent/historical weighting with boosted home advantage (1.20x)."
        }

def test_refined_model():
    """Test the refined contextual strength model with Liverpool vs Villa"""
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
    print(f"Context: {results['match_context']}")
    print(f"Production Features: {list(results['production_metrics'].keys())}")

if __name__ == "__main__":
    test_refined_model()
