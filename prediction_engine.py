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
        'away_penalty': 0.80,
        'min_edge': 0.08,
        'volatility_multiplier': 1.0,
        'avg_goals': 1.4,
        'home_advantage': 1.20
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

class ProductionLeagueCalibrator:
    def __init__(self):
        self.league_calibration = LEAGUE_PARAMS
        
    def get_league_params(self, league: str) -> Dict[str, float]:
        return self.league_calibration.get(league, self.league_calibration['default'])
    
    def get_league_avg_goals(self, league: str) -> float:
        params = self.get_league_params(league)
        return params['avg_goals']
    
    def get_home_advantage(self, league: str) -> float:
        params = self.get_league_params(league)
        return params['home_advantage']
    
    def get_away_penalty(self, league: str) -> float:
        params = self.get_league_params(league)
        return params['away_penalty']
    
    def get_min_edge(self, league: str) -> float:
        params = self.get_league_params(league)
        return params['min_edge']

class ProductionFeatureEngine:
    def __init__(self):
        self.calibrator = ProductionLeagueCalibrator()
        
    def get_historical_context(self, team_tier: str) -> Tuple[float, float]:
        historical_context = {
            'ELITE': (1.30, 0.75),
            'STRONG': (1.12, 0.88),  
            'MEDIUM': (1.00, 1.00),
            'WEAK': (0.82, 1.20)
        }
        return historical_context.get(team_tier, (1.00, 1.00))
    
    def calculate_contextual_strength(self, goals: int, conceded: int, team_tier: str, 
                                    league_avg: float, games_played: int = 6) -> Tuple[float, float]:
        recent_attack = goals / (games_played * league_avg)
        recent_defense = (games_played * league_avg) / max(conceded, 0.5)
        
        historical_attack, historical_defense = self.get_historical_context(team_tier)
        
        # YOUR 60/40 REFINEMENT
        attack_strength = (0.6 * recent_attack) + (0.4 * historical_attack)
        defense_strength = (0.6 * recent_defense) + (0.4 * historical_defense)
        
        return attack_strength, defense_strength
    
    def calculate_contextual_xg(self, home_goals: int, home_conceded: int, home_tier: str,
                              away_goals: int, away_conceded: int, away_tier: str, 
                              league: str) -> Tuple[float, float, float, float]:
        league_avg = self.calibrator.get_league_avg_goals(league)
        home_advantage = self.calibrator.get_home_advantage(league)
        away_penalty = self.calibrator.get_away_penalty(league)
        
        home_attack, home_defense = self.calculate_contextual_strength(
            home_goals, home_conceded, home_tier, league_avg
        )
        away_attack, away_defense = self.calculate_contextual_strength(
            away_goals, away_conceded, away_tier, league_avg
        )
        
        home_xg = league_avg * home_attack * away_defense * home_advantage
        away_xg = league_avg * away_attack * home_defense * away_penalty
        
        home_xg = max(0.4, min(3.8, home_xg))
        away_xg = max(0.3, min(2.8, away_xg))
        
        home_uncertainty = home_xg * 0.08
        away_uncertainty = away_xg * 0.08
        
        return home_xg, away_xg, home_uncertainty, away_uncertainty

class BivariatePoissonSimulator:
    def __init__(self, n_simulations: int = 25000):
        self.n_simulations = n_simulations
        
    def simulate_match(self, home_xg: float, away_xg: float, correlation: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
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
        home_wins = np.mean(home_goals > away_goals)
        draws = np.mean(home_goals == away_goals)
        away_wins = np.mean(home_goals < away_goals)
        
        btts_yes = np.mean((home_goals > 0) & (away_goals > 0))
        
        total_goals = home_goals + away_goals
        over_25 = np.mean(total_goals > 2.5)
        
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
    def remove_vig_1x2(self, home_odds: float, draw_odds: float, away_odds: float) -> Dict[str, float]:
        home_implied = 1.0 / home_odds
        draw_implied = 1.0 / draw_odds
        away_implied = 1.0 / away_odds
        
        total_implied = home_implied + draw_implied + away_implied
        
        home_true = home_implied / total_implied
        draw_true = draw_implied / total_implied
        away_true = away_implied / total_implied
        
        return {'home': home_true, 'draw': draw_true, 'away': away_true}
    
    def remove_vig_two_way(self, yes_odds: float, no_odds: float) -> Dict[str, float]:
        yes_implied = 1.0 / yes_odds
        no_implied = 1.0 / no_odds
        
        total_implied = yes_implied + no_implied
        
        yes_true = yes_implied / total_implied
        no_true = no_implied / total_implied
        
        return {'yes': yes_true, 'no': no_true}
    
    def calculate_edges(self, model_probs: Dict[str, float], market_odds: Dict[str, float]) -> Dict[str, float]:
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
    def __init__(self):
        self.calibrator = ProductionLeagueCalibrator()
    
    def calculate_kelly_stake(self, model_prob: float, odds: float, bankroll: float, 
                            kelly_fraction: float = 0.2) -> float:
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

class EnhancedTeamTierCalibrator:
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
        league_teams = self.team_databases.get(league, {})
        return league_teams.get(team, 'MEDIUM')
    
    def get_all_teams_for_league(self, league: str) -> List[str]:
        return list(self.team_databases.get(league, {}).keys())

# MAIN ENGINE CLASS - FIXED NAME
class ApexProductionEngine:
    def __init__(self, match_data: Dict[str, Any]):
        self.data = self._production_data_validation(match_data)
        self.calibrator = ProductionLeagueCalibrator()
        self.feature_engine = ProductionFeatureEngine()
        self.simulator = BivariatePoissonSimulator()
        self.market_analyzer = MarketAnalyzer()
        self.staking_engine = ProductionStakingEngine()
        self.tier_calibrator = EnhancedTeamTierCalibrator()
        
    def _production_data_validation(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
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
        
        return validated_data

    def _calculate_contextual_xg(self) -> Tuple[float, float, float, float]:
        league = self.data.get('league', 'premier_league')
        
        home_tier = self.tier_calibrator.get_team_tier(self.data['home_team'], league)
        away_tier = self.tier_calibrator.get_team_tier(self.data['away_team'], league)
        
        home_xg, away_xg, home_uncertainty, away_uncertainty = self.feature_engine.calculate_contextual_xg(
            self.data.get('home_goals', 0),
            self.data.get('home_conceded', 0),
            home_tier,
            self.data.get('away_goals', 0),
            self.data.get('away_conceded', 0), 
            away_tier,
            league
        )
        
        return home_xg, away_xg, home_uncertainty, away_uncertainty

    def _run_production_simulation(self, home_xg: float, away_xg: float, 
                                 home_uncertainty: float, away_uncertainty: float) -> Dict[str, float]:
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
        home_xg, away_xg, home_uncertainty, away_uncertainty = self._calculate_contextual_xg()
        
        simulation_results = self._run_production_simulation(
            home_xg, away_xg, home_uncertainty, away_uncertainty
        )
        
        league = self.data.get('league', 'premier_league')
        home_tier = self.tier_calibrator.get_team_tier(self.data['home_team'], league)
        away_tier = self.tier_calibrator.get_team_tier(self.data['away_team'], league)
        
        context = self._determine_descriptive_context(home_xg, away_xg)
        
        market_edges = self.market_analyzer.calculate_edges(
            simulation_results, self.data['market_odds']
        )
        
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
            'confidence_score': max(simulation_results['home_win'], simulation_results['draw'], simulation_results['away_win']) * 100,
            'production_metrics': {
                'refined_contextual_model': True,
                '60_40_historical_weighting': True,
                'boosted_home_advantage': True,
                'stronger_away_penalty': True,
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
            'production_summary': f"REFINED 60/40 contextual model with 1.20x home advantage applied."
        }
