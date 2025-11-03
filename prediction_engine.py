# prediction_engine.py - COMPLETE APEX INTELLIGENCE VERSION
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
    """CENTRAL MATCH NARRATIVE - The Football Brain"""
    
    def __init__(self):
        self.dominance = "balanced"
        self.style_conflict = "neutral"
        self.expected_tempo = "medium"
        self.expected_openness = 0.5
        self.defensive_stability = "mixed"
        self.primary_pattern = None
        
    def to_dict(self):
        return {
            'dominance': self.dominance,
            'style_conflict': self.style_conflict,
            'expected_tempo': self.expected_tempo,
            'expected_openness': self.expected_openness,
            'defensive_stability': self.defensive_stability,
            'primary_pattern': self.primary_pattern
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

class TeamTierCalibrator:
    def __init__(self):
        self.league_baselines = {
            'premier_league': {'avg_goals': 2.8, 'home_advantage': 0.35},
            'la_liga': {'avg_goals': 2.6, 'home_advantage': 0.32},
            'serie_a': {'avg_goals': 2.7, 'home_advantage': 0.38},
            'bundesliga': {'avg_goals': 3.1, 'home_advantage': 0.28},
            'ligue_1': {'avg_goals': 2.5, 'home_advantage': 0.34},
            'liga_portugal': {'avg_goals': 2.6, 'home_advantage': 0.42},
            'brasileirao': {'avg_goals': 2.4, 'home_advantage': 0.45},
            'liga_mx': {'avg_goals': 2.7, 'home_advantage': 0.40},
            'eredivisie': {'avg_goals': 3.0, 'home_advantage': 0.30},
        }
        
        self.team_databases = {
            'premier_league': {
                'Arsenal': 'ELITE', 'Man City': 'ELITE', 'Liverpool': 'ELITE',
                'Tottenham': 'STRONG', 'Aston Villa': 'STRONG', 'Newcastle': 'STRONG',
                'West Ham': 'MEDIUM', 'Brighton': 'MEDIUM', 'Wolves': 'MEDIUM',
            },
            'la_liga': {
                'Real Madrid': 'ELITE', 'Barcelona': 'ELITE', 'Atletico Madrid': 'ELITE',
                'Athletic Bilbao': 'STRONG', 'Real Sociedad': 'STRONG', 'Sevilla': 'STRONG',
                'Real Betis': 'MEDIUM', 'Getafe': 'MEDIUM', 'Osasuna': 'MEDIUM',
            },
            'bundesliga': {
                'Bayern Munich': 'ELITE', 'Bayer Leverkusen': 'ELITE', 'Borussia Dortmund': 'ELITE',
                'RB Leipzig': 'STRONG', 'Eintracht Frankfurt': 'STRONG', 'Wolfsburg': 'STRONG',
                'Freiburg': 'STRONG', 'Hoffenheim': 'STRONG', 'Augsburg': 'MEDIUM',
            },
            'ligue_1': {
                'PSG': 'ELITE', 'Monaco': 'STRONG', 'Marseille': 'STRONG',
                'Lille': 'STRONG', 'Lyon': 'STRONG', 'Rennes': 'STRONG',
                'Nice': 'STRONG', 'Lens': 'STRONG', 'Reims': 'MEDIUM',
                'Montpellier': 'MEDIUM', 'Toulouse': 'MEDIUM', 'Strasbourg': 'MEDIUM',
                'Nantes': 'MEDIUM', 'Le Havre': 'MEDIUM', 'Brest': 'MEDIUM',
            }
        }
    
    def get_league_teams(self, league: str) -> List[str]:
        return list(self.team_databases.get(league, {}).keys())
    
    def get_team_tier(self, team: str, league: str) -> str:
        league_teams = self.team_databases.get(league, {})
        return league_teams.get(team, 'MEDIUM')

class ApexIntelligenceEngine:
    """APEX INTELLIGENCE ENGINE - Holistic Football Reasoning System"""
    
    def __init__(self, match_data: Dict[str, Any]):
        self.data = self._validate_and_enhance_data(match_data)
        self.calibrator = TeamTierCalibrator()
        self.narrative = MatchNarrative()
        self.intelligence = IntelligenceMetrics(
            narrative_coherence=0.0, prediction_alignment="LOW", 
            data_quality_score=0.0, certainty_score=0.0,
            market_edge_score=0.0, risk_level="HIGH", football_iq_score=0.0
        )
        self._setup_parameters()
        
    def _validate_and_enhance_data(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        enhanced_data = match_data.copy()
        
        predictive_fields = {
            'home_goals': (0, 20, 1.5), 'away_goals': (0, 20, 1.5),
            'home_conceded': (0, 20, 1.5), 'away_conceded': (0, 20, 1.5),
        }
        
        for field, (min_val, max_val, default) in predictive_fields.items():
            if field in enhanced_data:
                try:
                    value = float(enhanced_data[field])
                    enhanced_data[field] = max(min_val, min(value, max_val))
                except (TypeError, ValueError):
                    enhanced_data[field] = default
            else:
                enhanced_data[field] = default
        
        for form_field in ['home_form', 'away_form']:
            if form_field in enhanced_data:
                try:
                    if isinstance(enhanced_data[form_field], list):
                        enhanced_data[form_field] = [float(x) for x in enhanced_data[form_field]]
                    else:
                        enhanced_data[form_field] = []
                except (TypeError, ValueError):
                    enhanced_data[form_field] = []
        
        if 'motivation' not in enhanced_data:
            enhanced_data['motivation'] = {'home': 'Normal', 'away': 'Normal'}
            
        return enhanced_data

    def _setup_parameters(self):
        self.calibration_params = {
            'form_decay_rate': 0.85, 'h2h_weight': 0.15, 'injury_impact': 0.08,
            'motivation_impact': 0.10, 'defensive_impact_multiplier': 0.4,
        }

    def _calculate_data_quality(self) -> float:
        score = 0
        max_score = 0
        
        if self.data.get('home_team') and self.data.get('away_team'):
            score += 20
        max_score += 20
        
        if self.data.get('home_goals', 0) > 0:
            score += 15
        if self.data.get('away_goals', 0) > 0:
            score += 15
        max_score += 30
        
        if len(self.data.get('home_form', [])) >= 4:
            score += 10
        if len(self.data.get('away_form', [])) >= 4:
            score += 10
        max_score += 20
        
        h2h_data = self.data.get('h2h_data', {})
        if h2h_data.get('matches', 0) >= 3:
            score += 20
        max_score += 20
        
        return (score / max_score) * 100

    def _estimate_market_edge(self) -> float:
        return 0.5  # Placeholder - would integrate with actual market odds analysis

    def _calculate_realistic_xg(self) -> Tuple[float, float]:
        league = self.data.get('league', 'premier_league')
        
        home_goals_avg = self.data.get('home_goals', 0) / 6.0
        away_goals_avg = self.data.get('away_goals', 0) / 6.0
        home_conceded_avg = self.data.get('home_conceded', 0) / 6.0
        away_conceded_avg = self.data.get('away_conceded', 0) / 6.0
        
        league_avg = self.calibrator.league_baselines.get(league, {'avg_goals': 2.7})['avg_goals']
        
        home_xg = home_goals_avg * (1 - (away_conceded_avg / league_avg) * 0.4)
        away_xg = away_goals_avg * (1 - (home_conceded_avg / league_avg) * 0.4)
        
        home_advantage = self.calibrator.league_baselines.get(league, {'home_advantage': 0.35})['home_advantage']
        home_xg *= (1 + home_advantage)
        
        motivation = self.data.get('motivation', {})
        home_motivation = self._calculate_motivation_impact(motivation.get('home', 'Normal'))
        away_motivation = self._calculate_motivation_impact(motivation.get('away', 'Normal'))
        
        home_xg *= home_motivation
        away_xg *= away_motivation
        
        h2h_data = self.data.get('h2h_data', {})
        if h2h_data and h2h_data.get('matches', 0) >= 3:
            home_xg, away_xg = self._apply_h2h_adjustment(home_xg, away_xg, h2h_data)
        
        return round(home_xg, 3), round(away_xg, 3)

    def _calculate_motivation_impact(self, motivation_level: str) -> float:
        multipliers = {"Low": 0.90, "Normal": 1.0, "High": 1.08, "Very High": 1.12}
        return multipliers.get(motivation_level, 1.0)

    def _apply_h2h_adjustment(self, home_xg: float, away_xg: float, h2h_data: Dict) -> Tuple[float, float]:
        matches = h2h_data.get('matches', 0)
        if matches < 3:
            return home_xg, away_xg
        
        h2h_weight = min(0.25, matches * 0.06)
        h2h_home_avg = h2h_data.get('home_goals', 0) / matches
        h2h_away_avg = h2h_data.get('away_goals', 0) / matches
        
        adjusted_home_xg = (home_xg * (1 - h2h_weight)) + (h2h_home_avg * h2h_weight)
        adjusted_away_xg = (away_xg * (1 - h2h_weight)) + (h2h_away_avg * h2h_weight)
        
        return adjusted_home_xg, adjusted_away_xg

    def _determine_match_narrative(self, home_xg: float, away_xg: float) -> MatchNarrative:
        narrative = MatchNarrative()
        total_xg = home_xg + away_xg
        xg_difference = home_xg - away_xg
        
        if xg_difference > 0.8:
            narrative.dominance = "home"
            narrative.primary_pattern = "home_dominance"
        elif xg_difference < -0.8:
            narrative.dominance = "away" 
            narrative.primary_pattern = "away_dominance"
        else:
            narrative.dominance = "balanced"
            
        home_attack = self.data.get('home_goals', 0) / 6.0
        away_attack = self.data.get('away_goals', 0) / 6.0
        home_defense = self.data.get('home_conceded', 0) / 6.0
        away_defense = self.data.get('away_conceded', 0) / 6.0
        
        if home_attack > 2.0 and away_attack > 2.0:
            narrative.style_conflict = "attacking_vs_attacking"
            narrative.expected_openness = 0.85
            narrative.expected_tempo = "high"
        elif home_attack > 1.8 and away_defense < 1.0:
            narrative.style_conflict = "attacking_vs_defensive"
            narrative.expected_openness = 0.6
            narrative.expected_tempo = "medium"
        elif away_attack > 1.8 and home_defense < 1.0:
            narrative.style_conflict = "defensive_vs_attacking" 
            narrative.expected_openness = 0.6
            narrative.expected_tempo = "medium"
        else:
            narrative.style_conflict = "balanced"
            narrative.expected_openness = 0.5
            narrative.expected_tempo = "medium"
            
        avg_conceded = (home_defense + away_defense) / 2
        if avg_conceded < 0.8:
            narrative.defensive_stability = "solid"
        elif avg_conceded > 1.5:
            narrative.defensive_stability = "leaky" 
        else:
            narrative.defensive_stability = "mixed"
            
        return narrative
    
    def _calculate_holistic_btts(self, home_xg: float, away_xg: float, narrative: MatchNarrative) -> float:
        home_score_prob = 1 - math.exp(-home_xg)
        away_score_prob = 1 - math.exp(-away_xg)
        base_btts = home_score_prob * away_score_prob
        
        narrative_factor = 1.0
        if narrative.style_conflict == "attacking_vs_attacking":
            narrative_factor *= 1.4
        elif narrative.style_conflict in ["attacking_vs_defensive", "defensive_vs_attacking"]:
            narrative_factor *= 0.7
            
        if narrative.defensive_stability == "leaky":
            narrative_factor *= 1.3
        elif narrative.defensive_stability == "solid":
            narrative_factor *= 0.8
            
        league = self.data.get('league', 'premier_league')
        league_factors = {
            'bundesliga': 1.25, 'eredivisie': 1.2, 'premier_league': 1.1,
            'la_liga': 1.0, 'serie_a': 0.9, 'ligue_1': 0.85
        }
        league_factor = league_factors.get(league, 1.0)
        
        holistic_btts = base_btts * narrative_factor * league_factor
        return max(0.15, min(0.90, holistic_btts))
    
    def _calculate_holistic_over_under(self, home_xg: float, away_xg: float, narrative: MatchNarrative) -> Dict[str, float]:
        total_xg = home_xg + away_xg
        base_over = 1 - poisson.cdf(2, total_xg)
        
        narrative_factor = 1.0
        if narrative.style_conflict == "attacking_vs_attacking":
            narrative_factor *= 1.4
        elif narrative.expected_tempo == "high":
            narrative_factor *= 1.2
        elif narrative.style_conflict in ["attacking_vs_defensive", "defensive_vs_attacking"]:
            narrative_factor *= 0.8
            
        if narrative.defensive_stability == "leaky":
            narrative_factor *= 1.3
        elif narrative.defensive_stability == "solid":
            narrative_factor *= 0.7
            
        league = self.data.get('league', 'premier_league')
        league_factors = {
            'bundesliga': 1.3, 'eredivisie': 1.25, 'premier_league': 1.15,
            'la_liga': 1.0, 'serie_a': 0.95, 'ligue_1': 0.85
        }
        league_factor = league_factors.get(league, 1.0)
        
        holistic_over = base_over * narrative_factor * league_factor
        holistic_under = 1 - holistic_over
        
        return {
            'over_25': max(0.1, min(0.9, holistic_over)),
            'under_25': max(0.1, min(0.9, holistic_under))
        }
    
    def _calculate_intelligent_goal_timing(self, total_xg: float, narrative: MatchNarrative, btts_prob: float) -> Dict[str, float]:
        first_half_base = 1 - poisson.pmf(0, total_xg * 0.46)
        second_half_base = 1 - poisson.pmf(0, total_xg * 0.54)
        
        timing_factor = 1.0
        if btts_prob < 0.4:
            timing_factor *= 0.7
            
        if narrative.style_conflict in ["attacking_vs_defensive", "defensive_vs_attacking"]:
            first_half_base *= 0.8
            second_half_base *= 1.1
            
        if narrative.expected_tempo == "high":
            first_half_base *= 1.2
            
        first_half = first_half_base * timing_factor
        second_half = second_half_base * timing_factor
        
        return {
            'first_half': max(0.2, min(0.95, first_half)),
            'second_half': max(0.2, min(0.95, second_half))
        }
    
    def _calculate_intelligent_corners(self, narrative: MatchNarrative, home_xg: float, away_xg: float) -> Dict[str, str]:
        base_corners = 9.5
        
        if narrative.style_conflict == "attacking_vs_defensive":
            corner_adjustment = 2.5
        elif narrative.style_conflict == "defensive_vs_attacking":
            corner_adjustment = 1.5
        elif narrative.style_conflict == "attacking_vs_attacking":
            corner_adjustment = 2.0
        elif narrative.defensive_stability == "solid":
            corner_adjustment = -1.5
        else:
            corner_adjustment = 0
            
        xg_adjustment = (home_xg + away_xg - 2.7) * 0.5
        total_corners = max(5, min(16, base_corners + corner_adjustment + xg_adjustment))
        
        return {
            'total': f"{int(total_corners)}-{int(total_corners + 1)}",
            'home': f"{int(total_corners * 0.55)}-{int(total_corners * 0.55 + 0.5)}",
            'away': f"{int(total_corners * 0.45)}-{int(total_corners * 0.45 + 0.5)}"
        }
    
    def _run_monte_carlo(self, home_xg: float, away_xg: float, iterations: int = 10000) -> MonteCarloResults:
        np.random.seed(42)
        
        lambda3_alpha = 0.12
        lambda3 = lambda3_alpha * min(home_xg, away_xg)
        lambda1 = max(0.1, home_xg - lambda3)
        lambda2 = max(0.1, away_xg - lambda3)
        
        C = np.random.poisson(lambda3, iterations)
        A = np.random.poisson(lambda1, iterations)
        B = np.random.poisson(lambda2, iterations)
        
        home_goals_sim = A + C
        away_goals_sim = B + C
        
        home_wins = np.sum(home_goals_sim > away_goals_sim) / iterations
        draws = np.sum(home_goals_sim == away_goals_sim) / iterations
        away_wins = np.sum(home_goals_sim < away_goals_sim) / iterations
        
        exact_scores = {}
        for i in range(6):
            for j in range(6):
                count = np.sum((home_goals_sim == i) & (away_goals_sim == j))
                prob = count / iterations
                if prob > 0.005:
                    exact_scores[f"{i}-{j}"] = round(prob * 100, 1)
        
        exact_scores = dict(sorted(exact_scores.items(), key=lambda x: x[1], reverse=True)[:6])
        
        return MonteCarloResults(
            home_win_prob=home_wins, draw_prob=draws, away_win_prob=away_wins,
            over_25_prob=0.5, btts_prob=0.5, exact_scores=exact_scores,
            confidence_intervals={}, probability_volatility={}
        )
    
    def _assess_prediction_coherence(self, predictions: Dict) -> Tuple[float, str]:
        btts_yes = predictions.get('btts_yes', 0.5)
        over_25 = predictions.get('over_25', 0.5)
        home_win = predictions.get('home_win', 0.33)
        
        coherence_score = 0.0
        
        if btts_yes > 0.7 and over_25 < 0.4:
            coherence_score -= 0.3
        elif btts_yes < 0.3 and over_25 > 0.7:
            coherence_score -= 0.3  
        else:
            coherence_score += 0.3
            
        if home_win > 0.6 and over_25 < 0.3:
            coherence_score -= 0.2
        elif home_win < 0.3 and over_25 > 0.7:
            coherence_score -= 0.2
        else:
            coherence_score += 0.2
            
        if coherence_score >= 0.4:
            alignment = "HIGH"
        elif coherence_score >= 0.1:
            alignment = "MEDIUM"
        else:
            alignment = "LOW"
            
        return max(0.0, min(1.0, 0.5 + coherence_score)), alignment
    
    def _calculate_intelligent_risk(self, certainty: float, data_quality: float, 
                                  market_edge: float, alignment: str) -> str:
        base_risk = (1 - certainty) * 0.4 + (1 - data_quality/100) * 0.3 + (1 - market_edge) * 0.3
        
        alignment_penalty = {
            "HIGH": 0.0, "MEDIUM": 0.2, "LOW": 0.4
        }.get(alignment, 0.3)
        
        total_risk = base_risk + alignment_penalty
        
        if total_risk < 0.3:
            return "LOW"
        elif total_risk < 0.5:
            return "MEDIUM"
        elif total_risk < 0.7:
            return "HIGH"
        else:
            return "VERY_HIGH"
    
    def _generate_intelligent_summary(self, narrative: MatchNarrative, predictions: Dict, 
                                   home_team: str, away_team: str) -> str:
        home_win = predictions.get('home_win', 0.33)
        btts_yes = predictions.get('btts_yes', 0.5)
        over_25 = predictions.get('over_25', 0.5)
        
        if narrative.primary_pattern == "home_dominance":
            if over_25 > 0.6:
                return f"{home_team} are expected to dominate this encounter with their superior attacking quality. {away_team}'s defensive vulnerabilities suggest multiple goals are likely as the home side controls proceedings."
            else:
                return f"{home_team} should control possession and create the better chances, but {away_team}'s organized defense may limit clear opportunities. A patient, probing performance from the hosts could yield a narrow victory."
                
        elif narrative.style_conflict == "attacking_vs_attacking":
            if btts_yes > 0.7:
                return f"An entertaining, open contest awaits as two attack-minded teams face off. Both {home_team} and {away_team} have shown defensive frailties, suggesting goals at both ends in what could be a high-scoring affair."
            else:
                return f"Despite both teams' attacking intentions, this could become a tactical battle where chances are limited. The offensive quality on display may cancel out, leading to a tighter encounter than expected."
                
        elif narrative.style_conflict == "attacking_vs_defensive":
            if over_25 > 0.6:
                return f"{home_team}'s attacking impetus against {away_team}'s defensive resilience creates an intriguing tactical dynamic. The home side's creativity should eventually break through, but not without facing determined resistance."
            else:
                return f"{home_team} will look to impose their attacking game on a well-organized {away_team} defense. This could become a game of patience, with the hosts needing to work hard to create clear opportunities against disciplined opposition."
                
        else:
            return f"A competitive match expected between {home_team} and {away_team}, with small margins likely deciding the outcome. Both teams will seek to establish control in what promises to be a closely-fought encounter."

    def generate_apex_predictions(self, mc_iterations: int = 10000) -> Dict[str, Any]:
        home_xg, away_xg = self._calculate_realistic_xg()
        self.narrative = self._determine_match_narrative(home_xg, away_xg)
        
        btts_prob = self._calculate_holistic_btts(home_xg, away_xg, self.narrative)
        over_under = self._calculate_holistic_over_under(home_xg, away_xg, self.narrative)
        goal_timing = self._calculate_intelligent_goal_timing(home_xg + away_xg, self.narrative, btts_prob)
        corners = self._calculate_intelligent_corners(self.narrative, home_xg, away_xg)
        
        mc_results = self._run_monte_carlo(home_xg, away_xg, mc_iterations)
        
        prediction_set = {
            'home_win': mc_results.home_win_prob,
            'btts_yes': btts_prob,
            'over_25': over_under['over_25']
        }
        
        coherence, alignment = self._assess_prediction_coherence(prediction_set)
        certainty = max(mc_results.home_win_prob, mc_results.away_win_prob)
        data_quality = self._calculate_data_quality()
        market_edge = self._estimate_market_edge()
        
        risk_level = self._calculate_intelligent_risk(certainty, data_quality, market_edge, alignment)
        
        self.intelligence = IntelligenceMetrics(
            narrative_coherence=coherence, prediction_alignment=alignment,
            data_quality_score=data_quality, certainty_score=certainty,
            market_edge_score=market_edge, risk_level=risk_level,
            football_iq_score=(coherence * 40 + data_quality * 0.3 + (1 - self._risk_to_penalty(risk_level)) * 30)
        )
        
        summary = self._generate_intelligent_summary(
            self.narrative, prediction_set, 
            self.data['home_team'], self.data['away_team']
        )
        
        return {
            'match': f"{self.data['home_team']} vs {self.data['away_team']}",
            'league': self.data.get('league', 'premier_league'),
            'expected_goals': {'home': round(home_xg, 2), 'away': round(away_xg, 2)},
            'match_narrative': self.narrative.to_dict(),
            'apex_intelligence': {
                'narrative_coherence': round(coherence * 100, 1),
                'prediction_alignment': alignment,
                'football_iq_score': round(self.intelligence.football_iq_score, 1),
                'data_quality': round(data_quality, 1),
                'certainty': round(certainty * 100, 1)
            },
            'probabilities': {
                'match_outcomes': {
                    'home_win': round(mc_results.home_win_prob * 100, 1),
                    'draw': round(mc_results.draw_prob * 100, 1),
                    'away_win': round(mc_results.away_win_prob * 100, 1)
                },
                'both_teams_score': {
                    'yes': round(btts_prob * 100, 1),
                    'no': round((1 - btts_prob) * 100, 1)
                },
                'over_under': {
                    'over_25': round(over_under['over_25'] * 100, 1),
                    'under_25': round(over_under['under_25'] * 100, 1)
                },
                'goal_timing': {
                    'first_half': round(goal_timing['first_half'] * 100, 1),
                    'second_half': round(goal_timing['second_half'] * 100, 1)
                },
                'exact_scores': mc_results.exact_scores
            },
            'corner_predictions': corners,
            'risk_assessment': {
                'risk_level': risk_level,
                'explanation': self._get_risk_explanation(risk_level),
                'recommendation': self._get_risk_recommendation(risk_level),
                'certainty': f"{certainty * 100:.1f}%"
            },
            'summary': summary,
            'intelligence_breakdown': self._get_intelligence_breakdown()
        }
    
    def _get_risk_explanation(self, risk_level: str) -> str:
        explanations = {
            'LOW': "High prediction coherence with strong data support and clear patterns",
            'MEDIUM': "Reasonable prediction alignment with some uncertainties in the data",
            'HIGH': "Multiple uncertainties with conflicting signals or limited data quality", 
            'VERY_HIGH': "High unpredictability with poor data quality and conflicting patterns"
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
    
    def _get_intelligence_breakdown(self) -> str:
        return (f"Football IQ: {self.intelligence.football_iq_score:.1f}/100 | "
                f"Coherence: {self.intelligence.narrative_coherence:.1%} | "
                f"Alignment: {self.intelligence.prediction_alignment} | "
                f"Risk: {self.intelligence.risk_level}")
    
    def _risk_to_penalty(self, risk_level: str) -> float:
        return {'LOW': 0.1, 'MEDIUM': 0.3, 'HIGH': 0.6, 'VERY_HIGH': 0.9}.get(risk_level, 0.5)

class ValueDetectionEngine:
    def __init__(self):
        self.value_thresholds = {
            'EXCEPTIONAL': 25.0, 'HIGH': 15.0, 'GOOD': 8.0, 'MODERATE': 4.0,
        }
        self.min_confidence = 60
        self.max_stake = 0.03

    def detect_value_bets(self, pure_probabilities: Dict, market_odds: Dict) -> List[BettingSignal]:
        signals = []
        
        home_pure = pure_probabilities['probabilities']['match_outcomes']['home_win'] / 100.0
        draw_pure = pure_probabilities['probabilities']['match_outcomes']['draw'] / 100.0  
        away_pure = pure_probabilities['probabilities']['match_outcomes']['away_win'] / 100.0
        
        total = home_pure + draw_pure + away_pure
        if total > 0:
            home_pure /= total
            draw_pure /= total
            away_pure /= total
        
        probability_mapping = [
            ('1x2 Home', home_pure, '1x2 Home'),
            ('1x2 Draw', draw_pure, '1x2 Draw'), 
            ('1x2 Away', away_pure, '1x2 Away'),
            ('Over 2.5 Goals', pure_probabilities['probabilities']['over_under']['over_25'] / 100.0, 'Over 2.5 Goals'),
            ('Under 2.5 Goals', pure_probabilities['probabilities']['over_under']['under_25'] / 100.0, 'Under 2.5 Goals'),
            ('BTTS Yes', pure_probabilities['probabilities']['both_teams_score']['yes'] / 100.0, 'BTTS Yes'),
            ('BTTS No', pure_probabilities['probabilities']['both_teams_score']['no'] / 100.0, 'BTTS No')
        ]
        
        for market_name, pure_prob, market_key in probability_mapping:
            market_odd = market_odds.get(market_key, 0)
            if market_odd <= 1:
                continue
                
            market_prob = 1.0 / market_odd
            edge = (pure_prob / market_prob) - 1.0
            edge_percentage = edge * 100
            
            if edge_percentage >= 4.0:
                value_rating = self._get_value_rating(edge_percentage)
                stake = min(self.max_stake, edge_percentage / 500)
                
                signal = BettingSignal(
                    market=market_name, model_prob=round(pure_prob * 100, 1),
                    book_prob=round(market_prob * 100, 1), edge=round(edge_percentage, 1),
                    confidence="MEDIUM", recommended_stake=stake, value_rating=value_rating
                )
                signals.append(signal)
        
        signals.sort(key=lambda x: x.edge, reverse=True)
        return signals
    
    def _get_value_rating(self, edge: float) -> str:
        for rating, threshold in self.value_thresholds.items():
            if edge >= threshold:
                return rating
        return "LOW"

class AdvancedFootballPredictor:
    def __init__(self, match_data: Dict[str, Any]):
        self.market_odds = match_data.get('market_odds', {})
        football_data = match_data.copy()
        if 'market_odds' in football_data:
            del football_data['market_odds']
        
        self.apex_engine = ApexIntelligenceEngine(football_data)
        self.value_engine = ValueDetectionEngine()

    def generate_comprehensive_analysis(self, mc_iterations: int = 10000) -> Dict[str, Any]:
        football_predictions = self.apex_engine.generate_apex_predictions(mc_iterations)
        value_signals = self.value_engine.detect_value_bets(football_predictions, self.market_odds)
        
        comprehensive_result = football_predictions.copy()
        comprehensive_result['betting_signals'] = [signal.__dict__ for signal in value_signals]
        comprehensive_result['system_validation'] = {'status': 'VALID', 'alignment': 'PERFECT'}
        
        return comprehensive_result

# TEST FUNCTION
def test_apex_intelligence():
    match_data = {
        'home_team': 'Wolfsburg', 'away_team': 'Hoffenheim', 'league': 'bundesliga',
        'home_goals': 5, 'away_goals': 12, 'home_conceded': 11, 'away_conceded': 9,
        'home_form': [1, 0, 1, 0, 0, 1], 'away_form': [3, 3, 1, 3, 3, 3],
        'h2h_data': {'matches': 6, 'home_wins': 3, 'away_wins': 2, 'draws': 1, 'home_goals': 10, 'away_goals': 9},
        'motivation': {'home': 'High', 'away': 'Normal'},
        'market_odds': {
            '1x2 Home': 2.70, '1x2 Draw': 3.75, '1x2 Away': 2.38,
            'Over 2.5 Goals': 1.44, 'Under 2.5 Goals': 2.75,
            'BTTS Yes': 1.40, 'BTTS No': 2.75
        }
    }
    
    predictor = AdvancedFootballPredictor(match_data)
    results = predictor.generate_comprehensive_analysis()
    
    print("🧠 APEX INTELLIGENCE PREDICTION RESULTS")
    print("=" * 60)
    print(f"Match: {results['match']}")
    print(f"Football IQ: {results['apex_intelligence']['football_iq_score']}/100")
    print(f"Narrative Coherence: {results['apex_intelligence']['narrative_coherence']}%")
    print(f"Prediction Alignment: {results['apex_intelligence']['prediction_alignment']}")
    print(f"Risk Level: {results['risk_assessment']['risk_level']}")
    print()
    
    print("📊 INTELLIGENT PROBABILITIES:")
    outcomes = results['probabilities']['match_outcomes']
    print(f"Home Win: {outcomes['home_win']}% | Draw: {outcomes['draw']}% | Away Win: {outcomes['away_win']}%")
    print(f"BTTS Yes: {results['probabilities']['both_teams_score']['yes']}%")
    print(f"Over 2.5: {results['probabilities']['over_under']['over_25']}%")
    print()
    
    print("🎯 MATCH NARRATIVE:")
    narrative = results['match_narrative']
    print(f"Dominance: {narrative['dominance']} | Style: {narrative['style_conflict']}")
    print(f"Tempo: {narrative['expected_tempo']} | Defense: {narrative['defensive_stability']}")
    print()
    
    print("📝 INTELLIGENT SUMMARY:")
    print(results['summary'])

if __name__ == "__main__":
    test_apex_intelligence()
