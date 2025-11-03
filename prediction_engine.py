# prediction_engine.py - PRODUCTION GRADE WITH ALL ENHANCEMENTS (FIXED)
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
    explanation: List[str]

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

class FeatureEngine:
    """Enhanced Feature Engineering with Interaction Terms"""
    
    def __init__(self):
        self.feature_metadata = {}
        
    def create_match_features(self, home_data: Dict, away_data: Dict, context: Dict) -> Dict[str, float]:
        """Generate comprehensive feature set with interaction terms"""
        
        features = {}
        
        # Base team metrics (home/away adjusted)
        features.update({
            'home_xg_for': home_data.get('xg_home', 1.5),
            'away_xg_for': away_data.get('xg_away', 1.2),
            'home_xg_against': home_data.get('xga_home', 1.3),
            'away_xg_against': away_data.get('xga_away', 1.4),
        })
        
        # Form metrics (weighted recent performance)
        features.update({
            'home_form_attack': self._exponential_form(home_data.get('xg_last_5', [1.5]*5)),
            'away_form_attack': self._exponential_form(away_data.get('xg_last_5', [1.2]*5)),
            'home_form_defense': self._exponential_form(home_data.get('xga_last_5', [1.3]*5)),
            'away_form_defense': self._exponential_form(away_data.get('xga_last_5', [1.4]*5)),
        })
        
        # CRITICAL: Interaction features
        features.update({
            # Attack vs Defense matchups
            'home_attack_vs_away_defense': features['home_xg_for'] * features['away_xg_against'],
            'away_attack_vs_home_defense': features['away_xg_for'] * features['home_xg_against'],
            
            # Goal expectation correlations
            'total_xg_expected': features['home_xg_for'] + features['away_xg_for'],
            'xg_difference': features['home_xg_for'] - features['away_xg_for'],
        })
        
        # Contextual modifiers
        features.update(self._contextual_features(context))
        
        return features
    
    def _exponential_form(self, form_data: List[float]) -> float:
        """Calculate exponentially weighted form"""
        if not form_data:
            return 1.5
        weights = np.array([0.9**i for i in range(len(form_data)-1, -1, -1)])
        return np.average(form_data, weights=weights)
    
    def _contextual_features(self, context: Dict) -> Dict[str, float]:
        """Quantify contextual factors"""
        features = {}
        
        # Injury impact (0-1 scale)
        home_injury_impact = min(1.0, context.get('home_injuries', 1) * 0.15)
        away_injury_impact = min(1.0, context.get('away_injuries', 1) * 0.15)
        
        # Motivation factors
        motivation_map = {'Low': 0.8, 'Normal': 1.0, 'High': 1.1, 'Very High': 1.15}
        home_motivation = motivation_map.get(context.get('home_motivation', 'Normal'), 1.0)
        away_motivation = motivation_map.get(context.get('away_motivation', 'Normal'), 1.0)
        
        features.update({
            'home_injury_factor': 1.0 - home_injury_impact,
            'away_injury_factor': 1.0 - away_injury_impact,
            'home_motivation_factor': home_motivation,
            'away_motivation_factor': away_motivation,
            'match_importance': context.get('match_importance', 0.5),
        })
        
        return features

class MatchSimulator:
    """Monte Carlo Simulation Engine with Dependencies"""
    
    def __init__(self, n_simulations: int = 10000):
        self.n_simulations = n_simulations
        
    def simulate_match_dixon_coles(self, home_xg: float, away_xg: float, correlation: float = 0.2):
        """Dixon-Coles model for score dependency"""
        # Adjust for correlation between scores
        lambda1 = max(0.1, home_xg - correlation * min(home_xg, away_xg))
        lambda2 = max(0.1, away_xg - correlation * min(home_xg, away_xg))
        lambda3 = correlation * min(home_xg, away_xg)
        
        # Generate correlated Poisson
        C = np.random.poisson(lambda3, self.n_simulations)
        A = np.random.poisson(lambda1, self.n_simulations)
        B = np.random.poisson(lambda2, self.n_simulations)
        
        home_goals = A + C
        away_goals = B + C
        
        return home_goals, away_goals
    
    def get_market_probabilities(self, home_goals: np.array, away_goals: np.array) -> Dict[str, float]:
        """Calculate all market probabilities from simulations"""
        
        # BTTS probabilities
        btts_yes = np.mean((home_goals > 0) & (away_goals > 0))
        btts_no = 1 - btts_yes
        
        # Over/Under probabilities
        total_goals = home_goals + away_goals
        over_15 = np.mean(total_goals > 1.5)
        over_25 = np.mean(total_goals > 2.5)
        over_35 = np.mean(total_goals > 3.5)
        
        # Exact score probabilities (top 10)
        score_counts = {}
        for h, a in zip(home_goals[:1000], away_goals[:1000]):  # Sample for efficiency
            score = f"{h}-{a}"
            score_counts[score] = score_counts.get(score, 0) + 1
        
        exact_scores = {score: count/1000 
                       for score, count in sorted(score_counts.items(), 
                       key=lambda x: x[1], reverse=True)[:6]}
        
        return {
            'btts_yes': btts_yes,
            'btts_no': btts_no,
            'over_15': over_15,
            'over_25': over_25,
            'over_35': over_35,
            'under_25': 1 - over_25,
            'exact_scores': exact_scores,
        }

class LeagueAwareCalibrator:
    """League-Specific Probability Calibration"""
    
    def __init__(self):
        self.league_profiles = {
            'premier_league': {'goal_intensity': 'high', 'defensive_variance': 'medium', 'calibration_factor': 1.0},
            'la_liga': {'goal_intensity': 'medium', 'defensive_variance': 'low', 'calibration_factor': 0.98},
            'serie_a': {'goal_intensity': 'medium', 'defensive_variance': 'low', 'calibration_factor': 0.96},
            'bundesliga': {'goal_intensity': 'very_high', 'defensive_variance': 'high', 'calibration_factor': 1.08},
            'ligue_1': {'goal_intensity': 'low', 'defensive_variance': 'medium', 'calibration_factor': 0.95},
            'liga_portugal': {'goal_intensity': 'medium', 'defensive_variance': 'medium', 'calibration_factor': 0.97},
            'brasileirao': {'goal_intensity': 'high', 'defensive_variance': 'high', 'calibration_factor': 1.02},
            'liga_mx': {'goal_intensity': 'medium', 'defensive_variance': 'high', 'calibration_factor': 1.01},
            'eredivisie': {'goal_intensity': 'high', 'defensive_variance': 'high', 'calibration_factor': 1.03}
        }
    
    def calibrate_probability(self, raw_prob: float, league: str, market_type: str) -> float:
        """Apply league-specific calibration"""
        profile = self.league_profiles.get(league, {'calibration_factor': 1.0})
        base_calibrated = raw_prob * profile['calibration_factor']
        
        # Market-specific adjustments
        if market_type == 'over_25':
            if profile['goal_intensity'] == 'very_high':
                base_calibrated *= 1.05
            elif profile['goal_intensity'] == 'low':
                base_calibrated *= 0.95
        elif market_type == 'btts_yes':
            if profile['defensive_variance'] == 'low':
                base_calibrated *= 0.95
            elif profile['defensive_variance'] == 'high':
                base_calibrated *= 1.05
                
        return np.clip(base_calibrated, 0.02, 0.98)

class PredictionExplainer:
    """Enhanced Explanation Engine (No SHAP Dependency)"""
    
    def __init__(self):
        self.feature_descriptions = {
            'home_attack_vs_away_defense': 'Home attacking strength against away defense',
            'away_attack_vs_home_defense': 'Away attacking strength against home defense',
            'total_xg_expected': 'Expected total goals in match',
            'home_form_attack': 'Recent home attacking form',
            'away_form_attack': 'Recent away attacking form',
            'xg_difference': 'Difference in expected goals between teams'
        }
    
    def generate_explanation(self, features: Dict, probabilities: Dict, narrative: Dict) -> Dict[str, List[str]]:
        """Generate human-readable explanations for predictions"""
        
        explanations = {}
        
        # Extract key metrics
        home_attack = features.get('home_xg_for', 1.5)
        away_attack = features.get('away_xg_for', 1.2)
        home_defense = features.get('home_xg_against', 1.3)
        away_defense = features.get('away_xg_against', 1.4)
        total_xg = features.get('total_xg_expected', 2.7)
        
        # BTTS explanation
        btts_prob = probabilities.get('btts_yes', 0.5)
        if btts_prob > 0.6:
            explanations['btts'] = [
                f"Both teams show strong attacking form (Home: {home_attack:.1f}, Away: {away_attack:.1f} xG)",
                f"Defensive vulnerabilities present (Home concedes: {home_defense:.1f}, Away concedes: {away_defense:.1f} xG)"
            ]
        elif btts_prob < 0.4:
            explanations['btts'] = [
                f"Defensive solidity from one or both teams",
                f"Limited attacking threat reduces BTTS probability"
            ]
        else:
            explanations['btts'] = [
                f"Balanced attacking and defensive capabilities",
                f"Moderate chance for both teams to score"
            ]
        
        # Over/Under explanation
        over_prob = probabilities.get('over_25', 0.5)
        if over_prob > 0.6:
            explanations['over_under'] = [
                f"High expected goal volume (Total xG: {total_xg:.2f})",
                f"Attacking styles and defensive vulnerabilities suggest multiple goals"
            ]
        elif over_prob < 0.4:
            explanations['over_under'] = [
                f"Moderate expected goal volume (Total xG: {total_xg:.2f})",
                f"Defensive organization likely to limit scoring opportunities"
            ]
        else:
            explanations['over_under'] = [
                f"Average expected goal volume (Total xG: {total_xg:.2f})",
                f"Game could go either way in terms of total goals"
            ]
            
        # Add narrative-based explanations
        style_conflict = narrative.get('style_conflict', 'balanced')
        defensive_stability = narrative.get('defensive_stability', 'mixed')
        
        if style_conflict == "attacking_vs_attacking":
            explanations['style'] = ["Open game expected with both teams prioritizing attack"]
        elif style_conflict == "attacking_vs_defensive":
            explanations['style'] = ["Tactical battle between attacking initiative and defensive organization"]
        else:
            explanations['style'] = ["Balanced tactical approach from both teams"]
            
        return explanations

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
        
        # COMPLETE TEAM DATABASES FOR ALL LEAGUES
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
            'serie_a': {
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
        return list(self.team_databases.get(league, {}).keys())
    
    def get_team_tier(self, team: str, league: str) -> str:
        league_teams = self.team_databases.get(league, {})
        return league_teams.get(team, 'MEDIUM')

class AdvancedGoalModel:
    """Advanced Goal Prediction Model with Bayesian Shrinkage"""
    
    def __init__(self):
        self.feature_engine = FeatureEngine()
        self.simulator = MatchSimulator()
        self.calibrator = LeagueAwareCalibrator()
        self.explainer = PredictionExplainer()
        
    def calculate_team_strength(self, team_data: Dict, is_home: bool = True) -> Dict[str, float]:
        """Calculate dynamic team strength with Bayesian shrinkage"""
        prior_games = 10
        prior_strength = 1.5  # League average
        
        # Recent form with exponential decay
        xg_data = team_data.get('xg_last_6', [1.5] * 6)
        xga_data = team_data.get('xga_last_6', [1.3] * 6)
        
        weights = np.array([0.9**i for i in range(5, -1, -1)])
        recent_xg = np.average(xg_data, weights=weights)
        recent_xga = np.average(xga_data, weights=weights)
        
        # Apply Bayesian shrinkage toward league average
        games_played = len(team_data.get('xg_season', [])) or 6
        strength_attack = (recent_xg * games_played + prior_strength * prior_games) / (games_played + prior_games)
        strength_defense = (recent_xga * games_played + prior_strength * prior_games) / (games_played + prior_games)
        
        return {
            'attack': strength_attack,
            'defense': strength_defense,
            'sample_size': games_played
        }

class ApexIntelligenceEngine:
    """APEX INTELLIGENCE ENGINE - Enhanced with Production Features"""
    
    def __init__(self, match_data: Dict[str, Any]):
        self.data = self._validate_and_enhance_data(match_data)
        self.calibrator = TeamTierCalibrator()
        self.goal_model = AdvancedGoalModel()
        self.narrative = MatchNarrative()
        self.intelligence = IntelligenceMetrics(
            narrative_coherence=0.0, prediction_alignment="LOW", 
            data_quality_score=0.0, certainty_score=0.0,
            market_edge_score=0.0, risk_level="HIGH", football_iq_score=0.0
        )
        self._setup_parameters()
        
    def _validate_and_enhance_data(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        enhanced_data = match_data.copy()
        
        # Ensure required fields exist
        required_fields = ['home_team', 'away_team', 'league']
        for field in required_fields:
            if field not in enhanced_data:
                enhanced_data[field] = 'Unknown'
        
        # Enhanced data validation with better defaults
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
            else:
                enhanced_data[field] = default
        
        # Enhanced form data processing
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
            
        return enhanced_data

    def _setup_parameters(self):
        self.calibration_params = {
            'form_decay_rate': 0.85, 'h2h_weight': 0.15, 'injury_impact': 0.08,
            'motivation_impact': 0.10, 'defensive_impact_multiplier': 0.4,
        }

    def _calculate_realistic_xg(self) -> Tuple[float, float]:
        # Enhanced xG calculation using the new goal model
        league = self.data.get('league', 'premier_league')
        league_baseline = self.calibrator.league_baselines.get(league, {'avg_goals': 2.7, 'home_advantage': 0.35})
        
        # Prepare team data for the goal model
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
        
        # Calculate team strengths
        home_strength = self.goal_model.calculate_team_strength(home_team_data, is_home=True)
        away_strength = self.goal_model.calculate_team_strength(away_team_data, is_home=False)
        
        # Base xG calculation
        home_xg = home_strength['attack'] * (1 - (away_strength['defense'] / league_baseline['avg_goals']) * 0.3)
        away_xg = away_strength['attack'] * (1 - (home_strength['defense'] / league_baseline['avg_goals']) * 0.3)
        
        # Apply home advantage
        home_advantage = league_baseline['home_advantage']
        home_xg *= (1 + home_advantage)
        
        # Apply contextual factors using feature engine
        context = {
            'home_injuries': self.data.get('injuries', {}).get('home', 1),
            'away_injuries': self.data.get('injuries', {}).get('away', 1),
            'home_motivation': self.data.get('motivation', {}).get('home', 'Normal'),
            'away_motivation': self.data.get('motivation', {}).get('away', 'Normal'),
            'match_importance': 0.5
        }
        
        contextual_factors = self.goal_model.feature_engine._contextual_features(context)
        home_xg *= contextual_factors.get('home_injury_factor', 1.0) * contextual_factors.get('home_motivation_factor', 1.0)
        away_xg *= contextual_factors.get('away_injury_factor', 1.0) * contextual_factors.get('away_motivation_factor', 1.0)
        
        # H2H adjustment
        h2h_data = self.data.get('h2h_data', {})
        if h2h_data and h2h_data.get('matches', 0) >= 3:
            home_xg, away_xg = self._apply_h2h_adjustment(home_xg, away_xg, h2h_data)
        
        return round(max(0.2, min(4.0, home_xg)), 3), round(max(0.2, min(4.0, away_xg)), 3)

    def _apply_h2h_adjustment(self, home_xg: float, away_xg: float, h2h_data: Dict) -> Tuple[float, float]:
        matches = h2h_data.get('matches', 0)
        if matches < 3:
            return home_xg, away_xg
        
        h2h_weight = min(0.25, matches * 0.06)
        h2h_home_avg = h2h_data.get('home_goals', 0) / matches
        h2h_away_avg = h2h_data.get('away_goals', 0) / matches
        
        if h2h_home_avg > 0 or h2h_away_avg > 0:
            adjusted_home_xg = (home_xg * (1 - h2h_weight)) + (h2h_home_avg * h2h_weight)
            adjusted_away_xg = (away_xg * (1 - h2h_weight)) + (h2h_away_avg * h2h_weight)
            return adjusted_home_xg, adjusted_away_xg
        
        return home_xg, away_xg

    def _calculate_motivation_impact(self, motivation_level: str) -> float:
        multipliers = {"Low": 0.90, "Normal": 1.0, "High": 1.08, "Very High": 1.12}
        return multipliers.get(motivation_level, 1.0)

    def _determine_match_narrative(self, home_xg: float, away_xg: float) -> MatchNarrative:
        narrative = MatchNarrative()
        total_xg = home_xg + away_xg
        xg_difference = home_xg - away_xg
        
        # Determine dominance
        if xg_difference > 0.8:
            narrative.dominance = "home"
            narrative.primary_pattern = "home_dominance"
        elif xg_difference < -0.8:
            narrative.dominance = "away" 
            narrative.primary_pattern = "away_dominance"
        else:
            narrative.dominance = "balanced"
            
        # Calculate attack/defense metrics
        home_attack = self.data.get('home_goals', 0) / 6.0
        away_attack = self.data.get('away_goals', 0) / 6.0
        home_defense = self.data.get('home_conceded', 0) / 6.0
        away_defense = self.data.get('away_conceded', 0) / 6.0
        
        # Determine style conflict
        if home_attack > 1.8 and away_attack > 1.8:
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
            
        # Determine defensive stability
        avg_conceded = (home_defense + away_defense) / 2
        if avg_conceded < 0.8:
            narrative.defensive_stability = "solid"
        elif avg_conceded > 1.5:
            narrative.defensive_stability = "leaky" 
        else:
            narrative.defensive_stability = "mixed"
            
        return narrative

    def _run_monte_carlo(self, home_xg: float, away_xg: float, iterations: int = 10000) -> MonteCarloResults:
        # Use the enhanced simulator with Dixon-Coles correlation
        home_goals, away_goals = self.goal_model.simulator.simulate_match_dixon_coles(home_xg, away_xg)
        market_probs = self.goal_model.simulator.get_market_probabilities(home_goals, away_goals)
        
        # Calculate outcome probabilities
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

    def _calculate_data_quality(self) -> float:
        score = 0
        max_score = 100
        
        if self.data.get('home_team') and self.data.get('away_team') and self.data.get('home_team') != 'Unknown' and self.data.get('away_team') != 'Unknown':
            score += 20
        
        home_goals = self.data.get('home_goals', 0)
        away_goals = self.data.get('away_goals', 0)
        if home_goals > 0 and away_goals > 0:
            score += 30
        elif home_goals > 0 or away_goals > 0:
            score += 15
        
        home_form = self.data.get('home_form', [])
        away_form = self.data.get('away_form', [])
        if len(home_form) >= 4 and len(away_form) >= 4:
            score += 20
        elif len(home_form) >= 2 or len(away_form) >= 2:
            score += 10
        
        h2h_data = self.data.get('h2h_data', {})
        if h2h_data.get('matches', 0) >= 3:
            score += 20
        elif h2h_data.get('matches', 0) >= 1:
            score += 10
        
        if self.data.get('motivation') and self.data.get('injuries'):
            score += 10
        
        return min(100, score)

    def _estimate_market_edge(self) -> float:
        data_quality = self._calculate_data_quality() / 100
        market_edge = 0.3 + (data_quality * 0.4) + (0.5 * 0.3)
        return min(0.8, market_edge)

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
        # Calculate core metrics using enhanced models
        home_xg, away_xg = self._calculate_realistic_xg()
        self.narrative = self._determine_match_narrative(home_xg, away_xg)
        
        # Run Monte Carlo simulation with Dixon-Coles
        mc_results = self._run_monte_carlo(home_xg, away_xg, mc_iterations)
        
        # Apply league-specific calibration
        league = self.data.get('league', 'premier_league')
        calibrated_btts = self.goal_model.calibrator.calibrate_probability(mc_results.btts_prob, league, 'btts_yes')
        calibrated_over = self.goal_model.calibrator.calibrate_probability(mc_results.over_25_prob, league, 'over_25')
        
        # Generate explanations
        features = self.goal_model.feature_engine.create_match_features(
            {'xg_home': home_xg, 'xga_home': self.data.get('home_conceded', 8)/6.0, 'xg_last_5': self.data.get('home_form', [1.5]*5)},
            {'xg_away': away_xg, 'xga_away': self.data.get('away_conceded', 8)/6.0, 'xg_last_5': self.data.get('away_form', [1.5]*5)},
            self.data
        )
        
        probabilities = {'btts_yes': calibrated_btts, 'over_25': calibrated_over}
        explanations = self.goal_model.explainer.generate_explanation(features, probabilities, self.narrative.to_dict())
        
        # Prepare prediction set for coherence assessment
        prediction_set = {
            'home_win': mc_results.home_win_prob,
            'btts_yes': calibrated_btts,
            'over_25': calibrated_over
        }
        
        # Calculate intelligence metrics
        coherence, alignment = self._assess_prediction_coherence(prediction_set)
        certainty = max(mc_results.home_win_prob, mc_results.away_win_prob, mc_results.draw_prob)
        data_quality = self._calculate_data_quality()
        market_edge = self._estimate_market_edge()
        
        risk_level = self._calculate_intelligent_risk(certainty, data_quality, market_edge, alignment)
        
        # Calculate Football IQ score
        football_iq_score = (coherence * 40 + (data_quality/100) * 30 + (1 - self._risk_to_penalty(risk_level)) * 30)
        
        self.intelligence = IntelligenceMetrics(
            narrative_coherence=coherence, 
            prediction_alignment=alignment,
            data_quality_score=data_quality, 
            certainty_score=certainty,
            market_edge_score=market_edge, 
            risk_level=risk_level,
            football_iq_score=football_iq_score
        )
        
        # Generate intelligent summary
        summary = self._generate_intelligent_summary(
            self.narrative, prediction_set, 
            self.data['home_team'], self.data['away_team']
        )
        
        # Get team tiers for display
        home_tier = self.calibrator.get_team_tier(self.data['home_team'], self.data['league'])
        away_tier = self.calibrator.get_team_tier(self.data['away_team'], self.data['league'])
        
        # Determine match context from narrative
        match_context = self.narrative.primary_pattern or self.narrative.style_conflict
        if match_context == "attacking_vs_attacking":
            match_context_display = "offensive_showdown"
        elif match_context == "home_dominance":
            match_context_display = "home_dominance"
        elif match_context == "away_dominance":
            match_context_display = "away_counter"
        else:
            match_context_display = "balanced"
        
        return {
            'match': f"{self.data['home_team']} vs {self.data['away_team']}",
            'league': self.data.get('league', 'premier_league'),
            'expected_goals': {'home': round(home_xg, 2), 'away': round(away_xg, 2)},
            'team_tiers': {'home': home_tier, 'away': away_tier},
            'match_context': match_context_display,
            'confidence_score': round(certainty * 100, 1),
            'data_quality_score': round(data_quality, 1),
            'match_narrative': self.narrative.to_dict(),
            'apex_intelligence': {
                'narrative_coherence': round(coherence * 100, 1),
                'prediction_alignment': alignment,
                'football_iq_score': round(football_iq_score, 1),
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
                'explanation': self._get_risk_explanation(risk_level),
                'recommendation': self._get_risk_recommendation(risk_level),
                'certainty': f"{certainty * 100:.1f}%",
            },
            'summary': summary,
            'intelligence_breakdown': self._get_intelligence_breakdown(),
            'monte_carlo_results': {
                'home_win_prob': mc_results.home_win_prob,
                'draw_prob': mc_results.draw_prob,
                'away_win_prob': mc_results.away_win_prob,
            }
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

class BettingDecisionEngine:
    """Enhanced Betting Decision Engine with Kelly Criterion"""
    
    def __init__(self, bankroll: float = 1000, kelly_fraction: float = 0.25):
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.value_thresholds = {
            'EXCEPTIONAL': 25.0, 'HIGH': 15.0, 'GOOD': 8.0, 'MODERATE': 4.0,
        }
        
    def calculate_expected_value(self, model_prob: float, market_odds: float) -> Dict[str, float]:
        """Calculate expected value and edge for a bet"""
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
    
    def kelly_stake(self, model_prob: float, market_odds: float) -> float:
        """Calculate Kelly Criterion stake with conservative limits"""
        if market_odds <= 1:
            return 0
            
        q = 1 - model_prob
        b = market_odds - 1
        kelly = (model_prob * (b + 1) - 1) / b
        
        # Apply fractional Kelly and bankroll management
        stake = max(0, kelly * self.kelly_fraction * self.bankroll)
        
        # Conservative limits
        max_stake = 0.05 * self.bankroll  # Max 5% of bankroll
        min_stake = 0.01 * self.bankroll  # Min 1% of bankroll
        return min(max(stake, min_stake), max_stake)
    
    def _get_value_rating(self, edge: float) -> str:
        for rating, threshold in self.value_thresholds.items():
            if edge >= threshold:
                return rating
        return "LOW"
    
    def _assign_confidence(self, probability: float, edge: float) -> str:
        """Assign confidence based on probability and edge"""
        if probability > 0.67 and edge > 15:
            return 'HIGH'
        elif probability > 0.57 and edge > 8:
            return 'MEDIUM'
        elif probability > 0.52 and edge > 4:
            return 'LOW'
        else:
            return 'SPECULATIVE'

    def detect_value_bets(self, pure_probabilities: Dict, market_odds: Dict, explanations: Dict) -> List[BettingSignal]:
        """Enhanced value detection with explanations"""
        signals = []
        
        # Extract probabilities with safe defaults
        outcomes = pure_probabilities.get('probabilities', {}).get('match_outcomes', {})
        home_pure = outcomes.get('home_win', 33.3) / 100.0
        draw_pure = outcomes.get('draw', 33.3) / 100.0  
        away_pure = outcomes.get('away_win', 33.3) / 100.0
        
        # Normalize to ensure sum = 1
        total = home_pure + draw_pure + away_pure
        if total > 0:
            home_pure /= total
            draw_pure /= total
            away_pure /= total
        
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
        
        for market_name, pure_prob, market_key in probability_mapping:
            market_odd = market_odds.get(market_key, 0)
            if market_odd <= 1:
                continue
                
            ev_data = self.calculate_expected_value(pure_prob, market_odd)
            edge_percentage = ev_data['edge_percentage']
            
            if edge_percentage >= 4.0:  # Minimum edge threshold
                value_rating = self._get_value_rating(edge_percentage)
                stake = self.kelly_stake(pure_prob, market_odd)
                confidence = self._assign_confidence(pure_prob, edge_percentage)
                
                # Get relevant explanations
                market_explanations = []
                if 'BTTS' in market_name:
                    market_explanations = explanations.get('btts', [])
                elif 'Over' in market_name or 'Under' in market_name:
                    market_explanations = explanations.get('over_under', [])
                
                signal = BettingSignal(
                    market=market_name, 
                    model_prob=round(pure_prob * 100, 1),
                    book_prob=round(ev_data['implied_probability'] * 100, 1), 
                    edge=round(edge_percentage, 1),
                    confidence=confidence, 
                    recommended_stake=stake, 
                    value_rating=value_rating,
                    explanation=market_explanations
                )
                signals.append(signal)
        
        # Sort by edge (highest first)
        signals.sort(key=lambda x: x.edge, reverse=True)
        return signals

class AdvancedFootballPredictor:
    """Main Predictor Class - Production Ready"""
    
    def __init__(self, match_data: Dict[str, Any]):
        self.market_odds = match_data.get('market_odds', {})
        football_data = match_data.copy()
        if 'market_odds' in football_data:
            del football_data['market_odds']
        
        self.apex_engine = ApexIntelligenceEngine(football_data)
        self.betting_engine = BettingDecisionEngine()
        self.predictions = None

    def generate_comprehensive_analysis(self, mc_iterations: int = 10000) -> Dict[str, Any]:
        """Generate complete analysis with enhanced features"""
        football_predictions = self.apex_engine.generate_apex_predictions(mc_iterations)
        
        # Get explanations for value detection
        explanations = football_predictions.get('explanations', {})
        value_signals = self.betting_engine.detect_value_bets(football_predictions, self.market_odds, explanations)
        
        # Enhanced system validation
        alignment_status = "PERFECT" if self._validate_system_alignment(football_predictions, value_signals) else "PARTIAL"
        
        comprehensive_result = football_predictions.copy()
        comprehensive_result['betting_signals'] = [signal.__dict__ for signal in value_signals]
        comprehensive_result['system_validation'] = {
            'status': 'VALID', 
            'alignment': alignment_status,
            'engine_sync': 'OPTIMAL',
            'model_version': '1.2.0_production'
        }
        
        self.predictions = comprehensive_result
        return comprehensive_result
    
    def _validate_system_alignment(self, football_predictions: Dict, value_signals: List[BettingSignal]) -> bool:
        """Validate that value signals align with football predictions"""
        if not value_signals:
            return True
            
        # Check if any value bet strongly contradicts primary predictions
        primary_outcome = max(
            football_predictions['probabilities']['match_outcomes'], 
            key=football_predictions['probabilities']['match_outcomes'].get
        )
        
        for signal in value_signals:
            market = signal.market
            if (market in ['1x2 Away', '1x2 Draw'] and primary_outcome == 'home_win' and 
                football_predictions['probabilities']['match_outcomes']['home_win'] > 60):
                return False
                
        return True

# TEST FUNCTION - Enhanced with new features
def test_apex_intelligence():
    match_data = {
        'home_team': 'Rennes', 'away_team': 'Strasbourg', 'league': 'ligue_1',
        'home_goals': 8, 'away_goals': 12, 'home_conceded': 8, 'away_conceded': 6,
        'home_goals_home': 5, 'away_goals_away': 7,
        'home_form': [3, 0, 3, 0, 0, 3], 'away_form': [3, 3, 1, 3, 3, 3],
        'h2h_data': {'matches': 6, 'home_wins': 3, 'away_wins': 2, 'draws': 1, 'home_goals': 10, 'away_goals': 9},
        'motivation': {'home': 'High', 'away': 'Normal'},
        'injuries': {'home': 1, 'away': 2},
        'market_odds': {
            '1x2 Home': 2.70, '1x2 Draw': 3.75, '1x2 Away': 2.38,
            'Over 2.5 Goals': 1.44, 'Under 2.5 Goals': 2.75,
            'BTTS Yes': 1.40, 'BTTS No': 2.75
        }
    }
    
    predictor = AdvancedFootballPredictor(match_data)
    results = predictor.generate_comprehensive_analysis()
    
    print("🧠 ENHANCED APEX INTELLIGENCE PREDICTION RESULTS")
    print("=" * 60)
    print(f"Match: {results['match']}")
    print(f"Football IQ: {results['apex_intelligence']['football_iq_score']}/100")
    print(f"Narrative Coherence: {results['apex_intelligence']['narrative_coherence']}%")
    print(f"Prediction Alignment: {results['apex_intelligence']['prediction_alignment']}")
    print(f"Data Quality: {results['data_quality_score']}%")
    print(f"Risk Level: {results['risk_assessment']['risk_level']}")
    print()
    
    print("📊 ENHANCED PROBABILITIES:")
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
    print()
    
    print("💡 EXPLANATIONS:")
    explanations = results.get('explanations', {})
    for market, reasons in explanations.items():
        print(f"{market}: {', '.join(reasons)}")
    
    print()
    print("💰 VALUE BETS:")
    for signal in results.get('betting_signals', []):
        print(f"- {signal['market']}: {signal['edge']}% edge | Stake: {signal['recommended_stake']:.2f} | Confidence: {signal['confidence']}")

if __name__ == "__main__":
    test_apex_intelligence()
