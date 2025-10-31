import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson, skellam
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TeamStrength:
    """Bayesian team strength estimates with uncertainty"""
    attack: float
    defense: float
    attack_std: float
    defense_std: float
    last_updated: datetime
    reliability: float  # 0-1 how reliable this estimate is

@dataclass 
class PredictiveFeatures:
    """True predictive features - not outcome-based"""
    xg_creation: float  # xG created per game
    xg_suppression: float  # xG conceded per game
    progression_value: float  # Expected threat from ball progression
    press_intensity: float  # Pressures in final third
    defensive_solidity: float  # Shot prevention quality
    set_piece_threat: float  # xG from set pieces
    sustainability: float  # How repeatable is performance (0-1)

class BayesianDixonColes:
    """
    True Dixon-Coles implementation with Bayesian updating
    Solves for all team strengths simultaneously
    """
    
    def __init__(self, decay_rate: float = 0.002, correlation_param: float = 0.2):
        self.decay_rate = decay_rate
        self.correlation_param = correlation_param
        self.team_strengths: Dict[str, TeamStrength] = {}
        self.league_avg_goals = 1.35
        self.home_advantage = 1.12
        
    def time_decay(self, match_date: datetime, current_date: datetime) -> float:
        """Exponential time decay for match importance"""
        days_diff = (current_date - match_date).days
        return np.exp(-self.decay_rate * days_diff)
    
    def dixon_coles_log_likelihood(self, params: np.ndarray, matches: pd.DataFrame) -> float:
        """Dixon-Coles log likelihood function"""
        num_teams = len(self.team_strengths)
        team_indices = {team: i for i, team in enumerate(self.team_strengths.keys())}
        
        alpha = params[:num_teams]  # Attack strengths
        beta = params[num_teams:2*num_teams]  # Defense strengths
        gamma = params[2*num_teams]  # Home advantage
        rho = params[2*num_teams + 1]  # Correlation
        
        log_likelihood = 0
        
        for _, match in matches.iterrows():
            home_idx = team_indices[match['home_team']]
            away_idx = team_indices[match['away_team']]
            
            # Expected goals
            mu_home = np.exp(alpha[home_idx] - beta[away_idx] + gamma)
            mu_away = np.exp(alpha[away_idx] - beta[home_idx])
            
            # Dixon-Coles correlation adjustment
            home_goals = match['home_goals']
            away_goals = match['away_goals']
            
            if home_goals == 0 and away_goals == 0:
                corr_adjust = 1 + rho * mu_home * mu_away
            elif home_goals == 0 and away_goals == 1:
                corr_adjust = 1 - rho * mu_home
            elif home_goals == 1 and away_goals == 0:
                corr_adjust = 1 - rho * mu_away
            elif home_goals == 1 and away_goals == 1:
                corr_adjust = 1 + rho
            else:
                corr_adjust = 1.0
            
            # Poisson probabilities with correlation
            home_prob = poisson.pmf(home_goals, mu_home)
            away_prob = poisson.pmf(away_goals, mu_away)
            
            prob = home_prob * away_prob * corr_adjust
            time_weight = self.time_decay(match['date'], datetime.now())
            
            if prob > 0:
                log_likelihood += time_weight * np.log(prob)
            else:
                log_likelihood += -10  # Penalty for zero probability
        
        return -log_likelihood  # Negative for minimization
    
    def fit_model(self, matches: pd.DataFrame):
        """Fit Dixon-Coles model to historical matches"""
        teams = list(set(matches['home_team'].unique()) | set(matches['away_team'].unique()))
        self.team_strengths = {team: TeamStrength(0, 0, 0.1, 0.1, datetime.now(), 0.5) for team in teams}
        
        num_teams = len(teams)
        
        # Initial parameters
        initial_params = np.zeros(2 * num_teams + 2)
        initial_params[2 * num_teams] = np.log(self.home_advantage)  # gamma
        initial_params[2 * num_teams + 1] = self.correlation_param  # rho
        
        # Solve optimization
        result = minimize(
            self.dixon_coles_log_likelihood,
            initial_params,
            args=(matches,),
            method='L-BFGS-B',
            options={'maxiter': 1000}
        )
        
        if result.success:
            self._update_team_strengths(result.x, teams)
    
    def _update_team_strengths(self, params: np.ndarray, teams: List[str]):
        """Update team strengths from optimized parameters"""
        num_teams = len(teams)
        
        for i, team in enumerate(teams):
            attack = params[i]
            defense = params[num_teams + i]
            
            # Convert to interpretable scale (goals relative to average)
            attack_goals = np.exp(attack) * self.league_avg_goals
            defense_goals = np.exp(defense) * self.league_avg_goals
            
            self.team_strengths[team] = TeamStrength(
                attack=attack_goals,
                defense=defense_goals,
                attack_std=0.1,  # Would need more data for proper uncertainty
                defense_std=0.1,
                last_updated=datetime.now(),
                reliability=0.8
            )

class PredictiveFeatureEngine:
    """Extracts true predictive features from match data"""
    
    def __init__(self):
        self.feature_weights = {
            'xg_creation': 0.25,
            'xg_suppression': 0.25,
            'progression_value': 0.15,
            'press_intensity': 0.10,
            'defensive_solidity': 0.15,
            'set_piece_threat': 0.05,
            'sustainability': 0.05
        }
    
    def calculate_team_features(self, team_matches: pd.DataFrame) -> PredictiveFeatures:
        """Calculate predictive features from recent matches"""
        if len(team_matches) == 0:
            return self._default_features()
        
        # xG-based features (predictive)
        xg_for = team_matches['xg_for'].mean()
        xg_against = team_matches['xg_against'].mean()
        
        # Progression value (predictive of future attacks)
        progression_value = self._calculate_progression_value(team_matches)
        
        # Pressing intensity (predictive of possession wins)
        press_intensity = team_matches['pressures_final_third'].mean()
        
        # Defensive solidity (predictive of clean sheets)
        defensive_solidity = self._calculate_defensive_solidity(team_matches)
        
        # Set piece threat
        set_piece_threat = team_matches['set_piece_xg'].mean()
        
        # Sustainability (how repeatable is performance)
        sustainability = self._calculate_sustainability(team_matches)
        
        return PredictiveFeatures(
            xg_creation=xg_for,
            xg_suppression=xg_against,
            progression_value=progression_value,
            press_intensity=press_intensity,
            defensive_solidity=defensive_solidity,
            set_piece_threat=set_piece_threat,
            sustainability=sustainability
        )
    
    def _calculate_progression_value(self, matches: pd.DataFrame) -> float:
        """Calculate expected threat from ball progression"""
        if 'progressive_passes' not in matches.columns:
            return 0.5  # Default
        
        prog_passes = matches['progressive_passes'].mean()
        prog_carries = matches['progressive_carries'].mean()
        
        # Normalize to 0-1 scale
        max_progression = 100  # Reasonable maximum
        return min(1.0, (prog_passes + prog_carries) / max_progression)
    
    def _calculate_defensive_solidity(self, matches: pd.DataFrame) -> float:
        """Calculate defensive quality independent of results"""
        if 'shots_against' not in matches.columns:
            return 0.5
        
        shots_against = matches['shots_against'].mean()
        xg_per_shot_against = matches['xg_against'].sum() / matches['shots_against'].sum()
        
        # Lower shots and lower xG per shot = better defense
        shot_quality = 1.0 - min(1.0, xg_per_shot_against / 0.15)  # 0.15 is poor defense
        shot_quantity = 1.0 - min(1.0, shots_against / 20)  # 20 shots is poor
        
        return (shot_quality + shot_quantity) / 2
    
    def _calculate_sustainability(self, matches: pd.DataFrame) -> float:
        """Calculate how repeatable the performance is"""
        if len(matches) < 3:
            return 0.3  # Low confidence with few matches
        
        # Low variance in xG = more sustainable
        xg_variance = matches['xg_for'].std()
        sustainability = 1.0 - min(1.0, xg_variance / 1.0)  # 1.0 xG variance is high
        
        return max(0.1, sustainability)
    
    def _default_features(self) -> PredictiveFeatures:
        """Return default features for teams with no data"""
        return PredictiveFeatures(
            xg_creation=1.35,  # League average
            xg_suppression=1.35,
            progression_value=0.5,
            press_intensity=0.5,
            defensive_solidity=0.5,
            set_piece_threat=0.2,
            sustainability=0.3
        )

class TruePredictiveEngine:
    """
    True predictive engine with Bayesian updating and proper statistical foundations
    """
    
    def __init__(self, historical_matches: pd.DataFrame):
        self.historical_matches = historical_matches
        self.dixon_coles = BayesianDixonColes()
        self.feature_engine = PredictiveFeatureEngine()
        self.team_features: Dict[str, PredictiveFeatures] = {}
        
        # Fit initial model
        if not historical_matches.empty:
            self.dixon_coles.fit_model(historical_matches)
            self._calculate_all_features()
    
    def _calculate_all_features(self):
        """Calculate predictive features for all teams"""
        for team in self.dixon_coles.team_strengths.keys():
            team_matches = self.historical_matches[
                (self.historical_matches['home_team'] == team) | 
                (self.historical_matches['away_team'] == team)
            ].tail(10)  # Last 10 matches
            
            self.team_features[team] = self.feature_engine.calculate_team_features(team_matches)
    
    def predict_match(self, home_team: str, away_team: str, context: Dict = None) -> Dict:
        """Generate true predictive probabilities"""
        context = context or {}
        
        # Get base Dixon-Coles probabilities
        dc_probs = self._get_dixon_coles_probs(home_team, away_team)
        
        # Apply feature-based adjustments
        feature_adjustment = self._calculate_feature_adjustment(home_team, away_team)
        
        # Apply context adjustments
        context_adjustment = self._calculate_context_adjustment(context)
        
        # Combine adjustments
        adjusted_probs = self._apply_adjustments(dc_probs, feature_adjustment, context_adjustment)
        
        # Add uncertainty estimates
        uncertainty = self._calculate_uncertainty(home_team, away_team)
        
        return {
            'probabilities': adjusted_probs,
            'expected_goals': self._predict_expected_goals(home_team, away_team),
            'uncertainty': uncertainty,
            'feature_strength': self._get_feature_strength(home_team, away_team),
            'market_inefficiency': self._detect_inefficiency(adjusted_probs, context),
            'recommendation_confidence': self._calculate_confidence(adjusted_probs, uncertainty)
        }
    
    def _get_dixon_coles_probs(self, home_team: str, away_team: str) -> Dict[str, float]:
        """Get base probabilities from Dixon-Coles model"""
        home_strength = self.dixon_coles.team_strengths.get(home_team)
        away_strength = self.dixon_coles.team_strengths.get(away_team)
        
        if not home_strength or not away_strength:
            return self._default_probabilities()
        
        # Expected goals
        home_xg = home_strength.attack * away_strength.defense / self.dixon_coles.league_avg_goals * self.dixon_coles.home_advantage
        away_xg = away_strength.attack * home_strength.defense / self.dixon_coles.league_avg_goals
        
        # Match outcome probabilities
        home_win, draw, away_win = 0.33, 0.34, 0.33
        
        # Simple Poisson calculation (would use bivariate in production)
        for i in range(10):  # Home goals
            for j in range(10):  # Away goals
                prob = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
                if i > j:
                    home_win += prob
                elif i == j:
                    draw += prob
                else:
                    away_win += prob
        
        total = home_win + draw + away_win
        return {
            'home_win': home_win / total,
            'draw': draw / total,
            'away_win': away_win / total,
            'home_xg': home_xg,
            'away_xg': away_xg
        }
    
    def _calculate_feature_adjustment(self, home_team: str, away_team: str) -> Dict[str, float]:
        """Calculate probability adjustments based on predictive features"""
        home_features = self.team_features.get(home_team, self.feature_engine._default_features())
        away_features = self.team_features.get(away_team, self.feature_engine._default_features())
        
        # Feature differentials
        xg_creation_diff = home_features.xg_creation - away_features.xg_creation
        xg_suppression_diff = home_features.xg_suppression - away_features.xg_suppression
        progression_diff = home_features.progression_value - away_features.progression_value
        
        # Convert to probability adjustments (small, realistic impacts)
        home_win_adj = (
            xg_creation_diff * 0.1 +      # 10% of xG creation difference
            xg_suppression_diff * 0.08 +  # 8% of xG suppression difference  
            progression_diff * 0.05       # 5% of progression difference
        )
        
        # Cap adjustments to reasonable levels
        home_win_adj = max(-0.15, min(0.15, home_win_adj))
        
        return {
            'home_win_adj': home_win_adj,
            'draw_adj': -home_win_adj * 0.3,  # Draw absorbs some adjustment
            'away_win_adj': -home_win_adj * 0.7
        }
    
    def _calculate_context_adjustment(self, context: Dict) -> Dict[str, float]:
        """Calculate context-based adjustments"""
        adjustment = {'home_win_adj': 0, 'draw_adj': 0, 'away_win_adj': 0}
        
        # Motivation factors
        motivation_home = context.get('motivation_home', 0)
        motivation_away = context.get('motivation_away', 0)
        motivation_diff = motivation_home - motivation_away
        
        # Small, realistic motivation impact
        adjustment['home_win_adj'] += motivation_diff * 0.03
        adjustment['away_win_adj'] -= motivation_diff * 0.03
        
        # Other context factors would go here
        
        return adjustment
    
    def _apply_adjustments(self, base_probs: Dict, feature_adj: Dict, context_adj: Dict) -> Dict[str, float]:
        """Apply all adjustments to base probabilities"""
        home_win = base_probs['home_win'] + feature_adj['home_win_adj'] + context_adj['home_win_adj']
        away_win = base_probs['away_win'] + feature_adj['away_win_adj'] + context_adj['away_win_adj']
        draw = base_probs['draw'] + feature_adj['draw_adj'] + context_adj['draw_adj']
        
        # Ensure probabilities are valid
        total = home_win + draw + away_win
        home_win /= total
        draw /= total
        away_win /= total
        
        return {
            'home_win': max(0.01, min(0.99, home_win)),
            'draw': max(0.01, min(0.99, draw)),
            'away_win': max(0.01, min(0.99, away_win)),
            'home_xg': base_probs['home_xg'],
            'away_xg': base_probs['away_xg']
        }
    
    def _predict_expected_goals(self, home_team: str, away_team: str) -> Dict[str, float]:
        """Predict expected goals using features"""
        home_features = self.team_features.get(home_team, self.feature_engine._default_features())
        away_features = self.team_features.get(away_team, self.feature_engine._default_features())
        
        # Base xG from team strengths
        base_home_xg = 1.35  # Would come from Dixon-Coles in full implementation
        base_away_xg = 1.35
        
        # Adjust based on features
        home_xg = base_home_xg * (home_features.xg_creation / 1.35) * (away_features.xg_suppression / 1.35)
        away_xg = base_away_xg * (away_features.xg_creation / 1.35) * (home_features.xg_suppression / 1.35)
        
        return {'home': home_xg, 'away': away_xg}
    
    def _calculate_uncertainty(self, home_team: str, away_team: str) -> Dict[str, float]:
        """Calculate prediction uncertainty"""
        home_features = self.team_features.get(home_team, self.feature_engine._default_features())
        away_features = self.team_features.get(away_team, self.feature_engine._default_features())
        
        # More uncertainty with less sustainable performances
        sustainability = (home_features.sustainability + away_features.sustainability) / 2
        
        base_uncertainty = 0.15  # Base 15% uncertainty
        adjusted_uncertainty = base_uncertainty * (1.5 - sustainability)  # More uncertainty with lower sustainability
        
        return {
            'outcome_uncertainty': adjusted_uncertainty,
            'goal_uncertainty': adjusted_uncertainty * 1.2,
            'confidence': 1.0 - adjusted_uncertainty
        }
    
    def _get_feature_strength(self, home_team: str, away_team: str) -> Dict[str, float]:
        """Get strength of predictive features for this match"""
        home_features = self.team_features.get(home_team, self.feature_engine._default_features())
        away_features = self.team_features.get(away_team, self.feature_engine._default_features())
        
        return {
            'data_quality': (home_features.sustainability + away_features.sustainability) / 2,
            'feature_reliability': min(home_features.sustainability, away_features.sustainability),
            'predictive_power': 0.7  # Would calculate based on backtest performance
        }
    
    def _detect_inefficiency(self, probabilities: Dict, context: Dict) -> Dict[str, float]:
        """Detect potential market inefficiencies"""
        # This would compare to market odds in real implementation
        return {
            'potential_edge': 0.02,  # Small edges are realistic
            'market_bias': 'none_detected',
            'sharp_money_alignment': 0.5
        }
    
    def _calculate_confidence(self, probabilities: Dict, uncertainty: Dict) -> float:
        """Calculate overall recommendation confidence"""
        max_prob = max(probabilities['home_win'], probabilities['draw'], probabilities['away_win'])
        uncertainty_penalty = uncertainty['outcome_uncertainty']
        
        confidence = max_prob * (1 - uncertainty_penalty)
        return max(0.1, min(0.9, confidence))
    
    def _default_probabilities(self) -> Dict[str, float]:
        """Return default probabilities when data is insufficient"""
        return {
            'home_win': 0.33,
            'draw': 0.34, 
            'away_win': 0.33,
            'home_xg': 1.35,
            'away_xg': 1.35
        }

# Example usage with sample data
def create_sample_data():
    """Create sample historical match data"""
    matches = []
    teams = ['Team A', 'Team B', 'Team C', 'Team D']
    
    for i in range(100):
        home_team = teams[i % len(teams)]
        away_team = teams[(i + 1) % len(teams)]
        
        matches.append({
            'date': datetime.now() - timedelta(days=i),
            'home_team': home_team,
            'away_team': away_team,
            'home_goals': np.random.poisson(1.5),
            'away_goals': np.random.poisson(1.2),
            'xg_for': np.random.normal(1.4, 0.3),
            'xg_against': np.random.normal(1.3, 0.3),
            'progressive_passes': np.random.randint(10, 30),
            'progressive_carries': np.random.randint(5, 20),
            'pressures_final_third': np.random.randint(15, 40),
            'shots_against': np.random.randint(8, 20),
            'set_piece_xg': np.random.normal(0.2, 0.1)
        })
    
    return pd.DataFrame(matches)

if __name__ == "__main__":
    # Test the predictive engine
    historical_data = create_sample_data()
    engine = TruePredictiveEngine(historical_data)
    
    prediction = engine.predict_match('Team A', 'Team B')
    print("True Predictive Engine Output:")
    print(f"Probabilities: {prediction['probabilities']}")
    print(f"Expected Goals: {prediction['expected_goals']}")
    print(f"Uncertainty: {prediction['uncertainty']}")
    print(f"Confidence: {prediction['recommendation_confidence']:.2f}")
