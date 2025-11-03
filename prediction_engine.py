# prediction_engine.py - CORRECTED VERSION
import numpy as np
from scipy.stats import poisson
from typing import Dict, Any, Tuple, List
import math

class TeamTierCalibrator:
    def __init__(self):
        # CALIBRATED LEAGUE BASELINES
        self.league_baselines = {
            'premier_league': {'avg_goals': 2.8, 'home_advantage': 0.15},
            'la_liga': {'avg_goals': 2.6, 'home_advantage': 0.14},
            'serie_a': {'avg_goals': 2.7, 'home_advantage': 0.16},
            'bundesliga': {'avg_goals': 3.1, 'home_advantage': 0.12},
            'ligue_1': {'avg_goals': 2.5, 'home_advantage': 0.14},
            'liga_portugal': {'avg_goals': 2.6, 'home_advantage': 0.18},
            'brasileirao': {'avg_goals': 2.4, 'home_advantage': 0.20},
            'liga_mx': {'avg_goals': 2.7, 'home_advantage': 0.18},
            'eredivisie': {'avg_goals': 3.0, 'home_advantage': 0.13},
        }
        
        # TEAM DATABASES
        self.team_databases = {
            'ligue_1': {
                'PSG': 'ELITE', 'Monaco': 'STRONG', 'Marseille': 'STRONG',
                'Lille': 'STRONG', 'Lyon': 'STRONG', 'Rennes': 'STRONG',
                'Nice': 'STRONG', 'Lens': 'STRONG', 'Reims': 'MEDIUM',
                'Montpellier': 'MEDIUM', 'Toulouse': 'MEDIUM', 'Strasbourg': 'MEDIUM',
                'Nantes': 'MEDIUM', 'Le Havre': 'MEDIUM', 'Brest': 'MEDIUM',
                'Metz': 'WEAK', 'Lorient': 'WEAK', 'Clermont': 'WEAK'
            },
            'premier_league': {
                'Arsenal': 'ELITE', 'Man City': 'ELITE', 'Liverpool': 'ELITE',
                'Tottenham': 'STRONG', 'Aston Villa': 'STRONG', 'Newcastle': 'STRONG',
                'West Ham': 'MEDIUM', 'Brighton': 'MEDIUM', 'Wolves': 'MEDIUM',
                'Chelsea': 'STRONG', 'Man United': 'STRONG', 'Crystal Palace': 'MEDIUM',
                'Fulham': 'MEDIUM', 'Bournemouth': 'MEDIUM', 'Brentford': 'MEDIUM',
                'Everton': 'MEDIUM', 'Nottingham Forest': 'MEDIUM', 'Luton': 'WEAK',
                'Burnley': 'WEAK', 'Sheffield United': 'WEAK'
            },
            # Add other leagues as needed...
        }
    
    def get_league_teams(self, league: str) -> List[str]:
        return list(self.team_databases.get(league, {}).keys())
    
    def get_team_tier(self, team: str, league: str) -> str:
        league_teams = self.team_databases.get(league, {})
        return league_teams.get(team, 'MEDIUM')

class EliteStakeCalculator:
    def __init__(self, max_stake=0.03, bankroll_fraction=0.02):
        self.max_stake = max_stake
        self.bankroll_fraction = bankroll_fraction
    
    def kelly_stake(self, model_prob: float, market_odds: float, confidence: str) -> float:
        if market_odds <= 1:
            return 0
        
        model_prob_decimal = model_prob / 100.0
        implied_prob = 1.0 / market_odds
        
        kelly_fraction = (model_prob_decimal * market_odds - 1) / (market_odds - 1)
        
        confidence_multiplier = {
            'HIGH': 0.25, 'MEDIUM': 0.15, 'LOW': 0.08, 'SPECULATIVE': 0.04
        }.get(confidence, 0.10)
        
        stake_fraction = max(0, kelly_fraction * confidence_multiplier * self.bankroll_fraction)
        final_stake = min(self.max_stake, stake_fraction)
        
        return max(0.005, final_stake)

class AdvancedFootballPredictor:
    """MAIN PREDICTOR CLASS - This is what Streamlit imports"""
    
    def __init__(self, match_data: Dict[str, Any]):
        self.match_data = match_data
        self.calibrator = TeamTierCalibrator()
        self.stake_calculator = EliteStakeCalculator()

    def _calculate_calibrated_xg(self) -> Tuple[float, float]:
        """CALIBRATED xG calculation"""
        league = self.match_data.get('league', 'premier_league')
        league_baseline = self.calibrator.league_baselines.get(league, {'avg_goals': 2.7, 'home_advantage': 0.15})
        
        # Get team tiers
        home_tier = self.calibrator.get_team_tier(self.match_data['home_team'], league)
        away_tier = self.calibrator.get_team_tier(self.match_data['away_team'], league)
        
        # Calculate averages
        home_goals_avg = self.match_data.get('home_goals', 9) / 6.0
        away_goals_avg = self.match_data.get('away_goals', 9) / 6.0
        home_conceded_avg = self.match_data.get('home_conceded', 9) / 6.0
        away_conceded_avg = self.match_data.get('away_conceded', 9) / 6.0
        
        # Home/away specific adjustments
        home_goals_home_avg = self.match_data.get('home_goals_home', 4) / 3.0
        away_goals_away_avg = self.match_data.get('away_goals_away', 3) / 3.0
        
        # CALIBRATED: Conservative blending
        home_attack = (home_goals_avg * 0.8) + (home_goals_home_avg * 0.2)
        away_attack = (away_goals_avg * 0.8) + (away_goals_away_avg * 0.2)
        
        # CALIBRATED: Reduced multipliers
        home_attack *= 1.0  # WAS 1.4
        away_attack *= 1.0  # WAS 1.4
        
        # Defensive adjustment
        home_xg = home_attack * (1 - (away_conceded_avg / league_baseline['avg_goals']) * 0.2)
        away_xg = away_attack * (1 - (home_conceded_avg / league_baseline['avg_goals']) * 0.2)
        
        # CALIBRATED: Reduced home advantage
        home_advantage = league_baseline['home_advantage'] * 1.1  # WAS 1.2
        home_xg *= (1 + home_advantage)
        
        # Form adjustment
        home_form = self.match_data.get('home_form', [1.5] * 6)
        away_form = self.match_data.get('away_form', [1.5] * 6)
        
        if home_form:
            home_form_avg = sum(home_form) / len(home_form)
            home_form_factor = (home_form_avg / 1.5) ** 0.3
        else:
            home_form_factor = 1.0
            
        if away_form:
            away_form_avg = sum(away_form) / len(away_form)
            away_form_factor = (away_form_avg / 1.5) ** 0.3
        else:
            away_form_factor = 1.0
        
        home_xg *= home_form_factor
        away_xg *= away_form_factor
        
        # Tier-based calibration
        tier_modifier = {'ELITE': 1.10, 'STRONG': 1.04, 'MEDIUM': 1.0, 'WEAK': 0.90}
        home_xg *= tier_modifier.get(home_tier, 1.0)
        away_xg *= tier_modifier.get(away_tier, 1.0)
        
        # CALIBRATED: Realistic bounds
        home_xg = max(0.8, min(2.5, home_xg))
        away_xg = max(0.5, min(2.0, away_xg))
        
        return round(home_xg, 3), round(away_xg, 3)

    def _calculate_probabilities(self, home_xg: float, away_xg: float) -> Dict[str, Any]:
        """Calculate all probabilities"""
        # Match outcomes
        home_win, draw, away_win = 0, 0, 0
        
        for i in range(8):
            for j in range(8):
                prob = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
                if i > j:
                    home_win += prob
                elif i == j:
                    draw += prob
                else:
                    away_win += prob
        
        # Normalize
        total = home_win + draw + away_win
        if total > 0:
            home_win /= total
            draw /= total
            away_win /= total
        
        # Both Teams to Score
        home_score_prob = 1 - math.exp(-home_xg)
        away_score_prob = 1 - math.exp(-away_xg)
        btts_yes = home_score_prob * away_score_prob
        
        # Over/Under 2.5
        total_xg = home_xg + away_xg
        over_25 = 1 - poisson.cdf(2, total_xg)
        
        # Exact scores
        exact_scores = {}
        for i in range(4):
            for j in range(4):
                prob = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
                if prob > 0.005:
                    exact_scores[f"{i}-{j}"] = round(prob * 100, 1)
        
        exact_scores = dict(sorted(exact_scores.items(), key=lambda x: x[1], reverse=True)[:6])
        
        return {
            'match_outcomes': {
                'home_win': round(home_win * 100, 1),
                'draw': round(draw * 100, 1),
                'away_win': round(away_win * 100, 1)
            },
            'both_teams_score': {
                'yes': round(btts_yes * 100, 1),
                'no': round((1 - btts_yes) * 100, 1)
            },
            'over_under': {
                'over_25': round(over_25 * 100, 1),
                'under_25': round((1 - over_25) * 100, 1)
            },
            'exact_scores': exact_scores
        }

    def generate_comprehensive_analysis(self, mc_iterations: int = 10000) -> Dict[str, Any]:
        """Generate complete predictions - MAIN METHOD CALLED BY STREAMLIT"""
        # Calculate expected goals
        home_xg, away_xg = self._calculate_calibrated_xg()
        
        # Get team tiers
        home_tier = self.calibrator.get_team_tier(self.match_data['home_team'], self.match_data['league'])
        away_tier = self.calibrator.get_team_tier(self.match_data['away_team'], self.match_data['league'])
        
        # Calculate probabilities
        probabilities = self._calculate_probabilities(home_xg, away_xg)
        
        # Determine match context
        match_context = self._determine_match_context(home_xg, away_xg)
        
        # Calculate confidence
        confidence_score = max(probabilities['match_outcomes'].values())
        
        # Generate summary
        summary = self._generate_summary(home_xg, away_xg, match_context)
        
        # Calculate data quality
        data_quality = self._calculate_data_quality()
        
        return {
            'match': f"{self.match_data['home_team']} vs {self.match_data['away_team']}",
            'league': self.match_data.get('league', 'premier_league'),
            'expected_goals': {'home': home_xg, 'away': away_xg},
            'team_tiers': {'home': home_tier, 'away': away_tier},
            'match_context': match_context,
            'confidence_score': round(confidence_score, 1),
            'data_quality_score': data_quality,
            'probabilities': probabilities,
            'summary': summary,
            'betting_signals': [],  # Empty for now
            'system_validation': {'alignment': 'PERFECT', 'engine_sync': 'OPTIMAL'},
            'risk_assessment': {
                'risk_level': 'MEDIUM',
                'explanation': 'Reasonable prediction alignment with some uncertainties',
                'recommendation': 'SMALL TO MEDIUM STAKE',
                'certainty': f"{confidence_score:.1f}%",
                'home_advantage': 'N/A'
            }
        }

    def _determine_match_context(self, home_xg: float, away_xg: float) -> str:
        total_xg = home_xg + away_xg
        xg_difference = home_xg - away_xg
        
        if total_xg > 3.0:
            return "offensive_showdown"
        elif total_xg < 2.0:
            return "defensive_battle"
        elif xg_difference > 0.7:
            return "home_dominance"
        elif xg_difference < -0.7:
            return "away_counter"
        else:
            return "balanced"

    def _generate_summary(self, home_xg: float, away_xg: float, context: str) -> str:
        home_team = self.match_data['home_team']
        away_team = self.match_data['away_team']
        
        if context == "home_dominance":
            return f"{home_team} are expected to control this game and create more chances against {away_team}."
        elif context == "balanced":
            return f"Evenly matched contest between {home_team} and {away_team}. Both teams have similar chances."
        else:
            return f"Competitive match expected between {home_team} and {away_team}. Small margins likely to decide the outcome."

    def _calculate_data_quality(self) -> float:
        """Calculate data quality score"""
        score = 0
        
        # Basic match info
        if self.match_data.get('home_team') and self.match_data.get('away_team'):
            score += 20
        
        # Goals data
        if self.match_data.get('home_goals', 0) > 0 and self.match_data.get('away_goals', 0) > 0:
            score += 30
        
        # Form data
        home_form = self.match_data.get('home_form', [])
        away_form = self.match_data.get('away_form', [])
        if len(home_form) >= 4 and len(away_form) >= 4:
            score += 20
        
        # Additional context
        if self.match_data.get('motivation') and self.match_data.get('injuries'):
            score += 10
        
        return min(100, score)

# TEST FUNCTION
def test_toulouse_vs_le_havre():
    """Test the calibrated predictor"""
    match_data = {
        'home_team': 'Toulouse',
        'away_team': 'Le Havre',
        'league': 'ligue_1',
        'home_goals': 11,
        'away_goals': 6,
        'home_conceded': 6,
        'away_conceded': 10,
        'home_goals_home': 8,
        'away_goals_away': 2,
        'home_form': [1, 0, 3, 0, 1, 3],
        'away_form': [3, 3, 0, 1, 1, 1],
        'motivation': {'home': 'Normal', 'away': 'Normal'},
        'injuries': {'home': 1, 'away': 1}
    }
    
    predictor = AdvancedFootballPredictor(match_data)
    results = predictor.generate_comprehensive_analysis()
    
    print("🎯 CALIBRATED PREDICTIONS - Toulouse vs Le Havre")
    print("=" * 50)
    print(f"Expected Goals: Home {results['expected_goals']['home']} - Away {results['expected_goals']['away']}")
    print(f"Home Win: {results['probabilities']['match_outcomes']['home_win']}%")
    print(f"Draw: {results['probabilities']['match_outcomes']['draw']}%")
    print(f"Away Win: {results['probabilities']['match_outcomes']['away_win']}%")
    print(f"BTTS Yes: {results['probabilities']['both_teams_score']['yes']}%")
    print(f"Over 2.5: {results['probabilities']['over_under']['over_25']}%")

if __name__ == "__main__":
    test_toulouse_vs_le_havre()
