# prediction_engine.py - DEBUGGED & CALIBRATED VERSION
import numpy as np
from scipy.stats import poisson
from typing import Dict, Any, Tuple, List
import math

class TeamTierCalibrator:
    def __init__(self):
        self.league_baselines = {
            'premier_league': {'avg_goals': 2.8, 'home_advantage': 0.35},
            'la_liga': {'avg_goals': 2.6, 'home_advantage': 0.32},
            'serie_a': {'avg_goals': 2.7, 'home_advantage': 0.38},
            'bundesliga': {'avg_goals': 3.1, 'home_advantage': 0.28},
            'ligue_1': {'avg_goals': 2.5, 'home_advantage': 0.34},
        }
        
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
            },
            # Add other leagues as needed
        }
    
    def get_team_tier(self, team: str, league: str) -> str:
        league_teams = self.team_databases.get(league, {})
        return league_teams.get(team, 'MEDIUM')

class DebuggedFootballPredictor:
    """DEBUGGED & CALIBRATED PREDICTION ENGINE"""
    
    def __init__(self, match_data: Dict[str, Any]):
        self.data = self._validate_data(match_data)
        self.calibrator = TeamTierCalibrator()
        
        # CALIBRATION CONSTANTS - TUNED FOR REALISM
        self.calibration = {
            'attack_factor': 1.4,        # Increased to boost goal counts
            'defense_factor': 0.7,       # Reduced defensive impact
            'form_weight': 0.4,          # Increased form importance
            'tier_impact': 0.15,         # Tier-based adjustments
            'home_boost': 1.2,           # Additional home advantage
            'min_xg': 0.3,               # Minimum expected goals
            'max_xg': 4.5                # Maximum expected goals
        }
        
    def _validate_data(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced data validation with better defaults"""
        enhanced_data = match_data.copy()
        
        # Ensure required fields
        required_fields = ['home_team', 'away_team', 'league']
        for field in required_fields:
            if field not in enhanced_data:
                enhanced_data[field] = 'Unknown'
        
        # Set realistic defaults for missing data
        predictive_defaults = {
            'home_goals': 9, 'away_goals': 9,      # Increased defaults
            'home_conceded': 9, 'away_conceded': 9,
            'home_goals_home': 5, 'away_goals_away': 4,
        }
        
        for field, default in predictive_defaults.items():
            if field not in enhanced_data:
                enhanced_data[field] = default
            else:
                # Ensure values are reasonable
                enhanced_data[field] = max(0, min(30, enhanced_data[field]))
        
        # Form validation
        for form_field in ['home_form', 'away_form']:
            if form_field not in enhanced_data:
                enhanced_data[form_field] = [1, 1, 1, 1, 1, 1]  # Neutral form
            elif len(enhanced_data[form_field]) < 6:
                # Pad with neutral form points
                current_form = enhanced_data[form_field]
                padding = [1] * (6 - len(current_form))
                enhanced_data[form_field] = current_form + padding
        
        return enhanced_data

    def _calculate_realistic_xg(self) -> Tuple[float, float]:
        """DEBUGGED xG calculation with proper scaling"""
        league = self.data.get('league', 'premier_league')
        league_baseline = self.calibrator.league_baselines.get(league, {'avg_goals': 2.7, 'home_advantage': 0.35})
        
        # Get team tiers for calibration
        home_tier = self.calibrator.get_team_tier(self.data['home_team'], league)
        away_tier = self.calibrator.get_team_tier(self.data['away_team'], league)
        
        # Calculate averages per game with better normalization
        home_goals_avg = self.data.get('home_goals', 9) / 6.0
        away_goals_avg = self.data.get('away_goals', 9) / 6.0
        home_conceded_avg = self.data.get('home_conceded', 9) / 6.0
        away_conceded_avg = self.data.get('away_conceded', 9) / 6.0
        
        # Home/away specific adjustments
        home_goals_home_avg = self.data.get('home_goals_home', 5) / 3.0
        away_goals_away_avg = self.data.get('away_goals_away', 4) / 3.0
        
        # DEBUG: Print intermediate values for diagnosis
        print(f"DEBUG - Home attack: {home_goals_avg:.2f}, Away attack: {away_goals_avg:.2f}")
        print(f"DEBUG - Home defense: {home_conceded_avg:.2f}, Away defense: {away_conceded_avg:.2f}")
        
        # CALIBRATED Attack calculation with league adjustment
        home_attack = (home_goals_avg * 0.6) + (home_goals_home_avg * 0.4)
        away_attack = (away_goals_avg * 0.6) + (away_goals_away_avg * 0.4)
        
        # Apply attack factor calibration
        home_attack *= self.calibration['attack_factor']
        away_attack *= self.calibration['attack_factor']
        
        # CALIBRATED Defense adjustment (less aggressive)
        home_xg = home_attack * (1 - (away_conceded_avg / 10) * self.calibration['defense_factor'])
        away_xg = away_attack * (1 - (home_conceded_avg / 10) * self.calibration['defense_factor'])
        
        # Apply league baseline normalization
        league_adjustment = league_baseline['avg_goals'] / 2.7
        home_xg *= league_adjustment
        away_xg *= league_adjustment
        
        # Apply home advantage
        home_advantage = league_baseline['home_advantage'] * self.calibration['home_boost']
        home_xg *= (1 + home_advantage)
        
        # CALIBRATED Form adjustment
        home_form = self.data.get('home_form', [1, 1, 1, 1, 1, 1])
        away_form = self.data.get('away_form', [1, 1, 1, 1, 1, 1])
        
        # Weight recent form more heavily
        home_form_weights = [0.1, 0.1, 0.15, 0.15, 0.2, 0.3]  # Most recent = highest weight
        away_form_weights = [0.1, 0.1, 0.15, 0.15, 0.2, 0.3]
        
        home_form_weighted = sum(home_form[i] * home_form_weights[i] for i in range(6))
        away_form_weighted = sum(away_form[i] * away_form_weights[i] for i in range(6))
        
        # Convert form points to multiplier (1.5 pts/game = 1.0 multiplier)
        home_form_factor = (home_form_weighted / 1.5) ** 0.3  # Reduced impact
        away_form_factor = (away_form_weighted / 1.5) ** 0.3
        
        home_xg *= (1 + (home_form_factor - 1) * self.calibration['form_weight'])
        away_xg *= (1 + (away_form_factor - 1) * self.calibration['form_weight'])
        
        # APPLY TIER-BASED CALIBRATION (CRITICAL FIX)
        tier_modifier = {
            'ELITE': 1.15, 
            'STRONG': 1.05, 
            'MEDIUM': 1.0, 
            'WEAK': 0.85
        }
        
        home_xg *= tier_modifier.get(home_tier, 1.0)
        away_xg *= tier_modifier.get(away_tier, 1.0)
        
        # Ensure realistic bounds
        home_xg = max(self.calibration['min_xg'], min(self.calibration['max_xg'], home_xg))
        away_xg = max(self.calibration['min_xg'], min(self.calibration['max_xg'], away_xg))
        
        print(f"DEBUG - Final xG: Home {home_xg:.2f}, Away {away_xg:.2f}")
        print(f"DEBUG - Team Tiers: {home_tier} vs {away_tier}")
        
        return round(home_xg, 3), round(away_xg, 3)

    def _calculate_accurate_probabilities(self, home_xg: float, away_xg: float) -> Dict[str, float]:
        """FIXED probability calculation with proper normalization"""
        home_win_prob = 0
        draw_prob = 0
        away_win_prob = 0
        
        # EXTENDED Poisson distribution (0-7 goals)
        for i in range(8):  # 0-7 goals for home
            for j in range(8):  # 0-7 goals for away
                prob = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
                if i > j:
                    home_win_prob += prob
                elif i == j:
                    draw_prob += prob
                else:
                    away_win_prob += prob
        
        # CRITICAL: Normalize to 100%
        total = home_win_prob + draw_prob + away_win_prob
        if total > 0:
            home_win_prob /= total
            draw_prob /= total
            away_win_prob /= total
        else:
            # Fallback if all probabilities are zero
            home_win_prob, draw_prob, away_win_prob = 0.33, 0.34, 0.33
        
        # Convert to percentages
        return {
            'home_win': home_win_prob * 100,
            'draw': draw_prob * 100,
            'away_win': away_win_prob * 100
        }

    def _calculate_goals_probabilities(self, home_xg: float, away_xg: float) -> Dict[str, Any]:
        """FIXED goals probabilities calculation"""
        total_xg = home_xg + away_xg
        
        # FIXED Over/Under 2.5 calculation
        over_25_prob = 0
        under_25_prob = 0
        
        for i in range(8):
            for j in range(8):
                prob = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
                if i + j > 2.5:
                    over_25_prob += prob
                else:
                    under_25_prob += prob
        
        # FIXED Both Teams to Score calculation
        btts_yes_prob = 0
        btts_no_prob = 0
        
        for i in range(8):
            for j in range(8):
                prob = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
                if i > 0 and j > 0:
                    btts_yes_prob += prob
                else:
                    btts_no_prob += prob
        
        return {
            'over_under': {
                'over_25': over_25_prob * 100,
                'under_25': under_25_prob * 100
            },
            'both_teams_score': {
                'yes': btts_yes_prob * 100,
                'no': btts_no_prob * 100
            }
        }

    def _calculate_exact_scores(self, home_xg: float, away_xg: float) -> Dict[str, float]:
        """Calculate most likely exact scores"""
        exact_scores = {}
        
        for i in range(5):  # 0-4 goals for home
            for j in range(5):  # 0-4 goals for away
                prob = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
                if prob > 0.005:  # Only include probabilities > 0.5%
                    exact_scores[f"{i}-{j}"] = round(prob * 100, 1)
        
        # Return top 6 most likely scores
        return dict(sorted(exact_scores.items(), key=lambda x: x[1], reverse=True)[:6])

    def _calculate_confidence_score(self, probabilities: Dict[str, float]) -> float:
        """Improved confidence calculation using probability spread"""
        probs = list(probabilities.values())
        max_prob = max(probs)
        
        # Calculate entropy (uncertainty measure)
        entropy = -sum(p/100 * math.log(p/100 + 1e-10) for p in probs if p > 0)
        max_entropy = math.log(3)  # Maximum entropy for 3 outcomes
        
        # Confidence is inverse of normalized entropy
        normalized_entropy = entropy / max_entropy
        confidence = (1 - normalized_entropy) * 100
        
        return round(confidence, 1)

    def generate_predictions(self) -> Dict[str, Any]:
        """Main prediction generation with debugging"""
        print("🔧 GENERATING CALIBRATED PREDICTIONS...")
        
        # Calculate expected goals
        home_xg, away_xg = self._calculate_realistic_xg()
        
        # Get team tiers
        home_tier = self.calibrator.get_team_tier(self.data['home_team'], self.data['league'])
        away_tier = self.calibrator.get_team_tier(self.data['away_team'], self.data['league'])
        
        # Calculate probabilities
        match_outcomes = self._calculate_accurate_probabilities(home_xg, away_xg)
        goals_probs = self._calculate_goals_probabilities(home_xg, away_xg)
        exact_scores = self._calculate_exact_scores(home_xg, away_xg)
        
        # Determine match context
        match_context = self._determine_match_context(home_xg, away_xg)
        
        # Calculate improved confidence score
        confidence_score = self._calculate_confidence_score(match_outcomes)
        
        print("✅ PREDICTIONS GENERATED SUCCESSFULLY")
        
        return {
            'match': f"{self.data['home_team']} vs {self.data['away_team']}",
            'league': self.data.get('league', 'premier_league'),
            'expected_goals': {'home': home_xg, 'away': away_xg},
            'team_tiers': {'home': home_tier, 'away': away_tier},
            'match_context': match_context,
            'confidence_score': confidence_score,
            'probabilities': {
                'match_outcomes': {k: round(v, 1) for k, v in match_outcomes.items()},
                'both_teams_score': {k: round(v, 1) for k, v in goals_probs['both_teams_score'].items()},
                'over_under': {k: round(v, 1) for k, v in goals_probs['over_under'].items()},
                'exact_scores': exact_scores
            },
            'summary': self._generate_summary(home_xg, away_xg, match_context),
            'debug_info': {
                'home_goals_input': self.data.get('home_goals'),
                'away_goals_input': self.data.get('away_goals'),
                'home_form_input': self.data.get('home_form'),
                'away_form_input': self.data.get('away_form')
            }
        }

    def _determine_match_context(self, home_xg: float, away_xg: float) -> str:
        """Determine the match context based on xG values"""
        total_xg = home_xg + away_xg
        xg_difference = home_xg - away_xg
        
        if total_xg > 3.5:
            return "offensive_showdown"
        elif total_xg < 2.0:
            return "defensive_battle"
        elif xg_difference > 1.0:
            return "home_dominance"
        elif xg_difference < -1.0:
            return "away_counter"
        elif abs(xg_difference) < 0.3:
            return "balanced"
        else:
            return "competitive"

    def _generate_summary(self, home_xg: float, away_xg: float, context: str) -> str:
        """Generate match summary based on predictions"""
        home_team = self.data['home_team']
        away_team = self.data['away_team']
        total_xg = home_xg + away_xg
        
        if context == "offensive_showdown":
            return f"🔥 High-scoring thriller expected! {home_team} and {away_team} both look dangerous going forward. Expect goals at both ends."
        elif context == "defensive_battle":
            return f"🛡️ Tight, tactical affair anticipated. Both {home_team} and {away_team} are well-organized defensively. Few clear chances expected."
        elif context == "home_dominance":
            return f"🏠 {home_team} should control this game. Their superior attacking quality and home advantage should see them create more chances against {away_team}."
        elif context == "away_counter":
            return f"✈️ {away_team} look dangerous on the break. They could exploit {home_team}'s defensive vulnerabilities with quick counter-attacks."
        elif context == "balanced":
            return f"⚖️ Evenly matched contest between {home_team} and {away_team}. Both teams have similar quality and could cancel each other out."
        else:
            return f"⚽ Competitive match expected between {home_team} and {away_team}. Small margins likely to decide this encounter."

# TEST WITH REALISTIC DATA
def test_debugged_predictor():
    """Test the debugged predictor with realistic scenarios"""
    
    # Scenario 1: Strong home team vs weak away team
    match_data_1 = {
        'home_team': 'Man City',
        'away_team': 'Sheffield United', 
        'league': 'premier_league',
        'home_goals': 15,  # High scoring
        'away_goals': 6,   # Low scoring
        'home_conceded': 8,
        'away_conceded': 18, # Leaky defense
        'home_goals_home': 8,
        'away_goals_away': 2,
        'home_form': [3, 3, 3, 3, 3, 3],  # Perfect form
        'away_form': [0, 0, 1, 0, 0, 0]   # Poor form
    }
    
    print("TEST 1: Strong Home vs Weak Away")
    print("=" * 50)
    predictor = DebuggedFootballPredictor(match_data_1)
    results = predictor.generate_predictions()
    
    display_predictions(results)
    
    # Scenario 2: Evenly matched teams
    match_data_2 = {
        'home_team': 'Brighton',
        'away_team': 'West Ham', 
        'league': 'premier_league',
        'home_goals': 10,
        'away_goals': 11,
        'home_conceded': 12,
        'away_conceded': 13,
        'home_goals_home': 6,
        'away_goals_away': 5,
        'home_form': [1, 3, 0, 3, 1, 3],
        'away_form': [3, 0, 1, 3, 0, 3]
    }
    
    print("\nTEST 2: Evenly Matched Teams")
    print("=" * 50)
    predictor2 = DebuggedFootballPredictor(match_data_2)
    results2 = predictor2.generate_predictions()
    
    display_predictions(results2)

def display_predictions(results: Dict[str, Any]):
    """Display prediction results clearly"""
    print(f"📊 Match: {results['match']}")
    print(f"🏆 Tiers: {results['team_tiers']['home']} vs {results['team_tiers']['away']}")
    print(f"🎯 Expected Goals: Home {results['expected_goals']['home']} - Away {results['expected_goals']['away']}")
    print(f"📈 Context: {results['match_context'].replace('_', ' ').title()}")
    print(f"🎲 Confidence: {results['confidence_score']}%")
    print()
    
    outcomes = results['probabilities']['match_outcomes']
    print("MATCH OUTCOMES:")
    print(f"  Home Win: {outcomes['home_win']}%")
    print(f"  Draw: {outcomes['draw']}%") 
    print(f"  Away Win: {outcomes['away_win']}%")
    print()
    
    btts = results['probabilities']['both_teams_score']
    print("BOTH TEAMS TO SCORE:")
    print(f"  Yes: {btts['yes']}% | No: {btts['no']}%")
    
    over_under = results['probabilities']['over_under']
    print("OVER/UNDER 2.5 GOALS:")
    print(f"  Over: {over_under['over_25']}% | Under: {over_under['under_25']}%")
    print()
    
    print("MOST LIKELY SCORES:")
    for score, prob in results['probabilities']['exact_scores'].items():
        print(f"  {score}: {prob}%")
    print()
    
    print("SUMMARY:")
    print(f"  {results['summary']}")
    print("-" * 50)

if __name__ == "__main__":
    test_debugged_predictor()
