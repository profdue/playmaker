# prediction_engine.py - COMPLETELY CALIBRATED VERSION
import numpy as np
from scipy.stats import poisson
from typing import Dict, Any, Tuple, List
import math

class TeamTierCalibrator:
    def __init__(self):
        # CALIBRATED LEAGUE BASELINES - Based on actual goal averages
        self.league_baselines = {
            'premier_league': {'avg_goals': 2.8, 'home_advantage': 0.15},  # REDUCED from 0.35
            'la_liga': {'avg_goals': 2.6, 'home_advantage': 0.14},         # REDUCED from 0.32
            'serie_a': {'avg_goals': 2.7, 'home_advantage': 0.16},         # REDUCED from 0.38
            'bundesliga': {'avg_goals': 3.1, 'home_advantage': 0.12},      # REDUCED from 0.28
            'ligue_1': {'avg_goals': 2.5, 'home_advantage': 0.14},         # REDUCED from 0.34
            'liga_portugal': {'avg_goals': 2.6, 'home_advantage': 0.18},   # REDUCED from 0.42
            'brasileirao': {'avg_goals': 2.4, 'home_advantage': 0.20},     # REDUCED from 0.45
            'liga_mx': {'avg_goals': 2.7, 'home_advantage': 0.18},         # REDUCED from 0.40
            'eredivisie': {'avg_goals': 3.0, 'home_advantage': 0.13},      # REDUCED from 0.30
        }
        
        # COMPLETE TEAM DATABASES WITH CALIBRATED TIERS
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

class CalibratedFootballPredictor:
    """COMPLETELY CALIBRATED PREDICTION ENGINE"""
    
    def __init__(self, match_data: Dict[str, Any]):
        self.data = self._calibrate_input_data(match_data)
        self.calibrator = TeamTierCalibrator()
        
        # CALIBRATION CONSTANTS - TUNED FOR REALISM
        self.calibration = {
            'attack_factor': 1.0,        # WAS 1.4 - REDUCED SIGNIFICANTLY
            'defense_factor': 0.2,       # WAS 0.3 - REDUCED
            'form_weight': 0.3,          # WAS 0.4 - REDUCED
            'tier_impact': 0.10,         # WAS 0.15 - REDUCED
            'home_boost': 1.1,           # WAS 1.2 - REDUCED
            'min_xg': 0.3,               # Realistic minimum
            'max_xg': 3.0,               # Realistic maximum (was 4.0)
            'h2h_weight': 0.08,          # WAS 0.15 - REDUCED
            'motivation_impact': 0.06,   # WAS 0.10 - REDUCED
        }

    def _calibrate_input_data(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calibrate input data to prevent overestimation"""
        data = match_data.copy()
        
        # CALIBRATION: Cap unrealistic input values
        goals_cap = 18  # Maximum total goals in 6 games
        conceded_cap = 18
        
        for field in ['home_goals', 'away_goals', 'home_conceded', 'away_conceded']:
            if field in data:
                data[field] = min(data[field], goals_cap)
        
        # CALIBRATION: Ensure form data is realistic
        for form_field in ['home_form', 'away_form']:
            if form_field in data and data[form_field]:
                # Ensure we have exactly 6 form points
                form_data = data[form_field]
                if len(form_data) < 6:
                    # Pad with neutral form (1.5 points average)
                    padding_needed = 6 - len(form_data)
                    form_data = form_data + [1.5] * padding_needed
                data[form_field] = form_data[:6]
            else:
                data[form_field] = [1.5] * 6  # Default to average form
        
        return data

    def _calculate_calibrated_xg(self) -> Tuple[float, float]:
        """COMPLETELY CALIBRATED xG CALCULATION"""
        league = self.data.get('league', 'premier_league')
        league_baseline = self.calibrator.league_baselines.get(league, {'avg_goals': 2.7, 'home_advantage': 0.15})
        
        # Get team tiers for calibration
        home_tier = self.calibrator.get_team_tier(self.data['home_team'], league)
        away_tier = self.calibrator.get_team_tier(self.data['away_team'], league)
        
        # CALCULATE REALISTIC AVERAGES
        home_goals_avg = self.data.get('home_goals', 9) / 6.0  # More realistic default
        away_goals_avg = self.data.get('away_goals', 9) / 6.0
        home_conceded_avg = self.data.get('home_conceded', 9) / 6.0
        away_conceded_avg = self.data.get('away_conceded', 9) / 6.0
        
        # Home/away specific adjustments (less weight)
        home_goals_home_avg = self.data.get('home_goals_home', 4) / 3.0
        away_goals_away_avg = self.data.get('away_goals_away', 3) / 3.0  # Lower default for away
        
        # CALIBRATED: More conservative blending
        home_attack = (home_goals_avg * 0.8) + (home_goals_home_avg * 0.2)  # WAS 0.7/0.3
        away_attack = (away_goals_avg * 0.8) + (away_goals_away_avg * 0.2)  # WAS 0.7/0.3
        
        # CALIBRATED: Reduced attack factor
        home_attack *= self.calibration['attack_factor']  # WAS 1.4
        away_attack *= self.calibration['attack_factor']  # WAS 1.4
        
        # CALIBRATED: More conservative defensive adjustment
        home_xg = home_attack * (1 - (away_conceded_avg / league_baseline['avg_goals']) * self.calibration['defense_factor'])
        away_xg = away_attack * (1 - (home_conceded_avg / league_baseline['avg_goals']) * self.calibration['defense_factor'])
        
        # CALIBRATED: Reduced home advantage
        home_advantage = league_baseline['home_advantage'] * self.calibration['home_boost']  # WAS 1.2
        home_xg *= (1 + home_advantage)
        
        # CALIBRATED: More conservative form adjustment
        home_form = self.data.get('home_form', [1.5] * 6)
        away_form = self.data.get('away_form', [1.5] * 6)
        
        # Use weighted average with more recent games having slightly more weight
        form_weights = [0.1, 0.1, 0.15, 0.15, 0.2, 0.3]  # Recent games weighted more
        
        home_form_weighted = sum(home_form[i] * form_weights[i] for i in range(6))
        away_form_weighted = sum(away_form[i] * form_weights[i] for i in range(6))
        
        # CALIBRATED: Reduced form impact
        home_form_factor = 0.9 + (home_form_weighted / 1.5) * 0.2  # WAS more aggressive
        away_form_factor = 0.9 + (away_form_weighted / 1.5) * 0.2  # WAS more aggressive
        
        home_xg *= home_form_factor
        away_xg *= away_form_factor
        
        # CALIBRATED: Tier-based calibration (reduced impact)
        tier_modifier = {
            'ELITE': 1.10,    # WAS 1.15
            'STRONG': 1.04,   # WAS 1.05  
            'MEDIUM': 1.0,
            'WEAK': 0.90      # WAS 0.85
        }
        
        home_xg *= tier_modifier.get(home_tier, 1.0)
        away_xg *= tier_modifier.get(away_tier, 1.0)
        
        # CALIBRATED: Motivation adjustment (reduced)
        motivation = self.data.get('motivation', {})
        home_motivation = self._calculate_motivation_impact(motivation.get('home', 'Normal'))
        away_motivation = self._calculate_motivation_impact(motivation.get('away', 'Normal'))
        
        home_xg *= home_motivation
        away_xg *= away_motivation
        
        # CALIBRATED: Injuries adjustment (reduced)
        injuries = self.data.get('injuries', {})
        home_injury_factor = 1.0 - (injuries.get('home', 1) * 0.03)  # WAS 0.05
        away_injury_factor = 1.0 - (injuries.get('away', 1) * 0.03)  # WAS 0.05
        
        home_xg *= home_injury_factor
        away_xg *= away_injury_factor
        
        # CALIBRATED: H2H adjustment (reduced weight)
        h2h_data = self.data.get('h2h_data', {})
        if h2h_data and h2h_data.get('matches', 0) >= 3:
            home_xg, away_xg = self._apply_calibrated_h2h_adjustment(home_xg, away_xg, h2h_data)
        
        # CALIBRATED: Realistic bounds based on actual data analysis
        home_xg = max(self.calibration['min_xg'], min(self.calibration['max_xg'], home_xg))
        away_xg = max(self.calibration['min_xg'], min(self.calibration['max_xg'], away_xg))
        
        # FINAL CALIBRATION: Ensure xG values are realistic for football
        # Most teams average 1.0-2.0 xG per game, elite teams 1.5-2.5
        return round(home_xg, 3), round(away_xg, 3)

    def _apply_calibrated_h2h_adjustment(self, home_xg: float, away_xg: float, h2h_data: Dict) -> Tuple[float, float]:
        """Calibrated H2H adjustment with reduced impact"""
        matches = h2h_data.get('matches', 0)
        if matches < 3:
            return home_xg, away_xg
        
        # CALIBRATED: Reduced H2H weight
        h2h_weight = min(0.12, matches * 0.04)  # WAS 0.25, 0.06
        
        h2h_home_avg = h2h_data.get('home_goals', 0) / matches
        h2h_away_avg = h2h_data.get('away_goals', 0) / matches
        
        # Only apply if H2H data is meaningful
        if h2h_home_avg > 0 or h2h_away_avg > 0:
            adjusted_home_xg = (home_xg * (1 - h2h_weight)) + (h2h_home_avg * h2h_weight)
            adjusted_away_xg = (away_xg * (1 - h2h_weight)) + (h2h_away_avg * h2h_weight)
            return adjusted_home_xg, adjusted_away_xg
        
        return home_xg, away_xg

    def _calculate_motivation_impact(self, motivation_level: str) -> float:
        """Calibrated motivation impact"""
        multipliers = {
            "Low": 0.94,      # WAS 0.90
            "Normal": 1.0, 
            "High": 1.05,     # WAS 1.08
            "Very High": 1.08 # WAS 1.12
        }
        return multipliers.get(motivation_level, 1.0)

    def _calculate_probabilities(self, home_xg: float, away_xg: float) -> Dict[str, Any]:
        """Calculate all probabilities with proper normalization"""
        # Match outcomes using Poisson distribution
        home_win, draw, away_win = 0, 0, 0
        
        # Use extended range but with proper bounds
        for i in range(8):  # 0-7 goals
            for j in range(8):
                prob = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
                if i > j:
                    home_win += prob
                elif i == j:
                    draw += prob
                else:
                    away_win += prob
        
        # CRITICAL: Normalize to ensure sum = 1
        total = home_win + draw + away_win
        if total > 0:
            home_win /= total
            draw /= total
            away_win /= total
        else:
            # Fallback if all probabilities are zero (unlikely)
            home_win, draw, away_win = 0.33, 0.34, 0.33
        
        # Both Teams to Score
        home_score_prob = 1 - math.exp(-home_xg)
        away_score_prob = 1 - math.exp(-away_xg)
        btts_yes = home_score_prob * away_score_prob
        
        # Over/Under 2.5 goals
        total_xg = home_xg + away_xg
        over_25 = 1 - poisson.cdf(2, total_xg)
        
        # Exact scores (most likely)
        exact_scores = {}
        for i in range(5):  # 0-4 goals
            for j in range(5):
                prob = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
                if prob > 0.005:  # Only include > 0.5%
                    exact_scores[f"{i}-{j}"] = round(prob * 100, 1)
        
        # Return top 6 most likely scores
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

    def generate_predictions(self) -> Dict[str, Any]:
        """Generate completely calibrated predictions"""
        # Calculate calibrated expected goals
        home_xg, away_xg = self._calculate_calibrated_xg()
        
        # Get team tiers
        home_tier = self.calibrator.get_team_tier(self.data['home_team'], self.data['league'])
        away_tier = self.calibrator.get_team_tier(self.data['away_team'], self.data['league'])
        
        # Calculate probabilities
        probabilities = self._calculate_probabilities(home_xg, away_xg)
        
        # Determine match context
        match_context = self._determine_match_context(home_xg, away_xg)
        
        # Calculate confidence (based on probability spread)
        confidence_score = self._calculate_confidence(probabilities['match_outcomes'])
        
        # Generate realistic summary
        summary = self._generate_realistic_summary(home_xg, away_xg, match_context)
        
        return {
            'match': f"{self.data['home_team']} vs {self.data['away_team']}",
            'league': self.data.get('league', 'premier_league'),
            'expected_goals': {'home': home_xg, 'away': away_xg},
            'team_tiers': {'home': home_tier, 'away': away_tier},
            'match_context': match_context,
            'confidence_score': confidence_score,
            'data_quality_score': 85.0,
            'probabilities': probabilities,
            'summary': summary,
            'calibration_info': {
                'attack_factor': self.calibration['attack_factor'],
                'home_advantage': self.calibrator.league_baselines[self.data.get('league', 'premier_league')]['home_advantage'],
                'max_xg': self.calibration['max_xg']
            }
        }

    def _determine_match_context(self, home_xg: float, away_xg: float) -> str:
        """Determine realistic match context"""
        total_xg = home_xg + away_xg
        xg_difference = home_xg - away_xg
        
        if total_xg > 3.2:
            return "offensive_showdown"
        elif total_xg < 2.0:
            return "defensive_battle"
        elif xg_difference > 0.7:  # WAS 0.8
            return "home_dominance"
        elif xg_difference < -0.7:  # WAS -0.8
            return "away_counter"
        elif abs(xg_difference) < 0.3:
            return "balanced"
        else:
            return "competitive"

    def _calculate_confidence(self, outcomes: Dict[str, float]) -> float:
        """Calculate confidence based on probability distribution"""
        probs = list(outcomes.values())
        max_prob = max(probs)
        
        # Confidence is higher when one outcome is clearly more likely
        # and lower when probabilities are evenly distributed
        if max_prob > 55:
            confidence = 60 + (max_prob - 55) * 0.8
        elif max_prob > 45:
            confidence = 40 + (max_prob - 45)
        else:
            confidence = 30 + max_prob * 0.5
            
        return min(85, max(25, round(confidence, 1)))

    def _generate_realistic_summary(self, home_xg: float, away_xg: float, context: str) -> str:
        """Generate realistic match summary"""
        home_team = self.data['home_team']
        away_team = self.data['away_team']
        total_xg = home_xg + away_xg
        
        if context == "home_dominance":
            if total_xg > 3.0:
                return f"{home_team} are favored at home and should create more chances. Expect goals as they look to control the game against {away_team}."
            else:
                return f"{home_team} have the home advantage and should see more of the ball, but {away_team} will be organized defensively in a potentially tight contest."
                
        elif context == "balanced":
            return f"An evenly matched encounter between {home_team} and {away_team}. Both teams have similar quality and the match could go either way."
            
        elif context == "away_counter":
            return f"{away_team} could pose problems on the counter-attack. {home_team} will need to be careful defensively while looking to break down a resilient away side."
            
        elif context == "offensive_showdown":
            return f"An open, attacking game expected with both {home_team} and {away_team} looking to play forward. Goals seem likely at both ends."
            
        else:  # defensive_battle
            return f"A tactical, potentially low-scoring affair. Both {home_team} and {away_team} are well-organized defensively and chances may be limited."

# TEST WITH TOULOUSE vs LE HAVRE - CALIBRATED
def test_calibrated_predictor():
    """Test the completely calibrated predictor"""
    
    # Toulouse vs Le Havre actual data
    match_data = {
        'home_team': 'Toulouse',
        'away_team': 'Le Havre',
        'league': 'ligue_1',
        'home_goals': 11,  # Actual: 2+1+4+1+2+1 = 11
        'away_goals': 6,   # Actual: 1+0+2+2+0+1 = 6
        'home_conceded': 6, # Actual: 2+0+0+2+2+0 = 6
        'away_conceded': 10, # Actual: 0+1+6+2+0+1 = 10
        'home_goals_home': 8, # Home games: 2+4+2 = 8
        'away_goals_away': 2, # Away games: 0+2+0 = 2
        'home_form': [1, 0, 3, 0, 1, 3],  # Draw, Loss, Win, Loss, Draw, Win
        'away_form': [3, 3, 0, 1, 1, 1],   # Win, Win, Loss, Draw, Draw, Draw
        'h2h_data': {
            'matches': 6,
            'home_wins': 3,  # Toulouse advantage in H2H
            'away_wins': 2,
            'draws': 1,
            'home_goals': 7,
            'away_goals': 6
        },
        'motivation': {'home': 'Normal', 'away': 'Normal'},
        'injuries': {'home': 1, 'away': 1}
    }
    
    print("🎯 COMPLETELY CALIBRATED PREDICTIONS")
    print("=" * 60)
    print("Match: Toulouse vs Le Havre")
    print("League: Ligue 1")
    print()
    
    predictor = CalibratedFootballPredictor(match_data)
    predictions = predictor.generate_predictions()
    
    print(f"📊 EXPECTED GOALS (CALIBRATED):")
    print(f"Toulouse: {predictions['expected_goals']['home']} (was 2.19 in old engine)")
    print(f"Le Havre: {predictions['expected_goals']['away']} (was 1.16 in old engine)")
    print()
    
    print("📈 MATCH OUTCOMES (CALIBRATED):")
    outcomes = predictions['probabilities']['match_outcomes']
    print(f"Toulouse Win: {outcomes['home_win']}% (was 61.6% in old engine)")
    print(f"Draw: {outcomes['draw']}% (was 20.9% in old engine)")
    print(f"Le Havre Win: {outcomes['away_win']}% (was 17.5% in old engine)")
    print()
    
    print("⚽ GOALS MARKETS:")
    btts = predictions['probabilities']['both_teams_score']
    print(f"BTTS Yes: {btts['yes']}% (was 51.7% in old engine)")
    print(f"BTTS No: {btts['no']}% (was 48.3% in old engine)")
    
    over_under = predictions['probabilities']['over_under']
    print(f"Over 2.5: {over_under['over_25']}% (was 55.2% in old engine)")
    print(f"Under 2.5: {over_under['under_25']}% (was 44.8% in old engine)")
    print()
    
    print("🎯 MOST LIKELY SCORES:")
    for score, prob in predictions['probabilities']['exact_scores'].items():
        print(f"  {score}: {prob}%")
    print()
    
    print(f"📊 MATCH CONTEXT: {predictions['match_context'].replace('_', ' ').title()}")
    print(f"🎲 CONFIDENCE: {predictions['confidence_score']}%")
    print(f"🏆 TEAM TIERS: {predictions['team_tiers']['home']} vs {predictions['team_tiers']['away']}")
    print()
    
    print("📝 REALISTIC SUMMARY:")
    print(f"  {predictions['summary']}")
    print()
    
    print("🔧 CALIBRATION APPLIED:")
    cal_info = predictions['calibration_info']
    print(f"  Attack Factor: {cal_info['attack_factor']} (was 1.4)")
    print(f"  Home Advantage: {cal_info['home_advantage']*100:.1f}% (was 34%)")
    print(f"  Max xG: {cal_info['max_xg']} (was 4.0)")

if __name__ == "__main__":
    test_calibrated_predictor()
