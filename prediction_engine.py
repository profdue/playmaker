import numpy as np
from scipy.stats import poisson, skellam
from typing import Dict, Any, Tuple
import math

class AdvancedPredictionEngine:
    """Advanced Football Prediction Engine with Balanced Calculations"""
    
    def __init__(self, match_data: Dict[str, Any]):
        self.data = match_data
        self.league_contexts = self._initialize_league_contexts()
    
    def _initialize_league_contexts(self) -> Dict[str, Dict]:
        """Initialize league-specific parameters for better calibration"""
        return {
            'premier_league': {'avg_goals': 2.8, 'avg_corners': 10.5, 'home_advantage': 1.1},
            'la_liga': {'avg_goals': 2.6, 'avg_corners': 9.8, 'home_advantage': 1.15},
            'serie_a': {'avg_goals': 2.7, 'avg_corners': 10.2, 'home_advantage': 1.08},
            'bundesliga': {'avg_goals': 3.1, 'avg_corners': 9.5, 'home_advantage': 1.12},
            'ligue_1': {'avg_goals': 2.5, 'avg_corners': 9.2, 'home_advantage': 1.1},
            'default': {'avg_goals': 2.7, 'avg_corners': 10.0, 'home_advantage': 1.1}
        }
    
    def generate_advanced_predictions(self) -> Dict[str, Any]:
        """Generate precise predictions with balanced calculations"""
        
        # Extract and validate data
        home_team = self.data.get('home_team', 'Home Team')
        away_team = self.data.get('away_team', 'Away Team')
        league = self.data.get('league', 'default')
        
        # Get advanced metrics
        home_goals = self.data.get('home_goals', 0)
        away_goals = self.data.get('away_goals', 0)
        home_conceded = self.data.get('home_conceded', 0)
        away_conceded = self.data.get('away_conceded', 0)
        
        # Advanced inputs
        home_goals_home = self.data.get('home_goals_home', home_goals)
        away_goals_away = self.data.get('away_goals_away', away_goals)
        home_form = self.data.get('home_form', [])  # Last 5 results as points: [3, 1, 0, 3, 3]
        away_form = self.data.get('away_form', [])
        h2h_data = self.data.get('h2h_data', {})
        injuries = self.data.get('injuries', {'home': 0, 'away': 0})
        motivation = self.data.get('motivation', {'home': 1.0, 'away': 1.0})
        
        # Calculate precise expected goals
        home_xg, away_xg = self._calculate_precise_xg(
            home_goals, away_goals, home_conceded, away_conceded,
            home_goals_home, away_goals_away, home_form, away_form,
            injuries, motivation, league
        )
        
        # Apply H2H adjustments if available
        if h2h_data:
            home_xg, away_xg = self._apply_h2h_adjustment(home_xg, away_xg, h2h_data)
        
        # Calculate all probabilities with high precision
        probabilities = self._calculate_all_probabilities(home_xg, away_xg, league)
        
        # Generate corner predictions
        corner_predictions = self._calculate_corner_predictions(home_xg, away_xg, league)
        
        # Generate timing predictions
        timing_predictions = self._calculate_timing_predictions(home_xg, away_xg)
        
        # Generate betting recommendations
        betting_recommendations = self._generate_betting_recommendations(probabilities)
        
        # Calculate overall confidence
        confidence_score = self._calculate_advanced_confidence(
            home_goals, away_goals, home_form, away_form, h2h_data
        )
        
        return {
            'match': f"{home_team} vs {away_team}",
            'expected_goals': {'home': round(home_xg, 2), 'away': round(away_xg, 2)},
            'probabilities': probabilities,
            'corner_predictions': corner_predictions,
            'timing_predictions': timing_predictions,
            'betting_recommendations': betting_recommendations,
            'summary': self._generate_detailed_summary(home_team, away_team, probabilities, home_xg, away_xg),
            'confidence_score': confidence_score,
            'risk_assessment': self._assess_prediction_risk(probabilities, confidence_score)
        }
    
    def _calculate_precise_xg(self, home_goals: int, away_goals: int, home_conceded: int, 
                            away_conceded: int, home_goals_home: int, away_goals_away: int,
                            home_form: list, away_form: list, injuries: Dict, 
                            motivation: Dict, league: str) -> Tuple[float, float]:
        """Calculate precise expected goals with multiple factors"""
        
        league_params = self.league_contexts.get(league, self.league_contexts['default'])
        
        # Base attack/defense strength (per game)
        home_attack = max(0.3, home_goals / 6.0)
        away_attack = max(0.2, away_goals / 6.0)
        home_defense = max(0.3, home_conceded / 6.0)
        away_defense = max(0.4, away_conceded / 6.0)
        
        # Home/away specific adjustments
        home_attack_home = max(0.3, home_goals_home / 3.0) if home_goals_home > 0 else home_attack
        away_attack_away = max(0.2, away_goals_away / 3.0) if away_goals_away > 0 else away_attack
        
        # Form adjustments (weighted average of recent form)
        home_form_factor = self._calculate_form_factor(home_form)
        away_form_factor = self._calculate_form_factor(away_form)
        
        # Injury adjustments
        home_injury_factor = max(0.7, 1.0 - (injuries.get('home', 0) * 0.1))
        away_injury_factor = max(0.7, 1.0 - (injuries.get('away', 0) * 0.1))
        
        # Motivation adjustments
        home_motivation = motivation.get('home', 1.0)
        away_motivation = motivation.get('away', 1.0)
        
        # Calculate base xG with league normalization
        base_home_xg = (home_attack_home * away_defense * league_params['home_advantage'] * 
                       home_form_factor * home_injury_factor * home_motivation)
        base_away_xg = (away_attack_away * home_defense * 
                       away_form_factor * away_injury_factor * away_motivation)
        
        # Apply regression to league mean to reduce extremes
        home_xg = (base_home_xg + league_params['avg_goals'] * 0.3) / 1.3
        away_xg = (base_away_xg + league_params['avg_goals'] * 0.3) / 1.3
        
        # Ensure reasonable bounds
        home_xg = max(0.1, min(4.0, home_xg))
        away_xg = max(0.1, min(3.5, away_xg))
        
        return round(home_xg, 3), round(away_xg, 3)
    
    def _calculate_form_factor(self, form: list) -> float:
        """Calculate form factor from recent results"""
        if not form or len(form) == 0:
            return 1.0
        
        # Recent matches weighted more heavily
        weights = [1.2, 1.1, 1.0, 0.9, 0.8][:len(form)]
        total_points = sum(score * weight for score, weight in zip(form, weights))
        max_possible = sum(3 * weight for weight in weights)
        
        form_ratio = total_points / max_possible if max_possible > 0 else 0.5
        
        # Convert to multiplier (0.8 to 1.2 range)
        return 0.8 + (form_ratio * 0.4)
    
    def _apply_h2h_adjustment(self, home_xg: float, away_xg: float, h2h_data: Dict) -> Tuple[float, float]:
        """Apply head-to-head historical adjustments"""
        matches = h2h_data.get('matches', 0)
        home_wins = h2h_data.get('home_wins', 0)
        away_wins = h2h_data.get('away_wins', 0)
        draws = h2h_data.get('draws', 0)
        home_goals = h2h_data.get('home_goals', 0)
        away_goals = h2h_data.get('away_goals', 0)
        
        if matches < 3:  # Not enough H2H data
            return home_xg, away_xg
        
        # Calculate H2H dominance factor
        home_dominance = (home_wins + draws * 0.5) / matches
        h2h_home_avg = home_goals / matches if matches > 0 else home_xg
        h2h_away_avg = away_goals / matches if matches > 0 else away_xg
        
        # Blend current form with H2H history (70/30 weight)
        adjusted_home_xg = (home_xg * 0.7) + (h2h_home_avg * 0.3)
        adjusted_away_xg = (away_xg * 0.7) + (h2h_away_avg * 0.3)
        
        # Apply dominance adjustment
        if home_dominance > 0.6:  # Strong home dominance
            adjusted_home_xg *= 1.1
            adjusted_away_xg *= 0.9
        elif home_dominance < 0.4:  # Weak home performance
            adjusted_home_xg *= 0.9
            adjusted_away_xg *= 1.1
        
        return adjusted_home_xg, adjusted_away_xg
    
    def _calculate_all_probabilities(self, home_xg: float, away_xg: float, league: str) -> Dict[str, Any]:
        """Calculate all match probabilities with high precision"""
        
        # Match outcome probabilities using Poisson distribution
        home_win_prob, draw_prob, away_win_prob = self._calculate_match_outcomes(home_xg, away_xg)
        
        # Goal timing probabilities
        first_half_prob = self._calculate_goal_timing_probability(home_xg, away_xg, 'first_half')
        second_half_prob = self._calculate_goal_timing_probability(home_xg, away_xg, 'second_half')
        
        # Both teams to score probability
        btts_prob = self._calculate_btts_probability(home_xg, away_xg)
        
        # Over/under probabilities
        over_15_prob = 1 - (poisson.pmf(0, home_xg) * poisson.pmf(0, away_xg) + 
                           poisson.pmf(1, home_xg) * poisson.pmf(0, away_xg) + 
                           poisson.pmf(0, home_xg) * poisson.pmf(1, away_xg))
        
        over_25_prob = 1 - sum(poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg) 
                              for i in range(3) for j in range(3 - i))
        
        over_35_prob = 1 - sum(poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg) 
                              for i in range(4) for j in range(4 - i))
        
        # Exact score probabilities for most likely outcomes
        exact_scores = {}
        for i in range(4):  # Home goals
            for j in range(4):  # Away goals
                prob = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
                if prob > 0.01:  # Only include probabilities > 1%
                    exact_scores[f"{i}-{j}"] = round(prob * 100, 1)
        
        return {
            'match_outcomes': {
                'home_win': round(home_win_prob * 100, 1),
                'draw': round(draw_prob * 100, 1),
                'away_win': round(away_win_prob * 100, 1)
            },
            'goal_timing': {
                'first_half': round(first_half_prob * 100, 1),
                'second_half': round(second_half_prob * 100, 1)
            },
            'both_teams_score': round(btts_prob * 100, 1),
            'over_under': {
                'over_1.5': round(over_15_prob * 100, 1),
                'over_2.5': round(over_25_prob * 100, 1),
                'over_3.5': round(over_35_prob * 100, 1)
            },
            'exact_scores': dict(sorted(exact_scores.items(), key=lambda x: x[1], reverse=True)[:6])
        }
    
    def _calculate_match_outcomes(self, home_xg: float, away_xg: float) -> Tuple[float, float, float]:
        """Calculate precise match outcome probabilities using Poisson distribution"""
        max_goals = 8  # Increased for better accuracy
        
        home_probs = [poisson.pmf(i, home_xg) for i in range(max_goals)]
        away_probs = [poisson.pmf(i, away_xg) for i in range(max_goals)]
        
        home_win = 0.0
        draw = 0.0
        away_win = 0.0
        
        for i in range(max_goals):
            for j in range(max_goals):
                prob = home_probs[i] * away_probs[j]
                if i > j:
                    home_win += prob
                elif i == j:
                    draw += prob
                else:
                    away_win += prob
        
        # Normalize to account for probabilities beyond max_goals
        total = home_win + draw + away_win
        return home_win/total, draw/total, away_win/total
    
    def _calculate_goal_timing_probability(self, home_xg: float, away_xg: float, half: str) -> float:
        """Calculate probability of goals in specific half"""
        if half == 'first_half':
            # Goals are typically 45% in first half, 55% in second half
            first_half_xg_home = home_xg * 0.45
            first_half_xg_away = away_xg * 0.45
            prob_no_goals = poisson.pmf(0, first_half_xg_home) * poisson.pmf(0, first_half_xg_away)
            return 1 - prob_no_goals
        else:  # second_half
            second_half_xg_home = home_xg * 0.55
            second_half_xg_away = away_xg * 0.55
            prob_no_goals = poisson.pmf(0, second_half_xg_home) * poisson.pmf(0, second_half_xg_away)
            return 1 - prob_no_goals
    
    def _calculate_btts_probability(self, home_xg: float, away_xg: float) -> float:
        """Calculate Both Teams to Score probability"""
        prob_home_scores = 1 - poisson.pmf(0, home_xg)
        prob_away_scores = 1 - poisson.pmf(0, away_xg)
        return prob_home_scores * prob_away_scores
    
    def _calculate_corner_predictions(self, home_xg: float, away_xg: float, league: str) -> Dict[str, Any]:
        """Calculate realistic corner predictions"""
        league_params = self.league_contexts.get(league, self.league_contexts['default'])
        
        # Corner prediction based on expected goals and attacking play
        base_corners = league_params['avg_corners']
        attacking_bonus = (home_xg + away_xg - league_params['avg_goals']) * 1.5
        
        total_corners = base_corners + attacking_bonus
        total_corners = max(4, min(16, total_corners))  # Realistic bounds
        
        # Home teams typically get more corners
        home_corners = total_corners * 0.55
        away_corners = total_corners * 0.45
        
        return {
            'total': f"{int(total_corners)}-{int(total_corners + 1)}",
            'home': f"{int(home_corners)}-{int(home_corners + 0.5)}",
            'away': f"{int(away_corners)}-{int(away_corners + 0.5)}",
            'over_9.5': 'YES' if total_corners > 9.5 else 'NO'
        }
    
    def _calculate_timing_predictions(self, home_xg: float, away_xg: float) -> Dict[str, Any]:
        """Calculate goal timing predictions"""
        total_xg = home_xg + away_xg
        
        if total_xg < 1.5:
            first_goal = "35+ minutes"
            late_goals = "UNLIKELY"
        elif total_xg < 2.5:
            first_goal = "25-35 minutes"
            late_goals = "POSSIBLE"
        else:
            first_goal = "15-30 minutes"
            late_goals = "LIKELY"
        
        return {
            'first_goal': first_goal,
            'late_goals': late_goals,
            'most_action': "Last 20 minutes of each half" if total_xg > 2.0 else "Scattered throughout"
        }
    
    def _generate_betting_recommendations(self, probabilities: Dict) -> Dict[str, Any]:
        """Generate precise betting recommendations"""
        recs = []
        confidence_levels = []
        
        outcomes = probabilities['match_outcomes']
        btts = probabilities['both_teams_score']
        over_under = probabilities['over_under']
        
        # Match outcome recommendations
        if outcomes['home_win'] > 55:
            recs.append(f"HOME WIN ({outcomes['home_win']}%)")
            confidence_levels.append(min(95, outcomes['home_win']))
        elif outcomes['away_win'] > 55:
            recs.append(f"AWAY WIN ({outcomes['away_win']}%)")
            confidence_levels.append(min(95, outcomes['away_win']))
        elif outcomes['draw'] > 35:
            recs.append(f"DRAW ({outcomes['draw']}%)")
            confidence_levels.append(min(80, outcomes['draw']))
        
        # BTTS recommendations
        if btts > 65:
            recs.append(f"BOTH TEAMS SCORE: YES ({btts}%)")
            confidence_levels.append(min(90, btts))
        elif btts < 35:
            recs.append(f"BOTH TEAMS SCORE: NO ({100-btts}%)")
            confidence_levels.append(min(90, 100-btts))
        
        # Over/under recommendations
        if over_under['over_2.5'] > 60:
            recs.append(f"OVER 2.5 GOALS ({over_under['over_2.5']}%)")
            confidence_levels.append(min(85, over_under['over_2.5']))
        elif over_under['over_2.5'] < 40:
            recs.append(f"UNDER 2.5 GOALS ({100-over_under['over_2.5']}%)")
            confidence_levels.append(min(85, 100-over_under['over_2.5']))
        
        # Always include corners based on analysis
        recs.append("OVER 9.5 TOTAL CORNERS")
        confidence_levels.append(65)
        
        # Top confidence bet
        if confidence_levels:
            top_confidence = max(confidence_levels)
            top_index = confidence_levels.index(top_confidence)
            top_bet = recs[top_index]
        else:
            top_confidence = 60
            top_bet = "OVER 1.5 GOALS"
        
        return {
            'recommendations': recs,
            'top_bet': f"{top_bet} - {self._get_confidence_label(top_confidence)} Confidence",
            'confidence_scores': confidence_levels
        }
    
    def _calculate_advanced_confidence(self, home_goals: int, away_goals: int, 
                                    home_form: list, away_form: list, h2h_data: Dict) -> int:
        """Calculate advanced confidence score considering multiple factors"""
        confidence = 0
        
        # Data completeness (max 40 points)
        if home_goals > 0: confidence += 10
        if away_goals > 0: confidence += 10
        if home_form and len(home_form) >= 3: confidence += 10
        if away_form and len(away_form) >= 3: confidence += 10
        
        # Historical data (max 30 points)
        if h2h_data and h2h_data.get('matches', 0) >= 3:
            confidence += 30
        elif h2h_data and h2h_data.get('matches', 0) > 0:
            confidence += 15
        
        # Form consistency (max 30 points)
        if home_form and len(home_form) >= 3:
            form_std = np.std(home_form)
            if form_std < 1.0: confidence += 15
        
        if away_form and len(away_form) >= 3:
            form_std = np.std(away_form)
            if form_std < 1.0: confidence += 15
        
        return min(95, confidence)  # Never 100% in sports
    
    def _assess_prediction_risk(self, probabilities: Dict, confidence: int) -> Dict[str, str]:
        """Assess the risk level of predictions"""
        outcomes = probabilities['match_outcomes']
        highest_prob = max(outcomes.values())
        
        if highest_prob > 70 and confidence > 80:
            risk_level = "LOW"
            explanation = "Strong favorite with good data support"
        elif highest_prob > 55 and confidence > 65:
            risk_level = "MEDIUM"
            explanation = "Moderate favorite with reasonable data"
        else:
            risk_level = "HIGH"
            explanation = "Uncertain outcome or limited data"
        
        return {
            'risk_level': risk_level,
            'explanation': explanation,
            'certainty': f"{highest_prob}%"
        }
    
    def _get_confidence_label(self, confidence: float) -> str:
        """Convert confidence percentage to label"""
        if confidence >= 85:
            return "Very High"
        elif confidence >= 75:
            return "High"
        elif confidence >= 65:
            return "Good"
        elif confidence >= 55:
            return "Moderate"
        else:
            return "Low"
    
    def _generate_detailed_summary(self, home_team: str, away_team: str, probabilities: Dict, 
                                 home_xg: float, away_xg: float) -> str:
        """Generate detailed match summary"""
        outcomes = probabilities['match_outcomes']
        
        if outcomes['home_win'] > 60:
            return f"{home_team} are strong favorites to win this match. Expect controlled possession and multiple scoring opportunities against {away_team}."
        elif outcomes['home_win'] > 45:
            return f"{home_team} have a slight advantage playing at home. This could be a close match with {away_team} posing a threat on the counter-attack."
        elif outcomes['away_win'] > 45:
            return f"{away_team} might cause an upset here. {home_team} will need to be careful defensively while pushing for goals at home."
        else:
            return f"This match appears evenly balanced with a draw being a distinct possibility. Both teams will fancy their chances in what should be a competitive encounter."
