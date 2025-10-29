import numpy as np
from scipy.stats import poisson
from typing import Dict, Any

class SimplePredictionEngine:
    """Simple and Clear Prediction Engine - No Complex Analysis"""
    
    def __init__(self, match_data: Dict[str, Any]):
        self.data = match_data
    
    def generate_simple_predictions(self) -> Dict[str, Any]:
        """Generate clear, simple predictions anyone can understand"""
        
        # Extract basic data
        home_team = self.data.get('home_team', 'Home Team')
        away_team = self.data.get('away_team', 'Away Team')
        
        # Calculate simple probabilities
        home_goals = self.data.get('home_goals', 0)
        away_goals = self.data.get('away_goals', 0)
        home_conceded = self.data.get('home_conceded', 0)
        away_conceded = self.data.get('away_conceded', 0)
        
        # Simple goal expectation
        home_attack = home_goals / 6.0 if home_goals > 0 else 1.0
        away_attack = away_goals / 6.0 if away_goals > 0 else 0.8
        home_defense = home_conceded / 6.0 if home_conceded > 0 else 1.2
        away_defense = away_conceded / 6.0 if away_conceded > 0 else 1.5
        
        # Expected goals
        home_xg = (home_attack * away_defense * 1.1)  # Home advantage
        away_xg = (away_attack * home_defense)
        
        # Ensure reasonable bounds
        home_xg = max(0.3, min(3.0, home_xg))
        away_xg = max(0.2, min(2.5, away_xg))
        
        # Calculate probabilities
        # 1st half goal probability
        first_half_goal_prob = 1 - (poisson.pmf(0, home_xg * 0.45) * poisson.pmf(0, away_xg * 0.45))
        
        # 2nd half goal probability  
        second_half_goal_prob = 1 - (poisson.pmf(0, home_xg * 0.55) * poisson.pmf(0, away_xg * 0.55))
        
        # Total goals probability
        total_goals_prob = home_xg + away_xg
        
        # BTTS probability
        home_score_prob = 1 - poisson.pmf(0, home_xg)
        away_score_prob = 1 - poisson.pmf(0, away_xg)
        btts_prob = home_score_prob * away_score_prob
        
        # Win probabilities
        home_win_prob, draw_prob, away_win_prob = self._calculate_1x2_probabilities(home_xg, away_xg)
        
        # Corner prediction (simple average based on goals)
        total_corners = (home_goals + away_goals) * 1.8 + 6
        home_corners = total_corners * 0.6
        away_corners = total_corners * 0.4
        
        # Determine confidence levels
        confidence = self._calculate_confidence(home_goals, away_goals, home_conceded, away_conceded)
        
        return {
            'match': f"{home_team} vs {away_team}",
            'goals_prediction': {
                'first_half_goal': {
                    'answer': 'YES' if first_half_goal_prob > 0.6 else 'NO',
                    'probability': round(first_half_goal_prob * 100),
                    'confidence': 'HIGH' if first_half_goal_prob > 0.7 else 'MEDIUM' if first_half_goal_prob > 0.55 else 'LOW'
                },
                'second_half_goal': {
                    'answer': 'YES' if second_half_goal_prob > 0.6 else 'NO', 
                    'probability': round(second_half_goal_prob * 100),
                    'confidence': 'HIGH' if second_half_goal_prob > 0.8 else 'MEDIUM' if second_half_goal_prob > 0.65 else 'LOW'
                },
                'total_goals_range': self._get_goals_range(total_goals_prob),
                'btts': {
                    'answer': 'YES' if btts_prob > 0.5 else 'NO',
                    'probability': round(btts_prob * 100)
                }
            },
            'who_scores': {
                'home_team': home_team,
                'away_team': away_team,
                'home_likely': 'VERY LIKELY' if home_score_prob > 0.7 else 'LIKELY' if home_score_prob > 0.5 else 'MIGHT SCORE',
                'away_likely': 'VERY LIKELY' if away_score_prob > 0.7 else 'LIKELY' if away_score_prob > 0.5 else 'MIGHT SCORE'
            },
            'corners_prediction': {
                'total_corners_range': f"{int(total_corners-1)}-{int(total_corners+1)}",
                'home_corners': f"{int(home_corners)}-{int(home_corners+1)}",
                'away_corners': f"{int(away_corners)}-{int(away_corners+1)}"
            },
            'key_timing': {
                'first_goal': '25-40 minutes',
                'late_goals': 'YES' if second_half_goal_prob > 0.7 else 'POSSIBLE',
                'most_action': 'Last 15 minutes of each half'
            },
            'best_bets': self._get_best_bets(home_win_prob, draw_prob, away_win_prob, btts_prob, total_goals_prob),
            'top_confidence_bet': self._get_top_bet(home_win_prob, second_half_goal_prob, btts_prob),
            'summary': self._generate_summary(home_team, away_team, home_win_prob, total_goals_prob),
            'confidence_score': confidence
        }
    
    def _calculate_1x2_probabilities(self, home_xg: float, away_xg: float) -> tuple:
        """Calculate simple 1X2 probabilities"""
        max_goals = 5
        home_probs = [poisson.pmf(i, home_xg) for i in range(max_goals)]
        away_probs = [poisson.pmf(i, away_xg) for i in range(max_goals)]
        
        home_win = 0
        draw = 0
        away_win = 0
        
        for i in range(max_goals):
            for j in range(max_goals):
                prob = home_probs[i] * away_probs[j]
                if i > j:
                    home_win += prob
                elif i == j:
                    draw += prob
                else:
                    away_win += prob
        
        total = home_win + draw + away_win
        return home_win/total, draw/total, away_win/total
    
    def _get_goals_range(self, total_goals: float) -> str:
        """Convert total goals to range"""
        if total_goals >= 3.5:
            return "3-4 GOALS"
        elif total_goals >= 2.8:
            return "2-3 GOALS" 
        elif total_goals >= 2.2:
            return "2 GOALS"
        else:
            return "1-2 GOALS"
    
    def _calculate_confidence(self, home_goals: int, away_goals: int, home_conceded: int, away_conceded: int) -> int:
        """Calculate simple confidence score"""
        data_points = 0
        
        if home_goals > 0: data_points += 25
        if away_goals > 0: data_points += 25  
        if home_conceded > 0: data_points += 25
        if away_conceded > 0: data_points += 25
        
        return min(100, data_points)
    
    def _get_best_bets(self, home_win: float, draw: float, away_win: float, btts: float, total_goals: float) -> list:
        """Determine best betting options"""
        bets = []
        
        if home_win > 0.45:
            bets.append("HOME TEAM TO WIN")
        elif away_win > 0.45:
            bets.append("AWAY TEAM TO WIN")
        else:
            bets.append("DRAW OR HOME WIN")
            
        if btts > 0.55:
            bets.append("BOTH TEAMS TO SCORE: YES")
        else:
            bets.append("BOTH TEAMS TO SCORE: NO")
            
        if total_goals > 2.7:
            bets.append("OVER 2.5 GOALS")
        elif total_goals < 2.0:
            bets.append("UNDER 2.5 GOALS")
        else:
            bets.append("OVER 1.5 GOALS")
            
        # Always include corners
        bets.append("OVER 8.5 TOTAL CORNERS")
        
        return bets
    
    def _get_top_bet(self, home_win: float, second_half_goal: float, btts: float) -> str:
        """Get the top confidence bet"""
        if second_half_goal > 0.75:
            return "HOME TEAM TO SCORE IN SECOND HALF - Very High Confidence"
        elif home_win > 0.6:
            return "HOME TEAM TO WIN - High Confidence" 
        elif btts > 0.65:
            return "BOTH TEAMS TO SCORE - High Confidence"
        else:
            return "OVER 1.5 GOALS - Good Confidence"
    
    def _generate_summary(self, home_team: str, away_team: str, home_win_prob: float, total_goals: float) -> str:
        """Generate simple match summary"""
        if home_win_prob > 0.6:
            return f"Expect {home_team} to win in a game with plenty of goals and action."
        elif home_win_prob > 0.4:
            return f"Close game expected, {home_team} slight favorites with goals likely."
        else:
            return f"{away_team} might cause an upset in what should be an entertaining match."
