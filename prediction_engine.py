import numpy as np
from scipy.stats import poisson
from typing import Dict, Any, Tuple
import math

class ProductionPredictionEngine:
    """Production-ready Football Prediction Engine with Balanced Calculations"""
    
    def __init__(self, match_data: Dict[str, Any]):
        self.data = match_data
        self.league_contexts = self._initialize_league_contexts()
    
    def _initialize_league_contexts(self) -> Dict[str, Dict]:
        """Initialize league-specific parameters for better calibration"""
        return {
            'premier_league': {'avg_goals': 2.8, 'avg_corners': 10.5, 'home_advantage': 1.15, 'goal_variance': 1.1},
            'la_liga': {'avg_goals': 2.6, 'avg_corners': 9.8, 'home_advantage': 1.18, 'goal_variance': 1.05},
            'serie_a': {'avg_goals': 2.7, 'avg_corners': 10.2, 'home_advantage': 1.12, 'goal_variance': 0.95},
            'bundesliga': {'avg_goals': 3.1, 'avg_corners': 9.5, 'home_advantage': 1.16, 'goal_variance': 1.2},
            'ligue_1': {'avg_goals': 2.5, 'avg_corners': 9.2, 'home_advantage': 1.14, 'goal_variance': 1.0},
            'default': {'avg_goals': 2.7, 'avg_corners': 10.0, 'home_advantage': 1.15, 'goal_variance': 1.0}
        }
    
    def generate_predictions(self) -> Dict[str, Any]:
        """Generate production-ready predictions with balanced calculations"""
        
        # Extract and validate data
        home_team = self.data.get('home_team', 'Home Team')
        away_team = self.data.get('away_team', 'Away Team')
        league = self.data.get('league', 'default')
        
        # Get advanced metrics
        home_goals = self.data.get('home_goals', 0)
        away_goals = self.data.get('away_goals', 0)
        home_conceded = self.data.get('home_conceded', 0)
        away_conceded = self.data.get('away_conceded', 0)
        
        # Advanced inputs with defaults
        home_goals_home = self.data.get('home_goals_home', home_goals)
        away_goals_away = self.data.get('away_goals_away', away_goals)
        home_form = self.data.get('home_form', [])
        away_form = self.data.get('away_form', [])
        h2h_data = self.data.get('h2h_data', {})
        injuries = self.data.get('injuries', {'home': 0, 'away': 0})
        motivation = self.data.get('motivation', {'home': 1.0, 'away': 1.0})
        
        # Calculate defensive-aware expected goals
        home_xg, away_xg = self._calculate_defensive_aware_xg(
            home_goals, away_goals, home_conceded, away_conceded,
            home_goals_home, away_goals_away, home_form, away_form,
            injuries, motivation, league
        )
        
        # Apply H2H adjustments if available
        if h2h_data and h2h_data.get('matches', 0) >= 2:
            home_xg, away_xg = self._apply_conservative_h2h_adjustment(home_xg, away_xg, h2h_data)
        
        # Apply defensive game detection
        home_xg, away_xg = self._apply_defensive_game_detection(home_xg, away_xg, home_conceded, away_conceded)
        
        # Calculate all probabilities with professional calibration
        probabilities = self._calculate_professional_probabilities(home_xg, away_xg, league)
        
        # Generate realistic corner predictions
        corner_predictions = self._calculate_precise_corners(home_xg, away_xg, league)
        
        # Generate timing predictions with late goal awareness
        timing_predictions = self._calculate_enhanced_timing_predictions(home_xg, away_xg, probabilities)
        
        # Generate betting recommendations
        betting_recommendations = self._generate_professional_betting_recommendations(probabilities, home_xg, away_xg)
        
        # Calculate overall confidence
        confidence_score = self._calculate_production_confidence(
            home_goals, away_goals, home_form, away_form, h2h_data
        )
        
        # Generate accurate summary
        summary = self._generate_accurate_summary(home_team, away_team, probabilities, home_xg, away_xg)
        
        return {
            'match': f"{home_team} vs {away_team}",
            'expected_goals': {'home': round(home_xg, 2), 'away': round(away_xg, 2)},
            'probabilities': probabilities,
            'corner_predictions': corner_predictions,
            'timing_predictions': timing_predictions,
            'betting_recommendations': betting_recommendations,
            'summary': summary,
            'confidence_score': confidence_score,
            'risk_assessment': self._assess_professional_risk(probabilities, confidence_score, home_xg, away_xg)
        }
    
    def _calculate_defensive_aware_xg(self, home_goals: int, away_goals: int, home_conceded: int, 
                                    away_conceded: int, home_goals_home: int, away_goals_away: int,
                                    home_form: list, away_form: list, injuries: Dict, 
                                    motivation: Dict, league: str) -> Tuple[float, float]:
        """Calculate defensive-aware expected goals with strong regression"""
        
        league_params = self.league_contexts.get(league, self.league_contexts['default'])
        
        # Calculate base rates with heavy regression to league mean
        home_attack_base = max(0.3, home_goals / 6.0)
        away_attack_base = max(0.2, away_goals / 6.0)
        home_defense_base = max(0.3, home_conceded / 6.0)
        away_defense_base = max(0.4, away_conceded / 6.0)
        
        # Apply strong Bayesian regression - prioritize league averages
        home_attack = self._apply_strong_regression(home_attack_base, league_params['avg_goals'] / 2, 8)
        away_attack = self._apply_strong_regression(away_attack_base, league_params['avg_goals'] / 2, 8)
        home_defense = self._apply_strong_regression(home_defense_base, league_params['avg_goals'] / 2, 8)
        away_defense = self._apply_strong_regression(away_defense_base, league_params['avg_goals'] / 2, 8)
        
        # Home/away specific adjustments with conservative caps
        home_attack_home = min(2.0, max(0.3, home_goals_home / 3.0)) if home_goals_home > 0 else home_attack
        away_attack_away = min(1.8, max(0.2, away_goals_away / 3.0)) if away_goals_away > 0 else away_attack
        
        # Conservative form adjustments
        home_form_factor = self._calculate_conservative_form_factor(home_form)
        away_form_factor = self._calculate_conservative_form_factor(away_form)
        
        # Injury adjustments (moderate impact)
        home_injury_factor = max(0.85, 1.0 - (injuries.get('home', 0) * 0.05))
        away_injury_factor = max(0.85, 1.0 - (injuries.get('away', 0) * 0.05))
        
        # Motivation adjustments (moderate)
        home_motivation = motivation.get('home', 1.0)
        away_motivation = motivation.get('away', 1.0)
        
        # Calculate base xG with defensive strength emphasis
        base_home_xg = (home_attack_home * (1.3 - away_defense) * league_params['home_advantage'] * 
                       home_form_factor * home_injury_factor * home_motivation)
        base_away_xg = (away_attack_away * (1.3 - home_defense) * 
                       away_form_factor * away_injury_factor * away_motivation)
        
        # Apply heavy regression and ensure realistic values
        home_xg = min(2.5, max(0.3, (base_home_xg * 0.6 + league_params['avg_goals'] * 0.5 * 0.4)))
        away_xg = min(2.0, max(0.2, (base_away_xg * 0.6 + league_params['avg_goals'] * 0.5 * 0.4)))
        
        return round(home_xg, 2), round(away_xg, 2)
    
    def _apply_strong_regression(self, observed: float, prior: float, sample_size: int) -> float:
        """Apply strong Bayesian regression to prevent overfitting"""
        # Heavy regression toward prior, especially for small samples
        weight = min(0.5, sample_size / (sample_size + 15))
        return (observed * weight) + (prior * (1 - weight))
    
    def _apply_conservative_h2h_adjustment(self, home_xg: float, away_xg: float, h2h_data: Dict) -> Tuple[float, float]:
        """Apply conservative H2H adjustments with minimal impact"""
        matches = h2h_data.get('matches', 0)
        home_wins = h2h_data.get('home_wins', 0)
        away_wins = h2h_data.get('away_wins', 0)
        draws = h2h_data.get('draws', 0)
        home_goals = h2h_data.get('home_goals', 0)
        away_goals = h2h_data.get('away_goals', 0)
        
        if matches < 3:
            return home_xg, away_xg
        
        # Calculate H2H dominance with very conservative impact
        home_dominance = (home_wins + draws * 0.5) / matches
        h2h_home_avg = home_goals / matches if matches > 0 else home_xg
        h2h_away_avg = away_goals / matches if matches > 0 else away_xg
        
        # Very light blend with H2H history (90/10 weight)
        adjusted_home_xg = (home_xg * 0.9) + (h2h_home_avg * 0.1)
        adjusted_away_xg = (away_xg * 0.9) + (h2h_away_avg * 0.1)
        
        # Minimal dominance adjustment
        if home_dominance > 0.75:  # Very strong home dominance
            adjusted_home_xg *= 1.03
            adjusted_away_xg *= 0.97
        elif home_dominance < 0.25:  # Very weak home performance
            adjusted_home_xg *= 0.97
            adjusted_away_xg *= 1.03
        
        return min(2.5, adjusted_home_xg), min(2.0, adjusted_away_xg)
    
    def _apply_defensive_game_detection(self, home_xg: float, away_xg: float, home_conceded: int, away_conceded: int) -> Tuple[float, float]:
        """Detect and adjust for likely defensive games"""
        total_xg = home_xg + away_xg
        
        # If both teams have strong defensive records, reduce xG
        home_defensive_strength = home_conceded / 6.0 if home_conceded > 0 else 1.5
        away_defensive_strength = away_conceded / 6.0 if away_conceded > 0 else 1.5
        
        # Apply defensive game adjustment
        if home_defensive_strength < 1.0 and away_defensive_strength < 1.0:
            # Both teams strong defensively
            reduction_factor = 0.85
            home_xg *= reduction_factor
            away_xg *= reduction_factor
        elif home_defensive_strength < 1.0 or away_defensive_strength < 1.0:
            # One team strong defensively
            reduction_factor = 0.92
            home_xg *= reduction_factor
            away_xg *= reduction_factor
        
        return home_xg, away_xg
    
    def _calculate_conservative_form_factor(self, form: list) -> float:
        """Calculate very conservative form factor"""
        if not form or len(form) == 0:
            return 1.0
        
        # Very conservative weights
        weights = [0.8, 0.6, 0.4, 0.2, 0.1][:len(form)]
        total_points = sum(score * weight for score, weight in zip(form, weights))
        max_possible = sum(3 * weight for weight in weights)
        
        form_ratio = total_points / max_possible if max_possible > 0 else 0.5
        
        # Very limited impact range
        return 0.95 + (form_ratio * 0.1)
    
    def _calculate_professional_probabilities(self, home_xg: float, away_xg: float, league: str) -> Dict[str, Any]:
        """Calculate professionally calibrated probabilities"""
        
        # Match outcome probabilities using Poisson distribution
        home_win_prob, draw_prob, away_win_prob = self._calculate_match_outcomes(home_xg, away_xg)
        
        # Apply professional calibration - prevent overconfidence
        home_win_prob = self._professional_calibration(home_win_prob)
        away_win_prob = self._professional_calibration(away_win_prob)
        draw_prob = self._professional_calibration(draw_prob)
        
        # Normalize after calibration
        total = home_win_prob + draw_prob + away_win_prob
        home_win_prob /= total
        draw_prob /= total
        away_win_prob /= total
        
        # Goal timing probabilities with realistic caps
        first_half_prob = min(0.80, self._calculate_goal_timing_probability(home_xg, away_xg, 'first_half'))
        second_half_prob = min(0.85, self._calculate_goal_timing_probability(home_xg, away_xg, 'second_half'))
        
        # Both teams to score probability
        btts_prob = self._professional_calibration(self._calculate_btts_probability(home_xg, away_xg))
        
        # Over/under probabilities with professional calibration
        over_15_prob = self._professional_calibration(1 - (poisson.pmf(0, home_xg) * poisson.pmf(0, away_xg) + 
                           poisson.pmf(1, home_xg) * poisson.pmf(0, away_xg) + 
                           poisson.pmf(0, home_xg) * poisson.pmf(1, away_xg)))
        
        over_25_prob = self._professional_calibration(1 - sum(poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg) 
                              for i in range(3) for j in range(3 - i)))
        
        over_35_prob = self._professional_calibration(1 - sum(poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg) 
                              for i in range(4) for j in range(4 - i)))
        
        # Exact score probabilities with improved calculation
        exact_scores = self._calculate_realistic_scores(home_xg, away_xg)
        
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
            'exact_scores': exact_scores
        }
    
    def _professional_calibration(self, prob: float) -> float:
        """Professional probability calibration to prevent overconfidence"""
        # Strong calibration that pulls extremes toward realistic ranges
        if prob > 0.7:
            return 0.6 + (prob - 0.7) * 0.5  # Heavy reduction for high probabilities
        elif prob > 0.5:
            return 0.5 + (prob - 0.5) * 0.7  # Moderate reduction
        else:
            return prob * 1.1  # Slight increase for low probabilities
    
    def _calculate_match_outcomes(self, home_xg: float, away_xg: float) -> Tuple[float, float, float]:
        """Calculate match outcome probabilities using Poisson distribution"""
        max_goals = 8
        
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
        
        total = home_win + draw + away_win
        return home_win/total, draw/total, away_win/total
    
    def _calculate_realistic_scores(self, home_xg: float, away_xg: float) -> Dict[str, float]:
        """Calculate realistic score probabilities with improved distribution"""
        exact_scores = {}
        
        # Calculate probabilities for common scorelines
        for i in range(4):  # Home goals
            for j in range(4):  # Away goals
                prob = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
                if prob > 0.015:  # Only include probabilities > 1.5%
                    exact_scores[f"{i}-{j}"] = round(prob * 100, 1)
        
        # Ensure 1-0 and 0-1 have reasonable probabilities for low-scoring expectations
        if home_xg + away_xg < 2.5:
            prob_1_0 = poisson.pmf(1, home_xg) * poisson.pmf(0, away_xg)
            prob_0_1 = poisson.pmf(0, home_xg) * poisson.pmf(1, away_xg)
            if prob_1_0 > 0.01:
                exact_scores["1-0"] = round(prob_1_0 * 100, 1)
            if prob_0_1 > 0.01:
                exact_scores["0-1"] = round(prob_0_1 * 100, 1)
        
        return dict(sorted(exact_scores.items(), key=lambda x: x[1], reverse=True)[:6])
    
    def _calculate_goal_timing_probability(self, home_xg: float, away_xg: float, half: str) -> float:
        """Calculate probability of goals in specific half"""
        if half == 'first_half':
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
    
    def _calculate_precise_corners(self, home_xg: float, away_xg: float, league: str) -> Dict[str, Any]:
        """Calculate precise corner predictions with realistic bounds"""
        league_params = self.league_contexts.get(league, self.league_contexts['default'])
        
        # Corner prediction based on expected goals with tight bounds
        base_corners = league_params['avg_corners']
        attacking_bonus = (home_xg + away_xg - league_params['avg_goals']) * 0.8  # Reduced impact
        
        total_corners = base_corners + attacking_bonus
        # Very tight realistic bounds for corners (6-12 range)
        total_corners = max(6, min(12, total_corners))
        
        # Home teams typically get more corners
        home_corners = total_corners * 0.55
        away_corners = total_corners * 0.45
        
        # Round to nearest integer for ranges
        total_int = int(round(total_corners))
        home_int = int(round(home_corners))
        away_int = int(round(away_corners))
        
        return {
            'total': f"{total_int}-{total_int + 1}",
            'home': f"{home_int}-{home_int + 1}",
            'away': f"{away_int}-{away_int + 1}",
            'over_9.5': 'YES' if total_corners > 9.5 else 'NO'
        }
    
    def _calculate_enhanced_timing_predictions(self, home_xg: float, away_xg: float, probabilities: Dict) -> Dict[str, Any]:
        """Calculate enhanced timing predictions with late goal awareness"""
        total_xg = home_xg + away_xg
        btts_prob = probabilities['both_teams_score'] / 100
        
        if total_xg < 1.8:
            first_goal = "30+ minutes"
            late_goals = "UNLIKELY"
        elif total_xg < 2.5:
            first_goal = "25-35 minutes"
            # Higher chance of late goals if BTTS is likely
            late_goals = "POSSIBLE" if btts_prob > 0.5 else "UNLIKELY"
        else:
            first_goal = "15-30 minutes"
            late_goals = "LIKELY"
        
        # Adjust for defensive games
        if total_xg < 2.0:
            action_pattern = "Scattered throughout"
        elif total_xg < 3.0:
            action_pattern = "Last 15 minutes of each half"
        else:
            action_pattern = "Last 20 minutes of each half"
        
        return {
            'first_goal': first_goal,
            'late_goals': late_goals,
            'most_action': action_pattern
        }
    
    def _generate_professional_betting_recommendations(self, probabilities: Dict, home_xg: float, away_xg: float) -> Dict[str, Any]:
        """Generate professional betting recommendations"""
        recs = []
        confidence_levels = []
        
        outcomes = probabilities['match_outcomes']
        btts = probabilities['both_teams_score']
        over_under = probabilities['over_under']
        total_xg = home_xg + away_xg
        
        # Match outcome recommendations with professional thresholds
        if outcomes['home_win'] > 55:
            recs.append(f"HOME WIN ({outcomes['home_win']}%)")
            confidence_levels.append(min(75, outcomes['home_win']))
        elif outcomes['away_win'] > 55:
            recs.append(f"AWAY WIN ({outcomes['away_win']}%)")
            confidence_levels.append(min(75, outcomes['away_win']))
        elif outcomes['draw'] > 35:
            recs.append(f"DRAW ({outcomes['draw']}%)")
            confidence_levels.append(min(70, outcomes['draw']))
        
        # BTTS recommendations with balanced thresholds
        if btts > 65:
            recs.append(f"BOTH TEAMS SCORE: YES ({btts}%)")
            confidence_levels.append(min(80, btts))
        elif btts < 35:
            recs.append(f"BOTH TEAMS SCORE: NO ({100-btts}%)")
            confidence_levels.append(min(80, 100-btts))
        
        # Over/under recommendations with xG consideration
        if over_under['over_2.5'] > 65 and total_xg > 2.8:
            recs.append(f"OVER 2.5 GOALS ({over_under['over_2.5']}%)")
            confidence_levels.append(min(75, over_under['over_2.5']))
        elif over_under['over_2.5'] < 35 and total_xg < 2.2:
            recs.append(f"UNDER 2.5 GOALS ({100-over_under['over_2.5']}%)")
            confidence_levels.append(min(75, 100-over_under['over_2.5']))
        
        # Corners recommendation only if confident
        if total_xg > 2.5:
            recs.append("OVER 9.5 TOTAL CORNERS")
            confidence_levels.append(65)
        else:
            recs.append("OVER 8.5 TOTAL CORNERS")
            confidence_levels.append(60)
        
        # Select top bet - prioritize match outcomes over corners
        top_bet = "OVER 1.5 GOALS"
        top_confidence = 55
        
        for i, rec in enumerate(recs):
            if confidence_levels[i] > top_confidence and not "CORNERS" in rec:
                top_bet = rec
                top_confidence = confidence_levels[i]
        
        # If no good match bets, use corners as last resort
        if top_confidence < 60:
            for i, rec in enumerate(recs):
                if "CORNERS" in rec and confidence_levels[i] > top_confidence:
                    top_bet = rec
                    top_confidence = confidence_levels[i]
        
        return {
            'recommendations': recs,
            'top_bet': f"{top_bet} - {self._get_confidence_label(top_confidence)} Confidence",
            'confidence_scores': confidence_levels
        }
    
    def _calculate_production_confidence(self, home_goals: int, away_goals: int, 
                                      home_form: list, away_form: list, h2h_data: Dict) -> int:
        """Calculate production-ready confidence score"""
        confidence = 0
        
        # Data completeness (max 40 points)
        if home_goals > 0: confidence += 10
        if away_goals > 0: confidence += 10
        if home_form and len(home_form) >= 3: confidence += 10
        if away_form and len(away_form) >= 3: confidence += 10
        
        # Historical data (max 30 points)
        if h2h_data and h2h_data.get('matches', 0) >= 5:
            confidence += 30
        elif h2h_data and h2h_data.get('matches', 0) >= 3:
            confidence += 20
        elif h2h_data and h2h_data.get('matches', 0) > 0:
            confidence += 10
        
        # Form consistency (max 30 points)
        if home_form and len(home_form) >= 3:
            form_std = np.std(home_form) if len(home_form) > 1 else 0
            if form_std < 1.0: confidence += 15
            elif form_std < 1.5: confidence += 10
        
        if away_form and len(away_form) >= 3:
            form_std = np.std(away_form) if len(away_form) > 1 else 0
            if form_std < 1.0: confidence += 15
            elif form_std < 1.5: confidence += 10
        
        return min(85, confidence)  # Conservative maximum
    
    def _assess_professional_risk(self, probabilities: Dict, confidence: int, home_xg: float, away_xg: float) -> Dict[str, str]:
        """Assess professional risk level"""
        outcomes = probabilities['match_outcomes']
        highest_prob = max(outcomes.values())
        prob_diff = max(outcomes.values()) - sorted(outcomes.values())[1]  # Difference between top 2
        
        # Enhanced risk assessment considering multiple factors
        if highest_prob > 70 and prob_diff > 25 and confidence > 75:
            risk_level = "LOW"
            explanation = "Clear favorite with strong data support"
        elif highest_prob > 60 and prob_diff > 15 and confidence > 65:
            risk_level = "MEDIUM-LOW"
            explanation = "Strong favorite with good data"
        elif highest_prob > 52 and prob_diff > 8 and confidence > 60:
            risk_level = "MEDIUM"
            explanation = "Moderate favorite with reasonable data"
        elif highest_prob > 45:
            risk_level = "MEDIUM-HIGH"
            explanation = "Slight favorite, competitive match expected"
        else:
            risk_level = "HIGH"
            explanation = "Highly uncertain outcome"
        
        return {
            'risk_level': risk_level,
            'explanation': explanation,
            'certainty': f"{highest_prob}%"
        }
    
    def _generate_accurate_summary(self, home_team: str, away_team: str, probabilities: Dict, 
                                 home_xg: float, away_xg: float) -> str:
        """Generate accurate match summary based on probabilities"""
        outcomes = probabilities['match_outcomes']
        home_win = outcomes['home_win']
        away_win = outcomes['away_win']
        draw = outcomes['draw']
        
        prob_diff = abs(home_win - away_win)
        total_xg = home_xg + away_xg
        
        if home_win > 60 and prob_diff > 20:
            return f"{home_team} are strong favorites to win this match. They should control the game and create more scoring opportunities against {away_team}."
        elif home_win > 55 and prob_diff > 15:
            return f"{home_team} have a clear advantage playing at home and are expected to secure a victory against {away_team}."
        elif home_win > 50 and prob_diff > 10:
            return f"{home_team} are slight favorites at home. This could be a competitive match with {away_team} looking to counter-attack."
        elif away_win > 50 and prob_diff > 10:
            return f"{away_team} might cause an upset here. {home_team} will need to be organized defensively to get a result at home."
        elif prob_diff < 8:
            if total_xg > 2.8:
                return f"This match appears evenly balanced and could be an entertaining affair with goals expected from both sides."
            else:
                return f"This match appears evenly balanced and could be a tight, tactical encounter with few clear chances."
        else:
            return f"Both teams will fancy their chances in what should be a competitive encounter with no clear favorite."
    
    def _get_confidence_label(self, confidence: float) -> str:
        """Convert confidence percentage to label"""
        if confidence >= 80:
            return "Very High"
        elif confidence >= 70:
            return "High"
        elif confidence >= 60:
            return "Good"
        elif confidence >= 50:
            return "Moderate"
        else:
            return "Low"
