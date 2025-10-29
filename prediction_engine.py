import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import math

class QualitativeIntelligence:
    """Qualitative Intelligence Module - The Edge That Beats Other Models"""
    
    def __init__(self):
        self.motivation_factors = {}
        self.tactical_insights = {}
        self.contextual_boosters = {}
        self.trap_game_signals = []
        
    def analyze_complete_context(self, home_data: Dict, away_data: Dict, 
                               league_context: Dict, market_data: Dict) -> Dict[str, Any]:
        """Comprehensive qualitative analysis that other models miss"""
        
        analysis = {
            'motivation_score': 0.0,
            'trap_signals': [],
            'qualitative_boosts': {'home': 1.0, 'away': 1.0, 'draw': 1.0},
            'confidence_adjustment': 0.0,
            'key_insights': [],
            'risk_alerts': []
        }
        
        # 1. MOTIVATION ANALYSIS (Massive Edge)
        motivation_score = self._analyze_motivation_gap(home_data, away_data, league_context)
        analysis['motivation_score'] = motivation_score
        
        # 2. TRAP GAME DETECTION
        trap_signals = self._detect_trap_games(home_data, away_data, market_data)
        analysis['trap_signals'] = trap_signals
        
        # 3. Apply qualitative boosts
        analysis['qualitative_boosts'] = self._calculate_qualitative_boosts(
            motivation_score, trap_signals, home_data, away_data
        )
        
        # 4. Generate insights for betting recommendations
        analysis['key_insights'] = self._generate_qualitative_insights(
            motivation_score, trap_signals, home_data, away_data
        )
        
        # 5. Risk alerts
        analysis['risk_alerts'] = self._generate_risk_alerts(trap_signals)
        
        return analysis
    
    def _analyze_motivation_gap(self, home_data: Dict, away_data: Dict, league_context: Dict) -> float:
        """Quantify motivation differential - CRUSH other models here"""
        motivation_score = 0.0
        
        # Extract key data with safe defaults
        home_pos = home_data.get('league_position', 10)
        away_pos = away_data.get('league_position', 10)
        home_pts = home_data.get('total_points', 0)
        away_pts = away_data.get('total_points', 0)
        
        # 1. RELEGATION BATTLE BOOSTER (Massive Edge)
        if home_pos >= 15 and away_pos <= 8:
            motivation_score += 0.25
            self.motivation_factors['relegation_desperation'] = 'HIGH'
            self.motivation_factors['home_fighting_survival'] = True
        
        elif away_pos >= 15 and home_pos <= 8:
            motivation_score -= 0.25  # Away team desperate
            self.motivation_factors['away_fighting_survival'] = True
        
        # 2. EUROPEAN QUALIFICATION STAKES
        if home_pos <= 6 and away_pos <= 6:
            # Both teams fighting for Europe - intense match
            motivation_score += 0.15
            self.motivation_factors['european_qualification_battle'] = True
        
        # 3. EUROPEAN HANGOVER / FATIGUE
        if away_data.get('had_european_match', False) and not home_data.get('had_european_match', False):
            motivation_score += 0.18
            self.motivation_factors['european_fatigue'] = 'AWAY_TEAM'
        
        if home_data.get('had_european_match', False) and not away_data.get('had_european_match', False):
            motivation_score -= 0.15
            self.motivation_factors['european_fatigue'] = 'HOME_TEAM'
        
        # 4. NEW MANAGER BOOST (Proven statistical edge)
        home_manager_matches = home_data.get('manager_matches_in_charge', 20)
        away_manager_matches = away_data.get('manager_matches_in_charge', 20)
        
        if home_manager_matches <= 3:
            motivation_score += 0.12
            self.motivation_factors['new_manager_bounce'] = 'HOME'
        
        if away_manager_matches <= 3:
            motivation_score -= 0.10
            self.motivation_factors['new_manager_bounce'] = 'AWAY'
        
        # 5. DERBY/SPECIAL CONTEXT
        if self._is_special_fixture(home_data, away_data):
            motivation_score += 0.20
            self.motivation_factors['special_fixture'] = 'HIGH_INTENSITY'
        
        # 6. FORM MOMENTUM vs TABLE POSITION
        home_recent_pts = home_data.get('recent_points', 0)  # Last 5 games
        away_recent_pts = away_data.get('recent_points', 0)
        
        if home_recent_pts > away_recent_pts + 4:  # Home team in better form
            motivation_score += 0.10
        
        # 7. GOAL DROUGHT / SCORING PRESSURE
        if home_data.get('goals_scored', 0) <= 2 and home_data.get('matches_played', 6) >= 5:
            motivation_score += 0.08  # Extra motivation to score
        
        return max(-0.4, min(0.4, motivation_score))  # Cap at ±40%
    
    def _detect_trap_games(self, home_data: Dict, away_data: Dict, market_data: Dict) -> List[str]:
        """Find games where statistical models get fooled"""
        trap_signals = []
        
        # Market data extraction
        home_odds = market_data.get('home_odds', 2.0)
        away_odds = market_data.get('away_odds', 2.0)
        home_implied = 1 / home_odds if home_odds > 0 else 0.33
        away_implied = 1 / away_odds if away_odds > 0 else 0.33
        
        # 1. PUBLIC DARLING TRAP
        if (away_implied > 0.65 and  # Market heavily favors away
            home_data.get('league_position', 10) - away_data.get('league_position', 10) <= 8):
            trap_signals.append('public_overbet_favorite')
        
        # 2. STATISTICAL DOMINANCE BUT NO SUBSTANCE
        home_xg = home_data.get('expected_goals', 1.0)
        away_xg = away_data.get('expected_goals', 1.0)
        home_actual = home_data.get('goals_scored', 1.0)
        away_actual = away_data.get('goals_scored', 1.0)
        
        if (away_xg - home_xg > 0.8 and  # Away team has better underlying stats
            away_actual - home_actual < 0.3):  # But not converting to goals
            trap_signals.append('xg_merchant_team')
        
        # 3. TRAVEL FATIGUE
        home_travel = home_data.get('travel_distance_km', 0)
        away_travel = away_data.get('travel_distance_km', 0)
        
        if away_travel > home_travel + 500:  # Significant travel disadvantage
            trap_signals.append('travel_disadvantage')
        
        # 4. INJURY CRISIS
        home_key_missing = home_data.get('key_players_missing', 0)
        away_key_missing = away_data.get('key_players_missing', 0)
        
        if away_key_missing >= 2 and away_implied > 0.5:
            trap_signals.append('key_players_missing')
        
        # 5. SCHEDULE CONGESTION
        if away_data.get('matches_last_14_days', 0) >= 4:
            trap_signals.append('schedule_congestion')
        
        return trap_signals
    
    def _calculate_qualitative_boosts(self, motivation_score: float, trap_signals: List[str],
                                    home_data: Dict, away_data: Dict) -> Dict[str, float]:
        """Convert qualitative analysis into probability boosts"""
        boosts = {'home': 1.0, 'away': 1.0, 'draw': 1.0}
        
        # 1. Apply motivation boosts
        if motivation_score > 0:
            # Home team motivated
            boosts['home'] *= (1.0 + motivation_score * 0.8)  # Up to 32% boost
            boosts['away'] *= (1.0 - motivation_score * 0.6)  # Up to 24% reduction
            boosts['draw'] *= (1.0 + motivation_score * 0.3)  # Moderate draw boost
        elif motivation_score < 0:
            # Away team motivated
            boosts['away'] *= (1.0 + abs(motivation_score) * 0.8)
            boosts['home'] *= (1.0 - abs(motivation_score) * 0.6)
            boosts['draw'] *= (1.0 + abs(motivation_score) * 0.3)
        
        # 2. Apply trap signal adjustments
        for signal in trap_signals:
            if signal == 'public_overbet_favorite':
                boosts['away'] *= 0.7   # Reduce overhyped away team
                boosts['home'] *= 1.3   # Boost underdog home team
                boosts['draw'] *= 1.2   # Increase draw probability
            
            elif signal == 'travel_disadvantage':
                boosts['away'] *= 0.85
                boosts['home'] *= 1.15
            
            elif signal == 'key_players_missing':
                boosts['away'] *= 0.8
                boosts['home'] *= 1.2
            
            elif signal == 'schedule_congestion':
                boosts['away'] *= 0.9
                boosts['home'] *= 1.1
        
        # Ensure reasonable bounds
        for key in boosts:
            boosts[key] = max(0.5, min(1.8, boosts[key]))
        
        return boosts
    
    def _generate_qualitative_insights(self, motivation_score: float, trap_signals: List[str],
                                     home_data: Dict, away_data: Dict) -> List[str]:
        """Generate human-readable insights for betting decisions"""
        insights = []
        
        if motivation_score > 0.15:
            insights.append(f"🏠 STRONG HOME MOTIVATION: Home team has significant motivational advantage (+{motivation_score:.0%})")
        elif motivation_score < -0.15:
            insights.append(f"✈️ STRONG AWAY MOTIVATION: Away team has significant motivational advantage (+{abs(motivation_score):.0%})")
        
        if 'public_overbet_favorite' in trap_signals:
            insights.append("🎯 TRAP GAME DETECTED: Public overbetting favorite - value on underdog")
        
        if 'travel_disadvantage' in trap_signals:
            insights.append("🚗 TRAVEL FATIGUE: Away team has significant travel disadvantage")
        
        if 'key_players_missing' in trap_signals:
            insights.append("🏥 KEY ABSENCES: Critical players missing for away team")
        
        # Specific context insights
        if home_data.get('league_position', 10) >= 16:
            insights.append("🔥 RELEGATION BATTLE: Home team fighting for survival")
        
        if away_data.get('had_european_match', False):
            insights.append("🌍 EUROPEAN HANGOVER: Away team played in Europe recently")
        
        return insights
    
    def _generate_risk_alerts(self, trap_signals: List[str]) -> List[str]:
        """Generate risk management alerts"""
        alerts = []
        
        if 'public_overbet_favorite' in trap_signals:
            alerts.append("HIGH_RISK: Public darling trap - reduced stakes recommended")
        
        if len(trap_signals) >= 3:
            alerts.append("HIGH_RISK: Multiple trap signals detected - exercise caution")
        
        return alerts
    
    def _is_special_fixture(self, home_data: Dict, away_data: Dict) -> bool:
        """Check if this is a derby or special fixture"""
        derby_keywords = ['derby', 'classico', 'rivalry', 'local']
        home_name = home_data.get('team_name', '').lower()
        away_name = away_data.get('team_name', '').lower()
        
        # Simple implementation - expand based on your league knowledge
        known_derbies = [
            ('lazio', 'roma'), ('inter', 'milan'), ('barcelona', 'real madrid'),
            ('celtic', 'rangers'), ('dortmund', 'schalke'), ('man utd', 'man city')
        ]
        
        for derby in known_derbies:
            if (derby[0] in home_name and derby[1] in away_name) or \
               (derby[1] in home_name and derby[0] in away_name):
                return True
        
        return False

class ProfessionalPredictionEngine:
    """Enhanced Institutional Prediction Engine with Qualitative Intelligence"""
    
    def __init__(self, match_data: Dict[str, Any]):
        self.data = match_data
        self.qualitative_engine = QualitativeIntelligence()
        self.logger = self._setup_logger()
        self.enhancements_active = {
            'timing_intelligence': False,
            'upset_detection': False,
            'pattern_detection': False,
            'market_analysis': False,
            'qualitative_intelligence': False
        }
        
    def _setup_logger(self):
        """Setup professional logging"""
        logger = logging.getLogger('InstitutionalPredictor')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def generate_all_predictions(self) -> Dict[str, Any]:
        """Generate comprehensive institutional predictions with Qualitative Intelligence"""
        self.logger.info("🚀 Starting Enhanced Institutional Prediction Engine with Qualitative Intelligence")
        
        # Calculate base probabilities
        base_predictions = self._calculate_base_probabilities()
        
        # Apply qualitative intelligence (NEW - BEATS OTHER MODELS)
        qualitative_analysis = self._apply_qualitative_intelligence()
        self.enhancements_active['qualitative_intelligence'] = qualitative_analysis.get('active', False)
        
        # Apply timing intelligence
        timing_intelligence = self._calculate_goal_timing_intelligence()
        self.enhancements_active['timing_intelligence'] = timing_intelligence.get('active', False)
        
        # Apply pattern intelligence
        pattern_intelligence = self._detect_pattern_intelligence()
        self.enhancements_active['pattern_detection'] = len(pattern_intelligence.get('patterns_detected', [])) > 0
        
        # Apply upset detection
        upset_analysis = self._detect_contextual_upset_potential()
        self.enhancements_active['upset_detection'] = upset_analysis.get('upset_detected', False)
        
        # Market analysis
        market_comparison = self._analyze_market_efficiency()
        self.enhancements_active['market_analysis'] = market_comparison.get('efficiency_score', 0) > 0
        
        # Combine all enhancements for final probabilities
        final_probabilities = self._apply_enhancements(
            base_predictions, 
            timing_intelligence, 
            pattern_intelligence,
            upset_analysis,
            qualitative_analysis
        )
        
        # Generate betting recommendations
        betting_recommendations = self._generate_betting_recommendations(
            final_probabilities, 
            timing_intelligence,
            upset_analysis,
            qualitative_analysis
        )
        
        # Risk assessment
        risk_assessment = self._calculate_risk_assessment(final_probabilities, qualitative_analysis)
        
        # Compile final results
        results = {
            'predictions': base_predictions,
            'final_probabilities': final_probabilities,
            'qualitative_intelligence': qualitative_analysis,
            'goal_timing_intelligence': timing_intelligence,
            'pattern_intelligence': pattern_intelligence,
            'upset_analysis': upset_analysis,
            'market_comparison': market_comparison,
            'betting_recommendations': betting_recommendations,
            'risk_assessment': risk_assessment,
            'uncertainty': self._calculate_uncertainty(final_probabilities, qualitative_analysis),
            'value_analysis': self._analyze_value_opportunities(final_probabilities),
            'model_type': 'Enhanced Poisson + Pattern Detection + Automated Timing Intelligence + Qualitative Overlay',
            'confidence_score': self._calculate_confidence_score(qualitative_analysis),
            'market_regime': self._determine_market_regime(),
            'precision_metrics': {
                'timing_intelligence_included': self.enhancements_active['timing_intelligence'],
                'upset_detection_active': self.enhancements_active['upset_detection'],
                'pattern_detection_active': self.enhancements_active['pattern_detection'],
                'market_analysis_active': self.enhancements_active['market_analysis'],
                'qualitative_intelligence_active': self.enhancements_active['qualitative_intelligence']
            }
        }
        
        self.logger.info("✅ Enhanced Institutional Predictions with Qualitative Intelligence Complete")
        self._log_enhancement_status()
        
        return results

    def _apply_qualitative_intelligence(self) -> Dict[str, Any]:
        """QUALITATIVE INTELLIGENCE - The Edge That Beats Other Models"""
        self.logger.info("🎯 Applying Qualitative Intelligence Overlay...")
        
        try:
            # Prepare data for qualitative analysis
            home_data = self._prepare_team_data('home')
            away_data = self._prepare_team_data('away')
            market_data = self._prepare_market_data()
            league_context = self._prepare_league_context()
            
            # Run comprehensive qualitative analysis
            qualitative_analysis = self.qualitative_engine.analyze_complete_context(
                home_data, away_data, league_context, market_data
            )
            
            qualitative_analysis['active'] = True
            self.logger.info(f"✅ QUALITATIVE INTELLIGENCE ACTIVE - Motivation Score: {qualitative_analysis['motivation_score']:.3f}")
            
            if qualitative_analysis['trap_signals']:
                self.logger.info(f"🎯 TRAP SIGNALS DETECTED: {qualitative_analysis['trap_signals']}")
            
            return qualitative_analysis
            
        except Exception as e:
            self.logger.error(f"Error in qualitative intelligence: {e}")
            return {
                'active': False,
                'motivation_score': 0.0,
                'trap_signals': [],
                'qualitative_boosts': {'home': 1.0, 'away': 1.0, 'draw': 1.0},
                'key_insights': ['Qualitative analysis unavailable'],
                'risk_alerts': []
            }
    
    def _prepare_team_data(self, team_type: str) -> Dict[str, Any]:
        """Prepare team data for qualitative analysis"""
        prefix = 'home' if team_type == 'home' else 'away'
        
        goals_data = self.data.get(f'{prefix}_goals_data', {})
        standing = self.data.get(f'{prefix}_standing', [10, 0, 0, 0])
        
        return {
            'team_name': self.data.get(f'{prefix}_team', 'Unknown'),
            'league_position': standing[0] if standing else 10,
            'total_points': standing[1] if len(standing) > 1 else 0,
            'matches_played': standing[2] if len(standing) > 2 else 0,
            'goal_difference': standing[3] if len(standing) > 3 else 0,
            'goals_scored': goals_data.get('goals_scored', 0),
            'goals_conceded': goals_data.get('goals_conceded', 0),
            'scoring_frequency': goals_data.get('scoring_frequency', 0),
            'manager_matches_in_charge': 20,  # Default - implement actual tracking
            'had_european_match': False,  # Implement European match detection
            'recent_points': 6,  # Implement recent form tracking
            'travel_distance_km': 0,  # Implement travel distance
            'key_players_missing': 0,  # Implement injury tracking
            'matches_last_14_days': 3,  # Implement schedule tracking
            'expected_goals': goals_data.get('goals_scored', 0) / 6.0  # Simple xG proxy
        }
    
    def _prepare_market_data(self) -> Dict[str, Any]:
        """Prepare market data for qualitative analysis"""
        odds_1x2 = self.data.get('odds_1x2', [2.0, 3.0, 2.0])
        return {
            'home_odds': odds_1x2[0] if len(odds_1x2) > 0 else 2.0,
            'draw_odds': odds_1x2[1] if len(odds_1x2) > 1 else 3.0,
            'away_odds': odds_1x2[2] if len(odds_1x2) > 2 else 2.0
        }
    
    def _prepare_league_context(self) -> Dict[str, Any]:
        """Prepare league context for qualitative analysis"""
        return {
            'league_type': self.data.get('league_type', 'Standard'),
            'match_importance': self.data.get('match_importance', 'Normal League'),
            'venue_context': self.data.get('venue_context', 'Normal'),
            'season_stage': 'mid'  # Implement season stage detection
        }

    def _apply_enhancements(self, base_predictions: Dict, timing_intelligence: Dict, 
                          pattern_intelligence: Dict, upset_analysis: Dict,
                          qualitative_analysis: Dict) -> Dict[str, Any]:
        """Apply all enhancements to base probabilities including Qualitative Intelligence"""
        
        # Start with base probabilities
        enhanced_probs = base_predictions.copy()
        
        # Apply qualitative intelligence FIRST (Most Important)
        if qualitative_analysis.get('active', False):
            enhanced_probs = self._apply_qualitative_enhancements(enhanced_probs, qualitative_analysis)
        
        # Apply timing intelligence to goal-based markets
        if timing_intelligence.get('active', False):
            enhanced_probs = self._apply_timing_enhancements(enhanced_probs, timing_intelligence)
        
        # Apply pattern intelligence
        if pattern_intelligence.get('patterns_detected'):
            enhanced_probs = self._apply_pattern_enhancements(enhanced_probs, pattern_intelligence)
        
        # Apply upset analysis
        if upset_analysis.get('active', False) and upset_analysis.get('upset_detected', False):
            enhanced_probs = self._apply_upset_enhancements(enhanced_probs, upset_analysis)
        
        return enhanced_probs

    def _apply_qualitative_enhancements(self, probs: Dict, qualitative: Dict) -> Dict:
        """Apply qualitative intelligence enhancements"""
        boosts = qualitative.get('qualitative_boosts', {'home': 1.0, 'away': 1.0, 'draw': 1.0})
        
        # Apply boosts to 1X2 market
        probs['1X2']['Home Win'] *= boosts['home']
        probs['1X2']['Away Win'] *= boosts['away'] 
        probs['1X2']['Draw'] *= boosts['draw']
        
        # Normalize 1X2 probabilities
        total_1x2 = sum(probs['1X2'].values())
        for outcome in probs['1X2']:
            probs['1X2'][outcome] = (probs['1X2'][outcome] / total_1x2) * 100
        
        # Adjust Over/Under based on motivation (motivated teams attack more)
        motivation_score = qualitative.get('motivation_score', 0.0)
        if abs(motivation_score) > 0.1:
            # Higher motivation often leads to more attacking play
            ou_adjustment = 1.0 + (abs(motivation_score) * 0.3)
            probs['Over/Under']['Over 2.5'] = min(95, probs['Over/Under']['Over 2.5'] * ou_adjustment)
            probs['Over/Under']['Under 2.5'] = 100 - probs['Over/Under']['Over 2.5']
        
        # Adjust BTTS based on trap signals
        if 'public_overbet_favorite' in qualitative.get('trap_signals', []):
            # Trap games often see unexpected goals
            probs['BTTS']['Yes'] = min(90, probs['BTTS']['Yes'] * 1.15)
            probs['BTTS']['No'] = 100 - probs['BTTS']['Yes']
        
        return probs

    def _calculate_base_probabilities(self) -> Dict[str, Any]:
        """Calculate base probabilities using enhanced Poisson model"""
        try:
            # Extract goal data from the structure
            home_goals_data = self.data.get('home_goals_data', {})
            away_goals_data = self.data.get('away_goals_data', {})
            
            # Calculate averages if not provided
            home_goals_scored = home_goals_data.get('goals_scored', 0)
            home_goals_conceded = home_goals_data.get('goals_conceded', 0)
            away_goals_scored = away_goals_data.get('goals_scored', 0)
            away_goals_conceded = away_goals_data.get('goals_conceded', 0)
            
            # Calculate attack and defense strengths from raw goals
            home_attack = home_goals_scored / 6.0 if home_goals_scored > 0 else 1.0
            home_defense = home_goals_conceded / 6.0 if home_goals_conceded > 0 else 1.0
            away_attack = away_goals_scored / 6.0 if away_goals_scored > 0 else 1.0
            away_defense = away_goals_conceded / 6.0 if away_goals_conceded > 0 else 1.0
            
            # League adjustment factors
            league_factor = self._get_league_factor()
            home_advantage = 1.1  # 10% home advantage
            
            # Calculate expected goals
            home_xg = (home_attack * away_defense * home_advantage * league_factor)
            away_xg = (away_attack * home_defense * league_factor)
            
            # Apply H2H adjustments if available
            h2h_adjustment = self._calculate_h2h_adjustment()
            home_xg *= h2h_adjustment['home_boost']
            away_xg *= h2h_adjustment['away_boost']
            
            # Ensure reasonable bounds
            home_xg = max(0.1, min(4.0, home_xg))
            away_xg = max(0.1, min(4.0, away_xg))
            
            # Calculate probabilities using Poisson distribution
            max_goals = 8
            home_probs = [poisson.pmf(i, home_xg) for i in range(max_goals)]
            away_probs = [poisson.pmf(i, away_xg) for i in range(max_goals)]
            
            # 1X2 probabilities
            home_win_prob = 0
            draw_prob = 0
            away_win_prob = 0
            
            for i in range(max_goals):
                for j in range(max_goals):
                    prob = home_probs[i] * away_probs[j]
                    if i > j:
                        home_win_prob += prob
                    elif i == j:
                        draw_prob += prob
                    else:
                        away_win_prob += prob
            
            # Normalize to 100%
            total = home_win_prob + draw_prob + away_win_prob
            home_win_prob /= total
            draw_prob /= total
            away_win_prob /= total
            
            # Over/Under probabilities
            over_25_prob = 0
            under_25_prob = 0
            
            for i in range(max_goals):
                for j in range(max_goals):
                    prob = home_probs[i] * away_probs[j]
                    if i + j > 2.5:
                        over_25_prob += prob
                    else:
                        under_25_prob += prob
            
            # BTTS probabilities
            btts_yes = 0
            btts_no = 0
            
            for i in range(1, max_goals):
                for j in range(1, max_goals):
                    btts_yes += home_probs[i] * away_probs[j]
            
            btts_no = 1 - btts_yes
            
            return {
                '1X2': {
                    'Home Win': home_win_prob * 100,
                    'Draw': draw_prob * 100,
                    'Away Win': away_win_prob * 100
                },
                'Over/Under': {
                    'Over 2.5': over_25_prob * 100,
                    'Under 2.5': under_25_prob * 100
                },
                'BTTS': {
                    'Yes': btts_yes * 100,
                    'No': btts_no * 100
                },
                'goal_expectancy': {
                    'home_xg': home_xg,
                    'away_xg': away_xg,
                    'total_xg': home_xg + away_xg
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in base probability calculation: {e}")
            # Return safe defaults
            return {
                '1X2': {'Home Win': 33.3, 'Draw': 33.3, 'Away Win': 33.3},
                'Over/Under': {'Over 2.5': 50.0, 'Under 2.5': 50.0},
                'BTTS': {'Yes': 50.0, 'No': 50.0},
                'goal_expectancy': {'home_xg': 1.3, 'away_xg': 1.3, 'total_xg': 2.6}
            }

    def _calculate_goal_timing_intelligence(self) -> Dict[str, Any]:
        """Calculate goal timing intelligence using H2H goal data"""
        self.logger.info("🎯 Calculating Enhanced Goal Timing Intelligence...")
        
        try:
            # Check for H2H goal data first
            h2h_home_goals = self.data.get('h2h_home_goals', 0)
            h2h_away_goals = self.data.get('h2h_away_goals', 0)
            h2h_total_matches = self.data.get('h2h_aggregate', {}).get('total_matches', 0)
            
            # Check if we have sufficient H2H goal data for timing intelligence
            has_sufficient_h2h_goals = (
                h2h_total_matches > 0 and 
                (h2h_home_goals > 0 or h2h_away_goals > 0)
            )
            
            if has_sufficient_h2h_goals:
                self.logger.info("✅ H2H GOAL DATA FOUND - ACTIVATING TIMING INTELLIGENCE")
                
                # Calculate average goals from H2H data
                h2h_home_avg = h2h_home_goals / h2h_total_matches
                h2h_away_avg = h2h_away_goals / h2h_total_matches
                h2h_total_avg = (h2h_home_goals + h2h_away_goals) / h2h_total_matches
                
                # Use H2H data for timing patterns
                home_1h_ratio = 0.45  # Slightly more goals in 1H for home teams in H2H
                away_1h_ratio = 0.40  # Away teams score more evenly
                
                # Adjust based on H2H goal dominance
                if h2h_home_avg > h2h_away_avg * 1.5:
                    home_1h_ratio = 0.50  # Dominant home teams score early
                elif h2h_away_avg > h2h_home_avg * 1.5:
                    away_1h_ratio = 0.45  # Dominant away teams may score early
                
            else:
                self.logger.warning("❌ INSUFFICIENT H2H GOAL DATA - USING DEFAULT TIMING PATTERNS")
                # Fall back to default timing patterns based on recent form
                home_goals_data = self.data.get('home_goals_data', {})
                away_goals_data = self.data.get('away_goals_data', {})
                
                home_attack_strength = home_goals_data.get('goals_scored', 0) / 6.0
                away_attack_strength = away_goals_data.get('goals_scored', 0) / 6.0
                
                # Default timing ratios based on attack strength
                home_1h_ratio = 0.42 + (home_attack_strength * 0.08)
                away_1h_ratio = 0.38 + (away_attack_strength * 0.08)
                
                # Ensure reasonable bounds
                home_1h_ratio = max(0.35, min(0.65, home_1h_ratio))
                away_1h_ratio = max(0.35, min(0.65, away_1h_ratio))
            
            # Calculate timing probabilities
            base_probs = self._calculate_base_probabilities()
            home_xg = base_probs['goal_expectancy']['home_xg']
            away_xg = base_probs['goal_expectancy']['away_xg']
            
            # 1st half goal probabilities
            home_1h_xg = home_xg * home_1h_ratio
            away_1h_xg = away_xg * away_1h_ratio
            total_1h_xg = home_1h_xg + away_1h_xg
            
            # 2nd half goal probabilities
            home_2h_xg = home_xg * (1 - home_1h_ratio)
            away_2h_xg = away_xg * (1 - away_1h_ratio)
            total_2h_xg = home_2h_xg + away_2h_xg
            
            # Probability calculations
            prob_1h_goal = (1 - poisson.pmf(0, total_1h_xg)) * 100
            prob_2h_goal = (1 - poisson.pmf(0, total_2h_xg)) * 100
            
            # Late goals probability (75+ minutes)
            late_goal_ratio = 0.25
            late_goals_xg = total_2h_xg * late_goal_ratio
            prob_late_goals = (1 - poisson.pmf(0, late_goals_xg)) * 100
            
            # Determine scoring momentum
            if total_1h_xg > total_2h_xg * 1.2:
                scoring_momentum = 'front_loaded'
            elif total_2h_xg > total_1h_xg * 1.2:
                scoring_momentum = 'back_loaded'
            else:
                scoring_momentum = 'balanced'
            
            # Expected goal timing windows
            timing_windows = self._calculate_goal_timing_windows(total_1h_xg, total_2h_xg)
            
            # Key insights
            key_insights = self._generate_timing_insights(
                home_1h_ratio, away_1h_ratio, scoring_momentum, 
                has_sufficient_h2h_goals, h2h_total_avg if has_sufficient_h2h_goals else None
            )
            
            timing_intelligence = {
                'active': True,
                'data_source': 'h2h_goals' if has_sufficient_h2h_goals else 'recent_form',
                '1h_goal_probability': round(prob_1h_goal, 1),
                '2h_goal_probability': round(prob_2h_goal, 1),
                'late_goals_75plus_prob': round(prob_late_goals, 1),
                'scoring_momentum': scoring_momentum,
                'expected_goal_timing': timing_windows,
                'team_timing_analysis': {
                    'home_1h_ratio': round(home_1h_ratio, 3),
                    'away_1h_ratio': round(away_1h_ratio, 3),
                    'home_1h_xg': round(home_1h_xg, 2),
                    'home_2h_xg': round(home_2h_xg, 2),
                    'away_1h_xg': round(away_1h_xg, 2),
                    'away_2h_xg': round(away_2h_xg, 2)
                },
                'key_insights': key_insights,
                'h2h_goal_data_used': has_sufficient_h2h_goals
            }
            
            self.logger.info(f"✅ TIMING INTELLIGENCE ACTIVATED - Data Source: {timing_intelligence['data_source']}")
            return timing_intelligence
            
        except Exception as e:
            self.logger.error(f"Error in timing intelligence calculation: {e}")
            return {
                'active': False,
                'error': str(e),
                '1h_goal_probability': 45.0,
                '2h_goal_probability': 55.0,
                'late_goals_75plus_prob': 25.0,
                'scoring_momentum': 'balanced'
            }

    def _calculate_goal_timing_windows(self, total_1h_xg: float, total_2h_xg: float) -> Dict[str, str]:
        """Calculate expected goal timing windows"""
        segments = {
            '0-15': total_1h_xg * 0.25,
            '16-30': total_1h_xg * 0.35,
            '31-45': total_1h_xg * 0.40,
            '46-60': total_2h_xg * 0.30,
            '61-75': total_2h_xg * 0.35,
            '76-90': total_2h_xg * 0.35
        }
        
        first_goal_window = max(segments, key=segments.get)
        second_goal_window = max([k for k in segments.keys() if k != first_goal_window], 
                               key=lambda k: segments[k])
        late_goal_window = '76-90' if segments['76-90'] > 0.1 else '61-75'
        
        return {
            'first_goal_window': first_goal_window,
            'second_goal_window': second_goal_window,
            'late_goal_window': late_goal_window,
            'segments': {k: round(v, 3) for k, v in segments.items()}
        }

    def _generate_timing_insights(self, home_1h_ratio: float, away_1h_ratio: float, 
                                scoring_momentum: str, h2h_data_used: bool, h2h_avg_goals: float = None) -> List[str]:
        """Generate key timing insights"""
        insights = []
        
        if h2h_data_used and h2h_avg_goals:
            insights.append(f"H2H data shows {h2h_avg_goals:.1f} avg goals - using historical timing patterns")
        
        if home_1h_ratio > 0.5:
            insights.append(f"Home team tends to score early ({home_1h_ratio:.0%} of goals in 1H)")
        elif home_1h_ratio < 0.4:
            insights.append(f"Home team tends to score late ({(1-home_1h_ratio):.0%} of goals in 2H)")
            
        if away_1h_ratio > 0.45:
            insights.append(f"Away team shows early scoring tendency ({away_1h_ratio:.0%} in 1H)")
            
        if scoring_momentum == 'front_loaded':
            insights.append("Expect early match intensity with goals in first half")
        elif scoring_momentum == 'back_loaded':
            insights.append("Game likely to open up in second half with late goals")
        else:
            insights.append("Balanced goal distribution expected throughout match")
            
        return insights

    def _detect_pattern_intelligence(self) -> Dict[str, Any]:
        """Detect patterns in match data"""
        patterns_detected = []
        total_influence_strength = 0.0
        
        # H2H Pattern detection
        h2h_pattern = self.data.get('h2h_pattern')
        if h2h_pattern and h2h_pattern != "No H2H Data":
            pattern_strength = 0.3
            patterns_detected.append({
                'type': 'h2h_dominance_pattern',
                'direction': 'home' if 'Home' in h2h_pattern else 'away' if 'Away' in h2h_pattern else 'neutral',
                'strength': pattern_strength,
                'confidence': 0.7,
                'evidence': self.data.get('h2h_evidence', 'H2H pattern detected')
            })
            total_influence_strength += pattern_strength
        
        # Goal scoring pattern
        home_goals_data = self.data.get('home_goals_data', {})
        away_goals_data = self.data.get('away_goals_data', {})
        
        home_scoring_freq = home_goals_data.get('scoring_frequency', 0)
        away_scoring_freq = away_goals_data.get('scoring_frequency', 0)
        
        if home_scoring_freq >= 70 and away_scoring_freq >= 70:
            patterns_detected.append({
                'type': 'high_btts_likelihood',
                'direction': 'both',
                'strength': 0.25,
                'confidence': 0.8,
                'evidence': f"Both teams score frequently (Home: {home_scoring_freq}%, Away: {away_scoring_freq}%)"
            })
            total_influence_strength += 0.25
        
        # League position pattern
        home_standing = self.data.get('home_standing', [10, 0, 0, 0])
        away_standing = self.data.get('away_standing', [10, 0, 0, 0])
        
        if len(home_standing) > 0 and len(away_standing) > 0:
            home_pos, away_pos = home_standing[0], away_standing[0]
            if abs(home_pos - away_pos) >= 8:
                patterns_detected.append({
                    'type': 'table_disparity',
                    'direction': 'home' if home_pos < away_pos else 'away',
                    'strength': 0.2,
                    'confidence': 0.6,
                    'evidence': f"Significant table difference: {home_pos} vs {away_pos}"
                })
                total_influence_strength += 0.2
        
        return {
            'patterns_detected': patterns_detected,
            'pattern_count': len(patterns_detected),
            'total_influence_strength': total_influence_strength
        }

    def _detect_contextual_upset_potential(self) -> Dict[str, Any]:
        """Detect contextual upset potential using market odds"""
        self.logger.info("🎯 Analyzing Contextual Upset Potential...")
        
        try:
            odds_1x2 = self.data.get('odds_1x2', [])
            
            if not odds_1x2 or len(odds_1x2) != 3:
                self.logger.warning("❌ NO MARKET ODDS DATA - UPSET DETECTION INACTIVE")
                return {
                    'upset_detected': False,
                    'upset_level': 'none',
                    'reason': 'No market odds data provided',
                    'home_boost': 1.0,
                    'away_boost': 1.0,
                    'total_upset_score': 0.0,
                    'factor_count': 0,
                    'active': False
                }
            
            self.logger.info("✅ MARKET ODDS FOUND - ACTIVATING UPSET DETECTION")
            
            # Extract odds
            home_odds, draw_odds, away_odds = odds_1x2
            
            # Calculate implied probabilities
            home_implied = 1 / home_odds
            away_implied = 1 / away_odds
            draw_implied = 1 / draw_odds
            
            # Get our model probabilities
            base_probs = self._calculate_base_probabilities()
            home_prob = base_probs['1X2']['Home Win'] / 100
            away_prob = base_probs['1X2']['Away Win'] / 100
            
            # Calculate market mispricing
            home_mispricing = home_prob - home_implied
            away_mispricing = away_prob - away_implied
            
            # Upset factors scoring
            upset_factors = []
            upset_score = 0.0
            
            # Factor 1: Significant market mispricing (>= 5%)
            if home_mispricing >= 0.05:
                upset_score += 0.3
                upset_factors.append({
                    'factor': 'market_mispricing',
                    'score': 0.3,
                    'reason': f'Market undervalues home team by {home_mispricing:.1%}'
                })
            
            if away_mispricing >= 0.05:
                upset_score += 0.3
                upset_factors.append({
                    'factor': 'market_mispricing', 
                    'score': 0.3,
                    'reason': f'Market undervalues away team by {away_mispricing:.1%}'
                })
            
            # Factor 2: Match importance context
            match_importance = self.data.get('match_importance', 'Normal League')
            importance_boost = {
                'Relegation Battle': 0.2,
                'Title Decider': 0.15,
                'Derby/Local Rivalry': 0.25,
                'Cup Final': 0.2,
                'European Qualification': 0.1
            }
            
            if match_importance in importance_boost:
                upset_score += importance_boost[match_importance]
                upset_factors.append({
                    'factor': 'match_context',
                    'score': importance_boost[match_importance],
                    'reason': f'High-stakes context: {match_importance}'
                })
            
            # Factor 3: Home advantage in important matches
            venue_context = self.data.get('venue_context', 'Normal')
            if venue_context in ['European Night', 'Rival Territory']:
                upset_score += 0.15
                upset_factors.append({
                    'factor': 'venue_impact',
                    'score': 0.15,
                    'reason': f'Special venue context: {venue_context}'
                })
            
            # Factor 4: Recent form vs market expectation
            home_goals_data = self.data.get('home_goals_data', {})
            away_goals_data = self.data.get('away_goals_data', {})
            
            home_goals_scored = home_goals_data.get('goals_scored', 0)
            away_goals_scored = away_goals_data.get('goals_scored', 0)
            
            if home_goals_scored >= 8 and home_mispricing > 0.02:
                upset_score += 0.1
                upset_factors.append({
                    'factor': 'recent_form',
                    'score': 0.1,
                    'reason': f'Home team scoring form ({home_goals_scored} goals) not reflected in odds'
                })
            
            # Determine upset level
            if upset_score >= 0.5:
                upset_level = 'high'
                home_boost = 1.15
                away_boost = 0.95
            elif upset_score >= 0.3:
                upset_level = 'medium' 
                home_boost = 1.08
                away_boost = 0.98
            elif upset_score >= 0.1:
                upset_level = 'low'
                home_boost = 1.03
                away_boost = 0.99
            else:
                upset_level = 'none'
                home_boost = 1.0
                away_boost = 1.0
            
            upset_detected = upset_level != 'none'
            
            result = {
                'upset_detected': upset_detected,
                'upset_level': upset_level,
                'home_boost': home_boost,
                'away_boost': away_boost,
                'total_upset_score': round(upset_score, 3),
                'factor_count': len(upset_factors),
                'factors': upset_factors,
                'market_analysis': {
                    'home_implied_prob': round(home_implied * 100, 1),
                    'away_implied_prob': round(away_implied * 100, 1),
                    'home_mispricing': round(home_mispricing * 100, 1),
                    'away_mispricing': round(away_mispricing * 100, 1)
                },
                'active': True
            }
            
            if upset_detected:
                self.logger.info(f"✅ UPSET POTENTIAL DETECTED - Level: {upset_level.upper()}, Score: {upset_score:.3f}")
            else:
                self.logger.info("✅ UPSET ANALYSIS COMPLETE - No significant upset potential")
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error in upset detection: {e}")
            return {
                'upset_detected': False,
                'upset_level': 'none',
                'reason': f'Error: {str(e)}',
                'home_boost': 1.0,
                'away_boost': 1.0,
                'total_upset_score': 0.0,
                'factor_count': 0,
                'active': False
            }

    def _apply_timing_enhancements(self, probs: Dict, timing: Dict) -> Dict:
        """Apply timing intelligence enhancements"""
        ou_probs = probs['Over/Under']
        timing_1h = timing['1h_goal_probability'] / 100
        timing_2h = timing['2h_goal_probability'] / 100
        
        if timing_1h > 0.5 and timing_2h > 0.5:
            ou_probs['Over 2.5'] = min(95, ou_probs['Over 2.5'] * 1.1)
            ou_probs['Under 2.5'] = 100 - ou_probs['Over 2.5']
        
        elif timing['scoring_momentum'] == 'back_loaded':
            ou_probs['Over 2.5'] = min(90, ou_probs['Over 2.5'] * 1.05)
            ou_probs['Under 2.5'] = 100 - ou_probs['Over 2.5']
        
        probs['Over/Under'] = ou_probs
        return probs

    def _apply_pattern_enhancements(self, probs: Dict, patterns: Dict) -> Dict:
        """Apply pattern-based enhancements"""
        total_influence = patterns['total_influence_strength']
        
        influence_factor = 1.0 + (total_influence * 0.1)
        
        for market in ['1X2', 'Over/Under', 'BTTS']:
            for outcome in probs[market]:
                probs[market][outcome] = min(95, max(5, probs[market][outcome] * influence_factor))
        
        for market in probs:
            if market != 'goal_expectancy':
                total = sum(probs[market].values())
                for outcome in probs[market]:
                    probs[market][outcome] = (probs[market][outcome] / total) * 100
        
        return probs

    def _apply_upset_enhancements(self, probs: Dict, upset: Dict) -> Dict:
        """Apply upset analysis enhancements"""
        home_boost = upset['home_boost']
        away_boost = upset['away_boost']
        
        home_win = probs['1X2']['Home Win'] * home_boost
        away_win = probs['1X2']['Away Win'] * away_boost
        draw = probs['1X2']['Draw'] * 1.0
        
        total = home_win + draw + away_win
        probs['1X2']['Home Win'] = (home_win / total) * 100
        probs['1X2']['Draw'] = (draw / total) * 100
        probs['1X2']['Away Win'] = (away_win / total) * 100
        
        return probs

    def _generate_betting_recommendations(self, final_probs: Dict, timing_intelligence: Dict, 
                                       upset_analysis: Dict, qualitative_analysis: Dict) -> Dict[str, Any]:
        """Generate institutional betting recommendations with Qualitative Intelligence"""
        opportunities = []
        timing_enhanced_bets = []
        upset_aware_bets = []
        qualitative_enhanced_bets = []
        
        # Market odds for comparison
        odds_1x2 = self.data.get('odds_1x2', [2.0, 3.0, 2.0])
        odds_ou = self.data.get('odds_over_under', [2.0, 1.85])
        odds_btts = self.data.get('odds_btts', [1.8, 2.0])
        
        # 1X2 Value Opportunities
        home_implied = 1 / odds_1x2[0] * 100 if odds_1x2[0] > 0 else 33.3
        draw_implied = 1 / odds_1x2[1] * 100 if odds_1x2[1] > 0 else 33.3
        away_implied = 1 / odds_1x2[2] * 100 if odds_1x2[2] > 0 else 33.3
        
        home_edge = final_probs['1X2']['Home Win'] - home_implied
        away_edge = final_probs['1X2']['Away Win'] - away_implied
        draw_edge = final_probs['1X2']['Draw'] - draw_implied
        
        # Home win opportunity
        if home_edge >= 5:
            bet = self._create_bet_recommendation(
                '1X2', 'Home Win', final_probs['1X2']['Home Win'], 
                home_implied, home_edge, odds_1x2[0], 'standard'
            )
            opportunities.append(bet)
            
            if upset_analysis.get('upset_detected', False) and upset_analysis.get('home_boost', 1.0) > 1.0:
                upset_bet = bet.copy()
                upset_bet['bet_type'] = 'upset_aware'
                upset_bet['reasoning'] += f" | Upset context boosts home probability"
                upset_bet['upset_factors'] = [f['reason'] for f in upset_analysis.get('factors', [])[:2]]
                upset_aware_bets.append(upset_bet)
        
        # Away win opportunity  
        if away_edge >= 5:
            bet = self._create_bet_recommendation(
                '1X2', 'Away Win', final_probs['1X2']['Away Win'],
                away_implied, away_edge, odds_1x2[2], 'standard'
            )
            opportunities.append(bet)
        
        # Over/Under opportunities
        over_implied = 1 / odds_ou[0] * 100 if odds_ou[0] > 0 else 50.0
        under_implied = 1 / odds_ou[1] * 100 if odds_ou[1] > 0 else 50.0
        
        over_edge = final_probs['Over/Under']['Over 2.5'] - over_implied
        under_edge = final_probs['Over/Under']['Under 2.5'] - under_implied
        
        if over_edge >= 5:
            bet = self._create_bet_recommendation(
                'Over/Under', 'Over 2.5', final_probs['Over/Under']['Over 2.5'],
                over_implied, over_edge, odds_ou[0], 'standard'
            )
            opportunities.append(bet)
            
            if timing_intelligence.get('active', False):
                if timing_intelligence['scoring_momentum'] == 'back_loaded':
                    timing_bet = bet.copy()
                    timing_bet['bet_type'] = 'timing_enhanced'
                    timing_bet['reasoning'] += f" | Late goal pattern ({timing_intelligence['2h_goal_probability']}% 2H goals) supports Over"
                    timing_enhanced_bets.append(timing_bet)
        
        if under_edge >= 5:
            bet = self._create_bet_recommendation(
                'Over/Under', 'Under 2.5', final_probs['Over/Under']['Under 2.5'],
                under_implied, under_edge, odds_ou[1], 'standard'
            )
            opportunities.append(bet)
        
        # BTTS opportunities
        yes_implied = 1 / odds_btts[0] * 100 if odds_btts[0] > 0 else 50.0
        no_implied = 1 / odds_btts[1] * 100 if odds_btts[1] > 0 else 50.0
        
        yes_edge = final_probs['BTTS']['Yes'] - yes_implied
        no_edge = final_probs['BTTS']['No'] - no_implied
        
        if yes_edge >= 5:
            bet = self._create_bet_recommendation(
                'BTTS', 'Yes', final_probs['BTTS']['Yes'],
                yes_implied, yes_edge, odds_btts[0], 'standard'
            )
            opportunities.append(bet)
        
        if no_edge >= 5:
            bet = self._create_bet_recommendation(
                'BTTS', 'No', final_probs['BTTS']['No'],
                no_implied, no_edge, odds_btts[1], 'standard'
            )
            opportunities.append(bet)
        
        # Add qualitative-enhanced bets
        if qualitative_analysis.get('active', False):
            qual_enhanced = self._create_qualitative_enhanced_bets(
                final_probs, qualitative_analysis, odds_1x2, odds_ou, odds_btts
            )
            qualitative_enhanced_bets.extend(qual_enhanced)
        
        return {
            'all_opportunities': opportunities,
            'timing_enhanced_bets': timing_enhanced_bets,
            'upset_aware_bets': upset_aware_bets,
            'qualitative_enhanced_bets': qualitative_enhanced_bets,
            'total_opportunities': len(opportunities),
            'enhanced_opportunities': len(timing_enhanced_bets) + len(upset_aware_bets) + len(qualitative_enhanced_bets)
        }
    
    def _create_qualitative_enhanced_bets(self, final_probs: Dict, qualitative_analysis: Dict,
                                        odds_1x2: List, odds_ou: List, odds_btts: List) -> List[Dict]:
        """Create bets enhanced by qualitative intelligence"""
        enhanced_bets = []
        motivation_score = qualitative_analysis.get('motivation_score', 0.0)
        trap_signals = qualitative_analysis.get('trap_signals', [])
        insights = qualitative_analysis.get('key_insights', [])
        
        # High motivation home team opportunities
        if motivation_score > 0.2:
            home_implied = 1 / odds_1x2[0] * 100 if odds_1x2[0] > 0 else 33.3
            home_edge = final_probs['1X2']['Home Win'] - home_implied
            
            if home_edge >= 3:
                bet = self._create_bet_recommendation(
                    '1X2', 'Home Win', final_probs['1X2']['Home Win'],
                    home_implied, home_edge, odds_1x2[0], 'qualitative_enhanced'
                )
                bet['qualitative_insights'] = insights
                bet['reasoning'] += " | " + " | ".join(insights[:2])
                enhanced_bets.append(bet)
        
        # Trap game opportunities
        if 'public_overbet_favorite' in trap_signals:
            home_implied = 1 / odds_1x2[0] * 100 if odds_1x2[0] > 0 else 33.3
            home_edge = final_probs['1X2']['Home Win'] - home_implied
            
            if home_edge >= 2:
                bet = self._create_bet_recommendation(
                    '1X2', 'Home Win', final_probs['1X2']['Home Win'],
                    home_implied, home_edge, odds_1x2[0], 'trap_game_opportunity'
                )
                bet['qualitative_insights'] = ["TRAP GAME: Public overbetting favorite"]
                bet['reasoning'] += " | TRAP GAME: Value on underestimated home team"
                enhanced_bets.append(bet)
        
        return enhanced_bets

    def _create_bet_recommendation(self, market: str, selection: str, our_prob: float, 
                                 implied_prob: float, edge: float, odds: float, bet_type: str) -> Dict[str, Any]:
        """Create a standardized bet recommendation"""
        expected_value_float = (edge / 100) * odds - 1
        expected_value_percent = expected_value_float * 100
        
        recommended_stake = min(5.0, max(1.0, (edge / 5.0)))
        
        confidence = 'High' if edge >= 10 else 'Medium' if edge >= 7 else 'Low'
        
        reasoning = f"Our model probability {our_prob:.1f}% vs market implied {implied_prob:.1f}%"
        
        return {
            'market': market,
            'selection': selection,
            'probability': round(our_prob, 1),
            'implied_prob': round(implied_prob, 1),
            'edge': f"+{edge:.1f}%",
            'odds': odds,
            'expected_value': expected_value_percent,
            'recommended_stake': round(recommended_stake, 1),
            'confidence': confidence,
            'reasoning': reasoning,
            'bet_type': bet_type
        }

    def _analyze_market_efficiency(self) -> Dict[str, Any]:
        """Analyze market efficiency and overround"""
        odds_1x2 = self.data.get('odds_1x2', [2.0, 3.0, 2.0])
        
        if len(odds_1x2) == 3:
            home_implied = 1 / odds_1x2[0]
            draw_implied = 1 / odds_1x2[1] 
            away_implied = 1 / odds_1x2[2]
            
            overround = (home_implied + draw_implied + away_implied - 1) * 100
            efficiency = 100 - overround
            
            return {
                'overround': round(overround, 2),
                'efficiency': round(efficiency, 2),
                'efficiency_score': efficiency / 100,
                'market_quality': 'High' if efficiency >= 97 else 'Medium' if efficiency >= 94 else 'Low'
            }
        
        return {
            'overround': 0.0,
            'efficiency': 100.0,
            'efficiency_score': 1.0,
            'market_quality': 'Unknown'
        }

    def _calculate_risk_assessment(self, probs: Dict, qualitative_analysis: Dict) -> Dict[str, Any]:
        """Calculate risk assessment metrics with qualitative considerations"""
        confidence = self._calculate_confidence_score(qualitative_analysis)
        
        base_risk = 'Low' if confidence >= 80 else 'Medium' if confidence >= 60 else 'High'
        
        if qualitative_analysis.get('trap_signals'):
            if 'public_overbet_favorite' in qualitative_analysis['trap_signals']:
                base_risk = 'High' if base_risk == 'Low' else 'Very High' if base_risk == 'Medium' else 'Extreme'
        
        return {
            'risk_level': base_risk,
            'confidence': 'High' if confidence >= 70 else 'Medium' if confidence >= 50 else 'Low',
            'recommended_max_stake': 3.0 if base_risk == 'High' else 5.0,
            'kelly_fraction': 0.15 if base_risk == 'High' else 0.25,
            'var_95': 0.05 if base_risk == 'High' else 0.02,
            'max_drawdown': 0.08 if base_risk == 'High' else 0.05,
            'qualitative_risk_factors': qualitative_analysis.get('risk_alerts', [])
        }

    def _calculate_uncertainty(self, probs: Dict, qualitative_analysis: Dict) -> Dict[str, Any]:
        """Calculate uncertainty metrics with qualitative considerations"""
        confidence = self._calculate_confidence_score(qualitative_analysis)
        
        base_uncertainty = max(2.0, 10 - (confidence / 10))
        
        if qualitative_analysis.get('motivation_score', 0) != 0:
            base_uncertainty *= 1.2
        
        if qualitative_analysis.get('trap_signals'):
            base_uncertainty *= 1.3
        
        home_win_prob = probs['1X2']['Home Win']
        error_margin = (base_uncertainty / 100) * home_win_prob
        
        lower_68 = max(0, home_win_prob - error_margin)
        upper_68 = min(100, home_win_prob + error_margin)
        
        lower_95 = max(0, home_win_prob - (2 * error_margin))
        upper_95 = min(100, home_win_prob + (2 * error_margin))
        
        return {
            'confidence_score': confidence,
            'standard_error': base_uncertainty,
            'home_win_68_interval': (round(lower_68, 1), round(upper_68, 1)),
            'home_win_95_interval': (round(lower_95, 1), round(upper_95, 1)),
            'data_quality_score': self._assess_data_quality(),
            'qualitative_uncertainty_factors': qualitative_analysis.get('trap_signals', [])
        }

    def _analyze_value_opportunities(self, probs: Dict) -> Dict[str, Any]:
        """Analyze value betting opportunities"""
        opportunities = []
        
        betting_recs = self._generate_betting_recommendations(probs, {}, {}, {})
        
        for bet in betting_recs['all_opportunities']:
            opportunities.append({
                'market': bet['market'],
                'selection': bet['selection'],
                'our_prob': bet['probability'],
                'implied_prob': bet['implied_prob'],
                'edge': float(bet['edge'].replace('+', '').replace('%', '')),
                'expected_value': bet['expected_value'],
                'recommended_stake': bet['recommended_stake'],
                'confidence': bet['confidence'],
                'reasoning': [bet['reasoning']]
            })
        
        return {
            'value_opportunities': opportunities,
            'total_opportunities': len(opportunities),
            'best_edge': max([opp['edge'] for opp in opportunities]) if opportunities else 0
        }

    def _calculate_confidence_score(self, qualitative_analysis: Dict = None) -> float:
        """Calculate overall model confidence score with qualitative considerations"""
        scores = []
        
        # Data quality score (0-100)
        data_quality = self._assess_data_quality()
        scores.append(data_quality * 0.25)
        
        # Pattern confidence (0-100)
        pattern_confidence = self._assess_pattern_confidence()
        scores.append(pattern_confidence * 0.20)
        
        # Market efficiency (0-100)
        market_eff = self._analyze_market_efficiency()
        scores.append(market_eff['efficiency'] * 0.15)
        
        # Enhancement activation (0-100)
        enhancement_score = sum([
            20 if self.enhancements_active['timing_intelligence'] else 0,
            20 if self.enhancements_active['upset_detection'] else 0,
            20 if self.enhancements_active['pattern_detection'] else 0,
            20 if self.enhancements_active['market_analysis'] else 0,
            20 if self.enhancements_active['qualitative_intelligence'] else 0
        ])
        scores.append(enhancement_score * 0.20)
        
        # QUALITATIVE CONFIDENCE BOOSTER (NEW - 20% weight)
        if qualitative_analysis and qualitative_analysis.get('active', False):
            qual_confidence = self._calculate_qualitative_confidence(qualitative_analysis)
            scores.append(qual_confidence * 0.20)
        else:
            scores.append(0.0)
        
        return min(100, sum(scores))
    
    def _calculate_qualitative_confidence(self, qualitative_analysis: Dict) -> float:
        """Calculate confidence boost from qualitative analysis"""
        confidence = 0.0
        
        motivation_score = abs(qualitative_analysis.get('motivation_score', 0.0))
        if motivation_score > 0.2:
            confidence += 40
        elif motivation_score > 0.1:
            confidence += 20
        
        if qualitative_analysis.get('trap_signals'):
            confidence += 30
        
        factor_count = len(qualitative_analysis.get('key_insights', []))
        confidence += min(30, factor_count * 10)
        
        return confidence

    def _assess_data_quality(self) -> float:
        """Assess overall data quality"""
        score = 0
        max_score = 100
        
        # Home goals data (20 points)
        home_goals = self.data.get('home_goals_data', {})
        if home_goals.get('goals_scored', 0) > 0 or home_goals.get('goals_conceded', 0) > 0:
            score += 20
        
        # Away goals data (20 points)
        away_goals = self.data.get('away_goals_data', {})
        if away_goals.get('goals_scored', 0) > 0 or away_goals.get('goals_conceded', 0) > 0:
            score += 20
        
        # H2H data (20 points)
        h2h_aggregate = self.data.get('h2h_aggregate', {})
        if h2h_aggregate.get('total_matches', 0) > 0:
            score += 20
            
        # H2H GOAL DATA BONUS (20 points)
        h2h_home_goals = self.data.get('h2h_home_goals', 0)
        h2h_away_goals = self.data.get('h2h_away_goals', 0)
        if h2h_home_goals > 0 or h2h_away_goals > 0:
            score += 20
        
        # Market odds (20 points)
        if self.data.get('odds_1x2') and len(self.data['odds_1x2']) == 3:
            score += 20
        
        return score

    def _assess_pattern_confidence(self) -> float:
        """Assess pattern detection confidence"""
        pattern_intel = self._detect_pattern_intelligence()
        pattern_count = pattern_intel['pattern_count']
        influence_strength = pattern_intel['total_influence_strength']
        
        confidence = min(100, (pattern_count * 20) + (influence_strength * 100))
        
        return confidence

    def _determine_market_regime(self) -> str:
        """Determine current market regime"""
        efficiency = self._analyze_market_efficiency()
        
        if efficiency['efficiency'] >= 97:
            return 'efficient'
        elif efficiency['efficiency'] >= 94:
            return 'normal'
        else:
            return 'inefficient'

    def _get_league_factor(self) -> float:
        """Get league-specific adjustment factor"""
        league_type = self.data.get('league_type', 'Standard')
        
        factors = {
            'English Premier League': 1.0,
            'Bundesliga': 1.1,
            'La Liga': 0.9,
            'Serie A': 0.95,
            'Champions League': 1.05,
            'International': 0.9,
            'Standard': 1.0
        }
        
        return factors.get(league_type, 1.0)

    def _calculate_h2h_adjustment(self) -> Dict[str, float]:
        """Calculate H2H-based adjustments"""
        h2h_aggregate = self.data.get('h2h_aggregate', {})
        total_matches = h2h_aggregate.get('total_matches', 0)
        
        if total_matches == 0:
            return {'home_boost': 1.0, 'away_boost': 1.0}
        
        home_wins = h2h_aggregate.get('home_wins', 0)
        away_wins = h2h_aggregate.get('away_wins', 0)
        
        home_win_rate = home_wins / total_matches
        away_win_rate = away_wins / total_matches
        
        home_boost = 1.0 + (home_win_rate - 0.33) * 0.3
        away_boost = 1.0 + (away_win_rate - 0.33) * 0.3
        
        return {
            'home_boost': max(0.9, min(1.1, home_boost)),
            'away_boost': max(0.9, min(1.1, away_boost))
        }

    def _log_enhancement_status(self):
        """Log the status of all enhancements including Qualitative Intelligence"""
        status = self.enhancements_active
        self.logger.info("🎯 ENHANCEMENT STATUS REPORT:")
        self.logger.info(f"   Qualitative Intelligence: {'✅ ACTIVE' if status['qualitative_intelligence'] else '❌ INACTIVE'}")
        self.logger.info(f"   Timing Intelligence: {'✅ ACTIVE' if status['timing_intelligence'] else '❌ INACTIVE'}")
        self.logger.info(f"   Upset Detection: {'✅ ACTIVE' if status['upset_detection'] else '❌ INACTIVE'}")
        self.logger.info(f"   Pattern Detection: {'✅ ACTIVE' if status['pattern_detection'] else '❌ INACTIVE'}")
        self.logger.info(f"   Market Analysis: {'✅ ACTIVE' if status['market_analysis'] else '❌ INACTIVE'}")
        
        if status['qualitative_intelligence']:
            qual = getattr(self, 'qualitative_analysis', {})
            if qual:
                self.logger.info(f"   Qualitative Motivation Score: {qual.get('motivation_score', 0):.3f}")
                self.logger.info(f"   Trap Signals: {len(qual.get('trap_signals', []))}")

# Example usage
def demonstrate_qualitative_edge():
    """Demonstrate how qualitative intelligence catches games that fool other models"""
    
    match_data = {
        'home_team': 'Angers',
        'away_team': 'Monaco',
        'home_goals_data': {
            'goals_scored': 2,
            'goals_conceded': 11,
            'scoring_frequency': 33.3
        },
        'away_goals_data': {
            'goals_scored': 13,
            'goals_conceded': 11,
            'scoring_frequency': 83.3
        },
        'home_standing': [17, 7, 6, -5],
        'away_standing': [5, 10, 6, 6],
        'h2h_aggregate': {
            'total_matches': 6,
            'home_wins': 0,
            'away_wins': 5,
            'draws': 1
        },
        'h2h_home_goals': 3,
        'h2h_away_goals': 13,
        'odds_1x2': [2.33, 3.44, 3.33],
        'odds_over_under': [2.00, 1.85],
        'odds_btts': [1.79, 2.05],
        'league_type': 'Ligue 1',
        'match_importance': 'Normal League',
        'venue_context': 'Normal'
    }
    
    engine = ProfessionalPredictionEngine(match_data)
    predictions = engine.generate_all_predictions()
    
    return predictions

# Run the demonstration
if __name__ == "__main__":
    results = demonstrate_qualitative_edge()
    print("🎯 Qualitative Intelligence in Action:")
    print(f"Final 1X2 Probabilities: {results['final_probabilities']['1X2']}")
    print(f"Qualitative Insights: {results['qualitative_intelligence'].get('key_insights', [])}")
