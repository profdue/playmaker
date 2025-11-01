"""
PROFESSIONAL PREDICTION ORCHESTRATOR
Properly integrates pattern intelligence with core statistical engine
MAINTAINS FULL COMPATIBILITY with existing streamlit_app.py
"""
import logging
from typing import Dict, Any, List, Optional
from prediction_engine import AdvancedFootballPredictor

logger = logging.getLogger(__name__)

class ProfessionalPredictionOrchestrator:
    def __init__(self, pattern_influence: float = 0.12):  # Conservative professional weighting
        self.pattern_influence = min(0.20, max(0.05, pattern_influence))  # 5-20% range
        self.core_engine_class = AdvancedFootballPredictor
        logger.info(f"🎯 Initialized ProfessionalPredictionOrchestrator with {pattern_influence*100:.1f}% pattern influence")
        
    def generate_all_predictions(self, match_data: Dict) -> Dict[str, Any]:
        """
        PROFESSIONAL ORCHESTRATION: Blends statistical engine with pattern intelligence
        MAINTAINS EXACT SAME INTERFACE as existing prediction_engine
        """
        try:
            # Step 1: Generate core predictions with built-in pattern detection
            logger.info("📊 Generating institutional-grade predictions...")
            core_engine = self.core_engine_class(match_data)
            core_predictions = core_engine.generate_comprehensive_analysis()
            
            # Step 2: Check if core engine already detected patterns
            core_patterns = core_predictions.get('pattern_intelligence', {})
            core_pattern_count = core_patterns.get('pattern_count', 0)
            
            if core_pattern_count > 0:
                # Step 3: ENHANCE existing pattern integration (don't override)
                enhanced_predictions = self._professionally_enhance_predictions(
                    core_predictions, 
                    core_patterns,
                    match_data
                )
                logger.info(f"✅ Professionally enhanced with {core_pattern_count} core patterns")
                return enhanced_predictions
            else:
                # Step 4: Apply CONSERVATIVE external pattern analysis if no core patterns
                external_patterns = self._analyze_external_patterns(match_data, core_predictions)
                if external_patterns['pattern_count'] > 0:
                    conservatively_enhanced = self._conservatively_apply_external_patterns(
                        core_predictions, 
                        external_patterns
                    )
                    logger.info(f"🔄 Conservatively applied {external_patterns['pattern_count']} external patterns")
                    return conservatively_enhanced
                else:
                    logger.info("ℹ️ No meaningful patterns detected - using pure statistical predictions")
                    return core_predictions
                
        except Exception as e:
            logger.error(f"❌ Orchestration failed: {e}")
            # Professional fallback - return core predictions without enhancement
            core_engine = self.core_engine_class(match_data)
            return core_engine.generate_comprehensive_analysis()
    
    def _professionally_enhance_predictions(self, core_predictions: Dict, 
                                          core_patterns: Dict, 
                                          match_data: Dict) -> Dict:
        """
        PROFESSIONAL ENHANCEMENT: Respects core engine's pattern detection
        while providing intelligent calibration
        """
        enhanced = core_predictions.copy()
        
        # Extract core pattern adjustment
        core_adjustment = core_patterns.get('net_adjustment', {'home': 0, 'away': 0, 'draw': 0})
        pattern_count = core_patterns.get('pattern_count', 0)
        
        # Calculate professional enhancement factor based on pattern quality
        enhancement_factor = self._calculate_enhancement_factor(core_patterns, match_data)
        
        # Apply CALIBRATED enhancement to 1X2 probabilities
        if 'probabilities' in enhanced and 'match_outcomes' in enhanced['probabilities']:
            pred_1x2 = enhanced['probabilities']['match_outcomes']
            
            # Convert to probability space (0-1)
            home_core = pred_1x2.get('home_win', 33.3) / 100.0
            draw_core = pred_1x2.get('draw', 33.3) / 100.0
            away_core = pred_1x2.get('away_win', 33.3) / 100.0
            
            # Apply CALIBRATED pattern enhancement
            home_enhanced = home_core + (core_adjustment['home'] * enhancement_factor)
            draw_enhanced = draw_core + (core_adjustment['draw'] * enhancement_factor)
            away_enhanced = away_core + (core_adjustment['away'] * enhancement_factor)
            
            # Professional renormalization
            home_enhanced, draw_enhanced, away_enhanced = self._professional_renormalize(
                home_enhanced, draw_enhanced, away_enhanced
            )
            
            # Update predictions
            enhanced['probabilities']['match_outcomes']['home_win'] = round(home_enhanced * 100, 1)
            enhanced['probabilities']['match_outcomes']['draw'] = round(draw_enhanced * 100, 1)
            enhanced['probabilities']['match_outcomes']['away_win'] = round(away_enhanced * 100, 1)
            
            # ENHANCE confidence score based on pattern quality
            if pattern_count > 0:
                original_confidence = enhanced.get('confidence_score', 50)
                pattern_boost = min(15, pattern_count * 2)  # Max 15% boost
                enhanced_confidence = min(95, original_confidence + pattern_boost)
                enhanced['confidence_score'] = round(enhanced_confidence, 1)
                
                # Update uncertainty intervals to reflect increased confidence
                enhanced = self._adjust_uncertainty_for_patterns(enhanced, pattern_count)
        
        # Add professional orchestration metadata
        enhanced['orchestration_metadata'] = {
            'enhancement_type': 'core_pattern_integration',
            'enhancement_factor': round(enhancement_factor, 3),
            'pattern_count': pattern_count,
            'confidence_boost': pattern_count * 2,
            'notes': 'Professionally calibrated pattern integration'
        }
        
        return enhanced
    
    def _calculate_enhancement_factor(self, core_patterns: Dict, match_data: Dict) -> float:
        """
        Calculate professional enhancement factor based on pattern quality and data completeness
        """
        base_factor = self.pattern_influence
        
        # Factor 1: Pattern confidence from core engine
        patterns = core_patterns.get('patterns_detected', [])
        if patterns:
            avg_confidence = sum(p.get('confidence', 0.5) for p in patterns) / len(patterns)
            confidence_factor = min(1.5, max(0.5, avg_confidence))
        else:
            confidence_factor = 0.8  # Conservative for unknown patterns
        
        # Factor 2: Data quality impact
        data_quality = match_data.get('data_quality_score', 50) / 100.0
        quality_factor = min(1.2, max(0.8, data_quality))
        
        # Factor 3: Pattern count scaling (diminishing returns)
        pattern_count = core_patterns.get('pattern_count', 0)
        count_factor = min(1.3, 1.0 + (pattern_count * 0.1))
        
        # Combined professional factor
        professional_factor = base_factor * confidence_factor * quality_factor * count_factor
        
        return min(0.25, professional_factor)  # Cap at 25% max influence
    
    def _professional_renormalize(self, home: float, draw: float, away: float) -> tuple:
        """
        Professional probability renormalization with bounds checking
        """
        # Ensure reasonable bounds
        home = max(0.05, min(0.85, home))
        draw = max(0.05, min(0.50, draw))
        away = max(0.05, min(0.85, away))
        
        # Renormalize to sum to 1
        total = home + draw + away
        if total > 0:
            home /= total
            draw /= total
            away /= total
        
        return home, draw, away
    
    def _adjust_uncertainty_for_patterns(self, predictions: Dict, pattern_count: int) -> Dict:
        """
        Adjust uncertainty intervals to reflect increased confidence from patterns
        """
        if 'monte_carlo_results' not in predictions:
            return predictions
        
        mc_results = predictions['monte_carlo_results']
        
        # Reduce uncertainty intervals when patterns are present
        reduction_factor = max(0.7, 1.0 - (pattern_count * 0.08))  # 8% reduction per pattern
        
        if 'confidence_intervals' in mc_results:
            for market, interval in mc_results['confidence_intervals'].items():
                if isinstance(interval, list) and len(interval) == 2:
                    width = interval[1] - interval[0]
                    new_width = width * reduction_factor
                    center = (interval[0] + interval[1]) / 2
                    mc_results['confidence_intervals'][market] = [
                        round(center - new_width/2, 3),
                        round(center + new_width/2, 3)
                    ]
        
        # Reduce probability volatility
        if 'probability_volatility' in mc_results:
            for market, volatility in mc_results['probability_volatility'].items():
                mc_results['probability_volatility'][market] = round(volatility * reduction_factor, 4)
        
        predictions['monte_carlo_results'] = mc_results
        return predictions
    
    def _analyze_external_patterns(self, match_data: Dict, core_predictions: Dict) -> Dict:
        """
        CONSERVATIVE external pattern analysis (only used when core engine finds no patterns)
        """
        patterns_detected = []
        
        # Only analyze if we have sufficient data
        data_quality = core_predictions.get('data_quality_score', 50)
        if data_quality < 60:
            return {"patterns_detected": [], "pattern_count": 0, "net_adjustment": {"home": 0, "away": 0, "draw": 0}}
        
        # Pattern 1: Strong H2H Dominance (conservative thresholds)
        h2h_pattern = self._detect_conservative_h2h_dominance(match_data)
        if h2h_pattern:
            patterns_detected.append(h2h_pattern)
        
        # Pattern 2: Clear Momentum Difference (conservative thresholds)
        momentum_pattern = self._detect_conservative_momentum(match_data)
        if momentum_pattern:
            patterns_detected.append(momentum_pattern)
        
        # Pattern 3: Standings Gap (very conservative)
        standings_pattern = self._detect_standings_gap(match_data)
        if standings_pattern:
            patterns_detected.append(standings_pattern)
        
        return {
            "patterns_detected": patterns_detected,
            "pattern_count": len(patterns_detected),
            "net_adjustment": self._calculate_conservative_adjustment(patterns_detected)
        }
    
    def _detect_conservative_h2h_dominance(self, match_data: Dict) -> Optional[Dict]:
        """Very conservative H2H pattern detection"""
        try:
            h2h_data = match_data.get('h2h_data', {})
            matches = h2h_data.get('matches', 0)
            home_wins = h2h_data.get('home_wins', 0)
            away_wins = h2h_data.get('away_wins', 0)
            
            if matches < 6:  # Higher threshold for external detection
                return None
            
            # Very conservative thresholds for external patterns
            if home_wins / matches >= 0.7:  # 70% home dominance
                return {
                    'type': 'external_h2h_home_dominance',
                    'direction': 'home',
                    'strength': min(0.08, (home_wins/matches - 0.6)),  # Reduced strength
                    'confidence': 0.6,  # Lower confidence for external patterns
                    'evidence': f"Strong H2H: {home_wins}/{matches} home wins"
                }
            elif away_wins / matches >= 0.7:
                return {
                    'type': 'external_h2h_away_dominance',
                    'direction': 'away',
                    'strength': min(0.08, (away_wins/matches - 0.6)),
                    'confidence': 0.6,
                    'evidence': f"Strong H2H: {away_wins}/{matches} away wins"
                }
        except Exception:
            pass
        return None
    
    def _detect_conservative_momentum(self, match_data: Dict) -> Optional[Dict]:
        """Very conservative momentum detection"""
        try:
            home_form = match_data.get('home_form', [])
            away_form = match_data.get('away_form', [])
            
            if len(home_form) < 5 or len(away_form) < 5:
                return None
            
            home_momentum = self._calculate_conservative_momentum(home_form)
            away_momentum = self._calculate_conservative_momentum(away_form)
            
            momentum_gap = home_momentum - away_momentum
            
            # Conservative threshold
            if abs(momentum_gap) > 0.5:  # Significant gap required
                direction = 'home' if momentum_gap > 0 else 'away'
                return {
                    'type': 'external_momentum',
                    'direction': direction,
                    'strength': min(0.06, abs(momentum_gap) / 5),  # Very conservative
                    'confidence': 0.5,
                    'evidence': f"Clear momentum: {momentum_gap:+.2f}"
                }
        except Exception:
            pass
        return None
    
    def _detect_standings_gap(self, match_data: Dict) -> Optional[Dict]:
        """Detect significant standings gap"""
        try:
            home_standing = match_data.get('home_standing')
            away_standing = match_data.get('away_standing')
            
            if not home_standing or not away_standing:
                return None
            
            # For simplicity, assume standings are provided as position integers
            home_pos = int(home_standing) if isinstance(home_standing, (int, str)) else 10
            away_pos = int(away_standing) if isinstance(away_standing, (int, str)) else 10
            
            # Calculate position gap
            pos_gap = abs(home_pos - away_pos)
            
            # Conservative thresholds
            if pos_gap >= 8:  # Significant gap
                direction = 'home' if home_pos < away_pos else 'away'  # Lower position = better
                return {
                    'type': 'standings_gap',
                    'direction': direction,
                    'strength': min(0.05, pos_gap / 40),  # Very conservative
                    'confidence': 0.5,
                    'evidence': f"Standings gap: {pos_gap} positions"
                }
        except Exception:
            pass
        return None
    
    def _calculate_conservative_momentum(self, form: List) -> float:
        """Conservative momentum calculation"""
        if not form or len(form) < 3:
            return 0.5
        
        # Convert form points to momentum score (assuming form is points per game)
        recent_form = form[:3]  # Last 3 games
        if len(recent_form) == 0:
            return 0.5
        
        avg_points = sum(recent_form) / len(recent_form)
        # Normalize to 0-1 scale (0=0 points per game, 1=3 points per game)
        return avg_points / 3.0
    
    def _calculate_conservative_adjustment(self, patterns: List[Dict]) -> Dict[str, float]:
        """Very conservative adjustment calculation for external patterns"""
        adjustment = {'home': 0.0, 'away': 0.0, 'draw': 0.0}
        
        for pattern in patterns:
            # Very conservative: 50% strength reduction for external patterns
            strength = pattern['strength'] * pattern['confidence'] * 0.5
            
            if pattern['direction'] == 'home':
                adjustment['home'] += strength
                adjustment['away'] -= strength * 0.3  # Reduced negative impact
                adjustment['draw'] -= strength * 0.2
            elif pattern['direction'] == 'away':
                adjustment['away'] += strength
                adjustment['home'] -= strength * 0.3
                adjustment['draw'] -= strength * 0.2
        
        # Very conservative caps
        for key in adjustment:
            adjustment[key] = max(-0.06, min(0.06, adjustment[key]))
            
        return adjustment
    
    def _conservatively_apply_external_patterns(self, core_predictions: Dict, 
                                              external_patterns: Dict) -> Dict:
        """
        VERY CONSERVATIVE application of external patterns
        """
        enhanced = core_predictions.copy()
        
        if 'probabilities' in enhanced and 'match_outcomes' in enhanced['probabilities']:
            pred_1x2 = enhanced['probabilities']['match_outcomes']
            
            # Convert to probability space
            home_core = pred_1x2.get('home_win', 33.3) / 100.0
            draw_core = pred_1x2.get('draw', 33.3) / 100.0
            away_core = pred_1x2.get('away_win', 33.3) / 100.0
            
            # Apply VERY CONSERVATIVE external pattern adjustment
            adjustment = external_patterns['net_adjustment']
            external_factor = self.pattern_influence * 0.5  # 50% reduction for external patterns
            
            home_enhanced = home_core + (adjustment['home'] * external_factor)
            draw_enhanced = draw_core + (adjustment['draw'] * external_factor)
            away_enhanced = away_core + (adjustment['away'] * external_factor)
            
            # Renormalize
            home_enhanced, draw_enhanced, away_enhanced = self._professional_renormalize(
                home_enhanced, draw_enhanced, away_enhanced
            )
            
            # Update predictions
            enhanced['probabilities']['match_outcomes']['home_win'] = round(home_enhanced * 100, 1)
            enhanced['probabilities']['match_outcomes']['draw'] = round(draw_enhanced * 100, 1)
            enhanced['probabilities']['match_outcomes']['away_win'] = round(away_enhanced * 100, 1)
            
            # Add external pattern metadata
            enhanced['orchestration_metadata'] = {
                'enhancement_type': 'external_pattern_integration',
                'enhancement_factor': round(external_factor, 3),
                'pattern_count': external_patterns['pattern_count'],
                'notes': 'Very conservative external pattern integration'
            }
        
        return enhanced

# Maintain compatibility with existing code
PredictionOrchestrator = ProfessionalPredictionOrchestrator
