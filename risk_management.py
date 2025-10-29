# risk_management.py
"""Institutional-grade risk management with performance learning and dynamic adjustments"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
import os
from scipy.stats import norm

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate" 
    AGGRESSIVE = "aggressive"
    INSTITUTIONAL = "institutional"

class BettingStrategy(Enum):
    FULL_KELLY = "full_kelly"
    HALF_KELLY = "half_kelly"
    QUARTER_KELLY = "quarter_kelly"
    FIXED_FRACTION = "fixed_fraction"
    PORTFOLIO_OPTIMIZED = "portfolio_optimized"
    ADAPTIVE_KELLY = "adaptive_kelly"  # NEW: Learning Kelly

@dataclass
class StakeRecommendation:
    strategy: BettingStrategy
    stake_percent: float
    stake_amount: float
    expected_value: float
    kelly_fraction: float
    confidence: str
    risk_adjustment: float
    reasoning: List[str]
    value_edge: float  # NEW: Edge from prediction engine
    pattern_boost: float  # NEW: Pattern-based adjustment

@dataclass
class PortfolioAllocation:
    total_bankroll: float
    total_allocated: float
    available_cash: float
    position_sizes: Dict[str, float]
    correlation_matrix: np.ndarray
    portfolio_var: float
    max_drawdown_risk: float
    recommended_allocations: Dict[str, float]
    performance_adjustments: Dict[str, float]  # NEW: Performance-based adjustments

@dataclass
class RiskMetrics:
    var_95: float
    var_99: float
    expected_shortfall: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    ulcer_index: float
    risk_of_ruin: float
    confidence_score: float
    adaptive_kelly_factor: float  # NEW: Learning Kelly factor
    recent_accuracy: float  # NEW: Recent prediction accuracy

class InstitutionalRiskManager:
    """
    Professional risk management with performance learning and dynamic adjustments
    """
    
    def __init__(self, initial_bankroll: float = 10000, risk_level: RiskLevel = RiskLevel.INSTITUTIONAL):
        self.bankroll = initial_bankroll
        self.initial_bankroll = initial_bankroll
        self.risk_level = risk_level
        self.active_positions: Dict[str, float] = {}
        self.performance_history: List[Dict] = []
        self.learning_state_file = "risk_learning_state.json"
        
        # Load learning state
        self.learning_state = self._load_learning_state()
        
        # Professional configuration
        self.config = {
            'max_portfolio_allocation': 0.20,
            'max_single_position': 0.05,
            'max_drawdown_limit': 0.15,
            'var_confidence_level': 0.95,
            'correlation_threshold': 0.7,
            'base_kelly_fraction': 0.25,
            'min_confidence_threshold': 0.60,
            'performance_window': 50,  # NEW: For adaptive learning
            'value_edge_threshold': 2.0,  # NEW: Minimum edge for betting
            'adaptive_learning_rate': 0.1  # NEW: Learning rate for adjustments
        }
        
        # NEW: Dynamic market correlations (updated based on performance)
        self.market_correlations = self._initialize_dynamic_correlations()
        
        logger.info(f"Initialized Professional Risk Manager with ${initial_bankroll:,.2f}")

    def _load_learning_state(self) -> Dict[str, Any]:
        """Load risk learning state for continuous improvement"""
        try:
            if os.path.exists(self.learning_state_file):
                with open(self.learning_state_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load learning state: {e}")
        
        # Default learning state
        return {
            'adaptive_kelly_factor': 1.0,
            'market_performance': {},
            'recent_accuracy': 0.55,
            'value_bet_success_rate': 0.55,
            'drawdown_protection_active': False,
            'last_learning_update': datetime.now().isoformat(),
            'performance_trend': 'stable'
        }

    def _save_learning_state(self):
        """Save learning state for persistence"""
        try:
            with open(self.learning_state_file, 'w') as f:
                json.dump(self.learning_state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving learning state: {e}")

    def _initialize_dynamic_correlations(self) -> Dict[tuple, float]:
        """Initialize dynamic correlations that will be updated based on performance"""
        return {
            ('match_odds', 'over_under'): 0.3,
            ('match_odds', 'btts'): 0.4,
            ('over_under', 'btts'): 0.6,
            # These will be updated based on actual performance correlation
        }

    def calculate_optimal_stake(self, prediction_data: Dict, market_type: str,
                              current_positions: Dict[str, float] = None) -> StakeRecommendation:
        """
        PROFESSIONAL stake calculation with performance learning integration
        """
        try:
            # NEW: Enhanced metric extraction with value opportunities
            probability, odds, edge, value_edge = self._extract_enhanced_prediction_metrics(prediction_data, market_type)
            confidence = prediction_data.get('confidence_score', 50) / 100
            
            # NEW: Extract pattern information for risk adjustment
            pattern_count = prediction_data.get('pattern_intelligence', {}).get('pattern_count', 0)
            pattern_boost = self._calculate_pattern_boost(pattern_count, prediction_data)
            
            # Check value edge threshold
            if value_edge < self.config['value_edge_threshold']:
                return self._generate_no_value_stake(value_edge)
            
            # Calculate adaptive Kelly stake
            kelly_fraction = self._calculate_adaptive_kelly_fraction(probability, odds, confidence, value_edge)
            
            # Apply professional risk adjustments
            risk_adjusted_kelly = self._apply_professional_risk_adjustments(
                kelly_fraction, confidence, edge, value_edge, pattern_boost
            )
            
            # Portfolio-level adjustments with learning
            portfolio_adjusted_stake = self._apply_learning_portfolio_adjustments(
                risk_adjusted_kelly, market_type, current_positions or {}
            )
            
            # Dynamic drawdown protection
            final_stake_percent = self._apply_dynamic_drawdown_protection(portfolio_adjusted_stake)
            
            # Calculate stake amount
            stake_amount = self.bankroll * final_stake_percent
            
            # Expected value with pattern boost
            expected_value = self._calculate_enhanced_expected_value(
                probability, odds, stake_amount, pattern_boost
            )
            
            # Generate professional reasoning
            reasoning = self._generate_professional_stake_reasoning(
                probability, odds, edge, value_edge, kelly_fraction, risk_adjusted_kelly,
                portfolio_adjusted_stake, final_stake_percent, confidence, pattern_boost
            )
            
            return StakeRecommendation(
                strategy=BettingStrategy.ADAPTIVE_KELLY,
                stake_percent=final_stake_percent * 100,
                stake_amount=stake_amount,
                expected_value=expected_value,
                kelly_fraction=kelly_fraction,
                confidence=self._classify_confidence(confidence),
                risk_adjustment=risk_adjusted_kelly / kelly_fraction if kelly_fraction > 0 else 1.0,
                reasoning=reasoning,
                value_edge=value_edge,
                pattern_boost=pattern_boost
            )
            
        except Exception as e:
            logger.error(f"Error calculating optimal stake: {e}")
            return self._generate_fallback_stake()

    def _extract_enhanced_prediction_metrics(self, prediction_data: Dict, market_type: str) -> Tuple[float, float, float, float]:
        """Enhanced metric extraction with value opportunity integration"""
        try:
            # Get value opportunities from prediction data
            value_opportunities = prediction_data.get('value_analysis', {}).get('value_opportunities', [])
            
            if value_opportunities:
                # Use the best value opportunity
                best_opportunity = max(value_opportunities, key=lambda x: x.get('edge', 0))
                probability = best_opportunity.get('our_prob', 50) / 100
                odds = best_opportunity.get('odds', 2.0)
                edge = best_opportunity.get('edge', 0) / 100
                value_edge = best_opportunity.get('edge', 0)
                
                return probability, odds, edge, value_edge
            
            # Fallback to standard extraction if no value opportunities
            if market_type == "match_odds":
                predictions = prediction_data['predictions']['1X2']
                # Find selection with best combination of probability and value
                best_value = -100
                best_selection = None
                best_probability = 0
                best_odds = 2.0
                
                odds_1x2 = prediction_data.get('odds_1x2', [2.5, 3.2, 2.8])
                selections = [
                    ("Home Win", predictions.get('Home Win', 33.3), odds_1x2[0]),
                    ("Draw", predictions.get('Draw', 33.3), odds_1x2[1]),
                    ("Away Win", predictions.get('Away Win', 33.3), odds_1x2[2])
                ]
                
                for selection, probability, odds in selections:
                    implied_prob = 1.0 / odds
                    edge = probability - (implied_prob * 100)
                    
                    if edge > best_value:
                        best_value = edge
                        best_selection = selection
                        best_probability = probability
                        best_odds = odds
                
                return best_probability/100, best_odds, best_value/100, best_value
                
            else:
                # Handle other markets (simplified)
                probability, odds, edge = self._extract_prediction_metrics(prediction_data, market_type)
                return probability, odds, edge, edge * 100
                
        except Exception as e:
            logger.error(f"Error extracting enhanced metrics: {e}")
            return 0.5, 2.0, 0.0, 0.0

    def _calculate_pattern_boost(self, pattern_count: int, prediction_data: Dict) -> float:
        """Calculate pattern-based stake boost"""
        if pattern_count == 0:
            return 1.0
        
        # Base boost from pattern count
        base_boost = 1.0 + (min(pattern_count, 3) * 0.05)  # 5% per pattern, max 15%
        
        # Adjust based on pattern confidence from prediction engine
        pattern_confidence = prediction_data.get('pattern_intelligence', {}).get('pattern_influence', 0) / 100
        confidence_factor = 0.8 + (pattern_confidence * 0.4)  # 0.8-1.2 range
        
        # Adjust based on learning state (pattern success rate)
        pattern_success = self.learning_state.get('pattern_success_rate', 0.55)
        success_factor = 0.9 + (pattern_success * 0.2)  # 0.9-1.1 range
        
        return base_boost * confidence_factor * success_factor

    def _calculate_adaptive_kelly_fraction(self, probability: float, odds: float, 
                                         confidence: float, value_edge: float) -> float:
        """Calculate adaptive Kelly fraction with performance learning"""
        try:
            # Standard Kelly formula
            b = odds - 1
            p = probability
            q = 1 - p
            
            raw_kelly = (b * p - q) / b if b > 0 else 0
            
            # Confidence weighting
            confidence_weighted = raw_kelly * confidence
            
            # Value edge boost (higher edge = more aggressive)
            edge_boost = 1.0 + (min(value_edge / 10, 0.3))  # Max 30% boost for high edge
            
            # Adaptive learning factor from performance
            adaptive_factor = self.learning_state.get('adaptive_kelly_factor', 1.0)
            
            # Base conservative fraction with adaptations
            base_fraction = confidence_weighted * self.config['base_kelly_fraction']
            adapted_fraction = base_fraction * edge_boost * adaptive_factor
            
            return max(0.0, min(adapted_fraction, self.config['max_single_position']))
            
        except Exception as e:
            logger.error(f"Error calculating adaptive Kelly: {e}")
            return 0.0

    def _apply_professional_risk_adjustments(self, kelly_fraction: float, confidence: float,
                                           edge: float, value_edge: float, pattern_boost: float) -> float:
        """Apply professional risk adjustments with learning"""
        # Base risk level adjustments
        if self.risk_level == RiskLevel.CONSERVATIVE:
            risk_multiplier = 0.5
        elif self.risk_level == RiskLevel.MODERATE:
            risk_multiplier = 0.75
        elif self.risk_level == RiskLevel.AGGRESSIVE:
            risk_multiplier = 1.0
        else:  # INSTITUTIONAL
            # Dynamic adjustment based on multiple factors
            if value_edge > 8.0 and confidence > 0.8:
                risk_multiplier = 0.8
            elif value_edge > 5.0 and confidence > 0.7:
                risk_multiplier = 0.6
            else:
                risk_multiplier = 0.4
        
        adjusted_fraction = kelly_fraction * risk_multiplier
        
        # Apply pattern boost
        adjusted_fraction *= pattern_boost
        
        # Apply maximum single position limit
        return min(adjusted_fraction, self.config['max_single_position'])

    def _apply_learning_portfolio_adjustments(self, stake_fraction: float, market_type: str,
                                            current_positions: Dict[str, float]) -> float:
        """Apply portfolio adjustments with performance learning"""
        try:
            if not current_positions:
                return stake_fraction
            
            # Calculate current portfolio exposure
            total_exposure = sum(current_positions.values())
            
            # Portfolio capacity check
            proposed_exposure = total_exposure + stake_fraction
            if proposed_exposure > self.config['max_portfolio_allocation']:
                available_capacity = self.config['max_portfolio_allocation'] - total_exposure
                stake_fraction = min(stake_fraction, max(0, available_capacity))
            
            # Learning-based correlation adjustment
            correlation_penalty = self._calculate_learning_correlation_penalty(market_type, current_positions)
            stake_fraction *= (1 - correlation_penalty)
            
            # Market performance adjustment
            market_performance = self.learning_state.get('market_performance', {}).get(market_type, 0.55)
            if market_performance < 0.5:  # Underperforming market
                performance_penalty = 0.3  # 30% reduction
                stake_fraction *= (1 - performance_penalty)
            
            return stake_fraction
            
        except Exception as e:
            logger.error(f"Error applying portfolio adjustments: {e}")
            return stake_fraction

    def _calculate_learning_correlation_penalty(self, new_market: str, current_positions: Dict[str, float]) -> float:
        """Calculate correlation penalty with performance learning"""
        try:
            total_penalty = 0.0
            total_weight = 0.0
            
            for existing_market, position_size in current_positions.items():
                # Get dynamic correlation
                correlation = self._get_dynamic_market_correlation(new_market, existing_market)
                
                # Calculate penalty based on correlation and position size
                if correlation > self.config['correlation_threshold']:
                    penalty = correlation * (position_size / self.config['max_portfolio_allocation'])
                    total_penalty += penalty
                    total_weight += 1
            
            return total_penalty / total_weight if total_weight > 0 else 0.0
            
        except Exception:
            return 0.0

    def _get_dynamic_market_correlation(self, market1: str, market2: str) -> float:
        """Get dynamic market correlation with performance updates"""
        key = tuple(sorted([market1, market2]))
        
        # In production, this would be updated based on actual performance correlation
        # For now, use base correlations but plan for dynamic updates
        return self.market_correlations.get(key, 0.2)

    def _apply_dynamic_drawdown_protection(self, stake_fraction: float) -> float:
        """Apply dynamic drawdown protection based on performance"""
        try:
            if len(self.performance_history) < 5:
                return stake_fraction
            
            # Calculate recent performance metrics
            recent_performance = self.performance_history[-self.config['performance_window']:]
            recent_returns = [p['return'] for p in recent_performance if 'return' in p]
            
            if not recent_returns:
                return stake_fraction
            
            # Check for recent losses
            recent_losses = sum(1 for r in recent_returns if r < 0)
            loss_ratio = recent_losses / len(recent_returns)
            
            # Calculate current drawdown
            current_drawdown = self._calculate_current_drawdown()
            
            # Dynamic drawdown protection
            if current_drawdown > self.config['max_drawdown_limit']:
                drawdown_multiplier = 0.3  # 70% reduction in severe drawdown
                self.learning_state['drawdown_protection_active'] = True
            elif loss_ratio > 0.6:
                drawdown_multiplier = 0.5
            elif loss_ratio > 0.4:
                drawdown_multiplier = 0.75
            else:
                drawdown_multiplier = 1.0
                self.learning_state['drawdown_protection_active'] = False
            
            return stake_fraction * drawdown_multiplier
            
        except Exception:
            return stake_fraction

    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak"""
        try:
            if not self.performance_history:
                return 0.0
            
            # Calculate cumulative returns
            cumulative_returns = np.cumsum([p.get('return', 0) for p in self.performance_history])
            current_value = self.initial_bankroll + cumulative_returns[-1] if len(cumulative_returns) > 0 else self.initial_bankroll
            
            # Find peak
            peak = max(self.initial_bankroll, current_value)
            for i in range(len(cumulative_returns)):
                portfolio_value = self.initial_bankroll + cumulative_returns[i]
                if portfolio_value > peak:
                    peak = portfolio_value
            
            # Calculate drawdown
            drawdown = (peak - current_value) / peak if peak > 0 else 0.0
            return drawdown
            
        except Exception:
            return 0.0

    def _calculate_enhanced_expected_value(self, probability: float, odds: float,
                                         stake_amount: float, pattern_boost: float) -> float:
        """Calculate enhanced expected value with pattern boost"""
        win_payout = stake_amount * (odds - 1) * pattern_boost
        loss_payout = -stake_amount
        
        expected_value = (probability * win_payout) + ((1 - probability) * loss_payout)
        return expected_value

    def _generate_professional_stake_reasoning(self, probability: float, odds: float, edge: float,
                                             value_edge: float, kelly_fraction: float, risk_adjusted: float,
                                             portfolio_adjusted: float, final_stake: float,
                                             confidence: float, pattern_boost: float) -> List[str]:
        """Generate professional reasoning for stake recommendation"""
        reasoning = []
        
        reasoning.append(f"Value edge: {value_edge:.1f}% (threshold: {self.config['value_edge_threshold']}%)")
        reasoning.append(f"Probability: {probability:.1%}, Odds: {odds:.2f}, Mathematical edge: {edge:.2%}")
        
        if pattern_boost > 1.0:
            reasoning.append(f"Pattern boost: +{(pattern_boost-1)*100:.1f}%")
        
        reasoning.append(f"Base Kelly: {kelly_fraction:.3f}")
        
        if risk_adjusted < kelly_fraction:
            reasoning.append(f"Risk-adjusted to {risk_adjusted:.3f} for {self.risk_level.value} risk level")
        
        if portfolio_adjusted < risk_adjusted:
            reasoning.append("Portfolio diversification applied")
        
        if confidence < self.config['min_confidence_threshold']:
            reasoning.append(f"Lower confidence ({confidence:.1%}) reduced stake size")
        
        # Adaptive learning factors
        adaptive_factor = self.learning_state.get('adaptive_kelly_factor', 1.0)
        if adaptive_factor != 1.0:
            reasoning.append(f"Adaptive learning factor: {adaptive_factor:.2f}")
        
        reasoning.append(f"Final recommended stake: {final_stake:.3%} of bankroll (${self.bankroll * final_stake:.2f})")
        
        return reasoning

    def _generate_no_value_stake(self, value_edge: float) -> StakeRecommendation:
        """Generate recommendation when no value is detected"""
        return StakeRecommendation(
            strategy=BettingStrategy.FIXED_FRACTION,
            stake_percent=0.0,  # No bet recommended
            stake_amount=0.0,
            expected_value=0.0,
            kelly_fraction=0.0,
            confidence="LOW",
            risk_adjustment=1.0,
            reasoning=[f"No value bet detected (edge: {value_edge:.1f}% < threshold: {self.config['value_edge_threshold']}%)"],
            value_edge=value_edge,
            pattern_boost=1.0
        )

    def update_performance_with_learning(self, bet_result: Dict):
        """Update performance with learning integration"""
        try:
            # Update performance history
            self.performance_history.append({
                'timestamp': datetime.now(),
                'return': bet_result.get('profit_loss', 0),
                'stake_percent': bet_result.get('stake_percent', 0),
                'market': bet_result.get('market', 'unknown'),
                'outcome': bet_result.get('outcome', 'unknown'),
                'value_edge': bet_result.get('value_edge', 0),
                'pattern_count': bet_result.get('pattern_count', 0)
            })
            
            # Update bankroll
            self.bankroll += bet_result.get('profit_loss', 0)
            
            # Update learning state
            self._update_learning_state(bet_result)
            
            # Keep manageable history
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
                
            logger.info(f"Updated performance: P&L ${bet_result.get('profit_loss', 0):.2f}, Bankroll: ${self.bankroll:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating performance with learning: {e}")

    def _update_learning_state(self, bet_result: Dict):
        """Update learning state based on bet results"""
        try:
            # Update recent accuracy
            recent_bets = self.performance_history[-self.config['performance_window']:]
            if recent_bets:
                accurate_bets = sum(1 for bet in recent_bets if bet.get('outcome') == 'win')
                self.learning_state['recent_accuracy'] = accurate_bets / len(recent_bets)
            
            # Update market performance
            market = bet_result.get('market', 'unknown')
            if market != 'unknown':
                market_bets = [b for b in self.performance_history if b.get('market') == market]
                if market_bets:
                    market_wins = sum(1 for b in market_bets if b.get('outcome') == 'win')
                    self.learning_state['market_performance'][market] = market_wins / len(market_bets)
            
            # Update value bet success rate
            value_bets = [b for b in self.performance_history if b.get('value_edge', 0) >= self.config['value_edge_threshold']]
            if value_bets:
                value_wins = sum(1 for b in value_bets if b.get('outcome') == 'win')
                self.learning_state['value_bet_success_rate'] = value_wins / len(value_bets)
            
            # Update pattern success rate
            pattern_bets = [b for b in self.performance_history if b.get('pattern_count', 0) > 0]
            if pattern_bets:
                pattern_wins = sum(1 for b in pattern_bets if b.get('outcome') == 'win')
                self.learning_state['pattern_success_rate'] = pattern_wins / len(pattern_bets)
            
            # Update adaptive Kelly factor based on performance
            self._update_adaptive_kelly_factor()
            
            # Save learning state
            self._save_learning_state()
            
        except Exception as e:
            logger.error(f"Error updating learning state: {e}")

    def _update_adaptive_kelly_factor(self):
        """Update adaptive Kelly factor based on recent performance"""
        try:
            recent_bets = self.performance_history[-self.config['performance_window']:]
            if len(recent_bets) < 10:
                return
            
            # Calculate performance metrics
            returns = [b.get('return', 0) for b in recent_bets]
            total_return = sum(returns)
            avg_return = total_return / len(returns) if returns else 0
            
            # Update Kelly factor based on performance
            current_factor = self.learning_state.get('adaptive_kelly_factor', 1.0)
            
            if avg_return > 0:
                # Positive performance - slightly increase aggression
                new_factor = min(1.3, current_factor * 1.05)
            else:
                # Negative performance - reduce aggression
                new_factor = max(0.7, current_factor * 0.95)
            
            self.learning_state['adaptive_kelly_factor'] = new_factor
            
        except Exception as e:
            logger.error(f"Error updating adaptive Kelly factor: {e}")

    # Keep existing methods for portfolio optimization and risk metrics
    # with enhanced learning integration...

    def optimize_portfolio_allocation(self, potential_bets: List[Dict]) -> PortfolioAllocation:
        """Enhanced portfolio optimization with learning"""
        try:
            bet_opportunities = []
            for bet in potential_bets:
                stake_rec = self.calculate_optimal_stake(bet['prediction'], bet['market_type'])
                bet_opportunities.append({
                    'market': bet['market_type'],
                    'stake_recommendation': stake_rec,
                    'value_edge': stake_rec.value_edge,
                    'pattern_boost': stake_rec.pattern_boost
                })
            
            # Calculate portfolio metrics with learning adjustments
            position_sizes = {}
            total_allocated = 0.0
            
            for opportunity in bet_opportunities:
                market = opportunity['market']
                stake_pct = opportunity['stake_recommendation'].stake_percent / 100
                
                # Apply learning-based adjustments
                market_perf = self.learning_state.get('market_performance', {}).get(market, 0.55)
                if market_perf < 0.5:
                    stake_pct *= 0.7  # Reduce allocation for underperforming markets
                
                position_sizes[market] = stake_pct
                total_allocated += stake_pct
            
            # Enhanced correlation matrix
            correlation_matrix = self._build_learning_correlation_matrix(
                [opp['market'] for opp in bet_opportunities]
            )
            
            # Calculate portfolio metrics
            portfolio_var = self._calculate_enhanced_portfolio_var(position_sizes, correlation_matrix)
            max_drawdown_risk = self._calculate_learning_drawdown_risk(position_sizes)
            
            # Generate recommended allocations
            recommended_allocations = self._diversify_with_learning(position_sizes, correlation_matrix)
            
            # Performance adjustments
            performance_adjustments = self._calculate_performance_adjustments(bet_opportunities)
            
            return PortfolioAllocation(
                total_bankroll=self.bankroll,
                total_allocated=total_allocated,
                available_cash=1.0 - total_allocated,
                position_sizes=position_sizes,
                correlation_matrix=correlation_matrix,
                portfolio_var=portfolio_var,
                max_drawdown_risk=max_drawdown_risk,
                recommended_allocations=recommended_allocations,
                performance_adjustments=performance_adjustments
            )
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio allocation: {e}")
            return self._generate_fallback_allocation()

    def _build_learning_correlation_matrix(self, markets: List[str]) -> np.ndarray:
        """Build correlation matrix with learning updates"""
        n = len(markets)
        corr_matrix = np.eye(n)
        
        for i in range(n):
            for j in range(i+1, n):
                correlation = self._get_dynamic_market_correlation(markets[i], markets[j])
                corr_matrix[i, j] = correlation
                corr_matrix[j, i] = correlation
                
        return corr_matrix

    def _calculate_enhanced_portfolio_var(self, position_sizes: Dict[str, float], 
                                        correlation_matrix: np.ndarray) -> float:
        """Calculate enhanced portfolio VaR with learning adjustments"""
        try:
            positions = np.array(list(position_sizes.values()))
            
            # Dynamic volatility based on recent performance
            recent_volatility = self._calculate_recent_volatility()
            avg_volatility = max(0.1, min(0.3, recent_volatility))  # Bound between 10-30%
            
            portfolio_variance = positions.T @ correlation_matrix @ positions * (avg_volatility ** 2)
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # VaR at 95% confidence
            var_95 = 1.645 * portfolio_volatility * self.bankroll
            
            return var_95
            
        except Exception:
            return 0.0

    def _calculate_recent_volatility(self) -> float:
        """Calculate recent portfolio volatility"""
        try:
            if len(self.performance_history) < 10:
                return 0.15  # Default
            
            recent_returns = [p.get('return', 0) for p in self.performance_history[-20:]]
            if not recent_returns:
                return 0.15
            
            # Calculate volatility as percentage of bankroll
            returns_pct = [r / self.bankroll for r in recent_returns]
            return np.std(returns_pct) if len(returns_pct) > 1 else 0.15
            
        except Exception:
            return 0.15

    def _calculate_learning_drawdown_risk(self, position_sizes: Dict[str, float]) -> float:
        """Calculate drawdown risk with learning integration"""
        try:
            total_exposure = sum(position_sizes.values())
            current_drawdown = self._calculate_current_drawdown()
            
            # Enhanced risk calculation
            base_risk = total_exposure / self.config['max_portfolio_allocation']
            drawdown_risk = current_drawdown / self.config['max_drawdown_limit']
            
            # Combined risk score
            combined_risk = (base_risk * 0.6 + drawdown_risk * 0.4)
            
            return min(1.0, combined_risk)
                
        except Exception:
            return 0.5

    def _diversify_with_learning(self, position_sizes: Dict[str, float],
                               correlation_matrix: np.ndarray) -> Dict[str, float]:
        """Diversify allocations with learning-based adjustments"""
        try:
            diversified = position_sizes.copy()
            markets = list(position_sizes.keys())
            
            # Reduce allocations based on performance and correlations
            for i, market1 in enumerate(markets):
                market_perf1 = self.learning_state.get('market_performance', {}).get(market1, 0.55)
                
                for j, market2 in enumerate(markets[i+1:], i+1):
                    market_perf2 = self.learning_state.get('market_performance', {}).get(market2, 0.55)
                    
                    # Higher penalty for correlated underperforming markets
                    if correlation_matrix[i, j] > self.config['correlation_threshold']:
                        if market_perf1 < 0.5 and market_perf2 < 0.5:
                            reduction = 0.5  # 50% reduction for both underperforming
                        elif market_perf1 < 0.5 or market_perf2 < 0.5:
                            reduction = 0.3  # 30% reduction for one underperforming
                        else:
                            reduction = 0.2  # 20% reduction normally
                        
                        # Apply reduction to smaller position
                        if diversified[market1] > diversified[market2]:
                            diversified[market2] = max(0.001, diversified[market2] * (1 - reduction))
                        else:
                            diversified[market1] = max(0.001, diversified[market1] * (1 - reduction))
            
            return diversified
            
        except Exception:
            return position_sizes

    def _calculate_performance_adjustments(self, bet_opportunities: List[Dict]) -> Dict[str, float]:
        """Calculate performance-based allocation adjustments"""
        adjustments = {}
        
        for opportunity in bet_opportunities:
            market = opportunity['market']
            market_perf = self.learning_state.get('market_performance', {}).get(market, 0.55)
            
            if market_perf > 0.6:
                adjustments[market] = 1.2  # 20% boost for strong markets
            elif market_perf < 0.5:
                adjustments[market] = 0.8  # 20% reduction for weak markets
            else:
                adjustments[market] = 1.0  # Neutral
        
        return adjustments

    def _classify_confidence(self, confidence: float) -> str:
        """Classify confidence level for reporting"""
        if confidence >= 0.8:
            return "HIGH"
        elif confidence >= 0.7:
            return "MEDIUM-HIGH"
        elif confidence >= 0.6:
            return "MEDIUM"
        elif confidence >= 0.5:
            return "MEDIUM-LOW"
        else:
            return "LOW"

    def _generate_fallback_stake(self) -> StakeRecommendation:
        """Generate professional fallback stake"""
        return StakeRecommendation(
            strategy=BettingStrategy.FIXED_FRACTION,
            stake_percent=1.0,
            stake_amount=self.bankroll * 0.01,
            expected_value=0.0,
            kelly_fraction=0.0,
            confidence="LOW",
            risk_adjustment=1.0,
            reasoning=["Fallback stake due to calculation error"],
            value_edge=0.0,
            pattern_boost=1.0
        )

    def _generate_fallback_allocation(self) -> PortfolioAllocation:
        """Generate fallback portfolio allocation"""
        return PortfolioAllocation(
            total_bankroll=self.bankroll,
            total_allocated=0.0,
            available_cash=1.0,
            position_sizes={},
            correlation_matrix=np.eye(1),
            portfolio_var=0.0,
            max_drawdown_risk=0.0,
            recommended_allocations={},
            performance_adjustments={}
        )

# Enhanced utility functions
def create_risk_manager(bankroll: float = 10000, risk_level: str = "institutional") -> InstitutionalRiskManager:
    """Create a professional risk manager with learning"""
    risk_enum = RiskLevel(risk_level.lower())
    return InstitutionalRiskManager(bankroll, risk_enum)

def record_bet_result(risk_manager: InstitutionalRiskManager, bet_result: Dict):
    """Professional function to record bet results with learning"""
    risk_manager.update_performance_with_learning(bet_result)

def get_learning_insights(risk_manager: InstitutionalRiskManager) -> Dict[str, Any]:
    """Get risk management learning insights"""
    return {
        'adaptive_kelly_factor': risk_manager.learning_state.get('adaptive_kelly_factor', 1.0),
        'recent_accuracy': risk_manager.learning_state.get('recent_accuracy', 0.55),
        'value_bet_success_rate': risk_manager.learning_state.get('value_bet_success_rate', 0.55),
        'market_performance': risk_manager.learning_state.get('market_performance', {}),
        'drawdown_protection_active': risk_manager.learning_state.get('drawdown_protection_active', False)
    }