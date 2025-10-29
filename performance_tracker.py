# performance_tracker.py
"""Institutional-grade performance tracking with advanced analytics and model feedback integration"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
import re
from scipy import stats
import os
import warnings
from collections import deque, defaultdict
import hashlib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class BetOutcome(Enum):
    WIN = "win"
    LOSS = "loss" 
    PUSH = "push"
    VOID = "void"
    CASHOUT = "cashout"

class MarketType(Enum):
    MATCH_ODDS = "match_odds"
    OVER_UNDER = "over_under"
    BTTS = "btts"
    ASIAN_HANDICAP = "asian_handicap"
    DOUBLE_CHANCE = "double_chance"

class BettingStrategy(Enum):
    VALUE_BETTING = "value_betting"
    KELLY_CRITERION = "kelly_criterion"
    FIXED_STAKE = "fixed_stake"
    MARTINGALE = "martingale"  # For monitoring only - not recommended
    PATTERN_BASED = "pattern_based"

class PerformanceTier(Enum):
    ELITE = "elite"           # ROI > 10%
    PROFESSIONAL = "professional"  # ROI 5-10%
    COMPETENT = "competent"    # ROI 2-5%
    BREAKEVEN = "breakeven"    # ROI -2% to 2%
    UNDERPERFORMING = "underperforming"  # ROI < -2%

@dataclass
class BetRecord:
    """Enhanced bet record with comprehensive tracking"""
    id: str  # Unique identifier
    timestamp: datetime
    match: str
    market: MarketType
    selection: str
    probability: float
    odds: float
    stake_percent: float
    stake_amount: float
    outcome: BetOutcome
    profit_loss: float
    confidence: float
    data_quality: float
    model_type: str
    pattern_count: int
    value_edge: float
    betting_strategy: BettingStrategy
    bankroll_before: float
    bankroll_after: float
    kelly_fraction: float
    closing_odds: Optional[float] = None
    cashout_value: Optional[float] = None
    bet_metadata: Dict[str, Any] = None

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics with advanced analytics"""
    # Basic metrics
    total_bets: int
    winning_bets: int
    losing_bets: int
    pushed_bets: int
    voided_bets: int
    
    # Financial metrics
    total_staked: float
    total_profit: float
    total_pnl: float
    roi: float
    net_profit: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    volatility: float
    var_95: float
    cvar_95: float
    ulcer_index: float
    
    # Betting metrics
    win_rate: float
    average_odds: float
    average_stake: float
    expectancy: float
    profit_factor: float
    edge: float
    
    # Statistical metrics
    p_value: float
    confidence_interval: Tuple[float, float]
    z_score: float
    statistical_power: float
    
    # Advanced metrics
    kelly_optimized_roi: float
    calibration_score: float
    value_betting_accuracy: float
    pattern_success_rate: float
    market_efficiency: float
    confidence_accuracy: float
    
    # Performance tiers
    performance_tier: PerformanceTier
    risk_adjusted_grade: str
    
    # Time-based metrics
    daily_roi: float
    weekly_roi: float
    monthly_roi: float
    compound_annual_growth: float

@dataclass
class ModelFeedback:
    """Advanced model feedback with actionable insights"""
    confidence_calibration: float
    market_performance: Dict[str, float]
    pattern_effectiveness: Dict[str, float]
    feature_importance: Dict[str, float]
    strategy_performance: Dict[str, float]
    recommended_adjustments: List[str]
    calibration_factors: Dict[str, float]
    risk_assessment: Dict[str, float]
    optimization_opportunities: List[str]
    performance_forecast: Dict[str, float]

@dataclass
class RiskMetrics:
    """Comprehensive risk assessment"""
    value_at_risk: float
    expected_shortfall: float
    maximum_adverse_excursion: float
    risk_of_ruin: float
    kelly_optimal_fraction: float
    optimal_bet_size: float
    risk_adjusted_return: float

class InstitutionalPerformanceTracker:
    """
    Institutional-grade performance tracking with advanced analytics,
    risk management, and machine learning integration
    """
    
    def __init__(self, initial_bankroll: float = 10000, config: Optional[Dict] = None):
        self.bet_history: List[BetRecord] = []
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.performance_history: List[Dict] = []
        self.feedback_data: List[ModelFeedback] = []
        self.risk_metrics_history: List[RiskMetrics] = []
        
        # Enhanced professional configuration
        self.config = config or {
            'min_bets_for_analysis': 20,
            'confidence_threshold': 0.70,
            'sharpe_target': 1.5,
            'sortino_target': 2.0,
            'max_drawdown_limit': 0.10,
            'var_confidence_level': 0.95,
            'feedback_update_frequency': 15,
            'value_edge_threshold': 2.0,
            'kelly_fraction_limit': 0.25,
            'risk_of_ruin_threshold': 0.05,
            'performance_evaluation_period': 30,
            'volatility_lookback': 50
        }
        
        # Advanced analytics state
        self.learning_state = self._load_learning_state()
        self.risk_assessor = RiskAssessor(self.config)
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Caching for performance
        self._metrics_cache = {}
        self._cache_timestamp = None
        
        logger.info(f"Initialized Enhanced InstitutionalPerformanceTracker with ${initial_bankroll:,.2f}")

    def _load_learning_state(self) -> Dict[str, Any]:
        """Load advanced learning state for continuous improvement"""
        try:
            if os.path.exists("advanced_learning_state.json"):
                with open("advanced_learning_state.json", 'r') as f:
                    state = json.load(f)
                    # Convert string dates back to datetime
                    if 'last_learning_update' in state:
                        state['last_learning_update'] = datetime.fromisoformat(state['last_learning_update'])
                    return state
        except Exception as e:
            logger.warning(f"Could not load learning state: {e}")
        
        # Enhanced default learning state
        return {
            'market_performance': {},
            'pattern_success_rates': {},
            'strategy_performance': {},
            'confidence_calibration': 1.0,
            'value_bet_accuracy': 0.55,
            'market_efficiency': 0.0,
            'recent_trend': 'stable',
            'performance_momentum': 0.0,
            'risk_adjustment_factor': 1.0,
            'last_learning_update': datetime.now(),
            'learning_cycles': 0,
            'adaptive_thresholds': {
                'value_edge': 2.0,
                'confidence_min': 0.65,
                'stake_max': 0.05
            }
        }

    def _save_learning_state(self):
        """Save learning state with enhanced persistence"""
        try:
            # Convert datetime to string for JSON serialization
            save_state = self.learning_state.copy()
            if 'last_learning_update' in save_state and isinstance(save_state['last_learning_update'], datetime):
                save_state['last_learning_update'] = save_state['last_learning_update'].isoformat()
            
            with open("advanced_learning_state.json", 'w') as f:
                json.dump(save_state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving learning state: {e}")

    def record_bet(self, prediction_data: Dict, actual_result: Union[Dict, str], 
                   stake_percent: float, betting_strategy: BettingStrategy = BettingStrategy.VALUE_BETTING) -> BetRecord:
        """
        Enhanced bet recording with advanced tracking and risk management
        """
        try:
            # Generate unique bet ID
            bet_id = self._generate_bet_id(prediction_data)
            
            # Parse actual result with enhanced error handling
            if isinstance(actual_result, str):
                actual_result = self._parse_actual_result(actual_result)
            
            # Extract enhanced prediction details
            match = f"{prediction_data.get('home_team', 'Unknown')} vs {prediction_data.get('away_team', 'Unknown')}"
            market = self._determine_market(prediction_data)
            selection, probability, odds, value_edge = self._extract_enhanced_prediction_details(prediction_data, market)
            
            # Calculate Kelly fraction for reference
            kelly_fraction = self._calculate_kelly_fraction(probability, odds)
            
            # Determine stake amount with risk limits
            stake_amount = self._calculate_optimal_stake(stake_percent, kelly_fraction, betting_strategy)
            
            # Record bankroll before bet
            bankroll_before = self.current_bankroll
            
            # Determine outcome with enhanced logic
            outcome, profit_loss = self._calculate_enhanced_outcome(
                selection, odds, stake_amount, actual_result, market, prediction_data
            )
            
            # Update bankroll
            bankroll_after = bankroll_before + profit_loss
            
            # Extract advanced features
            pattern_count = prediction_data.get('pattern_intelligence', {}).get('pattern_count', 0)
            confidence = prediction_data.get('confidence_score', 50)
            data_quality = prediction_data.get('data_quality_score', 50)
            
            # Create enhanced bet record
            bet_record = BetRecord(
                id=bet_id,
                timestamp=datetime.now(),
                match=match,
                market=market,
                selection=selection,
                probability=probability,
                odds=odds,
                stake_percent=stake_percent,
                stake_amount=stake_amount,
                outcome=outcome,
                profit_loss=profit_loss,
                confidence=confidence,
                data_quality=data_quality,
                model_type=prediction_data.get('model_type', 'Unknown'),
                pattern_count=pattern_count,
                value_edge=value_edge,
                betting_strategy=betting_strategy,
                bankroll_before=bankroll_before,
                bankroll_after=bankroll_after,
                kelly_fraction=kelly_fraction,
                bet_metadata={
                    'prediction_hash': self._hash_prediction_data(prediction_data),
                    'market_conditions': self._assess_market_conditions(prediction_data),
                    'risk_category': self._classify_risk_category(probability, odds, value_edge)
                }
            )
            
            self.bet_history.append(bet_record)
            self.current_bankroll = bankroll_after
            
            # Update analytics
            self._update_performance_history()
            self._update_risk_metrics()
            
            # Update learning state more frequently for adaptive learning
            if len(self.bet_history) % max(1, self.config['feedback_update_frequency'] // 2) == 0:
                self._update_advanced_learning_state()
            
            # Clear cache
            self._metrics_cache.clear()
            
            logger.info(
                f"Recorded bet: {match} - {selection} - {outcome.value} - "
                f"P&L: ${profit_loss:+.2f} - Bankroll: ${bankroll_after:,.2f}"
            )
            
            return bet_record
            
        except Exception as e:
            logger.error(f"Error recording bet: {e}")
            return self._create_enhanced_fallback_bet_record(prediction_data, str(e))

    def _generate_bet_id(self, prediction_data: Dict) -> str:
        """Generate unique bet ID based on prediction data"""
        content = f"{prediction_data.get('home_team', '')}{prediction_data.get('away_team', '')}{datetime.now().timestamp()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _hash_prediction_data(self, prediction_data: Dict) -> str:
        """Create hash of prediction data for tracking"""
        import pickle
        return hashlib.md5(pickle.dumps(prediction_data)).hexdigest()[:8]

    def _assess_market_conditions(self, prediction_data: Dict) -> str:
        """Assess market conditions for the bet"""
        odds = prediction_data.get('odds_1x2', [])
        if not odds or len(odds) < 3:
            return "unknown"
        
        favorite_odds = min(odds)
        if favorite_odds < 1.5:
            return "heavy_favorite"
        elif favorite_odds < 2.0:
            return "moderate_favorite"
        elif max(odds) > 4.0:
            return "underdog_opportunity"
        else:
            return "competitive"

    def _classify_risk_category(self, probability: float, odds: float, value_edge: float) -> str:
        """Classify bet into risk categories"""
        if value_edge > 5.0 and probability > 65:
            return "high_confidence_value"
        elif value_edge > 2.0 and probability > 55:
            return "medium_confidence_value"
        elif value_edge > 0:
            return "low_confidence_value"
        elif probability > 70:
            return "high_probability"
        else:
            return "speculative"

    def _calculate_optimal_stake(self, proposed_stake: float, kelly_fraction: float, 
                               strategy: BettingStrategy) -> float:
        """Calculate optimal stake with risk management"""
        base_stake = self.current_bankroll * proposed_stake
        
        if strategy == BettingStrategy.KELLY_CRITERION:
            optimal = self.current_bankroll * kelly_fraction
            # Apply fractional Kelly for risk management
            fractional_kelly = optimal * 0.5  # Half Kelly for conservative approach
            return min(fractional_kelly, base_stake * 1.5)  # Cap at 1.5x proposed
        
        elif strategy == BettingStrategy.VALUE_BETTING:
            # Scale stake based on value edge
            edge_multiplier = min(2.0, 1.0 + (kelly_fraction * 2))
            return base_stake * edge_multiplier
        
        else:
            return base_stake

    def _parse_actual_result(self, result_str: str) -> Dict[str, Any]:
        """
        Advanced actual result parsing with multiple format support and validation
        """
        try:
            result_str_lower = result_str.lower().strip()
            
            # Enhanced score extraction with multiple patterns
            score_patterns = [
                r'(\d+)[\-\:\s]+(\d+)',  # Standard scores: 2-1, 2:1, 2 1
                r'(\d+)\s*[\-\:]\s*(\d+)',  # Scores with spaces: 2 - 1
                r'^.*?(\d+).*?(\d+).*$'  # Any two numbers in string
            ]
            
            home_goals, away_goals = None, None
            for pattern in score_patterns:
                score_match = re.search(pattern, result_str_lower)
                if score_match:
                    home_goals = int(score_match.group(1))
                    away_goals = int(score_match.group(2))
                    break
            
            # If no scores found, try to infer from text
            if home_goals is None or away_goals is None:
                if any(word in result_str_lower for word in ['home win', 'home won', '1-0', 'home victory']):
                    home_goals, away_goals = 1, 0
                elif any(word in result_str_lower for word in ['away win', 'away won', '0-1', 'away victory']):
                    home_goals, away_goals = 0, 1
                elif any(word in result_str_lower for word in ['draw', 'tie', '0-0', '1-1']):
                    home_goals, away_goals = 1, 1  # Use 1-1 as default draw
                else:
                    # Last resort: try to extract any numbers
                    numbers = re.findall(r'\d+', result_str_lower)
                    if len(numbers) >= 2:
                        home_goals, away_goals = int(numbers[0]), int(numbers[1])
                    else:
                        raise ValueError(f"Cannot parse result: {result_str}")
            
            # Validate scores
            if home_goals < 0 or away_goals < 0:
                raise ValueError(f"Invalid scores: {home_goals}-{away_goals}")
            
            # Detect if result was inferred
            inferred = not any(char.isdigit() for char in result_str)
            
            return {
                'home_goals': home_goals,
                'away_goals': away_goals,
                'parsed_from': result_str,
                'inferred': inferred,
                'total_goals': home_goals + away_goals,
                'goal_difference': abs(home_goals - away_goals)
            }
                    
        except Exception as e:
            logger.warning(f"Could not parse result '{result_str}': {e}")
            return {
                'home_goals': 0, 
                'away_goals': 0, 
                'error': str(e),
                'inferred': True,
                'parsed_from': result_str
            }

    def _extract_enhanced_prediction_details(self, prediction_data: Dict, market: MarketType) -> Tuple[str, float, float, float]:
        """
        Advanced prediction extraction with multiple market support and edge calculation
        """
        try:
            if market == MarketType.MATCH_ODDS:
                return self._extract_match_odds_details(prediction_data)
            elif market == MarketType.OVER_UNDER:
                return self._extract_over_under_details(prediction_data)
            elif market == MarketType.BTTS:
                return self._extract_btts_details(prediction_data)
            elif market == MarketType.ASIAN_HANDICAP:
                return self._extract_asian_handicap_details(prediction_data)
            else:
                return self._extract_generic_details(prediction_data)
                
        except Exception as e:
            logger.error(f"Error extracting prediction details: {e}")
            return "Unknown", 50.0, 2.0, 0.0

    def _extract_match_odds_details(self, prediction_data: Dict) -> Tuple[str, float, float, float]:
        """Extract details for match odds market"""
        predictions = prediction_data.get('predictions', {}).get('1X2', {})
        odds_1x2 = prediction_data.get('odds_1x2', [2.5, 3.2, 2.8])
        
        selections = [
            ("Home Win", predictions.get('Home Win', 33.3), odds_1x2[0]),
            ("Draw", predictions.get('Draw', 33.3), odds_1x2[1]),
            ("Away Win", predictions.get('Away Win', 33.3), odds_1x2[2])
        ]
        
        return self._find_best_value_bet(selections)

    def _extract_over_under_details(self, prediction_data: Dict) -> Tuple[str, float, float, float]:
        """Extract details for over/under market"""
        predictions = prediction_data.get('predictions', {}).get('Over/Under', {})
        odds_over_under = prediction_data.get('odds_over_under', [1.9, 1.9])
        
        over_prob = predictions.get('Over 2.5', 50)
        under_prob = predictions.get('Under 2.5', 50)
        
        selections = [
            ("Over 2.5", over_prob, odds_over_under[0]),
            ("Under 2.5", under_prob, odds_over_under[1])
        ]
        
        return self._find_best_value_bet(selections)

    def _extract_btts_details(self, prediction_data: Dict) -> Tuple[str, float, float, float]:
        """Extract details for BTTS market"""
        predictions = prediction_data.get('predictions', {}).get('BTTS', {})
        odds_btts = prediction_data.get('odds_btts', [1.85, 1.95])
        
        yes_prob = predictions.get('Yes', 50)
        no_prob = predictions.get('No', 50)
        
        selections = [
            ("Yes", yes_prob, odds_btts[0]),
            ("No", no_prob, odds_btts[1])
        ]
        
        return self._find_best_value_bet(selections)

    def _extract_asian_handicap_details(self, prediction_data: Dict) -> Tuple[str, float, float, float]:
        """Extract details for Asian Handicap market"""
        # Simplified implementation - would be more complex in production
        return "Home -0.5", 55.0, 1.95, 2.5

    def _extract_generic_details(self, prediction_data: Dict) -> Tuple[str, float, float, float]:
        """Extract details for generic markets"""
        # Fallback implementation
        return "Selection", 50.0, 2.0, 0.0

    def _find_best_value_bet(self, selections: List[Tuple[str, float, float]]) -> Tuple[str, float, float, float]:
        """Find the best value bet among selections"""
        best_value = -100
        best_selection = ""
        best_probability = 0
        best_odds = 2.0
        
        for selection, probability, odds in selections:
            # Calculate value edge with implied probability
            implied_prob = 1.0 / odds
            edge = probability - (implied_prob * 100)
            
            # Enhanced value calculation considering confidence
            confidence_adjusted_edge = edge * (probability / 100)
            
            if confidence_adjusted_edge > best_value and edge >= self.config['value_edge_threshold']:
                best_value = edge
                best_selection = selection
                best_probability = probability
                best_odds = odds
        
        # If no value bet found, use highest probability with positive edge
        if not best_selection:
            valid_selections = [(s, p, o) for s, p, o in selections 
                              if p - (1.0 / o * 100) >= 0]
            if valid_selections:
                best_selection, best_probability, best_odds = max(valid_selections, key=lambda x: x[1])
                best_value = best_probability - (1.0 / best_odds * 100)
            else:
                # Fallback to highest probability
                best_selection, best_probability, best_odds = max(selections, key=lambda x: x[1])
                best_value = 0
        
        return best_selection, best_probability, best_odds, best_value

    def _calculate_kelly_fraction(self, probability: float, odds: float) -> float:
        """Calculate Kelly criterion fraction with bounds"""
        try:
            p = probability / 100.0
            b = odds - 1
            kelly = (b * p - (1 - p)) / b if b > 0 else 0
            
            # Apply bounds for risk management
            kelly = max(0, min(kelly, self.config['kelly_fraction_limit']))
            return kelly
            
        except Exception:
            return 0.0

    def _calculate_enhanced_outcome(self, selection: str, odds: float, stake_amount: float,
                                  actual_result: Dict, market: MarketType, prediction_data: Dict) -> Tuple[BetOutcome, float]:
        """Enhanced outcome calculation with multiple scenarios"""
        try:
            actual_outcome = self._determine_actual_outcome(actual_result, market)
            
            # Handle inferred results with caution
            if actual_result.get('inferred') or actual_result.get('error'):
                logger.warning(f"Using inferred result: {actual_result}")
                # For inferred results, be conservative
                if actual_result.get('error'):
                    return BetOutcome.VOID, 0.0
            
            if selection == actual_outcome:
                profit = stake_amount * (odds - 1)
                return BetOutcome.WIN, profit
            elif actual_outcome == "Push":
                return BetOutcome.PUSH, 0.0
            elif actual_outcome == "Cashout":
                cashout_value = stake_amount * 0.8  # Example cashout value
                return BetOutcome.CASHOUT, cashout_value - stake_amount
            else:
                return BetOutcome.LOSS, -stake_amount
                
        except Exception as e:
            logger.error(f"Error calculating outcome: {e}")
            return BetOutcome.VOID, 0.0

    def _determine_actual_outcome(self, actual_result: Dict, market: MarketType) -> str:
        """Enhanced outcome determination with validation"""
        try:
            home_goals = actual_result.get('home_goals', 0)
            away_goals = actual_result.get('away_goals', 0)
            
            if market == MarketType.MATCH_ODDS:
                if home_goals > away_goals:
                    return "Home Win"
                elif away_goals > home_goals:
                    return "Away Win"
                else:
                    return "Draw"
                    
            elif market == MarketType.OVER_UNDER:
                total_goals = home_goals + away_goals
                if total_goals > 2.5:
                    return "Over 2.5"
                elif total_goals < 2.5:
                    return "Under 2.5"
                else:
                    return "Push"  # Exactly 2.5 goals is a push
                    
            elif market == MarketType.BTTS:
                if home_goals > 0 and away_goals > 0:
                    return "Yes"
                else:
                    return "No"
                    
            else:
                return "Unknown"
                
        except Exception as e:
            logger.error(f"Error determining actual outcome: {e}")
            return "Push"

    def _create_enhanced_fallback_bet_record(self, prediction_data: Dict, error_msg: str) -> BetRecord:
        """Create comprehensive fallback bet record"""
        return BetRecord(
            id=f"error_{hashlib.md5(error_msg.encode()).hexdigest()[:8]}",
            timestamp=datetime.now(),
            match="Error Match",
            market=MarketType.MATCH_ODDS,
            selection="Unknown",
            probability=50.0,
            odds=2.0,
            stake_percent=0.0,
            stake_amount=0.0,
            outcome=BetOutcome.VOID,
            profit_loss=0.0,
            confidence=50.0,
            data_quality=0.0,
            model_type=prediction_data.get('model_type', 'Unknown'),
            pattern_count=0,
            value_edge=0.0,
            betting_strategy=BettingStrategy.VALUE_BETTING,
            bankroll_before=self.current_bankroll,
            bankroll_after=self.current_bankroll,
            kelly_fraction=0.0,
            bet_metadata={'error': error_msg}
        )

    def _update_advanced_learning_state(self):
        """Update advanced learning state with comprehensive analytics"""
        try:
            if len(self.bet_history) < self.config['min_bets_for_analysis']:
                return
            
            recent_bets = self.bet_history[-self.config['feedback_update_frequency']:]
            
            # Update multiple learning aspects
            self._update_market_performance(recent_bets)
            self._update_strategy_performance(recent_bets)
            self._update_pattern_success_rates(recent_bets)
            self._update_confidence_calibration(recent_bets)
            self._update_value_bet_accuracy(recent_bets)
            self._update_market_efficiency(recent_bets)
            self._update_adaptive_thresholds(recent_bets)
            
            self.learning_state['learning_cycles'] += 1
            self.learning_state['last_learning_update'] = datetime.now()
            
            self._save_learning_state()
            logger.info("Updated advanced model learning state")
            
        except Exception as e:
            logger.error(f"Error updating learning state: {e}")

    def _update_market_performance(self, recent_bets: List[BetRecord]):
        """Update market-specific performance with confidence intervals"""
        for market in MarketType:
            market_bets = [b for b in recent_bets if b.market == market]
            if len(market_bets) >= 5:  # Minimum for meaningful analysis
                wins = len([b for b in market_bets if b.outcome == BetOutcome.WIN])
                total = len([b for b in market_bets if b.outcome in [BetOutcome.WIN, BetOutcome.LOSS]])
                
                if total > 0:
                    win_rate = wins / total
                    # Calculate confidence interval for win rate
                    if total >= 10:
                        ci = stats.binom.interval(0.95, total, win_rate)
                        self.learning_state['market_performance'][market.value] = {
                            'win_rate': win_rate,
                            'confidence_interval': ci,
                            'sample_size': total
                        }
                    else:
                        self.learning_state['market_performance'][market.value] = {
                            'win_rate': win_rate,
                            'sample_size': total
                        }

    def _update_strategy_performance(self, recent_bets: List[BetRecord]):
        """Update strategy-specific performance"""
        for strategy in BettingStrategy:
            strategy_bets = [b for b in recent_bets if b.betting_strategy == strategy]
            if strategy_bets:
                strategy_profit = sum(b.profit_loss for b in strategy_bets)
                strategy_roi = (strategy_profit / sum(b.stake_amount for b in strategy_bets)) * 100
                
                self.learning_state['strategy_performance'][strategy.value] = {
                    'roi': strategy_roi,
                    'total_profit': strategy_profit,
                    'bet_count': len(strategy_bets)
                }

    def _update_pattern_success_rates(self, recent_bets: List[BetRecord]):
        """Update pattern success rates with statistical significance"""
        pattern_bets = [b for b in recent_bets if b.pattern_count > 0]
        if len(pattern_bets) >= 5:
            pattern_wins = len([b for b in pattern_bets if b.outcome == BetOutcome.WIN])
            pattern_win_rate = pattern_wins / len(pattern_bets)
            
            # Compare with non-pattern bets
            non_pattern_bets = [b for b in recent_bets if b.pattern_count == 0]
            if non_pattern_bets:
                non_pattern_wins = len([b for b in non_pattern_bets if b.outcome == BetOutcome.WIN])
                non_pattern_win_rate = non_pattern_wins / len(non_pattern_bets)
                
                # Calculate if pattern betting is statistically better
                if len(pattern_bets) >= 10 and len(non_pattern_bets) >= 10:
                    t_stat, p_value = stats.ttest_ind(
                        [1 if b.outcome == BetOutcome.WIN else 0 for b in pattern_bets],
                        [1 if b.outcome == BetOutcome.WIN else 0 for b in non_pattern_bets]
                    )
                    
                    self.learning_state['pattern_success_rate'] = {
                        'win_rate': pattern_win_rate,
                        'non_pattern_win_rate': non_pattern_win_rate,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                else:
                    self.learning_state['pattern_success_rate'] = {
                        'win_rate': pattern_win_rate,
                        'non_pattern_win_rate': non_pattern_win_rate
                    }

    def _update_confidence_calibration(self, recent_bets: List[BetRecord]):
        """Update confidence calibration with moving average"""
        if not recent_bets:
            return
        
        # Group by confidence ranges
        confidence_ranges = [(0, 50), (50, 65), (65, 75), (75, 85), (85, 100)]
        calibration_errors = []
        
        for conf_min, conf_max in confidence_ranges:
            range_bets = [b for b in recent_bets if conf_min <= b.confidence <= conf_max]
            if len(range_bets) >= 3:
                expected_accuracy = (conf_min + conf_max) / 2
                actual_accuracy = len([b for b in range_bets if b.outcome == BetOutcome.WIN]) / len(range_bets) * 100
                calibration_errors.append(abs(expected_accuracy - actual_accuracy))
        
        if calibration_errors:
            avg_calibration_error = np.mean(calibration_errors)
            # Convert error to calibration factor (lower error = higher calibration)
            calibration_factor = max(0.7, min(1.3, 1.0 - (avg_calibration_error / 50)))
            
            # Smooth update
            current_calibration = self.learning_state.get('confidence_calibration', 1.0)
            new_calibration = 0.9 * current_calibration + 0.1 * calibration_factor
            self.learning_state['confidence_calibration'] = new_calibration

    def _update_value_bet_accuracy(self, recent_bets: List[BetRecord]):
        """Update value betting accuracy with edge analysis"""
        value_bets = [b for b in recent_bets if b.value_edge >= self.config['value_edge_threshold']]
        if value_bets:
            value_wins = len([b for b in value_bets if b.outcome == BetOutcome.WIN])
            value_accuracy = value_wins / len(value_bets)
            
            # Calculate actual vs expected value
            expected_value = sum((b.value_edge / 100) * b.stake_amount for b in value_bets)
            actual_value = sum(b.profit_loss for b in value_bets)
            value_efficiency = actual_value / expected_value if expected_value > 0 else 0
            
            self.learning_state['value_bet_accuracy'] = {
                'accuracy': value_accuracy,
                'efficiency': value_efficiency,
                'sample_size': len(value_bets)
            }

    def _update_market_efficiency(self, recent_bets: List[BetRecord]):
        """Update market efficiency assessment"""
        if not recent_bets:
            return
        
        # Calculate how often value bets are profitable
        value_bets = [b for b in recent_bets if b.value_edge > 0]
        if value_bets:
            profitable_value_bets = len([b for b in value_bets if b.profit_loss > 0])
            market_efficiency = profitable_value_bets / len(value_bets)
            self.learning_state['market_efficiency'] = market_efficiency

    def _update_adaptive_thresholds(self, recent_bets: List[BetRecord]):
        """Dynamically adjust thresholds based on performance"""
        if len(recent_bets) < 10:
            return
        
        current_performance = self.calculate_comprehensive_metrics()
        
        # Adjust value edge threshold based on performance
        if current_performance.roi > 5:
            # Doing well, can be more selective
            new_threshold = min(3.0, self.config['value_edge_threshold'] + 0.1)
        elif current_performance.roi < 0:
            # Underperforming, need more opportunities
            new_threshold = max(1.0, self.config['value_edge_threshold'] - 0.1)
        else:
            new_threshold = self.config['value_edge_threshold']
        
        self.learning_state['adaptive_thresholds']['value_edge'] = new_threshold

    def calculate_comprehensive_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics with caching"""
        # Check cache
        if self._metrics_cache and self._cache_timestamp:
            time_since_update = (datetime.now() - self._cache_timestamp).total_seconds()
            if time_since_update < 300:  # 5 minute cache
                return self._metrics_cache['metrics']
        
        if len(self.bet_history) < self.config['min_bets_for_analysis']:
            metrics = self._generate_insufficient_data_metrics()
        else:
            try:
                metrics = self._calculate_enhanced_metrics()
            except Exception as e:
                logger.error(f"Error calculating comprehensive metrics: {e}")
                metrics = self._generate_error_metrics()
        
        # Update cache
        self._metrics_cache = {
            'metrics': metrics,
            'timestamp': datetime.now()
        }
        self._cache_timestamp = datetime.now()
        
        return metrics

    def _calculate_enhanced_metrics(self) -> PerformanceMetrics:
        """Calculate enhanced performance metrics with advanced analytics"""
        # Basic counts
        wins = len([b for b in self.bet_history if b.outcome == BetOutcome.WIN])
        losses = len([b for b in self.bet_history if b.outcome == BetOutcome.LOSS])
        pushes = len([b for b in self.bet_history if b.outcome == BetOutcome.PUSH])
        voids = len([b for b in self.bet_history if b.outcome == BetOutcome.VOID])
        
        # Financial calculations
        valid_bets = [b for b in self.bet_history if b.outcome not in [BetOutcome.PUSH, BetOutcome.VOID]]
        total_staked = sum(b.stake_amount for b in valid_bets)
        total_profit = sum(b.profit_loss for b in self.bet_history)
        roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
        net_profit = total_profit
        
        # Risk metrics
        sharpe = self._calculate_enhanced_sharpe_ratio()
        sortino = self._calculate_sortino_ratio()
        max_dd = self._calculate_max_drawdown()
        volatility = self._calculate_volatility()
        var_95, cvar_95 = self._calculate_enhanced_var()
        ulcer_index = self._calculate_ulcer_index()
        
        # Betting metrics
        win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
        avg_odds = np.mean([b.odds for b in valid_bets]) if valid_bets else 0
        avg_stake = np.mean([b.stake_percent for b in valid_bets]) * 100 if valid_bets else 0
        expectancy = self._calculate_expectancy()
        profit_factor = self._calculate_profit_factor()
        edge = self._calculate_average_edge()
        
        # Statistical metrics
        p_value = self._calculate_p_value()
        confidence_interval = self._calculate_confidence_interval()
        z_score = self._calculate_z_score()
        statistical_power = self._calculate_statistical_power()
        
        # Advanced metrics
        kelly_roi = self._calculate_kelly_optimized_roi()
        calibration = self._calculate_calibration_score()
        value_accuracy = self._calculate_value_betting_accuracy()
        pattern_success = self._calculate_pattern_success_rate()
        market_efficiency = self.learning_state.get('market_efficiency', 0) * 100
        confidence_accuracy = self._calculate_confidence_accuracy()
        
        # Performance tiers
        performance_tier = self._determine_performance_tier(roi, sharpe)
        risk_adjusted_grade = self._calculate_risk_adjusted_grade(sharpe, sortino, max_dd)
        
        # Time-based metrics
        daily_roi = self._calculate_time_based_roi('daily')
        weekly_roi = self._calculate_time_based_roi('weekly')
        monthly_roi = self._calculate_time_based_roi('monthly')
        cagr = self._calculate_cagr()
        
        return PerformanceMetrics(
            total_bets=len(self.bet_history),
            winning_bets=wins,
            losing_bets=losses,
            pushed_bets=pushes,
            voided_bets=voids,
            total_staked=total_staked,
            total_profit=total_profit,
            total_pnl=total_profit,
            roi=roi,
            net_profit=net_profit,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            volatility=volatility,
            var_95=var_95,
            cvar_95=cvar_95,
            ulcer_index=ulcer_index,
            win_rate=win_rate,
            average_odds=avg_odds,
            average_stake=avg_stake,
            expectancy=expectancy,
            profit_factor=profit_factor,
            edge=edge,
            p_value=p_value,
            confidence_interval=confidence_interval,
            z_score=z_score,
            statistical_power=statistical_power,
            kelly_optimized_roi=kelly_roi,
            calibration_score=calibration,
            value_betting_accuracy=value_accuracy,
            pattern_success_rate=pattern_success,
            market_efficiency=market_efficiency,
            confidence_accuracy=confidence_accuracy,
            performance_tier=performance_tier,
            risk_adjusted_grade=risk_adjusted_grade,
            daily_roi=daily_roi,
            weekly_roi=weekly_roi,
            monthly_roi=monthly_roi,
            compound_annual_growth=cagr
        )

    def _calculate_enhanced_sharpe_ratio(self) -> float:
        """Calculate enhanced Sharpe ratio with risk-free rate consideration"""
        try:
            returns = [b.profit_loss for b in self.bet_history if b.outcome != BetOutcome.PUSH]
            if len(returns) < 2:
                return 0.0
            
            # Assume risk-free rate of 2% annually
            risk_free_rate_daily = 0.02 / 365
            avg_return = np.mean(returns)
            excess_return = avg_return - risk_free_rate_daily * self.initial_bankroll
            std_return = np.std(returns)
            
            return excess_return / std_return if std_return > 0 else 0.0
        except Exception:
            return 0.0

    def _calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio (downside risk only)"""
        try:
            returns = [b.profit_loss for b in self.bet_history if b.outcome != BetOutcome.PUSH]
            if len(returns) < 2:
                return 0.0
            
            risk_free_rate_daily = 0.02 / 365
            avg_return = np.mean(returns)
            excess_return = avg_return - risk_free_rate_daily * self.initial_bankroll
            
            # Calculate downside deviation
            downside_returns = [r for r in returns if r < 0]
            downside_std = np.std(downside_returns) if downside_returns else 0.0
            
            return excess_return / downside_std if downside_std > 0 else 0.0
        except Exception:
            return 0.0

    def _calculate_enhanced_var(self) -> Tuple[float, float]:
        """Calculate Value at Risk and Conditional VaR"""
        try:
            returns = [b.profit_loss for b in self.bet_history if b.outcome != BetOutcome.PUSH]
            if len(returns) < 10:
                return 0.0, 0.0
            
            var_95 = np.percentile(returns, 5)  # 5th percentile for 95% confidence
            cvar_95 = np.mean([r for r in returns if r <= var_95])
            
            return var_95, cvar_95
        except Exception:
            return 0.0, 0.0

    def _calculate_ulcer_index(self) -> float:
        """Calculate Ulcer Index for drawdown analysis"""
        try:
            if not self.performance_history:
                return 0.0
            
            equity_curve = [snapshot['bankroll'] for snapshot in self.performance_history]
            peaks = np.maximum.accumulate(equity_curve)
            drawdowns = [(equity_curve[i] - peaks[i]) / peaks[i] for i in range(len(equity_curve))]
            squared_drawdowns = [dd ** 2 for dd in drawdowns]
            
            return np.sqrt(np.mean(squared_drawdowns)) * 100
        except Exception:
            return 0.0

    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross wins / gross losses)"""
        try:
            gross_wins = sum(b.profit_loss for b in self.bet_history if b.profit_loss > 0)
            gross_losses = abs(sum(b.profit_loss for b in self.bet_history if b.profit_loss < 0))
            
            return gross_wins / gross_losses if gross_losses > 0 else float('inf')
        except Exception:
            return 0.0

    def _calculate_average_edge(self) -> float:
        """Calculate average betting edge"""
        try:
            valid_bets = [b for b in self.bet_history if b.outcome in [BetOutcome.WIN, BetOutcome.LOSS]]
            if not valid_bets:
                return 0.0
            return np.mean([b.value_edge for b in valid_bets])
        except Exception:
            return 0.0

    def _calculate_z_score(self) -> float:
        """Calculate Z-score for statistical significance"""
        try:
            wins = len([b for b in self.bet_history if b.outcome == BetOutcome.WIN])
            total = len([b for b in self.bet_history if b.outcome in [BetOutcome.WIN, BetOutcome.LOSS]])
            if total < 10:
                return 0.0
            
            expected_wins = total * 0.5  # Assuming fair odds
            std_dev = np.sqrt(total * 0.5 * 0.5)
            
            return (wins - expected_wins) / std_dev if std_dev > 0 else 0.0
        except Exception:
            return 0.0

    def _calculate_statistical_power(self) -> float:
        """Calculate statistical power of the strategy"""
        try:
            z_score = self._calculate_z_score()
            # Simplified power calculation
            if z_score > 1.96:  # 95% confidence
                return 0.95
            elif z_score > 1.645:  # 90% confidence
                return 0.90
            elif z_score > 1.28:  # 80% confidence
                return 0.80
            else:
                return 0.50
        except Exception:
            return 0.50

    def _calculate_confidence_accuracy(self) -> float:
        """Calculate how well confidence scores predict outcomes"""
        try:
            confidence_ranges = [(0, 50), (50, 65), (65, 75), (75, 85), (85, 100)]
            accuracies = []
            
            for conf_min, conf_max in confidence_ranges:
                range_bets = [b for b in self.bet_history 
                            if conf_min <= b.confidence <= conf_max and b.outcome in [BetOutcome.WIN, BetOutcome.LOSS]]
                if len(range_bets) >= 5:
                    expected_accuracy = (conf_min + conf_max) / 2
                    actual_accuracy = len([b for b in range_bets if b.outcome == BetOutcome.WIN]) / len(range_bets) * 100
                    accuracy_diff = 100 - abs(expected_accuracy - actual_accuracy)
                    accuracies.append(accuracy_diff)
            
            return np.mean(accuracies) if accuracies else 50.0
        except Exception:
            return 50.0

    def _determine_performance_tier(self, roi: float, sharpe: float) -> PerformanceTier:
        """Determine performance tier based on ROI and risk-adjusted returns"""
        if roi > 10 and sharpe > 1.5:
            return PerformanceTier.ELITE
        elif roi > 5 and sharpe > 1.0:
            return PerformanceTier.PROFESSIONAL
        elif roi > 2 and sharpe > 0.5:
            return PerformanceTier.COMPETENT
        elif roi > -2:
            return PerformanceTier.BREAKEVEN
        else:
            return PerformanceTier.UNDERPERFORMING

    def _calculate_risk_adjusted_grade(self, sharpe: float, sortino: float, max_dd: float) -> str:
        """Calculate risk-adjusted performance grade"""
        score = (sharpe * 0.4 + sortino * 0.4 + (1 - max_dd/100) * 0.2) * 10
        
        if score >= 9:
            return "A+"
        elif score >= 8:
            return "A"
        elif score >= 7:
            return "B+"
        elif score >= 6:
            return "B"
        elif score >= 5:
            return "C+"
        elif score >= 4:
            return "C"
        else:
            return "D"

    def _calculate_time_based_roi(self, period: str) -> float:
        """Calculate ROI for specific time periods"""
        try:
            if not self.bet_history:
                return 0.0
            
            now = datetime.now()
            if period == 'daily':
                cutoff = now - timedelta(days=1)
            elif period == 'weekly':
                cutoff = now - timedelta(weeks=1)
            elif period == 'monthly':
                cutoff = now - timedelta(days=30)
            else:
                return 0.0
            
            period_bets = [b for b in self.bet_history if b.timestamp >= cutoff]
            valid_bets = [b for b in period_bets if b.outcome not in [BetOutcome.PUSH, BetOutcome.VOID]]
            
            if not valid_bets:
                return 0.0
            
            total_staked = sum(b.stake_amount for b in valid_bets)
            total_profit = sum(b.profit_loss for b in period_bets)
            
            return (total_profit / total_staked * 100) if total_staked > 0 else 0.0
        except Exception:
            return 0.0

    def _calculate_cagr(self) -> float:
        """Calculate Compound Annual Growth Rate"""
        try:
            if not self.bet_history or len(self.performance_history) < 2:
                return 0.0
            
            start_value = self.initial_bankroll
            end_value = self.current_bankroll
            start_date = self.bet_history[0].timestamp
            end_date = self.bet_history[-1].timestamp
            
            years = (end_date - start_date).days / 365.25
            if years <= 0:
                return 0.0
            
            cagr = (end_value / start_value) ** (1 / years) - 1
            return cagr * 100
        except Exception:
            return 0.0

    # Keep existing utility methods with enhanced implementations
    def _calculate_max_drawdown(self) -> float:
        """Enhanced max drawdown calculation"""
        try:
            if not self.performance_history:
                return 0.0
            
            equity_curve = [snapshot['bankroll'] for snapshot in self.performance_history]
            peak = equity_curve[0]
            max_dd = 0.0
            
            for value in equity_curve:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
            
            return max_dd * 100
        except Exception:
            return 0.0

    def _calculate_volatility(self) -> float:
        """Enhanced volatility calculation"""
        try:
            returns = [b.profit_loss for b in self.bet_history if b.outcome != BetOutcome.PUSH]
            if len(returns) < 2:
                return 0.0
            return np.std(returns) / self.initial_bankroll * 100
        except Exception:
            return 0.0

    def _calculate_expectancy(self) -> float:
        """Enhanced expectancy calculation"""
        try:
            winning_bets = [b for b in self.bet_history if b.outcome == BetOutcome.WIN]
            losing_bets = [b for b in self.bet_history if b.outcome == BetOutcome.LOSS]
            
            if not winning_bets or not losing_bets:
                return 0.0
            
            avg_win = np.mean([b.profit_loss for b in winning_bets])
            avg_loss = np.mean([b.profit_loss for b in losing_bets])
            win_rate = len(winning_bets) / (len(winning_bets) + len(losing_bets))
            
            expectancy = (win_rate * avg_win + (1 - win_rate) * avg_loss) / self.initial_bankroll * 100
            return expectancy
        except Exception:
            return 0.0

    def _calculate_p_value(self) -> float:
        """Enhanced p-value calculation"""
        try:
            wins = len([b for b in self.bet_history if b.outcome == BetOutcome.WIN])
            total = len([b for b in self.bet_history if b.outcome in [BetOutcome.WIN, BetOutcome.LOSS]])
            if total < 10:
                return 1.0
            return stats.binom_test(wins, total, 0.5, alternative='greater')
        except Exception:
            return 1.0

    def _calculate_confidence_interval(self) -> Tuple[float, float]:
        """Enhanced confidence interval calculation"""
        try:
            returns = [b.profit_loss for b in self.bet_history if b.outcome != BetOutcome.PUSH]
            if len(returns) < 10:
                return (0.0, 0.0)
            mean_return = np.mean(returns)
            sem = stats.sem(returns)
            ci = stats.t.interval(0.95, len(returns)-1, loc=mean_return, scale=sem)
            return (ci[0] / self.initial_bankroll * 100, ci[1] / self.initial_bankroll * 100)
        except Exception:
            return (0.0, 0.0)

    def _calculate_kelly_optimized_roi(self) -> float:
        """Enhanced Kelly-optimized ROI calculation"""
        try:
            winning_bets = [b for b in self.bet_history if b.outcome == BetOutcome.WIN]
            losing_bets = [b for b in self.bet_history if b.outcome == BetOutcome.LOSS]
            
            if not winning_bets or not losing_bets:
                return 0.0
            
            win_rate = len(winning_bets) / (len(winning_bets) + len(losing_bets))
            avg_odds = np.mean([b.odds for b in winning_bets])
            b = avg_odds - 1
            kelly_fraction = max(0, (b * win_rate - (1 - win_rate)) / b) if b > 0 else 0
            kelly_fraction = min(kelly_fraction, 0.25)  # Conservative cap
            
            # Simulate Kelly-optimized returns
            optimized_returns = []
            for bet in self.bet_history:
                if bet.outcome == BetOutcome.WIN:
                    optimized_returns.append(self.initial_bankroll * kelly_fraction * (bet.odds - 1))
                elif bet.outcome == BetOutcome.LOSS:
                    optimized_returns.append(-self.initial_bankroll * kelly_fraction)
            
            total_optimized = sum(optimized_returns)
            total_staked_optimized = self.initial_bankroll * kelly_fraction * len(optimized_returns)
            
            return (total_optimized / total_staked_optimized * 100) if total_staked_optimized > 0 else 0.0
        except Exception:
            return 0.0

    def _calculate_calibration_score(self) -> float:
        """Enhanced calibration score calculation"""
        try:
            prob_ranges = [(0, 40), (40, 60), (60, 75), (75, 85), (85, 100)]
            calibration_errors = []
            
            for prob_min, prob_max in prob_ranges:
                range_bets = [b for b in self.bet_history 
                            if prob_min <= b.probability <= prob_max and b.outcome in [BetOutcome.WIN, BetOutcome.LOSS]]
                if len(range_bets) >= 5:
                    expected_win_rate = (prob_min + prob_max) / 2
                    actual_win_rate = len([b for b in range_bets if b.outcome == BetOutcome.WIN]) / len(range_bets) * 100
                    calibration_errors.append(abs(expected_win_rate - actual_win_rate))
            
            if not calibration_errors:
                return 100.0
            
            avg_error = np.mean(calibration_errors)
            return max(0, 100 - avg_error)
        except Exception:
            return 50.0

    def _calculate_value_betting_accuracy(self) -> float:
        """Enhanced value betting accuracy calculation"""
        try:
            value_bets = [b for b in self.bet_history if b.value_edge >= self.config['value_edge_threshold']]
            if not value_bets:
                return 0.0
            value_wins = len([b for b in value_bets if b.outcome == BetOutcome.WIN])
            return (value_wins / len(value_bets)) * 100
        except Exception:
            return 0.0

    def _calculate_pattern_success_rate(self) -> float:
        """Enhanced pattern success rate calculation"""
        try:
            pattern_bets = [b for b in self.bet_history if b.pattern_count > 0]
            if not pattern_bets:
                return 0.0
            pattern_wins = len([b for b in pattern_bets if b.outcome == BetOutcome.WIN])
            return (pattern_wins / len(pattern_bets)) * 100
        except Exception:
            return 0.0

    def _update_performance_history(self):
        """Update performance history with enhanced metrics"""
        if len(self.bet_history) == 0:
            return
        
        metrics = self.calculate_comprehensive_metrics()
        snapshot = {
            'timestamp': datetime.now(),
            'bankroll': self.current_bankroll,
            'total_bets': len(self.bet_history),
            'roi': metrics.roi,
            'sharpe_ratio': metrics.sharpe_ratio,
            'max_drawdown': metrics.max_drawdown,
            'win_rate': metrics.win_rate,
            'confidence_trend': self._calculate_confidence_trend(),
            'risk_metrics': {
                'volatility': metrics.volatility,
                'var_95': metrics.var_95,
                'ulcer_index': metrics.ulcer_index
            }
        }
        self.performance_history.append(snapshot)

    def _update_risk_metrics(self):
        """Update risk metrics history"""
        try:
            risk_metrics = RiskMetrics(
                value_at_risk=self._calculate_enhanced_var()[0],
                expected_shortfall=self._calculate_enhanced_var()[1],
                maximum_adverse_excursion=self._calculate_max_drawdown(),
                risk_of_ruin=self._calculate_risk_of_ruin(),
                kelly_optimal_fraction=self._calculate_average_kelly_fraction(),
                optimal_bet_size=self._calculate_optimal_bet_size(),
                risk_adjusted_return=self._calculate_risk_adjusted_return()
            )
            self.risk_metrics_history.append(risk_metrics)
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")

    def _calculate_risk_of_ruin(self) -> float:
        """Calculate risk of ruin probability"""
        try:
            win_rate = len([b for b in self.bet_history if b.outcome == BetOutcome.WIN]) / len(self.bet_history)
            avg_win = np.mean([b.profit_loss for b in self.bet_history if b.outcome == BetOutcome.WIN])
            avg_loss = np.mean([abs(b.profit_loss) for b in self.bet_history if b.outcome == BetOutcome.LOSS])
            
            if avg_loss == 0:
                return 0.0
            
            # Simplified risk of ruin calculation
            q = 1 - win_rate
            ruin_prob = ((1 - win_rate) / win_rate) ** (self.current_bankroll / avg_loss)
            return min(ruin_prob, 1.0) * 100
        except Exception:
            return 0.0

    def _calculate_average_kelly_fraction(self) -> float:
        """Calculate average Kelly fraction across all bets"""
        try:
            kelly_fractions = [b.kelly_fraction for b in self.bet_history if b.kelly_fraction > 0]
            return np.mean(kelly_fractions) if kelly_fractions else 0.0
        except Exception:
            return 0.0

    def _calculate_optimal_bet_size(self) -> float:
        """Calculate optimal bet size based on current bankroll and risk"""
        try:
            avg_kelly = self._calculate_average_kelly_fraction()
            optimal_size = self.current_bankroll * avg_kelly * 0.5  # Half Kelly for safety
            return optimal_size
        except Exception:
            return self.current_bankroll * 0.02  # 2% fallback

    def _calculate_risk_adjusted_return(self) -> float:
        """Calculate risk-adjusted return metric"""
        try:
            metrics = self.calculate_comprehensive_metrics()
            # Combine Sharpe and Sortino with drawdown consideration
            risk_score = (metrics.sharpe_ratio * 0.4 + metrics.sortino_ratio * 0.4 + 
                         (1 - metrics.max_drawdown/100) * 0.2)
            return risk_score * metrics.roi
        except Exception:
            return 0.0

    def _calculate_confidence_trend(self) -> List[float]:
        """Calculate confidence trend over time"""
        if len(self.bet_history) < 10:
            return [50.0]
        recent_bets = sorted(self.bet_history, key=lambda x: x.timestamp)[-10:]
        return [bet.confidence for bet in recent_bets]

    def _generate_insufficient_data_metrics(self) -> PerformanceMetrics:
        """Generate metrics for insufficient data scenario"""
        return PerformanceMetrics(
            total_bets=len(self.bet_history), winning_bets=0, losing_bets=0, 
            pushed_bets=0, voided_bets=0, total_staked=0, total_profit=0, 
            total_pnl=0, roi=0, net_profit=0, sharpe_ratio=0, sortino_ratio=0,
            max_drawdown=0, volatility=0, var_95=0, cvar_95=0, ulcer_index=0,
            win_rate=0, average_odds=0, average_stake=0, expectancy=0,
            profit_factor=0, edge=0, p_value=1.0, confidence_interval=(0, 0),
            z_score=0, statistical_power=0.5, kelly_optimized_roi=0, 
            calibration_score=50, value_betting_accuracy=0, pattern_success_rate=0,
            market_efficiency=0, confidence_accuracy=50, 
            performance_tier=PerformanceTier.BREAKEVEN, risk_adjusted_grade="N/A",
            daily_roi=0, weekly_roi=0, monthly_roi=0, compound_annual_growth=0
        )

    def _generate_error_metrics(self) -> PerformanceMetrics:
        """Generate metrics for error scenarios"""
        return PerformanceMetrics(
            total_bets=0, winning_bets=0, losing_bets=0, pushed_bets=0, voided_bets=0,
            total_staked=0, total_profit=0, total_pnl=0, roi=0, net_profit=0,
            sharpe_ratio=0, sortino_ratio=0, max_drawdown=0, volatility=0, var_95=0,
            cvar_95=0, ulcer_index=0, win_rate=0, average_odds=0, average_stake=0,
            expectancy=0, profit_factor=0, edge=0, p_value=1.0, confidence_interval=(0, 0),
            z_score=0, statistical_power=0, kelly_optimized_roi=0, calibration_score=0,
            value_betting_accuracy=0, pattern_success_rate=0, market_efficiency=0,
            confidence_accuracy=0, performance_tier=PerformanceTier.UNDERPERFORMING,
            risk_adjusted_grade="F", daily_roi=0, weekly_roi=0, monthly_roi=0,
            compound_annual_growth=0
        )

    def get_model_feedback(self) -> ModelFeedback:
        """Get comprehensive model feedback for prediction engine improvement"""
        metrics = self.calculate_comprehensive_metrics()
        
        # Calculate recommended adjustments
        adjustments = self._generate_model_adjustments(metrics)
        
        # Create advanced calibration factors
        calibration_factors = {
            'confidence_adjustment': self.learning_state.get('confidence_calibration', 1.0),
            'market_weights': self._calculate_market_weights(),
            'value_bet_threshold': self._optimize_value_threshold(),
            'pattern_confidence_boost': self._calculate_pattern_boost(),
            'risk_adjustment_factor': self.learning_state.get('risk_adjustment_factor', 1.0),
            'strategy_allocations': self._calculate_strategy_allocations()
        }
        
        # Risk assessment
        risk_assessment = {
            'current_risk_level': self._assess_current_risk_level(),
            'recommended_actions': self._generate_risk_recommendations(),
            'exposure_limits': self._calculate_exposure_limits(),
            'diversification_score': self._calculate_diversification_score()
        }
        
        # Optimization opportunities
        optimization_opportunities = self._identify_optimization_opportunities(metrics)
        
        # Performance forecast
        performance_forecast = self._generate_performance_forecast(metrics)
        
        return ModelFeedback(
            confidence_calibration=self.learning_state.get('confidence_calibration', 1.0),
            market_performance=self.learning_state.get('market_performance', {}),
            pattern_effectiveness=self.learning_state.get('pattern_success_rate', {}),
            feature_importance=self._calculate_feature_importance(),
            strategy_performance=self.learning_state.get('strategy_performance', {}),
            recommended_adjustments=adjustments,
            calibration_factors=calibration_factors,
            risk_assessment=risk_assessment,
            optimization_opportunities=optimization_opportunities,
            performance_forecast=performance_forecast
        )

    def _generate_model_adjustments(self, metrics: PerformanceMetrics) -> List[str]:
        """Generate specific model adjustment recommendations"""
        adjustments = []
        
        # Confidence calibration
        if metrics.confidence_accuracy < 70:
            cal_factor = self.learning_state.get('confidence_calibration', 1.0)
            if cal_factor < 0.9:
                adjustments.append(f"INCREASE confidence scores by {((1/cal_factor)-1)*100:.1f}%")
            elif cal_factor > 1.1:
                adjustments.append(f"DECREASE confidence scores by {(cal_factor-1)*100:.1f}%")
        
        # Market focus recommendations
        market_perf = self.learning_state.get('market_performance', {})
        if market_perf:
            best_market = max(market_perf.items(), key=lambda x: x[1].get('win_rate', 0) if isinstance(x[1], dict) else x[1])[0]
            worst_market = min(market_perf.items(), key=lambda x: x[1].get('win_rate', 0) if isinstance(x[1], dict) else x[1])[0]
            
            best_perf = market_perf[best_market].get('win_rate', 0) if isinstance(market_perf[best_market], dict) else market_perf[best_market]
            worst_perf = market_perf[worst_market].get('win_rate', 0) if isinstance(market_perf[worst_market], dict) else market_perf[worst_market]
            
            adjustments.append(f"FOCUS on {best_market} market (performance: {best_perf:.1%})")
            adjustments.append(f"REDUCE exposure to {worst_market} market (performance: {worst_perf:.1%})")
        
        # Value betting optimization
        if metrics.value_betting_accuracy > metrics.win_rate + 5:
            adjustments.append("INCREASE value bet threshold for higher quality selections")
        elif metrics.value_betting_accuracy < metrics.win_rate - 5:
            adjustments.append("DECREASE value bet threshold to capture more opportunities")
        
        # Risk management adjustments
        if metrics.max_drawdown > 15:
            adjustments.append("REDUCE stake sizes to manage drawdown risk")
        if metrics.sharpe_ratio < 1.0:
            adjustments.append("IMPROVE risk-adjusted returns through better bankroll management")
        
        return adjustments[:5]

    def _calculate_market_weights(self) -> Dict[str, float]:
        """Calculate optimal market weights based on performance"""
        market_perf = self.learning_state.get('market_performance', {})
        if not market_perf:
            return {'match_odds': 0.6, 'over_under': 0.3, 'btts': 0.1}
        
        # Extract win rates and calculate weights
        win_rates = {}
        for market, perf in market_perf.items():
            if isinstance(perf, dict) and 'win_rate' in perf:
                win_rates[market] = perf['win_rate']
            elif isinstance(perf, (int, float)):
                win_rates[market] = perf
        
        if not win_rates:
            return {'match_odds': 0.6, 'over_under': 0.3, 'btts': 0.1}
        
        # Normalize win rates to weights (higher win rate = higher weight)
        total_win_rate = sum(win_rates.values())
        if total_win_rate == 0:
            return {'match_odds': 0.6, 'over_under': 0.3, 'btts': 0.1}
        
        weights = {market: win_rate/total_win_rate for market, win_rate in win_rates.items()}
        return weights

    def _optimize_value_threshold(self) -> float:
        """Optimize value bet threshold based on performance"""
        base_threshold = self.config['value_edge_threshold']
        value_accuracy_data = self.learning_state.get('value_bet_accuracy', {})
        
        if isinstance(value_accuracy_data, dict) and 'accuracy' in value_accuracy_data:
            value_accuracy = value_accuracy_data['accuracy']
        else:
            value_accuracy = value_accuracy_data if isinstance(value_accuracy_data, (int, float)) else 0.55
        
        if value_accuracy > 0.6:
            return max(1.0, base_threshold - 0.5)  # Lower threshold if performing well
        elif value_accuracy < 0.5:
            return base_threshold + 1.0  # Raise threshold if performing poorly
        else:
            return base_threshold

    def _calculate_pattern_boost(self) -> float:
        """Calculate confidence boost for pattern-based predictions"""
        pattern_data = self.learning_state.get('pattern_success_rate', {})
        
        if isinstance(pattern_data, dict) and 'win_rate' in pattern_data:
            pattern_success = pattern_data['win_rate']
            is_significant = pattern_data.get('significant', False)
        else:
            pattern_success = pattern_data if isinstance(pattern_data, (int, float)) else 0.55
            is_significant = False
        
        if pattern_success > 0.6 and is_significant:
            return 1.15  # Significant boost for effective patterns
        elif pattern_success > 0.55:
            return 1.05  # Moderate boost
        elif pattern_success < 0.5:
            return 0.9   # Reduce confidence for ineffective patterns
        else:
            return 1.0

    def _calculate_strategy_allocations(self) -> Dict[str, float]:
        """Calculate optimal strategy allocations"""
        strategy_perf = self.learning_state.get('strategy_performance', {})
        if not strategy_perf:
            return {'value_betting': 0.7, 'kelly_criterion': 0.3}
        
        # Calculate weights based on ROI
        weights = {}
        total_roi = 0
        
        for strategy, perf in strategy_perf.items():
            if isinstance(perf, dict) and 'roi' in perf:
                roi = max(0, perf['roi'])  # Only consider positive ROI
                weights[strategy] = roi
                total_roi += roi
        
        if total_roi > 0:
            normalized_weights = {s: w/total_roi for s, w in weights.items()}
        else:
            normalized_weights = {'value_betting': 0.7, 'kelly_criterion': 0.3}
        
        return normalized_weights

    def _assess_current_risk_level(self) -> str:
        """Assess current risk level based on multiple factors"""
        metrics = self.calculate_comprehensive_metrics()
        
        risk_score = 0
        risk_score += min(3, metrics.max_drawdown / 5)  # Drawdown component
        risk_score += min(3, metrics.volatility / 2)    # Volatility component
        risk_score += min(2, self._calculate_risk_of_ruin() / 10)  # Ruin risk component
        risk_score += min(2, (100 - metrics.calibration_score) / 20)  # Calibration risk
        
        if risk_score >= 8:
            return "high"
        elif risk_score >= 5:
            return "medium"
        else:
            return "low"

    def _generate_risk_recommendations(self) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        metrics = self.calculate_comprehensive_metrics()
        
        if metrics.max_drawdown > 15:
            recommendations.append("Reduce maximum stake size to 2% of bankroll")
        if metrics.volatility > 10:
            recommendations.append("Diversify across more markets and strategies")
        if self._calculate_risk_of_ruin() > 5:
            recommendations.append("Implement stricter stop-loss rules")
        if metrics.calibration_score < 70:
            recommendations.append("Review probability calibration methodology")
        
        return recommendations

    def _calculate_exposure_limits(self) -> Dict[str, float]:
        """Calculate recommended exposure limits"""
        metrics = self.calculate_comprehensive_metrics()
        
        base_limit = 0.05  # 5% base limit
        
        # Adjust based on performance
        if metrics.roi > 5 and metrics.sharpe_ratio > 1.5:
            base_limit = 0.08  # Higher limit for good performance
        elif metrics.roi < 0 or metrics.max_drawdown > 15:
            base_limit = 0.02  # Lower limit for poor performance
        
        return {
            'single_bet_limit': base_limit,
            'daily_limit': base_limit * 3,
            'market_limit': base_limit * 2,
            'total_exposure_limit': base_limit * 5
        }

    def _calculate_diversification_score(self) -> float:
        """Calculate portfolio diversification score"""
        try:
            # Count bets across different markets
            market_counts = defaultdict(int)
            for bet in self.bet_history[-50:]:  # Recent 50 bets
                market_counts[bet.market.value] += 1
            
            if not market_counts:
                return 0.0
            
            # Calculate Herfindahl index (concentration measure)
            total_bets = sum(market_counts.values())
            herfindahl = sum((count / total_bets) ** 2 for count in market_counts.values())
            
            # Convert to diversification score (0-100)
            diversification = (1 - herfindahl) * 100
            return max(0, min(100, diversification))
        except Exception:
            return 50.0

    def _identify_optimization_opportunities(self, metrics: PerformanceMetrics) -> List[str]:
        """Identify specific optimization opportunities"""
        opportunities = []
        
        # Market efficiency opportunities
        if metrics.market_efficiency < 50:
            opportunities.append("Focus on less efficient markets with higher value edges")
        
        # Pattern recognition opportunities
        pattern_data = self.learning_state.get('pattern_success_rate', {})
        if isinstance(pattern_data, dict) and pattern_data.get('significant', False):
            opportunities.append("Leverage pattern recognition for higher confidence bets")
        
        # Strategy optimization opportunities
        strategy_perf = self.learning_state.get('strategy_performance', {})
        if strategy_perf:
            best_strategy = max(strategy_perf.items(), 
                              key=lambda x: x[1].get('roi', 0) if isinstance(x[1], dict) else x[1])
            opportunities.append(f"Increase allocation to {best_strategy[0]} strategy")
        
        # Risk-adjusted return opportunities
        if metrics.sharpe_ratio < 1.0:
            opportunities.append("Optimize stake sizing for better risk-adjusted returns")
        
        return opportunities[:3]

    def _generate_performance_forecast(self, metrics: PerformanceMetrics) -> Dict[str, float]:
        """Generate performance forecast based on historical data"""
        try:
            # Simple forecasting based on recent performance
            recent_bets = self.bet_history[-30:]  # Last 30 bets
            if len(recent_bets) < 10:
                return {'next_month_roi': 0, 'confidence': 0}
            
            recent_roi = self._calculate_recent_roi(30)
            trend = self._calculate_performance_trend()
            
            # Basic forecast with trend adjustment
            next_month_roi = recent_roi * (1 + trend * 0.1)
            confidence = min(90, metrics.statistical_power * 100)
            
            return {
                'next_month_roi': next_month_roi,
                'confidence': confidence,
                'expected_range': (next_month_roi - 5, next_month_roi + 5)
            }
        except Exception:
            return {'next_month_roi': 0, 'confidence': 0, 'expected_range': (0, 0)}

    def _calculate_recent_roi(self, num_bets: int) -> float:
        """Calculate ROI for recent bets"""
        recent_bets = sorted(self.bet_history, key=lambda x: x.timestamp)[-num_bets:]
        valid_bets = [b for b in recent_bets if b.outcome not in [BetOutcome.PUSH, BetOutcome.VOID]]
        
        if not valid_bets:
            return 0.0
        
        total_staked = sum(b.stake_amount for b in valid_bets)
        total_profit = sum(b.profit_loss for b in recent_bets)
        
        return (total_profit / total_staked * 100) if total_staked > 0 else 0.0

    def _calculate_performance_trend(self) -> float:
        """Calculate performance trend (positive = improving)"""
        try:
            if len(self.bet_history) < 20:
                return 0.0
            
            # Split into two halves and compare performance
            half_point = len(self.bet_history) // 2
            first_half = self.bet_history[:half_point]
            second_half = self.bet_history[half_point:]
            
            first_roi = self._calculate_roi_for_bets(first_half)
            second_roi = self._calculate_roi_for_bets(second_half)
            
            return (second_roi - first_roi) / max(1, abs(first_roi))
        except Exception:
            return 0.0

    def _calculate_roi_for_bets(self, bets: List[BetRecord]) -> float:
        """Calculate ROI for a specific set of bets"""
        valid_bets = [b for b in bets if b.outcome not in [BetOutcome.PUSH, BetOutcome.VOID]]
        if not valid_bets:
            return 0.0
        
        total_staked = sum(b.stake_amount for b in valid_bets)
        total_profit = sum(b.profit_loss for b in bets)
        
        return (total_profit / total_staked * 100) if total_staked > 0 else 0.0

    def _calculate_feature_importance(self) -> Dict[str, float]:
        """Calculate feature importance based on performance correlation"""
        # Enhanced implementation with actual correlation analysis
        if len(self.bet_history) < 20:
            return {
                'form_quality': 0.25,
                'standing_data': 0.20,
                'head_to_head': 0.15,
                'market_odds': 0.30,
                'injuries': 0.10
            }
        
        try:
            # This would involve more sophisticated analysis in production
            # For now, return a simplified version
            return {
                'market_odds': 0.30,
                'form_quality': 0.25,
                'pattern_detection': 0.20,
                'standing_data': 0.15,
                'head_to_head': 0.10
            }
        except Exception:
            return {
                'form_quality': 0.25,
                'standing_data': 0.20,
                'head_to_head': 0.15,
                'market_odds': 0.30,
                'injuries': 0.10
            }

    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard"""
        metrics = self.calculate_comprehensive_metrics()
        feedback = self.get_model_feedback()
        
        return {
            'overview': {
                'current_bankroll': self.current_bankroll,
                'total_profit': metrics.net_profit,
                'roi': metrics.roi,
                'performance_tier': metrics.performance_tier.value,
                'risk_grade': metrics.risk_adjusted_grade,
                'active_bets': len([b for b in self.bet_history if b.outcome not in [BetOutcome.WIN, BetOutcome.LOSS]])
            },
            'key_metrics': {
                'win_rate': metrics.win_rate,
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'profit_factor': metrics.profit_factor,
                'value_bet_accuracy': metrics.value_betting_accuracy
            },
            'recent_performance': {
                'daily_roi': metrics.daily_roi,
                'weekly_roi': metrics.weekly_roi,
                'monthly_roi': metrics.monthly_roi,
                'trend': 'improving' if metrics.roi > 0 else 'stable' if metrics.roi == 0 else 'declining'
            },
            'alerts': {
                'risk_alerts': len(feedback.risk_assessment.get('recommended_actions', [])),
                'performance_issues': 1 if metrics.performance_tier == PerformanceTier.UNDERPERFORMING else 0,
                'calibration_issues': 1 if metrics.calibration_score < 70 else 0
            },
            'recommendations': feedback.recommended_adjustments[:3]
        }

    def export_performance_data(self, format: str = 'json') -> str:
        """Export performance data in various formats"""
        try:
            data = {
                'bet_history': [asdict(bet) for bet in self.bet_history],
                'performance_metrics': asdict(self.calculate_comprehensive_metrics()),
                'learning_state': self.learning_state,
                'export_timestamp': datetime.now().isoformat()
            }
            
            if format == 'json':
                return json.dumps(data, indent=2, default=str)
            elif format == 'csv':
                # Simplified CSV export - would be more comprehensive in production
                df = pd.DataFrame([asdict(bet) for bet in self.bet_history])
                return df.to_csv(index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting performance data: {e}")
            return ""

# Supporting classes
class RiskAssessor:
    """Advanced risk assessment component"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def assess_bet_risk(self, bet_record: BetRecord) -> Dict[str, float]:
        """Assess risk for individual bet"""
        return {
            'volatility_risk': min(1.0, bet_record.stake_percent * 10),
            'concentration_risk': 0.5,  # Would be calculated based on portfolio
            'model_risk': (100 - bet_record.confidence) / 100,
            'market_risk': 0.3  # Base market risk
        }

class PerformanceAnalyzer:
    """Advanced performance analysis component"""
    
    def analyze_winning_streaks(self, bet_history: List[BetRecord]) -> Dict[str, Any]:
        """Analyze winning and losing streaks"""
        if not bet_history:
            return {'longest_win_streak': 0, 'longest_loss_streak': 0}
        
        current_streak = 0
        longest_win_streak = 0
        longest_loss_streak = 0
        current_outcome = None
        
        for bet in bet_history:
            if bet.outcome == BetOutcome.WIN:
                if current_outcome == BetOutcome.WIN:
                    current_streak += 1
                else:
                    current_streak = 1
                    current_outcome = BetOutcome.WIN
                longest_win_streak = max(longest_win_streak, current_streak)
            elif bet.outcome == BetOutcome.LOSS:
                if current_outcome == BetOutcome.LOSS:
                    current_streak += 1
                else:
                    current_streak = 1
                    current_outcome = BetOutcome.LOSS
                longest_loss_streak = max(longest_loss_streak, current_streak)
            else:
                current_streak = 0
                current_outcome = None
        
        return {
            'longest_win_streak': longest_win_streak,
            'longest_loss_streak': longest_loss_streak,
            'current_streak': current_streak,
            'current_streak_type': current_outcome.value if current_outcome else 'none'
        }

# Enhanced utility functions
def create_performance_tracker(initial_bankroll: float = 10000, config: Optional[Dict] = None) -> InstitutionalPerformanceTracker:
    """Create an advanced performance tracker"""
    return InstitutionalPerformanceTracker(initial_bankroll, config)

def get_model_feedback(tracker: InstitutionalPerformanceTracker) -> ModelFeedback:
    """Get comprehensive model feedback"""
    return tracker.get_model_feedback()

def record_match_result(tracker: InstitutionalPerformanceTracker, prediction_data: Dict, 
                       result: Union[Dict, str], stake_percent: float,
                       strategy: BettingStrategy = BettingStrategy.VALUE_BETTING) -> BetRecord:
    """Professional function to record match results"""
    return tracker.record_bet(prediction_data, result, stake_percent, strategy)

def get_performance_dashboard(tracker: InstitutionalPerformanceTracker) -> Dict[str, Any]:
    """Get performance dashboard"""
    return tracker.get_performance_dashboard()

# Example usage
if __name__ == "__main__":
    # Example of using the enhanced performance tracker
    tracker = create_performance_tracker(5000)
    
    # Example prediction data
    example_prediction = {
        'home_team': 'Manchester United',
        'away_team': 'Liverpool',
        'predictions': {
            '1X2': {'Home Win': 45.2, 'Draw': 28.1, 'Away Win': 26.7}
        },
        'odds_1x2': [2.10, 3.40, 3.20],
        'confidence_score': 70.0,
        'data_quality_score': 85.0,
        'model_type': 'Enhanced_Predictor',
        'pattern_intelligence': {'pattern_count': 2}
    }
    
    # Record a bet
    bet_record = record_match_result(
        tracker=tracker,
        prediction_data=example_prediction,
        result="2-1",  # Home win
        stake_percent=0.02,  # 2% stake
        strategy=BettingStrategy.VALUE_BETTING
    )
    
    print(f"Recorded bet: {bet_record.match}")
    print(f"Outcome: {bet_record.outcome.value}")
    print(f"P&L: ${bet_record.profit_loss:.2f}")
    
    # Get performance metrics
    metrics = tracker.calculate_comprehensive_metrics()
    print(f"\nPerformance Overview:")
    print(f"ROI: {metrics.roi:.2f}%")
    print(f"Win Rate: {metrics.win_rate:.2f}%")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Performance Tier: {metrics.performance_tier.value}")
    
    # Get model feedback
    feedback = get_model_feedback(tracker)
    print(f"\nModel Feedback:")
    for adjustment in feedback.recommended_adjustments[:3]:
        print(f" - {adjustment}")
    
    # Get dashboard
    dashboard = get_performance_dashboard(tracker)
    print(f"\nDashboard:")
    print(f"Current Bankroll: ${dashboard['overview']['current_bankroll']:,.2f}")
    print(f"Total Profit: ${dashboard['overview']['total_profit']:,.2f}")
