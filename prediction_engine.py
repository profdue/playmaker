# betting_engine.py - COMPLETE PRODUCTION-READY VERSION
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BettingSignal:
    """Container for betting recommendations"""
    market: str
    model_probability: float
    market_odds: float
    edge_percentage: float
    expected_value: float
    recommended_stake: float
    confidence: str
    kelly_fraction: float
    explanation: List[str]

@dataclass
class Bankroll:
    """Bankroll management system"""
    total_amount: float
    max_stake_percent: float = 0.05  # Max 5% per bet
    kelly_fraction: float = 0.25     # Fractional Kelly
    min_stake: float = 10.0          # Minimum stake amount
    
    def validate_stake(self, stake: float) -> Tuple[bool, str]:
        """Validate stake against bankroll constraints"""
        stake_percent = stake / self.total_amount
        
        if stake < self.min_stake:
            return False, f"Stake below minimum: {stake} < {self.min_stake}"
        elif stake_percent > self.max_stake_percent:
            return False, f"Stake exceeds maximum percentage: {stake_percent:.1%} > {self.max_stake_percent:.1%}"
        elif stake > self.total_amount:
            return False, "Stake exceeds bankroll"
        
        return True, "Valid stake"

class ExpectedValueCalculator:
    """Expected value calculation engine"""
    
    @staticmethod
    def calculate_ev(probability: float, odds: float) -> Dict:
        """Calculate expected value and edge"""
        if odds <= 1.0:
            return {'edge_percentage': 0.0, 'expected_value': 0.0, 'implied_probability': 0.0}
        
        implied_prob = 1.0 / odds
        edge = probability - implied_prob
        ev = (probability * (odds - 1)) - ((1 - probability) * 1)
        
        return {
            'edge_percentage': edge * 100,
            'expected_value': ev,
            'implied_probability': implied_prob,
            'market_vig': (implied_prob - probability) * 100 if implied_prob > probability else 0.0
        }
    
    @staticmethod
    def kelly_criterion(probability: float, odds: float) -> float:
        """Calculate Kelly Criterion stake percentage"""
        if odds <= 1.0:
            return 0.0
        
        b = odds - 1  # Decimal odds to multiplier
        q = 1 - probability
        kelly = (probability * (b + 1) - 1) / b
        
        return max(0.0, kelly)  # Only positive bets

class MarketAnalyzer:
    """Analyze betting markets for value opportunities"""
    
    def __init__(self):
        self.min_edge = 0.02  # Minimum 2% edge
        self.min_confidence_prob = 0.52  # Minimum probability for consideration
        self.market_mappings = {
            'btts_yes': 'BTTS Yes',
            'btts_no': 'BTTS No', 
            'over_25': 'Over 2.5 Goals',
            'under_25': 'Under 2.5 Goals',
            'home_win': '1x2 Home',
            'draw': '1x2 Draw',
            'away_win': '1x2 Away'
        }
    
    def analyze_markets(self, predictions: Dict, market_odds: Dict, 
                       bankroll: Bankroll) -> List[BettingSignal]:
        """Analyze all markets for value bets"""
        
        signals = []
        
        # Analyze BTTS markets
        btts_signals = self._analyze_btts_markets(predictions, market_odds, bankroll)
        signals.extend(btts_signals)
        
        # Analyze Over/Under markets
        ou_signals = self._analyze_ou_markets(predictions, market_odds, bankroll)
        signals.extend(ou_signals)
        
        # Analyze 1X2 markets
        match_signals = self._analyze_match_markets(predictions, market_odds, bankroll)
        signals.extend(match_signals)
        
        # Sort by expected value (highest first)
        signals.sort(key=lambda x: x.expected_value, reverse=True)
        
        return signals
    
    def _analyze_btts_markets(self, predictions: Dict, market_odds: Dict, 
                            bankroll: Bankroll) -> List[BettingSignal]:
        """Analyze Both Teams to Score markets"""
        signals = []
        
        btts_probs = predictions['probabilities']['both_teams_score']
        btts_yes_prob = btts_probs['yes'] / 100.0
        btts_no_prob = btts_probs['no'] / 100.0
        
        # BTTS Yes
        if 'BTTS Yes' in market_odds:
            odds = market_odds['BTTS Yes']
            ev_data = ExpectedValueCalculator.calculate_ev(btts_yes_prob, odds)
            
            if self._is_valuable_bet(btts_yes_prob, ev_data):
                signal = self._create_signal(
                    market='BTTS Yes',
                    probability=btts_yes_prob,
                    odds=odds,
                    ev_data=ev_data,
                    bankroll=bankroll,
                    explanation=self._generate_btts_explanation(btts_yes_prob, 'yes', predictions)
                )
                signals.append(signal)
        
        # BTTS No
        if 'BTTS No' in market_odds:
            odds = market_odds['BTTS No']
            ev_data = ExpectedValueCalculator.calculate_ev(btts_no_prob, odds)
            
            if self._is_valuable_bet(btts_no_prob, ev_data):
                signal = self._create_signal(
                    market='BTTS No',
                    probability=btts_no_prob,
                    odds=odds,
                    ev_data=ev_data,
                    bankroll=bankroll,
                    explanation=self._generate_btts_explanation(btts_no_prob, 'no', predictions)
                )
                signals.append(signal)
        
        return signals
    
    def _analyze_ou_markets(self, predictions: Dict, market_odds: Dict, 
                          bankroll: Bankroll) -> List[BettingSignal]:
        """Analyze Over/Under markets"""
        signals = []
        
        ou_probs = predictions['probabilities']['over_under']
        over_25_prob = ou_probs['over_25'] / 100.0
        under_25_prob = ou_probs['under_25'] / 100.0
        
        # Over 2.5
        if 'Over 2.5 Goals' in market_odds:
            odds = market_odds['Over 2.5 Goals']
            ev_data = ExpectedValueCalculator.calculate_ev(over_25_prob, odds)
            
            if self._is_valuable_bet(over_25_prob, ev_data):
                signal = self._create_signal(
                    market='Over 2.5 Goals',
                    probability=over_25_prob,
                    odds=odds,
                    ev_data=ev_data,
                    bankroll=bankroll,
                    explanation=self._generate_ou_explanation(over_25_prob, 'over', predictions)
                )
                signals.append(signal)
        
        # Under 2.5
        if 'Under 2.5 Goals' in market_odds:
            odds = market_odds['Under 2.5 Goals']
            ev_data = ExpectedValueCalculator.calculate_ev(under_25_prob, odds)
            
            if self._is_valuable_bet(under_25_prob, ev_data):
                signal = self._create_signal(
                    market='Under 2.5 Goals',
                    probability=under_25_prob,
                    odds=odds,
                    ev_data=ev_data,
                    bankroll=bankroll,
                    explanation=self._generate_ou_explanation(under_25_prob, 'under', predictions)
                )
                signals.append(signal)
        
        return signals
    
    def _analyze_match_markets(self, predictions: Dict, market_odds: Dict, 
                             bankroll: Bankroll) -> List[BettingSignal]:
        """Analyze 1X2 match outcome markets"""
        signals = []
        
        match_probs = predictions['probabilities']['match_outcomes']
        home_prob = match_probs['home_win'] / 100.0
        draw_prob = match_probs['draw'] / 100.0
        away_prob = match_probs['away_win'] / 100.0
        
        # Home Win
        if '1x2 Home' in market_odds:
            odds = market_odds['1x2 Home']
            ev_data = ExpectedValueCalculator.calculate_ev(home_prob, odds)
            
            if self._is_valuable_bet(home_prob, ev_data):
                signal = self._create_signal(
                    market='1x2 Home',
                    probability=home_prob,
                    odds=odds,
                    ev_data=ev_data,
                    bankroll=bankroll,
                    explanation=[f"Model predicts {home_prob:.1%} chance of home win"]
                )
                signals.append(signal)
        
        # Draw
        if '1x2 Draw' in market_odds:
            odds = market_odds['1x2 Draw']
            ev_data = ExpectedValueCalculator.calculate_ev(draw_prob, odds)
            
            if self._is_valuable_bet(draw_prob, ev_data):
                signal = self._create_signal(
                    market='1x2 Draw',
                    probability=draw_prob,
                    odds=odds,
                    ev_data=ev_data,
                    bankroll=bankroll,
                    explanation=[f"Model predicts {draw_prob:.1%} chance of draw"]
                )
                signals.append(signal)
        
        # Away Win
        if '1x2 Away' in market_odds:
            odds = market_odds['1x2 Away']
            ev_data = ExpectedValueCalculator.calculate_ev(away_prob, odds)
            
            if self._is_valuable_bet(away_prob, ev_data):
                signal = self._create_signal(
                    market='1x2 Away',
                    probability=away_prob,
                    odds=odds,
                    ev_data=ev_data,
                    bankroll=bankroll,
                    explanation=[f"Model predicts {away_prob:.1%} chance of away win"]
                )
                signals.append(signal)
        
        return signals
    
    def _is_valuable_bet(self, probability: float, ev_data: Dict) -> bool:
        """Determine if a bet meets value criteria"""
        return (probability >= self.min_confidence_prob and 
                ev_data['expected_value'] >= self.min_edge and
                ev_data['edge_percentage'] > 0.0)
    
    def _create_signal(self, market: str, probability: float, odds: float, 
                      ev_data: Dict, bankroll: Bankroll, explanation: List[str]) -> BettingSignal:
        """Create a betting signal with proper stake calculation"""
        
        # Calculate Kelly stake
        kelly_pct = ExpectedValueCalculator.kelly_criterion(probability, odds)
        fractional_kelly = kelly_pct * bankroll.kelly_fraction
        
        # Calculate stake amount
        stake_amount = fractional_kelly * bankroll.total_amount
        
        # Validate and adjust stake
        is_valid, validation_msg = bankroll.validate_stake(stake_amount)
        if not is_valid:
            logger.warning(f"Stake validation failed for {market}: {validation_msg}")
            stake_amount = bankroll.min_stake  # Fallback to minimum stake
        
        # Determine confidence level
        confidence = self._determine_confidence(probability, ev_data['edge_percentage'])
        
        return BettingSignal(
            market=market,
            model_probability=probability,
            market_odds=odds,
            edge_percentage=ev_data['edge_percentage'],
            expected_value=ev_data['expected_value'],
            recommended_stake=stake_amount,
            confidence=confidence,
            kelly_fraction=fractional_kelly,
            explanation=explanation
        )
    
    def _determine_confidence(self, probability: float, edge_percentage: float) -> str:
        """Determine confidence level based on probability and edge"""
        if probability > 0.67 and edge_percentage > 15:
            return "HIGH"
        elif probability > 0.57 and edge_percentage > 8:
            return "MEDIUM"
        elif probability > 0.52 and edge_percentage > 4:
            return "LOW"
        else:
            return "SPECULATIVE"
    
    def _generate_btts_explanation(self, probability: float, side: str, 
                                 predictions: Dict) -> List[str]:
        """Generate BTTS market explanations"""
        explanations = []
        
        exp_goals = predictions['expected_goals']
        clean_sheet = predictions['simulation_stats']['clean_sheet_prob']
        
        if side == 'yes':
            if probability > 0.6:
                explanations.append("Both teams show strong attacking capabilities")
                explanations.append(f"Expected goals: Home {exp_goals['home']}, Away {exp_goals['away']}")
            if clean_sheet['home'] < 0.3 or clean_sheet['away'] < 0.3:
                explanations.append("Low clean sheet probability for both teams")
        else:
            if probability > 0.6:
                explanations.append("Strong defensive organization from one or both teams")
                if clean_sheet['home'] > 0.4 or clean_sheet['away'] > 0.4:
                    explanations.append("High clean sheet probability indicates defensive strength")
        
        return explanations
    
    def _generate_ou_explanation(self, probability: float, side: str, 
                               predictions: Dict) -> List[str]:
        """Generate Over/Under market explanations"""
        explanations = []
        
        total_xg = predictions['expected_goals']['total']
        exp_goals = predictions['expected_goals']
        
        if side == 'over':
            if probability > 0.6:
                explanations.append(f"High expected goal total: {total_xg}")
                explanations.append(f"Attacking strength: Home {exp_goals['home']}, Away {exp_goals['away']}")
            if total_xg > 3.0:
                explanations.append("Very high combined expected goals")
        else:
            if probability > 0.6:
                explanations.append(f"Low expected goal total: {total_xg}")
                explanations.append("Defensive stability from both teams")
            if total_xg < 2.0:
                explanations.append("Very low combined expected goals")
        
        return explanations

class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self, max_daily_bets: int = 5, max_daily_exposure: float = 0.15):
        self.max_daily_bets = max_daily_bets
        self.max_daily_exposure = max_daily_exposure  # Max 15% of bankroll per day
        self.daily_tracking = {}
    
    def assess_portfolio_risk(self, signals: List[BettingSignal], bankroll: Bankroll) -> Dict:
        """Assess overall portfolio risk"""
        
        total_exposure = sum(signal.recommended_stake for signal in signals)
        exposure_percent = total_exposure / bankroll.total_amount
        
        risk_assessment = {
            'total_exposure': total_exposure,
            'exposure_percentage': exposure_percent,
            'number_of_bets': len(signals),
            'average_confidence': np.mean([self._confidence_to_number(s.confidence) for s in signals]),
            'risk_level': 'LOW',
            'recommendations': []
        }
        
        # Risk level determination
        if exposure_percent > self.max_daily_exposure:
            risk_assessment['risk_level'] = 'HIGH'
            risk_assessment['recommendations'].append(
                f"Reduce exposure: {exposure_percent:.1%} > {self.max_daily_exposure:.1%}"
            )
        
        if len(signals) > self.max_daily_bets:
            risk_assessment['risk_level'] = 'MEDIUM'
            risk_assessment['recommendations'].append(
                f"Reduce number of bets: {len(signals)} > {self.max_daily_bets}"
            )
        
        # Correlation risk (simplified - in practice would use more sophisticated correlation matrix)
        market_types = [s.market for s in signals]
        unique_markets = len(set(market_types))
        if unique_markets < len(market_types) / 2:
            risk_assessment['risk_level'] = 'MEDIUM'
            risk_assessment['recommendations'].append("High correlation between recommended bets")
        
        return risk_assessment
    
    def _confidence_to_number(self, confidence: str) -> float:
        """Convert confidence to numerical value"""
        confidence_map = {
            'HIGH': 0.9,
            'MEDIUM': 0.7, 
            'LOW': 0.5,
            'SPECULATIVE': 0.3
        }
        return confidence_map.get(confidence, 0.5)

class BettingDecisionEngine:
    """Main betting decision engine"""
    
    def __init__(self, bankroll_amount: float = 1000.0):
        self.bankroll = Bankroll(total_amount=bankroll_amount)
        self.market_analyzer = MarketAnalyzer()
        self.risk_manager = RiskManager()
        self.betting_history = []
    
    def generate_recommendations(self, predictions: Dict, market_odds: Dict) -> Dict:
        """Generate comprehensive betting recommendations"""
        
        logger.info("Generating betting recommendations...")
        
        # Analyze markets for value
        signals = self.market_analyzer.analyze_markets(predictions, market_odds, self.bankroll)
        
        # Assess portfolio risk
        risk_assessment = self.risk_manager.assess_portfolio_risk(signals, self.bankroll)
        
        # Generate summary
        summary = self._generate_summary(signals, risk_assessment)
        
        # Record in history
        self._record_betting_decision(signals, predictions['match'])
        
        return {
            'match': predictions['match'],
            'timestamp': datetime.now().isoformat(),
            'bankroll': self.bankroll.total_amount,
            'signals': [signal.__dict__ for signal in signals],
            'risk_assessment': risk_assessment,
            'summary': summary,
            'recommendation_count': len(signals)
        }
    
    def _generate_summary(self, signals: List[BettingSignal], risk_assessment: Dict) -> Dict:
        """Generate betting summary"""
        
        if not signals:
            return {
                'overview': 'No value bets identified',
                'total_stake': 0.0,
                'total_ev': 0.0,
                'average_edge': 0.0
            }
        
        total_stake = sum(s.recommended_stake for s in signals)
        total_ev = sum(s.expected_value * s.recommended_stake for s in signals)
        average_edge = np.mean([s.edge_percentage for s in signals])
        
        # Find best bet
        best_bet = max(signals, key=lambda x: x.expected_value) if signals else None
        
        return {
            'overview': f"Found {len(signals)} value bets with {risk_assessment['risk_level']} risk",
            'total_stake': total_stake,
            'total_ev': total_ev,
            'average_edge': average_edge,
            'best_bet': best_bet.market if best_bet else 'None',
            'best_bet_ev': best_bet.expected_value if best_bet else 0.0,
            'risk_level': risk_assessment['risk_level']
        }
    
    def _record_betting_decision(self, signals: List[BettingSignal], match: str):
        """Record betting decision in history"""
        decision = {
            'timestamp': datetime.now().isoformat(),
            'match': match,
            'signals': [{
                'market': s.market,
                'stake': s.recommended_stake,
                'odds': s.market_odds,
                'probability': s.model_probability,
                'edge': s.edge_percentage
            } for s in signals],
            'total_stake': sum(s.recommended_stake for s in signals)
        }
        self.betting_history.append(decision)
        
        # Keep only last 100 decisions
        if len(self.betting_history) > 100:
            self.betting_history = self.betting_history[-100:]
    
    def update_bankroll(self, amount: float):
        """Update bankroll amount"""
        self.bankroll.total_amount = amount
        logger.info(f"Bankroll updated to: {amount}")

# Example usage and test
def test_betting_engine():
    """Test the betting engine with sample data"""
    
    # Sample predictions (from prediction engine)
    sample_predictions = {
        'match': 'Liverpool vs Arsenal',
        'league': 'premier_league',
        'expected_goals': {'home': 2.1, 'away': 1.4, 'total': 3.5},
        'probabilities': {
            'both_teams_score': {'yes': 68.5, 'no': 31.5},
            'over_under': {'over_25': 72.3, 'under_25': 27.7},
            'match_outcomes': {'home_win': 55.2, 'draw': 25.1, 'away_win': 19.7}
        },
        'simulation_stats': {
            'clean_sheet_prob': {'home': 25.3, 'away': 18.2}
        },
        'confidence': 'HIGH'
    }
    
    # Sample market odds
    sample_odds = {
        'BTTS Yes': 1.80,
        'BTTS No': 2.10,
        'Over 2.5 Goals': 1.65,
        'Under 2.5 Goals': 2.20,
        '1x2 Home': 2.10,
        '1x2 Draw': 3.50,
        '1x2 Away': 3.80
    }
    
    # Initialize and run betting engine
    betting_engine = BettingDecisionEngine(bankroll_amount=1000.0)
    recommendations = betting_engine.generate_recommendations(sample_predictions, sample_odds)
    
    return recommendations

if __name__ == "__main__":
    # Run test
    results = test_betting_engine()
    
    print("💰 BETTING ENGINE TEST RESULTS")
    print("=" * 50)
    print(f"Match: {results['match']}")
    print(f"Bankroll: ${results['bankroll']:,.2f}")
    print(f"Recommendations: {results['recommendation_count']}")
    print(f"Risk Level: {results['risk_assessment']['risk_level']}")
    
    print("\n🎯 BETTING SIGNALS:")
    for signal in results['signals']:
        print(f"  {signal['market']}:")
        print(f"    Probability: {signal['model_probability']:.1%}")
        print(f"    Odds: {signal['market_odds']}")
        print(f"    Edge: +{signal['edge_percentage']:.1f}%")
        print(f"    Stake: ${signal['recommended_stake']:.2f}")
        print(f"    Confidence: {signal['confidence']}")
    
    print(f"\n📊 SUMMARY:")
    print(f"  Total Stake: ${results['summary']['total_stake']:.2f}")
    print(f"  Total EV: ${results['summary']['total_ev']:.2f}")
    print(f"  Best Bet: {results['summary']['best_bet']}")
