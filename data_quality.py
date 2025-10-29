import logging
from typing import Dict, Any, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

def calculate_data_quality(match_data: Dict[str, Any]) -> float:
    """
    Calculate comprehensive data quality score (0-100) - UPDATED FOR STREAMLIT APP STRUCTURE
    """
    try:
        score = 0
        max_score = 100
        
        # 1. Basic match info (20 points)
        if match_data.get('home_team') and match_data.get('away_team'):
            score += 20
        
        # 2. Team performance data (30 points) - USING GOALS DATA STRUCTURE
        home_goals = match_data.get('home_goals_data', {})
        away_goals = match_data.get('away_goals_data', {})
        
        # Check if goals data exists and has reasonable values
        if home_goals and isinstance(home_goals, dict):
            if home_goals.get('goals_scored', 0) > 0 or home_goals.get('goals_conceded', 0) > 0:
                score += 15
        else:
            # Partial credit if basic goals data exists
            if match_data.get('home_goals_scored') is not None:
                score += 10
                
        if away_goals and isinstance(away_goals, dict):
            if away_goals.get('goals_scored', 0) > 0 or away_goals.get('goals_conceded', 0) > 0:
                score += 15
        else:
            # Partial credit if basic goals data exists
            if match_data.get('away_goals_scored') is not None:
                score += 10
        
        # 3. Head-to-head data (25 points) - UPDATED FOR HYBRID H2H STRUCTURE
        h2h_aggregate = match_data.get('h2h_aggregate', {})
        h2h_recent = match_data.get('h2h_recent_matches', [])
        h2h_legacy = match_data.get('head_to_head', [])
        
        # Check aggregate H2H
        if h2h_aggregate.get('total_matches', 0) >= 3:
            score += 10
        elif h2h_aggregate.get('total_matches', 0) > 0:
            score += 5
            
        # Check recent H2H matches
        if len(h2h_recent) >= 2:
            score += 10
        elif len(h2h_recent) > 0:
            score += 5
            
        # Check legacy H2H format
        if len(h2h_legacy) >= 3:
            score += 5
        
        # 4. Market odds (15 points)
        odds_1x2 = match_data.get('odds_1x2')
        if odds_1x2 and len(odds_1x2) == 3 and all(odd > 1.0 for odd in odds_1x2):
            score += 15
        elif odds_1x2 and len(odds_1x2) == 3:
            score += 10  # Partial credit if odds exist but might be invalid
        
        # 5. League context and standings (10 points)
        has_league_context = (match_data.get('league_type') and 
                             match_data.get('match_importance') and 
                             match_data.get('venue_context'))
        
        has_standings = (match_data.get('home_standing') and 
                        match_data.get('away_standing'))
        
        if has_league_context and has_standings:
            score += 10
        elif has_league_context or has_standings:
            score += 5
        
        return min(100, score)
        
    except Exception as e:
        logger.error(f"Data quality calculation error: {e}")
        return 50.0  # Return 50 for graceful degradation

def get_data_quality_message(quality_score: float) -> str:
    """Get appropriate message for data quality score"""
    if quality_score >= 90:
        return "🎯 EXCELLENT QUALITY - Optimal prediction conditions"
    elif quality_score >= 75:
        return "✅ VERY GOOD QUALITY - Strong prediction foundation"
    elif quality_score >= 60:
        return "👍 GOOD QUALITY - Solid prediction foundation"
    elif quality_score >= 45:
        return "⚠️ FAIR QUALITY - Predictions should be used with caution"
    elif quality_score >= 30:
        return "🔶 LIMITED QUALITY - Significant data gaps present"
    else:
        return "❌ POOR QUALITY - Predictions highly uncertain"

def get_missing_data_suggestions(match_data: Dict[str, Any]) -> List[str]:
    """Get suggestions for improving data quality - UPDATED FOR STREAMLIT APP"""
    suggestions = []
    
    # Check goals data (primary performance metric)
    home_goals = match_data.get('home_goals_data', {})
    away_goals = match_data.get('away_goals_data', {})
    
    if not home_goals or not isinstance(home_goals, dict) or home_goals.get('goals_scored', 0) == 0:
        suggestions.append("🎯 Add home team recent performance data (goals scored/conceded in last 6 matches)")
    if not away_goals or not isinstance(away_goals, dict) or away_goals.get('goals_scored', 0) == 0:
        suggestions.append("🎯 Add away team recent performance data (goals scored/conceded in last 6 matches)")
    
    # Check H2H data - comprehensive check
    h2h_aggregate = match_data.get('h2h_aggregate', {})
    h2h_recent = match_data.get('h2h_recent_matches', [])
    
    total_h2h_matches = h2h_aggregate.get('total_matches', 0)
    if total_h2h_matches < 3:
        suggestions.append("🤝 Add more head-to-head history (3+ matches in aggregate data)")
    
    if len(h2h_recent) < 2:
        suggestions.append("📊 Add detailed recent H2H matches for enhanced pattern analysis")
    
    # Check if H2H has goal data (important for timing intelligence)
    if total_h2h_matches > 0 and (h2h_aggregate.get('home_goals', 0) == 0 and h2h_aggregate.get('away_goals', 0) == 0):
        suggestions.append("⚽ Add H2H goal data for enhanced timing intelligence")
    
    # Check odds
    odds_1x2 = match_data.get('odds_1x2')
    if not odds_1x2 or len(odds_1x2) != 3:
        suggestions.append("💰 Add match odds (1X2 market) for value analysis")
    elif any(odd <= 1.0 for odd in odds_1x2):
        suggestions.append("💰 Verify odds values (should be greater than 1.0)")
    
    # Check context data
    if not match_data.get('league_type'):
        suggestions.append("🏆 Specify league type for context-aware predictions")
    
    if not match_data.get('match_importance'):
        suggestions.append("🎯 Specify match importance for motivational factors")
        
    if not match_data.get('venue_context'):
        suggestions.append("🏟️ Specify venue context for home advantage analysis")
    
    # Check standings
    if not match_data.get('home_standing') or not match_data.get('away_standing'):
        suggestions.append("📈 Add league standings data for team strength assessment")
    
    return suggestions

def validate_match_data_legacy(match_data: Dict[str, Any]) -> List[str]:
    """
    Validate match data for legacy compatibility - UPDATED FOR STREAMLIT STRUCTURE
    Returns list of error messages, empty if valid
    """
    errors = []
    
    try:
        # Required fields
        if not match_data.get('home_team'):
            errors.append("Home team name is required")
        if not match_data.get('away_team'):
            errors.append("Away team name is required")
        
        # Validate goals data structure
        home_goals = match_data.get('home_goals_data', {})
        away_goals = match_data.get('away_goals_data', {})
        
        if home_goals and not isinstance(home_goals, dict):
            errors.append("Home goals data should be a dictionary")
        else:
            if 'goals_scored' in home_goals and not isinstance(home_goals['goals_scored'], (int, float)):
                errors.append("Home goals_scored should be a number")
            if 'goals_conceded' in home_goals and not isinstance(home_goals['goals_conceded'], (int, float)):
                errors.append("Home goals_conceded should be a number")
        
        if away_goals and not isinstance(away_goals, dict):
            errors.append("Away goals data should be a dictionary")
        else:
            if 'goals_scored' in away_goals and not isinstance(away_goals['goals_scored'], (int, float)):
                errors.append("Away goals_scored should be a number")
            if 'goals_conceded' in away_goals and not isinstance(away_goals['goals_conceded'], (int, float)):
                errors.append("Away goals_conceded should be a number")
        
        # Validate H2H aggregate data
        h2h_aggregate = match_data.get('h2h_aggregate', {})
        if h2h_aggregate:
            total_matches = h2h_aggregate.get('total_matches', 0)
            home_wins = h2h_aggregate.get('home_wins', 0)
            away_wins = h2h_aggregate.get('away_wins', 0)
            draws = h2h_aggregate.get('draws', 0)
            
            if total_matches > 0 and (home_wins + away_wins + draws) != total_matches:
                errors.append(f"H2H stats don't match: {home_wins}W + {away_wins}W + {draws}D = {home_wins + away_wins + draws}, but total = {total_matches}")
        
        # Validate recent H2H matches
        h2h_recent = match_data.get('h2h_recent_matches', [])
        for i, match in enumerate(h2h_recent):
            if not isinstance(match, dict):
                errors.append(f"Recent H2H match {i+1} should be a dictionary")
                continue
                
            if 'home_goals' not in match or 'away_goals' not in match:
                errors.append(f"Recent H2H match {i+1} missing home_goals or away_goals")
            else:
                if not isinstance(match['home_goals'], (int, float)):
                    errors.append(f"Recent H2H match {i+1} home_goals should be a number")
                if not isinstance(match['away_goals'], (int, float)):
                    errors.append(f"Recent H2H match {i+1} away_goals should be a number")
        
        # Validate legacy H2H format
        h2h_legacy = match_data.get('head_to_head', [])
        for i, match in enumerate(h2h_legacy):
            if not isinstance(match, list) or len(match) != 2:
                errors.append(f"Legacy H2H match {i+1} should be [home_goals, away_goals]")
                continue
            if not all(isinstance(goals, (int, float)) for goals in match):
                errors.append(f"Legacy H2H match {i+1} goals should be numbers")
        
        # Validate odds
        odds_1x2 = match_data.get('odds_1x2')
        if odds_1x2:
            if len(odds_1x2) != 3:
                errors.append("Odds 1X2 should have exactly 3 values [home, draw, away]")
            elif any(odd <= 1.0 for odd in odds_1x2):
                errors.append("All odds should be greater than 1.0")
            elif any(odd > 100.0 for odd in odds_1x2):
                errors.append("Odds seem unrealistically high (check values)")
        
        # Validate standings
        home_standing = match_data.get('home_standing')
        away_standing = match_data.get('away_standing')
        
        if home_standing and len(home_standing) < 4:
            errors.append("Home standing data incomplete (need position, points, played, goal difference)")
        if away_standing and len(away_standing) < 4:
            errors.append("Away standing data incomplete (need position, points, played, goal difference)")
        
        return errors
        
    except Exception as e:
        errors.append(f"Data validation error: {str(e)}")
        return errors

def validate_match_data(match_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Enhanced match data validation
    Returns (is_valid, error_messages)
    """
    errors = validate_match_data_legacy(match_data)
    return len(errors) == 0, errors

def get_data_completeness(match_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Get completeness scores for different data categories - UPDATED FOR STREAMLIT APP
    """
    completeness = {}
    
    try:
        # Basic info completeness (20%)
        basic_fields = ['home_team', 'away_team']
        complete_basic = sum(1 for field in basic_fields if match_data.get(field)) / len(basic_fields)
        completeness['basic_info'] = complete_basic * 100
        
        # Performance data completeness (30%) - USING GOALS DATA
        home_goals = match_data.get('home_goals_data', {})
        away_goals = match_data.get('away_goals_data', {})
        
        performance_score = 0
        if home_goals and isinstance(home_goals, dict) and (home_goals.get('goals_scored', 0) > 0 or home_goals.get('goals_conceded', 0) > 0):
            performance_score += 0.5
        if away_goals and isinstance(away_goals, dict) and (away_goals.get('goals_scored', 0) > 0 or away_goals.get('goals_conceded', 0) > 0):
            performance_score += 0.5
        completeness['performance'] = performance_score * 100
        
        # H2H data completeness (25%)
        h2h_aggregate = match_data.get('h2h_aggregate', {})
        h2h_recent = match_data.get('h2h_recent_matches', [])
        
        h2h_score = 0
        if h2h_aggregate.get('total_matches', 0) >= 3:
            h2h_score += 0.5
        elif h2h_aggregate.get('total_matches', 0) > 0:
            h2h_score += 0.3
            
        if len(h2h_recent) >= 2:
            h2h_score += 0.5
        elif len(h2h_recent) > 0:
            h2h_score += 0.2
            
        completeness['h2h'] = min(100, h2h_score * 100)
        
        # Market data completeness (15%)
        odds_present = bool(match_data.get('odds_1x2') and len(match_data['odds_1x2']) == 3)
        completeness['market_data'] = 100 if odds_present else 0
        
        # Context completeness (10%)
        context_fields = ['league_type', 'match_importance', 'venue_context']
        context_complete = sum(1 for field in context_fields if match_data.get(field)) / len(context_fields)
        completeness['context'] = context_complete * 100
        
        return completeness
        
    except Exception as e:
        logger.error(f"Data completeness calculation error: {e}")
        return {category: 0.0 for category in ['basic_info', 'performance', 'h2h', 'market_data', 'context']}

def enhance_data_quality_suggestions(match_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get detailed suggestions for enhancing data quality - UPDATED FOR STREAMLIT APP
    """
    suggestions = {
        'critical': [],
        'important': [],
        'optional': []
    }
    
    quality_score = calculate_data_quality(match_data)
    completeness = get_data_completeness(match_data)
    
    # Critical suggestions (score < 60)
    if quality_score < 60:
        if completeness['basic_info'] < 100:
            suggestions['critical'].append("Add both team names")
        if completeness['performance'] < 50:
            suggestions['critical'].append("Add recent performance data (goals scored/conceded for both teams)")
        if completeness['h2h'] < 30:
            suggestions['critical'].append("Add basic head-to-head history (3+ matches)")
    
    # Important suggestions (score 60-80)
    if quality_score < 80:
        if completeness['h2h'] < 70:
            suggestions['important'].append("Add more detailed H2H data (recent matches + goal data)")
        if completeness['market_data'] < 100:
            suggestions['important'].append("Add current market odds for value analysis")
        if completeness['context'] < 70:
            suggestions['important'].append("Add match context (league type, importance, venue)")
        if not match_data.get('home_standing') or not match_data.get('away_standing'):
            suggestions['important'].append("Add league standings for team strength assessment")
    
    # Optional enhancements (score 80+)
    if quality_score >= 80:
        if completeness['h2h'] < 90:
            suggestions['optional'].append("Add comprehensive H2H data for enhanced pattern detection")
        if completeness['context'] < 100:
            suggestions['optional'].append("Complete all context fields for optimal predictions")
        if completeness['market_data'] < 100:
            suggestions['optional'].append("Add additional market odds (BTTS, Over/Under)")
    
    return suggestions

def calculate_prediction_confidence(data_quality: float, data_completeness: Dict[str, float]) -> float:
    """
    Calculate prediction confidence based on data quality and completeness
    """
    try:
        # Base confidence from overall quality
        base_confidence = data_quality / 100.0
        
        # Weight completeness factors for institutional predictions
        weights = {
            'basic_info': 0.10,
            'performance': 0.35,  # Higher weight for performance data
            'h2h': 0.25,          # High weight for historical patterns
            'market_data': 0.20,   # Important for value analysis
            'context': 0.10        # Contextual factors
        }
        
        # Calculate weighted completeness
        weighted_completeness = sum(
            data_completeness.get(category, 0) / 100.0 * weight
            for category, weight in weights.items()
        )
        
        # Combined confidence (0.0 to 1.0)
        combined_confidence = (base_confidence * 0.6) + (weighted_completeness * 0.4)
        
        # Apply confidence scaling for institutional use
        institutional_confidence = combined_confidence * 0.9  # Conservative adjustment
        
        return min(1.0, max(0.0, institutional_confidence))
        
    except Exception as e:
        logger.error(f"Prediction confidence calculation error: {e}")
        return 0.5

# Legacy compatibility functions
def validate_input_data(match_data: Dict[str, Any]) -> List[str]:
    """Legacy alias for validate_match_data_legacy"""
    return validate_match_data_legacy(match_data)

def get_quality_score(match_data: Dict[str, Any]) -> float:
    """Legacy alias for calculate_data_quality"""
    return calculate_data_quality(match_data)

# Test function to verify everything works
def test_data_quality_module():
    """Test the data quality module with sample Streamlit app data"""
    test_data = {
        'home_team': 'Union Berlin',
        'away_team': 'Borussia M\'gladbach',
        'home_goals_data': {'goals_scored': 8, 'goals_conceded': 13, 'matches_scored': 3},
        'away_goals_data': {'goals_scored': 5, 'goals_conceded': 12, 'matches_scored': 2},
        'h2h_aggregate': {'total_matches': 6, 'home_wins': 3, 'away_wins': 2, 'draws': 1, 'home_goals': 8, 'away_goals': 5},
        'h2h_recent_matches': [
            {'home_team': 'Union Berlin', 'away_team': 'Borussia M\'gladbach', 'home_goals': 2, 'away_goals': 0},
            {'home_team': 'Borussia M\'gladbach', 'away_team': 'Union Berlin', 'home_goals': 1, 'away_goals': 1}
        ],
        'odds_1x2': [2.33, 3.44, 3.33],
        'league_type': 'Bundesliga',
        'match_importance': 'Normal League',
        'venue_context': 'Normal',
        'home_standing': [13, 7, 6, -5],
        'away_standing': [17, 3, 6, -7]
    }
    
    print("🧪 Testing Data Quality Module...")
    
    # Test quality score
    score = calculate_data_quality(test_data)
    print(f"📊 Quality Score: {score}/100")
    
    # Test quality message
    message = get_data_quality_message(score)
    print(f"💬 Quality Message: {message}")
    
    # Test suggestions
    suggestions = get_missing_data_suggestions(test_data)
    print(f"🎯 Suggestions: {suggestions}")
    
    # Test validation
    is_valid, errors = validate_match_data(test_data)
    print(f"✅ Validation: {'PASS' if is_valid else 'FAIL'}")
    if errors:
        print(f"❌ Errors: {errors}")
    
    # Test completeness
    completeness = get_data_completeness(test_data)
    print(f"📈 Completeness: {completeness}")
    
    # Test confidence
    confidence = calculate_prediction_confidence(score, completeness)
    print(f"🎯 Prediction Confidence: {confidence:.1%}")
    
    print("✅ Data Quality Module Test Complete!")

# Run test if executed directly
if __name__ == "__main__":
    test_data_quality_module()
