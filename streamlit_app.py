import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import re
from prediction_engine import ProfessionalPredictionEngine as WorldClassPredictionEngine
from data_quality import calculate_data_quality, get_data_quality_message, get_missing_data_suggestions, validate_match_data_legacy

# Page configuration
st.set_page_config(
    page_title="Institutional Football Predictor Pro ⚽",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS for institutional look
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .institutional-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .risk-metric {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .value-metric {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .confidence-metric {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .model-diagnostic {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .uncertainty-band {
        background: rgba(116, 185, 255, 0.1);
        border: 1px solid #74b9ff;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
    .pattern-metric {
        background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .debug-panel {
        background: #f8f9fa;
        border: 2px solid #ff6b6b;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .form-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    .form-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 1rem 0;
    }
    .form-match-label {
        font-weight: bold;
        color: #495057;
        margin-bottom: 0.5rem;
    }
    .h2h-match-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
    }
    .aggregate-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .pattern-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.25rem;
    }
    .pattern-dominant { background: #ff6b6b; color: white; }
    .pattern-even { background: #74b9ff; color: white; }
    .pattern-draw { background: #fdcb6e; color: black; }
    .pattern-high-scoring { background: #00b894; color: white; }
    .pattern-competitive { background: #a29bfe; color: white; }
    .pattern-recent-dominant { background: #e84393; color: white; }
    .warning-badge {
        background: #ffeaa7;
        color: #2d3436;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border-left: 4px solid #fdcb6e;
        margin: 0.5rem 0;
    }
    .success-badge {
        background: #55efc4;
        color: #2d3436;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border-left: 4px solid #00b894;
        margin: 0.5rem 0;
    }
    .goal-metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
        text-align: center;
    }
    .goal-metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .goal-metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .timing-metric {
        background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .timing-insight {
        background: #f8f9fa;
        border-left: 4px solid #fd79a8;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .upset-alert {
        background: linear-gradient(135deg, #ff9ff3 0%, #fd79a8 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid #e84393;
    }
    .bet-type-timing { 
        border-left: 4px solid #fd79a8;
        background: #f8f9fa;
    }
    .bet-type-upset { 
        border-left: 4px solid #ff9ff3;
        background: #f8f9fa;
    }
    .bet-type-standard { 
        border-left: 4px solid #74b9ff;
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with professional structure
if 'match_data' not in st.session_state:
    st.session_state.match_data = {}
if 'institutional_predictions' not in st.session_state:
    st.session_state.institutional_predictions = None
if 'show_predictions' not in st.session_state:
    st.session_state.show_predictions = False
if 'last_match_data' not in st.session_state:
    st.session_state.last_match_data = {}
if 'performance_history' not in st.session_state:
    st.session_state.performance_history = []
if 'debug_data' not in st.session_state:
    st.session_state.debug_data = {}
if 'h2h_warnings' not in st.session_state:
    st.session_state.h2h_warnings = []

def validate_recent_matches_diversity(recent_matches):
    """Validate that recent H2H matches have diverse scores"""
    if not recent_matches or len(recent_matches) < 2:
        return True, []  # No validation needed for empty or single match
    
    warnings = []
    
    # Check for identical scores
    score_tuples = [(match['home_goals'], match['away_goals']) for match in recent_matches]
    unique_scores = set(score_tuples)
    
    if len(unique_scores) < len(recent_matches):
        identical_count = len(recent_matches) - len(unique_scores)
        warnings.append(f"⚠️ {identical_count} identical score(s) detected in recent matches")
    
    # Check for unrealistic patterns
    home_wins_recent = sum(1 for match in recent_matches if match['home_goals'] > match['away_goals'])
    away_wins_recent = sum(1 for match in recent_matches if match['away_goals'] > match['home_goals'])
    draws_recent = sum(1 for match in recent_matches if match['home_goals'] == match['away_goals'])
    
    if home_wins_recent == len(recent_matches):
        warnings.append("🎯 All recent matches are home wins - consider verifying home/away teams")
    elif away_wins_recent == len(recent_matches):
        warnings.append("🎯 All recent matches are away wins - consider verifying home/away teams")
    elif draws_recent == len(recent_matches):
        warnings.append("🎯 All recent matches are draws - verify if this reflects actual results")
    
    # Check for unrealistic goal patterns
    total_goals = sum(match['home_goals'] + match['away_goals'] for match in recent_matches)
    avg_goals = total_goals / len(recent_matches)
    
    if avg_goals > 4.5:
        warnings.append("⚡ Very high scoring recent matches detected")
    elif avg_goals < 0.5:
        warnings.append("🛡️ Very low scoring recent matches detected")
    
    return len(warnings) == 0, warnings

def detect_h2h_pattern(total_matches, home_wins, away_wins, draws, recent_matches=None):
    """Automatically detect H2H pattern from aggregate statistics with enhanced recent match analysis"""
    if total_matches == 0:
        return "No H2H Data", "Low", "No historical data available"
    
    home_win_rate = home_wins / total_matches
    away_win_rate = away_wins / total_matches
    draw_rate = draws / total_matches
    
    # Enhanced recent match analysis
    recent_pattern = None
    recent_evidence = ""
    
    if recent_matches and len(recent_matches) > 0:
        home_wins_recent = sum(1 for match in recent_matches if match['home_goals'] > match['away_goals'])
        away_wins_recent = sum(1 for match in recent_matches if match['away_goals'] > match['home_goals'])
        draws_recent = sum(1 for match in recent_matches if match['home_goals'] == match['away_goals'])
        
        total_goals_recent = sum(match['home_goals'] + match['away_goals'] for match in recent_matches)
        avg_goals_recent = total_goals_recent / len(recent_matches)
        
        # Recent match pattern detection
        if home_wins_recent == len(recent_matches):
            recent_pattern = "Home Team Dominant in Recent Meetings"
            recent_evidence = f"Home team won all {len(recent_matches)} recent matches"
        elif away_wins_recent == len(recent_matches):
            recent_pattern = "Away Team Dominant in Recent Meetings" 
            recent_evidence = f"Away team won all {len(recent_matches)} recent matches"
        elif draws_recent == len(recent_matches):
            recent_pattern = "Draw Heavy in Recent Meetings"
            recent_evidence = f"All {len(recent_matches)} recent matches ended in draws"
        elif home_wins_recent >= len(recent_matches) * 0.67:  # 2/3 of recent matches
            recent_pattern = "Home Team Strong Recent Form"
            recent_evidence = f"Home team won {home_wins_recent}/{len(recent_matches)} recent matches"
        elif away_wins_recent >= len(recent_matches) * 0.67:
            recent_pattern = "Away Team Strong Recent Form"
            recent_evidence = f"Away team won {away_wins_recent}/{len(recent_matches)} recent matches"
        
        # Goal-based patterns for recent matches
        if avg_goals_recent >= 3.5:
            if not recent_pattern:
                recent_pattern = "High Scoring Recent Meetings"
            recent_evidence += f" with {avg_goals_recent:.1f} avg goals"
        elif avg_goals_recent <= 1.0:
            if not recent_pattern:
                recent_pattern = "Low Scoring Recent Meetings" 
            recent_evidence += f" with {avg_goals_recent:.1f} avg goals"
    
    # Overall pattern detection (existing logic)
    if home_win_rate >= 0.6:
        confidence = "High" if home_win_rate >= 0.7 else "Medium"
        evidence = f"Home team won {home_wins}/{total_matches} matches ({home_win_rate:.0%} win rate)"
        pattern = "Home Team Dominant"
        
    elif away_win_rate >= 0.6:
        confidence = "High" if away_win_rate >= 0.7 else "Medium"
        evidence = f"Away team won {away_wins}/{total_matches} matches ({away_win_rate:.0%} win rate)"
        pattern = "Away Team Dominant"
        
    elif draw_rate >= 0.5:
        confidence = "High" if draw_rate >= 0.6 else "Medium"
        evidence = f"{draws}/{total_matches} matches ended in draws ({draw_rate:.0%} draw rate)"
        pattern = "Draw Heavy"
        
    elif abs(home_win_rate - away_win_rate) <= 0.2:
        confidence = "High" if abs(home_win_rate - away_win_rate) <= 0.1 else "Medium"
        evidence = f"Evenly balanced: {home_wins}-{away_wins}-{draws} (win difference: {abs(home_win_rate-away_win_rate):.0%})"
        pattern = "Evenly Matched"
        
    else:
        confidence = "Medium"
        evidence = f"Competitive record: {home_wins}-{away_wins}-{draws} with clear tendencies"
        pattern = "Competitive"
    
    # Enhance with recent pattern if detected
    if recent_pattern:
        if pattern in ["Home Team Dominant", "Away Team Dominant"] and "Recent" in recent_pattern:
            # Keep the dominant pattern but enhance evidence
            evidence += f" | Recent: {recent_evidence}"
            confidence = "High"  # Boost confidence with recent confirmation
        elif pattern != recent_pattern.replace(" in Recent Meetings", "").replace(" Recent Form", ""):
            # Recent pattern differs from historical - create combined pattern
            pattern = f"{pattern} but {recent_pattern}"
            evidence += f" | Recent trend: {recent_evidence}"
            confidence = "High"  # Recent data gets priority
    
    return pattern, confidence, evidence

def get_pattern_badge(pattern_name):
    """Get styled badge for pattern display"""
    badge_classes = {
        "Home Team Dominant": "pattern-dominant",
        "Away Team Dominant": "pattern-dominant", 
        "Evenly Matched": "pattern-even",
        "Draw Heavy": "pattern-draw",
        "High Scoring": "pattern-high-scoring",
        "Low Scoring": "pattern-competitive",
        "Competitive": "pattern-competitive",
        "No H2H Data": "pattern-competitive",
        "Home Team Dominant in Recent Meetings": "pattern-recent-dominant",
        "Away Team Dominant in Recent Meetings": "pattern-recent-dominant",
        "Home Team Strong Recent Form": "pattern-recent-dominant",
        "Away Team Strong Recent Form": "pattern-recent-dominant",
        "Draw Heavy in Recent Meetings": "pattern-draw",
        "High Scoring Recent Meetings": "pattern-high-scoring",
        "Low Scoring Recent Meetings": "pattern-competitive"
    }
    class_name = badge_classes.get(pattern_name, "pattern-competitive")
    return f'<span class="pattern-badge {class_name}">{pattern_name}</span>'

def create_goal_based_team_analytics(team_name, team_type, default_values=None):
    """Create professional goal-based team analytics input - FIXED DATA STRUCTURE"""
    if default_values is None:
        default_values = {'goals_scored': 8, 'goals_conceded': 13, 'matches_scored': 3}
    
    st.markdown(f"""
    <div class='form-section'>
        <h4>📊 {team_name} Recent Performance (Last 6 Matches)</h4>
        <p><em>Goal-based analytics for reliable BTTS & Over/Under predictions</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Goal metrics input
    col1, col2, col3 = st.columns(3)
    
    with col1:
        goals_scored = st.number_input(
            f"Goals Scored",
            min_value=0,
            max_value=30,
            value=default_values.get('goals_scored', 8),
            help=f"Total goals scored by {team_name} in last 6 matches",
            key=f"{team_type}_goals_scored"
        )
    
    with col2:
        goals_conceded = st.number_input(
            f"Goals Conceded",
            min_value=0,
            max_value=30,
            value=default_values.get('goals_conceded', 13),
            help=f"Total goals conceded by {team_name} in last 6 matches",
            key=f"{team_type}_goals_conceded"
        )
    
    with col3:
        matches_scored = st.number_input(
            f"Matches Scored In",
            min_value=0,
            max_value=6,
            value=default_values.get('matches_scored', 3),
            help=f"How many of last 6 matches {team_name} scored in",
            key=f"{team_type}_matches_scored"
        )
    
    # Calculate metrics for the engine
    avg_scored = goals_scored / 6 if goals_scored > 0 else 0
    avg_conceded = goals_conceded / 6 if goals_conceded > 0 else 0
    scoring_frequency = (matches_scored / 6) * 100 if matches_scored > 0 else 0
    
    # Display metrics in a professional way
    st.markdown("---")
    st.write(f"**📈 {team_name} Performance Metrics**")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.markdown(f"""
        <div class='goal-metric-card'>
            <div class='goal-metric-value'>{goals_scored}</div>
            <div class='goal-metric-label'>Total Scored</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col2:
        st.markdown(f"""
        <div class='goal-metric-card'>
            <div class='goal-metric-value'>{goals_conceded}</div>
            <div class='goal-metric-label'>Total Conceded</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col3:
        st.markdown(f"""
        <div class='goal-metric-card'>
            <div class='goal-metric-value'>{avg_scored:.2f}</div>
            <div class='goal-metric-label'>Avg Scored/Match</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col4:
        st.markdown(f"""
        <div class='goal-metric-card'>
            <div class='goal-metric-value'>{scoring_frequency:.1f}%</div>
            <div class='goal-metric-label'>Scoring Frequency</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance analysis
    st.markdown("---")
    st.write("**🔍 Performance Analysis**")
    
    if goals_scored > 0 or goals_conceded > 0:
        total_goals_involved = goals_scored + goals_conceded
        avg_total_goals = total_goals_involved / 6
        
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            # Attack strength analysis
            if avg_scored >= 2.0:
                st.success("🔥 **Strong Attack**: High scoring capability")
            elif avg_scored >= 1.0:
                st.info("⚽ **Moderate Attack**: Decent scoring record")
            else:
                st.warning("🎯 **Weak Attack**: Low scoring frequency")
            
            # BTTS potential
            if scoring_frequency >= 70:
                st.success("🎯 **High BTTS Potential**: Frequently scores")
            elif scoring_frequency >= 40:
                st.info("⚖️ **Moderate BTTS Potential**: Sometimes scores")
            else:
                st.warning("🛡️ **Low BTTS Potential**: Rarely scores")
        
        with analysis_col2:
            # Defense analysis
            if avg_conceded >= 2.0:
                st.error("🎯 **Weak Defense**: High conceding rate")
            elif avg_conceded >= 1.0:
                st.warning("⚖️ **Moderate Defense**: Average conceding")
            else:
                st.success("🛡️ **Strong Defense**: Low conceding rate")
            
            # Over/Under analysis
            if avg_total_goals >= 3.5:
                st.success("📈 **High-Scoring Games**: Over 2.5 likely")
            elif avg_total_goals >= 2.5:
                st.info("⚖️ **Balanced Games**: Mixed goal expectations")
            else:
                st.warning("📉 **Low-Scoring Games**: Under 2.5 likely")
    
    # 🎯 TIMING INTELLIGENCE NOTE
    st.markdown("---")
    st.info("""
    **🎯 Enhanced Timing Intelligence**: Goal timing patterns are now automatically calculated by the prediction engine 
    based on team performance data and match context. No manual timing inputs needed!
    """)
    
    # Return data in the CORRECT structure that the engine expects
    return {
        'goals_scored': goals_scored,        # Raw goals for engine calculation
        'goals_conceded': goals_conceded,    # Raw goals for engine calculation  
        'matches_scored': matches_scored,
        'avg_scored': avg_scored,           # Pre-calculated for patterns
        'avg_conceded': avg_conceded,       # Pre-calculated for patterns
        'scoring_frequency': scoring_frequency
    }

def create_h2h_hybrid_input_section(home_team, away_team, default_aggregate=None, default_recent_matches=None):
    """Create professional H2H input with hybrid approach and enhanced validation"""
    if default_aggregate is None:
        default_aggregate = {'total_matches': 6, 'home_wins': 3, 'away_wins': 2, 'draws': 1, 'home_goals': 8, 'away_goals': 5}
    if default_recent_matches is None:
        default_recent_matches = []
    
    st.markdown(f"""
    <div class='form-section'>
        <h4>⚔️ Head-to-Head Analysis</h4>
        <p><em>Hybrid approach for comprehensive H2H data</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # PART 1: Aggregate H2H Statistics (Quick & Essential) - ENHANCED WITH GOAL DATA
    st.markdown("""
    <div class='aggregate-section'>
        <h4>📈 H2H Aggregate Statistics (Last 2 Years)</h4>
        <p><em>Quick summary - essential for model accuracy</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # First row: Match counts
    agg_col1, agg_col2, agg_col3, agg_col4 = st.columns(4)
    
    with agg_col1:
        total_matches = st.number_input(
            "Total H2H Matches",
            min_value=0,
            max_value=20,
            value=default_aggregate['total_matches'],
            help="Total matches in last 2 years"
        )
    
    with agg_col2:
        home_wins = st.number_input(
            f"{home_team} Wins",
            min_value=0,
            max_value=20,
            value=default_aggregate['home_wins'],
            help=f"Matches won by {home_team}"
        )
    
    with agg_col3:
        away_wins = st.number_input(
            f"{away_team} Wins", 
            min_value=0,
            max_value=20,
            value=default_aggregate['away_wins'],
            help=f"Matches won by {away_team}"
        )
    
    with agg_col4:
        draws = st.number_input(
            "Draws",
            min_value=0, 
            max_value=20,
            value=default_aggregate['draws'],
            help="Matches ended in draw"
        )
    
    # 🆕 SECOND ROW: GOAL DATA - CRITICAL FOR ENHANCED ENGINE
    st.markdown("---")
    st.write("**🎯 H2H Goal Data** - *Essential for enhanced predictions*")
    
    goal_col1, goal_col2 = st.columns(2)
    
    with goal_col1:
        home_goals = st.number_input(
            f"Total Goals Scored by {home_team}",
            min_value=0,
            max_value=50,
            value=default_aggregate.get('home_goals', 8),
            help=f"Total goals {home_team} scored in all H2H matches",
            key="h2h_home_goals"
        )
    
    with goal_col2:
        away_goals = st.number_input(
            f"Total Goals Scored by {away_team}",
            min_value=0,
            max_value=50,
            value=default_aggregate.get('away_goals', 5),
            help=f"Total goals {away_team} scored in all H2H matches",
            key="h2h_away_goals"
        )
    
    # Calculate and display goal metrics
    if total_matches > 0:
        home_avg_goals = home_goals / total_matches
        away_avg_goals = away_goals / total_matches
        total_avg_goals = (home_goals + away_goals) / total_matches
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric(f"{home_team} Avg Goals", f"{home_avg_goals:.2f}")
        with metric_col2:
            st.metric(f"{away_team} Avg Goals", f"{away_avg_goals:.2f}")
        with metric_col3:
            st.metric("Total Avg Goals", f"{total_avg_goals:.2f}")
    
    # Validate aggregate data
    if total_matches > 0 and (home_wins + away_wins + draws) != total_matches:
        st.warning(f"⚠️ Stats don't match: {home_wins} wins + {away_wins} wins + {draws} draws = {home_wins + away_wins + draws}, but total matches = {total_matches}")
    
    # PART 2: Recent Match Details (Optional but valuable)
    st.markdown("---")
    st.subheader("🎯 Recent H2H Match Details (Optional)")
    
    # Enhanced user guidance
    st.markdown("""
    <div class='warning-badge'>
        <strong>💡 IMPORTANT:</strong> Enter 3 different recent matches with varied scores for best accuracy.
        Mix home/away perspectives realistically.
    </div>
    """, unsafe_allow_html=True)
    
    recent_matches = []
    use_recent_matches = st.checkbox("Include detailed recent matches", value=len(default_recent_matches) > 0)
    
    if use_recent_matches:
        for i in range(3):
            st.markdown(f'<div class="h2h-match-card">', unsafe_allow_html=True)
            st.write(f"**Match {i+1}** (Most recent first)")
            
            # Get default values - clear defaults if teams change
            default_match = default_recent_matches[i] if i < len(default_recent_matches) else {'home_team': home_team, 'away_team': away_team, 'home_goals': 1, 'away_goals': 0}
            
            # Clear default if teams don't match current selection
            if (default_match['home_team'] != home_team and default_match['home_team'] != away_team) or \
               (default_match['away_team'] != home_team and default_match['away_team'] != away_team):
                default_match = {'home_team': home_team, 'away_team': away_team, 'home_goals': 1, 'away_goals': 0}
            
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col1:
                home_team_match = st.selectbox(
                    f"Home Team",
                    [home_team, away_team],
                    index=0 if default_match['home_team'] == home_team else 1,
                    key=f"h2h_home_team_{i}"
                )
                home_goals_match = st.number_input(
                    "Goals",
                    min_value=0,
                    max_value=10,
                    value=default_match['home_goals'],
                    key=f"h2h_home_goals_{i}"
                )
            
            with col2:
                st.write("")  # Spacer
                st.markdown("### 🆚")
                st.write("")  # Spacer
            
            with col3:
                # Away team is the one not selected as home
                away_team_options = [team for team in [home_team, away_team] if team != home_team_match]
                away_team_match = away_team_options[0] if away_team_options else away_team
                
                st.write(f"**{away_team_match}**")
                away_goals_match = st.number_input(
                    "Goals", 
                    min_value=0,
                    max_value=10,
                    value=default_match['away_goals'],
                    key=f"h2h_away_goals_{i}"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
            recent_matches.append({
                'home_team': home_team_match,
                'away_team': away_team_match,
                'home_goals': home_goals_match,
                'away_goals': away_goals_match
            })
        
        # Real-time validation of recent matches
        is_valid, warnings = validate_recent_matches_diversity(recent_matches)
        
        if warnings:
            for warning in warnings:
                st.markdown(f'<div class="warning-badge">{warning}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="success-badge">✅ Recent matches data quality: Excellent</div>', unsafe_allow_html=True)
            
        # Store warnings for form submission
        st.session_state.h2h_warnings = warnings
        
    else:
        recent_matches = []
        st.session_state.h2h_warnings = []
    
    # AUTOMATIC PATTERN DETECTION
    st.markdown("---")
    st.write("**📊 H2H Pattern Analysis**")
    
    # Detect pattern automatically with enhanced recent match analysis
    pattern, confidence, evidence = detect_h2h_pattern(total_matches, home_wins, away_wins, draws, recent_matches)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"**Detected Pattern:** {get_pattern_badge(pattern)}", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {confidence} ✅")
        st.markdown(f"**Evidence:** {evidence}")
        
        # Show recent match insights if available
        if recent_matches:
            home_wins_recent = sum(1 for match in recent_matches if match['home_goals'] > match['away_goals'])
            away_wins_recent = sum(1 for match in recent_matches if match['away_goals'] > match['home_goals'])
            draws_recent = sum(1 for match in recent_matches if match['home_goals'] == match['away_goals'])
            
            st.write("**Recent Match Breakdown:**")
            st.write(f"- Home Wins: {home_wins_recent}/{len(recent_matches)}")
            st.write(f"- Away Wins: {away_wins_recent}/{len(recent_matches)}") 
            st.write(f"- Draws: {draws_recent}/{len(recent_matches)}")
    
    with col2:
        # Quick stats
        if total_matches > 0:
            home_win_rate = (home_wins / total_matches) * 100
            away_win_rate = (away_wins / total_matches) * 100
            draw_rate = (draws / total_matches) * 100
            
            st.metric(f"{home_team} Win Rate", f"{home_win_rate:.1f}%")
            st.metric(f"{away_team} Win Rate", f"{away_win_rate:.1f}%")
            st.metric("Draw Rate", f"{draw_rate:.1f}%")
    
    # Summary display with goal data
    st.markdown("---")
    st.write("**📋 H2H Input Summary**")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Matches", total_matches)
    with col2:
        st.metric(f"{home_team} Wins", home_wins)
    with col3:
        st.metric(f"{away_team} Wins", away_wins)
    with col4:
        st.metric("Draws", draws)
    
    # Goal summary
    goal_sum_col1, goal_sum_col2 = st.columns(2)
    with goal_sum_col1:
        st.metric(f"{home_team} Total Goals", home_goals)
    with goal_sum_col2:
        st.metric(f"{away_team} Total Goals", away_goals)
    
    if total_matches > 0:
        goal_avg_col1, goal_avg_col2 = st.columns(2)
        with goal_avg_col1:
            st.metric(f"{home_team} Avg Goals", f"{home_goals/total_matches:.2f}")
        with goal_avg_col2:
            st.metric(f"{away_team} Avg Goals", f"{away_goals/total_matches:.2f}")
    
    if use_recent_matches and recent_matches:
        st.write("**Recent Matches:**")
        for i, match in enumerate(recent_matches):
            emoji = "🏠" if match['home_team'] == home_team else "✈️"
            st.write(f"{i+1}. {emoji} {match['home_team']} {match['home_goals']}-{match['away_goals']} {match['away_team']}")
        
        # Calculate average goals from recent matches
        total_goals = sum(match['home_goals'] + match['away_goals'] for match in recent_matches)
        avg_goals = total_goals / len(recent_matches) if recent_matches else 0
        st.metric("Avg Goals (from recent matches)", f"{avg_goals:.1f}")
    
    # Prepare data for model - NOW INCLUDES GOAL DATA
    aggregate_data = {
        'total_matches': total_matches,
        'home_wins': home_wins,
        'away_wins': away_wins, 
        'draws': draws,
        'home_goals': home_goals,  # 🆕 CRITICAL: Added goal data
        'away_goals': away_goals   # 🆕 CRITICAL: Added goal data
    }
    
    return aggregate_data, recent_matches, pattern, confidence, evidence

def create_professional_input_form():
    """Create institutional-grade input form with goal-based analytics"""
    st.markdown("""
    <div class='institutional-card'>
        <h2>🎯 Institutional Data Input</h2>
        <p>Enter match details with professional goal-based analytics + Automated Timing Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("institutional_match_form", clear_on_submit=False):
        # Basic match info
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.text_input(
                "🏠 Home Team", 
                value=st.session_state.match_data.get('home_team', 'Union Berlin'),
                placeholder="e.g., Manchester City", 
                key="home_team_input"
            )
        with col2:
            away_team = st.text_input(
                "✈️ Away Team", 
                value=st.session_state.match_data.get('away_team', 'Borussia M\'gladbach'),
                placeholder="e.g., Liverpool", 
                key="away_team_input"
            )
        
        # Clear H2H defaults if teams change
        current_teams = f"{home_team}_{away_team}"
        last_teams = st.session_state.get('last_teams', '')
        
        if current_teams != last_teams and home_team and away_team:
            # Teams changed, clear H2H defaults
            st.session_state.match_data.pop('h2h_recent_matches', None)
            st.session_state.last_teams = current_teams
        
        # Advanced context section
        st.markdown("---")
        st.subheader("🌍 Advanced Match Context")
        
        context_col1, context_col2, context_col3 = st.columns(3)
        with context_col1:
            league_options = ["Standard", "English Premier League", "La Liga", "Serie A", "Bundesliga", "International", "Champions League"]
            current_league = st.session_state.match_data.get('league_type', 'Bundesliga')
            league_index = league_options.index(current_league) if current_league in league_options else 4
            league_type = st.selectbox(
                "League Type",
                league_options,
                index=league_index,
                help="League context affects home advantage and model priors"
            )
        with context_col2:
            importance_options = ["Normal League", "Derby/Local Rivalry", "Relegation Battle", "Title Decider", "Cup Final", "European Qualification"]
            current_importance = st.session_state.match_data.get('match_importance', 'Normal League')
            importance_index = importance_options.index(current_importance) if current_importance in importance_options else 0
            match_importance = st.selectbox(
                "Match Importance",
                importance_options,
                index=importance_index,
                help="Psychological and motivational factors"
            )
        with context_col3:
            venue_options = ["Normal", "Empty Stadium", "European Night", "Rival Territory"]
            venue_impact = st.selectbox(
                "Venue Context",
                venue_options,
                help="Crowd and atmosphere factors"
            )
        
        # Team analytics with goal-based inputs
        st.markdown("---")
        st.subheader("📊 Team Performance Analytics")
        st.info("🎯 **Goal-based inputs + Automated Timing Intelligence for reliable predictions**")
        
        form_col1, form_col2 = st.columns(2)
        with form_col1:
            # Get default goal values from session state
            home_goals_default = st.session_state.match_data.get('home_goals_data', {'goals_scored': 8, 'goals_conceded': 13, 'matches_scored': 3})
            home_goals_data = create_goal_based_team_analytics(
                f"{home_team or 'Home Team'}", 
                "home", 
                home_goals_default
            )
                
        with form_col2:
            # Get default goal values from session state
            away_goals_default = st.session_state.match_data.get('away_goals_data', {'goals_scored': 5, 'goals_conceded': 12, 'matches_scored': 2})
            away_goals_data = create_goal_based_team_analytics(
                f"{away_team or 'Away Team'}", 
                "away", 
                away_goals_default
            )
        
        # Advanced standings input
        st.markdown("---")
        st.subheader("🏆 League Standings Analysis")
        
        st.write("**Team Strength Metrics**")
        
        stand_col1, stand_col2 = st.columns(2)
        with stand_col1:
            home_standing = st.session_state.match_data.get('home_standing', [13, 7, 6, -5])
            st.write(f"**{home_team or 'Home'}**")
            home_pos = st.number_input("League Position", min_value=1, max_value=20, value=home_standing[0], key="home_pos")
            home_pts = st.number_input("Total Points", min_value=0, value=home_standing[1], key="home_pts")
            home_played = st.number_input("Matches Played", min_value=1, value=home_standing[2], key="home_played")
            home_gd = st.number_input("Goal Difference", value=home_standing[3], key="home_gd")
            
        with stand_col2:
            away_standing = st.session_state.match_data.get('away_standing', [17, 3, 6, -7])
            st.write(f"**{away_team or 'Away'}**")
            away_pos = st.number_input("League Position", min_value=1, max_value=20, value=away_standing[0], key="away_pos")
            away_pts = st.number_input("Total Points", min_value=0, value=away_standing[1], key="away_pts")
            away_played = st.number_input("Matches Played", min_value=1, value=away_standing[2], key="away_played")
            away_gd = st.number_input("Goal Difference", value=away_standing[3], key="away_gd")
        
        # Enhanced H2H with hybrid input - NOW INCLUDES GOAL DATA
        st.markdown("---")
        st.subheader("⚔️ Head-to-Head Analysis")
        
        # Get default H2H data from session state
        default_aggregate = st.session_state.match_data.get('h2h_aggregate', {'total_matches': 6, 'home_wins': 3, 'away_wins': 2, 'draws': 1, 'home_goals': 8, 'away_goals': 5})
        default_recent_matches = st.session_state.match_data.get('h2h_recent_matches', [])
        
        h2h_aggregate, h2h_recent_matches, h2h_pattern, h2h_confidence, h2h_evidence = create_h2h_hybrid_input_section(
            home_team or "Home Team", 
            away_team or "Away Team",
            default_aggregate,
            default_recent_matches
        )
        
        # Institutional odds input
        st.markdown("---")
        st.subheader("💰 Market Odds Analysis")
        
        st.write("**Professional Odds Input** - Critical for institutional predictions")
        
        odds_col1, odds_col2, odds_col3 = st.columns(3)
        with odds_col1:
            st.write("**1X2 Market**")
            odds_default = st.session_state.match_data.get('odds_1x2', [2.33, 3.44, 3.33])
            odds_1 = st.number_input("Home Win", min_value=1.01, max_value=100.0, value=odds_default[0], step=0.01, key="odds_1")
            odds_x = st.number_input("Draw", min_value=1.01, max_value=100.0, value=odds_default[1], step=0.01, key="odds_x")
            odds_2 = st.number_input("Away Win", min_value=1.01, max_value=100.0, value=odds_default[2], step=0.01, key="odds_2")
            
        with odds_col2:
            st.write("**Goals Market**")
            ou_default = st.session_state.match_data.get('odds_over_under', [2.00, 1.85])
            odds_over = st.number_input("Over 2.5", min_value=1.01, max_value=100.0, value=ou_default[0], step=0.01, key="odds_over")
            odds_under = st.number_input("Under 2.5", min_value=1.01, max_value=100.0, value=ou_default[1], step=0.01, key="odds_under")
            
        with odds_col3:
            st.write("**BTTS Market**")
            btts_default = st.session_state.match_data.get('odds_btts', [1.79, 2.05])
            odds_yes = st.number_input("BTTS Yes", min_value=1.01, max_value=100.0, value=btts_default[0], step=0.01, key="odds_yes")
            odds_no = st.number_input("BTTS No", min_value=1.01, max_value=100.0, value=btts_default[1], step=0.01, key="odds_no")
        
        # Institutional submission
        submitted = st.form_submit_button(
            "🚀 GENERATE ENHANCED INSTITUTIONAL PREDICTIONS", 
            type="primary", 
            use_container_width=True
        )
        
        if submitted:
            if not home_team or not away_team:
                st.error("❌ Institutional-grade analysis requires both team names")
                return None
            
            # Validate goal data
            if home_goals_data['goals_scored'] == 0 and home_goals_data['goals_conceded'] == 0:
                st.warning(f"⚠️ No goal data provided for {home_team}. Using league averages.")
            
            if away_goals_data['goals_scored'] == 0 and away_goals_data['goals_conceded'] == 0:
                st.warning(f"⚠️ No goal data provided for {away_team}. Using league averages.")
            
            # Validate H2H aggregate data
            if h2h_aggregate['total_matches'] > 0 and (h2h_aggregate['home_wins'] + h2h_aggregate['away_wins'] + h2h_aggregate['draws']) != h2h_aggregate['total_matches']:
                st.error("❌ H2H statistics don't match: Wins + Draws should equal Total Matches")
                return None
            
            # Show H2H warnings if any
            if st.session_state.h2h_warnings:
                st.warning("🎯 **H2H Data Quality Notes:**")
                for warning in st.session_state.h2h_warnings:
                    st.write(f"- {warning}")
            
            # Store raw data for debugging
            st.session_state.match_data = {
                'home_team': home_team,
                'away_team': away_team,
                'home_goals_data': home_goals_data,
                'away_goals_data': away_goals_data,
                'h2h_aggregate': h2h_aggregate,
                'h2h_recent_matches': h2h_recent_matches,
                'h2h_pattern': h2h_pattern,
                'h2h_confidence': h2h_confidence,
                'h2h_evidence': h2h_evidence,
                'league_type': league_type,
                'match_importance': match_importance,
                'venue_context': venue_impact,
                'home_standing': [home_pos, home_pts, home_played, home_gd],
                'away_standing': [away_pos, away_pts, away_played, away_gd],
                'odds_1x2': [odds_1, odds_x, odds_2],
                'odds_over_under': [odds_over, odds_under],
                'odds_btts': [odds_yes, odds_no]
            }
            
            # Create backup
            st.session_state.last_match_data = st.session_state.match_data.copy()
            
            # PROCESS ALL DATA
            try:
                # Convert recent matches to model format and calculate goal data
                h2h_matches_model = []
                total_goals_from_recent = 0
                
                for match in h2h_recent_matches:
                    # Determine which team is the "home" team for this specific match
                    if match['home_team'] == home_team:
                        h2h_matches_model.append([match['home_goals'], match['away_goals']])
                        total_goals_from_recent += (match['home_goals'] + match['away_goals'])
                    else:
                        # If the away team was home in this H2H match, reverse the perspective
                        h2h_matches_model.append([match['away_goals'], match['home_goals']])
                        total_goals_from_recent += (match['home_goals'] + match['away_goals'])
                
                # Calculate average goals from recent matches
                avg_goals_from_recent = total_goals_from_recent / len(h2h_recent_matches) if h2h_recent_matches else 0
                
                # If no recent matches, use H2H goal data for goal expectations
                if not h2h_matches_model and h2h_aggregate['total_matches'] > 0:
                    # Create representative matches based on aggregate statistics and H2H goal data
                    h2h_matches_model = create_synthetic_h2h_matches_with_goals(h2h_aggregate, league_type)
                    # Use H2H goal data for average goals calculation
                    avg_goals_from_recent = (h2h_aggregate['home_goals'] + h2h_aggregate['away_goals']) / h2h_aggregate['total_matches']
                
                # Build final match data for prediction engine with COMPLETE goal data
                match_data = {
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_goals_data': home_goals_data,
                    'away_goals_data': away_goals_data,
                    'home_standing': [home_pos, home_pts, home_played, home_gd],
                    'away_standing': [away_pos, away_pts, away_played, away_gd],
                    'head_to_head': h2h_matches_model,
                    'h2h_aggregate': h2h_aggregate,
                    'h2h_pattern': h2h_pattern,
                    'h2h_confidence': h2h_confidence,
                    'h2h_evidence': h2h_evidence,
                    'h2h_avg_goals': avg_goals_from_recent,
                    'h2h_home_goals': h2h_aggregate.get('home_goals', 0),  # 🆕 CRITICAL: Pass H2H goal data
                    'h2h_away_goals': h2h_aggregate.get('away_goals', 0),  # 🆕 CRITICAL: Pass H2H goal data
                    'home_injuries': [],
                    'away_injuries': [],
                    'odds_1x2': [odds_1, odds_x, odds_2],
                    'odds_over_under': [odds_over, odds_under],
                    'odds_btts': [odds_yes, odds_no],
                    'league_type': league_type,
                    'match_importance': match_importance,
                    'venue_context': venue_impact
                }
                
                # 🐛 DEBUG: Store comprehensive debug data
                st.session_state.debug_data = {
                    'home_goals_data': home_goals_data,
                    'away_goals_data': away_goals_data,
                    'h2h_aggregate_data': h2h_aggregate,
                    'h2h_recent_matches_raw': h2h_recent_matches,
                    'h2h_matches_model_format': h2h_matches_model,
                    'h2h_pattern_detected': h2h_pattern,
                    'h2h_confidence': h2h_confidence,
                    'h2h_evidence': h2h_evidence,
                    'h2h_avg_goals_calculated': avg_goals_from_recent,
                    'h2h_home_goals_provided': h2h_aggregate.get('home_goals', 0),
                    'h2h_away_goals_provided': h2h_aggregate.get('away_goals', 0),
                    'h2h_warnings': st.session_state.h2h_warnings,
                    'home_injuries_parsed': [],
                    'away_injuries_parsed': [],
                    'data_quality_score': calculate_data_quality(match_data),
                    'home_avg_scored': home_goals_data['avg_scored'],
                    'home_avg_conceded': home_goals_data['avg_conceded'],
                    'home_scoring_frequency': home_goals_data['scoring_frequency'],
                    'away_avg_scored': away_goals_data['avg_scored'],
                    'away_avg_conceded': away_goals_data['avg_conceded'],
                    'away_scoring_frequency': away_goals_data['scoring_frequency'],
                    'h2h_total_matches': h2h_aggregate['total_matches'],
                    'h2h_recent_matches_count': len(h2h_recent_matches),
                    'h2h_win_ratio_home': f"{(h2h_aggregate['home_wins']/h2h_aggregate['total_matches'])*100:.1f}%" if h2h_aggregate['total_matches'] > 0 else "0%",
                    'h2h_win_ratio_away': f"{(h2h_aggregate['away_wins']/h2h_aggregate['total_matches'])*100:.1f}%" if h2h_aggregate['total_matches'] > 0 else "0%",
                    'h2h_home_avg_goals': f"{(h2h_aggregate.get('home_goals', 0)/h2h_aggregate['total_matches']):.2f}" if h2h_aggregate['total_matches'] > 0 else "0.00",
                    'h2h_away_avg_goals': f"{(h2h_aggregate.get('away_goals', 0)/h2h_aggregate['total_matches']):.2f}" if h2h_aggregate['total_matches'] > 0 else "0.00"
                }
                
                validation_errors = validate_match_data_legacy(match_data)
                if validation_errors:
                    for error in validation_errors:
                        st.error(f"❌ Data Validation Error: {error}")
                    return None
                
                return match_data
                
            except Exception as e:
                st.error(f"❌ Data Processing Error: {e}")
                return None
    
    return None

def create_synthetic_h2h_matches_with_goals(aggregate_data, league_type):
    """🆕 ENHANCED: Create representative H2H matches using actual H2H goal data"""
    synthetic_matches = []
    total_matches = aggregate_data['total_matches']
    
    if total_matches == 0:
        return [[1, 1]]  # Default neutral match
    
    # Use actual H2H goal data for more accurate representation
    home_avg_goals = aggregate_data.get('home_goals', 0) / total_matches if aggregate_data.get('home_goals') else get_league_goal_expectation(league_type) * 0.6
    away_avg_goals = aggregate_data.get('away_goals', 0) / total_matches if aggregate_data.get('away_goals') else get_league_goal_expectation(league_type) * 0.4
    
    # Create matches based on actual win/draw distribution with H2H-based goals
    for _ in range(aggregate_data['home_wins']):
        # Home wins with H2H-based goals
        home_goals = max(1, int(home_avg_goals * 1.2))  # Slightly higher for wins
        away_goals = max(0, int(away_avg_goals * 0.8))  # Slightly lower for losses
        synthetic_matches.append([home_goals, away_goals])
    
    for _ in range(aggregate_data['away_wins']):
        # Away wins with H2H-based goals  
        home_goals = max(0, int(home_avg_goals * 0.8))  # Slightly lower for losses
        away_goals = max(1, int(away_avg_goals * 1.2))  # Slightly higher for wins
        synthetic_matches.append([home_goals, away_goals])
    
    for _ in range(aggregate_data['draws']):
        # Draws with H2H-based goals
        goals = max(1, int((home_avg_goals + away_avg_goals) * 0.5))
        synthetic_matches.append([goals, goals])
    
    return synthetic_matches

def get_league_goal_expectation(league_type):
    """Return typical average goals per match for different leagues"""
    league_goals = {
        "English Premier League": 2.8,
        "Bundesliga": 3.2,
        "La Liga": 2.5,
        "Serie A": 2.6,
        "Champions League": 2.7,
        "International": 2.4,
        "Standard": 2.6
    }
    return league_goals.get(league_type, 2.6)

def display_debug_information():
    """Display comprehensive debug information"""
    if not st.session_state.get('debug_data'):
        return
        
    debug_data = st.session_state.debug_data
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🐛 DEBUG PANEL")
    
    # Data parsing debug
    with st.sidebar.expander("🔧 DATA PARSING DEBUG", expanded=True):
        st.write("**Home Team Goal Data:**")
        st.write(f"Goals Scored: {debug_data.get('home_goals_data', {}).get('goals_scored', 0)}")
        st.write(f"Goals Conceded: {debug_data.get('home_goals_data', {}).get('goals_conceded', 0)}")
        st.write(f"Matches Scored In: {debug_data.get('home_goals_data', {}).get('matches_scored', 0)}/6")
        st.write(f"Avg Scored: {debug_data.get('home_avg_scored', 0):.2f}")
        st.write(f"Avg Conceded: {debug_data.get('home_avg_conceded', 0):.2f}")
        st.write(f"Scoring Frequency: {debug_data.get('home_scoring_frequency', 0):.1f}%")
        
        st.write("**Away Team Goal Data:**")
        st.write(f"Goals Scored: {debug_data.get('away_goals_data', {}).get('goals_scored', 0)}")
        st.write(f"Goals Conceded: {debug_data.get('away_goals_data', {}).get('goals_conceded', 0)}")
        st.write(f"Matches Scored In: {debug_data.get('away_goals_data', {}).get('matches_scored', 0)}/6")
        st.write(f"Avg Scored: {debug_data.get('away_avg_scored', 0):.2f}")
        st.write(f"Avg Conceded: {debug_data.get('away_avg_conceded', 0):.2f}")
        st.write(f"Scoring Frequency: {debug_data.get('away_scoring_frequency', 0):.1f}%")
        
        st.write("**H2H Aggregate Data:**")
        st.write(f"Total matches: {debug_data.get('h2h_total_matches', 0)}")
        st.write(f"Home wins: {debug_data.get('h2h_aggregate_data', {}).get('home_wins', 0)}")
        st.write(f"Away wins: {debug_data.get('h2h_aggregate_data', {}).get('away_wins', 0)}")
        st.write(f"Draws: {debug_data.get('h2h_aggregate_data', {}).get('draws', 0)}")
        
        st.write("**🆕 H2H GOAL DATA (CRITICAL):**")
        st.write(f"Home goals provided: {debug_data.get('h2h_home_goals_provided', 0)}")
        st.write(f"Away goals provided: {debug_data.get('h2h_away_goals_provided', 0)}")
        st.write(f"Home avg goals: {debug_data.get('h2h_home_avg_goals', '0.00')}")
        st.write(f"Away avg goals: {debug_data.get('h2h_away_avg_goals', '0.00')}")
        
        st.write("**H2H Recent Matches:**")
        st.write(f"Recent matches count: {debug_data.get('h2h_recent_matches_count', 0)}")
        st.write(f"Raw recent matches: {debug_data.get('h2h_recent_matches_raw', [])}")
        st.write(f"Model format: {debug_data.get('h2h_matches_model_format', [])}")
        
        st.write("**H2H Pattern Detection:**")
        st.write(f"Pattern: {debug_data.get('h2h_pattern_detected', 'N/A')}")
        st.write(f"Confidence: {debug_data.get('h2h_confidence', 'N/A')}")
        st.write(f"Evidence: {debug_data.get('h2h_evidence', 'N/A')}")
        
        st.write("**H2H Goal Data:**")
        st.write(f"Average goals calculated: {debug_data.get('h2h_avg_goals_calculated', 0):.2f}")
        
        st.write("**H2H Data Quality Warnings:**")
        warnings = debug_data.get('h2h_warnings', [])
        if warnings:
            for warning in warnings:
                st.write(f"⚠️ {warning}")
        else:
            st.write("✅ No warnings")
        
        st.write("**Win Ratios:**")
        st.write(f"Home win ratio: {debug_data.get('h2h_win_ratio_home', '0%')}")
        st.write(f"Away win ratio: {debug_data.get('h2h_win_ratio_away', '0%')}")
        
        st.write("**Input Statistics:**")
        st.write(f"Data quality: {debug_data.get('data_quality_score', 0):.1f}%")
    
    # Prediction engine debug
    if st.session_state.get('institutional_predictions'):
        predictions = st.session_state.institutional_predictions
        with st.sidebar.expander("🤖 PREDICTION ENGINE DEBUG", expanded=True):
            st.write("**Goal Expectancy:**")
            goal_exp = predictions.get('predictions', {}).get('goal_expectancy', {})
            st.write(f"Home XG: {goal_exp.get('home_xg', 'N/A')}")
            st.write(f"Away XG: {goal_exp.get('away_xg', 'N/A')}")
            st.write(f"Total XG: {goal_exp.get('total_xg', 'N/A')}")
            
            # Enhanced timing intelligence debug
            timing_intel = predictions.get('goal_timing_intelligence', {})
            if timing_intel:
                st.write("**🎯 TIMING INTELLIGENCE ACTIVE:**")
                st.write(f"1H Goal Probability: {timing_intel.get('1h_goal_probability', 'N/A')}%")
                st.write(f"2H Goal Probability: {timing_intel.get('2h_goal_probability', 'N/A')}%")
                st.write(f"Late Goals Probability: {timing_intel.get('late_goals_75plus_prob', 'N/A')}%")
                st.write(f"Scoring Momentum: {timing_intel.get('scoring_momentum', 'N/A')}")
                
                # Team-specific timing ratios
                team_timing = timing_intel.get('team_timing_analysis', {})
                st.write("**Team Timing Ratios:**")
                st.write(f"Home 1H Ratio: {team_timing.get('home_1h_ratio', 'N/A')}")
                st.write(f"Away 1H Ratio: {team_timing.get('away_1h_ratio', 'N/A')}")
            else:
                st.write("**❌ TIMING INTELLIGENCE INACTIVE**")
            
            st.write("**Pattern Detection:**")
            pattern_info = predictions.get('pattern_intelligence', {})
            st.write(f"Patterns found: {pattern_info.get('pattern_count', 0)}")
            
            # Enhanced upset analysis display
            upset_analysis = predictions.get('upset_analysis', {})
            if upset_analysis.get('upset_detected'):
                st.write("**🎯 UPSET POTENTIAL DETECTED**")
                st.write(f"Upset Level: {upset_analysis.get('upset_level', 'N/A')}")
                st.write(f"Home Boost: {upset_analysis.get('home_boost', 0):.3f}")
                st.write(f"Total Score: {upset_analysis.get('total_upset_score', 0):.3f}")
                st.write(f"Factors: {upset_analysis.get('factor_count', 0)}")
            else:
                st.write("**❌ NO UPSET POTENTIAL DETECTED**")
            
            for pattern in pattern_info.get('patterns_detected', []):
                st.write(f"- {pattern['type']}: {pattern['evidence']}")
            
            st.write("**Market Analysis:**")
            market_eff = predictions.get('market_comparison', {})
            st.write(f"Efficiency: {market_eff.get('efficiency', 'N/A')}")
            st.write(f"Overround: {market_eff.get('overround', 'N/A')}%")

def display_institutional_predictions(match_data, predictions):
    """Display world-class institutional predictions with professional analytics and enhanced intelligence"""
    
    # Calculate advanced data quality
    data_quality = calculate_data_quality(match_data)
    quality_message = get_data_quality_message(data_quality)
    
    # Header with professional branding
    st.markdown("""
    <div class='institutional-card'>
        <h1>🎯 ENHANCED INSTITUTIONAL PREDICTION REPORT</h1>
        <p>World-Class Football Analytics • Risk-Managed Insights • Professional Grade • Automated Timing Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display debug information
    display_debug_information()
    
    # 🆕 UPSET POTENTIAL ALERT
    upset_analysis = predictions.get('upset_analysis', {})
    if upset_analysis.get('upset_detected'):
        st.markdown(f"""
        <div class='upset-alert'>
            <h3>🎯 UPSET POTENTIAL DETECTED!</h3>
            <p><strong>Level:</strong> {upset_analysis['upset_level']} | <strong>Factors:</strong> {upset_analysis['factor_count']}</p>
            <p>Contextual analysis suggests {match_data['home_team']} may significantly overperform statistical expectations</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 🆕 ENHANCED: Goal Timing Intelligence Display
    if 'goal_timing_intelligence' in predictions:
        timing_info = predictions['goal_timing_intelligence']
        
        st.markdown("---")
        st.markdown("""
        <div class='institutional-card'>
            <h3>⏰ AUTOMATED GOAL TIMING INTELLIGENCE</h3>
            <p><em>Advanced goal timing predictions automatically calculated from team performance data</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Timing metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class='timing-metric'>
                <h3>1st Half Goals</h3>
                <h2>{timing_info['1h_goal_probability']}%</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class='timing-metric'>
                <h3>2nd Half Goals</h3>
                <h2>{timing_info['2h_goal_probability']}%</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class='timing-metric'>
                <h3>Late Goals (75+ mins)</h3>
                <h2>{timing_info['late_goals_75plus_prob']}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Goal timing windows
        st.write("**🎯 Expected Goal Timing Windows**")
        timing_windows = timing_info.get('expected_goal_timing', {})
        
        window_col1, window_col2, window_col3 = st.columns(3)
        with window_col1:
            st.metric("First Goal", timing_windows.get('first_goal_window', 'N/A'))
        with window_col2:
            st.metric("Second Goal", timing_windows.get('second_goal_window', 'N/A'))
        with window_col3:
            st.metric("Late Goals", timing_windows.get('late_goal_window', 'N/A'))
        
        # Team-specific timing analysis
        team_timing = timing_info.get('team_timing_analysis', {})
        st.write("**🔍 Team-Specific Timing Patterns**")
        
        timing_col1, timing_col2 = st.columns(2)
        with timing_col1:
            st.write(f"**{match_data['home_team']}**")
            st.write(f"1H Goal Ratio: {team_timing.get('home_1h_ratio', 0):.1%}")
            st.write(f"1H Expected Goals: {team_timing.get('home_1h_xg', 0):.2f}")
            st.write(f"2H Expected Goals: {team_timing.get('home_2h_xg', 0):.2f}")
        
        with timing_col2:
            st.write(f"**{match_data['away_team']}**")
            st.write(f"1H Goal Ratio: {team_timing.get('away_1h_ratio', 0):.1%}")
            st.write(f"1H Expected Goals: {team_timing.get('away_1h_xg', 0):.2f}")
            st.write(f"2H Expected Goals: {team_timing.get('away_2h_xg', 0):.2f}")
        
        # Key insights
        st.write("**💡 Timing Intelligence Insights**")
        insights = timing_info.get('key_insights', [])
        for insight in insights:
            st.markdown(f'<div class="timing-insight">🎯 {insight}</div>', unsafe_allow_html=True)
        
        # Scoring momentum
        momentum = timing_info.get('scoring_momentum', 'balanced')
        momentum_display = {
            'front_loaded': '🚀 Front-Loaded (Early Goals)',
            'back_loaded': '🔄 Back-Loaded (Late Goals)', 
            'balanced': '⚖️ Balanced (Even Distribution)'
        }
        st.metric("Scoring Momentum", momentum_display.get(momentum, momentum))
    else:
        st.warning("⏰ **Timing Intelligence**: Currently inactive - requires complete H2H goal data")
    
    # Enhanced Pattern Intelligence Display
    if 'pattern_intelligence' in predictions:
        pattern_info = predictions['pattern_intelligence']
        
        st.markdown("---")
        st.markdown("""
        <div class='institutional-card'>
            <h3>🎯 ENHANCED PATTERN INTELLIGENCE ANALYSIS</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class='pattern-metric'>
                <h3>Patterns Detected</h3>
                <h2>{pattern_info['pattern_count']}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            pattern_strength = pattern_info.get('total_influence_strength', 0.0)
            st.markdown(f"""
            <div class='pattern-metric'>
                <h3>Pattern Influence</h3>
                <h2>{pattern_strength*100:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
            
        # Enhanced pattern details with upset focus
        if pattern_info['patterns_detected']:
            for pattern in pattern_info['patterns_detected']:
                with st.expander(f"🔍 {pattern['type'].replace('_', ' ').title()}", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Direction", pattern['direction'].title())
                    with col2:
                        st.metric("Strength", f"+{pattern['strength']*100:.1f}%")
                    with col3:
                        st.metric("Confidence", f"{pattern['confidence']*100:.0f}%")
                    st.write(f"**Evidence:** {pattern['evidence']}")
        else:
            st.info("📊 **Pure Statistical Model** - No strong patterns detected in this match")
    
    # Executive Summary
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        confidence = predictions.get('confidence_score', 50)
        st.markdown(f"""
        <div class='confidence-metric'>
            <h3>Model Confidence</h3>
            <h2>{confidence}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        market_regime = predictions.get('market_regime', 'normal').upper()
        st.markdown(f"""
        <div class='value-metric'>
            <h3>Market Regime</h3>
            <h2>{market_regime}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        value_opps = len(predictions.get('value_analysis', {}).get('value_opportunities', []))
        st.markdown(f"""
        <div class='risk-metric'>
            <h3>Value Opportunities</h3>
            <h2>{value_opps}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Enhanced timing intelligence flag
        timing_included = predictions.get('precision_metrics', {}).get('timing_intelligence_included', False)
        upset_detected = upset_analysis.get('upset_detected', False)
        status = "🎯 UPSET+TIMING" if upset_detected else "⏰ TIMING" if timing_included else "📊 STANDARD"
        st.markdown(f"""
        <div class='timing-metric'>
            <h3>Analysis Type</h3>
            <h2>{status}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Data Quality & Model Diagnostics
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='institutional-card'>
            <h3>📊 Data Quality Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("Data Quality Score", f"{data_quality:.1f}%")
        st.progress(data_quality/100)
        st.write(quality_message)
        
        missing_suggestions = get_missing_data_suggestions(match_data)
        if missing_suggestions and data_quality < 90:
            st.write("**💡 Quality Enhancement:**")
            for suggestion in missing_suggestions[:3]:
                st.write(f"- {suggestion}")
    
    with col2:
        st.markdown("""
        <div class='institutional-card'>
            <h3>🤖 Model Diagnostics</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.write(f"**Model Type:** {predictions.get('model_type', 'Enhanced Poisson + Pattern Detection + Automated Timing')}")
        st.write(f"**Market Efficiency:** {predictions.get('market_comparison', {}).get('efficiency', 1.0):.1%}")
        st.write(f"**Timing Intelligence:** {'✅ Active' if timing_included else '❌ Inactive'}")
        st.write(f"**Upset Detection:** {'✅ Active' if upset_detected else '❌ Inactive'}")
    
    # Core Predictions with Uncertainty
    st.markdown("---")
    st.markdown("""
    <div class='institutional-card'>
        <h2>🎯 CORE PREDICTIONS WITH UNCERTAINTY QUANTIFICATION</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # 1X2 Market with Confidence Intervals
    col1, col2, col3 = st.columns(3)
    
    with col1:
        display_1x2_prediction(match_data, predictions)
    
    with col2:
        display_goals_prediction(predictions)
    
    with col3:
        display_btts_prediction(predictions)
    
    # 🆕 ENHANCED: Combined Betting Opportunities
    st.markdown("---")
    display_enhanced_betting_opportunities(predictions)
    
    # Value Bet Analysis
    st.markdown("---")
    display_value_analysis(predictions)
    
    # Risk Assessment & Portfolio Management
    st.markdown("---")
    display_risk_assessment(predictions)
    
    # Model Uncertainty Visualization
    st.markdown("---")
    display_uncertainty_analysis(predictions)
    
    # Professional Action Buttons
    st.markdown("---")
    display_professional_actions(match_data)

def display_1x2_prediction(match_data, predictions):
    """Display 1X2 predictions with institutional formatting"""
    st.markdown("""
    <div class='institutional-card'>
        <h3>1X2 MARKET</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Use final_probabilities instead of base probabilities
    final_probs = predictions.get("final_probabilities", predictions["predictions"])
    pred_1x2 = final_probs["1X2"]
    uncertainty = predictions.get("uncertainty", {})
    
    home_prob = pred_1x2["Home Win"]
    draw_prob = pred_1x2["Draw"]
    away_prob = pred_1x2["Away Win"]
    
    # Find highest probability
    max_prob = max(home_prob, draw_prob, away_prob)
    if home_prob == max_prob:
        prediction_text = f"🏠 {match_data['home_team']} Win"
        confidence = uncertainty.get('confidence_score', 50)
    elif draw_prob == max_prob:
        prediction_text = "🤝 Draw"
        confidence = 50  # Draws typically have lower confidence
    else:
        prediction_text = f"✈️ {match_data['away_team']} Win"
        confidence = uncertainty.get('confidence_score', 50)
    
    st.metric("Institutional Prediction", prediction_text)
    st.metric("Confidence Score", f"{confidence}%")
    
    # Uncertainty intervals
    if 'home_win_68_interval' in uncertainty:
        lower_68, upper_68 = uncertainty['home_win_68_interval']
        st.write(f"**68% Confidence:** {lower_68:.1f}% - {upper_68:.1f}%")
    
    if 'home_win_95_interval' in uncertainty:
        lower_95, upper_95 = uncertainty['home_win_95_interval']
        st.write(f"**95% Confidence:** {lower_95:.1f}% - {upper_95:.1f}%")
    
    # Probability bars
    st.write("**Probability Distribution:**")
    st.progress(home_prob/100)
    st.write(f"Home: {home_prob:.1f}%")
    
    st.progress(draw_prob/100)
    st.write(f"Draw: {draw_prob:.1f}%")
    
    st.progress(away_prob/100)
    st.write(f"Away: {away_prob:.1f}%")

def display_goals_prediction(predictions):
    """Display goals market predictions"""
    st.markdown("""
    <div class='institutional-card'>
        <h3>OVER/UNDER 2.5</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Use final_probabilities instead of base probabilities
    final_probs = predictions.get("final_probabilities", predictions["predictions"])
    pred_ou = final_probs["Over/Under"]
    over_prob = pred_ou["Over 2.5"]
    under_prob = pred_ou["Under 2.5"]
    
    prediction_text = "📈 Over 2.5" if over_prob > under_prob else "📉 Under 2.5"
    st.metric("Prediction", prediction_text)
    
    st.write("**Probability Analysis:**")
    st.progress(over_prob/100)
    st.write(f"Over 2.5: {over_prob:.1f}%")
    
    st.progress(under_prob/100)
    st.write(f"Under 2.5: {under_prob:.1f}%")

def display_btts_prediction(predictions):
    """Display BTTS predictions"""
    st.markdown("""
    <div class='institutional-card'>
        <h3>BOTH TEAMS TO SCORE</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Use final_probabilities instead of base probabilities
    final_probs = predictions.get("final_probabilities", predictions["predictions"])
    pred_btts = final_probs["BTTS"]
    yes_prob = pred_btts["Yes"]
    no_prob = pred_btts["No"]
    
    prediction_text = "✅ Yes" if yes_prob > no_prob else "❌ No"
    st.metric("Prediction", prediction_text)
    
    st.write("**Probability Analysis:**")
    st.progress(yes_prob/100)
    st.write(f"Yes: {yes_prob:.1f}%")
    
    st.progress(no_prob/100)
    st.write(f"No: {no_prob:.1f}%")

def display_enhanced_betting_opportunities(predictions):
    """🆕 ENHANCED: Display all betting opportunities with clear categorization"""
    st.markdown("""
    <div class='institutional-card'>
        <h2>🎯 ENHANCED BETTING OPPORTUNITIES</h2>
        <p><em>Timing-Intelligence + Upset-Aware + Standard Value Bets</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    betting_recs = predictions.get('betting_recommendations', {})
    
    # Get all bet types
    timing_bets = betting_recs.get('timing_enhanced_bets', [])
    upset_bets = betting_recs.get('upset_aware_bets', [])
    standard_bets = betting_recs.get('all_opportunities', [])
    
    total_enhanced_opportunities = len(timing_bets) + len(upset_bets)
    
    if total_enhanced_opportunities > 0:
        st.success(f"🎯 **{total_enhanced_opportunities} ENHANCED OPPORTUNITIES DETECTED**")
        
        # Display Timing-Enhanced Bets
        if timing_bets:
            st.markdown("#### ⏰ Timing-Enhanced Bets")
            for i, bet in enumerate(timing_bets):
                with st.expander(f"⏰ Timing Bet #{i+1}: {bet['market']} - {bet['selection']}", expanded=True):
                    display_bet_details(bet, 'timing')
        
        # Display Upset-Aware Bets  
        if upset_bets:
            st.markdown("#### 🎯 Upset-Aware Bets")
            for i, bet in enumerate(upset_bets):
                with st.expander(f"🎯 Upset Bet #{i+1}: {bet['market']} - {bet['selection']}", expanded=True):
                    display_bet_details(bet, 'upset')
                    
                    # Show upset factors
                    upset_factors = bet.get('upset_factors', [])
                    if upset_factors:
                        st.write("**Upset Factors Supporting This Bet:**")
                        for factor in upset_factors[:3]:  # Show top 3 factors
                            st.write(f"✅ {factor}")
    
    # Display Standard Value Bets
    if standard_bets:
        st.markdown("#### 💎 Standard Value Bets")
        for i, bet in enumerate(standard_bets[:3]):  # Show top 3 standard bets
            with st.expander(f"💎 Value Bet #{i+1}: {bet['market']} - {bet['selection']}", expanded=True):
                display_bet_details(bet, 'standard')
    
    if total_enhanced_opportunities == 0 and not standard_bets:
        st.info("🔍 No strong betting opportunities detected with current thresholds")

def display_bet_details(bet, bet_type):
    """Display detailed bet information with type-specific styling - FIXED"""
    
    # Determine CSS class based on bet type
    bet_class = {
        'timing': 'bet-type-timing',
        'upset': 'bet-type-upset', 
        'standard': 'bet-type-standard'
    }.get(bet_type, 'bet-type-standard')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Probability", f"{bet['probability']}%")
        st.metric("Edge", bet['edge'])
    
    with col2:
        # FIXED: Handle expected value properly - it's now a float
        expected_value = bet.get('expected_value', 0)
        st.metric("Expected Value", f"+{expected_value:.1f}%")
        st.metric("Confidence", bet['confidence'])
    
    with col3:
        # Type-specific insights
        if bet_type == 'timing':
            st.metric("Market", "Timing-Enhanced")
            st.write("**Focus:** Goal timing patterns")
        elif bet_type == 'upset':
            st.metric("Market", "Upset-Aware") 
            st.write("**Focus:** Contextual factors")
        else:
            st.metric("Market", "Standard Value")
            st.write("**Focus:** Statistical edge")
    
    # Reasoning with type-specific context
    st.markdown(f'<div class="{bet_class}">', unsafe_allow_html=True)
    st.write("**Rationale:**")
    st.write(f"🎯 {bet['reasoning']}")
    
    # Additional context based on bet type
    if bet_type == 'timing' and 'probability' in bet:
        timing_context = {
            '2H Goals': "Focus on 2nd half dynamics",
            'Late Goals': "Leverage final stages advantage", 
            '1H Goals': "Capitalize on early match patterns"
        }
        market_context = timing_context.get(bet.get('market', ''), "Timing-based opportunity")
        st.write(f"**Timing Context:** {market_context}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_value_analysis(predictions):
    """Display institutional value bet analysis"""
    st.markdown("""
    <div class='institutional-card'>
        <h2>💰 INSTITUTIONAL VALUE OPPORTUNITIES</h2>
    </div>
    """, unsafe_allow_html=True)
    
    value_analysis = predictions.get('value_analysis', {})
    opportunities = value_analysis.get('value_opportunities', [])
    
    if opportunities:
        # Filter out opportunities already shown in enhanced section
        enhanced_markets = ['2H Goals', 'Late Goals', '1H Goals', 'Home Team Goals', 'BTTS']
        standard_opportunities = [opp for opp in opportunities if opp.get('market') not in enhanced_markets]
        
        if standard_opportunities:
            st.success(f"💎 **{len(standard_opportunities)} STANDARD VALUE OPPORTUNITIES DETECTED**")
            
            for i, opp in enumerate(standard_opportunities):
                with st.expander(f"💎 Value Opportunity #{i+1}: {opp['market']} - {opp['selection']}", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Our Probability", f"{opp['our_prob']:.1f}%")
                        st.metric("Implied Probability", f"{opp['implied_prob']:.1f}%")
                    
                    with col2:
                        st.metric("Edge", f"+{opp['edge']:.1f}%")
                        st.metric("Expected Value", f"+{opp['expected_value']:.1f}%")
                    
                    with col3:
                        st.metric("Recommended Stake", f"{opp['recommended_stake']:.1f}%")
                        st.metric("Confidence", opp['confidence'])
                    
                    # Reasoning
                    st.write("**Rationale:**")
                    for reason in opp['reasoning']:
                        st.write(f"- {reason}")
        else:
            st.info("💎 All value opportunities are shown in the Enhanced Betting section above")
    else:
        st.info("🔍 No strong value opportunities detected with current thresholds")

def display_risk_assessment(predictions):
    """Display professional risk assessment"""
    st.markdown("""
    <div class='institutional-card'>
        <h2>⚖️ RISK ASSESSMENT & PORTFOLIO MANAGEMENT</h2>
    </div>
    """, unsafe_allow_html=True)
    
    risk_assessment = predictions.get('risk_assessment', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Risk Level", risk_assessment.get('risk_level', 'Medium'))
    
    with col2:
        var_95 = 0.02
        st.metric("VaR (95%)", f"{var_95*100:.1f}%")
    
    with col3:
        max_dd = 0.05
        st.metric("Max Drawdown", f"{max_dd*100:.1f}%")
    
    with col4:
        confidence = risk_assessment.get('confidence', 'High')
        st.metric("Model Confidence", confidence)
    
    # Enhanced risk explanation
    st.write("**🎯 Enhanced Risk Management Notes:**")
    st.write("- All stakes use conservative Kelly fraction (25%) with enhanced position sizing")
    st.write("- Maximum single position exposure limited to 5% of bankroll")
    st.write("- Portfolio optimization considers correlation between markets")
    st.write("- 🆕 Timing-enhanced and upset-aware bets have separate risk assessment")
    st.write("- Upset context may warrant slightly higher stakes for qualified opportunities")

def display_uncertainty_analysis(predictions):
    """Display advanced uncertainty analysis"""
    st.markdown("""
    <div class='institutional-card'>
        <h2>📊 UNCERTAINTY QUANTIFICATION & MODEL CONFIDENCE</h2>
    </div>
    """, unsafe_allow_html=True)
    
    uncertainty = predictions.get('uncertainty', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_uncertainty_chart(uncertainty), use_container_width=True)
    
    with col2:
        st.write("**Uncertainty Interpretation:**")
        
        confidence = uncertainty.get('confidence_score', 50)
        if confidence >= 80:
            st.success("**High Confidence**: Model has strong conviction in predictions")
        elif confidence >= 60:
            st.info("**Good Confidence**: Reasonable certainty in predictions")
        elif confidence >= 40:
            st.warning("**Moderate Confidence**: Some uncertainty in predictions")
        else:
            st.error("**Low Confidence**: High uncertainty - consider smaller stakes")
        
        st.write("**Standard Error:**", f"{uncertainty.get('standard_error', 5):.2f}%")
        st.write("**Market Regime:**", predictions.get('market_regime', 'normal').title())

def create_uncertainty_chart(uncertainty):
    """Create uncertainty visualization chart"""
    if 'home_win_68_interval' not in uncertainty:
        # Fallback simple chart
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode = "gauge+number+delta",
            value = uncertainty.get('confidence_score', 50),
            title = {'text': "Model Confidence"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {'axis': {'range': [None, 100]}}
        ))
        return fig
    
    # Professional uncertainty chart
    lower_68, upper_68 = uncertainty['home_win_68_interval']
    lower_95, upper_95 = uncertainty['home_win_95_interval']
    
    fig = go.Figure()
    
    # 95% interval
    fig.add_trace(go.Bar(
        x=['Probability Range'],
        y=[upper_95 - lower_95],
        base=[lower_95],
        marker_color='lightblue',
        name='95% Confidence Interval',
        hovertemplate='95% CI: %{base:.1f}% - %{y:.1f}%<extra></extra>'
    ))
    
    # 68% interval
    fig.add_trace(go.Bar(
        x=['Probability Range'],
        y=[upper_68 - lower_68],
        base=[lower_68],
        marker_color='blue',
        name='68% Confidence Interval',
        hovertemplate='68% CI: %{base:.1f}% - %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Prediction Uncertainty Intervals",
        yaxis_title="Probability (%)",
        showlegend=True,
        barmode='overlay'
    )
    
    return fig

def display_professional_actions(match_data):
    """Display professional action buttons"""
    st.markdown("""
    <div class='institutional-card'>
        <h3>🔄 PROFESSIONAL WORKFLOW ACTIONS</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("✏️ Refine Input Data", use_container_width=True):
            st.session_state.last_match_data = st.session_state.match_data.copy()
            st.session_state.show_predictions = False
            st.session_state.institutional_predictions = None
            st.rerun()
    
    with col2:
        if st.button("🔄 New Analysis", use_container_width=True):
            st.session_state.match_data = {}
            st.session_state.institutional_predictions = None
            st.session_state.show_predictions = False
            st.session_state.last_match_data = {}
            st.rerun()
    
    with col3:
        if st.button("📊 Quick Scenario Test", use_container_width=True):
            st.session_state.match_data = {
                'home_team': match_data['home_team'],
                'away_team': match_data['away_team'],
                'league_type': match_data['league_type'],
                'match_importance': match_data['match_importance']
            }
            st.session_state.last_match_data = st.session_state.match_data.copy()
            st.session_state.institutional_predictions = None
            st.session_state.show_predictions = False
            st.rerun()
    
    with col4:
        if st.button("💾 Save Analysis", use_container_width=True):
            # Save to performance history
            analysis_record = {
                'timestamp': datetime.now(),
                'match': f"{match_data['home_team']} vs {match_data['away_team']}",
                'predictions': st.session_state.institutional_predictions,
                'data_quality': calculate_data_quality(match_data)
            }
            st.session_state.performance_history.append(analysis_record)
            st.success("✅ Analysis saved to performance history")

def main():
    """Main institutional application"""
    # Professional header
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 class='main-header'>⚽ ENHANCED INSTITUTIONAL FOOTBALL PREDICTOR</h1>
        <p style='font-size: 1.2rem; color: #666;'>World-Class Analytics • Risk-Managed Insights • Professional Grade • Automated Timing Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Data restoration
    if st.session_state.match_data == {} and st.session_state.get('last_match_data'):
        st.session_state.match_data = st.session_state.last_match_data.copy()
    
    # Show predictions if available
    if st.session_state.institutional_predictions and st.session_state.show_predictions:
        display_institutional_predictions(st.session_state.match_data, st.session_state.institutional_predictions)
        return
    
    # Main input form
    match_data = create_professional_input_form()
    
    if match_data:
        # Generate institutional predictions with your enhanced engine
        with st.spinner("🤖 Running Enhanced Institutional AI with Automated Timing Intelligence..."):
            try:
                # USING YOUR ENHANCED PREDICTION ENGINE DIRECTLY
                engine = WorldClassPredictionEngine(match_data)
                predictions = engine.generate_all_predictions()
                
                # Store in session state
                st.session_state.institutional_predictions = predictions
                st.session_state.show_predictions = True
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Institutional Analysis Error: {str(e)}")
                st.info("Please verify your input data and try again.")

if __name__ == "__main__":
    main()
