# streamlit_app.py - COMPLETE IMPROVED VERSION
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from typing import Dict, Any
from datetime import datetime

# Import the IMPROVED PREDICTION ENGINE
try:
    from prediction_engine import AdvancedFootballPredictor, DynamicTierCalibrator, EliteStakeCalculator
except ImportError as e:
    st.error(f"❌ Could not import prediction_engine: {str(e)}")
    st.info("💡 Make sure prediction_engine.py is in the same directory")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🧠 Smart Football Predictor",
    page_icon="⚽", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header { 
        font-size: 2.5rem !important; 
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.3rem !important;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .smart-feature {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .dynamic-tier {
        background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
    }
    .prediction-card { 
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .risk-low { border-left-color: #4CAF50 !important; }
    .risk-medium { border-left-color: #FF9800 !important; }
    .risk-high { border-left-color: #f44336 !important; }
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #f0f2f6;
    }
    .tier-badge {
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: bold;
        color: white;
        display: inline-block;
        margin-left: 0.5rem;
    }
    .tier-elite { background: #e74c3c; }
    .tier-strong { background: #e67e22; }
    .tier-medium { background: #f1c40f; color: black; }
    .tier-weak { background: #95a5a6; }
    .value-exceptional { border-left-color: #4CAF50 !important; background: #f8fff8; }
    .value-high { border-left-color: #8BC34A !important; background: #f9fff9; }
    .value-good { border-left-color: #FFC107 !important; background: #fffdf6; }
    .alignment-perfect {
        background: #f8fff8;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .alignment-warning {
        background: #fffaf2;
        border-left: 4px solid #FF9800;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .goals-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .recommendation-yes {
        border-left-color: #4CAF50 !important;
        background: #f8fff8;
    }
    .recommendation-no {
        border-left-color: #f44336 !important;
        background: #fff5f5;
    }
    .confidence-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        color: white;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .confidence-high { background: #4CAF50; }
    .confidence-medium { background: #FF9800; }
    .confidence-low { background: #f44336; }
</style>
""", unsafe_allow_html=True)

def safe_get(dictionary, *keys, default=None):
    """Safely get nested dictionary keys"""
    if dictionary is None:
        return default
        
    current = dictionary
    for key in keys:
        try:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        except (TypeError, KeyError):
            return default
    return current

def get_league_display_name(league_id: str) -> str:
    """Get display name for league"""
    league_names = {
        'premier_league': 'Premier League 🏴󠁧󠁢󠁥󠁮󠁧󠁿',
        'la_liga': 'La Liga 🇪🇸', 
        'serie_a': 'Serie A 🇮🇹',
        'bundesliga': 'Bundesliga 🇩🇪',
        'ligue_1': 'Ligue 1 🇫🇷'
    }
    return league_names.get(league_id, league_id)

def get_league_badge(league_id: str) -> str:
    """Get CSS class for league badge"""
    league_classes = {
        'premier_league': 'premier-league',
        'la_liga': 'la-liga',
        'serie_a': 'serie-a', 
        'bundesliga': 'bundesliga',
        'ligue_1': 'ligue-1'
    }
    return league_classes.get(league_id, 'premier-league')

def display_smart_features():
    """Display smart features overview"""
    st.markdown('<div class="smart-feature">', unsafe_allow_html=True)
    st.markdown("### 🧠 SMART FEATURES ACTIVATED")
    
    smart_features = {
        "Feature": ["Dynamic Tier System", "Context-Aware XG", "Performance Monitoring", 
                   "Smart Value Detection", "Real-time Adjustments"],
        "Status": ["✅ ACTIVE", "✅ ACTIVE", "✅ ACTIVE", "✅ ACTIVE", "✅ ACTIVE"],
        "Impact": ["Real-time Team Assessment", "Motivational Context", "Accuracy Tracking",
                  "Alignment Checking", "Live Model Updates"]
    }
    
    df = pd.DataFrame(smart_features)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

def create_input_form():
    """Create improved input form"""
    
    st.markdown('<p class="main-header">🧠 Smart Football Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Dynamic Multi-League Analysis with Real-time Intelligence</p>', unsafe_allow_html=True)
    
    # Display smart features
    display_smart_features()
    
    # Smart System Overview
    with st.expander("🏗️ Smart System Architecture", expanded=True):
        st.markdown("""
        ### 🎯 IMPROVED PREDICTION SYSTEM
        
        **Enhanced Intelligence** 🧠
        - **Dynamic Tier System** - Real-time team performance assessment
        - **Context-Aware Predictions** - Motivational and situational factors
        - **Performance Monitoring** - Continuous accuracy tracking
        - **Smart Value Detection** - Alignment-aware betting signals
        - **Simplified But Smarter** - Reduced complexity, increased accuracy
        
        **Key Improvements** 🚀
        - Better handling of team form fluctuations
        - Improved context integration
        - More accurate risk assessment
        - Enhanced prediction explanations
        """)
    
    # League Selection
    st.markdown("### 🌍 League Selection")
    league_options = {
        'premier_league': 'Premier League 🏴󠁧󠁢󠁥󠁮󠁧󠁿',
        'la_liga': 'La Liga 🇪🇸',
        'serie_a': 'Serie A 🇮🇹', 
        'bundesliga': 'Bundesliga 🇩🇪',
        'ligue_1': 'Ligue 1 🇫🇷'
    }
    
    selected_league = st.selectbox(
        "Select League",
        options=list(league_options.keys()),
        format_func=lambda x: league_options[x],
        key="league_selection"
    )
    
    # Display league badge
    league_badge_class = get_league_badge(selected_league)
    league_display_name = get_league_display_name(selected_league)
    st.markdown(f'<span class="league-badge {league_badge_class}">{league_display_name}</span>', unsafe_allow_html=True)
    
    # Initialize team calibrator
    calibrator = DynamicTierCalibrator()
    league_teams = calibrator.get_league_teams(selected_league)
    
    if not league_teams:
        st.error(f"❌ No teams found for {league_display_name}")
        return None
    
    tab1, tab2, tab3 = st.tabs(["🏠 Team Data", "🎯 Match Context", "💰 Market Data"])

    with tab1:
        st.markdown("### 🏠 Team Data Input")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Home Team")
            home_team = st.selectbox(
                "Team Name", 
                options=league_teams,
                index=min(5, len(league_teams) - 1),
                key="home_team"
            )
            
            home_goals = st.number_input("Total Goals (Last 6 Games)", min_value=0, value=8, key="home_goals")
            home_conceded = st.number_input("Total Conceded (Last 6 Games)", min_value=0, value=8, key="home_conceded")
            home_goals_home = st.number_input("Home Goals (Last 3 Home Games)", min_value=0, value=4, key="home_goals_home")
            
            st.write("**Recent Form (Last 6 Games)**")
            home_form = st.multiselect(
                f"{home_team} Results",
                options=["Win (3 pts)", "Draw (1 pt)", "Loss (0 pts)"],
                default=["Win (3 pts)", "Loss (0 pts)", "Win (3 pts)", "Loss (0 pts)", "Loss (0 pts)", "Win (3 pts)"],
                key="home_form"
            )
            
        with col2:
            st.subheader("Away Team")
            away_team = st.selectbox(
                "Team Name",
                options=league_teams,
                index=0,
                key="away_team"
            )
            
            away_goals = st.number_input("Total Goals (Last 6 Games)", min_value=0, value=12, key="away_goals")
            away_conceded = st.number_input("Total Conceded (Last 6 Games)", min_value=0, value=6, key="away_conceded")
            away_goals_away = st.number_input("Away Goals (Last 3 Away Games)", min_value=0, value=7, key="away_goals_away")
            
            st.write("**Recent Form (Last 6 Games)**")
            away_form = st.multiselect(
                f"{away_team} Results", 
                options=["Win (3 pts)", "Draw (1 pt)", "Loss (0 pts)"],
                default=["Win (3 pts)", "Win (3 pts)", "Draw (1 pt)", "Win (3 pts)", "Win (3 pts)", "Win (3 pts)"],
                key="away_form"
            )

    with tab2:
        st.markdown("### 🎯 Match Context Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Team Motivation**")
            home_motivation = st.select_slider(
                "Home Team Motivation",
                options=["Low", "Normal", "High", "Very High"],
                value="High",
                key="home_motivation"
            )
            away_motivation = st.select_slider(
                "Away Team Motivation", 
                options=["Low", "Normal", "High", "Very High"],
                value="Normal", 
                key="away_motivation"
            )
            
        with col2:
            st.write("**Match Circumstances**")
            is_derby = st.checkbox("Derby Match", value=False, key="is_derby")
            relegation_battle = st.checkbox("Relegation Battle", value=True, key="relegation_battle")
            
        # Head-to-head
        with st.expander("📊 Head-to-Head History"):
            h2h_col1, h2h_col2 = st.columns(2)
            with h2h_col1:
                h2h_matches = st.number_input("Total H2H Matches", min_value=0, value=5, key="h2h_matches")
                h2h_home_wins = st.number_input("Home Wins", min_value=0, value=3, key="h2h_home_wins")
            with h2h_col2:
                h2h_away_wins = st.number_input("Away Wins", min_value=0, value=2, key="h2h_away_wins")
                h2h_draws = st.number_input("Draws", min_value=0, value=1, key="h2h_draws")

    with tab3:
        st.markdown("### 💰 Market Data Input") 
        
        odds_col1, odds_col2 = st.columns(2)
        
        with odds_col1:
            st.write("**1X2 Market**")
            home_odds = st.number_input("Home Win Odds", min_value=1.01, value=2.70, step=0.01, key="home_odds")
            draw_odds = st.number_input("Draw Odds", min_value=1.01, value=3.75, step=0.01, key="draw_odds")
            away_odds = st.number_input("Away Win Odds", min_value=1.01, value=2.38, step=0.01, key="away_odds")
        
        with odds_col2:
            st.write("**Goals Markets**")
            over_25_odds = st.number_input("Over 2.5 Goals", min_value=1.01, value=1.80, step=0.01, key="over_25_odds")
            under_25_odds = st.number_input("Under 2.5 Goals", min_value=1.01, value=2.00, step=0.01, key="under_25_odds")
            btts_yes_odds = st.number_input("BTTS Yes", min_value=1.01, value=1.90, step=0.01, key="btts_yes_odds")
            btts_no_odds = st.number_input("BTTS No", min_value=1.01, value=1.90, step=0.01, key="btts_no_odds")

    # Submit button
    submitted = st.button("🧠 GENERATE SMART ANALYSIS", type="primary", use_container_width=True)
    
    if submitted:
        if not home_team or not away_team:
            st.error("❌ Please enter both team names")
            return None
        
        if home_team == away_team:
            st.error("❌ Home and away teams cannot be the same")
            return None
        
        # Convert form selections to points
        form_map = {"Win (3 pts)": 3, "Draw (1 pt)": 1, "Loss (0 pts)": 0}
        home_form_points = [form_map[result] for result in home_form]
        away_form_points = [form_map[result] for result in away_form]
        
        # Convert motivation
        motivation_map = {"Low": "Low", "Normal": "Normal", "High": "High", "Very High": "Very High"}
        
        # Market odds
        market_odds = {
            '1x2 Home': home_odds,
            '1x2 Draw': draw_odds,
            '1x2 Away': away_odds,
            'Over 2.5 Goals': over_25_odds,
            'Under 2.5 Goals': under_25_odds,
            'BTTS Yes': btts_yes_odds,
            'BTTS No': btts_no_odds,
        }
        
        # Complete match data
        match_data = {
            'home_team': home_team,
            'away_team': away_team,
            'league': selected_league,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'home_conceded': home_conceded,
            'away_conceded': away_conceded,
            'home_goals_home': home_goals_home,
            'away_goals_away': away_goals_away,
            'home_form': home_form_points,
            'away_form': away_form_points,
            'h2h_data': {
                'matches': h2h_matches,
                'home_wins': h2h_home_wins,
                'away_wins': h2h_away_wins,
                'draws': h2h_draws
            },
            'match_context': {
                'home_motivation': motivation_map[home_motivation],
                'away_motivation': motivation_map[away_motivation],
                'is_derby': is_derby,
                'relegation_battle': relegation_battle
            },
            'market_odds': market_odds
        }
        
        return match_data
    
    return None

def display_dynamic_tiers(predictions):
    """Display dynamic tier information"""
    team_tiers = safe_get(predictions, 'team_tiers') or {}
    home_tier = team_tiers.get('home', 'MEDIUM')
    away_tier = team_tiers.get('away', 'MEDIUM')
    
    st.markdown('<div class="dynamic-tier">', unsafe_allow_html=True)
    st.markdown(f"### 🎯 DYNAMIC TEAM ASSESSMENT")
    st.markdown(f"**{predictions['match'].split(' vs ')[0]}**: <span class='tier-badge tier-{home_tier.lower()}'>{home_tier}</span>", unsafe_allow_html=True)
    st.markdown(f"**{predictions['match'].split(' vs ')[1]}**: <span class='tier-badge tier-{away_tier.lower()}'>{away_tier}</span>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def display_goals_analysis(predictions):
    """Display goals analysis"""
    st.markdown('<div class="section-title">⚽ Goals Analysis</div>', unsafe_allow_html=True)
    
    # Get probabilities with safe defaults
    btts_yes = safe_get(predictions, 'probabilities', 'both_teams_score', 'yes') or 0
    btts_no = safe_get(predictions, 'probabilities', 'both_teams_score', 'no') or 0
    
    over_25 = safe_get(predictions, 'probabilities', 'over_under', 'over_25') or 0
    under_25 = safe_get(predictions, 'probabilities', 'over_under', 'under_25') or 0
    
    first_half = safe_get(predictions, 'probabilities', 'goal_timing', 'first_half') or 0
    second_half = safe_get(predictions, 'probabilities', 'goal_timing', 'second_half') or 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # BTTS - Show the HIGHER probability as primary
        if btts_no > btts_yes:
            recommendation = "NO"
            primary_prob = btts_no
            secondary_prob = btts_yes
            card_class = "recommendation-no"
            emoji = "❌"
        else:
            recommendation = "YES"
            primary_prob = btts_yes
            secondary_prob = btts_no
            card_class = "recommendation-yes"
            emoji = "✅"
        
        confidence = "HIGH" if abs(primary_prob - 50) > 20 else "MEDIUM" if abs(primary_prob - 50) > 10 else "LOW"
        
        st.markdown(f'''
        <div class="goals-card {card_class}">
            <h4>{emoji} Both Teams Score</h4>
            <div style="font-size: 1.8rem; font-weight: bold; color: #333; margin: 0.5rem 0;">
                {recommendation}: {primary_prob:.1f}%
            </div>
            <div style="font-size: 0.9rem; color: #666; margin: 0.3rem 0;">
                {('NO' if recommendation == 'YES' else 'YES')}: {secondary_prob:.1f}%
            </div>
            <span class="confidence-badge confidence-{confidence.lower()}">
                {confidence} CONFIDENCE
            </span>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        # Over/Under 2.5 - Show the HIGHER probability as primary
        if under_25 > over_25:
            recommendation = "UNDER"
            primary_prob = under_25
            secondary_prob = over_25
            card_class = "recommendation-no"
            emoji = "❌"
        else:
            recommendation = "OVER"
            primary_prob = over_25
            secondary_prob = under_25
            card_class = "recommendation-yes"
            emoji = "✅"
        
        confidence = "HIGH" if abs(primary_prob - 50) > 20 else "MEDIUM" if abs(primary_prob - 50) > 10 else "LOW"
        
        st.markdown(f'''
        <div class="goals-card {card_class}">
            <h4>{emoji} Over/Under 2.5</h4>
            <div style="font-size: 1.8rem; font-weight: bold; color: #333; margin: 0.5rem 0;">
                {recommendation}: {primary_prob:.1f}%
            </div>
            <div style="font-size: 0.9rem; color: #666; margin: 0.3rem 0;">
                {('OVER' if recommendation == 'UNDER' else 'UNDER')}: {secondary_prob:.1f}%
            </div>
            <span class="confidence-badge confidence-{confidence.lower()}">
                {confidence} CONFIDENCE
            </span>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        confidence = "HIGH" if first_half > 60 else "MEDIUM" if first_half > 40 else "LOW"
        emoji = "🟢" if first_half > 60 else "🟡" if first_half > 40 else "🔴"
        
        st.markdown(f'''
        <div class="goals-card">
            <h4>🎯 First Half Goal</h4>
            <div style="font-size: 1.8rem; font-weight: bold; color: #667eea; margin: 0.5rem 0;">
                {first_half:.1f}%
            </div>
            <span class="confidence-badge confidence-{confidence.lower()}">
                {emoji} {confidence} CONFIDENCE
            </span>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        confidence = "HIGH" if second_half > 60 else "MEDIUM" if second_half > 40 else "LOW"
        emoji = "🟢" if second_half > 60 else "🟡" if second_half > 40 else "🔴"
        
        st.markdown(f'''
        <div class="goals-card">
            <h4>🎯 Second Half Goal</h4>
            <div style="font-size: 1.8rem; font-weight: bold; color: #667eea; margin: 0.5rem 0;">
                {second_half:.1f}%
            </div>
            <span class="confidence-badge confidence-{confidence.lower()}">
                {emoji} {confidence} CONFIDENCE
            </span>
        </div>
        ''', unsafe_allow_html=True)

def display_predictions(predictions):
    """Display improved predictions"""
    
    st.markdown('<p class="main-header">🎯 Smart Football Predictions</p>', unsafe_allow_html=True)
    
    # Dynamic tiers display
    display_dynamic_tiers(predictions)
    
    # Key metrics
    xg = safe_get(predictions, 'expected_goals') or {'home': 0, 'away': 0}
    match_context = safe_get(predictions, 'match_context') or 'Unknown'
    confidence_score = safe_get(predictions, 'confidence_score') or 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏠 Expected Goals", f"{xg.get('home', 0):.2f}")
    with col2:
        st.metric("✈️ Expected Goals", f"{xg.get('away', 0):.2f}")
    with col3:
        context_emoji = {
            'home_dominance': '🏠',
            'away_counter': '✈️',
            'balanced': '⚖️'
        }.get(match_context, '❓')
        st.metric("Match Context", f"{context_emoji} {match_context.replace('_', ' ').title()}")
    with col4:
        st.metric("Confidence Score", f"{confidence_score}%")
    
    # System validation
    system_validation = safe_get(predictions, 'system_validation') or {}
    alignment_status = system_validation.get('alignment', 'UNKNOWN')
    
    if alignment_status == 'PERFECT':
        st.markdown('<div class="alignment-perfect">✅ <strong>PERFECT SYSTEM ALIGNMENT:</strong> All engines synchronized perfectly</div>', unsafe_allow_html=True)
    elif alignment_status in ['GOOD', 'PARTIAL']:
        st.markdown('<div class="alignment-warning">⚠️ <strong>GOOD ALIGNMENT:</strong> Minor inconsistencies detected</div>', unsafe_allow_html=True)
    else:
        st.warning("🔧 System alignment needs attention")
    
    # Match Outcomes
    st.markdown('<div class="section-title">📈 Match Outcome Probabilities</div>', unsafe_allow_html=True)
    
    outcomes = safe_get(predictions, 'probabilities', 'match_outcomes') or {'home_win': 0, 'draw': 0, 'away_win': 0}
    
    # Create probability bars
    col1, col2, col3 = st.columns(3)
    
    with col1:
        home_win = outcomes.get('home_win', 0)
        st.metric("Home Win", f"{home_win}%")
        st.progress(home_win / 100)
    
    with col2:
        draw = outcomes.get('draw', 0)
        st.metric("Draw", f"{draw}%")
        st.progress(draw / 100)
    
    with col3:
        away_win = outcomes.get('away_win', 0)
        st.metric("Away Win", f"{away_win}%")
        st.progress(away_win / 100)
    
    # Goals Analysis
    display_goals_analysis(predictions)
    
    # Exact Scores
    st.markdown('<div class="section-title">🎯 Most Likely Scores</div>', unsafe_allow_html=True)
    
    exact_scores = safe_get(predictions, 'probabilities', 'exact_scores') or {}
    if exact_scores:
        score_cols = st.columns(6)
        for idx, (score, prob) in enumerate(list(exact_scores.items())[:6]):
            with score_cols[idx]:
                st.metric(f"{score}", f"{prob}%")
    
    # Risk Assessment
    risk = safe_get(predictions, 'risk_assessment') or {}
    risk_class = f"risk-{risk.get('risk_level', 'unknown').lower()}"
    
    st.markdown(f'''
    <div class="prediction-card {risk_class}">
        <h3>📊 Risk Assessment</h3>
        <strong>Risk Level:</strong> {risk.get("risk_level", "UNKNOWN")}<br>
        <strong>Explanation:</strong> {risk.get("explanation", "No data available")}<br>
        <strong>Recommendation:</strong> {risk.get("recommendation", "N/A")}<br>
        <strong>Certainty:</strong> {risk.get("certainty", "N/A")}
    </div>
    ''', unsafe_allow_html=True)
    
    # Key Factors and Summary
    st.markdown('<div class="section-title">🔍 Key Insights</div>', unsafe_allow_html=True)
    
    key_factors = safe_get(predictions, 'key_factors') or []
    if key_factors:
        for factor in key_factors:
            st.write(f"• {factor}")
    
    st.markdown('<div class="section-title">📝 Match Summary</div>', unsafe_allow_html=True)
    summary = safe_get(predictions, 'summary') or "No summary available."
    st.info(summary)

def display_value_detection(predictions):
    """Display value detection results"""
    
    st.markdown('<p class="main-header">💰 Smart Value Detection</p>', unsafe_allow_html=True)
    
    betting_signals = safe_get(predictions, 'betting_signals') or []
    
    # Primary predictions context
    st.markdown('<div class="section-title">🎯 Primary Predictions</div>', unsafe_allow_html=True)
    
    outcomes = safe_get(predictions, 'probabilities', 'match_outcomes') or {}
    btts = safe_get(predictions, 'probabilities', 'both_teams_score') or {}
    over_under = safe_get(predictions, 'probabilities', 'over_under') or {}
    
    col1, col2, col3 = st.columns(3)
    with col1:
        primary_outcome = max(outcomes, key=outcomes.get) if outcomes else 'unknown'
        outcome_map = {'home_win': 'Home Win', 'draw': 'Draw', 'away_win': 'Away Win'}
        st.metric("Primary Outcome", outcome_map.get(primary_outcome, 'Unknown'))
    with col2:
        primary_btts = "YES" if btts.get('yes', 0) > btts.get('no', 0) else "NO"
        st.metric("Primary BTTS", primary_btts)
    with col3:
        primary_ou = "OVER" if over_under.get('over_25', 0) > over_under.get('under_25', 0) else "UNDER"
        st.metric("Primary Over/Under", f"{primary_ou} 2.5")
    
    # Kelly Criterion Explanation
    st.markdown("### 💰 Professional Stake Sizing")
    st.info("""
    **Kelly Criterion Active:**
    - HIGH Confidence: 1/4 Kelly (25% of optimal)
    - MEDIUM Confidence: 1/6 Kelly (16.7% of optimal)  
    - LOW Confidence: 1/12 Kelly (8.3% of optimal)
    - SPECULATIVE: 1/25 Kelly (4% of optimal)
    """)
    
    if not betting_signals:
        st.markdown('<div class="alignment-perfect">', unsafe_allow_html=True)
        st.success("""
        ## ✅ MARKET EFFICIENCY DETECTED
        
        **No significant value bets found:**
        - Market prices align with model probabilities
        - No edges above minimum thresholds  
        - System working as intended
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Value Bet Summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_signals = len(betting_signals)
        st.metric("Value Bets", total_signals)
    
    with col2:
        high_value = len([s for s in betting_signals if s.get('value_rating') in ['EXCEPTIONAL', 'HIGH']])
        st.metric("High Value Signals", high_value)
    
    with col3:
        aligned_count = sum(1 for s in betting_signals if s.get('aligns_with_primary', False))
        st.metric("Aligned Signals", f"{aligned_count}/{total_signals}")
    
    with col4:
        total_stake = sum(s.get('recommended_stake', 0) for s in betting_signals)
        st.metric("Total Portfolio", f"{total_stake * 100:.1f}%")
    
    # Display value bets
    st.markdown('<div class="section-title">🎯 Value Bet Recommendations</div>', unsafe_allow_html=True)
    
    for signal in betting_signals:
        value_class = f"value-{signal.get('value_rating', '').lower()}"
        aligns = signal.get('aligns_with_primary', False)
        alignment_emoji = "✅" if aligns else "⚠️"
        alignment_text = "ALIGNS" if aligns else "CONTRADICTS"
        
        confidence_emoji = {
            'HIGH': '🟢',
            'MEDIUM': '🟡', 
            'LOW': '🔴',
            'SPECULATIVE': '⚪'
        }.get(signal.get('confidence', 'SPECULATIVE'), '⚪')
        
        st.markdown(f'''
        <div class="prediction-card {value_class}">
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <div style="flex: 2;">
                    <strong>{signal.get('market', 'Unknown')}</strong><br>
                    <small>Model: {signal.get('model_prob', 0)}% | Market: {signal.get('book_prob', 0)}%</small>
                    <div style="margin-top: 0.5rem;">
                        <small>{alignment_emoji} <strong>{alignment_text}</strong> with primary prediction</small>
                    </div>
                </div>
                <div style="flex: 1; text-align: right;">
                    <strong style="color: #4CAF50; font-size: 1.1rem;">+{signal.get('edge', 0)}% Edge</strong><br>
                    <small>Stake: {signal.get('recommended_stake', 0)*100:.1f}%</small><br>
                    <small>{confidence_emoji} {signal.get('confidence', 'Unknown')}</small>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

def display_analytics(predictions):
    """Display analytics"""
    
    st.markdown('<p class="main-header">📈 Smart Analytics</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-title">📊 Prediction Intelligence</div>', unsafe_allow_html=True)
        
        confidence = safe_get(predictions, 'confidence_score') or 0
        risk_level = safe_get(predictions, 'risk_assessment', 'risk_level') or 'UNKNOWN'
        
        narrative = safe_get(predictions, 'match_narrative') or {}
        tempo = narrative.get('expected_tempo', 'Unknown')
        defense = narrative.get('defensive_stability', 'Unknown')
        
        st.metric("Overall Confidence", f"{confidence}%")
        st.metric("Risk Assessment", risk_level)
        st.metric("Expected Tempo", tempo.title())
        st.metric("Defensive Stability", defense.title())
    
    with col2:
        st.markdown('<div class="section-title">🎲 Additional Insights</div>', unsafe_allow_html=True)
        
        exact_scores = safe_get(predictions, 'probabilities', 'exact_scores') or {}
        most_likely_score = max(exact_scores, key=exact_scores.get) if exact_scores else "N/A"
        
        goal_timing = safe_get(predictions, 'probabilities', 'goal_timing') or {}
        first_half = goal_timing.get('first_half', 0)
        second_half = goal_timing.get('second_half', 0)
        
        st.metric("Most Likely Score", most_likely_score)
        st.metric("First Half Goal Prob", f"{first_half}%")
        st.metric("Second Half Goal Prob", f"{second_half}%")
        
        # System performance
        performance_monitor = PerformanceMonitor()
        performance_summary = performance_monitor.get_performance_summary()
        st.metric("System Status", performance_summary.get('health_status', 'HEALTHY'))
        st.metric("Recent Accuracy", f"{performance_summary.get('recent_accuracy', {}).get('1x2_accuracy', 0):.1f}%")

def main():
    """Main application function"""
    
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    
    if st.session_state.predictions:
        tab1, tab2, tab3 = st.tabs(["🎯 Predictions", "💰 Value Detection", "📈 Analytics"])
        
        with tab1:
            display_predictions(st.session_state.predictions)
        
        with tab2:
            display_value_detection(st.session_state.predictions)
        
        with tab3:
            display_analytics(st.session_state.predictions)
        
        st.markdown("---")
        if st.button("🔄 Analyze New Match", use_container_width=True):
            st.session_state.predictions = None
            st.rerun()
        
        return
    
    match_data = create_input_form()
    
    if match_data:
        with st.spinner("🧠 Running smart analysis with dynamic tier assessment..."):
            try:
                predictor = AdvancedFootballPredictor(match_data)
                predictions = predictor.generate_comprehensive_analysis()
                
                st.session_state.predictions = predictions
                
                # Check alignment status
                system_validation = safe_get(predictions, 'system_validation') or {}
                alignment_status = system_validation.get('alignment', 'UNKNOWN')
                
                if alignment_status == 'PERFECT':
                    st.success("✅ PERFECT SYSTEM ALIGNMENT! All engines synchronized with dynamic tier assessment!")
                else:
                    st.success("✅ Smart analysis completed successfully!")
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Analysis error: {str(e)}")
                st.info("💡 Check input parameters and try again")

if __name__ == "__main__":
    main()
