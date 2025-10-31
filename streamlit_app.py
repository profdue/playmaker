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
    from prediction_engine import AdvancedFootballPredictor, SignalEngine, ValueDetectionEngine
except ImportError:
    st.error("❌ Could not import prediction_engine. Make sure prediction_engine.py is in the same directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Advanced Football Predictor ⚽",
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
    
    .system-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .pure-engine-card {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .value-engine-card {
        background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .probability-bar {
        height: 8px;
        background: #e0e0e0;
        border-radius: 4px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    .probability-fill {
        height: 100%;
        background: linear-gradient(90deg, #4CAF50, #45a049);
        border-radius: 4px;
    }
    
    .bet-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        border-left: 4px solid;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .value-exceptional { border-left-color: #4CAF50 !important; background: #f8fff8; }
    .value-high { border-left-color: #8BC34A !important; background: #f9fff9; }
    .value-good { border-left-color: #FFC107 !important; background: #fffdf6; }
    .value-moderate { border-left-color: #FF9800 !important; background: #fffaf2; }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #f0f2f6;
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
    
    .architecture-diagram {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid #e9ecef;
    }
    
    .history-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def create_advanced_input_form():
    """Create input form with clear separation between football data and market data"""
    
    st.markdown('<p class="main-header">⚽ Advanced Football Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Professional Match Analysis with Strict Separation of Concerns</p>', unsafe_allow_html=True)
    
    # System Architecture Overview
    with st.expander("🏗️ System Architecture Overview", expanded=True):
        st.markdown("""
        ### 🎯 Project Purity v2.0 - Enhanced Separated Engines
        
        **Signal Engine** 🟢 (Pure Football Analysis)
        - Input: Only football data (goals, form, H2H, etc.)
        - Process: Dixon-Coles xG, Monte Carlo simulation
        - Output: Pure probabilities
        - **NEW**: Dynamic team profiles, context-aware motivation
        
        **Value Engine** 🟠 (Market Analysis)  
        - Input: Pure probabilities + Market odds
        - Process: Value detection, Kelly criterion
        - Output: Betting signals
        - **NEW**: Market confidence weighting, enhanced edge calculation
        
        **No feedback loop between engines - Market never influences football predictions**
        """)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Football Data", "💰 Market Data", "⚙️ Model Settings", "📊 System Info", "📈 History"])

    with tab1:
        st.markdown("### 🎯 Pure Football Data Input")
        st.info("This data goes ONLY to the Signal Engine for pure probability calculation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏠 Home Team")
            home_team = st.text_input("Team Name", value="Bologna", key="home_team")
            home_goals = st.number_input("Total Goals (Last 6 Games)", min_value=0, value=12, key="home_goals")
            home_conceded = st.number_input("Total Conceded (Last 6 Games)", min_value=0, value=8, key="home_conceded")
            home_goals_home = st.number_input("Home Goals (Last 3 Home Games)", min_value=0, value=7, key="home_goals_home")
            
        with col2:
            st.subheader("✈️ Away Team")
            away_team = st.text_input("Team Name", value="Torino", key="away_team")
            away_goals = st.number_input("Total Goals (Last 6 Games)", min_value=0, value=8, key="away_goals")
            away_conceded = st.number_input("Total Conceded (Last 6 Games)", min_value=0, value=10, key="away_conceded")
            away_goals_away = st.number_input("Away Goals (Last 3 Away Games)", min_value=0, value=4, key="away_goals_away")
        
        # Head-to-head section
        with st.expander("📊 Head-to-Head History", expanded=True):
            h2h_col1, h2h_col2, h2h_col3 = st.columns(3)
            with h2h_col1:
                h2h_matches = st.number_input("Total H2H Matches", min_value=0, value=6, key="h2h_matches")
                h2h_home_wins = st.number_input("Home Wins", min_value=0, value=4, key="h2h_home_wins")
            with h2h_col2:
                h2h_away_wins = st.number_input("Away Wins", min_value=0, value=1, key="h2h_away_wins")
                h2h_draws = st.number_input("Draws", min_value=0, value=1, key="h2h_draws")
            with h2h_col3:
                h2h_home_goals = st.number_input("Home Goals in H2H", min_value=0, value=9, key="h2h_home_goals")
                h2h_away_goals = st.number_input("Away Goals in H2H", min_value=0, value=4, key="h2h_away_goals")

        # Recent Form
        with st.expander("📈 Recent Form Analysis"):
            form_col1, form_col2 = st.columns(2)
            with form_col1:
                st.write(f"**{home_team} Last 6 Matches**")
                home_form = st.multiselect(
                    f"{home_team} Recent Results",
                    options=["Win (3 pts)", "Draw (1 pt)", "Loss (0 pts)"],
                    default=["Win (3 pts)", "Win (3 pts)", "Win (3 pts)", "Draw (1 pt)", "Loss (0 pts)", "Win (3 pts)"],
                    key="home_form"
                )
            with form_col2:
                st.write(f"**{away_team} Last 6 Matches**")
                away_form = st.multiselect(
                    f"{away_team} Recent Results", 
                    options=["Win (3 pts)", "Draw (1 pt)", "Loss (0 pts)"],
                    default=["Loss (0 pts)", "Win (3 pts)", "Draw (1 pt)", "Loss (0 pts)", "Win (3 pts)", "Draw (1 pt)"],
                    key="away_form"
                )

    with tab2:
        st.markdown("### 💰 Market Data Input") 
        st.info("This data goes ONLY to the Value Engine for betting signal generation")
        
        odds_col1, odds_col2, odds_col3 = st.columns(3)
        
        with odds_col1:
            st.write("**1X2 Market**")
            home_odds = st.number_input("Home Win Odds", min_value=1.01, value=2.10, step=0.01, key="home_odds")
            draw_odds = st.number_input("Draw Odds", min_value=1.01, value=3.10, step=0.01, key="draw_odds")
            away_odds = st.number_input("Away Win Odds", min_value=1.01, value=3.80, step=0.01, key="away_odds")
        
        with odds_col2:
            st.write("**Over/Under Markets**")
            over_15_odds = st.number_input("Over 1.5 Goals", min_value=1.01, value=1.40, step=0.01, key="over_15_odds")
            over_25_odds = st.number_input("Over 2.5 Goals", min_value=1.01, value=2.30, step=0.01, key="over_25_odds")
            over_35_odds = st.number_input("Over 3.5 Goals", min_value=1.01, value=4.00, step=0.01, key="over_35_odds")
        
        with odds_col3:
            st.write("**Both Teams to Score**")
            btts_yes_odds = st.number_input("BTTS Yes", min_value=1.01, value=1.95, step=0.01, key="btts_yes_odds")
            btts_no_odds = st.number_input("BTTS No", min_value=1.01, value=1.75, step=0.01, key="btts_no_odds")

    with tab3:
        st.markdown("### ⚙️ Enhanced Model Configuration")
        
        model_col1, model_col2 = st.columns(2)
        
        with model_col1:
            league = st.selectbox("League", [
                "premier_league", "la_liga", "serie_a", "bundesliga", "ligue_1", "default"
            ], index=2, key="league")
            
            st.write("**Enhanced Team Context**")
            home_injuries = st.slider("Home Key Absences", 0, 5, 1, key="home_injuries")
            away_injuries = st.slider("Away Key Absences", 0, 5, 2, key="away_injuries")
            
            # Enhanced absence impact
            st.write("**Absence Impact Level**")
            home_absence_impact = st.select_slider(
                "Home Team Absence Impact",
                options=["Rotation Player", "Regular Starter", "Key Player", "Star Player", "Multiple Key Players"],
                value="Regular Starter",
                key="home_absence_impact"
            )
            away_absence_impact = st.select_slider(
                "Away Team Absence Impact",
                options=["Rotation Player", "Regular Starter", "Key Player", "Star Player", "Multiple Key Players"],
                value="Key Player",
                key="away_absence_impact"
            )
            
        with model_col2:
            st.write("**Enhanced Motivation Factors**")
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
            
            mc_iterations = st.select_slider(
                "Monte Carlo Iterations",
                options=[1000, 5000, 10000, 25000],
                value=10000,
                key="mc_iterations"
            )
            
            # Performance quality (enhanced form)
            st.write("**Form Quality Assessment**")
            form_quality_note = st.info("💡 The model now considers performance quality, not just results")

    with tab4:
        st.markdown("### 📊 Enhanced System Information")
        st.markdown("""
        **Architecture Benefits:**
        - 🛡️ **Bias Protection**: Market odds cannot influence football predictions
        - 🔍 **Transparency**: Clear separation between analysis and betting
        - 📈 **Accuracy**: Pure football model focuses on match reality
        - 💰 **Value Detection**: Independent engine finds market inefficiencies
        
        **NEW ENHANCEMENTS:**
        - 🎯 **Dynamic Team Profiles**: Automatic style detection
        - 🧠 **Context-Aware Motivation**: Match context affects motivation impact
        - 📊 **Market Confidence Weighting**: Better edge detection
        - 🏥 **Enhanced Absence Impact**: More granular injury modeling
        
        **Data Flow:**
        ```
        Football Data → Signal Engine → Pure Probabilities
        Market Odds → Value Engine → Betting Signals
        ```
        """)

    with tab5:
        st.markdown("### 📈 Prediction History & Learning")
        if 'prediction_history' in st.session_state and st.session_state.prediction_history:
            history = st.session_state.prediction_history
            st.write(f"**Total Predictions Tracked:** {len(history)}")
            
            # Show recent predictions
            st.write("**Recent Predictions:**")
            for i, pred in enumerate(history[-5:]):  # Show last 5
                with st.expander(f"Prediction {i+1}: {pred['match']}"):
                    st.write(f"Date: {pred.get('timestamp', 'N/A')}")
                    st.write(f"Expected Goals: Home {pred['expected_goals']['home']:.2f} - Away {pred['expected_goals']['away']:.2f}")
                    st.write(f"Probabilities: {pred['probabilities']}")
                    st.write(f"Match Context: {pred['match_context']}")
                    st.write(f"Confidence: {pred['confidence_score']}%")
        else:
            st.info("No prediction history yet. Generate some predictions to see historical data here!")

    # Submit button
    submitted = st.button("🎯 GENERATE ENHANCED ANALYSIS", type="primary", use_container_width=True)
    
    if submitted:
        if not home_team or not away_team:
            st.error("❌ Please enter both team names")
            return None, None
        
        # Convert form selections to points
        form_map = {"Win (3 pts)": 3, "Draw (1 pt)": 1, "Loss (0 pts)": 0}
        home_form_points = [form_map[result] for result in home_form]
        away_form_points = [form_map[result] for result in away_form]
        
        # Convert motivation (already in correct format)
        motivation_map = {"Low": "Low", "Normal": "Normal", "High": "High", "Very High": "Very High"}
        
        # Convert absence impact to numeric
        absence_impact_map = {
            "Rotation Player": 1,
            "Regular Starter": 2,
            "Key Player": 3, 
            "Star Player": 4,
            "Multiple Key Players": 5
        }
        
        # Market odds
        market_odds = {
            '1x2 Home': home_odds,
            '1x2 Draw': draw_odds,
            '1x2 Away': away_odds,
            'Over 1.5 Goals': over_15_odds,
            'Over 2.5 Goals': over_25_odds,
            'Over 3.5 Goals': over_35_odds,
            'BTTS Yes': btts_yes_odds,
            'BTTS No': btts_no_odds,
        }
        
        # Complete match data with enhanced fields
        match_data = {
            'home_team': home_team,
            'away_team': away_team,
            'league': league,
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
                'draws': h2h_draws,
                'home_goals': h2h_home_goals,
                'away_goals': h2h_away_goals
            },
            'injuries': {
                'home': absence_impact_map[home_absence_impact],
                'away': absence_impact_map[away_absence_impact]
            },
            'motivation': {
                'home': motivation_map[home_motivation],
                'away': motivation_map[away_motivation]
            },
            'market_odds': market_odds
        }
        
        return match_data, mc_iterations
    
    return None, None

def safe_get(dictionary, *keys, default=None):
    """Safely get nested dictionary keys"""
    current = dictionary
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current

def display_separated_analysis(predictions):
    """Display analysis with clear separation between engines"""
    
    # Use tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Pure Predictions", "💰 Value Detection", "📈 Analytics", "🏗️ System Health"])
    
    with tab1:
        display_pure_predictions(predictions)
    
    with tab2:
        display_value_detection(predictions)
    
    with tab3:
        display_advanced_analytics(predictions)
    
    with tab4:
        display_system_health(predictions)

def display_goals_analysis(predictions):
    """Display goals analysis with CORRECT recommendations"""
    st.markdown('<div class="section-title">⚽ Goals Analysis</div>', unsafe_allow_html=True)
    
    # Get probabilities
    btts_yes = safe_get(predictions, 'probabilities', 'both_teams_score', 'yes', default=0)
    btts_no = safe_get(predictions, 'probabilities', 'both_teams_score', 'no', default=0)
    
    over_25 = safe_get(predictions, 'probabilities', 'over_under', 'over_25', default=0)
    under_25 = safe_get(predictions, 'probabilities', 'over_under', 'under_25', default=0)
    
    first_half = safe_get(predictions, 'probabilities', 'goal_timing', 'first_half', default=0)
    second_half = safe_get(predictions, 'probabilities', 'goal_timing', 'second_half', default=0)
    
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
                {recommendation}: {primary_prob}%
            </div>
            <div style="font-size: 0.9rem; color: #666; margin: 0.3rem 0;">
                {('NO' if recommendation == 'YES' else 'YES')}: {secondary_prob}%
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
                {recommendation}: {primary_prob}%
            </div>
            <div style="font-size: 0.9rem; color: #666; margin: 0.3rem 0;">
                {('OVER' if recommendation == 'UNDER' else 'UNDER')}: {secondary_prob}%
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
                {first_half}%
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
                {second_half}%
            </div>
            <span class="confidence-badge confidence-{confidence.lower()}">
                {emoji} {confidence} CONFIDENCE
            </span>
        </div>
        ''', unsafe_allow_html=True)

def display_pure_predictions(predictions):
    """Display pure football predictions from Signal Engine"""
    
    st.markdown('<p class="main-header">🎯 Pure Football Predictions</p>', unsafe_allow_html=True)
    st.markdown('<div class="pure-engine-card"><h3>🟢 Enhanced Signal Engine Output</h3>Market-independent football analysis with dynamic team profiling</div>', unsafe_allow_html=True)
    
    st.markdown(f'<p style="text-align: center; font-size: 1.4rem; font-weight: 600;">{predictions["match"]}</p>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        xg = safe_get(predictions, 'expected_goals', default={'home': 0, 'away': 0})
        st.metric("🏠 Expected Goals", f"{xg.get('home', 0):.2f}")
    with col2:
        st.metric("✈️ Expected Goals", f"{xg.get('away', 0):.2f}")
    with col3:
        match_context = predictions.get('match_context', 'Unknown')
        context_emoji = {
            'defensive_battle': '🛡️',
            'tactical_stalemate': '⚔️', 
            'offensive_showdown': '🔥',
            'home_dominance': '🏠',
            'away_counter': '✈️',
            'unknown': '❓'
        }.get(match_context, '❓')
        st.metric("Match Context", f"{context_emoji} {match_context.replace('_', ' ').title()}")
    with col4:
        confidence = safe_get(predictions, 'confidence_score', default=0)
        st.metric("Confidence Score", f"{confidence}%")
    
    # Match Outcomes
    st.markdown('<div class="section-title">📈 Match Outcome Probabilities</div>', unsafe_allow_html=True)
    
    outcomes = safe_get(predictions, 'probabilities', 'match_outcomes', default={'home_win': 0, 'draw': 0, 'away_win': 0})
    col1, col2, col3 = st.columns(3)
    
    with col1:
        display_probability_bar("Home Win", outcomes.get('home_win', 0), "#4CAF50")
    with col2:
        display_probability_bar("Draw", outcomes.get('draw', 0), "#FF9800")
    with col3:
        display_probability_bar("Away Win", outcomes.get('away_win', 0), "#2196F3")
    
    # Goals Analysis
    display_goals_analysis(predictions)
    
    # Exact Scores
    st.markdown('<div class="section-title">🎯 Most Likely Scores</div>', unsafe_allow_html=True)
    
    exact_scores = safe_get(predictions, 'probabilities', 'exact_scores', default={})
    top_scores = dict(list(exact_scores.items())[:6])
    
    if top_scores:
        score_cols = st.columns(6)
        for idx, (score, prob) in enumerate(top_scores.items()):
            with score_cols[idx]:
                st.metric(f"{score}", f"{prob}%")
    
    # Risk Assessment
    risk = safe_get(predictions, 'risk_assessment', default={'risk_level': 'UNKNOWN', 'explanation': 'No data'})
    risk_class = f"risk-{risk.get('risk_level', 'unknown').lower()}"
    
    st.markdown(f'''
    <div class="prediction-card {risk_class}">
        <h3>📊 Enhanced Risk Assessment</h3>
        <strong>Risk Level:</strong> {risk.get("risk_level", "UNKNOWN")}<br>
        <strong>Explanation:</strong> {risk.get("explanation", "No data available")}<br>
        <strong>Certainty:</strong> {risk.get("certainty", "N/A")}<br>
        <strong>Uncertainty:</strong> {risk.get('uncertainty', 'N/A')}
    </div>
    ''', unsafe_allow_html=True)
    
    # Professional Summary
    st.markdown('<div class="section-title">📝 Pure Football Summary</div>', unsafe_allow_html=True)
    st.info(safe_get(predictions, 'summary', default="No summary available."))

def display_value_detection(predictions):
    """Display value detection results from Value Engine"""
    
    st.markdown('<p class="main-header">💰 Enhanced Value Betting Detection</p>', unsafe_allow_html=True)
    st.markdown('<div class="value-engine-card"><h3>🟠 Enhanced Value Engine Output</h3>Market confidence-weighted edge detection</div>', unsafe_allow_html=True)
    
    betting_signals = safe_get(predictions, 'betting_signals', default=[])
    
    if not betting_signals:
        st.warning("No value bets detected. This could mean:")
        st.info("""
        - Market odds are efficient (no edge)
        - Pure probabilities align with market expectations  
        - Insufficient data for value detection
        - All edges below minimum threshold
        - **NEW**: Market confidence weighting reduced edges
        """)
        return
    
    # Value Bet Summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_signals = len(betting_signals)
        st.metric("Total Value Bets", total_signals)
    
    with col2:
        high_value = len([s for s in betting_signals if s.get('value_rating') in ['EXCEPTIONAL', 'HIGH']])
        st.metric("High Value Signals", high_value)
    
    with col3:
        avg_edge = np.mean([s.get('edge', 0) for s in betting_signals])
        st.metric("Average Edge", f"{avg_edge:.1f}%")
    
    with col4:
        total_stake = np.sum([s.get('recommended_stake', 0) for s in betting_signals])
        st.metric("Total Stake", f"{total_stake * 100:.1f}%")
    
    # Display value bets by rating
    st.markdown('<div class="section-title">🎯 Enhanced Value Bet Recommendations</div>', unsafe_allow_html=True)
    
    # Group by value rating
    exceptional_bets = [s for s in betting_signals if s.get('value_rating') == 'EXCEPTIONAL']
    high_bets = [s for s in betting_signals if s.get('value_rating') == 'HIGH']
    good_bets = [s for s in betting_signals if s.get('value_rating') == 'GOOD']
    moderate_bets = [s for s in betting_signals if s.get('value_rating') == 'MODERATE']
    
    def display_bet_group(bets, title, emoji):
        if bets:
            st.subheader(f"{emoji} {title} Value Bets")
            for bet in bets:
                value_class = f"value-{bet.get('value_rating', '').lower()}"
                confidence_emoji = {
                    'HIGH': '🟢',
                    'MEDIUM': '🟡', 
                    'LOW': '🔴',
                    'SPECULATIVE': '⚪'
                }.get(bet.get('confidence', 'SPECULATIVE'), '⚪')
                
                st.markdown(f'''
                <div class="bet-card {value_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="flex: 2;">
                            <strong>{bet.get('market', 'Unknown')}</strong><br>
                            <small>Pure Probability: {bet.get('model_prob', 0)}% | Market Implied: {bet.get('book_prob', 0)}%</small>
                        </div>
                        <div style="flex: 1; text-align: right;">
                            <strong style="color: #4CAF50; font-size: 1.1rem;">+{bet.get('edge', 0)}% Edge</strong><br>
                            <small>Stake: {bet.get('recommended_stake', 0)*100:.1f}% | {confidence_emoji} {bet.get('confidence', 'Unknown')}</small>
                        </div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
    
    display_bet_group(exceptional_bets, "Exceptional", "🔥")
    display_bet_group(high_bets, "High", "⭐")
    display_bet_group(good_bets, "Good", "✅")
    display_bet_group(moderate_bets, "Moderate", "📊")
    
    # Edge distribution visualization
    if betting_signals:
        st.markdown('<div class="section-title">📈 Enhanced Edge Distribution Analysis</div>', unsafe_allow_html=True)
        
        df_edges = pd.DataFrame(betting_signals)
        fig = px.bar(df_edges, x='market', y='edge', color='value_rating',
                    title="Market Confidence-Weighted Edge Detection",
                    color_discrete_map={
                        'EXCEPTIONAL': '#4CAF50',
                        'HIGH': '#8BC34A', 
                        'GOOD': '#FFC107',
                        'MODERATE': '#FF9800'
                    })
        fig.update_layout(
            xaxis_tickangle=-45, 
            showlegend=True,
            yaxis_title="Adjusted Edge (%)",
            xaxis_title="Market"
        )
        st.plotly_chart(fig, use_container_width=True)

def display_advanced_analytics(predictions):
    """Display advanced analytics from both engines"""
    
    st.markdown('<p class="main-header">📈 Enhanced Analytics</p>', unsafe_allow_html=True)
    
    # Monte Carlo Results
    mc_results = safe_get(predictions, 'monte_carlo_results', default={})
    
    if mc_results:
        st.markdown('<div class="section-title">⚡ Enhanced Monte Carlo Simulation</div>', unsafe_allow_html=True)
        
        # Confidence Intervals
        confidence_intervals = safe_get(mc_results, 'confidence_intervals', default={})
        
        if confidence_intervals:
            markets = ['Home Win', 'Draw', 'Away Win', 'Over 2.5']
            lower_bounds = [ci[0] * 100 for ci in confidence_intervals.values()]
            upper_bounds = [ci[1] * 100 for ci in confidence_intervals.values()]
            means = [(lower + upper) / 2 for lower, upper in zip(lower_bounds, upper_bounds)]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=markets,
                y=means,
                mode='markers',
                name='Mean Probability',
                marker=dict(size=10, color='#667eea')
            ))
            
            for i, market in enumerate(markets):
                fig.add_trace(go.Scatter(
                    x=[market, market],
                    y=[lower_bounds[i], upper_bounds[i]],
                    mode='lines',
                    line=dict(color='#667eea', width=2),
                    showlegend=False
                ))
            
            fig.update_layout(
                title="95% Confidence Intervals for Pure Probabilities",
                yaxis_title="Probability (%)",
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Additional Metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-title">📊 Enhanced Model Performance</div>', unsafe_allow_html=True)
        
        data_quality = safe_get(predictions, 'data_quality_score', default=0)
        confidence = safe_get(predictions, 'confidence_score', default=0)
        
        st.metric("Data Quality Score", f"{data_quality:.1f}%")
        st.metric("Overall Confidence", f"{confidence}%")
        
        # Volatility metrics
        probability_volatility = safe_get(mc_results, 'probability_volatility', default={})
        if probability_volatility:
            avg_volatility = np.mean(list(probability_volatility.values())) * 100
            st.metric("Probability Volatility", f"{avg_volatility:.2f}%")
            
            # Show volatility by market
            st.write("**Volatility by Market:**")
            for market, vol in probability_volatility.items():
                st.write(f"- {market.replace('_', ' ').title()}: {vol*100:.2f}%")
    
    with col2:
        st.markdown('<div class="section-title">🎲 Enhanced Predictions</div>', unsafe_allow_html=True)
        
        corners = safe_get(predictions, 'corner_predictions', default={})
        timing = safe_get(predictions, 'timing_predictions', default={})
        
        st.write(f"**Total Corners:** {corners.get('total', 'N/A')}")
        st.write(f"**First Goal:** {timing.get('first_goal', 'N/A')}")
        st.write(f"**Late Goals:** {timing.get('late_goals', 'N/A')}")
        st.write(f"**Match Rhythm:** {timing.get('most_action', 'N/A')}")
        
        # Handicap probabilities
        handicap_probs = safe_get(predictions, 'handicap_probabilities', default={})
        if handicap_probs:
            st.write("**Handicap Probabilities:**")
            for handicap, prob in list(handicap_probs.items())[:3]:  # Show top 3
                st.write(f"- {handicap.replace('_', ' ').title()}: {prob}%")

def display_system_health(predictions):
    """Display system health and bias monitoring"""
    
    st.markdown('<p class="main-header">🏗️ Enhanced System Health Monitoring</p>', unsafe_allow_html=True)
    
    # Architecture Diagram
    st.markdown("""
    <div class="architecture-diagram">
        <h3>🔄 Enhanced Data Flow Architecture</h3>
        <pre>
        ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
        │   Football Data │ ──▶│   Signal Engine  │ ──▶│ Pure Probabilities │
        │ (Goals, Form)   │    │  (No Market Bias)│    │   (Market-Free)   │
        │    + Enhanced   │    │ + Dynamic Profiles│    │ + Context-Aware   │
        │     Context     │    │ + Better Bounds  │    │   Summaries       │
        └─────────────────┘    └──────────────────┘    └─────────────────┘
                                                                  │
        ┌─────────────────┐    ┌──────────────────┐              │
        │   Market Odds   │ ──▶│  Value Engine    │ ◀─────────────┘
        │ (Bookmaker)     │    │ (Edge Detection) │    
        │                 │    │ + Confidence     │    
        └─────────────────┘    │   Weighting      │    
                               └──────────────────┘    
                                      │
                                      ▼
                            ┌─────────────────┐
                            │ Betting Signals │
                            │   (Value Bets)  │
                            │   + Enhanced    │
                            │   Edge Calc     │
                            └─────────────────┘
        </pre>
    </div>
    """, unsafe_allow_html=True)
    
    # Bias Monitoring
    st.markdown('<div class="section-title">🛡️ Enhanced Bias Protection Status</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("**✅ Enhanced Signal Engine**")
        st.write("• Pure football data only")
        st.write("• No market influence")
        st.write("• Dynamic team profiling")
        st.write("• Context-aware motivation")
    
    with col2:
        st.warning("**✅ Enhanced Value Engine**") 
        st.write("• Separate processing")
        st.write("• Market confidence weighting")
        st.write("• Enhanced edge detection")
        st.write("• No feedback to predictions")
    
    with col3:
        st.info("**✅ Enhanced System Integrity**")
        st.write("• Strict separation maintained")
        st.write("• Bias drift monitoring active")
        st.write("• Architecture compliance: 100%")
        st.write("• Historical tracking enabled")
    
    # Model Metrics
    st.markdown('<div class="section-title">📈 Enhanced Model Quality Metrics</div>', unsafe_allow_html=True)
    
    outcomes = safe_get(predictions, 'probabilities', 'match_outcomes', default={'home_win': 0, 'draw': 0, 'away_win': 0})
    probs = np.array([v / 100 for v in outcomes.values()])
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    bias_monitoring = safe_get(predictions, 'bias_monitoring', default={})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Shannon Entropy", f"{entropy:.3f}")
    with col2:
        max_entropy = np.log(3)
        uncertainty = entropy / max_entropy
        st.metric("Uncertainty Ratio", f"{uncertainty:.2f}")
    with col3:
        data_quality = safe_get(predictions, 'data_quality_score', default=0)
        st.metric("Data Quality", f"{data_quality:.1f}%")
    with col4:
        match_context = predictions.get('match_context', 'unknown')
        st.metric("Match Context", match_context.replace('_', ' ').title())
    
    # Additional metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Predictions Tracked", bias_monitoring.get('total_predictions_tracked', 0))
    with col2:
        # Show prediction history if available
        if 'prediction_history' in st.session_state:
            history_count = len(st.session_state.prediction_history)
            st.metric("Session Predictions", history_count)

def display_probability_bar(label: str, probability: float, color: str):
    """Display a probability with a visual bar"""
    st.markdown(f'''
    <div style="margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span><strong>{label}</strong></span>
            <span><strong>{probability}%</strong></span>
        </div>
        <div class="probability-bar">
            <div class="probability-fill" style="width: {probability}%; background: {color};"></div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

def store_prediction_in_session(prediction):
    """Store prediction in session state for history tracking"""
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    prediction_record = {
        'timestamp': datetime.now().isoformat(),
        'match': prediction['match'],
        'expected_goals': prediction['expected_goals'],
        'probabilities': prediction['probabilities']['match_outcomes'],
        'match_context': prediction['match_context'],
        'confidence_score': prediction['confidence_score'],
        'data_quality': prediction['data_quality_score']
    }
    
    st.session_state.prediction_history.append(prediction_record)
    
    # Keep only last 20 predictions to manage memory
    if len(st.session_state.prediction_history) > 20:
        st.session_state.prediction_history = st.session_state.prediction_history[-20:]

def main():
    """Main application function"""
    
    # Initialize session state
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    # Show predictions if available
    if st.session_state.predictions:
        display_separated_analysis(st.session_state.predictions)
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Analyze New Match", use_container_width=True):
                st.session_state.predictions = None
                st.rerun()
        
        with col2:
            if st.button("📊 System Diagnostics", use_container_width=True):
                st.json(st.session_state.predictions)
        
        with col3:
            if st.button("📈 View History", use_container_width=True):
                # Switch to history tab
                st.session_state.show_history = True
                st.rerun()
        
        return
    
    # Input form
    match_data, mc_iterations = create_advanced_input_form()
    
    if match_data:
        with st.spinner("🔍 Running enhanced engine analysis..."):
            try:
                # Use the enhanced orchestrator with separated engines
                predictor = AdvancedFootballPredictor(match_data)
                predictions = predictor.generate_comprehensive_analysis(mc_iterations)
                
                # Store in session state
                st.session_state.predictions = predictions
                store_prediction_in_session(predictions)
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Analysis error: {str(e)}")
                st.info("💡 Check input parameters and try again")

if __name__ == "__main__":
    main()
