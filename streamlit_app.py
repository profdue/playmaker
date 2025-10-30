import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from prediction_engine import AdvancedPredictionEngine, BettingSignal, MonteCarloResults
import json
from typing import Dict, Any

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
    
    .confidence-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .confidence-high { background: #4CAF50; color: white; }
    .confidence-medium { background: #FF9800; color: white; }
    .confidence-low { background: #f44336; color: white; }
    
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .value-bet-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .value-exceptional { border-left-color: #4CAF50 !important; background: #f8fff8; }
    .value-high { border-left-color: #8BC34A !important; background: #f9fff9; }
    .value-good { border-left-color: #FFC107 !important; background: #fffdf6; }
    .value-moderate { border-left-color: #FF9800 !important; background: #fffaf2; }
    .value-low { border-left-color: #f44336 !important; background: #fff5f5; }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #f0f2f6;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
        margin: 10px 0;
    }
    .stat-item {
        background: #f8f9fa;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.9rem;
    }
    .stat-value {
        font-weight: bold;
        float: right;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .handicap-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'match_data' not in st.session_state:
    st.session_state.match_data = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'calibration_data' not in st.session_state:
    st.session_state.calibration_data = {}

def create_advanced_input_form():
    """Create comprehensive input form with all advanced options"""
    
    st.markdown('<p class="main-header">⚽ Advanced Football Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Professional Match Analysis with Monte Carlo Simulation & Value Detection</p>', unsafe_allow_html=True)
    
    # Use tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Basic Match Info", "📊 Advanced Statistics", "💰 Market Odds", "⚙️ Model Settings"])
    
    with tab1:
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
            st.subheader("Head to Head")
            h2h_col1, h2h_col2, h2h_col3 = st.columns(3)
            with h2h_col1:
                h2h_matches = st.number_input("Total H2H Matches", min_value=0, value=4, key="h2h_matches")
                h2h_home_wins = st.number_input("Home Wins", min_value=0, value=3, key="h2h_home_wins")
            with h2h_col2:
                h2h_away_wins = st.number_input("Away Wins", min_value=0, value=0, key="h2h_away_wins")
                h2h_draws = st.number_input("Draws", min_value=0, value=1, key="h2h_draws")
            with h2h_col3:
                h2h_home_goals = st.number_input("Home Goals in H2H", min_value=0, value=8, key="h2h_home_goals")
                h2h_away_goals = st.number_input("Away Goals in H2H", min_value=0, value=2, key="h2h_away_goals")
    
    with tab2:
        # League Table Context
        with st.expander("🏆 League Table Context"):
            st.subheader("Serie A Italy League Table")
            league_col1, league_col2 = st.columns(2)
            with league_col1:
                home_position = st.number_input(f"{home_team} Position", min_value=1, value=4, key="home_position")
                home_points = st.number_input(f"{home_team} Points", min_value=0, value=45, key="home_points")
            with league_col2:
                away_position = st.number_input(f"{away_team} Position", min_value=1, value=11, key="away_position")
                away_points = st.number_input(f"{away_team} Points", min_value=0, value=32, key="away_points")
        
        # Recent Form Sections
        with st.expander("📈 Recent Form Analysis"):
            st.subheader("Last 6 Matches Form")
            
            form_col1, form_col2 = st.columns(2)
            
            with form_col1:
                st.write(f"**{home_team} Last 6 Matches**")
                home_form = st.multiselect(
                    f"{home_team} Recent Results",
                    options=["Win (3 pts)", "Draw (1 pt)", "Loss (0 pts)"],
                    default=["Win (3 pts)", "Win (3 pts)", "Win (3 pts)", "Draw (1 pt)", "Loss (0 pts)", "Win (3 pts)"],
                    help="Select results from most recent to oldest",
                    key="home_form"
                )
                
            with form_col2:
                st.write(f"**{away_team} Last 6 Matches**")
                away_form = st.multiselect(
                    f"{away_team} Recent Results",
                    options=["Win (3 pts)", "Draw (1 pt)", "Loss (0 pts)"],
                    default=["Loss (0 pts)", "Win (3 pts)", "Draw (1 pt)", "Loss (0 pts)", "Win (3 pts)", "Draw (1 pt)"],
                    help="Select results from most recent to oldest",
                    key="away_form"
                )
        
        # Home/Away Specific Statistics
        with st.expander("🏠✈️ Home/Away Specific Statistics", expanded=True):
            st.subheader("Team-Specific Performance Metrics")
            
            home_away_col1, home_away_col2 = st.columns(2)
            
            with home_away_col1:
                st.write(f"**{home_team} Average Home Statistics**")
                home_goals_scored = st.number_input("Goals scored", min_value=0.0, value=2.3, key="home_goals_scored")
                home_goals_conceded = st.number_input("Goals conceded", min_value=0.0, value=0.3, key="home_goals_conceded")
                home_time_first_goal = st.number_input("Time first goal scored", min_value=1, value=52, key="home_time_first_goal")
                home_time_first_conceded = st.number_input("Time first goal conceded", min_value=1, value=63, key="home_time_first_conceded")
                home_yellow_cards = st.number_input("Yellow cards", min_value=0.0, value=1.7, key="home_yellow_cards")
                home_subs_used = st.number_input("Subs used", min_value=0, value=5, key="home_subs_used")
            
            with home_away_col2:
                st.write(f"**{away_team} Average Away Statistics**")
                away_goals_scored = st.number_input("Goals scored", min_value=0.0, value=1.3, key="away_goals_scored")
                away_goals_conceded = st.number_input("Goals conceded", min_value=0.0, value=2.5, key="away_goals_conceded")
                away_time_first_goal = st.number_input("Time first goal scored", min_value=1, value=42, key="away_time_first_goal")
                away_time_first_conceded = st.number_input("Time first goal conceded", min_value=1, value=26, key="away_time_first_conceded")
                away_yellow_cards = st.number_input("Yellow cards", min_value=0.0, value=2.3, key="away_yellow_cards")
                away_subs_used = st.number_input("Subs used", min_value=0, value=5, key="away_subs_used")
    
    with tab3:
        st.subheader("💰 Market Odds Input")
        st.info("Enter current bookmaker odds for value betting analysis")
        
        odds_col1, odds_col2, odds_col3 = st.columns(3)
        
        with odds_col1:
            st.write("**1X2 Market**")
            home_odds = st.number_input("Home Win Odds", min_value=1.01, value=1.85, step=0.01, key="home_odds")
            draw_odds = st.number_input("Draw Odds", min_value=1.01, value=3.40, step=0.01, key="draw_odds")
            away_odds = st.number_input("Away Win Odds", min_value=1.01, value=4.50, step=0.01, key="away_odds")
        
        with odds_col2:
            st.write("**Over/Under Markets**")
            over_15_odds = st.number_input("Over 1.5 Goals", min_value=1.01, value=1.25, step=0.01, key="over_15_odds")
            over_25_odds = st.number_input("Over 2.5 Goals", min_value=1.01, value=2.10, step=0.01, key="over_25_odds")
            over_35_odds = st.number_input("Over 3.5 Goals", min_value=1.01, value=3.50, step=0.01, key="over_35_odds")
        
        with odds_col3:
            st.write("**Both Teams to Score**")
            btts_yes_odds = st.number_input("BTTS Yes", min_value=1.01, value=1.95, step=0.01, key="btts_yes_odds")
            btts_no_odds = st.number_input("BTTS No", min_value=1.01, value=1.80, step=0.01, key="btts_no_odds")
            
            st.write("**Asian Handicap**")
            handicap_home_odds = st.number_input("Home -0.5", min_value=1.01, value=1.75, step=0.01, key="handicap_home_odds")
    
    with tab4:
        st.subheader("⚙️ Advanced Model Settings")
        
        model_col1, model_col2 = st.columns(2)
        
        with model_col1:
            league = st.selectbox("League", [
                "premier_league", "la_liga", "serie_a", "bundesliga", 
                "ligue_1", "default"
            ], index=2, key="league")
            
            st.write("**Injuries & Suspensions**")
            home_injuries = st.slider("Home Key Absences", 0, 5, 0, key="home_injuries")
            away_injuries = st.slider("Away Key Absences", 0, 5, 1, key="away_injuries")
            
        with model_col2:
            st.write("**Match Motivation**")
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
            
            # Monte Carlo Settings
            st.write("**Simulation Settings**")
            mc_iterations = st.select_slider(
                "Monte Carlo Iterations",
                options=[1000, 5000, 10000, 25000],
                value=10000,
                key="mc_iterations"
            )
        
        # Calibration toggle
        use_calibration = st.checkbox("Use Advanced Calibration", value=True, 
                                    help="Apply historical data-driven calibration parameters")
    
    # Submit button
    submitted = st.button("🎯 GENERATE ADVANCED PREDICTION", type="primary", use_container_width=True)
    
    if submitted:
        if not home_team or not away_team:
            st.error("❌ Please enter both team names")
            return None, None
        
        # Convert form selections to points
        form_map = {"Win (3 pts)": 3, "Draw (1 pt)": 1, "Loss (0 pts)": 0}
        home_form_points = [form_map[result] for result in home_form]
        away_form_points = [form_map[result] for result in away_form]
        
        # Convert motivation to multipliers
        motivation_map = {"Low": 0.8, "Normal": 1.0, "High": 1.15, "Very High": 1.3}
        
        # Home/Away statistics
        home_avg_stats = {
            'goals_scored': home_goals_scored,
            'goals_conceded': home_goals_conceded,
            'time_first_goal_scored': home_time_first_goal,
            'time_first_goal_conceded': home_time_first_conceded,
            'yellow_cards': home_yellow_cards,
            'subs_used': home_subs_used
        }
        
        away_avg_stats = {
            'goals_scored': away_goals_scored,
            'goals_conceded': away_goals_conceded,
            'time_first_goal_scored': away_time_first_goal,
            'time_first_goal_conceded': away_time_first_conceded,
            'yellow_cards': away_yellow_cards,
            'subs_used': away_subs_used
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
            'Asian Handicap Home -0.5': handicap_home_odds
        }
        
        # Calibration data
        calibration_data = {}
        if use_calibration:
            calibration_data = {
                'home_attack_weight': 1.05,
                'away_attack_weight': 0.95,
                'defense_weight': 0.92,
                'form_decay_rate': 0.85,
                'h2h_weight': 0.25,
                'injury_impact': 0.08,
                'motivation_impact': 0.12,
                'regression_strength': 0.2
            }
        
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
            'injuries': {'home': home_injuries, 'away': away_injuries},
            'motivation': {
                'home': motivation_map[home_motivation],
                'away': motivation_map[away_motivation]
            },
            'home_avg_stats': home_avg_stats,
            'away_avg_stats': away_avg_stats,
            'market_odds': market_odds,
            'league_context': {
                'home_position': home_position,
                'away_position': away_position,
                'home_points': home_points,
                'away_points': away_points
            }
        }
        
        st.session_state.match_data = match_data
        st.session_state.calibration_data = calibration_data
        st.session_state.mc_iterations = mc_iterations
        return match_data, calibration_data
    
    return None, None

def display_advanced_predictions(predictions):
    """Display comprehensive predictions with all enhanced features"""
    
    # Use tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Prediction Overview", "💰 Betting Signals", "📈 Advanced Analytics", "⚙️ Model Metrics"])
    
    with tab1:
        display_prediction_overview(predictions)
    
    with tab2:
        display_betting_signals(predictions)
    
    with tab3:
        display_advanced_analytics(predictions)
    
    with tab4:
        display_model_metrics(predictions)

def display_prediction_overview(predictions):
    """Display the main prediction overview (preserves original beautiful design)"""
    
    st.markdown('<p class="main-header">🎯 Advanced Match Prediction</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align: center; font-size: 1.4rem; font-weight: 600;">{predictions["match"]}</p>', unsafe_allow_html=True)
    
    # Expected Goals and Risk Assessment
    col1, col2, col3 = st.columns(3)
    
    with col1:
        xg = predictions['expected_goals']
        st.metric("🏠 Expected Goals (Home)", f"{xg['home']}")
    with col2:
        st.metric("✈️ Expected Goals (Away)", f"{xg['away']}")
    with col3:
        risk = predictions['risk_assessment']
        risk_class = f"risk-{risk['risk_level'].lower()}"
        st.markdown(f'''
        <div class="prediction-card {risk_class}">
            <h3>📊 Risk Assessment</h3>
            <strong>{risk["risk_level"]} RISK</strong><br>
            {risk["explanation"]}<br>
            Certainty: {risk["certainty"]}<br>
            Uncertainty: {risk.get('uncertainty_index', 'N/A')}
        </div>
        ''', unsafe_allow_html=True)
    
    # Match Outcomes
    st.markdown('<div class="section-title">📈 Match Outcome Probabilities</div>', unsafe_allow_html=True)
    
    outcomes = predictions['probabilities']['match_outcomes']
    col1, col2, col3 = st.columns(3)
    
    with col1:
        display_probability_bar("Home Win", outcomes['home_win'], "#4CAF50")
    with col2:
        display_probability_bar("Draw", outcomes['draw'], "#FF9800")
    with col3:
        display_probability_bar("Away Win", outcomes['away_win'], "#2196F3")
    
    # Goals Analysis
    st.markdown('<div class="section-title">⚽ Goals Analysis</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        timing = predictions['probabilities']['goal_timing']
        display_probability_card("First Half Goal", timing['first_half'])
    with col2:
        display_probability_card("Second Half Goal", timing['second_half'])
    with col3:
        btts = predictions['probabilities']['both_teams_score']
        display_probability_card("Both Teams Score", btts)
    with col4:
        over_25 = predictions['probabilities']['over_under']['over_2.5']
        display_probability_card("Over 2.5 Goals", over_25)
    
    # Exact Score Probabilities - FIXED SECTION
    st.markdown('<div class="section-title">🎯 Most Likely Scores</div>', unsafe_allow_html=True)
    
    exact_scores = predictions['probabilities']['exact_scores']
    
    # Take only the top 6 scores to fit our 6 columns
    top_scores = dict(list(exact_scores.items())[:6])
    score_cols = st.columns(6)

    for idx, (score, prob) in enumerate(top_scores.items()):
        with score_cols[idx]:
            st.metric(f"{score}", f"{prob}%")
    
    # Asian Handicap Probabilities
    st.markdown('<div class="section-title">🎲 Asian Handicap Probabilities</div>', unsafe_allow_html=True)
    
    handicap_probs = predictions.get('handicap_probabilities', {})
    if handicap_probs:
        handicap_cols = st.columns(4)
        common_handicaps = ['handicap_-0.5', 'handicap_0', 'handicap_0.5', 'handicap_1.0']
        
        for idx, handicap in enumerate(common_handicaps):
            if handicap in handicap_probs:
                with handicap_cols[idx]:
                    handicap_label = handicap.replace('handicap_', '').replace('_', '.')
                    st.markdown(f'''
                    <div class="handicap-card">
                        <h4>Handicap {handicap_label}</h4>
                        <span style="font-size: 1.5rem; font-weight: bold; color: #667eea;">
                            {handicap_probs[handicap]}%
                        </span>
                    </div>
                    ''', unsafe_allow_html=True)
    
    # Corner Predictions
    st.markdown('<div class="section-title">📊 Corner Predictions</div>', unsafe_allow_html=True)
    
    corners = predictions['corner_predictions']
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f'<div class="prediction-card"><h3>Total Corners</h3><span style="font-size: 1.8rem; font-weight: bold;">{corners["total"]}</span></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="prediction-card"><h3>🏠 Home Corners</h3><span style="font-size: 1.8rem; font-weight: bold;">{corners["home"]}</span></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="prediction-card"><h3>✈️ Away Corners</h3><span style="font-size: 1.8rem; font-weight: bold;">{corners["away"]}</span></div>', unsafe_allow_html=True)
    
    # Timing Predictions
    st.markdown('<div class="section-title">⏰ Match Timing Analysis</div>', unsafe_allow_html=True)
    
    timing = predictions['timing_predictions']
    st.markdown(f'''
    <div class="prediction-card">
        <h3>⏰ Key Timing Patterns</h3>
        • <strong>First Goal:</strong> {timing['first_goal']}<br>
        • <strong>Late Goals:</strong> {timing['late_goals']}<br>
        • <strong>Most Action:</strong> {timing['most_action']}
    </div>
    ''', unsafe_allow_html=True)
    
    # Summary and Confidence
    st.markdown('<div class="section-title">📝 Professional Summary</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info(predictions['summary'])
    
    with col2:
        st.metric("Overall Confidence Score", f"{predictions['confidence_score']}%")

def display_betting_signals(predictions):
    """Display betting signals and value detection"""
    
    st.markdown('<p class="main-header">💰 Value Betting Signals</p>', unsafe_allow_html=True)
    
    betting_signals = predictions.get('betting_signals', [])
    
    if not betting_signals:
        st.warning("No betting signals generated. Please check market odds input.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_signals = len(betting_signals)
        st.metric("Total Signals", total_signals)
    
    with col2:
        high_value_signals = len([s for s in betting_signals if s['value_rating'] in ['EXCEPTIONAL', 'HIGH']])
        st.metric("High Value Signals", high_value_signals)
    
    with col3:
        avg_edge = np.mean([s['edge'] for s in betting_signals])
        st.metric("Average Edge", f"{avg_edge:.1f}%")
    
    with col4:
        total_stake = np.sum([s['recommended_stake'] for s in betting_signals])
        st.metric("Total Recommended Stake", f"{total_stake:.1f}%")
    
    # Value bets by rating
    st.markdown('<div class="section-title">🎯 Value Bet Recommendations</div>', unsafe_allow_html=True)
    
    # Sort by value rating and edge
    exceptional_bets = [s for s in betting_signals if s['value_rating'] == 'EXCEPTIONAL']
    high_bets = [s for s in betting_signals if s['value_rating'] == 'HIGH']
    good_bets = [s for s in betting_signals if s['value_rating'] == 'GOOD']
    moderate_bets = [s for s in betting_signals if s['value_rating'] == 'MODERATE']
    
    def display_bet_group(bets, title, emoji):
        if bets:
            st.subheader(f"{emoji} {title} Value Bets")
            for bet in bets:
                value_class = f"value-{bet['value_rating'].lower()}"
                st.markdown(f'''
                <div class="value-bet-card {value_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>{bet['market']}</strong><br>
                            <small>Model: {bet['model_prob']}% | Market: {bet['book_prob']}%</small>
                        </div>
                        <div style="text-align: right;">
                            <strong style="color: #4CAF50;">+{bet['edge']}% Edge</strong><br>
                            <small>Stake: {bet['recommended_stake']*100:.1f}% | {bet['confidence']} Confidence</small>
                        </div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
    
    display_bet_group(exceptional_bets, "Exceptional", "🔥")
    display_bet_group(high_bets, "High", "⭐")
    display_bet_group(good_bets, "Good", "✅")
    display_bet_group(moderate_bets, "Moderate", "📊")
    
    # Edge distribution chart
    if betting_signals:
        st.markdown('<div class="section-title">📈 Edge Distribution</div>', unsafe_allow_html=True)
        
        df_edges = pd.DataFrame(betting_signals)
        fig = px.bar(df_edges, x='market', y='edge', color='value_rating',
                    title="Value Edge by Market",
                    color_discrete_map={
                        'EXCEPTIONAL': '#4CAF50',
                        'HIGH': '#8BC34A', 
                        'GOOD': '#FFC107',
                        'MODERATE': '#FF9800',
                        'LOW': '#f44336'
                    })
        fig.update_layout(xaxis_tickangle=-45, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

def display_advanced_analytics(predictions):
    """Display Monte Carlo results and advanced analytics"""
    
    st.markdown('<p class="main-header">📈 Advanced Analytics</p>', unsafe_allow_html=True)
    
    mc_results = predictions.get('monte_carlo_results', {})
    
    if not mc_results:
        st.warning("Monte Carlo results not available.")
        return
    
    # Confidence Intervals
    st.markdown('<div class="section-title">📊 Probability Confidence Intervals</div>', unsafe_allow_html=True)
    
    confidence_intervals = mc_results.get('confidence_intervals', {})
    
    if confidence_intervals:
        # Create confidence interval visualization
        markets = list(confidence_intervals.keys())
        lower_bounds = [ci[0] * 100 for ci in confidence_intervals.values()]
        upper_bounds = [ci[1] * 100 for ci in confidence_intervals.values()]
        means = [(lower + upper) / 2 for lower, upper in zip(lower_bounds, upper_bounds)]
        
        fig = go.Figure()
        
        # Add confidence intervals
        fig.add_trace(go.Scatter(
            x=markets,
            y=means,
            mode='markers',
            name='Mean Probability',
            marker=dict(size=10, color='#667eea')
        ))
        
        # Add error bars
        for i, market in enumerate(markets):
            fig.add_trace(go.Scatter(
                x=[market, market],
                y=[lower_bounds[i], upper_bounds[i]],
                mode='lines',
                line=dict(color='#667eea', width=2),
                showlegend=False
            ))
        
        fig.update_layout(
            title="95% Confidence Intervals for Key Probabilities",
            yaxis_title="Probability (%)",
            xaxis_tickangle=-45,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Probability Volatility
    st.markdown('<div class="section-title">⚡ Probability Volatility</div>', unsafe_allow_html=True)
    
    probability_volatility = mc_results.get('probability_volatility', {})
    
    if probability_volatility:
        volatility_df = pd.DataFrame({
            'Market': list(probability_volatility.keys()),
            'Volatility': [v * 100 for v in probability_volatility.values()]
        })
        
        fig = px.bar(volatility_df, x='Market', y='Volatility',
                    title="Probability Volatility Across Simulation Runs",
                    color='Volatility',
                    color_continuous_scale='Viridis')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Volatility interpretation
        avg_volatility = np.mean(list(probability_volatility.values())) * 100
        if avg_volatility < 2:
            volatility_rating = "Very Stable"
            color = "green"
        elif avg_volatility < 4:
            volatility_rating = "Stable"
            color = "blue"
        elif avg_volatility < 6:
            volatility_rating = "Moderate"
            color = "orange"
        else:
            volatility_rating = "High Volatility"
            color = "red"
        
        st.metric("Average Probability Volatility", f"{avg_volatility:.2f}%", volatility_rating)

def display_model_metrics(predictions):
    """Display model performance and technical metrics"""
    
    st.markdown('<p class="main-header">⚙️ Model Performance Metrics</p>', unsafe_allow_html=True)
    
    model_metrics = predictions.get('model_metrics', {})
    
    if not model_metrics:
        st.warning("Model metrics not available.")
        return
    
    # Key metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        entropy = model_metrics.get('shannon_entropy', 0)
        st.metric("Shannon Entropy", f"{entropy:.3f}")
    
    with col2:
        volatility = model_metrics.get('avg_probability_volatility', 0)
        st.metric("Avg Probability Volatility", f"{volatility:.4f}")
    
    with col3:
        ci_width = model_metrics.get('avg_confidence_interval_width', 0)
        st.metric("Avg CI Width", f"{ci_width:.3f}")
    
    with col4:
        iterations = model_metrics.get('monte_carlo_iterations', 0)
        st.metric("Monte Carlo Iterations", f"{iterations:,}")
    
    # Entropy explanation
    st.markdown('<div class="section-title">🧠 Uncertainty Analysis</div>', unsafe_allow_html=True)
    
    entropy = model_metrics.get('shannon_entropy', 0)
    
    if entropy < 0.7:
        entropy_interpretation = "Low Uncertainty - Clear favorite"
        entropy_color = "green"
    elif entropy < 1.0:
        entropy_interpretation = "Moderate Uncertainty - Competitive match"
        entropy_color = "orange"
    else:
        entropy_interpretation = "High Uncertainty - Very unpredictable"
        entropy_color = "red"
    
    st.markdown(f'''
    <div class="prediction-card">
        <h3>Information Theory Metrics</h3>
        <strong>Shannon Entropy:</strong> {entropy:.3f}<br>
        <strong>Interpretation:</strong> <span style="color: {entropy_color}">{entropy_interpretation}</span><br>
        <small>Lower entropy indicates more predictable outcomes</small>
    </div>
    ''', unsafe_allow_html=True)
    
    # Model configuration
    st.markdown('<div class="section-title">⚙️ Model Configuration</div>', unsafe_allow_html=True)
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.write("**Prediction Engine**")
        st.write("• Bayesian xG Calculation")
        st.write("• Monte Carlo Simulation")
        st.write("• Skellam Distribution")
        st.write("• Market Integration")
    
    with config_col2:
        st.write("**Advanced Features**")
        st.write("• Value Detection")
        st.write("• Kelly Criterion Staking")
        st.write("• Uncertainty Quantification")
        st.write("• Historical Calibration")

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

def display_probability_card(label: str, probability: float):
    """Display a probability in a card format"""
    confidence_class = "confidence-high" if probability > 70 else "confidence-medium" if probability > 55 else "confidence-low"
    
    st.markdown(f'''
    <div class="prediction-card">
        <h4>{label}</h4>
        <span class="confidence-badge {confidence_class}">{probability}% probability</span>
    </div>
    ''', unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Show predictions if available
    if st.session_state.predictions:
        display_advanced_predictions(st.session_state.predictions)
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🔄 Analyze Another Match", use_container_width=True):
                st.session_state.match_data = {}
                st.session_state.predictions = None
                st.rerun()
        
        with col2:
            if st.button("📊 Download Analysis Report", use_container_width=True):
                # Generate downloadable report
                predictions = st.session_state.predictions
                report_data = {
                    'match': predictions['match'],
                    'timestamp': str(pd.Timestamp.now()),
                    'expected_goals': predictions['expected_goals'],
                    'key_probabilities': predictions['probabilities']['match_outcomes'],
                    'betting_signals': predictions.get('betting_signals', []),
                    'confidence_score': predictions['confidence_score'],
                    'risk_assessment': predictions['risk_assessment']
                }
                
                st.download_button(
                    label="📥 Download JSON Report",
                    data=json.dumps(report_data, indent=2),
                    file_name=f"football_prediction_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with col3:
            if st.button("📈 View Raw Data", use_container_width=True):
                st.json(st.session_state.predictions)
        
        return
    
    # Input form
    match_data, calibration_data = create_advanced_input_form()
    
    if match_data:
        with st.spinner("🔍 Performing advanced match analysis with Monte Carlo simulation..."):
            try:
                # Initialize engine with calibration data
                engine = AdvancedPredictionEngine(match_data, calibration_data)
                
                # Set Monte Carlo iterations if specified
                if hasattr(st.session_state, 'mc_iterations'):
                    engine.monte_carlo_iterations = st.session_state.mc_iterations
                
                # Generate predictions
                predictions = engine.generate_advanced_predictions()
                
                st.session_state.predictions = predictions
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error generating predictions: {str(e)}")
                st.info("💡 Try adjusting the input parameters or check for invalid values")

if __name__ == "__main__":
    main()
