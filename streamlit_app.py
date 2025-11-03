# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from typing import Dict, Any
from datetime import datetime

# Import the PREDICTION ENGINE
try:
    from prediction_engine import AdvancedFootballPredictor, TeamTierCalibrator
except ImportError as e:
    st.error(f"❌ Could not import prediction_engine: {str(e)}")
    st.info("💡 Make sure prediction_engine.py is in the same directory")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🌍 Advanced Football Predictor",
    page_icon="⚽", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling with league badges
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
    .league-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        color: white;
        display: inline-block;
        margin: 0.2rem;
    }
    .premier-league { background: #3D195B; }
    .la-liga { background: #FF0000; }
    .serie-a { background: #008C45; }
    .bundesliga { background: #DC052D; }
    .ligue-1 { background: #DA291C; }
    .liga-portugal { background: #006600; }
    .brasileirao { background: #FFCC00; color: black; }
    .liga-mx { background: #006847; }
    .eredivisie { background: #FF6B00; }
    
    .betting-activity-rank {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
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
    
    .success-banner {
        background: #f8fff8;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-banner {
        background: #fffaf2;
        border-left: 4px solid #FF9800;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .danger-banner {
        background: #fff5f5;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
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
    
    .alignment-danger {
        background: #fff5f5;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
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
        'ligue_1': 'Ligue 1 🇫🇷',
        'liga_portugal': 'Liga Portugal 🇵🇹',
        'brasileirao': 'Brasileirão 🇧🇷',
        'liga_mx': 'Liga MX 🇲🇽',
        'eredivisie': 'Eredivisie 🇳🇱'
    }
    return league_names.get(league_id, league_id)

def get_league_badge(league_id: str) -> str:
    """Get CSS class for league badge"""
    league_classes = {
        'premier_league': 'premier-league',
        'la_liga': 'la-liga',
        'serie_a': 'serie-a', 
        'bundesliga': 'bundesliga',
        'ligue_1': 'ligue-1',
        'liga_portugal': 'liga-portugal',
        'brasileirao': 'brasileirao',
        'liga_mx': 'liga-mx',
        'eredivisie': 'eredivisie'
    }
    return league_classes.get(league_id, 'premier-league')

def display_betting_activity_ranking():
    """Display betting activity ranking by league"""
    st.markdown('<div class="betting-activity-rank">', unsafe_allow_html=True)
    st.markdown("### 📊 Betting Activity Ranking")
    
    # Professional betting activity ranking (based on liquidity, market depth, etc.)
    ranking_data = {
        'League': ['Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1', 'Brasileirão', 'Liga MX', 'Eredivisie'],
        'Betting Activity': ['Very High', 'High', 'High', 'High', 'Medium', 'Medium', 'Medium', 'Medium'],
        'Market Depth': ['Excellent', 'Excellent', 'Very Good', 'Very Good', 'Good', 'Good', 'Good', 'Good'],
        'Liquidity': ['★★★★★', '★★★★★', '★★★★☆', '★★★★☆', '★★★☆☆', '★★★☆☆', '★★★☆☆', '★★★☆☆']
    }
    
    df = pd.DataFrame(ranking_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

def create_input_form():
    """Create input form with multi-league support"""
    
    st.markdown('<p class="main-header">🌍 Advanced Football Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Professional Multi-League Analysis with Tier-Based Calibration</p>', unsafe_allow_html=True)
    
    # Display betting activity ranking
    display_betting_activity_ranking()
    
    # System Architecture Overview
    with st.expander("🏗️ System Architecture", expanded=True):
        st.markdown("""
        ### 🎯 PROFESSIONAL MULTI-LEAGUE PREDICTOR
        
        **Supported Leagues** 🌍
        - **Premier League** 🏴󠁧󠁢󠁥󠁮󠁧󠁿, **La Liga** 🇪🇸, **Serie A** 🇮🇹
        - **Bundesliga** 🇩🇪, **Ligue 1** 🇫🇷, **Liga Portugal** 🇵🇹  
        - **Brasileirão** 🇧🇷, **Liga MX** 🇲🇽, **Eredivisie** 🇳🇱
        
        **League-Specific Calibration** ⚡
        - Different scoring profiles per league
        - League-specific home advantage
        - Tier-based team strength systems
        - Contextual probability adjustments
        """)
    
    # League Selection
    st.markdown("### 🌍 League Selection")
    league_options = {
        'premier_league': 'Premier League 🏴󠁧󠁢󠁥󠁮󠁧󠁿',
        'la_liga': 'La Liga 🇪🇸',
        'serie_a': 'Serie A 🇮🇹', 
        'bundesliga': 'Bundesliga 🇩🇪',
        'ligue_1': 'Ligue 1 🇫🇷',
        'liga_portugal': 'Liga Portugal 🇵🇹',
        'brasileirao': 'Brasileirão 🇧🇷',
        'liga_mx': 'Liga MX 🇲🇽',
        'eredivisie': 'Eredivisie 🇳🇱'
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
    
    tab1, tab2, tab3 = st.tabs(["🏠 Football Data", "💰 Market Data", "⚙️ Settings"])

    with tab1:
        st.markdown("### 🎯 Football Data Input")
        
        # Initialize team calibrator
        calibrator = TeamTierCalibrator()
        league_teams = calibrator.get_league_teams(selected_league)
        
        if not league_teams:
            st.error(f"❌ No teams found for {league_display_name}")
            return None, None
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏠 Home Team")
            home_team = st.selectbox(
                "Team Name", 
                options=league_teams,
                index=min(5, len(league_teams) - 1),  # Default to a mid-table team
                key="home_team"
            )
            
            home_goals = st.number_input("Total Goals (Last 6 Games)", min_value=0, value=8, key="home_goals")
            home_conceded = st.number_input("Total Conceded (Last 6 Games)", min_value=0, value=14, key="home_conceded")
            home_goals_home = st.number_input("Home Goals (Last 3 Home Games)", min_value=0, value=5, key="home_goals_home")
            
        with col2:
            st.subheader("✈️ Away Team")
            away_team = st.selectbox(
                "Team Name",
                options=league_teams,
                index=0,  # Default to top team
                key="away_team"
            )
            
            away_goals = st.number_input("Total Goals (Last 6 Games)", min_value=0, value=12, key="away_goals")
            away_conceded = st.number_input("Total Conceded (Last 6 Games)", min_value=0, value=6, key="away_conceded")
            away_goals_away = st.number_input("Away Goals (Last 3 Away Games)", min_value=0, value=7, key="away_goals_away")
        
        # Show team tiers
        home_tier = calibrator.get_team_tier(home_team, selected_league)
        away_tier = calibrator.get_team_tier(away_team, selected_league)
        
        st.markdown(f"""
        **Team Tiers:** 
        <span class="tier-badge tier-{home_tier.lower()}">{home_tier}</span> vs 
        <span class="tier-badge tier-{away_tier.lower()}">{away_tier}</span>
        """, unsafe_allow_html=True)
        
        # Head-to-head section
        with st.expander("📊 Head-to-Head History"):
            h2h_col1, h2h_col2, h2h_col3 = st.columns(3)
            with h2h_col1:
                h2h_matches = st.number_input("Total H2H Matches", min_value=0, value=5, key="h2h_matches")
                h2h_home_wins = st.number_input("Home Wins", min_value=0, value=1, key="h2h_home_wins")
            with h2h_col2:
                h2h_away_wins = st.number_input("Away Wins", min_value=0, value=3, key="h2h_away_wins")
                h2h_draws = st.number_input("Draws", min_value=0, value=1, key="h2h_draws")
            with h2h_col3:
                h2h_home_goals = st.number_input("Home Goals in H2H", min_value=0, value=4, key="h2h_home_goals")
                h2h_away_goals = st.number_input("Away Goals in H2H", min_value=0, value=9, key="h2h_away_goals")

        # Recent Form
        with st.expander("📈 Recent Form Analysis"):
            form_col1, form_col2 = st.columns(2)
            with form_col1:
                st.write(f"**{home_team} Last 6 Matches**")
                home_form = st.multiselect(
                    f"{home_team} Recent Results",
                    options=["Win (3 pts)", "Draw (1 pt)", "Loss (0 pts)"],
                    default=["Win (3 pts)", "Loss (0 pts)", "Win (3 pts)", "Loss (0 pts)", "Loss (0 pts)", "Win (3 pts)"],
                    key="home_form"
                )
            with form_col2:
                st.write(f"**{away_team} Last 6 Matches**")
                away_form = st.multiselect(
                    f"{away_team} Recent Results", 
                    options=["Win (3 pts)", "Draw (1 pt)", "Loss (0 pts)"],
                    default=["Win (3 pts)", "Win (3 pts)", "Draw (1 pt)", "Win (3 pts)", "Win (3 pts)", "Win (3 pts)"],
                    key="away_form"
                )

    with tab2:
        st.markdown("### 💰 Market Data Input") 
        
        odds_col1, odds_col2, odds_col3 = st.columns(3)
        
        with odds_col1:
            st.write("**1X2 Market**")
            home_odds = st.number_input("Home Win Odds", min_value=1.01, value=6.50, step=0.01, key="home_odds")
            draw_odds = st.number_input("Draw Odds", min_value=1.01, value=4.50, step=0.01, key="draw_odds")
            away_odds = st.number_input("Away Win Odds", min_value=1.01, value=1.50, step=0.01, key="away_odds")
        
        with odds_col2:
            st.write("**Over/Under Markets**")
            over_15_odds = st.number_input("Over 1.5 Goals", min_value=1.01, value=1.25, step=0.01, key="over_15_odds")
            over_25_odds = st.number_input("Over 2.5 Goals", min_value=1.01, value=1.80, step=0.01, key="over_25_odds")
            over_35_odds = st.number_input("Over 3.5 Goals", min_value=1.01, value=2.50, step=0.01, key="over_35_odds")
        
        with odds_col3:
            st.write("**Both Teams to Score**")
            btts_yes_odds = st.number_input("BTTS Yes", min_value=1.01, value=1.90, step=0.01, key="btts_yes_odds")
            btts_no_odds = st.number_input("BTTS No", min_value=1.01, value=1.90, step=0.01, key="btts_no_odds")

    with tab3:
        st.markdown("### ⚙️ Model Configuration")
        
        model_col1, model_col2 = st.columns(2)
        
        with model_col1:
            st.write("**Team Context**")
            home_injuries = st.slider("Home Key Absences", 0, 5, 1, key="home_injuries")
            away_injuries = st.slider("Away Key Absences", 0, 5, 2, key="away_injuries")
            
            home_absence_impact = st.select_slider(
                "Home Team Absence Impact",
                options=["Rotation Player", "Regular Starter", "Key Player", "Star Player", "Multiple Key Players"],
                value="Rotation Player",
                key="home_absence_impact"
            )
            away_absence_impact = st.select_slider(
                "Away Team Absence Impact",
                options=["Rotation Player", "Regular Starter", "Key Player", "Star Player", "Multiple Key Players"],
                value="Regular Starter",
                key="away_absence_impact"
            )
            
        with model_col2:
            st.write("**Motivation Factors**")
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

    # Submit button
    submitted = st.button("🎯 GENERATE MULTI-LEAGUE ANALYSIS", type="primary", use_container_width=True)
    
    if submitted:
        if not home_team or not away_team:
            st.error("❌ Please enter both team names")
            return None, None
        
        if home_team == away_team:
            st.error("❌ Home and away teams cannot be the same")
            return None, None
        
        # Convert form selections to points
        form_map = {"Win (3 pts)": 3, "Draw (1 pt)": 1, "Loss (0 pts)": 0}
        home_form_points = [form_map[result] for result in home_form]
        away_form_points = [form_map[result] for result in away_form]
        
        # Convert motivation
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

def display_probability_bar(label: str, probability: float, color: str):
    """Display a probability with a visual bar"""
    st.markdown(f'''
    <div style="margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span><strong>{label}</strong></span>
            <span><strong>{probability:.1f}%</strong></span>
        </div>
        <div class="probability-bar">
            <div class="probability-fill" style="width: {probability}%; background: {color};"></div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

def display_predictions(predictions):
    """Display football predictions with league context"""
    
    st.markdown('<p class="main-header">🎯 Football Predictions</p>', unsafe_allow_html=True)
    st.markdown('<div class="pure-engine-card"><h3>🟢 Signal Engine Output</h3>Professional Multi-League Tier-Calibrated Analysis</div>', unsafe_allow_html=True)
    
    # League and team tiers display
    team_tiers = safe_get(predictions, 'team_tiers') or {}
    home_tier = team_tiers.get('home', 'MEDIUM')
    away_tier = team_tiers.get('away', 'MEDIUM')
    
    # Get league from predictions data
    league = safe_get(predictions, 'league', default='premier_league')
    league_display_name = get_league_display_name(league)
    league_badge_class = get_league_badge(league)
    
    st.markdown(f'''
    <p style="text-align: center; font-size: 1.4rem; font-weight: 600;">
        {predictions["match"]} 
        <span class="tier-badge tier-{home_tier.lower()}">{home_tier}</span> vs 
        <span class="tier-badge tier-{away_tier.lower()}">{away_tier}</span>
    </p>
    <p style="text-align: center; margin-top: 0.5rem;">
        <span class="league-badge {league_badge_class}">{league_display_name}</span>
    </p>
    ''', unsafe_allow_html=True)
    
    # Key metrics with safe defaults
    xg = safe_get(predictions, 'expected_goals') or {'home': 0, 'away': 0}
    match_context = safe_get(predictions, 'match_context') or 'Unknown'
    confidence_score = safe_get(predictions, 'confidence_score') or 0
    data_quality = safe_get(predictions, 'data_quality_score') or 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏠 Expected Goals", f"{xg.get('home', 0):.2f}")
    with col2:
        st.metric("✈️ Expected Goals", f"{xg.get('away', 0):.2f}")
    with col3:
        context_emoji = {
            'defensive_battle': '🛡️',
            'tactical_stalemate': '⚔️', 
            'offensive_showdown': '🔥',
            'home_dominance': '🏠',
            'away_counter': '✈️',
            'unpredictable': '❓'
        }.get(match_context, '❓')
        st.metric("Match Context", f"{context_emoji} {match_context.replace('_', ' ').title()}")
    with col4:
        st.metric("Confidence Score", f"{confidence_score}%")
    
    # System validation
    system_validation = safe_get(predictions, 'system_validation') or {}
    alignment_status = system_validation.get('alignment', 'UNKNOWN')
    
    if alignment_status == 'PERFECT':
        st.markdown('<div class="alignment-perfect">✅ <strong>PERFECT ENGINE ALIGNMENT:</strong> Value Engine confirms Signal Engine predictions</div>', unsafe_allow_html=True)
    elif alignment_status == 'PARTIAL':
        st.markdown('<div class="alignment-warning">⚠️ <strong>PARTIAL ALIGNMENT:</strong> Some inconsistencies detected</div>', unsafe_allow_html=True)
    elif alignment_status == 'CONTRADICTORY':
        st.markdown('<div class="alignment-danger">❌ <strong>CONTRADICTORY ALIGNMENT:</strong> Engines disagree - system error</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-banner">✅ <strong>SYSTEM VALIDATION PASSED:</strong> Realistic probabilities generated</div>', unsafe_allow_html=True)
    
    # Match Outcomes
    st.markdown('<div class="section-title">📈 Match Outcome Probabilities</div>', unsafe_allow_html=True)
    
    outcomes = safe_get(predictions, 'probabilities', 'match_outcomes') or {'home_win': 0, 'draw': 0, 'away_win': 0}
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
    
    exact_scores = safe_get(predictions, 'probabilities', 'exact_scores') or {}
    top_scores = dict(list(exact_scores.items())[:6])
    
    if top_scores:
        score_cols = st.columns(6)
        for idx, (score, prob) in enumerate(top_scores.items()):
            with score_cols[idx]:
                st.metric(f"{score}", f"{prob}%")
    else:
        st.info("No exact score data available")
    
    # Risk Assessment
    risk = safe_get(predictions, 'risk_assessment') or {'risk_level': 'UNKNOWN', 'explanation': 'No data'}
    risk_class = f"risk-{risk.get('risk_level', 'unknown').lower()}"
    
    st.markdown(f'''
    <div class="prediction-card {risk_class}">
        <h3>📊 Risk Assessment</h3>
        <strong>Risk Level:</strong> {risk.get("risk_level", "UNKNOWN")}<br>
        <strong>Explanation:</strong> {risk.get("explanation", "No data available")}<br>
        <strong>Recommendation:</strong> {risk.get("recommendation", "N/A")}<br>
        <strong>Certainty:</strong> {risk.get("certainty", "N/A")}<br>
        <strong>Home Advantage:</strong> {risk.get('home_advantage', 'N/A')}
    </div>
    ''', unsafe_allow_html=True)
    
    # Professional Summary
    st.markdown('<div class="section-title">📝 Football Summary</div>', unsafe_allow_html=True)
    summary = safe_get(predictions, 'summary') or "No summary available."
    st.info(summary)

def display_value_detection(predictions):
    """Display value detection results"""
    
    st.markdown('<p class="main-header">💰 Value Betting Detection</p>', unsafe_allow_html=True)
    st.markdown('<div class="value-engine-card"><h3>🟠 Value Engine Output</h3>Perfectly aligned with Tier-Calibrated Signal Engine</div>', unsafe_allow_html=True)
    
    betting_signals = safe_get(predictions, 'betting_signals') or []
    
    # Get primary predictions for context
    outcomes = safe_get(predictions, 'probabilities', 'match_outcomes') or {}
    btts = safe_get(predictions, 'probabilities', 'both_teams_score') or {}
    over_under = safe_get(predictions, 'probabilities', 'over_under') or {}
    team_tiers = safe_get(predictions, 'team_tiers') or {}
    league = safe_get(predictions, 'league', 'premier_league')
    
    primary_outcome = max(outcomes, key=outcomes.get) if outcomes else 'unknown'
    primary_btts = 'yes' if btts.get('yes', 0) > btts.get('no', 0) else 'no'
    primary_over_under = 'over_25' if over_under.get('over_25', 0) > over_under.get('under_25', 0) else 'under_25'
    
    # Display primary predictions context
    st.markdown('<div class="section-title">🎯 Signal Engine Primary Predictions</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        outcome_map = {'home_win': 'Home Win', 'draw': 'Draw', 'away_win': 'Away Win'}
        st.metric("Primary Outcome", outcome_map.get(primary_outcome, 'Unknown'))
    with col2:
        st.metric("Primary BTTS", "YES" if primary_btts == 'yes' else "NO")
    with col3:
        st.metric("Primary Over/Under", "OVER 2.5" if primary_over_under == 'over_25' else "UNDER 2.5")
    
    if not betting_signals:
        st.markdown('<div class="alignment-perfect">', unsafe_allow_html=True)
        st.info("""
        ## ✅ NO VALUE BETS DETECTED - SYSTEM WORKING PERFECTLY!
        
        **This means:**
        - Pure probabilities align with market expectations  
        - No significant edges above realistic thresholds
        - Market is efficient for this match
        - **PERFECT ENGINE ALIGNMENT ACHIEVED**
        
        **Value Engine is properly confirming Signal Engine predictions without contradictions!**
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Check alignment status
    system_validation = safe_get(predictions, 'system_validation') or {}
    alignment_status = system_validation.get('alignment', 'UNKNOWN')
    
    if alignment_status == 'PERFECT':
        st.markdown('<div class="alignment-perfect">✅ <strong>PERFECT ALIGNMENT:</strong> All value bets confirm Signal Engine predictions</div>', unsafe_allow_html=True)
    elif alignment_status == 'PARTIAL':
        st.markdown('<div class="alignment-warning">⚠️ <strong>PARTIAL ALIGNMENT:</strong> Some inconsistencies detected</div>', unsafe_allow_html=True)
    elif alignment_status == 'CONTRADICTORY':
        st.markdown('<div class="alignment-danger">❌ <strong>CONTRADICTORY ALIGNMENT:</strong> Value bets contradict Signal Engine</div>', unsafe_allow_html=True)
    
    # Value Bet Summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_signals = len(betting_signals)
        st.metric("Value Bets", total_signals)
    
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
    st.markdown('<div class="section-title">🎯 Value Bet Recommendations</div>', unsafe_allow_html=True)
    
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
                    'MEDIUM-HIGH': '🟡',
                    'MEDIUM': '🟡', 
                    'LOW': '🔴',
                    'SPECULATIVE': '⚪'
                }.get(bet.get('confidence', 'SPECULATIVE'), '⚪')
                
                # Check if bet aligns with primary prediction
                aligns_with_primary = True
                alignment_emoji = "✅"
                
                if (bet.get('market') == 'BTTS Yes' and primary_btts == 'no') or \
                   (bet.get('market') == 'BTTS No' and primary_btts == 'yes'):
                    aligns_with_primary = False
                    alignment_emoji = "⚠️"
                
                # Tier-aware contradiction checks
                home_tier = team_tiers.get('home', 'MEDIUM')
                away_tier = team_tiers.get('away', 'MEDIUM')
                
                if (bet.get('market') in ['1x2 Draw', '1x2 Away'] and primary_outcome == 'home_win' and 
                    home_tier == 'ELITE' and away_tier == 'WEAK' and outcomes.get('home_win', 0) > 65):
                    aligns_with_primary = False
                    alignment_emoji = "❌"
                
                alignment_text = "ALIGNS" if aligns_with_primary else "CONTRADICTS"
                
                st.markdown(f'''
                <div class="bet-card {value_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="flex: 2;">
                            <strong>{bet.get('market', 'Unknown')}</strong><br>
                            <small>Model: {bet.get('model_prob', 0)}% | Market: {bet.get('book_prob', 0)}%</small>
                            <div style="margin-top: 0.3rem;">
                                <small>{alignment_emoji} <strong>{alignment_text}</strong> with Signal Engine</small>
                            </div>
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

def display_analytics(predictions):
    """Display advanced analytics"""
    
    st.markdown('<p class="main-header">📈 Advanced Analytics</p>', unsafe_allow_html=True)
    
    # Data Quality and Intelligence Metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-title">📊 Model Performance</div>', unsafe_allow_html=True)
        
        data_quality = safe_get(predictions, 'data_quality_score') or 0
        confidence = safe_get(predictions, 'confidence_score') or 0
        football_iq = safe_get(predictions, 'apex_intelligence', 'football_iq_score') or 0
        coherence = safe_get(predictions, 'apex_intelligence', 'narrative_coherence') or 0
        
        st.metric("Data Quality Score", f"{data_quality:.1f}%")
        st.metric("Overall Confidence", f"{confidence}%")
        st.metric("Football IQ Score", f"{football_iq:.1f}/100")
        st.metric("Narrative Coherence", f"{coherence}%")
    
    with col2:
        st.markdown('<div class="section-title">🎲 Additional Predictions</div>', unsafe_allow_html=True)
        
        corners = safe_get(predictions, 'corner_predictions') or {}
        timing = safe_get(predictions, 'probabilities', 'goal_timing') or {}
        
        st.write(f"**Total Corners:** {corners.get('total', 'N/A')}")
        st.write(f"**First Half Goal:** {timing.get('first_half', 'N/A')}%")
        st.write(f"**Second Half Goal:** {timing.get('second_half', 'N/A')}%")
        
        # Narrative insights
        narrative = safe_get(predictions, 'match_narrative') or {}
        st.write(f"**Match Rhythm:** {narrative.get('expected_tempo', 'N/A').title()}")
        st.write(f"**Defensive Stability:** {narrative.get('defensive_stability', 'N/A').title()}")

def display_analysis(predictions):
    """Display analysis"""
    
    tab1, tab2, tab3 = st.tabs(["🎯 Predictions", "💰 Value Detection", "📈 Analytics"])
    
    with tab1:
        display_predictions(predictions)
    
    with tab2:
        display_value_detection(predictions)
    
    with tab3:
        display_analytics(predictions)

def store_prediction_in_session(prediction):
    """Store prediction in session state for history tracking"""
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    prediction_record = {
        'timestamp': datetime.now().isoformat(),
        'match': prediction['match'],
        'league': prediction.get('league', 'premier_league'),
        'expected_goals': prediction['expected_goals'],
        'team_tiers': prediction.get('team_tiers', {}),
        'probabilities': prediction['probabilities']['match_outcomes'],
        'match_context': prediction['match_context'],
        'confidence_score': prediction['confidence_score'],
        'data_quality': prediction['data_quality_score']
    }
    
    st.session_state.prediction_history.append(prediction_record)
    
    if len(st.session_state.prediction_history) > 20:
        st.session_state.prediction_history = st.session_state.prediction_history[-20:]

def main():
    """Main application function"""
    
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    if st.session_state.predictions:
        display_analysis(st.session_state.predictions)
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Analyze New Match", use_container_width=True):
                st.session_state.predictions = None
                st.rerun()
        
        with col2:
            if st.button("📊 View History", use_container_width=True):
                if st.session_state.prediction_history:
                    st.write("**Recent Predictions:**")
                    for i, pred in enumerate(st.session_state.prediction_history[-5:]):
                        with st.expander(f"Prediction {i+1}: {pred['match']}"):
                            st.write(f"Date: {pred.get('timestamp', 'N/A')}")
                            st.write(f"League: {get_league_display_name(pred.get('league', 'premier_league'))}")
                            st.write(f"Expected Goals: Home {pred['expected_goals']['home']:.2f} - Away {pred['expected_goals']['away']:.2f}")
                            st.write(f"Team Tiers: {pred.get('team_tiers', {}).get('home', 'N/A')} vs {pred.get('team_tiers', {}).get('away', 'N/A')}")
                            st.write(f"Probabilities: {pred['probabilities']}")
                            st.write(f"Match Context: {pred['match_context']}")
                            st.write(f"Confidence: {pred['confidence_score']}%")
                else:
                    st.info("No prediction history yet.")
        
        return
    
    match_data, mc_iterations = create_input_form()
    
    if match_data:
        with st.spinner("🔍 Running multi-league calibrated analysis..."):
            try:
                predictor = AdvancedFootballPredictor(match_data)
                predictions = predictor.generate_comprehensive_analysis(mc_iterations)
                
                # Add league information to predictions for display
                predictions['league'] = match_data['league']
                
                st.session_state.predictions = predictions
                store_prediction_in_session(predictions)
                
                # Check alignment status
                system_validation = safe_get(predictions, 'system_validation') or {}
                alignment_status = system_validation.get('alignment', 'UNKNOWN')
                
                if alignment_status == 'PERFECT':
                    st.success("✅ PERFECT ALIGNMENT ACHIEVED! Value Engine confirms Signal Engine predictions!")
                elif alignment_status == 'PARTIAL':
                    st.warning("⚠️ PARTIAL ALIGNMENT: Some inconsistencies detected")
                elif alignment_status == 'CONTRADICTORY':
                    st.error("❌ CONTRADICTORY ALIGNMENT: Engines disagree - system error")
                else:
                    st.success("✅ Analysis completed with realistic probabilities!")
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Analysis error: {str(e)}")
                st.info("💡 Check input parameters and try again")

if __name__ == "__main__":
    main()
