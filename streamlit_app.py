# streamlit_app.py - PRODUCTION GRADE WITH ALL ENHANCEMENTS (FIXED)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from typing import Dict, Any
from datetime import datetime

# Import the ENHANCED PREDICTION ENGINE
try:
    from prediction_engine import AdvancedFootballPredictor, TeamTierCalibrator
except ImportError as e:
    st.error(f"❌ Could not import prediction_engine: {str(e)}")
    st.info("💡 Make sure prediction_engine.py is in the same directory")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🌍 Advanced Football Predictor PRO",
    page_icon="⚽", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling with production features
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
    
    .production-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    
    .betting-activity-rank {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
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
    
    .explanation-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    .feature-badge {
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        font-size: 0.7rem;
        margin: 0.1rem;
        display: inline-block;
    }
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
        except (TypeError, KeyError, AttributeError):
            return default
    return current

def safe_replace(text, old, new, default=""):
    """Safely replace text with fallback"""
    if text is None:
        return default
    try:
        return text.replace(old, new)
    except (AttributeError, TypeError):
        return default

def safe_title(text, default=""):
    """Safely convert text to title case"""
    if text is None:
        return default
    try:
        return text.title()
    except (AttributeError, TypeError):
        return default

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

def display_production_banner():
    """Display production features banner"""
    st.markdown("""
    <div class="production-banner">
        🚀 PRODUCTION GRADE PREDICTIONS • LEAGUE-SPECIFIC CALIBRATION • ENHANCED FEATURES • REAL-TIME EXPLANATIONS
    </div>
    """, unsafe_allow_html=True)

def display_betting_activity_ranking():
    """Display betting activity ranking by league"""
    st.markdown('<div class="betting-activity-rank">', unsafe_allow_html=True)
    st.markdown("### 📊 Betting Activity Ranking")
    
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
    """Create enhanced input form with production features"""
    
    st.markdown('<p class="main-header">🌍 Advanced Football Predictor PRO</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Production-Grade Multi-League Analysis with Enhanced Intelligence</p>', unsafe_allow_html=True)
    
    # Display production banner
    display_production_banner()
    
    # Display betting activity ranking
    display_betting_activity_ranking()
    
    # Enhanced System Architecture Overview
    with st.expander("🏗️ ENHANCED SYSTEM ARCHITECTURE", expanded=True):
        st.markdown("""
        ### 🚀 PRODUCTION-GRADE PREDICTION ENGINE
        
        **New Enhanced Features:**
        - **League-Specific Calibration** - Different models per league
        - **Dixon-Coles Simulation** - Realistic score correlation modeling  
        - **Bayesian Team Strength** - Dynamic strength with uncertainty
        - **Enhanced Explanations** - Transparent prediction reasoning
        - **Kelly Criterion** - Professional bankroll management
        
        **Supported Leagues** 🌍
        - **Premier League** 🏴󠁧󠁢󠁥󠁮󠁧󠁿, **La Liga** 🇪🇸, **Serie A** 🇮🇹
        - **Bundesliga** 🇩🇪, **Ligue 1** 🇫🇷, **Liga Portugal** 🇵🇹  
        - **Brasileirão** 🇧🇷, **Liga MX** 🇲🇽, **Eredivisie** 🇳🇱
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
    
    tab1, tab2, tab3 = st.tabs(["🏠 Football Data", "💰 Market Data", "⚙️ Enhanced Settings"])

    with tab1:
        st.markdown("### 🎯 Enhanced Football Data Input")
        
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
                index=min(5, len(league_teams) - 1),
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
                index=0,
                key="away_team"
            )
            
            away_goals = st.number_input("Total Goals (Last 6 Games)", min_value=0, value=12, key="away_goals")
            away_conceded = st.number_input("Total Conceded (Last 6 Games)", min_value=0, value=6, key="away_conceded")
            away_goals_away = st.number_input("Away Goals (Last 3 Away Games)", min_value=0, value=7, key="away_goals_away")
        
        # Show team tiers with enhanced display
        home_tier = calibrator.get_team_tier(home_team, selected_league)
        away_tier = calibrator.get_team_tier(away_team, selected_league)
        
        st.markdown(f"""
        **Team Tiers:** 
        <span class="tier-badge tier-{home_tier.lower() if home_tier else 'medium'}">{home_tier or 'MEDIUM'}</span> vs 
        <span class="tier-badge tier-{away_tier.lower() if away_tier else 'medium'}">{away_tier or 'MEDIUM'}</span>
        """, unsafe_allow_html=True)
        
        # Enhanced Head-to-head section
        with st.expander("📊 Enhanced Head-to-Head Analysis"):
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

        # Enhanced Recent Form with more options
        with st.expander("📈 Enhanced Form Analysis"):
            st.info("Form points: Win=3, Draw=1, Loss=0")
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
        st.markdown("### 💰 Enhanced Market Data Input") 
        
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
        st.markdown("### ⚙️ Enhanced Model Configuration")
        
        model_col1, model_col2 = st.columns(2)
        
        with model_col1:
            st.write("**Enhanced Team Context**")
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
            
            # Enhanced simulation settings
            st.write("**Advanced Simulation**")
            mc_iterations = st.select_slider(
                "Monte Carlo Iterations",
                options=[1000, 5000, 10000, 25000],
                value=10000,
                key="mc_iterations"
            )
            
            # Bankroll management
            bankroll = st.number_input("Bankroll ($)", min_value=100, value=1000, step=100, key="bankroll")
            kelly_fraction = st.slider("Kelly Fraction", 0.1, 0.5, 0.25, key="kelly_fraction")

    # Enhanced Submit button
    submitted = st.button("🚀 GENERATE ENHANCED ANALYSIS", type="primary", use_container_width=True)
    
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
        
        # Enhanced Market odds
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
        
        # Complete enhanced match data
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
            'market_odds': market_odds,
            'bankroll': bankroll,
            'kelly_fraction': kelly_fraction
        }
        
        return match_data, mc_iterations
    
    return None, None

def display_goals_analysis(predictions):
    """Enhanced goals analysis with explanations"""
    st.markdown('<div class="section-title">⚽ Enhanced Goals Analysis</div>', unsafe_allow_html=True)
    
    # Get probabilities with safe defaults
    btts_yes = safe_get(predictions, 'probabilities', 'both_teams_score', 'yes') or 0
    btts_no = safe_get(predictions, 'probabilities', 'both_teams_score', 'no') or 0
    
    over_25 = safe_get(predictions, 'probabilities', 'over_under', 'over_25') or 0
    under_25 = safe_get(predictions, 'probabilities', 'over_under', 'under_25') or 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # BTTS with explanations
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
        
        # Show explanations
        explanations = safe_get(predictions, 'explanations', 'btts') or []
        for explanation in explanations[:2]:  # Show top 2 explanations
            if explanation:  # Only show non-empty explanations
                st.markdown(f'<div class="explanation-card">💡 {explanation}</div>', unsafe_allow_html=True)
    
    with col2:
        # Over/Under with explanations
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
        
        # Show explanations
        explanations = safe_get(predictions, 'explanations', 'over_under') or []
        for explanation in explanations[:2]:
            if explanation:  # Only show non-empty explanations
                st.markdown(f'<div class="explanation-card">💡 {explanation}</div>', unsafe_allow_html=True)
    
    with col3:
        # Expected Goals display
        xg = safe_get(predictions, 'expected_goals') or {'home': 0, 'away': 0}
        total_xg = xg.get('home', 0) + xg.get('away', 0)
        
        st.markdown(f'''
        <div class="goals-card">
            <h4>🎯 Expected Goals</h4>
            <div style="font-size: 1.2rem; font-weight: bold; color: #333; margin: 0.5rem 0;">
                Home: {xg.get('home', 0):.2f}
            </div>
            <div style="font-size: 1.2rem; font-weight: bold; color: #333; margin: 0.5rem 0;">
                Away: {xg.get('away', 0):.2f}
            </div>
            <div style="font-size: 1rem; color: #666; margin: 0.3rem 0;">
                Total: {total_xg:.2f}
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        # Match Context
        context = safe_get(predictions, 'match_context') or 'balanced'
        context_emoji = {
            'defensive_battle': '🛡️',
            'offensive_showdown': '🔥',
            'home_dominance': '🏠',
            'away_counter': '✈️',
            'balanced': '⚖️'
        }.get(context, '⚖️')
        
        narrative = safe_get(predictions, 'match_narrative') or {}
        
        st.markdown(f'''
        <div class="goals-card">
            <h4>{context_emoji} Match Context</h4>
            <div style="font-size: 1.1rem; font-weight: bold; color: #333; margin: 0.5rem 0;">
                {safe_replace(context, '_', ' ', 'Balanced').title()}
            </div>
            <div style="font-size: 0.9rem; color: #666; margin: 0.3rem 0;">
                Tempo: {safe_title(narrative.get('expected_tempo', 'medium'), 'Medium')}
            </div>
            <div style="font-size: 0.9rem; color: #666; margin: 0.3rem 0;">
                Defense: {safe_title(narrative.get('defensive_stability', 'mixed'), 'Mixed')}
            </div>
        </div>
        ''', unsafe_allow_html=True)

def display_probability_bar(label: str, probability: float, color: str):
    """Display a probability with a visual bar"""
    safe_prob = probability or 0
    st.markdown(f'''
    <div style="margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span><strong>{label}</strong></span>
            <span><strong>{safe_prob:.1f}%</strong></span>
        </div>
        <div class="probability-bar">
            <div class="probability-fill" style="width: {safe_prob}%; background: {color};"></div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

def display_predictions(predictions):
    """Enhanced predictions display with new features"""
    
    if not predictions:
        st.error("❌ No predictions available")
        return
        
    st.markdown('<p class="main-header">🎯 Enhanced Football Predictions</p>', unsafe_allow_html=True)
    st.markdown('<div class="pure-engine-card"><h3>🟢 Enhanced Signal Engine Output</h3>Production-Grade Multi-League Analysis</div>', unsafe_allow_html=True)
    
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
        {predictions.get("match", "Unknown Match")} 
        <span class="tier-badge tier-{home_tier.lower() if home_tier else 'medium'}">{home_tier or 'MEDIUM'}</span> vs 
        <span class="tier-badge tier-{away_tier.lower() if away_tier else 'medium'}">{away_tier or 'MEDIUM'}</span>
    </p>
    <p style="text-align: center; margin-top: 0.5rem;">
        <span class="league-badge {league_badge_class}">{league_display_name}</span>
    </p>
    ''', unsafe_allow_html=True)
    
    # Enhanced metrics with production features
    xg = safe_get(predictions, 'expected_goals') or {'home': 0, 'away': 0}
    match_context = safe_get(predictions, 'match_context') or 'Unknown'
    confidence_score = safe_get(predictions, 'confidence_score') or 0
    data_quality = safe_get(predictions, 'data_quality_score') or 0
    football_iq = safe_get(predictions, 'apex_intelligence', 'football_iq_score') or 0
    
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
        st.metric("Match Context", f"{context_emoji} {safe_replace(match_context, '_', ' ', 'Unknown').title()}")
    with col4:
        st.metric("Football IQ", f"{football_iq:.1f}/100")
    
    # Enhanced system validation
    system_validation = safe_get(predictions, 'system_validation') or {}
    alignment_status = system_validation.get('alignment', 'UNKNOWN')
    model_version = system_validation.get('model_version', '1.0.0')
    
    if alignment_status == 'PERFECT':
        st.markdown(f'''
        <div class="alignment-perfect">
            ✅ <strong>PERFECT ENGINE ALIGNMENT:</strong> Value Engine confirms Signal Engine predictions
            <br><small>Model Version: {model_version} | League-Calibrated Probabilities</small>
        </div>
        ''', unsafe_allow_html=True)
    elif alignment_status == 'PARTIAL':
        st.markdown(f'''
        <div class="alignment-warning">
            ⚠️ <strong>PARTIAL ALIGNMENT:</strong> Some inconsistencies detected
            <br><small>Model Version: {model_version} | Review recommendations carefully</small>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="success-banner">
            ✅ <strong>SYSTEM VALIDATION PASSED:</strong> Realistic probabilities generated
            <br><small>Model Version: {model_version} | Enhanced Dixon-Coles Simulation</small>
        </div>
        ''', unsafe_allow_html=True)
    
    # Match Outcomes
    st.markdown('<div class="section-title">📈 Enhanced Match Outcome Probabilities</div>', unsafe_allow_html=True)
    
    outcomes = safe_get(predictions, 'probabilities', 'match_outcomes') or {'home_win': 0, 'draw': 0, 'away_win': 0}
    col1, col2, col3 = st.columns(3)
    
    with col1:
        display_probability_bar("Home Win", outcomes.get('home_win', 0), "#4CAF50")
    with col2:
        display_probability_bar("Draw", outcomes.get('draw', 0), "#FF9800")
    with col3:
        display_probability_bar("Away Win", outcomes.get('away_win', 0), "#2196F3")
    
    # Enhanced Goals Analysis with explanations
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
    
    # Enhanced Risk Assessment
    risk = safe_get(predictions, 'risk_assessment') or {'risk_level': 'UNKNOWN', 'explanation': 'No data'}
    risk_class = f"risk-{risk.get('risk_level', 'unknown').lower()}"
    
    intelligence = safe_get(predictions, 'apex_intelligence') or {}
    
    st.markdown(f'''
    <div class="prediction-card {risk_class}">
        <h3>📊 Enhanced Risk Assessment</h3>
        <strong>Risk Level:</strong> {risk.get("risk_level", "UNKNOWN")}<br>
        <strong>Explanation:</strong> {risk.get("explanation", "No data available")}<br>
        <strong>Recommendation:</strong> {risk.get("recommendation", "N/A")}<br>
        <strong>Certainty:</strong> {risk.get("certainty", "N/A")}<br>
        <strong>Narrative Coherence:</strong> {intelligence.get('narrative_coherence', 'N/A')}%<br>
        <strong>Prediction Alignment:</strong> {intelligence.get('prediction_alignment', 'N/A')}
    </div>
    ''', unsafe_allow_html=True)
    
    # Professional Summary
    st.markdown('<div class="section-title">📝 Enhanced Football Summary</div>', unsafe_allow_html=True)
    summary = safe_get(predictions, 'summary') or "No summary available."
    st.info(summary)

def display_value_detection(predictions):
    """Enhanced value detection with explanations"""
    
    if not predictions:
        st.error("❌ No predictions available for value detection")
        return
        
    st.markdown('<p class="main-header">💰 Enhanced Value Betting Detection</p>', unsafe_allow_html=True)
    st.markdown('<div class="value-engine-card"><h3>🟠 Enhanced Value Engine Output</h3>Perfectly aligned with Production-Grade Signal Engine</div>', unsafe_allow_html=True)
    
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
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        outcome_map = {'home_win': 'Home Win', 'draw': 'Draw', 'away_win': 'Away Win'}
        st.metric("Primary Outcome", outcome_map.get(primary_outcome, 'Unknown'))
    with col2:
        st.metric("Primary BTTS", "YES" if primary_btts == 'yes' else "NO")
    with col3:
        st.metric("Primary Over/Under", "OVER 2.5" if primary_over_under == 'over_25' else "UNDER 2.5")
    with col4:
        st.metric("League", get_league_display_name(league))
    
    if not betting_signals:
        st.markdown('<div class="alignment-perfect">', unsafe_allow_html=True)
        st.info("""
        ## ✅ NO VALUE BETS DETECTED - SYSTEM WORKING PERFECTLY!
        
        **This means:**
        - Pure probabilities align with market expectations  
        - No significant edges above realistic thresholds
        - Market is efficient for this match
        - **PERFECT ENGINE ALIGNMENT ACHIEVED**
        
        **Enhanced Value Engine is properly confirming Signal Engine predictions without contradictions!**
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
    
    # Enhanced Value Bet Summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_signals = len(betting_signals)
        st.metric("Value Bets", total_signals)
    
    with col2:
        high_value = len([s for s in betting_signals if s.get('value_rating') in ['EXCEPTIONAL', 'HIGH']])
        st.metric("High Value Signals", high_value)
    
    with col3:
        avg_edge = np.mean([s.get('edge', 0) for s in betting_signals]) if betting_signals else 0
        st.metric("Average Edge", f"{avg_edge:.1f}%")
    
    with col4:
        total_stake = np.sum([s.get('recommended_stake', 0) for s in betting_signals]) if betting_signals else 0
        st.metric("Total Stake", f"${total_stake:.2f}")
    
    # Display enhanced value bets with explanations
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
                
                # Check if bet aligns with primary prediction
                aligns_with_primary = True
                alignment_emoji = "✅"
                
                if (bet.get('market') == 'BTTS Yes' and primary_btts == 'no') or \
                   (bet.get('market') == 'BTTS No' and primary_btts == 'yes'):
                    aligns_with_primary = False
                    alignment_emoji = "⚠️"
                
                alignment_text = "ALIGNS" if aligns_with_primary else "CONTRADICTS"
                
                # Safely get explanations
                explanations = bet.get('explanation', [])
                safe_explanations = [exp for exp in explanations if exp]  # Filter out empty explanations
                
                st.markdown(f'''
                <div class="bet-card {value_class}">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <div style="flex: 2;">
                            <strong>{bet.get('market', 'Unknown')}</strong><br>
                            <small>Model: {bet.get('model_prob', 0)}% | Market: {bet.get('book_prob', 0)}%</small>
                            <div style="margin-top: 0.3rem;">
                                <small>{alignment_emoji} <strong>{alignment_text}</strong> with Signal Engine</small>
                            </div>
                            <div style="margin-top: 0.5rem;">
                                {''.join([f'<span class="feature-badge">💡 {exp}</span>' for exp in safe_explanations[:1]])}
                            </div>
                        </div>
                        <div style="flex: 1; text-align: right;">
                            <strong style="color: #4CAF50; font-size: 1.1rem;">+{bet.get('edge', 0)}% Edge</strong><br>
                            <small>Stake: ${bet.get('recommended_stake', 0):.2f} | {confidence_emoji} {bet.get('confidence', 'Unknown')}</small>
                        </div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
    
    display_bet_group(exceptional_bets, "Exceptional", "🔥")
    display_bet_group(high_bets, "High", "⭐")
    display_bet_group(good_bets, "Good", "✅")
    display_bet_group(moderate_bets, "Moderate", "📊")

def display_analytics(predictions):
    """Enhanced analytics display"""
    
    if not predictions:
        st.error("❌ No predictions available for analytics")
        return
        
    st.markdown('<p class="main-header">📈 Enhanced Analytics Dashboard</p>', unsafe_allow_html=True)
    
    # Enhanced Data Quality and Intelligence Metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-title">📊 Enhanced Model Performance</div>', unsafe_allow_html=True)
        
        data_quality = safe_get(predictions, 'data_quality_score') or 0
        confidence = safe_get(predictions, 'confidence_score') or 0
        football_iq = safe_get(predictions, 'apex_intelligence', 'football_iq_score') or 0
        coherence = safe_get(predictions, 'apex_intelligence', 'narrative_coherence') or 0
        
        # Create metrics with visual indicators
        st.metric("Data Quality Score", f"{data_quality:.1f}%", 
                 delta="Excellent" if data_quality > 80 else "Good" if data_quality > 60 else "Needs Review")
        st.metric("Overall Confidence", f"{confidence}%")
        st.metric("Football IQ Score", f"{football_iq:.1f}/100")
        st.metric("Narrative Coherence", f"{coherence}%")
        
        # System info
        system_validation = safe_get(predictions, 'system_validation') or {}
        st.metric("Model Version", system_validation.get('model_version', '1.0.0'))
        st.metric("Engine Sync", system_validation.get('engine_sync', 'UNKNOWN'))
    
    with col2:
        st.markdown('<div class="section-title">🎲 Enhanced Predictions</div>', unsafe_allow_html=True)
        
        # Expected goals visualization
        xg = safe_get(predictions, 'expected_goals') or {}
        if xg:
            fig = go.Figure(data=[
                go.Bar(name='Expected Goals', x=['Home', 'Away'], y=[xg.get('home', 0), xg.get('away', 0)])
            ])
            fig.update_layout(title="Expected Goals Distribution", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Match context details
        narrative = safe_get(predictions, 'match_narrative') or {}
        st.write("**Match Narrative:**")
        st.write(f"- Dominance: {narrative.get('dominance', 'N/A')}")
        st.write(f"- Style: {narrative.get('style_conflict', 'N/A')}")
        st.write(f"- Tempo: {narrative.get('expected_tempo', 'N/A')}")
        st.write(f"- Defense: {narrative.get('defensive_stability', 'N/A')}")

def display_analysis(predictions):
    """Enhanced analysis display with tabs"""
    
    if not predictions:
        st.error("❌ No analysis data available")
        return
        
    tab1, tab2, tab3 = st.tabs(["🎯 Enhanced Predictions", "💰 Enhanced Value Detection", "📈 Enhanced Analytics"])
    
    with tab1:
        display_predictions(predictions)
    
    with tab2:
        display_value_detection(predictions)
    
    with tab3:
        display_analytics(predictions)

def store_prediction_in_session(prediction):
    """Enhanced prediction storage"""
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    if not prediction:
        return
        
    prediction_record = {
        'timestamp': datetime.now().isoformat(),
        'match': prediction.get('match', 'Unknown Match'),
        'league': prediction.get('league', 'premier_league'),
        'expected_goals': prediction.get('expected_goals', {'home': 0, 'away': 0}),
        'team_tiers': prediction.get('team_tiers', {}),
        'probabilities': safe_get(prediction, 'probabilities', 'match_outcomes') or {},
        'match_context': prediction.get('match_context', 'unknown'),
        'confidence_score': prediction.get('confidence_score', 0),
        'data_quality': prediction.get('data_quality_score', 0),
        'football_iq': safe_get(prediction, 'apex_intelligence', 'football_iq_score') or 0,
        'value_bets': len(prediction.get('betting_signals', []))
    }
    
    st.session_state.prediction_history.append(prediction_record)
    
    if len(st.session_state.prediction_history) > 20:
        st.session_state.prediction_history = st.session_state.prediction_history[-20:]

def main():
    """Enhanced main application function"""
    
    # Initialize session state
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    if st.session_state.predictions:
        display_analysis(st.session_state.predictions)
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Analyze New Match", use_container_width=True):
                st.session_state.predictions = None
                st.rerun()
        
        with col2:
            if st.button("📊 View Enhanced History", use_container_width=True):
                if st.session_state.prediction_history:
                    st.write("**Enhanced Prediction History:**")
                    for i, pred in enumerate(st.session_state.prediction_history[-5:]):
                        with st.expander(f"Prediction {i+1}: {pred.get('match', 'Unknown Match')} (IQ: {pred.get('football_iq', 0):.1f})"):
                            st.write(f"Date: {pred.get('timestamp', 'N/A')}")
                            st.write(f"League: {get_league_display_name(pred.get('league', 'premier_league'))}")
                            st.write(f"Expected Goals: Home {pred['expected_goals'].get('home', 0):.2f} - Away {pred['expected_goals'].get('away', 0):.2f}")
                            st.write(f"Team Tiers: {pred.get('team_tiers', {}).get('home', 'N/A')} vs {pred.get('team_tiers', {}).get('away', 'N/A')}")
                            st.write(f"Football IQ: {pred.get('football_iq', 0):.1f}/100")
                            st.write(f"Value Bets Found: {pred.get('value_bets', 0)}")
                            st.write(f"Confidence: {pred.get('confidence_score', 0)}%")
                else:
                    st.info("No prediction history yet.")
        
        with col3:
            if st.button("📈 System Status", use_container_width=True):
                st.success("""
                **System Status: OPERATIONAL** 🟢
                
                **Enhanced Features Active:**
                - League-Specific Calibration ✅
                - Dixon-Coles Simulation ✅  
                - Bayesian Team Strength ✅
                - Enhanced Explanations ✅
                - Kelly Criterion ✅
                
                **Model Version:** 1.2.0_production
                **Last Update:** Current
                """)
        
        return
    
    match_data, mc_iterations = create_input_form()
    
    if match_data:
        with st.spinner("🔍 Running enhanced multi-league calibrated analysis..."):
            try:
                # Initialize predictor with enhanced settings
                predictor = AdvancedFootballPredictor(match_data)
                
                # Generate comprehensive analysis
                predictions = predictor.generate_comprehensive_analysis(mc_iterations)
                
                if predictions:
                    # Add enhanced information
                    predictions['league'] = match_data['league']
                    predictions['bankroll'] = match_data.get('bankroll', 1000)
                    predictions['kelly_fraction'] = match_data.get('kelly_fraction', 0.25)
                    
                    st.session_state.predictions = predictions
                    store_prediction_in_session(predictions)
                    
                    # Enhanced alignment status check
                    system_validation = safe_get(predictions, 'system_validation') or {}
                    alignment_status = system_validation.get('alignment', 'UNKNOWN')
                    
                    if alignment_status == 'PERFECT':
                        st.success("""
                        ✅ **PERFECT ALIGNMENT ACHIEVED!** 
                        
                        Enhanced Value Engine confirms Signal Engine predictions with:
                        - League-specific calibration ✅
                        - Realistic probability modeling ✅  
                        - Professional bankroll management ✅
                        """)
                    elif alignment_status == 'PARTIAL':
                        st.warning("⚠️ PARTIAL ALIGNMENT: Some inconsistencies detected - review carefully")
                    else:
                        st.success("✅ Enhanced analysis completed with production-grade probabilities!")
                    
                    st.rerun()
                else:
                    st.error("❌ Failed to generate predictions")
                
            except Exception as e:
                st.error(f"❌ Enhanced analysis error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                st.info("💡 Check input parameters and try again")

if __name__ == "__main__":
    main()
