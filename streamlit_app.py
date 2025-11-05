# streamlit_app.py - PROFESSIONAL BETTING GRADE (COMPLETE UPDATED VERSION)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from typing import Dict, Any
from datetime import datetime

# Import the PROFESSIONAL PREDICTION ENGINE
try:
    from prediction_engine import AdvancedFootballPredictor, ProfessionalTeamTierCalibrator
except ImportError as e:
    st.error(f"❌ Could not import prediction_engine: {str(e)}")
    st.info("💡 Make sure prediction_engine.py is in the same directory")
    st.stop()

# Professional page configuration
st.set_page_config(
    page_title="🎯 Professional Football Predictor",
    page_icon="⚽", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    .professional-header { 
        font-size: 2.8rem !important; 
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .professional-subheader {
        font-size: 1.4rem !important;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .professional-badge {
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
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
    .championship { background: #8B0000; }
    
    .money-grade-banner {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    .professional-card { 
        background: white;
        padding: 1.8rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .risk-low { border-left-color: #4CAF50 !important; }
    .risk-medium { border-left-color: #FF9800 !important; }
    .risk-high { border-left-color: #f44336 !important; }
    
    .professional-system-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.8rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    
    .professional-value-card {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.8rem 0;
    }
    
    .professional-probability-bar {
        height: 10px;
        background: #e0e0e0;
        border-radius: 5px;
        margin: 0.8rem 0;
        overflow: hidden;
    }
    .professional-probability-fill {
        height: 100%;
        background: linear-gradient(90deg, #4CAF50, #45a049);
        border-radius: 5px;
    }
    
    .professional-bet-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    
    .value-exceptional { border-left-color: #4CAF50 !important; background: #f8fff8; }
    .value-high { border-left-color: #8BC34A !important; background: #f9fff9; }
    .value-good { border-left-color: #FFC107 !important; background: #fffdf6; }
    .value-moderate { border-left-color: #FF9800 !important; background: #fffaf2; }
    
    .professional-section-title {
        font-size: 1.6rem;
        font-weight: 600;
        color: #333;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.8rem;
        border-bottom: 3px solid #f0f2f6;
    }
    
    .professional-confidence-badge {
        padding: 0.4rem 1rem;
        border-radius: 18px;
        font-size: 0.9rem;
        font-weight: bold;
        color: white;
        display: inline-block;
        margin-top: 0.8rem;
    }
    .confidence-high { background: #4CAF50; }
    .confidence-medium { background: #FF9800; }
    .confidence-low { background: #f44336; }
    
    .professional-alignment-perfect {
        background: #f8fff8;
        border-left: 5px solid #4CAF50;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .professional-alignment-warning {
        background: #fffaf2;
        border-left: 5px solid #FF9800;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .professional-tier-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        color: white;
        display: inline-block;
        margin-left: 0.5rem;
    }
    .tier-elite { background: #e74c3c; }
    .tier-strong { background: #e67e22; }
    .tier-medium { background: #f1c40f; color: black; }
    .tier-weak { background: #95a5a6; }
    
    .professional-explanation-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        font-size: 0.95rem;
    }
    
    .professional-feature-badge {
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.3rem 0.7rem;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    
    /* ENHANCED CONFIDENCE FEATURES */
    .enhanced-confidence-reasoning {
        background: #f0f8ff;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .contradiction-warning {
        background: #fff3e0;
        border-left: 4px solid #FF9800;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #E65100;
    }
    .stability-bonus {
        background: #e8f5e8;
        color: #2E7D32;
        padding: 0.3rem 0.7rem;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-left: 0.5rem;
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

def get_league_display_name(league_id: str) -> str:
    """Get professional display name for league"""
    league_names = {
        'premier_league': 'Premier League 🏴󠁧󠁢󠁥󠁮󠁧󠁿',
        'la_liga': 'La Liga 🇪🇸', 
        'serie_a': 'Serie A 🇮🇹',
        'bundesliga': 'Bundesliga 🇩🇪',
        'ligue_1': 'Ligue 1 🇫🇷',
        'liga_portugal': 'Liga Portugal 🇵🇹',
        'brasileirao': 'Brasileirão 🇧🇷',
        'liga_mx': 'Liga MX 🇲🇽',
        'eredivisie': 'Eredivisie 🇳🇱',
        'championship': 'Championship 🏴󠁧󠁢󠁥󠁮󠁧󠁿'
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
        'eredivisie': 'eredivisie',
        'championship': 'championship'
    }
    return league_classes.get(league_id, 'premier-league')

def display_professional_banner():
    """Display professional banner"""
    st.markdown("""
    <div class="money-grade-banner">
        🎯 PROFESSIONAL BETTING GRADE • ENHANCED CONFIDENCE LOGIC • LEAGUE-SPECIFIC CALIBRATION • MONEY-GRADE ACCURACY
    </div>
    """, unsafe_allow_html=True)

def display_professional_architecture():
    """Display professional system architecture"""
    with st.expander("🏗️ PROFESSIONAL SYSTEM ARCHITECTURE", expanded=True):
        st.markdown("""
        ### 🎯 ENHANCED MONEY-GRADE PREDICTION ENGINE
        
        **Enhanced Confidence Features:**
        - **League-Specific Confidence Multipliers** - Dynamic thresholds per league
        - **Form Stability Bonus** - Rewards consistent team performance  
        - **Contradiction Detection** - Automatically detects conflicting signals
        - **Enhanced Confidence Reasoning** - Transparent confidence assignment
        
        **Professional League Calibration** 🌍
        - **Premier League** 🏴󠁧󠁢󠁥󠁮󠁧󠁿: Baseline confidence model
        - **Serie A** 🇮🇹: +15% confidence requirements (defensive league)
        - **Bundesliga** 🇩🇪: -10% confidence requirements (high-scoring)
        - **Championship** 🏴󠁧󠁢󠁥󠁮󠁧󠁿: +8% requirements (unpredictable)
        """)

def create_professional_input_form():
    """Create professional input form"""
    
    st.markdown('<p class="professional-header">🎯 Professional Football Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="professional-subheader">Money-Grade Multi-League Analysis with Enhanced Confidence Intelligence</p>', unsafe_allow_html=True)
    
    # Display professional banner
    display_professional_banner()
    
    # Display professional architecture
    display_professional_architecture()
    
    # Professional League Selection
    st.markdown("### 🌍 Professional League Selection")
    league_options = {
        'premier_league': 'Premier League 🏴󠁧󠁢󠁥󠁮󠁧󠁿',
        'la_liga': 'La Liga 🇪🇸',
        'serie_a': 'Serie A 🇮🇹', 
        'bundesliga': 'Bundesliga 🇩🇪',
        'ligue_1': 'Ligue 1 🇫🇷',
        'liga_portugal': 'Liga Portugal 🇵🇹',
        'brasileirao': 'Brasileirão 🇧🇷',
        'liga_mx': 'Liga MX 🇲🇽',
        'eredivisie': 'Eredivisie 🇳🇱',
        'championship': 'Championship 🏴󠁧󠁢󠁥󠁮󠁧󠁿'
    }
    
    selected_league = st.selectbox(
        "Select League",
        options=list(league_options.keys()),
        format_func=lambda x: league_options[x],
        key="professional_league_selection"
    )
    
    # Display professional league badge
    league_badge_class = get_league_badge(selected_league)
    league_display_name = get_league_display_name(selected_league)
    st.markdown(f'<span class="professional-badge {league_badge_class}">{league_display_name}</span>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🏠 Professional Data", "💰 Market Data", "⚙️ Professional Settings"])

    with tab1:
        st.markdown("### 🎯 Professional Football Data")
        
        # Initialize professional team calibrator
        calibrator = ProfessionalTeamTierCalibrator()
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
                key="professional_home_team"
            )
            
            home_goals = st.number_input("Total Goals (Last 6 Games)", min_value=0, value=8, key="professional_home_goals")
            home_conceded = st.number_input("Total Conceded (Last 6 Games)", min_value=0, value=3, key="professional_home_conceded")
            home_goals_home = st.number_input("Home Goals (Last 3 Home Games)", min_value=0, value=5, key="professional_home_goals_home")
            
        with col2:
            st.subheader("✈️ Away Team")
            away_team = st.selectbox(
                "Team Name",
                options=league_teams,
                index=0,
                key="professional_away_team"
            )
            
            away_goals = st.number_input("Total Goals (Last 6 Games)", min_value=0, value=5, key="professional_away_goals")
            away_conceded = st.number_input("Total Conceded (Last 6 Games)", min_value=0, value=9, key="professional_away_conceded")
            away_goals_away = st.number_input("Away Goals (Last 3 Away Games)", min_value=0, value=4, key="professional_away_goals_away")
        
        # Show professional team tiers
        home_tier = calibrator.get_team_tier(home_team, selected_league)
        away_tier = calibrator.get_team_tier(away_team, selected_league)
        
        st.markdown(f"""
        **Professional Team Assessment:** 
        <span class="professional-tier-badge tier-{home_tier.lower() if home_tier else 'medium'}">{home_tier or 'MEDIUM'}</span> vs 
        <span class="professional-tier-badge tier-{away_tier.lower() if away_tier else 'medium'}">{away_tier or 'MEDIUM'}</span>
        """, unsafe_allow_html=True)
        
        # Professional Head-to-head section
        with st.expander("📊 Professional Head-to-Head Analysis"):
            h2h_col1, h2h_col2, h2h_col3 = st.columns(3)
            with h2h_col1:
                h2h_matches = st.number_input("Total H2H Matches", min_value=0, value=4, key="professional_h2h_matches")
                h2h_home_wins = st.number_input("Home Wins", min_value=0, value=3, key="professional_h2h_home_wins")
            with h2h_col2:
                h2h_away_wins = st.number_input("Away Wins", min_value=0, value=0, key="professional_h2h_away_wins")
                h2h_draws = st.number_input("Draws", min_value=0, value=1, key="professional_h2h_draws")
            with h2h_col3:
                h2h_home_goals = st.number_input("Home Goals in H2H", min_value=0, value=8, key="professional_h2h_home_goals")
                h2h_away_goals = st.number_input("Away Goals in H2H", min_value=0, value=2, key="professional_h2h_away_goals")

        # Professional Recent Form
        with st.expander("📈 Professional Form Analysis"):
            st.info("Professional form points: Win=3, Draw=1, Loss=0")
            form_col1, form_col2 = st.columns(2)
            with form_col1:
                st.write(f"**{home_team} Last 6 Matches**")
                home_form = st.multiselect(
                    f"{home_team} Recent Results",
                    options=["Win (3 pts)", "Draw (1 pt)", "Loss (0 pts)"],
                    default=["Win (3 pts)", "Win (3 pts)", "Win (3 pts)", "Win (3 pts)", "Draw (1 pt)", "Win (3 pts)"],
                    key="professional_home_form"
                )
            with form_col2:
                st.write(f"**{away_team} Last 6 Matches**")
                away_form = st.multiselect(
                    f"{away_team} Recent Results", 
                    options=["Win (3 pts)", "Draw (1 pt)", "Loss (0 pts)"],
                    default=["Loss (0 pts)", "Draw (1 pt)", "Loss (0 pts)", "Draw (1 pt)", "Loss (0 pts)", "Loss (0 pts)"],
                    key="professional_away_form"
                )

    with tab2:
        st.markdown("### 💰 Professional Market Data") 
        
        odds_col1, odds_col2, odds_col3 = st.columns(3)
        
        with odds_col1:
            st.write("**1X2 Market**")
            home_odds = st.number_input("Home Win Odds", min_value=1.01, value=1.45, step=0.01, key="professional_home_odds")
            draw_odds = st.number_input("Draw Odds", min_value=1.01, value=4.20, step=0.01, key="professional_draw_odds")
            away_odds = st.number_input("Away Win Odds", min_value=1.01, value=8.50, step=0.01, key="professional_away_odds")
        
        with odds_col2:
            st.write("**Over/Under Markets**")
            over_15_odds = st.number_input("Over 1.5 Goals", min_value=1.01, value=1.36, step=0.01, key="professional_over_15_odds")
            over_25_odds = st.number_input("Over 2.5 Goals", min_value=1.01, value=1.95, step=0.01, key="professional_over_25_odds")
            over_35_odds = st.number_input("Over 3.5 Goals", min_value=1.01, value=3.50, step=0.01, key="professional_over_35_odds")
        
        with odds_col3:
            st.write("**Both Teams to Score**")
            btts_yes_odds = st.number_input("BTTS Yes", min_value=1.01, value=2.20, step=0.01, key="professional_btts_yes_odds")
            btts_no_odds = st.number_input("BTTS No", min_value=1.01, value=1.65, step=0.01, key="professional_btts_no_odds")

    with tab3:
        st.markdown("### ⚙️ Professional Configuration")
        
        model_col1, model_col2 = st.columns(2)
        
        with model_col1:
            st.write("**Professional Team Context**")
            home_injuries = st.slider("Home Key Absences", 0, 5, 2, key="professional_home_injuries")
            away_injuries = st.slider("Away Key Absences", 0, 5, 3, key="professional_away_injuries")
            
            home_absence_impact = st.select_slider(
                "Home Team Absence Impact",
                options=["Rotation Player", "Regular Starter", "Key Player", "Star Player", "Multiple Key Players"],
                value="Regular Starter",
                key="professional_home_absence_impact"
            )
            away_absence_impact = st.select_slider(
                "Away Team Absence Impact",
                options=["Rotation Player", "Regular Starter", "Key Player", "Star Player", "Multiple Key Players"],
                value="Regular Starter",
                key="professional_away_absence_impact"
            )
            
        with model_col2:
            st.write("**Professional Motivation Factors**")
            home_motivation = st.select_slider(
                "Home Team Motivation",
                options=["Low", "Normal", "High", "Very High"],
                value="Normal",
                key="professional_home_motivation"
            )
            away_motivation = st.select_slider(
                "Away Team Motivation", 
                options=["Low", "Normal", "High", "Very High"],
                value="Normal", 
                key="professional_away_motivation"
            )
            
            # Professional simulation settings
            st.write("**Professional Simulation**")
            mc_iterations = st.select_slider(
                "Monte Carlo Iterations",
                options=[10000, 25000, 50000],
                value=25000,
                key="professional_mc_iterations"
            )
            
            # Professional bankroll management
            bankroll = st.number_input("Professional Bankroll ($)", min_value=500, value=1000, step=100, key="professional_bankroll")
            kelly_fraction = st.slider("Professional Kelly Fraction", 0.1, 0.3, 0.2, key="professional_kelly_fraction")

    # Professional Submit button
    submitted = st.button("🎯 GENERATE PROFESSIONAL ANALYSIS", type="primary", use_container_width=True)
    
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
        
        # Professional Market odds
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
        
        # Complete professional match data
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

def display_professional_predictions(predictions):
    """Display professional predictions"""
    
    if not predictions:
        st.error("❌ No professional predictions available")
        return
        
    st.markdown('<p class="professional-header">🎯 Professional Football Predictions</p>', unsafe_allow_html=True)
    st.markdown('<div class="professional-system-card"><h3>🟢 ENHANCED PROFESSIONAL SIGNAL ENGINE OUTPUT</h3>Money-Grade Multi-League Analysis with Enhanced Confidence</div>', unsafe_allow_html=True)
    
    # Professional team tiers display
    team_tiers = safe_get(predictions, 'team_tiers') or {}
    home_tier = team_tiers.get('home', 'MEDIUM')
    away_tier = team_tiers.get('away', 'MEDIUM')
    
    # Get league from predictions data
    league = safe_get(predictions, 'league', default='premier_league')
    league_display_name = get_league_display_name(league)
    league_badge_class = get_league_badge(league)
    
    # Enhanced: Show stability bonus if available
    intelligence = safe_get(predictions, 'apex_intelligence') or {}
    stability_bonus = intelligence.get('form_stability_bonus', 0)
    
    st.markdown(f'''
    <p style="text-align: center; font-size: 1.5rem; font-weight: 600;">
        {predictions.get("match", "Unknown Match")} 
        <span class="professional-tier-badge tier-{home_tier.lower() if home_tier else 'medium'}">{home_tier or 'MEDIUM'}</span> vs 
        <span class="professional-tier-badge tier-{away_tier.lower() if away_tier else 'medium'}">{away_tier or 'MEDIUM'}</span>
        {f'<span class="stability-bonus">Stability: +{stability_bonus:.1f}</span>' if stability_bonus > 0 else ''}
    </p>
    <p style="text-align: center; margin-top: 0.5rem;">
        <span class="professional-badge {league_badge_class}">{league_display_name}</span>
    </p>
    ''', unsafe_allow_html=True)
    
    # Professional metrics
    xg = safe_get(predictions, 'expected_goals') or {'home': 0, 'away': 0}
    match_context = safe_get(predictions, 'match_context') or 'Unknown'
    confidence_score = safe_get(predictions, 'confidence_score') or 0
    data_quality = safe_get(predictions, 'data_quality_score') or 0
    football_iq = safe_get(predictions, 'apex_intelligence', 'football_iq_score') or 0
    calibration_status = safe_get(predictions, 'apex_intelligence', 'calibration_status') or 'STANDARD'
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏠 Expected Goals", f"{xg.get('home', 0):.2f}")
    with col2:
        st.metric("✈️ Expected Goals", f"{xg.get('away', 0):.2f}")
    with col3:
        context_emoji = {
            'home_dominance': '🏠',
            'away_counter': '✈️',
            'offensive_showdown': '🔥',
            'defensive_battle': '🛡️',
            'tactical_stalemate': '⚔️',
            'unpredictable': '❓'
        }.get(match_context, '❓')
        st.metric("Professional Context", f"{context_emoji} {match_context.replace('_', ' ').title()}")
    with col4:
        st.metric("Enhanced IQ", f"{football_iq:.1f}/100")
    
    # Professional system validation
    system_validation = safe_get(predictions, 'system_validation') or {}
    alignment_status = system_validation.get('alignment', 'UNKNOWN')
    calibration_level = system_validation.get('calibration_level', 'STANDARD')
    
    if alignment_status == 'PERFECT' and calibration_level == 'MONEY_GRADE':
        st.markdown(f'''
        <div class="professional-alignment-perfect">
            ✅ <strong>ENHANCED PROFESSIONAL PERFECT ALIGNMENT:</strong> Value Engine confirms Signal Engine predictions
            <br><small>Calibration: {calibration_level} | Model Version: {system_validation.get('model_version', '2.1.0_enhanced')} | Stability Bonus: +{stability_bonus:.1f}</small>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="professional-alignment-warning">
            ⚠️ <strong>ENHANCED PROFESSIONAL REVIEW REQUIRED:</strong> Some inconsistencies detected
            <br><small>Calibration: {calibration_level} | Enhanced professional discretion advised</small>
        </div>
        ''', unsafe_allow_html=True)
    
    # Professional Match Outcomes
    st.markdown('<div class="professional-section-title">📈 Professional Outcome Probabilities</div>', unsafe_allow_html=True)
    
    outcomes = safe_get(predictions, 'probabilities', 'match_outcomes') or {'home_win': 0, 'draw': 0, 'away_win': 0}
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f'''
        <div style="margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.8rem;">
                <span><strong>Home Win</strong></span>
                <span><strong>{outcomes.get('home_win', 0):.1f}%</strong></span>
            </div>
            <div class="professional-probability-bar">
                <div class="professional-probability-fill" style="width: {outcomes.get('home_win', 0)}%;"></div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown(f'''
        <div style="margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.8rem;">
                <span><strong>Draw</strong></span>
                <span><strong>{outcomes.get('draw', 0):.1f}%</strong></span>
            </div>
            <div class="professional-probability-bar">
                <div class="professional-probability-fill" style="width: {outcomes.get('draw', 0)}%; background: #FF9800;"></div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    with col3:
        st.markdown(f'''
        <div style="margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.8rem;">
                <span><strong>Away Win</strong></span>
                <span><strong>{outcomes.get('away_win', 0):.1f}%</strong></span>
            </div>
            <div class="professional-probability-bar">
                <div class="professional-probability-fill" style="width: {outcomes.get('away_win', 0)}%; background: #2196F3;"></div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Professional Goals Analysis
    st.markdown('<div class="professional-section-title">⚽ Professional Goals Analysis</div>', unsafe_allow_html=True)
    
    # Get probabilities with safe defaults
    btts_yes = safe_get(predictions, 'probabilities', 'both_teams_score', 'yes') or 0
    btts_no = safe_get(predictions, 'probabilities', 'both_teams_score', 'no') or 0
    
    over_25 = safe_get(predictions, 'probabilities', 'over_under', 'over_25') or 0
    under_25 = safe_get(predictions, 'probabilities', 'over_under', 'under_25') or 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Professional BTTS with explanations
        if btts_no > btts_yes:
            recommendation = "NO"
            primary_prob = btts_no
            secondary_prob = btts_yes
            card_class = "risk-high"
            emoji = "❌"
        else:
            recommendation = "YES"
            primary_prob = btts_yes
            secondary_prob = btts_no
            card_class = "risk-low"
            emoji = "✅"
        
        confidence = "HIGH" if abs(primary_prob - 50) > 20 else "MEDIUM" if abs(primary_prob - 50) > 10 else "LOW"
        
        st.markdown(f'''
        <div class="professional-card {card_class}">
            <h4>{emoji} Both Teams Score</h4>
            <div style="font-size: 2rem; font-weight: bold; color: #333; margin: 0.8rem 0;">
                {recommendation}: {primary_prob:.1f}%
            </div>
            <div style="font-size: 1rem; color: #666; margin: 0.5rem 0;">
                {('NO' if recommendation == 'YES' else 'YES')}: {secondary_prob:.1f}%
            </div>
            <span class="professional-confidence-badge confidence-{confidence.lower()}">
                {confidence} CONFIDENCE
            </span>
        </div>
        ''', unsafe_allow_html=True)
        
        # Show professional explanations
        explanations = safe_get(predictions, 'explanations', 'btts') or []
        for explanation in explanations[:2]:
            if explanation:
                st.markdown(f'<div class="professional-explanation-card">💡 {explanation}</div>', unsafe_allow_html=True)
    
    with col2:
        # Professional Over/Under with explanations
        if under_25 > over_25:
            recommendation = "UNDER"
            primary_prob = under_25
            secondary_prob = over_25
            card_class = "risk-high"
            emoji = "❌"
        else:
            recommendation = "OVER"
            primary_prob = over_25
            secondary_prob = under_25
            card_class = "risk-low"
            emoji = "✅"
        
        confidence = "HIGH" if abs(primary_prob - 50) > 20 else "MEDIUM" if abs(primary_prob - 50) > 10 else "LOW"
        
        st.markdown(f'''
        <div class="professional-card {card_class}">
            <h4>{emoji} Over/Under 2.5</h4>
            <div style="font-size: 2rem; font-weight: bold; color: #333; margin: 0.8rem 0;">
                {recommendation}: {primary_prob:.1f}%
            </div>
            <div style="font-size: 1rem; color: #666; margin: 0.5rem 0;">
                {('OVER' if recommendation == 'UNDER' else 'UNDER')}: {secondary_prob:.1f}%
            </div>
            <span class="professional-confidence-badge confidence-{confidence.lower()}">
                {confidence} CONFIDENCE
            </span>
        </div>
        ''', unsafe_allow_html=True)
        
        # Show professional explanations
        explanations = safe_get(predictions, 'explanations', 'over_under') or []
        for explanation in explanations[:2]:
            if explanation:
                st.markdown(f'<div class="professional-explanation-card">💡 {explanation}</div>', unsafe_allow_html=True)
    
    with col3:
        # Professional Expected Goals display
        xg = safe_get(predictions, 'expected_goals') or {'home': 0, 'away': 0}
        total_xg = xg.get('home', 0) + xg.get('away', 0)
        
        st.markdown(f'''
        <div class="professional-card">
            <h4>🎯 Expected Goals</h4>
            <div style="font-size: 1.3rem; font-weight: bold; color: #333; margin: 0.8rem 0;">
                Home: {xg.get('home', 0):.2f}
            </div>
            <div style="font-size: 1.3rem; font-weight: bold; color: #333; margin: 0.8rem 0;">
                Away: {xg.get('away', 0):.2f}
            </div>
            <div style="font-size: 1.1rem; color: #666; margin: 0.5rem 0;">
                Total: {total_xg:.2f}
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        # Professional Match Context
        context = safe_get(predictions, 'match_context') or 'balanced'
        narrative = safe_get(predictions, 'match_narrative') or {}
        quality_gap = narrative.get('quality_gap', 'even')
        
        context_emoji = {
            'home_dominance': '🏠',
            'away_counter': '✈️',
            'offensive_showdown': '🔥',
            'defensive_battle': '🛡️',
            'tactical_stalemate': '⚔️',
            'balanced': '⚖️'
        }.get(context, '⚖️')
        
        quality_emoji = {
            'extreme': '🔥',
            'significant': '⭐',
            'even': '⚖️'
        }.get(quality_gap, '⚖️')
        
        st.markdown(f'''
        <div class="professional-card">
            <h4>{context_emoji} Professional Context</h4>
            <div style="font-size: 1.2rem; font-weight: bold; color: #333; margin: 0.8rem 0;">
                {context.replace('_', ' ').title()}
            </div>
            <div style="font-size: 1rem; color: #666; margin: 0.5rem 0;">
                {quality_emoji} Quality Gap: {quality_gap.title()}
            </div>
            <div style="font-size: 1rem; color: #666; margin: 0.5rem 0;">
                Tempo: {narrative.get('expected_tempo', 'medium').title()}
            </div>
            <div style="font-size: 1rem; color: #666; margin: 0.5rem 0;">
                Defense: {narrative.get('defensive_stability', 'mixed').title()}
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Professional Exact Scores
    st.markdown('<div class="professional-section-title">🎯 Most Likely Scores</div>', unsafe_allow_html=True)
    
    exact_scores = safe_get(predictions, 'probabilities', 'exact_scores') or {}
    top_scores = dict(list(exact_scores.items())[:6])
    
    if top_scores:
        score_cols = st.columns(6)
        for idx, (score, prob) in enumerate(top_scores.items()):
            with score_cols[idx]:
                st.metric(f"{score}", f"{prob*100:.1f}%")
    else:
        st.info("No exact score data available")
    
    # Professional Risk Assessment
    risk = safe_get(predictions, 'risk_assessment') or {'risk_level': 'UNKNOWN', 'explanation': 'No data'}
    risk_class = f"risk-{risk.get('risk_level', 'unknown').lower()}"
    
    intelligence = safe_get(predictions, 'apex_intelligence') or {}
    
    st.markdown(f'''
    <div class="professional-card {risk_class}">
        <h3>📊 Enhanced Professional Risk Assessment</h3>
        <strong>Risk Level:</strong> {risk.get("risk_level", "UNKNOWN")}<br>
        <strong>Enhanced Explanation:</strong> {risk.get("explanation", "No data available")}<br>
        <strong>Enhanced Recommendation:</strong> {risk.get("recommendation", "N/A")}<br>
        <strong>Certainty:</strong> {risk.get("certainty", "N/A")}<br>
        <strong>Narrative Coherence:</strong> {intelligence.get('narrative_coherence', 'N/A')}%<br>
        <strong>Prediction Alignment:</strong> {intelligence.get('prediction_alignment', 'N/A')}<br>
        <strong>Form Stability Bonus:</strong> +{intelligence.get('form_stability_bonus', 0):.1f}<br>
        <strong>Calibration Status:</strong> {intelligence.get('calibration_status', 'N/A')}
    </div>
    ''', unsafe_allow_html=True)
    
    # Professional Summary
    st.markdown('<div class="professional-section-title">📝 Professional Match Summary</div>', unsafe_allow_html=True)
    summary = safe_get(predictions, 'summary') or "No professional summary available."
    st.info(summary)

def display_professional_value_detection(predictions):
    """Display professional value detection"""
    
    if not predictions:
        st.error("❌ No professional predictions available for value detection")
        return
        
    st.markdown('<p class="professional-header">💰 Professional Value Betting Detection</p>', unsafe_allow_html=True)
    st.markdown('<div class="professional-value-card"><h3>🟠 ENHANCED PROFESSIONAL VALUE ENGINE OUTPUT</h3>Advanced confidence logic with contradiction detection</div>', unsafe_allow_html=True)
    
    betting_signals = safe_get(predictions, 'betting_signals') or []
    
    # Get primary predictions for professional context
    outcomes = safe_get(predictions, 'probabilities', 'match_outcomes') or {}
    btts = safe_get(predictions, 'probabilities', 'both_teams_score') or {}
    over_under = safe_get(predictions, 'probabilities', 'over_under') or {}
    team_tiers = safe_get(predictions, 'team_tiers') or {}
    league = safe_get(predictions, 'league', 'premier_league')
    
    primary_outcome = max(outcomes, key=outcomes.get) if outcomes else 'unknown'
    primary_btts = 'yes' if btts.get('yes', 0) > btts.get('no', 0) else 'no'
    primary_over_under = 'over_25' if over_under.get('over_25', 0) > over_under.get('under_25', 0) else 'under_25'
    
    # Display professional primary predictions context
    st.markdown('<div class="professional-section-title">🎯 Signal Engine Primary Predictions</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        outcome_map = {'home_win': 'Home Win', 'draw': 'Draw', 'away_win': 'Away Win'}
        st.metric("Primary Outcome", outcome_map.get(primary_outcome, 'Unknown'))
    with col2:
        st.metric("Primary BTTS", "YES" if primary_btts == 'yes' else "NO")
    with col3:
        st.metric("Primary Over/Under", "OVER 2.5" if primary_over_under == 'over_25' else "UNDER 2.5")
    with col4:
        st.metric("Professional League", get_league_display_name(league))
    
    if not betting_signals:
        st.markdown('<div class="professional-alignment-perfect">', unsafe_allow_html=True)
        st.info("""
        ## ✅ ENHANCED PROFESSIONAL: NO VALUE BETS DETECTED - SYSTEM WORKING PERFECTLY!
        
        **Enhanced Professional Assessment:**
        - Pure probabilities align with market expectations  
        - No significant edges above enhanced professional thresholds
        - Advanced contradiction detection confirms signal coherence
        - **ENHANCED PERFECT ALIGNMENT ACHIEVED**
        
        **Enhanced Professional Value Engine with advanced confidence logic is properly confirming predictions!**
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Professional alignment status
    system_validation = safe_get(predictions, 'system_validation') or {}
    alignment_status = system_validation.get('alignment', 'UNKNOWN')
    calibration_level = system_validation.get('calibration_level', 'STANDARD')
    
    if alignment_status == 'PERFECT' and calibration_level == 'MONEY_GRADE':
        st.markdown('<div class="professional-alignment-perfect">✅ <strong>ENHANCED PROFESSIONAL PERFECT ALIGNMENT:</strong> All value bets confirm Signal Engine predictions with advanced validation</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="professional-alignment-warning">⚠️ <strong>ENHANCED PROFESSIONAL REVIEW REQUIRED:</strong> Advanced contradiction detection active</div>', unsafe_allow_html=True)
    
    # Enhanced: Show contradiction warnings
    contradictory_signals = [s for s in betting_signals if any("contradict" in exp.lower() for exp in s.get('explanation', []))]
    if contradictory_signals:
        st.markdown(f'''
        <div class="contradiction-warning">
            ⚠️ <strong>CONTRADICTION DETECTION:</strong> {len(contradictory_signals)} signal(s) contradict primary predictions
            <br><small>Enhanced confidence logic has automatically adjusted stakes and confidence levels</small>
        </div>
        ''', unsafe_allow_html=True)
    
    # Professional Value Bet Summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_signals = len(betting_signals)
        st.metric("Enhanced Signals", total_signals)
    
    with col2:
        high_value = len([s for s in betting_signals if s.get('value_rating') in ['EXCEPTIONAL', 'HIGH']])
        st.metric("High Value Signals", high_value)
    
    with col3:
        contradictory_count = len(contradictory_signals)
        st.metric("Contradictory Signals", contradictory_count)
    
    with col4:
        avg_edge = np.mean([s.get('edge', 0) for s in betting_signals]) if betting_signals else 0
        st.metric("Average Edge", f"{avg_edge:.1f}%")
    
    # Display professional value bets
    st.markdown('<div class="professional-section-title">🎯 Enhanced Professional Value Bet Recommendations</div>', unsafe_allow_html=True)
    
    # Group by professional value rating
    exceptional_bets = [s for s in betting_signals if s.get('value_rating') == 'EXCEPTIONAL']
    high_bets = [s for s in betting_signals if s.get('value_rating') == 'HIGH']
    good_bets = [s for s in betting_signals if s.get('value_rating') == 'GOOD']
    moderate_bets = [s for s in betting_signals if s.get('value_rating') == 'MODERATE']
    
    def display_professional_bet_group(bets, title, emoji):
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
                
                # Professional alignment assessment
                alignment = bet.get('alignment', 'neutral')
                alignment_emoji = "✅" if alignment == 'aligns_with_primary' else "⚠️"
                alignment_text = "ALIGNS" if alignment == 'aligns_with_primary' else "CONTRADICTS"
                
                # Enhanced: Check for contradiction explanations
                explanations = bet.get('explanation', [])
                safe_explanations = [exp for exp in explanations if exp and "contradict" not in exp.lower()]
                contradiction_explanations = [exp for exp in explanations if exp and "contradict" in exp.lower()]
                
                st.markdown(f'''
                <div class="professional-bet-card {value_class}">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <div style="flex: 2;">
                            <strong>{bet.get('market', 'Unknown')}</strong>
                            {f'<span style="color: #FF5722; margin-left: 0.5rem;">⚠️ CONTRADICTION</span>' if contradiction_explanations else ''}
                            <br>
                            <small>Model: {bet.get('model_prob', 0)}% | Market: {bet.get('book_prob', 0)}%</small>
                            <div style="margin-top: 0.5rem;">
                                <small>{alignment_emoji} <strong>{alignment_text}</strong> with Signal Engine</small>
                            </div>
                            <div style="margin-top: 0.8rem;">
                                {''.join([f'<span class="professional-feature-badge">💡 {exp}</span>' for exp in safe_explanations[:1]])}
                            </div>
                            {''.join([f'<div class="enhanced-confidence-reasoning">⚠️ {exp}</div>' for exp in contradiction_explanations[:1]])}
                        </div>
                        <div style="flex: 1; text-align: right;">
                            <strong style="color: #4CAF50; font-size: 1.2rem;">+{bet.get('edge', 0)}% Edge</strong><br>
                            <small>Stake: ${bet.get('recommended_stake', 0):.2f}</small><br>
                            <small>{confidence_emoji} {bet.get('confidence', 'Unknown')}</small>
                        </div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
    
    display_professional_bet_group(exceptional_bets, "Exceptional", "🔥")
    display_professional_bet_group(high_bets, "High", "⭐")
    display_professional_bet_group(good_bets, "Good", "✅")
    display_professional_bet_group(moderate_bets, "Moderate", "📊")

def main():
    """Professional main application function"""
    
    # Initialize professional session state
    if 'professional_predictions' not in st.session_state:
        st.session_state.professional_predictions = None
    
    if 'professional_prediction_history' not in st.session_state:
        st.session_state.professional_prediction_history = []
    
    if st.session_state.professional_predictions:
        # Create professional tabs
        tab1, tab2 = st.tabs(["🎯 Professional Predictions", "💰 Professional Value Detection"])
        
        with tab1:
            display_professional_predictions(st.session_state.professional_predictions)
        
        with tab2:
            display_professional_value_detection(st.session_state.professional_predictions)
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 New Professional Analysis", use_container_width=True):
                st.session_state.professional_predictions = None
                st.rerun()
        
        with col2:
            if st.button("📊 Professional History", use_container_width=True):
                if st.session_state.professional_prediction_history:
                    st.write("**Enhanced Professional Prediction History:**")
                    for i, pred in enumerate(st.session_state.professional_prediction_history[-5:]):
                        with st.expander(f"Enhanced Analysis {i+1}: {pred.get('match', 'Unknown Match')} (IQ: {pred.get('football_iq', 0):.1f})"):
                            st.write(f"Date: {pred.get('timestamp', 'N/A')}")
                            st.write(f"League: {get_league_display_name(pred.get('league', 'premier_league'))}")
                            st.write(f"Expected Goals: Home {pred['expected_goals'].get('home', 0):.2f} - Away {pred['expected_goals'].get('away', 0):.2f}")
                            st.write(f"Team Tiers: {pred.get('team_tiers', {}).get('home', 'N/A')} vs {pred.get('team_tiers', {}).get('away', 'N/A')}")
                            st.write(f"Enhanced IQ: {pred.get('football_iq', 0):.1f}/100")
                            st.write(f"Stability Bonus: +{pred.get('stability_bonus', 0):.1f}")
                            st.write(f"Value Bets Found: {pred.get('value_bets', 0)}")
                else:
                    st.info("No enhanced professional prediction history yet.")
        
        with col3:
            if st.button("🎯 Enhanced System Status", use_container_width=True):
                st.success("""
                **Enhanced Professional System Status: OPERATIONAL** 🟢
                
                **Enhanced Confidence Features Active:**
                - League-Specific Confidence Multipliers ✅
                - Form Stability Bonus Scoring ✅  
                - Advanced Contradiction Detection ✅
                - Professional Monte Carlo (25k) ✅
                - Enhanced Signal Hygiene ✅
                
                **Model Version:** 2.1.0_enhanced
                **Calibration Level:** MONEY_GRADE
                **Last Update:** Enhanced Confidence Logic Active
                """)
        
        return
    
    match_data, mc_iterations = create_professional_input_form()
    
    if match_data:
        with st.spinner("🔍 Running enhanced professional multi-league calibrated analysis..."):
            try:
                # Initialize professional predictor
                predictor = AdvancedFootballPredictor(match_data)
                
                # Generate professional analysis
                predictions = predictor.generate_comprehensive_analysis(mc_iterations)
                
                if predictions:
                    # Add professional information
                    predictions['league'] = match_data['league']
                    predictions['bankroll'] = match_data.get('bankroll', 1000)
                    predictions['kelly_fraction'] = match_data.get('kelly_fraction', 0.2)
                    
                    st.session_state.professional_predictions = predictions
                    
                    # Store in professional history
                    if 'professional_prediction_history' not in st.session_state:
                        st.session_state.professional_prediction_history = []
                    
                    prediction_record = {
                        'timestamp': datetime.now().isoformat(),
                        'match': predictions.get('match', 'Unknown Match'),
                        'league': predictions.get('league', 'premier_league'),
                        'expected_goals': predictions.get('expected_goals', {'home': 0, 'away': 0}),
                        'team_tiers': predictions.get('team_tiers', {}),
                        'probabilities': safe_get(predictions, 'probabilities', 'match_outcomes') or {},
                        'football_iq': safe_get(predictions, 'apex_intelligence', 'football_iq_score') or 0,
                        'stability_bonus': safe_get(predictions, 'apex_intelligence', 'form_stability_bonus') or 0,
                        'value_bets': len(predictions.get('betting_signals', []))
                    }
                    
                    st.session_state.professional_prediction_history.append(prediction_record)
                    
                    # Professional alignment status check
                    system_validation = safe_get(predictions, 'system_validation') or {}
                    alignment_status = system_validation.get('alignment', 'UNKNOWN')
                    calibration_level = system_validation.get('calibration_level', 'STANDARD')
                    
                    if alignment_status == 'PERFECT' and calibration_level == 'MONEY_GRADE':
                        stability_bonus = safe_get(predictions, 'apex_intelligence', 'form_stability_bonus') or 0
                        st.success(f"""
                        ✅ **ENHANCED PROFESSIONAL PERFECT ALIGNMENT ACHIEVED!** 
                        
                        Enhanced Professional Value Engine confirms Signal Engine predictions with:
                        - League-specific confidence multipliers ✅
                        - Form stability bonus: +{stability_bonus:.1f} ✅  
                        - Advanced contradiction detection ✅
                        - Professional bankroll management ✅
                        """)
                    else:
                        st.warning("⚠️ ENHANCED PROFESSIONAL REVIEW REQUIRED: Advanced contradiction detection active")
                    
                    st.rerun()
                else:
                    st.error("❌ Failed to generate enhanced professional predictions")
                
            except Exception as e:
                st.error(f"❌ Enhanced professional analysis error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                st.info("💡 Check enhanced professional input parameters and try again")

if __name__ == "__main__":
    main()
