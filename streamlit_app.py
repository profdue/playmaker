# streamlit_app.py - ENHANCED PROFESSIONAL BETTING GRADE
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from typing import Dict, Any
from datetime import datetime

# Import the ENHANCED PROFESSIONAL PREDICTION ENGINE
try:
    from prediction_engine import AdvancedFootballPredictor, ProfessionalTeamTierCalibrator
except ImportError as e:
    st.error(f"❌ Could not import prediction_engine: {str(e)}")
    st.info("💡 Make sure prediction_engine.py is in the same directory")
    st.stop()

# Professional page configuration
st.set_page_config(
    page_title="🎯 Enhanced Professional Football Predictor",
    page_icon="⚽", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Professional CSS styling
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
    .enhanced-badge {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
        box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
    }
    .contradiction-warning {
        background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #FF5722;
    }
    .stability-bonus {
        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
    .league-multiplier {
        background: linear-gradient(135deg, #9C27B0 0%, #7B1FA2 100%);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 12px;
        font-size: 0.7rem;
        margin-left: 0.3rem;
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

def display_enhanced_banner():
    """Display enhanced professional banner"""
    st.markdown("""
    <div class="money-grade-banner">
        🚀 ENHANCED PROFESSIONAL BETTING GRADE • LEAGUE-SPECIFIC CONFIDENCE • FORM STABILITY BONUS • CONTRADICTION DETECTION
    </div>
    """, unsafe_allow_html=True)

def display_enhanced_architecture():
    """Display enhanced system architecture"""
    with st.expander("🏗️ ENHANCED SYSTEM ARCHITECTURE", expanded=True):
        st.markdown("""
        ### 🎯 ENHANCED MONEY-GRADE PREDICTION ENGINE
        
        **New Professional Features:**
        - **League-Specific Confidence Multipliers** - Dynamic threshold adjustment per league
        - **Form Stability Bonus** - Rewards consistent team performance  
        - **Contradiction Detection** - Automatically detects conflicting signals
        - **Enhanced Confidence Reasoning** - Transparent confidence assignment
        - **Signal Hygiene** - Prevents overconfidence in contradictory scenarios
        
        **Enhanced League Calibration** 🌍
        - **Serie A** 🇮🇹: 15% higher confidence requirements (defensive league)
        - **Bundesliga** 🇩🇪: 10% lower confidence requirements (high-scoring)
        - **Championship** 🏴󠁧󠁢󠁥󠁮󠁧󠁿: 8% higher requirements (unpredictable)
        - **All Leagues**: Custom confidence thresholds based on league characteristics
        """)

def create_enhanced_input_form():
    """Create enhanced professional input form"""
    
    st.markdown('<p class="professional-header">🚀 Enhanced Professional Football Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="professional-subheader">Money-Grade Multi-League Analysis with Enhanced Intelligence</p>', unsafe_allow_html=True)
    
    display_enhanced_banner()
    display_enhanced_architecture()
    
    st.markdown("### 🌍 Enhanced Professional League Selection")
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
        key="enhanced_league_selection"
    )
    
    league_badge_class = get_league_badge(selected_league)
    league_display_name = get_league_display_name(selected_league)
    st.markdown(f'<span class="professional-badge {league_badge_class}">{league_display_name}</span>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🏠 Enhanced Data", "💰 Market Data", "⚙️ Enhanced Settings"])

    with tab1:
        st.markdown("### 🎯 Enhanced Football Data")
        
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
                key="enhanced_home_team"
            )
            
            home_goals = st.number_input("Total Goals (Last 6 Games)", min_value=0, value=8, key="enhanced_home_goals")
            home_conceded = st.number_input("Total Conceded (Last 6 Games)", min_value=0, value=3, key="enhanced_home_conceded")
            home_goals_home = st.number_input("Home Goals (Last 3 Home Games)", min_value=0, value=5, key="enhanced_home_goals_home")
            
        with col2:
            st.subheader("✈️ Away Team")
            away_team = st.selectbox(
                "Team Name",
                options=league_teams,
                index=0,
                key="enhanced_away_team"
            )
            
            away_goals = st.number_input("Total Goals (Last 6 Games)", min_value=0, value=5, key="enhanced_away_goals")
            away_conceded = st.number_input("Total Conceded (Last 6 Games)", min_value=0, value=9, key="enhanced_away_conceded")
            away_goals_away = st.number_input("Away Goals (Last 3 Away Games)", min_value=0, value=4, key="enhanced_away_goals_away")
        
        home_tier = calibrator.get_team_tier(home_team, selected_league)
        away_tier = calibrator.get_team_tier(away_team, selected_league)
        
        st.markdown(f"""
        **Enhanced Team Assessment:** 
        <span class="professional-tier-badge tier-{home_tier.lower() if home_tier else 'medium'}">{home_tier or 'MEDIUM'}</span> vs 
        <span class="professional-tier-badge tier-{away_tier.lower() if away_tier else 'medium'}">{away_tier or 'MEDIUM'}</span>
        """, unsafe_allow_html=True)
        
        with st.expander("📊 Enhanced Head-to-Head Analysis"):
            h2h_col1, h2h_col2, h2h_col3 = st.columns(3)
            with h2h_col1:
                h2h_matches = st.number_input("Total H2H Matches", min_value=0, value=4, key="enhanced_h2h_matches")
                h2h_home_wins = st.number_input("Home Wins", min_value=0, value=3, key="enhanced_h2h_home_wins")
            with h2h_col2:
                h2h_away_wins = st.number_input("Away Wins", min_value=0, value=0, key="enhanced_h2h_away_wins")
                h2h_draws = st.number_input("Draws", min_value=0, value=1, key="enhanced_h2h_draws")
            with h2h_col3:
                h2h_home_goals = st.number_input("Home Goals in H2H", min_value=0, value=8, key="enhanced_h2h_home_goals")
                h2h_away_goals = st.number_input("Away Goals in H2H", min_value=0, value=2, key="enhanced_h2h_away_goals")

        with st.expander("📈 Enhanced Form Analysis"):
            st.info("Enhanced form analysis includes stability scoring for confidence bonuses")
            form_col1, form_col2 = st.columns(2)
            with form_col1:
                st.write(f"**{home_team} Last 6 Matches**")
                home_form = st.multiselect(
                    f"{home_team} Recent Results",
                    options=["Win (3 pts)", "Draw (1 pt)", "Loss (0 pts)"],
                    default=["Win (3 pts)", "Win (3 pts)", "Win (3 pts)", "Win (3 pts)", "Draw (1 pt)", "Win (3 pts)"],
                    key="enhanced_home_form"
                )
            with form_col2:
                st.write(f"**{away_team} Last 6 Matches**")
                away_form = st.multiselect(
                    f"{away_team} Recent Results", 
                    options=["Win (3 pts)", "Draw (1 pt)", "Loss (0 pts)"],
                    default=["Loss (0 pts)", "Draw (1 pt)", "Loss (0 pts)", "Draw (1 pt)", "Loss (0 pts)", "Loss (0 pts)"],
                    key="enhanced_away_form"
                )

    with tab2:
        st.markdown("### 💰 Enhanced Market Data") 
        
        odds_col1, odds_col2, odds_col3 = st.columns(3)
        
        with odds_col1:
            st.write("**1X2 Market**")
            home_odds = st.number_input("Home Win Odds", min_value=1.01, value=1.45, step=0.01, key="enhanced_home_odds")
            draw_odds = st.number_input("Draw Odds", min_value=1.01, value=4.20, step=0.01, key="enhanced_draw_odds")
            away_odds = st.number_input("Away Win Odds", min_value=1.01, value=8.50, step=0.01, key="enhanced_away_odds")
        
        with odds_col2:
            st.write("**Over/Under Markets**")
            over_15_odds = st.number_input("Over 1.5 Goals", min_value=1.01, value=1.36, step=0.01, key="enhanced_over_15_odds")
            over_25_odds = st.number_input("Over 2.5 Goals", min_value=1.01, value=1.95, step=0.01, key="enhanced_over_25_odds")
            over_35_odds = st.number_input("Over 3.5 Goals", min_value=1.01, value=3.50, step=0.01, key="enhanced_over_35_odds")
        
        with odds_col3:
            st.write("**Both Teams to Score**")
            btts_yes_odds = st.number_input("BTTS Yes", min_value=1.01, value=2.20, step=0.01, key="enhanced_btts_yes_odds")
            btts_no_odds = st.number_input("BTTS No", min_value=1.01, value=1.65, step=0.01, key="enhanced_btts_no_odds")

    with tab3:
        st.markdown("### ⚙️ Enhanced Configuration")
        
        model_col1, model_col2 = st.columns(2)
        
        with model_col1:
            st.write("**Enhanced Team Context**")
            home_injuries = st.slider("Home Key Absences", 0, 5, 2, key="enhanced_home_injuries")
            away_injuries = st.slider("Away Key Absences", 0, 5, 3, key="enhanced_away_injuries")
            
            home_absence_impact = st.select_slider(
                "Home Team Absence Impact",
                options=["Rotation Player", "Regular Starter", "Key Player", "Star Player", "Multiple Key Players"],
                value="Regular Starter",
                key="enhanced_home_absence_impact"
            )
            away_absence_impact = st.select_slider(
                "Away Team Absence Impact",
                options=["Rotation Player", "Regular Starter", "Key Player", "Star Player", "Multiple Key Players"],
                value="Regular Starter",
                key="enhanced_away_absence_impact"
            )
            
        with model_col2:
            st.write("**Enhanced Motivation Factors**")
            home_motivation = st.select_slider(
                "Home Team Motivation",
                options=["Low", "Normal", "High", "Very High"],
                value="Normal",
                key="enhanced_home_motivation"
            )
            away_motivation = st.select_slider(
                "Away Team Motivation", 
                options=["Low", "Normal", "High", "Very High"],
                value="Normal", 
                key="enhanced_away_motivation"
            )
            
            st.write("**Enhanced Simulation**")
            mc_iterations = st.select_slider(
                "Monte Carlo Iterations",
                options=[10000, 25000, 50000],
                value=25000,
                key="enhanced_mc_iterations"
            )
            
            bankroll = st.number_input("Enhanced Bankroll ($)", min_value=500, value=1000, step=100, key="enhanced_bankroll")
            kelly_fraction = st.slider("Enhanced Kelly Fraction", 0.1, 0.3, 0.2, key="enhanced_kelly_fraction")

    submitted = st.button("🚀 GENERATE ENHANCED ANALYSIS", type="primary", use_container_width=True)
    
    if submitted:
        if not home_team or not away_team:
            st.error("❌ Please enter both team names")
            return None, None
        
        if home_team == away_team:
            st.error("❌ Home and away teams cannot be the same")
            return None, None
        
        form_map = {"Win (3 pts)": 3, "Draw (1 pt)": 1, "Loss (0 pts)": 0}
        home_form_points = [form_map[result] for result in home_form]
        away_form_points = [form_map[result] for result in away_form]
        
        motivation_map = {"Low": "Low", "Normal": "Normal", "High": "High", "Very High": "Very High"}
        
        absence_impact_map = {
            "Rotation Player": 1,
            "Regular Starter": 2,
            "Key Player": 3, 
            "Star Player": 4,
            "Multiple Key Players": 5
        }
        
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

def display_enhanced_predictions(predictions):
    """Display enhanced professional predictions"""
    
    if not predictions:
        st.error("❌ No enhanced predictions available")
        return
        
    st.markdown('<p class="professional-header">🚀 Enhanced Professional Football Predictions</p>', unsafe_allow_html=True)
    st.markdown('<div class="professional-system-card"><h3>🟢 ENHANCED SIGNAL ENGINE OUTPUT</h3>Money-Grade Multi-League Analysis with Advanced Features</div>', unsafe_allow_html=True)
    
    team_tiers = safe_get(predictions, 'team_tiers') or {}
    home_tier = team_tiers.get('home', 'MEDIUM')
    away_tier = team_tiers.get('away', 'MEDIUM')
    
    league = safe_get(predictions, 'league', default='premier_league')
    league_display_name = get_league_display_name(league)
    league_badge_class = get_league_badge(league)
    
    # Enhanced header with stability bonus
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
        <span class="league-multiplier">Enhanced Confidence</span>
    </p>
    ''', unsafe_allow_html=True)
    
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
        st.metric("Enhanced Context", f"{context_emoji} {match_context.replace('_', ' ').title()}")
    with col4:
        st.metric("Enhanced IQ", f"{football_iq:.1f}/100")
    
    system_validation = safe_get(predictions, 'system_validation') or {}
    alignment_status = system_validation.get('alignment', 'UNKNOWN')
    calibration_level = system_validation.get('calibration_level', 'STANDARD')
    model_version = system_validation.get('model_version', '2.0.0')
    
    if alignment_status == 'PERFECT' and calibration_level == 'MONEY_GRADE':
        st.markdown(f'''
        <div class="professional-alignment-perfect">
            ✅ <strong>ENHANCED PERFECT ALIGNMENT:</strong> Advanced Value Engine confirms Signal Engine predictions
            <br><small>Model: {model_version} | Calibration: {calibration_level} | Stability Bonus: +{stability_bonus:.1f}</small>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="professional-alignment-warning">
            ⚠️ <strong>ENHANCED REVIEW REQUIRED:</strong> Advanced contradiction detection active
            <br><small>Model: {model_version} | Enhanced discretion advised</small>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('<div class="professional-section-title">📈 Enhanced Outcome Probabilities</div>', unsafe_allow_html=True)
    
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
    
    st.markdown('<div class="professional-section-title">⚽ Enhanced Goals Analysis</div>', unsafe_allow_html=True)
    
    btts_yes = safe_get(predictions, 'probabilities', 'both_teams_score', 'yes') or 0
    btts_no = safe_get(predictions, 'probabilities', 'both_teams_score', 'no') or 0
    
    over_25 = safe_get(predictions, 'probabilities', 'over_under', 'over_25') or 0
    under_25 = safe_get(predictions, 'probabilities', 'over_under', 'under_25') or 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
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
        
        explanations = safe_get(predictions, 'explanations', 'btts') or []
        for explanation in explanations[:2]:
            if explanation:
                st.markdown(f'<div class="professional-explanation-card">💡 {explanation}</div>', unsafe_allow_html=True)
    
    with col2:
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
        
        explanations = safe_get(predictions, 'explanations', 'over_under') or []
        for explanation in explanations[:2]:
            if explanation:
                st.markdown(f'<div class="professional-explanation-card">💡 {explanation}</div>', unsafe_allow_html=True)
    
    with col3:
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
            <h4>{context_emoji} Enhanced Context</h4>
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
    
    risk = safe_get(predictions, 'risk_assessment') or {'risk_level': 'UNKNOWN', 'explanation': 'No data'}
    risk_class = f"risk-{risk.get('risk_level', 'unknown').lower()}"
    
    intelligence = safe_get(predictions, 'apex_intelligence') or {}
    
    st.markdown(f'''
    <div class="professional-card {risk_class}">
        <h3>📊 Enhanced Risk Assessment</h3>
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
    
    st.markdown('<div class="professional-section-title">📝 Enhanced Match Summary</div>', unsafe_allow_html=True)
    summary = safe_get(predictions, 'summary') or "No enhanced summary available."
    st.info(summary)

def display_enhanced_value_detection(predictions):
    """Display enhanced professional value detection"""
    
    if not predictions:
        st.error("❌ No enhanced predictions available for value detection")
        return
        
    st.markdown('<p class="professional-header">💰 Enhanced Professional Value Betting Detection</p>', unsafe_allow_html=True)
    st.markdown('<div class="professional-value-card"><h3>🟠 ENHANCED VALUE ENGINE OUTPUT</h3>Advanced contradiction detection and league-specific confidence</div>', unsafe_allow_html=True)
    
    betting_signals = safe_get(predictions, 'betting_signals') or []
    
    outcomes = safe_get(predictions, 'probabilities', 'match_outcomes') or {}
    btts = safe_get(predictions, 'probabilities', 'both_teams_score') or {}
    over_under = safe_get(predictions, 'probabilities', 'over_under') or {}
    team_tiers = safe_get(predictions, 'team_tiers') or {}
    league = safe_get(predictions, 'league', 'premier_league')
    intelligence = safe_get(predictions, 'apex_intelligence') or {}
    
    primary_outcome = max(outcomes, key=outcomes.get) if outcomes else 'unknown'
    primary_btts = 'yes' if btts.get('yes', 0) > btts.get('no', 0) else 'no'
    primary_over_under = 'over_25' if over_under.get('over_25', 0) > over_under.get('under_25', 0) else 'under_25'
    stability_bonus = intelligence.get('form_stability_bonus', 0)
    
    st.markdown('<div class="professional-section-title">🎯 Enhanced Signal Engine Primary Predictions</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        outcome_map = {'home_win': 'Home Win', 'draw': 'Draw', 'away_win': 'Away Win'}
        st.metric("Primary Outcome", outcome_map.get(primary_outcome, 'Unknown'))
    with col2:
        st.metric("Primary BTTS", "YES" if primary_btts == 'yes' else "NO")
    with col3:
        st.metric("Primary Over/Under", "OVER 2.5" if primary_over_under == 'over_25' else "UNDER 2.5")
    with col4:
        st.metric("Enhanced League", get_league_display_name(league))
    with col5:
        st.metric("Stability Bonus", f"+{stability_bonus:.1f}")
    
    if not betting_signals:
        st.markdown('<div class="professional-alignment-perfect">', unsafe_allow_html=True)
        st.info("""
        ## ✅ ENHANCED: NO VALUE BETS DETECTED - ADVANCED SYSTEM WORKING PERFECTLY!
        
        **Enhanced Assessment:**
        - Pure probabilities align with market expectations  
        - No significant edges above enhanced professional thresholds
        - Advanced contradiction detection confirms signal coherence
        - **ENHANCED PERFECT ALIGNMENT ACHIEVED**
        
        **Enhanced Value Engine with league-specific confidence and form stability is properly confirming predictions!**
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    system_validation = safe_get(predictions, 'system_validation') or {}
    alignment_status = system_validation.get('alignment', 'UNKNOWN')
    calibration_level = system_validation.get('calibration_level', 'STANDARD')
    
    if alignment_status == 'PERFECT' and calibration_level == 'MONEY_GRADE':
        st.markdown('<div class="professional-alignment-perfect">✅ <strong>ENHANCED PERFECT ALIGNMENT:</strong> All value bets confirm Signal Engine predictions with advanced validation</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="professional-alignment-warning">⚠️ <strong>ENHANCED REVIEW REQUIRED:</strong> Advanced contradiction detection active</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_signals = len(betting_signals)
        st.metric("Enhanced Signals", total_signals)
    
    with col2:
        high_value = len([s for s in betting_signals if s.get('value_rating') in ['EXCEPTIONAL', 'HIGH']])
        st.metric("High Value Signals", high_value)
    
    with col3:
        contradictory_signals = len([s for s in betting_signals if "contradicts" in s.get('alignment', '')])
        st.metric("Contradictory Signals", contradictory_signals)
    
    with col4:
        avg_edge = np.mean([s.get('edge', 0) for s in betting_signals]) if betting_signals else 0
        st.metric("Average Edge", f"{avg_edge:.1f}%")
    
    with col5:
        total_stake = np.sum([s.get('recommended_stake', 0) for s in betting_signals]) if betting_signals else 0
        st.metric("Total Stake", f"${total_stake:.2f}")
    
    st.markdown('<div class="professional-section-title">🎯 Enhanced Value Bet Recommendations</div>', unsafe_allow_html=True)
    
    exceptional_bets = [s for s in betting_signals if s.get('value_rating') == 'EXCEPTIONAL']
    high_bets = [s for s in betting_signals if s.get('value_rating') == 'HIGH']
    good_bets = [s for s in betting_signals if s.get('value_rating') == 'GOOD']
    moderate_bets = [s for s in betting_signals if s.get('value_rating') == 'MODERATE']
    
    def display_enhanced_bet_group(bets, title, emoji):
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
                
                alignment = bet.get('alignment', 'neutral')
                alignment_emoji = "✅" if alignment == 'aligns_with_primary' else "⚠️"
                alignment_text = "ALIGNS" if alignment == 'aligns_with_primary' else "CONTRADICTS"
                
                # Enhanced: Check for contradiction warnings
                has_contradiction = any("contradict" in exp.lower() for exp in bet.get('explanation', []))
                contradiction_warning = " 🚨 CONTRADICTION" if has_contradiction else ""
                
                explanations = bet.get('explanation', [])
                safe_explanations = [exp for exp in explanations if exp and "contradict" not in exp.lower()]
                contradiction_explanations = [exp for exp in explanations if exp and "contradict" in exp.lower()]
                
                st.markdown(f'''
                <div class="professional-bet-card {value_class}">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <div style="flex: 2;">
                            <strong>{bet.get('market', 'Unknown')}{contradiction_warning}</strong><br>
                            <small>Model: {bet.get('model_prob', 0)}% | Market: {bet.get('book_prob', 0)}%</small>
                            <div style="margin-top: 0.5rem;">
                                <small>{alignment_emoji} <strong>{alignment_text}</strong> with Signal Engine</small>
                            </div>
                            <div style="margin-top: 0.8rem;">
                                {''.join([f'<span class="professional-feature-badge">💡 {exp}</span>' for exp in safe_explanations[:1]])}
                            </div>
                            {''.join([f'<div style="color: #FF5722; font-size: 0.8rem; margin-top: 0.3rem;">⚠️ {exp}</div>' for exp in contradiction_explanations[:1]])}
                        </div>
                        <div style="flex: 1; text-align: right;">
                            <strong style="color: #4CAF50; font-size: 1.2rem;">+{bet.get('edge', 0)}% Edge</strong><br>
                            <small>Stake: ${bet.get('recommended_stake', 0):.2f}</small><br>
                            <small>{confidence_emoji} {bet.get('confidence', 'Unknown')}</small>
                        </div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
    
    display_enhanced_bet_group(exceptional_bets, "Exceptional", "🔥")
    display_enhanced_bet_group(high_bets, "High", "⭐")
    display_enhanced_bet_group(good_bets, "Good", "✅")
    display_enhanced_bet_group(moderate_bets, "Moderate", "📊")

def main():
    """Enhanced main application function"""
    
    if 'professional_predictions' not in st.session_state:
        st.session_state.professional_predictions = None
    
    if 'professional_prediction_history' not in st.session_state:
        st.session_state.professional_prediction_history = []
    
    if st.session_state.professional_predictions:
        tab1, tab2 = st.tabs(["🚀 Enhanced Predictions", "💰 Enhanced Value Detection"])
        
        with tab1:
            display_enhanced_predictions(st.session_state.professional_predictions)
        
        with tab2:
            display_enhanced_value_detection(st.session_state.professional_predictions)
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 New Enhanced Analysis", use_container_width=True):
                st.session_state.professional_predictions = None
                st.rerun()
        
        with col2:
            if st.button("📊 Enhanced History", use_container_width=True):
                if st.session_state.professional_prediction_history:
                    st.write("**Enhanced Prediction History:**")
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
                    st.info("No enhanced prediction history yet.")
        
        with col3:
            if st.button("🎯 Enhanced Status", use_container_width=True):
                st.success("""
                **Enhanced System Status: OPERATIONAL** 🟢
                
                **Enhanced Features Active:**
                - League-Specific Confidence Multipliers ✅
                - Form Stability Bonus Scoring ✅  
                - Advanced Contradiction Detection ✅
                - Enhanced Monte Carlo (25k) ✅
                - Signal Hygiene Protocols ✅
                
                **Model Version:** 2.1.0_enhanced
                **Calibration Level:** MONEY_GRADE
                **Last Update:** Enhanced Features Active
                """)
        
        return
    
    match_data, mc_iterations = create_enhanced_input_form()
    
    if match_data:
        with st.spinner("🔍 Running enhanced multi-league calibrated analysis..."):
            try:
                predictor = AdvancedFootballPredictor(match_data)
                predictions = predictor.generate_comprehensive_analysis(mc_iterations)
                
                if predictions:
                    predictions['league'] = match_data['league']
                    predictions['bankroll'] = match_data.get('bankroll', 1000)
                    predictions['kelly_fraction'] = match_data.get('kelly_fraction', 0.2)
                    
                    st.session_state.professional_predictions = predictions
                    
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
                    
                    system_validation = safe_get(predictions, 'system_validation') or {}
                    alignment_status = system_validation.get('alignment', 'UNKNOWN')
                    calibration_level = system_validation.get('calibration_level', 'STANDARD')
                    
                    if alignment_status == 'PERFECT' and calibration_level == 'MONEY_GRADE':
                        stability_bonus = safe_get(predictions, 'apex_intelligence', 'form_stability_bonus') or 0
                        st.success(f"""
                        ✅ **ENHANCED PERFECT ALIGNMENT ACHIEVED!** 
                        
                        Enhanced Value Engine with advanced features confirms predictions:
                        - League-specific confidence multipliers ✅
                        - Form stability bonus: +{stability_bonus:.1f} ✅  
                        - Advanced contradiction detection ✅
                        - Enhanced bankroll management ✅
                        """)
                    else:
                        st.warning("⚠️ ENHANCED REVIEW REQUIRED: Advanced contradiction detection active")
                    
                    st.rerun()
                else:
                    st.error("❌ Failed to generate enhanced predictions")
                
            except Exception as e:
                st.error(f"❌ Enhanced analysis error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                st.info("💡 Check enhanced input parameters and try again")

if __name__ == "__main__":
    main()
