# streamlit_app.py - PRODUCTION-READY ENHANCED PREDICTOR
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List
import json
from datetime import datetime

# Import from the production prediction engine
try:
    from prediction_engine import (
        ApexProductionEngine, EnhancedTeamTierCalibrator, 
        ProductionLeagueCalibrator, MarketAnalyzer
    )
except ImportError as e:
    st.error(f"❌ Import Error: {str(e)}")
    st.info("💡 Make sure prediction_engine.py is in the same directory and all class names match")
    st.stop()

# Clear cache to ensure fresh imports
st.cache_resource.clear()

st.set_page_config(
    page_title="🎯 Production Football Predictor",
    page_icon="⚽", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Production CSS
st.markdown("""
<style>
    .production-header { 
        font-size: 2.8rem !important; 
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .production-subheader {
        font-size: 1.4rem !important;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .production-badge {
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
    
    .production-mode-active {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
        border: 3px solid #FFD700;
    }
    
    .production-card { 
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
    
    .production-metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .value-opportunity-card {
        background: #e8f5e8;
        border-left: 5px solid #4CAF50;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.8rem 0;
    }
    
    .production-probability-bar {
        height: 12px;
        background: #e0e0e0;
        border-radius: 6px;
        margin: 0.8rem 0;
        overflow: hidden;
    }
    .production-probability-fill {
        height: 100%;
        background: linear-gradient(90deg, #4CAF50, #45a049);
        border-radius: 6px;
    }
    
    .uncertainty-indicator {
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.3rem 0.7rem;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-left: 0.5rem;
        border: 1px solid #2196F3;
    }
    
    .production-section-title {
        font-size: 1.6rem;
        font-weight: 600;
        color: #333;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.8rem;
        border-bottom: 3px solid #f0f2f6;
    }
    
    .edge-positive {
        background: #e8f5e8;
        color: #2E7D32;
        padding: 0.3rem 0.7rem;
        border-radius: 12px;
        font-size: 0.8rem;
        border: 1px solid #4CAF50;
    }
    .edge-negative {
        background: #ffebee;
        color: #c62828;
        padding: 0.3rem 0.7rem;
        border-radius: 12px;
        font-size: 0.8rem;
        border: 1px solid #f44336;
    }
    
    .stake-recommendation {
        background: #fff3e0;
        border-left: 4px solid #FF9800;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .production-feature-badge {
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.3rem 0.7rem;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    
    .context-explanation {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

def safe_get(dictionary, *keys, default=None):
    """Safely get nested dictionary values"""
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
    """Get formatted league display name"""
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

def get_context_display_name(context: str) -> str:
    """Get formatted context display name"""
    context_names = {
        'home_dominance': 'Home Dominance',
        'away_counter': 'Away Counter', 
        'offensive_showdown': 'Offensive Showdown',
        'defensive_battle': 'Defensive Battle',
        'tactical_stalemate': 'Tactical Stalemate',
        'balanced': 'Balanced Match'
    }
    return context_names.get(context, context.replace('_', ' ').title())

def get_context_emoji(context: str) -> str:
    """Get emoji for match context"""
    context_emojis = {
        'home_dominance': '🏠',
        'away_counter': '✈️',
        'offensive_showdown': '🔥',
        'defensive_battle': '🛡️',
        'tactical_stalemate': '⚔️',
        'balanced': '⚖️'
    }
    return context_emojis.get(context, '⚖️')

def display_production_predictions(predictions: dict, match_data: dict):
    """Display production-grade predictions"""
    if not predictions:
        st.error("❌ No production predictions available")
        return
        
    st.markdown('<p class="production-header">🎯 Production Football Predictions</p>', unsafe_allow_html=True)
    
    # Production mode header
    st.markdown('<div class="production-mode-active">🟢 PRODUCTION MODE ACTIVE • UNCERTAINTY PROPAGATION • VIG REMOVAL • RISK-MANAGED STAKING</div>', unsafe_allow_html=True)
    
    # Basic match info
    team_tiers = safe_get(predictions, 'team_tiers') or {}
    home_tier = team_tiers.get('home', 'MEDIUM')
    away_tier = team_tiers.get('away', 'MEDIUM')
    
    league = safe_get(predictions, 'league', default='premier_league')
    league_display_name = get_league_display_name(league)
    league_badge_class = get_league_badge(league)
    
    # Production metrics
    production_metrics = safe_get(predictions, 'production_metrics') or {}
    xg_data = safe_get(predictions, 'expected_goals') or {}
    
    st.markdown(f'''
    <div style="text-align: center; font-size: 1.5rem; font-weight: 600; margin-bottom: 1rem;">
        {predictions.get("match", "Unknown Match")} 
        <span class="production-badge tier-{home_tier.lower() if home_tier else 'medium'}">{home_tier or 'MEDIUM'}</span> vs 
        <span class="production-badge tier-{away_tier.lower() if away_tier else 'medium'}">{away_tier or 'MEDIUM'}</span>
    </div>
    <div style="text-align: center; margin-top: 0.5rem;">
        <span class="production-badge {league_badge_class}">{league_display_name}</span>
        <span class="production-feature-badge">🎯 Production Grade</span>
        <span class="production-feature-badge">📊 xG Uncertainty: ±{xg_data.get('home_uncertainty', 0):.2f}</span>
        {f'<span class="production-feature-badge">🔗 Goal Correlation</span>' if production_metrics.get('goal_correlation_modeled') else ''}
    </div>
    ''', unsafe_allow_html=True)
    
    # Expected goals with uncertainty
    st.markdown('<div class="production-section-title">📈 Expected Goals Analysis</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        home_xg = xg_data.get('home', 0)
        home_uncertainty = xg_data.get('home_uncertainty', 0)
        st.metric("🏠 Home xG", f"{home_xg:.2f}", f"±{home_uncertainty:.2f}")
    
    with col2:
        away_xg = xg_data.get('away', 0)
        away_uncertainty = xg_data.get('away_uncertainty', 0)
        st.metric("✈️ Away xG", f"{away_xg:.2f}", f"±{away_uncertainty:.2f}")
    
    with col3:
        total_xg = xg_data.get('total', 0)
        st.metric("⚽ Total xG", f"{total_xg:.2f}")
    
    with col4:
        context = safe_get(predictions, 'match_context', 'balanced')
        context_emoji = get_context_emoji(context)
        st.metric("🎯 Match Context", f"{context_emoji} {get_context_display_name(context)}")
    
    # Market probabilities
    st.markdown('<div class="production-section-title">📊 Market Probabilities</div>', unsafe_allow_html=True)
    
    outcomes = safe_get(predictions, 'probabilities', 'match_outcomes') or {}
    col1, col2, col3 = st.columns(3)
    
    with col1:
        home_win_prob = outcomes.get('home_win', 0)
        st.markdown(f'''
        <div style="margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.8rem;">
                <span><strong>Home Win</strong></span>
                <span><strong>{home_win_prob:.1f}%</strong></span>
            </div>
            <div class="production-probability-bar">
                <div class="production-probability-fill" style="width: {home_win_prob}%;"></div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        draw_prob = outcomes.get('draw', 0)
        st.markdown(f'''
        <div style="margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.8rem;">
                <span><strong>Draw</strong></span>
                <span><strong>{draw_prob:.1f}%</strong></span>
            </div>
            <div class="production-probability-bar">
                <div class="production-probability-fill" style="width: {draw_prob}%; background: #FF9800;"></div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        away_win_prob = outcomes.get('away_win', 0)
        st.markdown(f'''
        <div style="margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.8rem;">
                <span><strong>Away Win</strong></span>
                <span><strong>{away_win_prob:.1f}%</strong></span>
            </div>
            <div class="production-probability-bar">
                <div class="production-probability-fill" style="width: {away_win_prob}%; background: #2196F3;"></div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Goals markets
    st.markdown('<div class="production-section-title">⚽ Goals Markets</div>', unsafe_allow_html=True)
    
    btts_yes = safe_get(predictions, 'probabilities', 'both_teams_score', 'yes') or 0
    btts_no = safe_get(predictions, 'probabilities', 'both_teams_score', 'no') or 0
    over_25 = safe_get(predictions, 'probabilities', 'over_under', 'over_25') or 0
    under_25 = safe_get(predictions, 'probabilities', 'over_under', 'under_25') or 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("✅ BTTS Yes", f"{btts_yes:.1f}%")
    with col2:
        st.metric("❌ BTTS No", f"{btts_no:.1f}%")
    with col3:
        st.metric("📈 Over 2.5", f"{over_25:.1f}%")
    with col4:
        st.metric("📉 Under 2.5", f"{under_25:.1f}%")
    
    # Market edges analysis
    st.markdown('<div class="production-section-title">💰 Market Edge Analysis</div>', unsafe_allow_html=True)
    
    market_analysis = safe_get(predictions, 'market_analysis') or {}
    edges = market_analysis.get('edges', {})
    min_edge = market_analysis.get('min_edge_threshold', 0)
    
    if edges:
        edge_cols = st.columns(4)
        edge_display = {
            'home_win': ('🏠 Home Win', edges.get('home_win', 0)),
            'away_win': ('✈️ Away Win', edges.get('away_win', 0)),
            'btts_yes': ('✅ BTTS Yes', edges.get('btts_yes', 0)),
            'over_25': ('📈 Over 2.5', edges.get('over_25', 0))
        }
        
        for idx, (key, (display_name, edge)) in enumerate(edge_display.items()):
            with edge_cols[idx]:
                edge_pct = edge * 100
                if edge_pct >= min_edge:
                    st.metric(display_name, f"{edge_pct:+.1f}%", "✅ Value", delta_color="inverse")
                else:
                    st.metric(display_name, f"{edge_pct:+.1f}%", "❌ No Edge")
    
    # Betting recommendations
    betting_recommendations = safe_get(predictions, 'betting_recommendations') or []
    if betting_recommendations:
        st.markdown('<div class="production-section-title">🎯 Value Betting Opportunities</div>', unsafe_allow_html=True)
        
        for rec in betting_recommendations:
            edge_pct = rec['edge'] * 100
            stake_pct = rec['bankroll_percentage']
            
            st.markdown(f'''
            <div class="value-opportunity-card">
                <div style="display: flex; justify-content: between; align-items: center;">
                    <div style="flex: 1;">
                        <strong>{rec['market'].replace('_', ' ').title()}</strong><br>
                        <span>Edge: <span class="edge-positive">+{edge_pct:.1f}%</span> • Odds: {rec['odds']:.2f}</span>
                    </div>
                    <div style="text-align: right;">
                        <strong>${rec['stake']:.2f}</strong><br>
                        <span>{stake_pct:.1f}% of bankroll</span>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
    else:
        st.info("ℹ️ No value betting opportunities meeting minimum edge threshold")
    
    # Exact scores
    st.markdown('<div class="production-section-title">🎯 Most Likely Scores</div>', unsafe_allow_html=True)
    
    exact_scores = safe_get(predictions, 'probabilities', 'exact_scores') or {}
    if exact_scores:
        top_scores = dict(list(exact_scores.items())[:6])
        score_cols = st.columns(6)
        for idx, (score, prob) in enumerate(top_scores.items()):
            with score_cols[idx]:
                st.metric(f"{score}", f"{prob*100:.1f}%")
    else:
        st.info("No exact score data available")
    
    # Risk assessment
    risk_assessment = safe_get(predictions, 'risk_assessment') or {}
    risk_class = f"risk-{risk_assessment.get('risk_level', 'unknown').lower()}"
    
    st.markdown(f'''
    <div class="production-card {risk_class}">
        <h3>📊 Production Risk Assessment</h3>
        <strong>Risk Level:</strong> {risk_assessment.get("risk_level", "UNKNOWN")}<br>
        <strong>Explanation:</strong> {risk_assessment.get("explanation", "No data available")}<br>
        <strong>Recommendation:</strong> {risk_assessment.get("recommendation", "N/A")}<br>
        <strong>Certainty:</strong> {risk_assessment.get("certainty", "N/A")}<br>
        <strong>Production Features:</strong><br>
        {', '.join([f'✅ {feat}' for feat in production_metrics.keys() if production_metrics.get(feat)])}
    </div>
    ''', unsafe_allow_html=True)
    
    # Explanations
    explanations = safe_get(predictions, 'explanations') or []
    if explanations:
        st.markdown('<div class="production-section-title">📝 Match Analysis</div>', unsafe_allow_html=True)
        for explanation in explanations:
            st.markdown(f'<div class="context-explanation">💡 {explanation}</div>', unsafe_allow_html=True)
    
    # Summary
    summary = safe_get(predictions, 'production_summary', 'Production analysis complete.')
    st.info(summary)

def create_production_input_form():
    """Create production-grade input form"""
    st.markdown('<p class="production-header">🎯 Production Football Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="production-subheader">Professional-Grade Analysis with Uncertainty Propagation & Risk Management</p>', unsafe_allow_html=True)
    
    # League selection
    league_options = {
        'championship': 'Championship 🏴󠁧󠁢󠁥󠁮󠁧󠁿',
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
        index=0
    )
    
    league_badge_class = get_league_badge(selected_league)
    league_display_name = get_league_display_name(selected_league)
    st.markdown(f'<span class="production-badge {league_badge_class}">{league_display_name}</span>', unsafe_allow_html=True)
    
    # Team selection
    calibrator = EnhancedTeamTierCalibrator()
    league_teams = calibrator.team_databases.get(selected_league, {})
    
    if not league_teams:
        st.error(f"❌ No teams found for {league_display_name}")
        return None
    
    tab1, tab2, tab3 = st.tabs(["🏠 Team Data", "💰 Market Data", "⚙️ Risk Settings"])

    with tab1:
        st.markdown("### 🎯 Team Performance Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏠 Home Team")
            home_team = st.selectbox(
                "Team Name", 
                options=list(league_teams.keys()),
                index=list(league_teams.keys()).index('Charlton Athletic') if 'Charlton Athletic' in league_teams else 0,
                key="production_home_team"
            )
            
            home_goals = st.number_input("Total Goals (Last 6 Games)", min_value=0, value=8, key="production_home_goals")
            home_conceded = st.number_input("Total Conceded (Last 6 Games)", min_value=0, value=6, key="production_home_conceded")
            home_goals_home = st.number_input("Home Goals (Last 3 Home Games)", min_value=0, value=6, key="production_home_goals_home")
            
        with col2:
            st.subheader("✈️ Away Team")
            away_team = st.selectbox(
                "Team Name",
                options=list(league_teams.keys()),
                index=list(league_teams.keys()).index('West Brom') if 'West Brom' in league_teams else 1,
                key="production_away_team"
            )
            
            away_goals = st.number_input("Total Goals (Last 6 Games)", min_value=0, value=4, key="production_away_goals")
            away_conceded = st.number_input("Total Conceded (Last 6 Games)", min_value=0, value=7, key="production_away_conceded")
            away_goals_away = st.number_input("Away Goals (Last 3 Away Games)", min_value=0, value=1, key="production_away_goals_away")
        
        # Show team tiers
        home_tier = calibrator.get_team_tier(home_team, selected_league)
        away_tier = calibrator.get_team_tier(away_team, selected_league)
        
        st.markdown(f"""
        **Team Quality Assessment:** 
        <span class="production-badge tier-{home_tier.lower() if home_tier else 'medium'}">{home_tier or 'MEDIUM'}</span> vs 
        <span class="production-badge tier-{away_tier.lower() if away_tier else 'medium'}">{away_tier or 'MEDIUM'}</span>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### 💰 Market Odds")
        
        odds_col1, odds_col2, odds_col3 = st.columns(3)
        
        with odds_col1:
            st.write("**1X2 Market**")
            home_odds = st.number_input("Home Win Odds", min_value=1.01, value=3.10, step=0.01, key="production_home_odds")
            draw_odds = st.number_input("Draw Odds", min_value=1.01, value=3.50, step=0.01, key="production_draw_odds")
            away_odds = st.number_input("Away Win Odds", min_value=1.01, value=2.25, step=0.01, key="production_away_odds")
        
        with odds_col2:
            st.write("**Over/Under Markets**")
            over_25_odds = st.number_input("Over 2.5 Goals", min_value=1.01, value=1.80, step=0.01, key="production_over_25_odds")
            under_25_odds = st.number_input("Under 2.5 Goals", min_value=1.01, value=2.50, step=0.01, key="production_under_25_odds")
        
        with odds_col3:
            st.write("**Both Teams to Score**")
            btts_yes_odds = st.number_input("BTTS Yes", min_value=1.01, value=1.67, step=0.01, key="production_btts_yes_odds")
            btts_no_odds = st.number_input("BTTS No", min_value=1.01, value=2.10, step=0.01, key="production_btts_no_odds")

    with tab3:
        st.markdown("### ⚙️ Risk Management")
        
        risk_col1, risk_col2 = st.columns(2)
        
        with risk_col1:
            st.write("**Bankroll Management**")
            bankroll = st.number_input("Bankroll ($)", min_value=500, value=1000, step=100, key="production_bankroll")
            kelly_fraction = st.slider("Kelly Fraction", 0.1, 0.3, 0.2, key="production_kelly_fraction")
            st.info(f"Max stake: ${bankroll * 0.03:.2f} (3% of bankroll)")
        
        with risk_col2:
            st.write("**Production Features**")
            st.checkbox("xG Uncertainty Propagation", value=True, disabled=True)
            st.checkbox("Goal Correlation Modeling", value=True, disabled=True)
            st.checkbox("Proper Vig Removal", value=True, disabled=True)
            st.checkbox("Sensitivity Testing", value=True, disabled=True)

    submitted = st.button("🎯 RUN PRODUCTION ANALYSIS", type="primary", use_container_width=True)
    
    if submitted:
        if not home_team or not away_team:
            st.error("❌ Please enter both team names")
            return None
        
        if home_team == away_team:
            st.error("❌ Home and away teams cannot be the same")
            return None
        
        market_odds = {
            '1x2 Home': home_odds,
            '1x2 Draw': draw_odds,
            '1x2 Away': away_odds,
            'Over 2.5 Goals': over_25_odds,
            'Under 2.5 Goals': under_25_odds,
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
            'market_odds': market_odds,
            'bankroll': bankroll,
            'kelly_fraction': kelly_fraction
        }
        
        return match_data
    
    return None

def main():
    """Main application function"""
    # Initialize session state
    if 'production_predictions' not in st.session_state:
        st.session_state.production_predictions = None
    
    if 'production_history' not in st.session_state:
        st.session_state.production_history = []
    
    if 'match_data' not in st.session_state:
        st.session_state.match_data = None
    
    # Display existing predictions if available
    if st.session_state.production_predictions and st.session_state.match_data:
        display_production_predictions(st.session_state.production_predictions, st.session_state.match_data)
        
        # Navigation
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 New Production Analysis", use_container_width=True):
                st.session_state.production_predictions = None
                st.session_state.match_data = None
                st.rerun()
        
        with col2:
            if st.button("📊 Production Metrics", use_container_width=True):
                st.success("""
                **Production System Status: OPERATIONAL** 🟢
                
                **Active Production Features:**
                - ✅ xG Uncertainty Propagation
                - ✅ Bivariate Poisson Goal Correlation  
                - ✅ Proper Vig Removal
                - ✅ Risk-Managed Staking
                - ✅ Market Edge Verification
                - ✅ Sensitivity Testing Framework
                
                **Model Version:** 4.0.0_production
                **Calibration Level:** PRODUCTION_READY
                """)
        
        return
    
    # Get new match data and generate predictions
    match_data = create_production_input_form()
    
    if match_data:
        with st.spinner("🔍 Running production analysis with uncertainty propagation..."):
            try:
                engine = ApexProductionEngine(match_data)
                predictions = engine.generate_production_predictions()
                
                if predictions:
                    st.session_state.production_predictions = predictions
                    st.session_state.match_data = match_data
                    
                    # Add to history
                    if 'production_history' not in st.session_state:
                        st.session_state.production_history = []
                    
                    prediction_record = {
                        'timestamp': datetime.now().isoformat(),
                        'match': predictions.get('match', 'Unknown Match'),
                        'league': predictions.get('league', 'premier_league'),
                        'context': predictions.get('match_context', 'balanced'),
                        'expected_goals': predictions.get('expected_goals', {}),
                        'value_opportunities': len(predictions.get('betting_recommendations', [])),
                        'risk_level': safe_get(predictions, 'risk_assessment', 'risk_level') or 'UNKNOWN'
                    }
                    
                    st.session_state.production_history.append(prediction_record)
                    
                    st.success("""
                    ✅ **PRODUCTION ANALYSIS COMPLETE!**
                    
                    **Production Features Applied:**
                    - 🎯 xG Uncertainty Propagation
                    - 🔗 Goal Correlation Modeling  
                    - 💰 Proper Vig Removal
                    - 🛡️ Risk-Managed Staking
                    - 📊 Sensitivity Testing
                    """)
                    
                    st.rerun()
                else:
                    st.error("❌ Failed to generate production predictions")
                
            except Exception as e:
                st.error(f"❌ Production analysis error: {str(e)}")
                st.info("💡 Check input parameters and try again")

if __name__ == "__main__":
    main()
