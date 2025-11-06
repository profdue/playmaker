# streamlit_app.py - PROFESSIONAL MULTI-LEAGUE PREDICTOR
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime
from typing import Dict, Any, List
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

try:
    from prediction_engine import (
        MultiLeaguePredictionEngine, 
        LEAGUE_PARAMS,
        VOLATILITY_MULTIPLIERS
    )
except ImportError as e:
    st.error(f"❌ Import Error: {str(e)}")
    st.info("💡 Make sure prediction_engine.py is in the same directory")
    st.stop()

# Clear cache for fresh imports
st.cache_resource.clear()

# =============================================================================
# PROFESSIONAL STREAMLIT CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="🎯 Professional Multi-League Football Predictor",
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
    .professional-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .card-success { border-left-color: #4CAF50; background: #f8fff8; }
    .card-warning { border-left-color: #FF9800; background: #fffaf2; }
    .card-danger { border-left-color: #f44336; background: #fff5f5; }
    .card-info { border-left-color: #2196F3; background: #f8fdff; }
    
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
    .eredivisie { background: #FF6B00; }
    .liga-portugal { background: #006600; }
    .brasileirao { background: #FFCC00; color: black; }
    .liga-mx { background: #006847; }
    .championship { background: #8B0000; }
    
    .value-indicator {
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        color: white;
        margin: 0.1rem;
    }
    .value-exceptional { background: #4CAF50; }
    .value-high { background: #8BC34A; }
    .value-good { background: #FFC107; color: black; }
    .value-moderate { background: #FF9800; color: white; }
    .value-low { background: #f44336; }
    
    .robustness-high { background: #4CAF50; color: white; padding: 0.2rem 0.6rem; border-radius: 10px; }
    .robustness-medium { background: #FF9800; color: white; padding: 0.2rem 0.6rem; border-radius: 10px; }
    .robustness-low { background: #f44336; color: white; padding: 0.2rem 0.6rem; border-radius: 10px; }
    
    .stake-recommendation {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .diagnostic-panel {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .context-panel {
        background: #fff3e0;
        border-left: 4px solid #FF9800;
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
        border-radius: 4px;
    }
    
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #333;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #f0f2f6;
    }
    
    .league-intelligence {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .league-selector {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# PROFESSIONAL HELPER FUNCTIONS
# =============================================================================

def get_league_display_name(league_id: str) -> str:
    """Get formatted league display name"""
    return LEAGUE_PARAMS.get(league_id, {}).get('display_name', league_id.replace('_', ' ').title())

def get_league_badge_class(league_id: str) -> str:
    """Get CSS class for league badge"""
    league_classes = {
        'premier_league': 'premier-league',
        'la_liga': 'la-liga',
        'serie_a': 'serie-a',
        'bundesliga': 'bundesliga',
        'ligue_1': 'ligue-1',
        'eredivisie': 'eredivisie',
        'liga_portugal': 'liga-portugal',
        'brasileirao': 'brasileirao',
        'liga_mx': 'liga-mx',
        'championship': 'championship'
    }
    return league_classes.get(league_id, 'premier-league')

def get_context_display_name(context: str) -> str:
    """Get formatted context display name"""
    context_names = {
        'home_dominance': 'Home Dominance 🏠',
        'away_counter': 'Away Counter ✈️',
        'offensive_showdown': 'Offensive Showdown 🔥',
        'defensive_battle': 'Defensive Battle 🛡️',
        'tactical_stalemate': 'Tactical Stalemate ⚔️',
        'balanced': 'Balanced Match ⚖️'
    }
    return context_names.get(context, context.replace('_', ' ').title())

def get_value_indicator(edge: float) -> str:
    """Get value indicator based on edge size"""
    if abs(edge) >= 0.10:
        return "value-exceptional"
    elif abs(edge) >= 0.07:
        return "value-high"
    elif abs(edge) >= 0.05:
        return "value-good"
    elif abs(edge) >= 0.03:
        return "value-moderate"
    else:
        return "value-low"

def format_percentage(value: float) -> str:
    """Format percentage with sign"""
    return f"{value:+.1%}" if abs(value) >= 0.001 else "0.0%"

def safe_get(data, *keys, default=None):
    """Safely get nested dictionary values"""
    current = data
    for key in keys:
        try:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        except (TypeError, KeyError, AttributeError):
            return default
    return current

# =============================================================================
# PROFESSIONAL VISUALIZATION COMPONENTS
# =============================================================================

def create_probability_gauge(probability: float, title: str, color: str = "#4CAF50") -> go.Figure:
    """Create professional probability gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 16}},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 33], 'color': "lightgray"},
                {'range': [33, 66], 'color': "gray"},
                {'range': [66, 100], 'color': "darkgray"}
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_edge_comparison_chart(recommendations: List[Dict]) -> go.Figure:
    """Create professional edge comparison chart"""
    if not recommendations:
        return go.Figure()
    
    markets = [rec['market'] for rec in recommendations]
    edges = [rec['edge'] * 100 for rec in recommendations]  # Convert to percentage
    
    colors = ['#4CAF50' if edge > 0 else '#f44336' for edge in edges]
    
    fig = go.Figure(data=[
        go.Bar(
            x=markets,
            y=edges,
            marker_color=colors,
            text=[f"{edge:+.1f}%" for edge in edges],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Betting Edge Comparison",
        xaxis_title="Market",
        yaxis_title="Edge (%)",
        showlegend=False,
        height=400
    )
    
    return fig

def create_xg_comparison_plot(results: Dict) -> go.Figure:
    """Create xG comparison plot"""
    home_xg = safe_get(results, 'expected_goals', 'home', default=0)
    away_xg = safe_get(results, 'expected_goals', 'away', default=0)
    
    fig = go.Figure(data=[
        go.Bar(
            name='Expected Goals (xG)',
            x=['Home', 'Away'],
            y=[home_xg, away_xg],
            marker_color=['#FF6B6B', '#4ECDC4']
        )
    ])
    
    fig.update_layout(
        title="Expected Goals Comparison",
        yaxis_title="Expected Goals (xG)",
        showlegend=False,
        height=300
    )
    
    return fig

def create_score_probability_plot(exact_scores: Dict[str, float]) -> go.Figure:
    """Create exact score probability plot"""
    if not exact_scores:
        return go.Figure()
    
    scores = list(exact_scores.keys())[:6]
    probabilities = [exact_scores[score] * 100 for score in scores]
    
    fig = go.Figure(data=[
        go.Bar(
            x=scores,
            y=probabilities,
            marker_color='#667eea',
            text=[f"{prob:.1f}%" for prob in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Most Likely Exact Scores",
        xaxis_title="Score",
        yaxis_title="Probability (%)",
        height=300
    )
    
    return fig

# =============================================================================
# PROFESSIONAL STREAMLIT COMPONENTS
# =============================================================================

def display_professional_header():
    """Display professional header"""
    st.markdown('<p class="professional-header">🎯 Professional Multi-League Football Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="professional-subheader">10 League Support • Evidence-Based Predictions • Professional Risk Management</p>', unsafe_allow_html=True)

def display_league_selector():
    """Display professional league selector"""
    st.markdown('<div class="league-selector">', unsafe_allow_html=True)
    st.markdown('### 🌍 Select League')
    
    # Create two columns for league buttons
    col1, col2, col3, col4, col5 = st.columns(5)
    
    leagues = list(LEAGUE_PARAMS.keys())
    
    with col1:
        if st.button(LEAGUE_PARAMS['premier_league']['display_name'], use_container_width=True):
            st.session_state.selected_league = 'premier_league'
        if st.button(LEAGUE_PARAMS['la_liga']['display_name'], use_container_width=True):
            st.session_state.selected_league = 'la_liga'
    
    with col2:
        if st.button(LEAGUE_PARAMS['serie_a']['display_name'], use_container_width=True):
            st.session_state.selected_league = 'serie_a'
        if st.button(LEAGUE_PARAMS['bundesliga']['display_name'], use_container_width=True):
            st.session_state.selected_league = 'bundesliga'
    
    with col3:
        if st.button(LEAGUE_PARAMS['ligue_1']['display_name'], use_container_width=True):
            st.session_state.selected_league = 'ligue_1'
        if st.button(LEAGUE_PARAMS['eredivisie']['display_name'], use_container_width=True):
            st.session_state.selected_league = 'eredivisie'
    
    with col4:
        if st.button(LEAGUE_PARAMS['liga_portugal']['display_name'], use_container_width=True):
            st.session_state.selected_league = 'liga_portugal'
        if st.button(LEAGUE_PARAMS['brasileirao']['display_name'], use_container_width=True):
            st.session_state.selected_league = 'brasileirao'
    
    with col5:
        if st.button(LEAGUE_PARAMS['liga_mx']['display_name'], use_container_width=True):
            st.session_state.selected_league = 'liga_mx'
        if st.button(LEAGUE_PARAMS['championship']['display_name'], use_container_width=True):
            st.session_state.selected_league = 'championship'
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Default league
    if 'selected_league' not in st.session_state:
        st.session_state.selected_league = 'premier_league'
    
    return st.session_state.selected_league

def display_league_intelligence(league: str):
    """Display league intelligence panel"""
    league_params = LEAGUE_PARAMS.get(league, LEAGUE_PARAMS['premier_league'])
    volatility_multiplier = VOLATILITY_MULTIPLIERS.get(league_params['volatility'], 1.0)
    
    st.markdown(f'''
    <div class="league-intelligence">
        <h3>🌍 {league_params["display_name"]} Intelligence</h3>
        <strong>Goal Baseline:</strong> {league_params['goal_baseline']:.2f} xG per game<br>
        <strong>Volatility:</strong> {league_params['volatility'].replace('_', ' ').title()}<br>
        <strong>Minimum Edge:</strong> {league_params['min_edge']:.1%}<br>
        <strong>Stake Multiplier:</strong> {volatility_multiplier:.1f}x<br>
        <strong>Goal Intensity:</strong> {league_params['goal_intensity'].replace('_', ' ').title()}<br>
        <strong>Home Advantage:</strong> {league_params['home_advantage']:.2f}x
    </div>
    ''', unsafe_allow_html=True)

def display_match_overview(results: Dict):
    """Display professional match overview"""
    st.markdown('<div class="section-title">📊 Match Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        home_xg = safe_get(results, 'expected_goals', 'home', default=0)
        st.metric("🏠 Expected Goals", f"{home_xg:.2f}")
    
    with col2:
        away_xg = safe_get(results, 'expected_goals', 'away', default=0)
        st.metric("✈️ Expected Goals", f"{away_xg:.2f}")
    
    with col3:
        total_xg = safe_get(results, 'expected_goals', 'total', default=0)
        st.metric("🎯 Total xG", f"{total_xg:.2f}")
    
    with col4:
        context = safe_get(results, 'match_context', 'primary_pattern', default='balanced')
        st.metric("🔍 Match Context", get_context_display_name(context))

def display_probability_analysis(results: Dict):
    """Display professional probability analysis"""
    st.markdown('<div class="section-title">📈 Probability Analysis</div>', unsafe_allow_html=True)
    
    # Outcome probabilities
    outcomes = safe_get(results, 'probabilities', 'match_outcomes', default={})
    col1, col2, col3 = st.columns(3)
    
    with col1:
        home_win = outcomes.get('home_win', 0)
        st.plotly_chart(create_probability_gauge(home_win, "Home Win", "#FF6B6B"), use_container_width=True)
    
    with col2:
        draw = outcomes.get('draw', 0)
        st.plotly_chart(create_probability_gauge(draw, "Draw", "#FFC107"), use_container_width=True)
    
    with col3:
        away_win = outcomes.get('away_win', 0)
        st.plotly_chart(create_probability_gauge(away_win, "Away Win", "#4ECDC4"), use_container_width=True)
    
    # Goals markets
    st.markdown("#### ⚽ Goals Markets")
    goals_col1, goals_col2, goals_col3, goals_col4 = st.columns(4)
    
    with goals_col1:
        btts_yes = safe_get(results, 'probabilities', 'both_teams_score', 'yes', default=0)
        st.metric("Both Teams Score", f"{btts_yes:.1%}")
    
    with goals_col2:
        over_25 = safe_get(results, 'probabilities', 'over_under', 'over_25', default=0)
        st.metric("Over 2.5 Goals", f"{over_25:.1%}")
    
    with goals_col3:
        under_25 = safe_get(results, 'probabilities', 'over_under', 'under_25', default=0)
        st.metric("Under 2.5 Goals", f"{under_25:.1%}")
    
    with goals_col4:
        over_15 = safe_get(results, 'probabilities', 'over_under', 'over_15', default=0)
        st.metric("Over 1.5 Goals", f"{over_15:.1%}")
    
    # Visualizations
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.plotly_chart(create_xg_comparison_plot(results), use_container_width=True)
    
    with viz_col2:
        exact_scores = safe_get(results, 'probabilities', 'exact_scores', default={})
        st.plotly_chart(create_score_probability_plot(exact_scores), use_container_width=True)

def display_betting_recommendations(results: Dict):
    """Display professional betting recommendations"""
    st.markdown('<div class="section-title">💰 Betting Recommendations</div>', unsafe_allow_html=True)
    
    recommendations = safe_get(results, 'market_analysis', 'recommendations', default=[])
    
    if not recommendations:
        st.info("🔍 No betting recommendations meet the professional edge thresholds.")
        return
    
    # Edge comparison chart
    st.plotly_chart(create_edge_comparison_chart(recommendations), use_container_width=True)
    
    # Individual recommendations
    for rec in recommendations:
        edge = rec.get('edge', 0)
        robustness = rec.get('robustness', 'LOW')
        stake = rec.get('recommended_stake', 0)
        
        card_class = "card-success" if edge > 0 else "card-danger"
        
        st.markdown(f'''
        <div class="professional-card {card_class}">
            <div style="display: flex; justify-content: between; align-items: center;">
                <div style="flex: 1;">
                    <h4>{rec.get('market', 'Unknown')}</h4>
                    <div style="display: flex; align-items: center; gap: 1rem; margin: 0.5rem 0;">
                        <span class="value-indicator {get_value_indicator(edge)}">Edge: {format_percentage(edge)}</span>
                        <span class="robustness-{robustness.lower()}">Robustness: {robustness}</span>
                    </div>
                    <div style="font-size: 0.9rem; color: #666;">
                        Model: {rec.get('model_prob', 0):.1%} • Market: {rec.get('implied_prob', 0):.1%}
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.2rem; font-weight: bold; color: #333;">
                        ${stake:.2f}
                    </div>
                    <div style="font-size: 0.8rem; color: #666;">
                        Recommended Stake
                    </div>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Explanation
        with st.expander("📝 Recommendation Details"):
            explanations = rec.get('explanation', [])
            for explanation in explanations:
                st.write(f"• {explanation}")

def display_match_context(results: Dict):
    """Display professional match context analysis"""
    st.markdown('<div class="section-title">🔍 Match Context Analysis</div>', unsafe_allow_html=True)
    
    context = safe_get(results, 'match_context', default={})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'''
        <div class="context-panel">
            <h4>🎯 Primary Pattern</h4>
            <div style="font-size: 1.2rem; font-weight: bold; margin: 0.5rem 0;">
                {get_context_display_name(context.get('primary_pattern', 'balanced'))}
            </div>
            <div style="margin: 0.5rem 0;">
                <strong>Quality Gap:</strong> {context.get('quality_gap', 'even').title()}<br>
                <strong>Confidence Score:</strong> {context.get('confidence_score', 0):.1f}%<br>
                <strong>Expected Tempo:</strong> {context.get('expected_tempo', 'medium').title()}
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        flags = []
        if context.get('home_advantage_amplified'):
            flags.append("🏠 Home Advantage Amplified")
        if context.get('away_scoring_issues'):
            flags.append("✈️ Away Scoring Issues")
        
        if flags:
            st.markdown(f'''
            <div class="context-panel">
                <h4>🚩 Context Flags</h4>
                {"<br>".join([f"• {flag}" for flag in flags])}
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="context-panel">
                <h4>🚩 Context Flags</h4>
                • No significant context flags detected
            </div>
            ''', unsafe_allow_html=True)

def display_diagnostics(results: Dict):
    """Display professional model diagnostics"""
    st.markdown('<div class="section-title">📊 Model Diagnostics</div>', unsafe_allow_html=True)
    
    diagnostics = safe_get(results, 'diagnostics', default={})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        data_quality = diagnostics.get('data_quality_score', 0)
        st.metric("📈 Data Quality", f"{data_quality:.1f}/100")
    
    with col2:
        calibration = diagnostics.get('calibration_score', 0)
        st.metric("🎯 Calibration", f"{calibration:.1f}/100")
    
    with col3:
        market_align = diagnostics.get('market_alignment', 0)
        st.metric("📊 Market Alignment", f"{market_align:.1f}%")
    
    with col4:
        uncertainty = diagnostics.get('uncertainty_score', 0)
        st.metric("⚡ Uncertainty", f"{uncertainty:.1f}/100")
    
    # Recommended action
    action = diagnostics.get('recommended_action', 'NO_VALUE')
    action_colors = {
        'CONFIDENT_BETTING': '🟢',
        'CAUTIOUS_BETTING': '🟡', 
        'NO_VALUE': '🔴'
    }
    
    st.markdown(f'''
    <div class="diagnostic-panel">
        <h4>🎯 Recommended Action</h4>
        <div style="font-size: 1.2rem; font-weight: bold; margin: 0.5rem 0;">
            {action_colors.get(action, '⚪')} {action.replace('_', ' ').title()}
        </div>
        <div style="font-size: 0.9rem; color: #666;">
            Sensitivity Test: {"✅ Passed" if diagnostics.get('sensitivity_passed') else "❌ Failed"}
        </div>
    </div>
    ''', unsafe_allow_html=True)

def display_xg_debug(results: Dict):
    """Display xG calculation debug information"""
    with st.expander("🔧 xG Calculation Details"):
        xg_debug = safe_get(results, 'expected_goals', 'debug', default={})
        
        if xg_debug:
            st.write("**xG Calculation Steps:**")
            for key, value in xg_debug.items():
                st.write(f"- **{key}**: {value:.3f}")
        else:
            st.info("No debug information available")

def display_professional_metadata(results: Dict):
    """Display professional metadata"""
    with st.expander("🔧 Technical Details"):
        metadata = safe_get(results, 'professional_metadata', default={})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Model Information**")
            st.write(f"Version: {metadata.get('model_version', 'N/A')}")
            st.write(f"Calibration: {metadata.get('calibration_level', 'N/A')}")
            st.write(f"Risk Profile: {metadata.get('risk_profile', 'N/A')}")
            st.write(f"League Support: {metadata.get('league_supported', 'No')}")
        
        with col2:
            st.write("**Execution Details**")
            st.write(f"Generated: {metadata.get('timestamp', 'N/A')}")
            st.write(f"Match: {safe_get(results, 'match_info', 'match', 'N/A')}")
            st.write(f"League: {safe_get(results, 'match_info', 'league_display', 'N/A')}")

# =============================================================================
# PROFESSIONAL INPUT FORM
# =============================================================================

def create_professional_input_form(selected_league: str) -> tuple:
    """Create professional input form for selected league"""
    st.markdown('<div class="section-title">⚙️ Match Configuration</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🏟️ Match Data", "💰 Market Odds", "🎯 Professional Settings"])
    
    match_data = {'league': selected_league}
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏠 Home Team")
            match_data['home_team'] = st.text_input("Team Name", value="Tottenham Hotspur", key="home_team")
            match_data['home_goals'] = st.number_input("Total Goals (Last 6)", min_value=0, value=12, key="home_goals")
            match_data['home_conceded'] = st.number_input("Total Conceded (Last 6)", min_value=0, value=6, key="home_conceded")
            match_data['home_goals_home'] = st.number_input("Home Goals (Last 3 Home)", min_value=0, value=2, key="home_goals_home")
            match_data['home_injuries'] = st.slider("Key Absences", 1, 5, 2, key="home_injuries")
            match_data['home_motivation'] = st.selectbox("Motivation", ["Low", "Normal", "High", "Very High"], index=1, key="home_motivation")
            match_data['home_tier'] = st.selectbox("Team Tier", ["WEAK", "MEDIUM", "STRONG", "ELITE"], index=2, key="home_tier")
        
        with col2:
            st.subheader("✈️ Away Team")
            match_data['away_team'] = st.text_input("Away Team Name", value="Chelsea", key="away_team")
            match_data['away_goals'] = st.number_input("Away Goals (Last 6)", min_value=0, value=10, key="away_goals")
            match_data['away_conceded'] = st.number_input("Away Conceded (Last 6)", min_value=0, value=7, key="away_conceded")
            match_data['away_goals_away'] = st.number_input("Away Goals (Last 3 Away)", min_value=0, value=6, key="away_goals_away")
            match_data['away_injuries'] = st.slider("Away Key Absences", 1, 5, 2, key="away_injuries")
            match_data['away_motivation'] = st.selectbox("Away Motivation", ["Low", "Normal", "High", "Very High"], index=1, key="away_motivation")
            match_data['away_tier'] = st.selectbox("Away Team Tier", ["WEAK", "MEDIUM", "STRONG", "ELITE"], index=2, key="away_tier")
    
    with tab2:
        st.subheader("💰 Market Odds")
        
        odds_col1, odds_col2 = st.columns(2)
        
        with odds_col1:
            st.write("**1X2 Market**")
            match_data['market_odds'] = {
                '1x2_home': st.number_input("Home Win Odds", min_value=1.01, value=3.10, step=0.01, key="home_odds"),
                '1x2_draw': st.number_input("Draw Odds", min_value=1.01, value=3.40, step=0.01, key="draw_odds"),
                '1x2_away': st.number_input("Away Win Odds", min_value=1.01, value=2.30, step=0.01, key="away_odds")
            }
        
        with odds_col2:
            st.write("**Goals Markets**")
            match_data['market_odds']['over_25'] = st.number_input("Over 2.5 Goals", min_value=1.01, value=1.80, step=0.01, key="over_25")
            match_data['market_odds']['under_25'] = st.number_input("Under 2.5 Goals", min_value=1.01, value=2.00, step=0.01, key="under_25")
            match_data['market_odds']['btts_yes'] = st.number_input("BTTS Yes", min_value=1.01, value=1.90, step=0.01, key="btts_yes")
            match_data['market_odds']['btts_no'] = st.number_input("BTTS No", min_value=1.01, value=1.90, step=0.01, key="btts_no")
    
    with tab3:
        st.subheader("🎯 Professional Settings")
        
        settings_col1, settings_col2 = st.columns(2)
        
        with settings_col1:
            match_data['bankroll'] = st.number_input("Bankroll ($)", min_value=100, value=1000, step=100, key="bankroll")
            st.info("💡 Professional staking uses fractional Kelly with volatility adjustments")
        
        with settings_col2:
            st.write("**Model Configuration**")
            enable_sensitivity = st.checkbox("Enable Sensitivity Analysis", value=True, key="sensitivity")
            st.info("🔍 Sensitivity testing checks edge robustness to xG changes")
    
    # H2H Data
    with st.expander("📊 Head-to-Head Data (Optional)"):
        h2h_col1, h2h_col2 = st.columns(2)
        
        with h2h_col1:
            h2h_matches = st.number_input("Total H2H Matches", min_value=0, value=6, key="h2h_matches")
            h2h_home_wins = st.number_input("Home Wins", min_value=0, value=1, key="h2h_home_wins")
        
        with h2h_col2:
            h2h_away_wins = st.number_input("Away Wins", min_value=0, value=4, key="h2h_away_wins")
            h2h_draws = st.number_input("Draws", min_value=0, value=1, key="h2h_draws")
        
        match_data['h2h_data'] = {
            'matches': h2h_matches,
            'home_wins': h2h_home_wins,
            'away_wins': h2h_away_wins,
            'draws': h2h_draws,
            'home_goals': 7,  # Default values
            'away_goals': 9
        }
    
    generate = st.button("🎯 GENERATE PROFESSIONAL ANALYSIS", type="primary", use_container_width=True)
    
    return match_data if generate else None

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main professional multi-league application"""
    
    # Initialize session state
    if 'professional_results' not in st.session_state:
        st.session_state.professional_results = None
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    if 'selected_league' not in st.session_state:
        st.session_state.selected_league = 'premier_league'
    
    # Display professional header
    display_professional_header()
    
    # League selection
    selected_league = display_league_selector()
    
    # Display league intelligence
    display_league_intelligence(selected_league)
    
    # Sidebar for navigation and info
    with st.sidebar:
        st.markdown("## 🎯 Navigation")
        
        if st.session_state.professional_results:
            if st.button("🔄 New Analysis", use_container_width=True):
                st.session_state.professional_results = None
                st.rerun()
            
            if st.button("📊 View History", use_container_width=True):
                if st.session_state.prediction_history:
                    st.write("### Recent Analyses")
                    for i, pred in enumerate(st.session_state.prediction_history[-5:]):
                        with st.expander(f"Analysis {i+1}: {pred.get('match', 'Unknown')}"):
                            st.write(f"League: {pred.get('league_display', 'Unknown')}")
                            st.write(f"xG: {pred.get('home_xg', 0):.2f}-{pred.get('away_xg', 0):.2f}")
                            st.write(f"Recommendations: {len(pred.get('recommendations', []))}")
                else:
                    st.info("No analysis history yet.")
        
        st.markdown("---")
        st.markdown("## 🔧 System Status")
        st.success("**Multi-League Mode:** ACTIVE 🟢")
        st.info(f"**Current League:** {LEAGUE_PARAMS[selected_league]['display_name']}")
        st.info("**Model Version:** 4.0.0 Multi-League")
        st.info("**Risk Profile:** Conservative")
        
        st.markdown("---")
        st.markdown("### 🌍 Supported Leagues")
        for league_id, league_info in LEAGUE_PARAMS.items():
            st.write(f"• {league_info['display_name']}")
        
        st.markdown("---")
        st.markdown("### 💡 Professional Features")
        st.write("• 10 League Support")
        st.write("• Unified Adaptive xG Logic")
        st.write("• League-Aware Calibration")
        st.write("• Professional Risk Management")
        st.write("• Market Reality Checks")
    
    # Main content area
    if st.session_state.professional_results:
        # Display existing results
        results = st.session_state.professional_results
        
        # Current league badge
        league_badge_class = get_league_badge_class(results['match_info']['league'])
        league_display = results['match_info']['league_display']
        st.markdown(f'<div style="text-align: center;"><span class="professional-badge {league_badge_class}">{league_display}</span></div>', unsafe_allow_html=True)
        
        # Display all professional components
        display_match_overview(results)
        display_probability_analysis(results)
        display_betting_recommendations(results)
        display_match_context(results)
        display_diagnostics(results)
        display_xg_debug(results)
        display_professional_metadata(results)
        
    else:
        # Show input form for new analysis
        match_data = create_professional_input_form(selected_league)
        
        if match_data:
            with st.spinner("🔍 Running professional multi-league analysis..."):
                try:
                    # Generate professional predictions
                    engine = MultiLeaguePredictionEngine(match_data)
                    results = engine.generate_predictions()
                    
                    # Store in session state
                    st.session_state.professional_results = results
                    
                    # Add to history
                    history_entry = {
                        'timestamp': datetime.now().isoformat(),
                        'match': results['match_info']['match'],
                        'league': results['match_info']['league'],
                        'league_display': results['match_info']['league_display'],
                        'home_xg': results['expected_goals']['home'],
                        'away_xg': results['expected_goals']['away'],
                        'recommendations': results['market_analysis']['recommendations']
                    }
                    st.session_state.prediction_history.append(history_entry)
                    
                    # Success message
                    st.success("✅ Professional multi-league analysis completed successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.info("Please check your input data and try again.")

if __name__ == "__main__":
    main()
