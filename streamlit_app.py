# streamlit_app.py - PROFESSIONAL PREDICTION INTERFACE
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# Import the professional engine
from prediction_engine import ApexProfessionalEngine, LEAGUE_PARAMS, VOLATILITY_MULTIPLIERS

st.set_page_config(
    page_title="🎯 Professional Football Predictor",
    page_icon="⚽", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .professional-header { 
        font-size: 2.5rem !important; 
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .edge-positive { border-left-color: #4CAF50; background: #f8fff8; }
    .edge-negative { border-left-color: #f44336; background: #fff5f5; }
    .edge-neutral { border-left-color: #FF9800; background: #fffaf2; }
    
    .robust-true { background: #e8f5e8; color: #2E7D32; padding: 0.3rem 0.7rem; border-radius: 12px; }
    .robust-false { background: #ffebee; color: #c62828; padding: 0.3rem 0.7rem; border-radius: 12px; }
    
    .stake-recommended { background: #4CAF50; color: white; padding: 0.5rem 1rem; border-radius: 15px; font-weight: bold; }
    .stake-not-recommended { background: #9E9E9E; color: white; padding: 0.5rem 1rem; border-radius: 15px; }
    
    .context-badge {
        background: #667eea;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

def display_professional_dashboard(predictions: dict):
    """Display professional prediction dashboard"""
    st.markdown('<p class="professional-header">🎯 Professional Football Predictions</p>', unsafe_allow_html=True)
    
    # Key metrics header
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        xg_home = predictions['expected_goals']['home']
        xg_away = predictions['expected_goals']['away']
        st.metric("🏠 Expected Goals", f"{xg_home:.2f}")
    
    with col2:
        st.metric("✈️ Expected Goals", f"{xg_away:.2f}")
    
    with col3:
        total_xg = predictions['expected_goals']['total']
        st.metric("📊 Total xG", f"{total_xg:.2f}")
    
    with col4:
        context = predictions['descriptive_context']['match_context']
        st.metric("🎯 Match Context", context.replace('_', ' ').title())
    
    # Market edges section
    st.markdown("### 💰 Professional Market Analysis")
    
    edges = predictions['market_analysis']['edges']
    min_edge_required = predictions['market_analysis']['min_edge_required']
    
    edge_cols = st.columns(4)
    
    with edge_cols[0]:
        home_edge = edges.get('home_win', 0)
        edge_class = "edge-positive" if home_edge >= min_edge_required else "edge-negative"
        st.markdown(f'''
        <div class="metric-card {edge_class}">
            <h4>🏠 Home Win</h4>
            <div style="font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0;">
                {home_edge:+.1f}%
            </div>
            <div style="color: #666;">
                Min: {min_edge_required}%
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    with edge_cols[1]:
        away_edge = edges.get('away_win', 0)
        edge_class = "edge-positive" if away_edge >= min_edge_required else "edge-negative"
        st.markdown(f'''
        <div class="metric-card {edge_class}">
            <h4>✈️ Away Win</h4>
            <div style="font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0;">
                {away_edge:+.1f}%
            </div>
            <div style="color: #666;">
                Min: {min_edge_required}%
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    with edge_cols[2]:
        btts_edge = edges.get('btts_yes', 0)
        edge_class = "edge-positive" if btts_edge >= min_edge_required else "edge-negative"
        st.markdown(f'''
        <div class="metric-card {edge_class}">
            <h4>⚽ BTTS Yes</h4>
            <div style="font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0;">
                {btts_edge:+.1f}%
            </div>
            <div style="color: #666;">
                Min: {min_edge_required}%
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    with edge_cols[3]:
        over_edge = edges.get('over_25', 0)
        edge_class = "edge-positive" if over_edge >= min_edge_required else "edge-negative"
        st.markdown(f'''
        <div class="metric-card {edge_class}">
            <h4>📈 Over 2.5</h4>
            <div style="font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0;">
                {over_edge:+.1f}%
            </div>
            <div style="color: #666;">
                Min: {min_edge_required}%
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Staking recommendations
    st.markdown("### 💵 Professional Staking Recommendations")
    
    stakes = predictions['market_analysis']['recommended_stakes']
    bankroll = 1000  # Default
    
    stake_cols = st.columns(4)
    
    with stake_cols[0]:
        home_stake = stakes.get('home_win', 0)
        if home_edge >= min_edge_required and home_stake > 0:
            st.markdown(f'<div class="stake-recommended">${home_stake:.2f}</div>', unsafe_allow_html=True)
            st.metric("Home Win Stake", f"${home_stake:.2f}", f"{(home_stake/bankroll)*100:.1f}%")
        else:
            st.markdown('<div class="stake-not-recommended">NO BET</div>', unsafe_allow_html=True)
    
    with stake_cols[1]:
        away_stake = stakes.get('away_win', 0)
        if away_edge >= min_edge_required and away_stake > 0:
            st.markdown(f'<div class="stake-recommended">${away_stake:.2f}</div>', unsafe_allow_html=True)
            st.metric("Away Win Stake", f"${away_stake:.2f}", f"{(away_stake/bankroll)*100:.1f}%")
        else:
            st.markdown('<div class="stake-not-recommended">NO BET</div>', unsafe_allow_html=True)
    
    with stake_cols[2]:
        btts_stake = stakes.get('btts_yes', 0)
        if btts_edge >= min_edge_required and btts_stake > 0:
            st.markdown(f'<div class="stake-recommended">${btts_stake:.2f}</div>', unsafe_allow_html=True)
            st.metric("BTTS Yes Stake", f"${btts_stake:.2f}", f"{(btts_stake/bankroll)*100:.1f}%")
        else:
            st.markdown('<div class="stake-not-recommended">NO BET</div>', unsafe_allow_html=True)
    
    with stake_cols[3]:
        over_stake = stakes.get('over_25', 0)
        if over_edge >= min_edge_required and over_stake > 0:
            st.markdown(f'<div class="stake-recommended">${over_stake:.2f}</div>', unsafe_allow_html=True)
            st.metric("Over 2.5 Stake", f"${over_stake:.2f}", f"{(over_stake/bankroll)*100:.1f}%")
        else:
            st.markdown('<div class="stake-not-recommended">NO BET</div>', unsafe_allow_html=True)
    
    # Robustness analysis
    st.markdown("### 🔬 Robustness Analysis")
    
    robustness = predictions['robustness_analysis']
    col1, col2 = st.columns(2)
    
    with col1:
        if robustness['edge_survives']:
            st.markdown('<div class="robust-true">✅ EDGE SURVIVES SENSITIVITY TESTING</div>', unsafe_allow_html=True)
            st.success("This edge is robust to ±20% changes in xG estimates")
        else:
            st.markdown('<div class="robust-false">❌ EDGE FAILS SENSITIVITY TESTING</div>', unsafe_allow_html=True)
            st.warning("This edge is fragile to changes in xG estimates")
    
    with col2:
        base_edge = robustness['base_edge'].get('home_win', 0) * 100
        st.metric("Base Edge", f"{base_edge:+.1f}%")
        st.metric("Minimum Required", f"{min_edge_required}%")
    
    # Probability distributions
    st.markdown("### 📊 Probability Distributions")
    
    probs = predictions['probabilities']['match_outcomes']
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Home Win Probability", f"{probs['home_win']}%")
        st.progress(probs['home_win'] / 100)
    
    with col2:
        st.metric("Draw Probability", f"{probs['draw']}%")
        st.progress(probs['draw'] / 100)
    
    with col3:
        st.metric("Away Win Probability", f"{probs['away_win']}%")
        st.progress(probs['away_win'] / 100)
    
    # Exact scores
    st.markdown("### 🎯 Most Likely Scores")
    exact_scores = predictions['probabilities']['exact_scores']
    
    if exact_scores:
        score_cols = st.columns(6)
        for idx, (score, prob) in enumerate(list(exact_scores.items())[:6]):
            with score_cols[idx]:
                st.metric(f"{score}", f"{prob*100:.1f}%")
    
    # Professional metrics
    st.markdown("### ⚙️ Professional Metrics")
    
    metrics = predictions['professional_metrics']
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    
    with mcol1:
        st.metric("Data Quality", f"{metrics['data_quality_score']}/100")
    
    with mcol2:
        st.metric("Calibration", metrics['model_calibration_status'])
    
    with mcol3:
        st.metric("Uncertainty", metrics['uncertainty_propagation'])
    
    with mcol4:
        st.metric("Sensitivity Tested", "✅" if metrics['sensitivity_tested'] else "❌")
    
    # Context explanation
    st.markdown("### 📝 Match Analysis")
    context_info = predictions['descriptive_context']
    st.info(f"**{context_info['match_context'].replace('_', ' ').title()}**: {context_info['explanation']}")

def create_professional_input_form():
    """Create professional input form"""
    st.markdown('<p class="professional-header">🎯 Professional Football Predictor</p>', unsafe_allow_html=True)
    
    # League selection in sidebar
    st.sidebar.markdown("### 🌍 League Configuration")
    
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
    
    selected_league = st.sidebar.selectbox(
        "Select League",
        options=list(league_options.keys()),
        format_func=lambda x: league_options[x],
        key="professional_league"
    )
    
    # Display league intelligence
    league_params = LEAGUE_PARAMS.get(selected_league, LEAGUE_PARAMS['premier_league'])
    volatility_multiplier = VOLATILITY_MULTIPLIERS.get(league_params['volatility'], 1.0)
    
    st.sidebar.markdown(f"""
    **League Intelligence:**
    - Volatility: {league_params['volatility'].upper()}
    - Away Penalty: {league_params['away_penalty']}
    - Min Edge: {league_params['min_edge']*100:.1f}%
    - Stake Multiplier: {volatility_multiplier:.1f}x
    """)
    
    # Main input form
    st.markdown("### 📊 Match Data Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏠 Home Team")
        home_team = st.text_input("Home Team Name", "Tottenham", key="pro_home_team")
        home_goals = st.number_input("Home Goals (Last 6 Games)", min_value=0, value=8, key="pro_home_goals")
        home_conceded = st.number_input("Home Conceded (Last 6 Games)", min_value=0, value=6, key="pro_home_conceded")
    
    with col2:
        st.subheader("✈️ Away Team") 
        away_team = st.text_input("Away Team Name", "Chelsea", key="pro_away_team")
        away_goals = st.number_input("Away Goals (Last 6 Games)", min_value=0, value=10, key="pro_away_goals")
        away_conceded = st.number_input("Away Conceded (Last 6 Games)", min_value=0, value=7, key="pro_away_conceded")
    
    st.markdown("### 💰 Market Odds")
    
    odds_col1, odds_col2 = st.columns(2)
    
    with odds_col1:
        st.write("**1X2 Market**")
        home_odds = st.number_input("Home Win Odds", min_value=1.01, value=3.10, key="pro_home_odds")
        draw_odds = st.number_input("Draw Odds", min_value=1.01, value=3.40, key="pro_draw_odds")
        away_odds = st.number_input("Away Win Odds", min_value=1.01, value=2.30, key="pro_away_odds")
    
    with odds_col2:
        st.write("**Goals Markets**")
        over_25_odds = st.number_input("Over 2.5 Goals", min_value=1.01, value=1.80, key="pro_over_25_odds")
        under_25_odds = st.number_input("Under 2.5 Goals", min_value=1.01, value=2.00, key="pro_under_25_odds")
        btts_yes_odds = st.number_input("BTTS Yes", min_value=1.01, value=1.70, key="pro_btts_yes_odds")
        btts_no_odds = st.number_input("BTTS No", min_value=1.01, value=2.10, key="pro_btts_no_odds")
    
    # Bankroll configuration
    st.sidebar.markdown("### 💵 Bankroll Management")
    bankroll = st.sidebar.number_input("Bankroll ($)", min_value=500, value=1000, key="pro_bankroll")
    
    # Generate predictions
    if st.button("🎯 GENERATE PROFESSIONAL ANALYSIS", type="primary", use_container_width=True):
        if not home_team or not away_team:
            st.error("❌ Please enter both team names")
            return None
            
        if home_team == away_team:
            st.error("❌ Home and away teams cannot be the same")
            return None
        
        # Prepare market odds
        market_odds = {
            '1x2 Home': home_odds,
            '1x2 Draw': draw_odds,
            '1x2 Away': away_odds,
            'Over 2.5 Goals': over_25_odds,
            'Under 2.5 Goals': under_25_odds,
            'BTTS Yes': btts_yes_odds,
            'BTTS No': btts_no_odds
        }
        
        match_data = {
            'home_team': home_team,
            'away_team': away_team,
            'league': selected_league,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'home_conceded': home_conceded,
            'away_conceded': away_conceded,
            'market_odds': market_odds,
            'bankroll': bankroll
        }
        
        return match_data
    
    return None

def main():
    """Main application function"""
    # Initialize session state
    if 'professional_predictions' not in st.session_state:
        st.session_state.professional_predictions = None
    
    # Display existing predictions or create new form
    if st.session_state.professional_predictions:
        display_professional_dashboard(st.session_state.professional_predictions)
        
        st.markdown("---")
        if st.button("🔄 New Analysis", use_container_width=True):
            st.session_state.professional_predictions = None
            st.rerun()
    else:
        match_data = create_professional_input_form()
        
        if match_data:
            with st.spinner("🔍 Running professional analysis with sensitivity testing..."):
                try:
                    engine = ApexProfessionalEngine(match_data)
                    predictions = engine.generate_predictions()
                    
                    st.session_state.professional_predictions = predictions
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Analysis error: {str(e)}")
                    st.info("💡 Check input parameters and try again")

if __name__ == "__main__":
    main()
