# streamlit_app.py - COMPLETE PRODUCTION-READY DASHBOARD
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import json

# Import our engines
from prediction_engine import PredictionEngine, TeamData, MatchContext, test_prediction_engine
from betting_engine import BettingDecisionEngine, test_betting_engine

# Page configuration
st.set_page_config(
    page_title="⚽ Advanced Football Predictor Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem !important;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.4rem !important;
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
    .betting-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid #FF9800;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .risk-low { border-left-color: #4CAF50 !important; }
    .risk-medium { border-left-color: #FF9800 !important; }
    .risk-high { border-left-color: #f44336 !important; }
    .confidence-high { background: #4CAF50; color: white; }
    .confidence-medium { background: #FF9800; color: white; }
    .confidence-low { background: #f44336; color: white; }
    .value-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    .value-exceptional { background: #4CAF50; color: white; }
    .value-high { background: #8BC34A; color: white; }
    .value-good { background: #FFC107; color: white; }
    .value-moderate { background: #FF9800; color: white; }
    .section-title {
        font-size: 1.6rem;
        font-weight: 600;
        color: #333;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #f0f2f6;
    }
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'betting_recommendations' not in st.session_state:
        st.session_state.betting_recommendations = None
    if 'bankroll' not in st.session_state:
        st.session_state.bankroll = 1000.0

def create_team_inputs(team_type: str):
    """Create team data input form"""
    st.subheader(f"🏠 {team_type} Team Data" if team_type == "Home" else f"✈️ {team_type} Team Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        name = st.text_input(f"{team_type} Team Name", value="Liverpool" if team_type == "Home" else "Arsenal")
        xg_for = st.number_input(f"{team_type} xG For", min_value=0.5, max_value=4.0, value=2.1 if team_type == "Home" else 1.9, step=0.1)
        xg_against = st.number_input(f"{team_type} xG Against", min_value=0.5, max_value=4.0, value=1.2 if team_type == "Home" else 1.1, step=0.1)
    
    with col2:
        xg_home = st.number_input(f"{team_type} xG Home", min_value=0.5, max_value=4.0, value=2.3 if team_type == "Home" else 2.0, step=0.1)
        xg_away = st.number_input(f"{team_type} xG Away", min_value=0.5, max_value=4.0, value=1.9 if team_type == "Home" else 1.8, step=0.1)
        form_attack = st.number_input(f"{team_type} Form Attack", min_value=0.5, max_value=3.0, value=1.8 if team_type == "Home" else 1.7, step=0.1)
    
    with col3:
        form_defense = st.number_input(f"{team_type} Form Defense", min_value=0.5, max_value=3.0, value=1.1 if team_type == "Home" else 1.0, step=0.1)
        avg_possession = st.number_input(f"{team_type} Avg Possession %", min_value=30.0, max_value=80.0, value=58.5 if team_type == "Home" else 55.2, step=0.1)
        pressures_p90 = st.number_input(f"{team_type} Pressures p90", min_value=100.0, max_value=250.0, value=185.0 if team_type == "Home" else 192.0, step=1.0)
    
    return TeamData(
        name=name,
        xg_for=xg_for,
        xg_against=xg_against,
        xg_home=xg_home,
        xg_away=xg_away,
        form_attack=form_attack,
        form_defense=form_defense,
        avg_possession=avg_possession,
        pressures_p90=pressures_p90,
        shots_p90=16.2 if team_type == "Home" else 14.8,
        conversion_rate=0.12 if team_type == "Home" else 0.11
    )

def create_context_inputs():
    """Create match context input form"""
    st.subheader("🎭 Match Context")
    
    col1, col2 = st.columns(2)
    
    with col1:
        home_motivation = st.selectbox("Home Team Motivation", ["Low", "Normal", "High", "Very High"], index=2)
        home_injuries = st.slider("Home Key Injuries", 0, 5, 1)
        match_importance = st.slider("Match Importance", 0.0, 1.0, 0.8, 0.1)
    
    with col2:
        away_motivation = st.selectbox("Away Team Motivation", ["Low", "Normal", "High", "Very High"], index=2)
        away_injuries = st.slider("Away Key Injuries", 0, 5, 2)
        days_since_last = st.slider("Days Since Last Game", 2, 14, 7)
    
    is_derby = st.checkbox("Is Derby Match")
    
    return MatchContext(
        home_motivation=home_motivation,
        away_motivation=away_motivation,
        home_injuries=home_injuries,
        away_injuries=away_injuries,
        match_importance=match_importance,
        is_derby=is_derby,
        days_since_last=days_since_last
    )

def create_odds_inputs():
    """Create market odds input form"""
    st.subheader("💰 Market Odds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**1X2 Markets**")
        home_odds = st.number_input("Home Win Odds", min_value=1.01, value=2.10, step=0.01)
        draw_odds = st.number_input("Draw Odds", min_value=1.01, value=3.50, step=0.01)
        away_odds = st.number_input("Away Win Odds", min_value=1.01, value=3.80, step=0.01)
    
    with col2:
        st.write("**Goal Markets**")
        btts_yes = st.number_input("BTTS Yes Odds", min_value=1.01, value=1.80, step=0.01)
        btts_no = st.number_input("BTTS No Odds", min_value=1.01, value=2.10, step=0.01)
        over_25 = st.number_input("Over 2.5 Goals Odds", min_value=1.01, value=1.65, step=0.01)
        under_25 = st.number_input("Under 2.5 Goals Odds", min_value=1.01, value=2.20, step=0.01)
    
    return {
        '1x2 Home': home_odds,
        '1x2 Draw': draw_odds,
        '1x2 Away': away_odds,
        'BTTS Yes': btts_yes,
        'BTTS No': btts_no,
        'Over 2.5 Goals': over_25,
        'Under 2.5 Goals': under_25
    }

def display_predictions(predictions):
    """Display prediction results"""
    st.markdown('<div class="section-title">🎯 Football Predictions</div>', unsafe_allow_html=True)
    
    # Match header
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        st.markdown(f"### 🏠 {predictions['match'].split(' vs ')[0]}")
    with col2:
        st.markdown("### vs")
    with col3:
        st.markdown(f"### ✈️ {predictions['match'].split(' vs ')[1]}")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Expected Goals Home", f"{predictions['expected_goals']['home']:.2f}")
    with col2:
        st.metric("Expected Goals Away", f"{predictions['expected_goals']['away']:.2f}")
    with col3:
        st.metric("Total Expected Goals", f"{predictions['expected_goals']['total']:.2f}")
    with col4:
        confidence_color = {
            'HIGH': 'confidence-high',
            'MEDIUM': 'confidence-medium', 
            'LOW': 'confidence-low'
        }.get(predictions['confidence'], 'confidence-medium')
        st.markdown(f'<div class="value-badge {confidence_color}">Confidence: {predictions["confidence"]}</div>', unsafe_allow_html=True)
    
    # Match outcomes
    st.subheader("📈 Match Outcome Probabilities")
    outcomes = predictions['probabilities']['match_outcomes']
    
    fig_outcomes = go.Figure(data=[
        go.Bar(name='Probability', x=list(outcomes.keys()), y=list(outcomes.values()),
               text=[f"{v}%" for v in outcomes.values()], textposition='auto')
    ])
    fig_outcomes.update_layout(
        title="Match Outcome Probabilities",
        yaxis_title="Probability (%)",
        showlegend=False
    )
    st.plotly_chart(fig_outcomes, use_container_width=True)
    
    # Goal markets
    st.subheader("⚽ Goal Market Probabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # BTTS probabilities
        btts_probs = predictions['probabilities']['both_teams_score']
        fig_btts = go.Figure(data=[
            go.Pie(labels=list(btts_probs.keys()), values=list(btts_probs.values()),
                   textinfo='label+percent', hole=0.4)
        ])
        fig_btts.update_layout(title="Both Teams to Score")
        st.plotly_chart(fig_btts, use_container_width=True)
    
    with col2:
        # Over/Under probabilities
        ou_probs = predictions['probabilities']['over_under']
        fig_ou = go.Figure(data=[
            go.Bar(x=list(ou_probs.keys()), y=list(ou_probs.values()),
                   text=[f"{v}%" for v in ou_probs.values()], textposition='auto')
        ])
        fig_ou.update_layout(title="Over/Under Probabilities", yaxis_title="Probability (%)")
        st.plotly_chart(fig_ou, use_container_width=True)
    
    # Exact scores
    st.subheader("🎯 Most Likely Scores")
    exact_scores = predictions['probabilities']['exact_scores']
    
    if exact_scores:
        score_cols = st.columns(6)
        for idx, (score, prob) in enumerate(exact_scores.items()):
            with score_cols[idx]:
                st.metric(f"{score}", f"{prob}%")
    
    # Explanation
    st.subheader("📝 Analysis Explanation")
    explanation = predictions.get('explanation', {})
    
    if 'summary' in explanation:
        for point in explanation['summary']:
            st.info(point)
    
    if 'key_factors' in explanation and explanation['key_factors']:
        st.write("**Key Factors:**")
        for factor in explanation['key_factors']:
            st.write(f"• {factor}")

def display_betting_recommendations(recommendations):
    """Display betting recommendations"""
    st.markdown('<div class="section-title">💰 Betting Recommendations</div>', unsafe_allow_html=True)
    
    # Summary
    summary = recommendations['summary']
    risk_assessment = recommendations['risk_assessment']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Recommendations", recommendations['recommendation_count'])
    with col2:
        st.metric("Total Stake", f"${summary['total_stake']:.2f}")
    with col3:
        st.metric("Total Expected Value", f"${summary['total_ev']:.2f}")
    with col4:
        risk_color = {
            'LOW': 'risk-low',
            'MEDIUM': 'risk-medium',
            'HIGH': 'risk-high'
        }.get(risk_assessment['risk_level'], 'risk-medium')
        st.markdown(f'<div class="value-badge {risk_color}">Risk: {risk_assessment["risk_level"]}</div>', unsafe_allow_html=True)
    
    # Betting signals
    signals = recommendations['signals']
    
    if not signals:
        st.info("No value bets identified for this match.")
        return
    
    st.subheader("🎯 Recommended Bets")
    
    for signal in signals:
        confidence_color = {
            'HIGH': 'value-exceptional',
            'MEDIUM': 'value-high',
            'LOW': 'value-good',
            'SPECULATIVE': 'value-moderate'
        }.get(signal['confidence'], 'value-moderate')
        
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 2, 2, 3])
            
            with col1:
                st.write(f"**{signal['market']}**")
                st.write(f"Model Probability: {signal['model_probability']:.1%}")
                st.write(f"Market Odds: {signal['market_odds']}")
            
            with col2:
                st.write(f"**Edge:** +{signal['edge_percentage']:.1f}%")
                st.write(f"**EV:** {signal['expected_value']:.3f}")
                st.markdown(f'<div class="value-badge {confidence_color}">{signal["confidence"]}</div>', unsafe_allow_html=True)
            
            with col3:
                st.write(f"**Stake:** ${signal['recommended_stake']:.2f}")
                st.write(f"**Kelly:** {signal['kelly_fraction']:.1%}")
            
            with col4:
                if signal['explanation']:
                    with st.expander("Explanation"):
                        for exp in signal['explanation']:
                            st.write(f"• {exp}")
    
    # Risk assessment details
    st.subheader("📊 Risk Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Portfolio Metrics:**")
        st.write(f"• Total Exposure: ${risk_assessment['total_exposure']:.2f}")
        st.write(f"• Exposure Percentage: {risk_assessment['exposure_percentage']:.1%}")
        st.write(f"• Number of Bets: {risk_assessment['number_of_bets']}")
        st.write(f"• Average Confidence: {risk_assessment['average_confidence']:.2f}")
    
    with col2:
        st.write("**Recommendations:**")
        if risk_assessment['recommendations']:
            for rec in risk_assessment['recommendations']:
                st.warning(rec)
        else:
            st.success("Portfolio risk within acceptable limits")

def create_probability_gauge(probability, title, color):
    """Create a probability gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 33], 'color': "lightgray"},
                {'range': [33, 66], 'color': "gray"},
                {'range': [66, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=250)
    return fig

def main():
    """Main application function"""
    
    st.markdown('<p class="main-header">⚽ Advanced Football Predictor Pro</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Professional Multi-League Analysis & Betting Intelligence</p>', unsafe_allow_html=True)
    
    initialize_session_state()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode", 
                                   ["🏠 Home", "🎯 Predictions", "💰 Betting", "📊 Analytics"])
    
    if app_mode == "🏠 Home":
        display_home()
    elif app_mode == "🎯 Predictions":
        display_predictions_tab()
    elif app_mode == "💰 Betting":
        display_betting_tab()
    elif app_mode == "📊 Analytics":
        display_analytics_tab()

def display_home():
    """Display home page"""
    
    st.markdown("""
    ## 🎯 Welcome to Advanced Football Predictor Pro
    
    This professional tool combines advanced statistical modeling with intelligent betting analysis 
    to provide comprehensive football match predictions and value betting opportunities.
    
    ### 🚀 Key Features:
    
    **🎯 Prediction Engine**
    - Advanced expected goals (xG) modeling
    - Monte Carlo simulations with Dixon-Coles correlation
    - Team strength analysis with Bayesian shrinkage
    - Context-aware probability adjustments
    
    **💰 Betting Intelligence**
    - Expected Value (EV) calculation
    - Kelly Criterion stake management
    - Portfolio risk assessment
    - Market edge detection
    
    **📊 Professional Analytics**
    - Interactive visualizations
    - Real-time probability gauges
    - Risk management tools
    - Performance tracking
    
    ### 🎮 How to Use:
    
    1. **Navigate to Predictions** - Input team data and match context
    2. **Generate Predictions** - Get detailed probability analysis
    3. **Check Betting Tab** - Find value betting opportunities
    4. **Review Analytics** - Monitor performance and risk
    
    ### 🏗️ Technical Architecture:
    
    - **Machine Learning**: Gradient Boosting + Calibrated Classifiers
    - **Statistical Modeling**: Poisson & Negative Binomial Distributions
    - **Simulation**: Monte Carlo with Dixon-Coles Correlation
    - **Risk Management**: Fractional Kelly + Portfolio Optimization
    """)
    
    # Quick start example
    st.markdown("---")
    st.subheader("🚀 Quick Start Example")
    
    if st.button("Run Demo Analysis", type="primary"):
        with st.spinner("Running demo analysis..."):
            # Generate sample predictions
            predictions = test_prediction_engine()
            betting_recs = test_betting_engine()
            
            st.session_state.predictions = predictions
            st.session_state.betting_recommendations = betting_recs
            
            st.success("Demo analysis completed! Navigate to Predictions and Betting tabs to see results.")
            st.rerun()

def display_predictions_tab():
    """Display predictions tab"""
    
    st.header("🎯 Match Predictions")
    
    # Input form
    with st.form("prediction_form"):
        st.subheader("📋 Match Input Data")
        
        # League selection
        league = st.selectbox("League", 
                            ["premier_league", "la_liga", "serie_a", 
                             "bundesliga", "ligue_1", "eredivisie"])
        
        # Team inputs
        home_team = create_team_inputs("Home")
        away_team = create_team_inputs("Away")
        
        # Context inputs
        context = create_context_inputs()
        
        # Submit button
        submitted = st.form_submit_button("🎯 Generate Predictions", type="primary")
    
    # Generate predictions
    if submitted:
        with st.spinner("Running advanced prediction analysis..."):
            try:
                engine = PredictionEngine()
                predictions = engine.predict_match(home_team, away_team, context, league)
                st.session_state.predictions = predictions
                st.success("Predictions generated successfully!")
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
    
    # Display results
    if st.session_state.predictions:
        display_predictions(st.session_state.predictions)

def display_betting_tab():
    """Display betting tab"""
    
    st.header("💰 Betting Intelligence")
    
    if not st.session_state.predictions:
        st.warning("Please generate predictions first in the Predictions tab.")
        return
    
    # Bankroll management
    st.subheader("🏦 Bankroll Management")
    col1, col2 = st.columns(2)
    
    with col1:
        bankroll = st.number_input("Bankroll Amount ($)", min_value=100.0, max_value=10000.0, 
                                 value=st.session_state.bankroll, step=100.0)
        st.session_state.bankroll = bankroll
    
    with col2:
        st.metric("Available Bankroll", f"${bankroll:,.2f}")
    
    # Odds inputs
    st.subheader("📊 Market Odds")
    market_odds = create_odds_inputs()
    
    # Generate betting recommendations
    if st.button("💰 Generate Betting Recommendations", type="primary"):
        with st.spinner("Analyzing betting markets..."):
            try:
                betting_engine = BettingDecisionEngine(bankroll_amount=bankroll)
                recommendations = betting_engine.generate_recommendations(
                    st.session_state.predictions, market_odds
                )
                st.session_state.betting_recommendations = recommendations
                st.success("Betting analysis completed!")
            except Exception as e:
                st.error(f"Betting analysis error: {str(e)}")
    
    # Display recommendations
    if st.session_state.betting_recommendations:
        display_betting_recommendations(st.session_state.betting_recommendations)

def display_analytics_tab():
    """Display analytics tab"""
    
    st.header("📊 Advanced Analytics")
    
    if not st.session_state.predictions:
        st.warning("Please generate predictions first to view analytics.")
        return
    
    predictions = st.session_state.predictions
    
    # Probability gauges
    st.subheader("🎯 Probability Gauges")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        btts_yes_prob = predictions['probabilities']['both_teams_score']['yes']
        fig_btts = create_probability_gauge(btts_yes_prob, "BTTS Yes", "blue")
        st.plotly_chart(fig_btts, use_container_width=True)
    
    with col2:
        over_25_prob = predictions['probabilities']['over_under']['over_25']
        fig_over = create_probability_gauge(over_25_prob, "Over 2.5 Goals", "green")
        st.plotly_chart(fig_over, use_container_width=True)
    
    with col3:
        home_win_prob = predictions['probabilities']['match_outcomes']['home_win']
        fig_home = create_probability_gauge(home_win_prob, "Home Win", "orange")
        st.plotly_chart(fig_home, use_container_width=True)
    
    # Expected goals breakdown
    st.subheader("⚽ Expected Goals Analysis")
    
    xg_data = predictions['expected_goals']
    fig_xg = go.Figure(data=[
        go.Bar(name='Expected Goals', x=['Home', 'Away', 'Total'], 
               y=[xg_data['home'], xg_data['away'], xg_data['total']])
    ])
    fig_xg.update_layout(title="Expected Goals Breakdown", yaxis_title="Expected Goals")
    st.plotly_chart(fig_xg, use_container_width=True)
    
    # Simulation statistics
    if 'simulation_stats' in predictions:
        st.subheader("🎲 Simulation Statistics")
        
        sim_stats = predictions['simulation_stats']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Avg Home Goals", f"{sim_stats['avg_home_goals']:.2f}")
        with col2:
            st.metric("Avg Away Goals", f"{sim_stats['avg_away_goals']:.2f}")
        with col3:
            st.metric("Most Common Score", sim_stats['most_common_score'])
        with col4:
            st.metric("Home Clean Sheet", f"{sim_stats['clean_sheet_prob']['home']}%")
    
    # Betting performance (if available)
    if st.session_state.betting_recommendations:
        st.subheader("💰 Betting Performance")
        
        betting_data = st.session_state.betting_recommendations
        signals = betting_data['signals']
        
        if signals:
            # Create performance summary
            edges = [s['edge_percentage'] for s in signals]
            stakes = [s['recommended_stake'] for s in signals]
            evs = [s['expected_value'] for s in signals]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average Edge", f"{np.mean(edges):.1f}%")
            with col2:
                st.metric("Total Stake", f"${sum(stakes):.2f}")
            with col3:
                st.metric("Total EV", f"${sum(evs):.2f}")
            with col4:
                st.metric("Avg Confidence", betting_data['risk_assessment']['average_confidence'])

if __name__ == "__main__":
    main()
