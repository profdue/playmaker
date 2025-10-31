import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Import the PREDICTIVE engine
try:
    from prediction_engine import PredictiveFootballEngine, TruePredictiveEngine, SmartValueEngine
except ImportError:
    st.error("❌ Could not import prediction_engine. Make sure the file is in the same directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Predictive Football Analyst ⚽",
    page_icon="⚽", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Predictive-focused CSS
st.markdown("""
<style>
    .predictive-header { 
        font-size: 2.5rem !important; 
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .predictive-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .predictive-high { border-left-color: #4CAF50 !important; background: #f8fff8; }
    .predictive-medium { border-left-color: #FF9800 !important; background: #fffaf2; }
    .predictive-low { border-left-color: #f44336 !important; background: #fff5f5; }
    
    .insight-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .value-signal {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        border-left: 4px solid;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .value-exceptional { border-left-color: #4CAF50 !important; }
    .value-high { border-left-color: #8BC34A !important; }
    .value-good { border-left-color: #FFC107 !important; }
    .value-moderate { border-left-color: #FF9800 !important; }
</style>
""", unsafe_allow_html=True)

def create_predictive_input_form():
    """Create input form focused on PREDICTIVE data"""
    
    st.markdown('<p class="predictive-header">🔮 Predictive Football Analyst</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem;">Future Performance Forecasting, Not Past Results Analysis</p>', unsafe_allow_html=True)
    
    with st.expander("🎯 Predictive System Overview", expanded=True):
        st.markdown("""
        ### 🧠 True Predictive Intelligence
        
        **What Makes This Different:**
        - 🔍 **Future-Focused**: Predicts what WILL happen, not what already happened
        - 📈 **Trend Analysis**: Identifies improving/declining teams
        - 🎯 **Efficiency Metrics**: Values process over results
        - ⚖️ **Regression Awareness**: Knows teams revert to mean performance
        - 🛡️ **Uncertainty Quantification**: Measures prediction confidence
        
        **No More:**
        - "Team A scored more goals → they'll win"
        - Simple form-based predictions
        - Overconfidence in small samples
        """)
    
    tab1, tab2, tab3 = st.tabs(["📊 Predictive Data", "💰 Market Odds", "⚙️ Engine Settings"])

    with tab1:
        st.markdown("### 🎯 Predictive Football Data")
        st.info("Focus on underlying performance, not just results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏠 Home Team Analysis")
            home_team = st.text_input("Team Name", value="Bologna", key="home_team")
            
            st.write("**Underlying Performance**")
            home_xg = st.number_input("Expected Goals (xG)", min_value=0.0, value=1.8, step=0.1, key="home_xg")
            home_xg_against = st.number_input("xG Against", min_value=0.0, value=1.1, step=0.1, key="home_xg_against")
            home_shots = st.number_input("Shots per Game", min_value=0, value=14, key="home_shots")
            home_shots_on_target = st.number_input("Shots on Target", min_value=0, value=5, key="home_shots_on_target")
            
        with col2:
            st.subheader("✈️ Away Team Analysis")
            away_team = st.text_input("Team Name", value="Torino", key="away_team")
            
            st.write("**Underlying Performance**")
            away_xg = st.number_input("Expected Goals (xG)", min_value=0.0, value=1.2, step=0.1, key="away_xg")
            away_xg_against = st.number_input("xG Against", min_value=0.0, value=1.3, step=0.1, key="away_xg_against")
            away_shots = st.number_input("Shots per Game", min_value=0, value=11, key="away_shots")
            away_shots_on_target = st.number_input("Shots on Target", min_value=0, value=4, key="away_shots_on_target")
        
        # Performance Trends
        with st.expander("📈 Performance Trends & Momentum", expanded=True):
            trend_col1, trend_col2 = st.columns(2)
            
            with trend_col1:
                st.write(f"**{home_team} Recent Form (Last 6)**")
                home_form = st.multiselect(
                    "Performance Rating (1-10)",
                    options=[f"Rating {i}" for i in range(1, 11)],
                    default=["Rating 8", "Rating 7", "Rating 9", "Rating 6", "Rating 8", "Rating 7"],
                    key="home_form"
                )
                
            with trend_col2:
                st.write(f"**{away_team} Recent Form (Last 6)**")
                away_form = st.multiselect(
                    "Performance Rating (1-10)",
                    options=[f"Rating {i}" for i in range(1, 11)],
                    default=["Rating 6", "Rating 7", "Rating 5", "Rating 6", "Rating 7", "Rating 6"],
                    key="away_form"
                )

    with tab2:
        st.markdown("### 💰 Market Odds Input")
        
        odds_col1, odds_col2 = st.columns(2)
        
        with odds_col1:
            st.write("**Match Outcomes**")
            home_odds = st.number_input("Home Win", min_value=1.01, value=2.10, step=0.01, key="home_odds")
            draw_odds = st.number_input("Draw", min_value=1.01, value=3.10, step=0.01, key="draw_odds")
            away_odds = st.number_input("Away Win", min_value=1.01, value=3.80, step=0.01, key="away_odds")
        
        with odds_col2:
            st.write("**Goals Markets**")
            over_25_odds = st.number_input("Over 2.5 Goals", min_value=1.01, value=2.30, step=0.01, key="over_25_odds")
            under_25_odds = st.number_input("Under 2.5 Goals", min_value=1.01, value=1.65, step=0.01, key="under_25_odds")
            btts_yes_odds = st.number_input("BTTS Yes", min_value=1.01, value=1.95, step=0.01, key="btts_yes_odds")
            btts_no_odds = st.number_input("BTTS No", min_value=1.01, value=1.75, step=0.01, key="btts_no_odds")

    with tab3:
        st.markdown("### ⚙️ Predictive Engine Settings")
        
        league = st.selectbox("League", [
            "premier_league", "la_liga", "serie_a", "bundesliga", "ligue_1", "default"
        ], index=2, key="league")
        
        mc_iterations = st.select_slider(
            "Simulation Precision",
            options=[5000, 10000, 25000, 50000],
            value=10000,
            key="mc_iterations"
        )
        
        st.info("""
        **Predictive Features Active:**
        - Trend analysis ✓
        - Regression to mean ✓  
        - Efficiency metrics ✓
        - Consistency scoring ✓
        - Market wisdom adjustment ✓
        """)

    # Generate analysis
    submitted = st.button("🔮 GENERATE PREDICTIVE ANALYSIS", type="primary", use_container_width=True)
    
    if submitted:
        if not home_team or not away_team:
            st.error("❌ Please enter both team names")
            return None, None
        
        # Convert form to performance ratings
        home_form_ratings = [int(rating.split()[-1]) for rating in home_form]
        away_form_ratings = [int(rating.split()[-1]) for rating in away_form]
        
        # Market odds
        market_odds = {
            '1x2 Home': home_odds,
            '1x2 Draw': draw_odds,
            '1x2 Away': away_odds,
            'Over 2.5 Goals': over_25_odds,
            'Under 2.5 Goals': under_25_odds,
            'BTTS Yes': btts_yes_odds,
            'BTTS No': btts_no_odds,
        }
        
        # Complete predictive data
        match_data = {
            'home_team': home_team,
            'away_team': away_team,
            'league': league,
            'home_xg': home_xg,
            'away_xg': away_xg,
            'home_xg_against': home_xg_against,
            'away_xg_against': away_xg_against,
            'home_shots': home_shots,
            'away_shots': away_shots,
            'home_shots_on_target': home_shots_on_target,
            'away_shots_on_target': away_shots_on_target,
            'home_form': home_form_ratings,
            'away_form': away_form_ratings,
            'market_odds': market_odds
        }
        
        return match_data, mc_iterations
    
    return None, None

def display_predictive_analysis(predictions):
    """Display truly predictive analysis"""
    
    st.markdown('<p class="predictive-header">🔮 Predictive Analysis Results</p>', unsafe_allow_html=True)
    
    # Key predictive metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        xg = predictions['expected_goals']
        st.metric("🏠 Predictive xG", f"{xg['home']:.2f}")
    with col2:
        st.metric("✈️ Predictive xG", f"{xg['away']:.2f}")
    with col3:
        context = predictions['match_context']
        st.metric("Match Context", context.replace('_', ' ').title())
    with col4:
        confidence = predictions['confidence_score']
        st.metric("Predictive Confidence", f"{confidence}%")
    
    # Main insights
    st.markdown("### 🧠 Predictive Insights")
    insights = predictions.get('predictive_insights', {})
    
    if insights:
        st.markdown(f"""
        <div class="insight-card">
            <h4>🎯 {insights.get('main', 'No specific insight')}</h4>
            <p><strong>Recommendation:</strong> {insights.get('recommendation', 'No specific recommendation')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Trend insights
        if 'home_trend' in insights:
            st.info(f"📈 {insights['home_trend']}")
        if 'away_trend' in insights:
            st.info(f"📈 {insights['away_trend']}")
    
    # Probabilities
    st.markdown("### 📊 Predictive Probabilities")
    
    outcomes = predictions['probabilities']['match_outcomes']
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Home Win", f"{outcomes['home_win']}%")
    with col2:
        st.metric("Draw", f"{outcomes['draw']}%")
    with col3:
        st.metric("Away Win", f"{outcomes['away_win']}%")
    
    # Goals markets
    goals = predictions['probabilities']['over_under']
    btts = predictions['probabilities']['both_teams_score']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Over 2.5 Goals", f"{goals['over_25']}%")
        st.metric("Under 2.5 Goals", f"{goals['under_25']}%")
    with col2:
        st.metric("BTTS Yes", f"{btts['yes']}%")
        st.metric("BTTS No", f"{btts['no']}%")
    
    # Value bets
    display_value_signals(predictions)
    
    # Risk assessment
    risk = predictions['risk_assessment']
    risk_class = f"predictive-{risk['risk_level'].lower()}"
    
    st.markdown(f"""
    <div class="predictive-card {risk_class}">
        <h3>⚖️ Predictive Risk Assessment</h3>
        <p><strong>Level:</strong> {risk['risk_level']}</p>
        <p><strong>Analysis:</strong> {risk['explanation']}</p>
        <p><strong>Recommendation:</strong> {risk['recommendation']}</p>
    </div>
    """, unsafe_allow_html=True)

def display_value_signals(predictions):
    """Display value betting signals"""
    
    signals = predictions.get('betting_signals', [])
    
    if not signals:
        st.warning("🎯 No value bets detected")
        st.info("""
        This means:
        - Market is efficiently priced
        - No clear predictive edge found
        - Recommendation: Avoid betting or wait for better opportunities
        """)
        return
    
    st.markdown("### 💎 Value Betting Signals")
    
    for signal in signals:
        value_class = f"value-{signal['value_rating'].lower()}"
        
        st.markdown(f"""
        <div class="value-signal {value_class}">
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <div>
                    <h4>{signal['market']}</h4>
                    <p>Model: {signal['model_prob']}% | Market: {signal['book_prob']}%</p>
                    <p><strong>Edge: +{signal['edge']}%</strong> | Confidence: {signal['confidence']}</p>
                </div>
                <div style="text-align: right;">
                    <h3>Stake: {signal['recommended_stake']*100:.1f}%</h3>
                    <p><strong>{signal['value_rating']} VALUE</strong></p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application"""
    
    # Initialize session state
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    
    # Show predictions if available
    if st.session_state.predictions:
        display_predictive_analysis(st.session_state.predictions)
        
        st.markdown("---")
        if st.button("🔄 New Analysis", use_container_width=True):
            st.session_state.predictions = None
            st.rerun()
        
        return
    
    # Input form
    match_data, mc_iterations = create_predictive_input_form()
    
    if match_data:
        with st.spinner("🧠 Running predictive analysis..."):
            try:
                # Use predictive engine
                predictor = PredictiveFootballEngine(match_data)
                predictions = predictor.generate_predictive_analysis(mc_iterations)
                
                # Store in session state
                st.session_state.predictions = predictions
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Analysis error: {str(e)}")
                st.info("💡 Check your input data and try again")

if __name__ == "__main__":
    main()
