# streamlit_app.py - WORKING VERSION
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime

# Import from the production prediction engine
try:
    from prediction_engine import ApexProductionEngine, EnhancedTeamTierCalibrator
except ImportError as e:
    st.error(f"❌ Import Error: {str(e)}")
    st.info("💡 Make sure prediction_engine.py is in the same directory")
    st.stop()

st.set_page_config(
    page_title="🎯 Football Predictor",
    page_icon="⚽", 
    layout="wide"
)

st.markdown("""
<style>
    .production-header { 
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1rem;
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
    .production-card { 
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def safe_get(dictionary, *keys, default=None):
    current = dictionary
    for key in keys:
        try:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        except:
            return default
    return current

def main():
    st.markdown('<p class="production-header">🎯 Football Predictor</p>', unsafe_allow_html=True)
    
    # Initialize
    tier_calibrator = EnhancedTeamTierCalibrator()
    
    # League selection
    selected_league = st.selectbox("Select League", ['premier_league', 'championship'], index=0)
    
    # Get teams
    available_teams = tier_calibrator.get_all_teams_for_league(selected_league)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏠 Home Team")
        home_team = st.selectbox("Team Name", available_teams, index=available_teams.index('Liverpool') if 'Liverpool' in available_teams else 0)
        home_goals = st.number_input("Goals (Last 6)", min_value=0, value=8, key="home_goals")
        home_conceded = st.number_input("Conceded (Last 6)", min_value=0, value=10, key="home_conceded")
    
    with col2:
        st.subheader("✈️ Away Team")  
        away_team = st.selectbox("Team Name", available_teams, index=available_teams.index('Aston Villa') if 'Aston Villa' in available_teams else 1)
        away_goals = st.number_input("Goals (Last 6)", min_value=0, value=9, key="away_goals")
        away_conceded = st.number_input("Conceded (Last 6)", min_value=0, value=4, key="away_conceded")
    
    # Market odds
    st.subheader("💰 Market Odds")
    col1, col2, col3 = st.columns(3)
    with col1:
        home_odds = st.number_input("Home Win", min_value=1.01, value=1.62, step=0.01)
        draw_odds = st.number_input("Draw", min_value=1.01, value=4.33, step=0.01)
        away_odds = st.number_input("Away Win", min_value=1.01, value=4.50, step=0.01)
    
    if st.button("🎯 GENERATE PREDICTIONS", type="primary", use_container_width=True):
        match_data = {
            'home_team': home_team,
            'away_team': away_team, 
            'league': selected_league,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'home_conceded': home_conceded,
            'away_conceded': away_conceded,
            'market_odds': {
                '1x2 Home': home_odds,
                '1x2 Draw': draw_odds, 
                '1x2 Away': away_odds,
                'Over 2.5 Goals': 1.53,
                'Under 2.5 Goals': 2.50,
                'BTTS Yes': 1.62,
                'BTTS No': 2.20
            },
            'bankroll': 1000,
            'kelly_fraction': 0.2
        }
        
        with st.spinner("🔄 Running refined analysis..."):
            try:
                engine = ApexProductionEngine(match_data)
                predictions = engine.generate_production_predictions()
                
                if predictions:
                    st.success("✅ Analysis Complete!")
                    
                    # Display results
                    st.markdown("---")
                    st.markdown(f"## 🎯 {predictions['match']}")
                    
                    # Expected goals
                    xg_data = predictions['expected_goals']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🏠 Home xG", f"{xg_data['home']:.2f}", f"±{xg_data['home_uncertainty']:.2f}")
                    with col2:
                        st.metric("✈️ Away xG", f"{xg_data['away']:.2f}", f"±{xg_data['away_uncertainty']:.2f}")  
                    with col3:
                        st.metric("⚽ Total xG", f"{xg_data['total']:.2f}")
                    
                    # Probabilities
                    st.subheader("📊 Probabilities")
                    outcomes = predictions['probabilities']['match_outcomes']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Home Win", f"{outcomes['home_win']:.1f}%")
                    with col2:
                        st.metric("Draw", f"{outcomes['draw']:.1f}%")
                    with col3:
                        st.metric("Away Win", f"{outcomes['away_win']:.1f}%")
                    
                    # Goals markets
                    st.subheader("⚽ Goals Markets")
                    btts = predictions['probabilities']['both_teams_score']
                    over_under = predictions['probabilities']['over_under']
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("BTTS Yes", f"{btts['yes']:.1f}%")
                        st.metric("Over 2.5", f"{over_under['over_25']:.1f}%")
                    with col2:
                        st.metric("BTTS No", f"{btts['no']:.1f}%") 
                        st.metric("Under 2.5", f"{over_under['under_25']:.1f}%")
                    
                    # Value opportunities
                    betting_ops = predictions.get('betting_recommendations', [])
                    if betting_ops:
                        st.subheader("💰 Value Opportunities")
                        for op in betting_ops:
                            st.info(f"**{op['market']}** - Edge: +{op['edge']*100:.1f}% - Stake: ${op['stake']:.2f}")
                    
                    st.info(predictions['production_summary'])
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
