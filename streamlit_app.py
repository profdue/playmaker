# streamlit_app.py - ENHANCED CHAMPIONSHIP INTERFACE
import streamlit as st
st.cache_resource.clear()
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from typing import Dict, Any
from datetime import datetime

try:
    from prediction_engine import ApexEnhancedEngine, EnhancedTeamTierCalibrator
except ImportError as e:
    st.error(f"❌ Could not import prediction_engine: {str(e)}")
    st.info("💡 Make sure prediction_engine.py is in the same directory")
    st.stop()

# ... (Keep all the existing CSS and helper functions from previous version)

def display_enhanced_championship_banner():
    st.markdown("""
    <div class="money-grade-banner">
        🎯 ENHANCED CHAMPIONSHIP CALIBRATION • HOME ADVANTAGE BOOST • AWAY SCORING DETECTION • CONTEXT-AWARE CONFIDENCE
    </div>
    """, unsafe_allow_html=True)

def display_enhanced_championship_architecture():
    with st.expander("🏗️ ENHANCED CHAMPIONSHIP SYSTEM ARCHITECTURE", expanded=True):
        st.markdown("""
        ### 🎯 ENHANCED CHAMPIONSHIP PREDICTION ENGINE
        
        **Championship-Specific Enhancements:**
        - **Home Advantage Boost**: 25% home advantage multiplier (was 15%)
        - **Away Scoring Detection**: Automatic BTTS No trigger for poor away scorers
        - **Recent Form Weighting**: 35% weight on recent home/away form (was 25%)
        - **Enhanced Context Confidence**: Form-based confidence scoring
        - **Reduced BTTS Baseline**: 48% BTTS rate (was 51%)
        - **Lower Goal Expectations**: 2.5 avg goals (was 2.6)
        
        **Key Championship Fixes:**
        - Home advantage now properly overrides team reputation
        - Away scoring droughts correctly trigger defensive contexts
        - Recent form weighted more heavily than season-long tiers
        - Better detection of low-scoring Championship patterns
        """)

def create_enhanced_championship_form():
    st.markdown('<p class="professional-header">🎯 Enhanced Championship Football Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="professional-subheader">Championship-Specific Calibration with Enhanced Home Advantage Detection</p>', unsafe_allow_html=True)
    
    display_enhanced_championship_banner()
    display_enhanced_championship_architecture()
    
    # ... (Keep the existing form structure but with Championship as default)
    
    league_options = {
        'championship': 'Championship 🏴󠁧󠁢󠁥󠁮󠁧󠁿 *ENHANCED*',
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
        index=0,  # Championship as default
        key="enhanced_league_selection"
    )
    
    # ... (Rest of the form remains similar but with enhanced explanations)

def display_enhanced_championship_predictions(predictions):
    if not predictions:
        st.error("❌ No enhanced predictions available")
        return
        
    st.markdown('<p class="professional-header">🎯 Enhanced Championship Football Predictions</p>', unsafe_allow_html=True)
    
    # Add Championship-specific context display
    narrative = predictions.get('match_narrative', {})
    if predictions.get('league') == 'championship':
        st.markdown('<div class="professional-system-card"><h3>🟢 ENHANCED CHAMPIONSHIP CALIBRATION ACTIVE</h3>Home Advantage Boost + Away Scoring Detection + Recent Form Weighting</div>', unsafe_allow_html=True)
        
        # Display Championship-specific features
        col1, col2, col3 = st.columns(3)
        with col1:
            if narrative.get('home_advantage_amplified'):
                st.success("🏠 **Enhanced Home Advantage**")
                st.caption("Recent home form overriding team reputation")
        with col2:
            if narrative.get('away_scoring_issues'):
                st.warning("✈️ **Away Scoring Issues**")
                st.caption("Poor away form triggering defensive context")
        with col3:
            st.info("📊 **Recent Form Weighted**")
            st.caption("35% weight on recent performance")
    
    # ... (Rest of the display function remains similar but with enhanced context)

def main():
    if 'enhanced_predictions' not in st.session_state:
        st.session_state.enhanced_predictions = None
    
    if st.session_state.enhanced_predictions:
        display_enhanced_championship_predictions(st.session_state.enhanced_predictions)
        
        # Add Championship-specific analysis
        if st.session_state.enhanced_predictions.get('league') == 'championship':
            with st.expander("🔍 Enhanced Championship Analysis"):
                st.markdown("""
                **Championship-Specific Insights:**
                - 🏠 **Home Advantage**: 44% home win rate (higher than other leagues)
                - ✈️ **Away Struggles**: Away teams score 12% fewer goals
                - 🎯 **Form Over Reputation**: Recent performance > team reputation
                - ⚽ **Lower Scoring**: 2.5 avg goals per game (reduced from 2.6)
                - 🛡️ **Fewer BTTS**: 48% BTTS rate (reduced from 51%)
                
                **Betting Implications:**
                - Home teams with strong recent form offer enhanced value
                - Away teams with scoring droughts suggest BTTS No
                - Under 2.5 goals has higher probability in Championship
                """)
                
    else:
        match_data, mc_iterations = create_enhanced_championship_form()
        
        if match_data:
            with st.spinner("🔍 Running enhanced Championship analysis..."):
                try:
                    predictor = ApexEnhancedEngine(match_data)
                    predictions = predictor.generate_enhanced_predictions(mc_iterations)
                    
                    if predictions:
                        st.session_state.enhanced_predictions = predictions
                        
                        # Show Championship-specific success message
                        if predictions.get('league') == 'championship':
                            narrative = predictions.get('match_narrative', {})
                            st.success(f"""
                            ✅ **ENHANCED CHAMPIONSHIP ANALYSIS COMPLETE!**
                            
                            **Championship Features Activated:**
                            - 🏠 Home Advantage: {narrative.get('home_advantage_amplified', False)}
                            - ✈️ Away Scoring: {narrative.get('away_scoring_issues', False)}
                            - 🎯 Context Confidence: {predictions['enhanced_intelligence']['context_confidence']:.1f}%
                            - 📊 Recent Form Weight: 35%
                            """)
                        
                        st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Enhanced analysis error: {str(e)}")

if __name__ == "__main__":
    main()