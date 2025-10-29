import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import re
from prediction_engine import QuantumTimingArbitrageEngine

# Page configuration
st.set_page_config(
    page_title="QUANTUM TIMING ARBITRAGE ENGINE ⚽",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Quantum CSS for revolutionary look
st.markdown("""
<style>
    .quantum-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 20%, #f093fb 40%, #f5576c 60%, #4facfe 80%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .quantum-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    .timing-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    .edge-metric {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    .risk-metric {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    .correlation-metric {
        background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .segment-input {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .bet-opportunity {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 2px solid #667eea;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .cascade-prediction {
        background: linear-gradient(135deg, rgba(253, 121, 168, 0.1) 0%, rgba(232, 67, 147, 0.1) 100%);
        border: 2px solid #fd79a8;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .timing-overlap {
        background: linear-gradient(135deg, rgba(0, 184, 148, 0.1) 0%, rgba(0, 160, 133, 0.1) 100%);
        border: 2px solid #00b894;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .quantum-glow {
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'quantum_data' not in st.session_state:
    st.session_state.quantum_data = {}
if 'quantum_predictions' not in st.session_state:
    st.session_state.quantum_predictions = None
if 'show_quantum_analysis' not in st.session_state:
    st.session_state.show_quantum_analysis = False

def create_quantum_input_form():
    """Create revolutionary quantum timing input form"""
    
    st.markdown("""
    <div class='quantum-card quantum-glow'>
        <h1 class='quantum-header'>QUANTUM TIMING ARBITRAGE ENGINE</h1>
        <p style='text-align: center; font-size: 1.2rem; color: #666;'>
        🚀 Micro-Timing Analysis • Corner-Goal Correlations • Market Mispricing Detection
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("quantum_timing_form"):
        # Basic match info
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.text_input(
                "🏠 QUANTUM ANALYSIS SUBJECT A", 
                value=st.session_state.quantum_data.get('home_team', 'Dinamo Tbilisi'),
                placeholder="e.g., Team Alpha",
                key="quantum_home_team"
            )
        with col2:
            away_team = st.text_input(
                "✈️ QUANTUM ANALYSIS SUBJECT B", 
                value=st.session_state.quantum_data.get('away_team', 'Kolkheti Poti'),
                placeholder="e.g., Team Beta",
                key="quantum_away_team"
            )
        
        st.markdown("---")
        
        # Goal Timing DNA Analysis
        st.markdown("""
        <div class='quantum-card'>
            <h2>🎯 GOAL TIMING DNA PROFILING</h2>
            <p><em>Enter goals by 15-minute segments (last 6 matches)</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Home Team Goal Timing
        st.markdown(f"### 🏠 {home_team} - Goal Timing Distribution")
        home_goal_cols = st.columns(6)
        home_goals_data = st.session_state.quantum_data.get('home_timing_data', {})
        
        with home_goal_cols[0]:
            goals_0_15 = st.number_input("0-15min Goals", min_value=0, value=home_goals_data.get('goals_0_15', 2), key="home_0_15")
        with home_goal_cols[1]:
            goals_16_30 = st.number_input("16-30min Goals", min_value=0, value=home_goals_data.get('goals_16_30', 3), key="home_16_30")
        with home_goal_cols[2]:
            goals_31_45 = st.number_input("31-45min Goals", min_value=0, value=home_goals_data.get('goals_31_45', 5), key="home_31_45")
        with home_goal_cols[3]:
            goals_46_60 = st.number_input("46-60min Goals", min_value=0, value=home_goals_data.get('goals_46_60', 2), key="home_46_60")
        with home_goal_cols[4]:
            goals_61_75 = st.number_input("61-75min Goals", min_value=0, value=home_goals_data.get('goals_61_75', 4), key="home_61_75")
        with home_goal_cols[5]:
            goals_76_90 = st.number_input("76-90min Goals", min_value=0, value=home_goals_data.get('goals_76_90', 4), key="home_76_90")
        
        home_matches = st.number_input(f"Matches Analyzed for {home_team}", min_value=1, value=home_goals_data.get('matches_analyzed', 6), key="home_matches")
        
        # Away Team Goal Timing
        st.markdown(f"### ✈️ {away_team} - Goal Timing Distribution")
        away_goal_cols = st.columns(6)
        away_goals_data = st.session_state.quantum_data.get('away_timing_data', {})
        
        with away_goal_cols[0]:
            away_goals_0_15 = st.number_input("0-15min Goals", min_value=0, value=away_goals_data.get('goals_0_15', 1), key="away_0_15")
        with away_goal_cols[1]:
            away_goals_16_30 = st.number_input("16-30min Goals", min_value=0, value=away_goals_data.get('goals_16_30', 2), key="away_16_30")
        with away_goal_cols[2]:
            away_goals_31_45 = st.number_input("31-45min Goals", min_value=0, value=away_goals_data.get('goals_31_45', 3), key="away_31_45")
        with away_goal_cols[3]:
            away_goals_46_60 = st.number_input("46-60min Goals", min_value=0, value=away_goals_data.get('goals_46_60', 1), key="away_46_60")
        with away_goal_cols[4]:
            away_goals_61_75 = st.number_input("61-75min Goals", min_value=0, value=away_goals_data.get('goals_61_75', 2), key="away_61_75")
        with away_goal_cols[5]:
            away_goals_76_90 = st.number_input("76-90min Goals", min_value=0, value=away_goals_data.get('goals_76_90', 1), key="away_76_90")
        
        away_matches = st.number_input(f"Matches Analyzed for {away_team}", min_value=1, value=away_goals_data.get('matches_analyzed', 6), key="away_matches")
        
        st.markdown("---")
        
        # Corner Timing DNA Analysis
        st.markdown("""
        <div class='quantum-card'>
            <h2>🔄 CORNER TIMING DNA PROFILING</h2>
            <p><em>Enter corners by 15-minute segments (last 6 matches)</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Home Team Corner Timing
        st.markdown(f"### 🏠 {home_team} - Corner Timing Distribution")
        home_corner_cols = st.columns(6)
        home_corners_data = st.session_state.quantum_data.get('home_corner_data', {})
        
        with home_corner_cols[0]:
            corners_0_15 = st.number_input("0-15min Corners", min_value=0, value=home_corners_data.get('corners_0_15', 8), key="home_c_0_15")
        with home_corner_cols[1]:
            corners_16_30 = st.number_input("16-30min Corners", min_value=0, value=home_corners_data.get('corners_16_30', 12), key="home_c_16_30")
        with home_corner_cols[2]:
            corners_31_45 = st.number_input("31-45min Corners", min_value=0, value=home_corners_data.get('corners_31_45', 15), key="home_c_31_45")
        with home_corner_cols[3]:
            corners_46_60 = st.number_input("46-60min Corners", min_value=0, value=home_corners_data.get('corners_46_60', 10), key="home_c_46_60")
        with home_corner_cols[4]:
            corners_61_75 = st.number_input("61-75min Corners", min_value=0, value=home_corners_data.get('corners_61_75', 14), key="home_c_61_75")
        with home_corner_cols[5]:
            corners_76_90 = st.number_input("76-90min Corners", min_value=0, value=home_corners_data.get('corners_76_90', 16), key="home_c_76_90")
        
        home_corner_matches = st.number_input(f"Corner Matches Analyzed for {home_team}", min_value=1, value=home_corners_data.get('matches_analyzed', 6), key="home_c_matches")
        
        # Away Team Corner Timing
        st.markdown(f"### ✈️ {away_team} - Corner Timing Distribution")
        away_corner_cols = st.columns(6)
        away_corners_data = st.session_state.quantum_data.get('away_corner_data', {})
        
        with away_corner_cols[0]:
            away_corners_0_15 = st.number_input("0-15min Corners", min_value=0, value=away_corners_data.get('corners_0_15', 6), key="away_c_0_15")
        with away_corner_cols[1]:
            away_corners_16_30 = st.number_input("16-30min Corners", min_value=0, value=away_corners_data.get('corners_16_30', 8), key="away_c_16_30")
        with away_corner_cols[2]:
            away_corners_31_45 = st.number_input("31-45min Corners", min_value=0, value=away_corners_data.get('corners_31_45', 10), key="away_c_31_45")
        with away_corner_cols[3]:
            away_corners_46_60 = st.number_input("46-60min Corners", min_value=0, value=away_corners_data.get('corners_46_60', 7), key="away_c_46_60")
        with away_corner_cols[4]:
            away_corners_61_75 = st.number_input("61-75min Corners", min_value=0, value=away_corners_data.get('corners_61_75', 9), key="away_c_61_75")
        with away_corner_cols[5]:
            away_corners_76_90 = st.number_input("76-90min Corners", min_value=0, value=away_corners_data.get('corners_76_90', 11), key="away_c_76_90")
        
        away_corner_matches = st.number_input(f"Corner Matches Analyzed for {away_team}", min_value=1, value=away_corners_data.get('matches_analyzed', 6), key="away_c_matches")
        
        # Context Factors
        st.markdown("---")
        st.markdown("""
        <div class='quantum-card'>
            <h2>🌍 QUANTUM CONTEXT FACTORS</h2>
        </div>
        """, unsafe_allow_html=True)
        
        context_col1, context_col2, context_col3 = st.columns(3)
        with context_col1:
            match_context = st.selectbox(
                "Match Context",
                ["Normal League", "Derby/Rivalry", "Relegation Battle", "Title Decider", "Cup Match", "Must-Win Situation"],
                help="Psychological and motivational factors"
            )
        with context_col2:
            venue_impact = st.selectbox(
                "Venue Dynamics",
                ["Normal", "Home Fortress", "Away Stronghold", "Neutral Ground", "Intimidating Atmosphere"],
                help="Crowd and venue impact on timing patterns"
            )
        with context_col3:
            recent_form = st.selectbox(
                "Recent Momentum",
                ["Both In Form", "Home In Form", "Away In Form", "Both Struggling", "Mixed Form"],
                help="Current team momentum affecting timing"
            )
        
        # Quantum Analysis Submission
        submitted = st.form_submit_button(
            "🚀 ACTIVATE QUANTUM TIMING ARBITRAGE ENGINE", 
            type="primary", 
            use_container_width=True
        )
        
        if submitted:
            if not home_team or not away_team:
                st.error("❌ Quantum analysis requires both teams for temporal alignment")
                return None
            
            # Store quantum data
            st.session_state.quantum_data = {
                'home_team': home_team,
                'away_team': away_team,
                'home_timing_data': {
                    'matches_analyzed': home_matches,
                    'goals_0_15': goals_0_15, 'goals_16_30': goals_16_30, 'goals_31_45': goals_31_45,
                    'goals_46_60': goals_46_60, 'goals_61_75': goals_61_75, 'goals_76_90': goals_76_90
                },
                'away_timing_data': {
                    'matches_analyzed': away_matches,
                    'goals_0_15': away_goals_0_15, 'goals_16_30': away_goals_16_30, 'goals_31_45': away_goals_31_45,
                    'goals_46_60': away_goals_46_60, 'goals_61_75': away_goals_61_75, 'goals_76_90': away_goals_76_90
                },
                'home_corner_data': {
                    'matches_analyzed': home_corner_matches,
                    'corners_0_15': corners_0_15, 'corners_16_30': corners_16_30, 'corners_31_45': corners_31_45,
                    'corners_46_60': corners_46_60, 'corners_61_75': corners_61_75, 'corners_76_90': corners_76_90
                },
                'away_corner_data': {
                    'matches_analyzed': away_corner_matches,
                    'corners_0_15': away_corners_0_15, 'corners_16_30': away_corners_16_30, 'corners_31_45': away_corners_31_45,
                    'corners_46_60': away_corners_46_60, 'corners_61_75': away_corners_61_75, 'corners_76_90': away_corners_76_90
                },
                'context_factors': {
                    'match_context': match_context,
                    'venue_impact': venue_impact,
                    'recent_form': recent_form
                }
            }
            
            return st.session_state.quantum_data
    
    return None

def display_quantum_analysis(quantum_data, predictions):
    """Display revolutionary quantum timing analysis"""
    
    st.markdown("""
    <div class='quantum-card quantum-glow'>
        <h1 class='quantum-header'>QUANTUM TIMING ARBITRAGE REPORT</h1>
        <p style='text-align: center; font-size: 1.2rem; color: #666;'>
        🚀 Micro-Timing Edges • Corner-Goal Correlations • Market Mispricing • Cascade Predictions
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Executive Summary
    display_executive_summary(predictions)
    
    # Team Timing DNA Analysis
    display_team_timing_dna(predictions)
    
    # Timing Overlap Detection
    display_timing_overlaps(predictions)
    
    # Corner-Goal Correlations
    display_corner_goal_correlations(predictions)
    
    # Market Mispricing Opportunities
    display_market_mispricing(predictions)
    
    # Cascade Predictions
    display_cascade_predictions(predictions)
    
    # Betting Opportunities
    display_betting_opportunities(predictions)
    
    # Quantum Confidence Metrics
    display_quantum_metrics(predictions)

def display_executive_summary(predictions):
    """Display quantum executive summary"""
    
    st.markdown("""
    <div class='quantum-card'>
        <h2>🎯 QUANTUM EXECUTIVE SUMMARY</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        confidence = predictions.get('confidence_score', 50)
        st.markdown(f"""
        <div class='timing-metric'>
            <h3>Quantum Confidence</h3>
            <h2>{confidence}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        opportunities = len(predictions.get('betting_opportunities', []))
        st.markdown(f"""
        <div class='edge-metric'>
            <h3>Arbitrage Opportunities</h3>
            <h2>{opportunities}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        timing_overlaps = len(predictions.get('timing_overlaps', []))
        st.markdown(f"""
        <div class='correlation-metric'>
            <h3>Timing Overlaps</h3>
            <h2>{timing_overlaps}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        correlations = len(predictions.get('corner_goal_correlations', []))
        st.markdown(f"""
        <div class='risk-metric'>
            <h3>Corner-Goal Signals</h3>
            <h2>{correlations}</h2>
        </div>
        """, unsafe_allow_html=True)

def display_team_timing_dna(predictions):
    """Display team timing DNA analysis"""
    
    st.markdown("""
    <div class='quantum-card'>
        <h2>🔬 TEAM TIMING DNA ANALYSIS</h2>
    </div>
    """, unsafe_allow_html=True)
    
    team_dna = predictions.get('team_timing_dna', {})
    home_dna = team_dna.get('home', {})
    away_dna = team_dna.get('away', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### 🏠 {home_dna.get('team_name', 'Home Team')}")
        display_timing_segments(home_dna, "Goals")
        
    with col2:
        st.markdown(f"### ✈️ {away_dna.get('team_name', 'Away Team')}")
        display_timing_segments(away_dna, "Goals")
    
    # Corner DNA Analysis
    corner_dna = predictions.get('corner_timing_dna', {})
    home_corner = corner_dna.get('home', {})
    away_corner = corner_dna.get('away', {})
    
    st.markdown("---")
    st.markdown("### 🔄 CORNER TIMING DNA")
    
    corner_col1, corner_col2 = st.columns(2)
    
    with corner_col1:
        st.markdown(f"#### 🏠 {home_corner.get('team_name', 'Home Team')} - Corner Pressure")
        display_timing_segments(home_corner, "Corners")
        
    with corner_col2:
        st.markdown(f"#### ✈️ {away_corner.get('team_name', 'Away Team')} - Corner Pressure")
        display_timing_segments(away_corner, "Corners")

def display_timing_segments(team_dna, metric_type):
    """Display timing segments for a team"""
    
    segments = team_dna.get('segments_avg_goals', {}) if metric_type == "Goals" else team_dna.get('segments_avg_corners', {})
    probabilities = team_dna.get('segments_probabilities', {})
    
    for segment, avg_value in segments.items():
        prob = probabilities.get(segment, 0)
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.write(f"**{segment}**")
        with col2:
            progress = prob  # Use probability for progress bar
            st.progress(progress)
        with col3:
            st.write(f"{avg_value:.2f} {metric_type.lower()}")
        
        # Highlight peak period
        if segment == team_dna.get('peak_scoring_period' if metric_type == "Goals" else 'peak_corner_period'):
            st.success(f"⭐ **PEAK {metric_type.upper()} PERIOD** - {prob:.1%} probability")

def display_timing_overlaps(predictions):
    """Display timing overlap analysis"""
    
    st.markdown("""
    <div class='quantum-card'>
        <h2>🎯 TIMING OVERLAP DETECTION</h2>
        <p><em>Where attack strength meets defense weakness</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    overlaps = predictions.get('timing_overlaps', [])
    
    if overlaps:
        for i, overlap in enumerate(overlaps[:3]):  # Top 3 overlaps
            with st.expander(f"🎯 Timing Overlap #{i+1}: {overlap['segment']} (Score: {overlap['overlap_score']:.3f})", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Overlap Score", f"{overlap['overlap_score']:.3f}")
                    st.metric("Segment", overlap['segment'])
                
                with col2:
                    st.metric("Attack Strength", f"{overlap['attack_strength']:.1%}")
                    st.metric("Defense Weakness", f"{overlap['defense_weakness']:.1%}")
                
                with col3:
                    st.metric("Expected Goals", f"{overlap['expected_goals']:.2f}")
                    st.metric("Type", overlap['type'].replace('_', ' ').title())
                
                st.info(f"**Quantum Insight**: Strong timing overlap detected in {overlap['segment']} - optimal betting window")
    else:
        st.warning("🔍 No significant timing overlaps detected")

def display_corner_goal_correlations(predictions):
    """Display corner-goal correlation analysis"""
    
    st.markdown("""
    <div class='quantum-card'>
        <h2>🔄 CORNER-GOAL CORRELATION ANALYSIS</h2>
        <p><em>Predicting goals from corner timing patterns</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    correlations = predictions.get('corner_goal_correlations', [])
    
    if correlations:
        for i, correlation in enumerate(correlations[:3]):  # Top 3 correlations
            with st.expander(f"🔄 Correlation #{i+1}: {correlation['team']} in {correlation['segment']}", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Correlation Strength", f"{correlation['correlation_strength']:.3f}")
                    st.metric("Team", correlation['team'])
                
                with col2:
                    st.metric("Corner Pressure", f"{correlation['corner_pressure']:.2f}")
                    st.metric("Goal Probability", f"{correlation['goal_probability']:.1%}")
                
                with col3:
                    st.metric("Segment", correlation['segment'])
                    st.metric("Signal Type", "Corner→Goal")
                
                st.success(f"**Quantum Signal**: {correlation['team']} shows strong corner→goal correlation in {correlation['segment']}")
    else:
        st.info("🔍 No strong corner-goal correlations detected")

def display_market_mispricing(predictions):
    """Display market mispricing opportunities"""
    
    st.markdown("""
    <div class='quantum-card'>
        <h2>💰 MARKET TIMING MISPRICING DETECTION</h2>
        <p><em>Where quantum analysis beats market expectations</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    mispricings = predictions.get('market_mispricing', [])
    
    if mispricings:
        for i, mispricing in enumerate(mispricings[:4]):  # Top 4 mispricings
            if mispricing['type'] == 'goal_timing_mispricing':
                with st.expander(f"💰 Mispricing #{i+1}: {mispricing['segment']} Goals", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Actual Probability", f"{mispricing['actual_probability']:.1%}")
                        st.metric("Market Probability", f"{mispricing['market_probability']:.1%}")
                    
                    with col2:
                        st.metric("Quantum Edge", f"+{mispricing['edge']:.1%}")
                        st.metric("Expected Odds", f"{mispricing['expected_odds']:.2f}")
                    
                    with col3:
                        st.metric("Market Odds", f"{mispricing['market_odds']:.2f}")
                        value_ratio = mispricing['expected_odds'] / mispricing['market_odds']
                        st.metric("Value Ratio", f"{value_ratio:.2f}")
                    
                    if value_ratio > 1:
                        st.success(f"🎯 **VALUE BET**: Market undervalues {mispricing['segment']} goals by {mispricing['edge']:.1%}")
            else:
                with st.expander(f"💰 Mispricing #{i+1}: {mispricing['edge_description']}", expanded=True):
                    st.metric("Correlation Strength", f"{mispricing['correlation_strength']:.3f}")
                    st.metric("Expected Value", f"{mispricing['expected_value']:.2f}")
                    st.info(f"**Market Blind Spot**: {mispricing['edge_description']}")
    else:
        st.warning("🔍 No significant market mispricing detected")

def display_cascade_predictions(predictions):
    """Display goal cascade predictions"""
    
    st.markdown("""
    <div class='quantum-card'>
        <h2>🌊 GOAL CASCADE PREDICTIONS</h2>
        <p><em>Momentum shifts and sequential goal patterns</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    cascades = predictions.get('cascade_predictions', [])
    
    if cascades:
        for i, cascade in enumerate(cascades):
            with st.expander(f"🌊 Cascade #{i+1}: {cascade['type'].replace('_', ' ').title()}", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("First Goal Window", cascade.get('first_goal_segment', 'N/A'))
                    st.metric("Cascade Type", cascade['type'].replace('_', ' ').title())
                
                with col2:
                    st.metric("Next Goal Window", cascade.get('next_goal_window', 'N/A'))
                    st.metric("Confidence", f"{cascade.get('cascade_confidence', 0)*100:.0f}%")
                
                with col3:
                    probability = cascade.get('probability', 0)
                    if probability > 0:
                        st.metric("Probability", f"{probability:.1%}")
                    else:
                        st.metric("Response Pattern", "Momentum-based")
                
                st.info(f"**Cascade Insight**: {cascade.get('response_pattern', cascade.get('description', 'Momentum shift detected'))}")
    else:
        st.info("🔍 No strong cascade patterns detected")

def display_betting_opportunities(predictions):
    """Display quantum betting opportunities"""
    
    st.markdown("""
    <div class='quantum-card'>
        <h2>💎 QUANTUM BETTING OPPORTUNITIES</h2>
        <p><em>Actionable edges based on timing arbitrage</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    opportunities = predictions.get('betting_opportunities', [])
    
    if opportunities:
        for i, opportunity in enumerate(opportunities):
            with st.expander(f"💎 Opportunity #{i+1}: {opportunity['market']}", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Bet Type", opportunity['bet_type'].replace('_', ' ').title())
                    st.metric("Quantum Edge", f"+{opportunity['edge']:.1%}")
                
                with col2:
                    st.metric("Expected Odds", f"{opportunity['expected_odds']:.2f}")
                    st.metric("Confidence", f"{opportunity['confidence']:.0f}%")
                
                with col3:
                    st.metric("Stake Recommendation", opportunity['stake_recommendation'])
                    value_score = opportunity['edge'] * opportunity['confidence'] / 100
                    st.metric("Value Score", f"{value_score:.3f}")
                
                st.success(f"**Rationale**: {opportunity['reasoning']}")
                
                # Risk management note
                if opportunity['stake_recommendation'] == 'High':
                    st.warning("⚠️ **Risk Note**: High stake recommendation - ensure proper bankroll management")
                elif opportunity['stake_recommendation'] == 'Medium':
                    st.info("💡 **Risk Note**: Medium stake - standard position sizing recommended")
                else:
                    st.info("🔍 **Risk Note**: Smaller stake - consider as part of portfolio")
    else:
        st.error("❌ No quantum betting opportunities detected with current thresholds")

def display_quantum_metrics(predictions):
    """Display quantum confidence metrics"""
    
    st.markdown("""
    <div class='quantum-card'>
        <h2>📊 QUANTUM CONFIDENCE METRICS</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        confidence = predictions.get('confidence_score', 50)
        if confidence >= 80:
            st.success(f"🎯 HIGH CONFIDENCE: {confidence}%")
        elif confidence >= 60:
            st.info(f"⚡ GOOD CONFIDENCE: {confidence}%")
        else:
            st.warning(f"🔍 MODERATE CONFIDENCE: {confidence}%")
    
    with col2:
        model_type = predictions.get('model_type', 'QUANTUM_TIMING_ARBITRAGE')
        st.metric("Model Version", model_type)
    
    with col3:
        timestamp = predictions.get('timestamp', '')
        if timestamp:
            st.metric("Analysis Time", datetime.fromisoformat(timestamp).strftime("%H:%M:%S"))
    
    with col4:
        opportunities = len(predictions.get('betting_opportunities', []))
        st.metric("Total Opportunities", opportunities)
    
    # Quantum Recommendations
    st.markdown("---")
    st.markdown("### 🚀 QUANTUM RECOMMENDATIONS")
    
    if opportunities > 0:
        st.success("""
        **🎯 RECOMMENDED STRATEGY:**
        - Focus on timing-based markets (goal in specific segments)
        - Use corner signals to confirm goal opportunities  
        - Implement proper stake sizing based on edge
        - Monitor live for cascade opportunities
        """)
    else:
        st.warning("""
        **🔍 CONSERVATIVE APPROACH:**
        - No strong quantum edges detected
        - Consider lower-stake positions
        - Wait for better timing alignment
        - Focus on traditional markets
        """)

def main():
    """Main quantum timing application"""
    
    # Show quantum analysis if available
    if st.session_state.quantum_predictions and st.session_state.show_quantum_analysis:
        display_quantum_analysis(st.session_state.quantum_data, st.session_state.quantum_predictions)
        
        # Action buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 New Quantum Analysis", use_container_width=True):
                st.session_state.quantum_data = {}
                st.session_state.quantum_predictions = None
                st.session_state.show_quantum_analysis = False
                st.rerun()
        with col2:
            if st.button("📊 Adjust Input Parameters", use_container_width=True):
                st.session_state.show_quantum_analysis = False
                st.rerun()
        
        return
    
    # Main quantum input form
    quantum_data = create_quantum_input_form()
    
    if quantum_data:
        # Generate quantum predictions
        with st.spinner("🚀 ACTIVATING QUANTUM TIMING ARBITRAGE ENGINE..."):
            try:
                engine = QuantumTimingArbitrageEngine(quantum_data)
                predictions = engine.generate_quantum_predictions()
                
                # Store in session state
                st.session_state.quantum_predictions = predictions
                st.session_state.show_quantum_analysis = True
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ QUANTUM ANALYSIS ERROR: {str(e)}")
                st.info("Please verify your temporal data inputs and try again.")

if __name__ == "__main__":
    main()
