import streamlit as st
import pandas as pd
import numpy as np
from prediction_engine import AdvancedPredictionEngine

# Page configuration
st.set_page_config(
    page_title="Advanced Football Predictor ⚽",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS styling
st.markdown("""
<style>
    .main-header { 
        font-size: 2.5rem !important; 
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.3rem !important;
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
    .risk-low { border-left-color: #4CAF50 !important; }
    .risk-medium { border-left-color: #FF9800 !important; }
    .risk-high { border-left-color: #f44336 !important; }
    
    .confidence-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .confidence-high { background: #4CAF50; color: white; }
    .confidence-medium { background: #FF9800; color: white; }
    .confidence-low { background: #f44336; color: white; }
    
    .probability-bar {
        height: 8px;
        background: #e0e0e0;
        border-radius: 4px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    .probability-fill {
        height: 100%;
        background: linear-gradient(90deg, #4CAF50, #45a049);
        border-radius: 4px;
    }
    
    .bet-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #f0f2f6;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
        margin: 10px 0;
    }
    .stat-item {
        background: #f8f9fa;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.9rem;
    }
    .stat-value {
        font-weight: bold;
        float: right;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'match_data' not in st.session_state:
    st.session_state.match_data = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

def create_advanced_input_form():
    """Create comprehensive input form with advanced options"""
    
    st.markdown('<p class="main-header">⚽ Advanced Football Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Professional match analysis with precise probability calculations</p>', unsafe_allow_html=True)
    
    with st.form("advanced_match_form"):
        # Basic team information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏠 Home Team")
            home_team = st.text_input("Team Name", value="Bologna", key="home_team")
            home_goals = st.number_input("Total Goals (Last 6 Games)", min_value=0, value=12, key="home_goals")
            home_conceded = st.number_input("Total Conceded (Last 6 Games)", min_value=0, value=8, key="home_conceded")
            home_goals_home = st.number_input("Home Goals (Last 3 Home Games)", min_value=0, value=7, key="home_goals_home")
            
        with col2:
            st.subheader("✈️ Away Team")
            away_team = st.text_input("Team Name", value="Torino", key="away_team")
            away_goals = st.number_input("Total Goals (Last 6 Games)", min_value=0, value=8, key="away_goals")
            away_conceded = st.number_input("Total Conceded (Last 6 Games)", min_value=0, value=10, key="away_conceded")
            away_goals_away = st.number_input("Away Goals (Last 3 Away Games)", min_value=0, value=4, key="away_goals_away")
        
        # Head-to-head section
        with st.expander("📊 Head-to-Head History", expanded=True):
            st.subheader("Head to Head")
            h2h_col1, h2h_col2, h2h_col3 = st.columns(3)
            with h2h_col1:
                h2h_matches = st.number_input("Total H2H Matches", min_value=0, value=4, key="h2h_matches")
                h2h_home_wins = st.number_input("Home Wins", min_value=0, value=3, key="h2h_home_wins")
            with h2h_col2:
                h2h_away_wins = st.number_input("Away Wins", min_value=0, value=0, key="h2h_away_wins")
                h2h_draws = st.number_input("Draws", min_value=0, value=1, key="h2h_draws")
            with h2h_col3:
                h2h_home_goals = st.number_input("Home Goals in H2H", min_value=0, value=8, key="h2h_home_goals")
                h2h_away_goals = st.number_input("Away Goals in H2H", min_value=0, value=2, key="h2h_away_goals")
        
        # League Table Context
        with st.expander("🏆 League Table Context"):
            st.subheader("Serie A Italy League Table")
            league_col1, league_col2 = st.columns(2)
            with league_col1:
                home_position = st.number_input(f"{home_team} Position", min_value=1, value=4, key="home_position")
                home_points = st.number_input(f"{home_team} Points", min_value=0, value=45, key="home_points")
            with league_col2:
                away_position = st.number_input(f"{away_team} Position", min_value=1, value=11, key="away_position")
                away_points = st.number_input(f"{away_team} Points", min_value=0, value=32, key="away_points")
        
        # Recent Form Sections
        with st.expander("📈 Recent Form Analysis"):
            st.subheader("Last 6 Matches Form")
            
            form_col1, form_col2 = st.columns(2)
            
            with form_col1:
                st.write(f"**{home_team} Last 6 Matches**")
                home_form = st.multiselect(
                    f"{home_team} Recent Results",
                    options=["Win (3 pts)", "Draw (1 pt)", "Loss (0 pts)"],
                    default=["Win (3 pts)", "Win (3 pts)", "Win (3 pts)", "Draw (1 pt)", "Loss (0 pts)", "Win (3 pts)"],
                    help="Select results from most recent to oldest",
                    key="home_form"
                )
                
            with form_col2:
                st.write(f"**{away_team} Last 6 Matches**")
                away_form = st.multiselect(
                    f"{away_team} Recent Results",
                    options=["Win (3 pts)", "Draw (1 pt)", "Loss (0 pts)"],
                    default=["Loss (0 pts)", "Win (3 pts)", "Draw (1 pt)", "Loss (0 pts)", "Win (3 pts)", "Draw (1 pt)"],
                    help="Select results from most recent to oldest",
                    key="away_form"
                )
        
        # Home/Away Specific Statistics - THE MAIN ENHANCEMENT
        with st.expander("🏠✈️ Home/Away Specific Statistics", expanded=True):
            st.subheader("Team-Specific Performance Metrics")
            
            home_away_col1, home_away_col2 = st.columns(2)
            
            with home_away_col1:
                st.write(f"**{home_team} Average Home Statistics**")
                home_goals_scored = st.number_input("Goals scored", min_value=0.0, value=2.3, key="home_goals_scored")
                home_goals_conceded = st.number_input("Goals conceded", min_value=0.0, value=0.3, key="home_goals_conceded")
                home_time_first_goal = st.number_input("Time first goal scored", min_value=1, value=52, key="home_time_first_goal")
                home_time_first_conceded = st.number_input("Time first goal conceded", min_value=1, value=63, key="home_time_first_conceded")
                home_yellow_cards = st.number_input("Yellow cards", min_value=0.0, value=1.7, key="home_yellow_cards")
                home_subs_used = st.number_input("Subs used", min_value=0, value=5, key="home_subs_used")
            
            with home_away_col2:
                st.write(f"**{away_team} Average Away Statistics**")
                away_goals_scored = st.number_input("Goals scored", min_value=0.0, value=1.3, key="away_goals_scored")
                away_goals_conceded = st.number_input("Goals conceded", min_value=0.0, value=2.5, key="away_goals_conceded")
                away_time_first_goal = st.number_input("Time first goal scored", min_value=1, value=42, key="away_time_first_goal")
                away_time_first_conceded = st.number_input("Time first goal conceded", min_value=1, value=26, key="away_time_first_conceded")
                away_yellow_cards = st.number_input("Yellow cards", min_value=0.0, value=2.3, key="away_yellow_cards")
                away_subs_used = st.number_input("Subs used", min_value=0, value=5, key="away_subs_used")
        
        # Advanced options
        with st.expander("⚙️ Advanced Match Parameters"):
            adv_col1, adv_col2, adv_col3 = st.columns(3)
            
            with adv_col1:
                league = st.selectbox("League", [
                    "premier_league", "la_liga", "serie_a", "bundesliga", 
                    "ligue_1", "default"
                ], index=2, key="league")
                
            with adv_col2:
                st.write("**Injuries & Suspensions**")
                home_injuries = st.slider("Home Key Absences", 0, 5, 0, key="home_injuries")
                away_injuries = st.slider("Away Key Absences", 0, 5, 1, key="away_injuries")
                
            with adv_col3:
                st.write("**Match Motivation**")
                home_motivation = st.select_slider(
                    "Home Team Motivation",
                    options=["Low", "Normal", "High", "Very High"],
                    value="High",
                    key="home_motivation"
                )
                away_motivation = st.select_slider(
                    "Away Team Motivation", 
                    options=["Low", "Normal", "High", "Very High"],
                    value="Normal",
                    key="away_motivation"
                )
        
        submitted = st.form_submit_button("🎯 GENERATE ADVANCED PREDICTION", type="primary", use_container_width=True)
        
        if submitted:
            if not home_team or not away_team:
                st.error("❌ Please enter both team names")
                return None
            
            # Convert form selections to points
            form_map = {"Win (3 pts)": 3, "Draw (1 pt)": 1, "Loss (0 pts)": 0}
            home_form_points = [form_map[result] for result in home_form]
            away_form_points = [form_map[result] for result in away_form]
            
            # Convert motivation to multipliers
            motivation_map = {"Low": 0.8, "Normal": 1.0, "High": 1.15, "Very High": 1.3}
            
            # Home/Away statistics
            home_avg_stats = {
                'goals_scored': home_goals_scored,
                'goals_conceded': home_goals_conceded,
                'time_first_goal_scored': home_time_first_goal,
                'time_first_goal_conceded': home_time_first_conceded,
                'yellow_cards': home_yellow_cards,
                'subs_used': home_subs_used
            }
            
            away_avg_stats = {
                'goals_scored': away_goals_scored,
                'goals_conceded': away_goals_conceded,
                'time_first_goal_scored': away_time_first_goal,
                'time_first_goal_conceded': away_time_first_conceded,
                'yellow_cards': away_yellow_cards,
                'subs_used': away_subs_used
            }
            
            match_data = {
                'home_team': home_team,
                'away_team': away_team,
                'league': league,
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
                'injuries': {'home': home_injuries, 'away': away_injuries},
                'motivation': {
                    'home': motivation_map[home_motivation],
                    'away': motivation_map[away_motivation]
                },
                'home_avg_stats': home_avg_stats,
                'away_avg_stats': away_avg_stats,
                'league_context': {
                    'home_position': home_position,
                    'away_position': away_position,
                    'home_points': home_points,
                    'away_points': away_points
                }
            }
            
            st.session_state.match_data = match_data
            return match_data
    
    return None

def display_advanced_predictions(predictions):
    """Display comprehensive predictions with detailed analysis"""
    
    st.markdown('<p class="main-header">🎯 Advanced Match Prediction</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align: center; font-size: 1.4rem; font-weight: 600;">{predictions["match"]}</p>', unsafe_allow_html=True)
    
    # Expected Goals and Risk Assessment
    col1, col2, col3 = st.columns(3)
    
    with col1:
        xg = predictions['expected_goals']
        st.metric("🏠 Expected Goals (Home)", f"{xg['home']}")
    with col2:
        st.metric("✈️ Expected Goals (Away)", f"{xg['away']}")
    with col3:
        risk = predictions['risk_assessment']
        risk_class = f"risk-{risk['risk_level'].lower()}"
        st.markdown(f'<div class="prediction-card {risk_class}"><h3>📊 Risk Assessment</h3><strong>{risk["risk_level"]} RISK</strong><br>{risk["explanation"]}<br>Certainty: {risk["certainty"]}</div>', unsafe_allow_html=True)
    
    # Match Outcomes
    st.markdown('<div class="section-title">📈 Match Outcome Probabilities</div>', unsafe_allow_html=True)
    
    outcomes = predictions['probabilities']['match_outcomes']
    col1, col2, col3 = st.columns(3)
    
    with col1:
        display_probability_bar("Home Win", outcomes['home_win'], "#4CAF50")
    with col2:
        display_probability_bar("Draw", outcomes['draw'], "#FF9800")
    with col3:
        display_probability_bar("Away Win", outcomes['away_win'], "#2196F3")
    
    # Goals Analysis
    st.markdown('<div class="section-title">⚽ Goals Analysis</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        timing = predictions['probabilities']['goal_timing']
        display_probability_card("First Half Goal", timing['first_half'])
    with col2:
        display_probability_card("Second Half Goal", timing['second_half'])
    with col3:
        btts = predictions['probabilities']['both_teams_score']
        display_probability_card("Both Teams Score", btts)
    with col4:
        over_25 = predictions['probabilities']['over_under']['over_2.5']
        display_probability_card("Over 2.5 Goals", over_25)
    
    # Exact Score Probabilities
    st.markdown('<div class="section-title">🎯 Most Likely Scores</div>', unsafe_allow_html=True)
    
    exact_scores = predictions['probabilities']['exact_scores']
    score_cols = st.columns(6)
    
    for idx, (score, prob) in enumerate(exact_scores.items()):
        with score_cols[idx]:
            st.metric(f"{score}", f"{prob}%")
    
    # Corner Predictions
    st.markdown('<div class="section-title">📊 Corner Predictions</div>', unsafe_allow_html=True)
    
    corners = predictions['corner_predictions']
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f'<div class="prediction-card"><h3>Total Corners</h3><span style="font-size: 1.8rem; font-weight: bold;">{corners["total"]}</span></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="prediction-card"><h3>🏠 Home Corners</h3><span style="font-size: 1.8rem; font-weight: bold;">{corners["home"]}</span></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="prediction-card"><h3>✈️ Away Corners</h3><span style="font-size: 1.8rem; font-weight: bold;">{corners["away"]}</span></div>', unsafe_allow_html=True)
    
    # Enhanced Timing Predictions
    st.markdown('<div class="section-title">⏰ Enhanced Timing Analysis</div>', unsafe_allow_html=True)
    
    timing = predictions['timing_predictions']
    st.markdown(f'''
    <div class="prediction-card">
        <h3>⏰ Key Timing Patterns</h3>
        • <strong>First Goal:</strong> {timing['first_goal']}<br>
        • <strong>Late Goals:</strong> {timing['late_goals']}<br>
        • <strong>Most Action:</strong> {timing['most_action']}<br>
        • <strong>Avg First Goal Time:</strong> {timing.get('avg_first_goal_time', 'N/A')}<br>
        • <strong>Avg First Conceded:</strong> {timing.get('avg_first_conceded_time', 'N/A')}
    </div>
    ''', unsafe_allow_html=True)
    
    # Betting Recommendations
    st.markdown('<div class="section-title">💰 Professional Betting Recommendations</div>', unsafe_allow_html=True)
    
    bets = predictions['betting_recommendations']
    
    # Top Bet
    st.markdown(f'<div class="bet-card">🚀 TOP CONFIDENCE BET<br><strong>{bets["top_bet"]}</strong></div>', unsafe_allow_html=True)
    
    # Other Recommendations
    for i, recommendation in enumerate(bets['recommendations']):
        if recommendation not in bets['top_bet']:
            confidence = bets['confidence_scores'][i] if i < len(bets['confidence_scores']) else 65
            confidence_class = "confidence-high" if confidence > 75 else "confidence-medium" if confidence > 60 else "confidence-low"
            
            st.markdown(f'''
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #4CAF50;">
                ✅ {recommendation}
                <span class="confidence-badge {confidence_class}" style="float: right;">{confidence}% conf</span>
            </div>
            ''', unsafe_allow_html=True)
    
    # Summary and Confidence
    st.markdown('<div class="section-title">📝 Professional Summary</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info(predictions['summary'])
    
    with col2:
        st.metric("Overall Confidence Score", f"{predictions['confidence_score']}%")

def display_probability_bar(label: str, probability: float, color: str):
    """Display a probability with a visual bar"""
    st.markdown(f'''
    <div style="margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span><strong>{label}</strong></span>
            <span><strong>{probability}%</strong></span>
        </div>
        <div class="probability-bar">
            <div class="probability-fill" style="width: {probability}%; background: {color};"></div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

def display_probability_card(label: str, probability: float):
    """Display a probability in a card format"""
    confidence_class = "confidence-high" if probability > 70 else "confidence-medium" if probability > 55 else "confidence-low"
    
    st.markdown(f'''
    <div class="prediction-card">
        <h4>{label}</h4>
        <span class="confidence-badge {confidence_class}">{probability}% probability</span>
    </div>
    ''', unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Show predictions if available
    if st.session_state.predictions:
        display_advanced_predictions(st.session_state.predictions)
        
        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🔄 Analyze Another Match", use_container_width=True):
                st.session_state.match_data = {}
                st.session_state.predictions = None
                st.rerun()
        
        with col2:
            if st.button("📊 Download Analysis", use_container_width=True):
                # In a real app, this would generate a PDF report
                st.success("Analysis download feature would be implemented here")
        
        return
    
    # Input form
    match_data = create_advanced_input_form()
    
    if match_data:
        with st.spinner("🔍 Performing advanced match analysis..."):
            try:
                engine = AdvancedPredictionEngine(match_data)
                predictions = engine.generate_advanced_predictions()
                
                st.session_state.predictions = predictions
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error generating predictions: {str(e)}")
                st.info("💡 Try adjusting the input parameters or check for invalid values")

if __name__ == "__main__":
    main()
