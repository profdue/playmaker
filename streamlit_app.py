import streamlit as st
import pandas as pd
from prediction_engine import SimplePredictionEngine

# Page configuration
st.set_page_config(
    page_title="Simple Football Predictor ⚽",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, simple CSS
st.markdown("""
<style>
    .big-font { font-size: 2rem !important; font-weight: bold; }
    .medium-font { font-size: 1.5rem !important; font-weight: bold; }
    .prediction-card { 
        background: #f0f2f6; 
        padding: 1.5rem; 
        border-radius: 10px; 
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
    }
    .yes-badge { background: #4CAF50; color: white; padding: 0.5rem 1rem; border-radius: 20px; }
    .no-badge { background: #ff6b6b; color: white; padding: 0.5rem 1rem; border-radius: 20px; }
    .bet-card { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'match_data' not in st.session_state:
    st.session_state.match_data = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

def create_simple_input_form():
    """Create clean, simple input form"""
    
    st.markdown('<p class="big-font">⚽ Simple Football Predictor</p>', unsafe_allow_html=True)
    st.write("Enter basic match data for clear predictions")
    
    with st.form("simple_match_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            home_team = st.text_input("🏠 Home Team", value="Dinamo Tbilisi")
            home_goals = st.number_input("Home Team Goals (Last 6 Games)", min_value=0, value=8)
            home_conceded = st.number_input("Home Team Conceded (Last 6 Games)", min_value=0, value=13)
            
        with col2:
            away_team = st.text_input("✈️ Away Team", value="Kolkheti Poti") 
            away_goals = st.number_input("Away Team Goals (Last 6 Games)", min_value=0, value=5)
            away_conceded = st.number_input("Away Team Conceded (Last 6 Games)", min_value=0, value=12)
        
        submitted = st.form_submit_button("🎯 GET PREDICTIONS", type="primary", use_container_width=True)
        
        if submitted:
            if not home_team or not away_team:
                st.error("Please enter both team names")
                return None
            
            match_data = {
                'home_team': home_team,
                'away_team': away_team,
                'home_goals': home_goals,
                'away_goals': away_goals,
                'home_conceded': home_conceded,
                'away_conceded': away_conceded
            }
            
            st.session_state.match_data = match_data
            return match_data
    
    return None

def display_simple_predictions(predictions):
    """Display clean, simple predictions"""
    
    st.markdown('<p class="big-font">🎯 MATCH PREDICTION</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="medium-font">{predictions["match"]}</p>', unsafe_allow_html=True)
    
    # Goals Prediction
    st.markdown("### 📊 GOALS PREDICTION")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        first_half = predictions['goals_prediction']['first_half_goal']
        badge = "yes-badge" if first_half['answer'] == 'YES' else "no-badge"
        st.markdown(f'<div class="prediction-card"><h3>FIRST HALF GOAL</h3><span class="{badge}">{first_half["answer"]}</span><br>Probability: {first_half["probability"]}%<br>Confidence: {first_half["confidence"]}</div>', unsafe_allow_html=True)
    
    with col2:
        second_half = predictions['goals_prediction']['second_half_goal']
        badge = "yes-badge" if second_half['answer'] == 'YES' else "no-badge"
        st.markdown(f'<div class="prediction-card"><h3>SECOND HALF GOAL</h3><span class="{badge}">{second_half["answer"]}</span><br>Probability: {second_half["probability"]}%<br>Confidence: {second_half["confidence"]}</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="prediction-card"><h3>TOTAL GOALS</h3><span class="yes-badge">{predictions["goals_prediction"]["total_goals_range"]}</span><br>Both Teams Score: {predictions["goals_prediction"]["btts"]["answer"]}<br>Probability: {predictions["goals_prediction"]["btts"]["probability"]}%</div>', unsafe_allow_html=True)
    
    # Who Scores
    st.markdown("### ⚽ WHO SCORES")
    
    col1, col2 = st.columns(2)
    
    with col1:
        who = predictions['who_scores']
        st.markdown(f'<div class="prediction-card"><h3>🏠 {who["home_team"]}</h3>{who["home_likely"]} to score</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="prediction-card"><h3>✈️ {who["away_team"]}</h3>{who["away_likely"]} to score</div>', unsafe_allow_html=True)
    
    # Corners
    st.markdown("### 📊 CORNER PREDICTION")
    
    corners = predictions['corners_prediction']
    st.markdown(f'<div class="prediction-card"><h3>TOTAL CORNERS: {corners["total_corners_range"]}</h3>🏠 {corners["home_corners"]} corners<br>✈️ {corners["away_corners"]} corners</div>', unsafe_allow_html=True)
    
    # Timing
    st.markdown("### ⏰ KEY TIMING")
    
    timing = predictions['key_timing']
    st.markdown(f'<div class="prediction-card"><h3>⏰ MATCH TIMING</h3>First Goal: {timing["first_goal"]}<br>Late Goals: {timing["late_goals"]}<br>Most Action: {timing["most_action"]}</div>', unsafe_allow_html=True)
    
    # Best Bets
    st.markdown("### 💰 BEST BETS FOR BEGINNERS")
    
    for bet in predictions['best_bets']:
        st.markdown(f'<div class="bet-card">✅ {bet}</div>', unsafe_allow_html=True)
    
    # Top Bet
    st.markdown("### 🚀 TOP CONFIDENCE BET")
    st.markdown(f'<div class="bet-card" style="background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);">🎯 {predictions["top_confidence_bet"]}</div>', unsafe_allow_html=True)
    
    # Summary
    st.markdown("### 📝 SIMPLE SUMMARY")
    st.info(predictions['summary'])
    
    # Confidence
    st.metric("Overall Confidence Score", f"{predictions['confidence_score']}%")

def main():
    """Main application"""
    
    # Show predictions if available
    if st.session_state.predictions:
        display_simple_predictions(st.session_state.predictions)
        
        st.markdown("---")
        if st.button("🔄 ANALYZE ANOTHER MATCH", type="primary", use_container_width=True):
            st.session_state.match_data = {}
            st.session_state.predictions = None
            st.rerun()
        
        return
    
    # Input form
    match_data = create_simple_input_form()
    
    if match_data:
        with st.spinner("🔍 Analyzing match..."):
            try:
                engine = SimplePredictionEngine(match_data)
                predictions = engine.generate_simple_predictions()
                
                st.session_state.predictions = predictions
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")

if __name__ == "__main__":
    main()
