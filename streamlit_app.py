import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

# Import the TRUE predictive engine
try:
    from prediction_engine import TruePredictiveFootballEngine, PredictiveValueEngine, TeamMetrics
except ImportError:
    st.error("❌ Could not import predictive engine. Make sure prediction_engine.py is in the same directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="TRUE Predictive Football Engine ⚽",
    page_icon="🔮", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS styling
st.markdown("""
<style>
    .predictive-header {
        font-size: 2.8rem !important;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .predictive-subheader {
        font-size: 1.4rem !important;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    .predictive-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }
    .power-high { border-left-color: #4CAF50 !important; background: linear-gradient(135deg, #f8fff8 0%, #f0fff0 100%); }
    .power-medium { border-left-color: #FF9800 !important; background: linear-gradient(135deg, #fffaf2 0%, #fff5e6 100%); }
    .power-low { border-left-color: #f44336 !important; background: linear-gradient(135deg, #fff5f5 0%, #ffe6e6 100%); }
    
    .value-signal {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        border-left: 4px solid;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        transition: transform 0.2s ease;
    }
    .value-signal:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.12);
    }
    .value-exceptional { border-left-color: #4CAF50; background: linear-gradient(135deg, #f0fff0 0%, #e8f5e8 100%); }
    .value-high { border-left-color: #8BC34A; background: linear-gradient(135deg, #f9fff9 0%, #f1f8e9 100%); }
    .value-good { border-left-color: #FFC107; background: linear-gradient(135deg, #fffdf6 0%, #fff8e1 100%); }
    .value-moderate { border-left-color: #FF9800; background: linear-gradient(135deg, #fffaf2 0%, #fff3e0 100%); }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .predictive-badge {
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        color: white;
        display: inline-block;
        margin: 0.3rem;
    }
    .badge-high { background: linear-gradient(135deg, #4CAF50, #45a049); }
    .badge-medium { background: linear-gradient(135deg, #FF9800, #F57C00); }
    .badge-low { background: linear-gradient(135deg, #f44336, #d32f2f); }
    
    .section-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #333;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #f0f2f6;
    }
    
    .probability-gauge {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .performance-metric {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

def create_predictive_input_form():
    """Create input form for TRUE predictive features"""
    
    st.markdown('<p class="predictive-header">🔮 TRUE Predictive Football Engine</p>', unsafe_allow_html=True)
    st.markdown('<p class="predictive-subheader">Sustainable Performance Metrics • Market Mispricing Detection • Bayesian Learning</p>', unsafe_allow_html=True)
    
    # System Overview
    with st.expander("🎯 PREDICTIVE ENGINE ARCHITECTURE", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🔄 BAYESIAN LEARNING**
            - Prior beliefs updated with evidence
            - Separates skill from luck
            - Adapts to team quality changes
            """)
            
        with col2:
            st.markdown("""
            **📊 SUSTAINABLE METRICS** 
            - Non-penalty xG (removes luck)
            - xG against per shot (defensive quality)
            - Pressing intensity
            - Possession quality
            """)
            
        with col3:
            st.markdown("""
            **🎪 MARKET PSYCHOLOGY**
            - Recency bias detection
            - Big team bias correction  
            - Public sentiment analysis
            - Sharp money tracking
            """)
    
    tabs = st.tabs(["🎯 PREDICTIVE DATA", "💰 MARKET ODDS", "📈 PERFORMANCE TRACKING"])
    
    with tabs[0]:
        st.markdown("### 🎯 TRUE Predictive Features Input")
        st.info("These features actually PREDICT future performance - not describe past results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏠 Home Team Predictive Metrics")
            home_team = st.text_input("Team Name", value="Bologna", key="home_team")
            home_xg = st.slider("Non-Penalty xG (Last 5 Games)", 0.5, 3.0, 1.8, 0.1, key="home_xg")
            home_xg_against = st.slider("xG Against (Last 5 Games)", 0.5, 3.0, 0.9, 0.1, key="home_xg_against")
            home_possession = st.slider("Possession Quality Score", 0.3, 0.8, 0.58, 0.01, key="home_possession")
            home_press = st.slider("Press Intensity", 10.0, 25.0, 18.5, 0.1, key="home_press")
            
        with col2:
            st.subheader("✈️ Away Team Predictive Metrics")  
            away_team = st.text_input("Team Name", value="Torino", key="away_team")
            away_xg = st.slider("Non-Penalty xG (Last 5 Games)", 0.5, 3.0, 1.2, 0.1, key="away_xg")
            away_xg_against = st.slider("xG Against (Last 5 Games)", 0.5, 3.0, 1.1, 0.1, key="away_xg_against")
            away_possession = st.slider("Possession Quality Score", 0.3, 0.8, 0.52, 0.01, key="away_possession")
            away_press = st.slider("Press Intensity", 10.0, 25.0, 16.2, 0.1, key="away_press")
        
        # Recent form for bias detection
        st.markdown("---")
        st.subheader("📈 Recent Context (For Bias Detection)")
        form_col1, form_col2 = st.columns(2)
        with form_col1:
            home_recent_wins = st.slider(f"{home_team} Recent Wins", 0, 5, 2, key="home_recent_wins")
        with form_col2:
            away_recent_wins = st.slider(f"{away_team} Recent Wins", 0, 5, 1, key="away_recent_wins")
    
    with tabs[1]:
        st.markdown("### 💰 Market Odds Input")
        st.warning("Odds are used for VALUE detection only - never influence predictions")
        
        odds_col1, odds_col2, odds_col3 = st.columns(3)
        
        with odds_col1:
            st.write("**1X2 Market**")
            home_odds = st.number_input("Home Win", min_value=1.01, value=2.30, step=0.01, key="home_odds")
            draw_odds = st.number_input("Draw", min_value=1.01, value=3.10, step=0.01, key="draw_odds")
            away_odds = st.number_input("Away Win", min_value=1.01, value=3.40, step=0.01, key="away_odds")
        
        with odds_col2:
            st.write("**Goal Markets**")
            over_15_odds = st.number_input("Over 1.5", min_value=1.01, value=1.40, step=0.01, key="over_15_odds")
            over_25_odds = st.number_input("Over 2.5", min_value=1.01, value=2.10, step=0.01, key="over_25_odds")
            over_35_odds = st.number_input("Over 3.5", min_value=1.01, value=3.50, step=0.01, key="over_35_odds")
        
        with odds_col3:
            st.write("**Both Teams to Score**")
            btts_yes_odds = st.number_input("BTTS Yes", min_value=1.01, value=1.95, step=0.01, key="btts_yes_odds")
            btts_no_odds = st.number_input("BTTS No", min_value=1.01, value=1.85, step=0.01, key="btts_no_odds")
    
    with tabs[2]:
        st.markdown("### 📈 Performance Tracking & Learning")
        
        if 'performance_data' not in st.session_state:
            st.session_state.performance_data = []
            
        if st.session_state.performance_data:
            df = pd.DataFrame(st.session_state.performance_data)
            st.write(f"**Total Predictions Tracked:** {len(df)}")
            
            # Calculate accuracy
            accuracy = df['correct'].mean() * 100
            st.metric("Overall Accuracy", f"{accuracy:.1f}%")
            
            # Show recent performance
            st.write("**Recent Predictions:**")
            st.dataframe(df.tail(5))
        else:
            st.info("No performance data yet. Generate predictions to start tracking!")
    
    # Generate analysis button
    submitted = st.button("🔮 GENERATE PREDICTIVE ANALYSIS", type="primary", use_container_width=True)
    
    if submitted:
        if not home_team or not away_team:
            st.error("❌ Please enter both team names")
            return None
            
        # Compile predictive data
        predictive_data = {
            'home_team': home_team,
            'away_team': away_team,
            'home_xg': home_xg,
            'away_xg': away_xg,
            'home_xg_against': home_xg_against,
            'away_xg_against': away_xg_against,
            'home_possession_quality': home_possession,
            'away_possession_quality': away_possession,
            'home_press_intensity': home_press,
            'away_press_intensity': away_press,
            'home_recent_wins': home_recent_wins,
            'away_recent_wins': away_recent_wins,
            'market_odds': {
                '1x2 Home': home_odds,
                '1x2 Draw': draw_odds,
                '1x2 Away': away_odds,
                'Over 1.5 Goals': over_15_odds,
                'Over 2.5 Goals': over_25_odds,
                'Over 3.5 Goals': over_35_odds,
                'BTTS Yes': btts_yes_odds,
                'BTTS No': btts_no_odds,
            }
        }
        
        return predictive_data
    
    return None

def display_predictive_power_gauge(score: float, title: str):
    """Display predictive power as a gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 0.5},
        gauge = {
            'axis': {'range': [0, 1]},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, 0.4], 'color': "lightgray"},
                {'range': [0.4, 0.7], 'color': "yellow"},
                {'range': [0.7, 1], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.9
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def display_predictive_analysis(results):
    """Display TRUE predictive analysis"""
    
    st.markdown('<p class="predictive-header">🔮 PREDICTIVE ANALYSIS RESULTS</p>', unsafe_allow_html=True)
    
    # Main metrics overview
    predictive_results = results['predictive_results']
    predictive_metrics = predictive_results['predictive_metrics']
    power_rating = results['predictive_power_rating']
    
    # Power rating styling
    power_class = {
        'HIGH_PREDICTIVE_POWER': 'power-high',
        'MEDIUM_PREDICTIVE_POWER': 'power-medium', 
        'LOW_PREDICTIVE_POWER': 'power-low'
    }.get(power_rating, 'power-low')
    
    st.markdown(f'''
    <div class="predictive-card {power_class}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2 style="margin: 0; color: #333;">{results['match']}</h2>
                <p style="margin: 0.5rem 0 0 0; color: #666;">TRUE Predictive Analysis • {results['analysis_timestamp'][:16]}</p>
            </div>
            <div style="text-align: right;">
                <span class="predictive-badge badge-{power_rating.split('_')[0].lower()}">
                    {power_rating.replace('_', ' ')}
                </span>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏠 Expected Goals", f"{predictive_results['expected_goals']['home']:.2f}")
    with col2:
        st.metric("✈️ Expected Goals", f"{predictive_results['expected_goals']['away']:.2f}")
    with col3:
        st.metric("📊 Sustainability", f"{predictive_metrics['sustainability_score']:.3f}")
    with col4:
        st.metric("🎪 Market Mispricing", f"{predictive_metrics['market_mispricing']}%")
    
    # Use tabs for different analysis aspects
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 PREDICTIONS", "💰 VALUE SIGNALS", "📊 METRICS", "🏗️ SYSTEM HEALTH"])
    
    with tab1:
        display_predictive_probabilities(predictive_results)
    
    with tab2:
        display_value_signals(results)
    
    with tab3:
        display_predictive_metrics(predictive_results)
    
    with tab4:
        display_system_health(results)

def display_predictive_probabilities(predictive_results):
    """Display probabilities with predictive context"""
    st.markdown('<div class="section-title">🎯 Predictive Probabilities</div>', unsafe_allow_html=True)
    
    probabilities = predictive_results['probabilities']
    
    # Outcome probabilities
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = display_predictive_power_gauge(probabilities['home_win']/100, "Home Win")
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        fig = display_predictive_power_gauge(probabilities['draw']/100, "Draw")
        st.plotly_chart(fig, use_container_width=True)
        
    with col3:
        fig = display_predictive_power_gauge(probabilities['away_win']/100, "Away Win")
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional markets
    st.markdown("#### 📊 Additional Markets")
    add_col1, add_col2 = st.columns(2)
    
    with add_col1:
        fig = display_predictive_power_gauge(probabilities['over_25']/100, "Over 2.5 Goals")
        st.plotly_chart(fig, use_container_width=True)
        
    with add_col2:
        fig = display_predictive_power_gauge(probabilities['btts_yes']/100, "BTTS Yes")
        st.plotly_chart(fig, use_container_width=True)

def display_value_signals(results):
    """Display value signals with predictive power"""
    st.markdown('<div class="section-title">💰 Predictive Value Signals</div>', unsafe_allow_html=True)
    
    value_signals = results.get('value_signals', [])
    
    if not value_signals:
        st.warning("""
        ⚠️ No strong value signals detected. This means:
        - Market is efficiently priced
        - No clear predictive edges found
        - Recommendation: Avoid betting or wait for better opportunities
        """)
        return
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Signals", len(value_signals))
    with col2:
        avg_edge = np.mean([s['edge'] for s in value_signals])
        st.metric("Average Edge", f"{avg_edge:.1f}%")
    with col3:
        high_value = len([s for s in value_signals if s['value_rating'] in ['EXCEPTIONAL', 'HIGH']])
        st.metric("High Value Signals", high_value)
    
    # Display each signal
    for signal in value_signals:
        value_class = f"value-{signal['value_rating'].lower()}"
        
        st.markdown(f'''
        <div class="value-signal {value_class}">
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <div style="flex: 2;">
                    <h4 style="margin: 0 0 0.5rem 0;">{signal['market']}</h4>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.9rem;">
                        <div>Model Probability: <strong>{signal['model_prob']}%</strong></div>
                        <div>Market Implied: <strong>{signal['book_prob']}%</strong></div>
                        <div>Predictive Power: <strong>{signal['predictive_power']}</strong></div>
                        <div>Confidence: <strong>{signal['confidence']}</strong></div>
                    </div>
                </div>
                <div style="flex: 1; text-align: right;">
                    <div style="font-size: 1.5rem; font-weight: bold; color: #4CAF50; margin-bottom: 0.5rem;">
                        +{signal['edge']}% Edge
                    </div>
                    <div style="font-size: 0.9rem;">
                        Stake: <strong>{signal['recommended_stake']*100:.1f}%</strong>
                    </div>
                    <span class="predictive-badge badge-{signal['confidence'].lower()}">
                        {signal['value_rating']} VALUE
                    </span>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

def display_predictive_metrics(predictive_results):
    """Display detailed predictive metrics"""
    st.markdown('<div class="section-title">📊 Predictive Metrics Analysis</div>', unsafe_allow_html=True)
    
    metrics = predictive_results['predictive_metrics']
    expected_goals = predictive_results['expected_goals']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Sustainability Analysis")
        st.metric("Sustainability Score", f"{metrics['sustainability_score']:.3f}")
        st.metric("Predictive Confidence", f"{metrics['predictive_confidence']:.3f}")
        
        # Sustainability interpretation
        if metrics['sustainability_score'] > 0.7:
            st.success("✅ High sustainability - predictions are reliable")
        elif metrics['sustainability_score'] > 0.5:
            st.warning("⚠️ Medium sustainability - use caution")
        else:
            st.error("❌ Low sustainability - high uncertainty")
    
    with col2:
        st.markdown("#### 🎪 Market Psychology")
        st.metric("Estimated Mispricing", f"{metrics['market_mispricing']}%")
        
        # Mispricing interpretation
        if abs(metrics['market_mispricing']) > 8:
            st.success("🎯 Significant market inefficiency detected")
        elif abs(metrics['market_mispricing']) > 4:
            st.info("💡 Moderate market inefficiency")
        else:
            st.warning("⚡ Market appears efficient")
    
    # Expected goals comparison
    st.markdown("#### ⚽ Expected Goals Analysis")
    eg_col1, eg_col2, eg_col3 = st.columns(3)
    
    with eg_col1:
        st.metric("Home xG", f"{expected_goals['home']:.2f}")
    with eg_col2:
        st.metric("Away xG", f"{expected_goals['away']:.2f}")
    with eg_col3:
        total_xg = expected_goals['home'] + expected_goals['away']
        st.metric("Total xG", f"{total_xg:.2f}")

def display_system_health(results):
    """Display system health and learning status"""
    st.markdown('<div class="section-title">🏗️ Predictive System Health</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔄 Bayesian Learning Status")
        st.info("""
        **System Features:**
        - ✅ Prior beliefs initialized
        - ✅ Evidence updating active  
        - ✅ Market bias detection
        - ✅ Sustainability scoring
        - ✅ Predictive power weighting
        """)
        
    with col2:
        st.markdown("#### 📈 Performance Tracking")
        if 'performance_data' in st.session_state and st.session_state.performance_data:
            df = pd.DataFrame(st.session_state.performance_data)
            accuracy = df['correct'].mean() * 100
            
            st.metric("Tracked Predictions", len(df))
            st.metric("Current Accuracy", f"{accuracy:.1f}%")
            
            if accuracy > 55:
                st.success("🎯 Beating market expectations")
            elif accuracy > 50:
                st.warning("📊 Market efficiency level")
            else:
                st.error("🔧 Needs calibration")
        else:
            st.info("No performance data tracked yet")
    
    # System recommendations
    st.markdown("#### 💡 System Recommendations")
    
    power_rating = results['predictive_power_rating']
    
    if power_rating == 'HIGH_PREDICTIVE_POWER':
        st.success("""
        **✅ STRONG PREDICTIVE CONFIDENCE**
        - Model has high certainty in predictions
        - Sustainable metrics detected
        - Market may have pricing inefficiencies
        - Consider value signals seriously
        """)
    elif power_rating == 'MEDIUM_PREDICTIVE_POWER':
        st.warning("""
        **⚠️ MODERATE PREDICTIVE CONFIDENCE** 
        - Some predictive signals present
        - Mixed sustainability metrics
        - Use smaller stakes if betting
        - Combine with other analysis
        """)
    else:
        st.error("""
        **❌ LOW PREDICTIVE CONFIDENCE**
        - High uncertainty in predictions
        - Low sustainability scores
        - Market appears efficient
        - Recommendation: Avoid betting
        """)

def main():
    """Main application function"""
    
    # Initialize session state
    if 'predictive_results' not in st.session_state:
        st.session_state.predictive_results = None
        
    if 'performance_data' not in st.session_state:
        st.session_state.performance_data = []
    
    # Show results if available
    if st.session_state.predictive_results:
        display_predictive_analysis(st.session_state.predictive_results)
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Analyze New Match", use_container_width=True):
                st.session_state.predictive_results = None
                st.rerun()
        
        with col2:
            if st.button("📊 Track This Prediction", use_container_width=True):
                # Add to performance tracking
                st.session_state.performance_data.append({
                    'timestamp': datetime.now(),
                    'match': st.session_state.predictive_results['match'],
                    'prediction': st.session_state.predictive_results['predictive_results'],
                    'correct': None  # Would be updated with actual result
                })
                st.success("Prediction added to tracking!")
        
        with col3:
            if st.button("🔍 Raw Data", use_container_width=True):
                st.json(st.session_state.predictive_results)
        
        return
    
    # Input form
    predictive_data = create_predictive_input_form()
    
    if predictive_data:
        with st.spinner("🔮 Running TRUE predictive analysis..."):
            try:
                # Use the TRUE predictive engine
                engine = TruePredictiveFootballEngine(predictive_data)
                results = engine.generate_predictive_analysis()
                
                # Store in session state
                st.session_state.predictive_results = results
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Predictive analysis error: {str(e)}")
                st.info("💡 Check that all predictive features are provided")

if __name__ == "__main__":
    main()
