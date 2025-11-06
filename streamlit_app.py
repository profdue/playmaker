# streamlit_app.py - COMPLETE ENHANCED PROFESSIONAL PREDICTOR
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from typing import Dict, Any
from datetime import datetime

# Import from the enhanced prediction engine
try:
    from prediction_engine import ApexEnhancedEngine, EnhancedTeamTierCalibrator, ProfessionalLeagueCalibrator
except ImportError as e:
    st.error(f"❌ Could not import prediction_engine: {str(e)}")
    st.info("💡 Make sure prediction_engine.py is in the same directory")
    st.stop()

# Clear cache to ensure fresh imports
st.cache_resource.clear()

st.set_page_config(
    page_title="🎯 Enhanced Professional Football Predictor",
    page_icon="⚽", 
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .liga-portugal { background: #006600; }
    .brasileirao { background: #FFCC00; color: black; }
    .liga-mx { background: #006847; }
    .eredivisie { background: #FF6B00; }
    .championship { background: #8B0000; }
    
    .money-grade-banner {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    .professional-card { 
        background: white;
        padding: 1.8rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .risk-low { border-left-color: #4CAF50 !important; }
    .risk-medium { border-left-color: #FF9800 !important; }
    .risk-high { border-left-color: #f44336 !important; }
    
    .professional-system-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.8rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    
    .professional-value-card {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.8rem 0;
    }
    
    .professional-probability-bar {
        height: 10px;
        background: #e0e0e0;
        border-radius: 5px;
        margin: 0.8rem 0;
        overflow: hidden;
    }
    .professional-probability-fill {
        height: 100%;
        background: linear-gradient(90deg, #4CAF50, #45a049);
        border-radius: 5px;
    }
    
    .professional-bet-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    
    .value-exceptional { border-left-color: #4CAF50 !important; background: #f8fff8; }
    .value-high { border-left-color: #8BC34A !important; background: #f9fff9; }
    .value-good { border-left-color: #FFC107 !important; background: #fffdf6; }
    .value-moderate { border-left-color: #FF9800 !important; background: #fffaf2; }
    
    .professional-section-title {
        font-size: 1.6rem;
        font-weight: 600;
        color: #333;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.8rem;
        border-bottom: 3px solid #f0f2f6;
    }
    
    .professional-confidence-badge {
        padding: 0.4rem 1rem;
        border-radius: 18px;
        font-size: 0.9rem;
        font-weight: bold;
        color: white;
        display: inline-block;
        margin-top: 0.8rem;
    }
    .confidence-high { background: #4CAF50; }
    .confidence-medium { background: #FF9800; }
    .confidence-low { background: #f44336; }
    .confidence-speculative { background: #9E9E9E; }
    
    .professional-alignment-perfect {
        background: #f8fff8;
        border-left: 5px solid #4CAF50;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .professional-alignment-warning {
        background: #fffaf2;
        border-left: 5px solid #FF9800;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .professional-tier-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        color: white;
        display: inline-block;
        margin-left: 0.5rem;
    }
    .tier-elite { background: #e74c3c; }
    .tier-strong { background: #e67e22; }
    .tier-medium { background: #f1c40f; color: black; }
    .tier-weak { background: #95a5a6; }
    
    .professional-explanation-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        font-size: 0.95rem;
    }
    
    .professional-feature-badge {
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.3rem 0.7rem;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    
    .context-perfect {
        background: #e8f5e8;
        border-left: 4px solid #4CAF50;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #2E7D32;
    }
    .context-strong {
        background: #e3f2fd;
        border-left: 4px solid #2196F3;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #1565C0;
    }
    .context-contradictory {
        background: #ffebee;
        border-left: 4px solid #f44336;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #c62828;
    }
    
    .stability-bonus {
        background: #e8f5e8;
        color: #2E7D32;
        padding: 0.3rem 0.7rem;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-left: 0.5rem;
        border: 1px solid #4CAF50;
    }
    
    .confidence-explanation {
        background: #f0f8ff;
        border-left: 4px solid #2196F3;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        color: #1565C0;
    }
    
    .production-feature {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    
    .context-emoji {
        font-size: 1.5rem;
        margin-right: 0.5rem;
    }
    
    .betting-priority {
        background: #fff3e0;
        border-left: 4px solid #FF9800;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .context-confidence-high {
        background: #e8f5e8;
        color: #2E7D32;
        padding: 0.3rem 0.7rem;
        border-radius: 12px;
        font-size: 0.8rem;
        border: 1px solid #4CAF50;
    }
    .context-confidence-medium {
        background: #fff3e0;
        color: #EF6C00;
        padding: 0.3rem 0.7rem;
        border-radius: 12px;
        font-size: 0.8rem;
        border: 1px solid #FF9800;
    }
    
    .championship-feature {
        background: linear-gradient(135deg, #8B0000 0%, #B22222 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
        border: 2px solid #FFD700;
    }
    
    .enhanced-banner {
        background: linear-gradient(135deg, #8B0000 0%, #B22222 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.1rem;
        border: 3px solid #FFD700;
    }
    
    .pro-mode-active {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
        border: 3px solid #FFD700;
    }
    
    .league-intelligence-panel {
        background: #f8f9fa;
        border-left: 5px solid #667eea;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .calibration-dashboard {
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def safe_get(dictionary, *keys, default=None):
    """Safely get nested dictionary values"""
    if dictionary is None:
        return default
        
    current = dictionary
    for key in keys:
        try:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        except (TypeError, KeyError, AttributeError):
            return default
    return current

def get_league_display_name(league_id: str) -> str:
    """Get formatted league display name"""
    league_names = {
        'premier_league': 'Premier League 🏴󠁧󠁢󠁥󠁮󠁧󠁿',
        'la_liga': 'La Liga 🇪🇸', 
        'serie_a': 'Serie A 🇮🇹',
        'bundesliga': 'Bundesliga 🇩🇪',
        'ligue_1': 'Ligue 1 🇫🇷',
        'liga_portugal': 'Liga Portugal 🇵🇹',
        'brasileirao': 'Brasileirão 🇧🇷',
        'liga_mx': 'Liga MX 🇲🇽',
        'eredivisie': 'Eredivisie 🇳🇱',
        'championship': 'Championship 🏴󠁧󠁢󠁥󠁮󠁧󠁿 *ENHANCED*'
    }
    return league_names.get(league_id, league_id)

def get_league_badge(league_id: str) -> str:
    """Get CSS class for league badge"""
    league_classes = {
        'premier_league': 'premier-league',
        'la_liga': 'la-liga',
        'serie_a': 'serie-a', 
        'bundesliga': 'bundesliga',
        'ligue_1': 'ligue-1',
        'liga_portugal': 'liga-portugal',
        'brasileirao': 'brasileirao',
        'liga_mx': 'liga-mx',
        'eredivisie': 'eredivisie',
        'championship': 'championship'
    }
    return league_classes.get(league_id, 'premier-league')

def get_context_emoji(context: str) -> str:
    """Get emoji for match context"""
    context_emojis = {
        'home_dominance': '🏠',
        'away_counter': '✈️',
        'offensive_showdown': '🔥',
        'defensive_battle': '🛡️',
        'tactical_stalemate': '⚔️',
        'balanced': '⚖️'
    }
    return context_emojis.get(context, '⚖️')

def get_context_display_name(context: str) -> str:
    """Get formatted context display name"""
    context_names = {
        'home_dominance': 'Home Dominance',
        'away_counter': 'Away Counter', 
        'offensive_showdown': 'Offensive Showdown',
        'defensive_battle': 'Defensive Battle',
        'tactical_stalemate': 'Tactical Stalemate',
        'balanced': 'Balanced Match'
    }
    return context_names.get(context, context.replace('_', ' ').title())

def display_professional_league_calibration(predictions: dict, match_data: dict):
    """Display professional calibration dashboard"""
    
    st.markdown("---")
    st.markdown("### 🎯 PROFESSIONAL LEAGUE CALIBRATION")
    
    # Initialize calibrator
    calibrator = ProfessionalLeagueCalibrator()
    
    # Get calibration data
    raw_confidence = predictions.get('confidence_score', 0) / 100
    context = predictions.get('match_context', 'balanced')
    league = match_data.get('league', 'premier_league')
    
    calibration_result = calibrator.get_professional_confidence(
        raw_confidence, context, league
    )
    
    # Display calibration dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Raw Confidence", f"{calibration_result['raw_confidence']*100:.1f}%")
    
    with col2:
        adjustment_pct = calibration_result['league_adjusted'] - calibration_result['raw_confidence']
        st.metric("🎯 League Adjusted", 
                 f"{calibration_result['league_adjusted']*100:.1f}%",
                 f"{adjustment_pct*100:+.1f}%")
    
    with col3:
        st.metric("⚡ Professional Final", f"{calibration_result['final_professional']*100:.1f}%")
    
    with col4:
        st.metric("📉 Volatility Multiplier", f"{calibration_result['volatility_multiplier']:.1f}x")
    
    # League context insights
    st.markdown("#### 🧠 League Intelligence")
    preferred_contexts = calibration_result['preferred_contexts']
    context_display = [get_context_display_name(ctx) for ctx in preferred_contexts]
    
    if context in preferred_contexts or 'all' in preferred_contexts:
        st.success(f"✅ **Context Alignment**: {get_context_display_name(context)} matches {league.replace('_', ' ').title()} style")
    else:
        st.warning(f"⚠️ **Context Caution**: {get_context_display_name(context)} less reliable in {league.replace('_', ' ').title()}")
    
    st.info(f"**Preferred Contexts for {league.replace('_', ' ').title()}**: {', '.join(context_display)}")
    
    # Professional betting recommendations
    st.markdown("#### 💰 Professional Betting Advice")
    
    # Calculate professional stakes
    base_bankroll = match_data.get('bankroll', 1000)
    base_stake = base_bankroll * 0.02  # 2% base
    
    professional_stake = calibrator.calculate_professional_stake(
        calibration_result['final_professional'], base_stake, league, 'protected_single'
    )
    
    stake_percentage = (professional_stake / base_bankroll) * 100
    
    st.metric("🎯 Recommended Stake", f"${professional_stake:.2f}", f"{stake_percentage:.1f}% of bankroll")
    
    # Edge verification for key markets
    st.markdown("##### 📈 Edge Verification")
    
    market_odds = match_data.get('market_odds', {})
    probabilities = predictions.get('probabilities', {})
    
    edge_cols = st.columns(3)
    
    with edge_cols[0]:
        home_win_prob = probabilities.get('match_outcomes', {}).get('home_win', 0) / 100
        home_odds = market_odds.get('1x2 Home', 2.5)
        home_edge = home_win_prob - (1 / home_odds)
        should_bet_home = calibrator.should_place_bet(home_win_prob, home_odds, league)
        
        st.metric("Home Win Edge", f"{home_edge*100:+.1f}%", 
                 "✅ BET" if should_bet_home else "❌ PASS")
    
    with edge_cols[1]:
        btts_no_prob = probabilities.get('both_teams_score', {}).get('no', 0) / 100
        btts_no_odds = market_odds.get('BTTS No', 1.67)
        btts_edge = btts_no_prob - (1 / btts_no_odds)
        should_bet_btts = calibrator.should_place_bet(btts_no_prob, btts_no_odds, league)
        
        st.metric("BTTS No Edge", f"{btts_edge*100:+.1f}%",
                 "✅ BET" if should_bet_btts else "❌ PASS")
    
    with edge_cols[2]:
        under_prob = probabilities.get('over_under', {}).get('under_25', 0) / 100
        under_odds = market_odds.get('Under 2.5 Goals', 1.5)
        under_edge = under_prob - (1 / under_odds)
        should_bet_under = calibrator.should_place_bet(under_prob, under_odds, league)
        
        st.metric("Under 2.5 Edge", f"{under_edge*100:+.1f}%",
                 "✅ BET" if should_bet_under else "❌ PASS")
    
    # Visual calibration diagnostic
    st.markdown("##### 📊 Calibration Diagnostic")
    
    # Create confidence comparison chart
    confidence_data = {
        'Metric': ['Raw Model', 'League Adjusted', 'Professional Final'],
        'Confidence': [
            calibration_result['raw_confidence'] * 100,
            calibration_result['league_adjusted'] * 100, 
            calibration_result['final_professional'] * 100
        ]
    }
    
    fig = go.Figure(data=[
        go.Bar(name='Confidence Levels', x=confidence_data['Metric'], 
               y=confidence_data['Confidence'],
               marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ])
    
    fig.update_layout(
        title="Professional Confidence Calibration",
        yaxis_title="Confidence %",
        yaxis_range=[0, 100]
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_enhanced_predictions(predictions: dict, match_data: dict):
    """Display enhanced predictions with professional features"""
    if not predictions:
        st.error("❌ No enhanced predictions available")
        return
        
    st.markdown('<p class="professional-header">🎯 Enhanced Professional Football Predictions</p>', unsafe_allow_html=True)
    
    # Professional mode header
    if predictions.get('professional_calibration'):
        st.markdown('<div class="pro-mode-active"><h3>🟢 PROFESSIONAL LEAGUE MODE ACTIVE</h3>League-Aware Calibration • Dynamic Context Bonuses • Volatility-Adjusted Staking</div>', unsafe_allow_html=True)
    
    team_tiers = safe_get(predictions, 'team_tiers') or {}
    home_tier = team_tiers.get('home', 'MEDIUM')
    away_tier = team_tiers.get('away', 'MEDIUM')
    
    league = safe_get(predictions, 'league', default='premier_league')
    league_display_name = get_league_display_name(league)
    league_badge_class = get_league_badge(league)
    
    intelligence = safe_get(predictions, 'enhanced_intelligence') or {}
    stability_bonus = intelligence.get('form_stability_bonus', 0)
    context_confidence = intelligence.get('context_confidence', 0)
    
    betting_context = safe_get(predictions, 'betting_context') or {}
    primary_context = betting_context.get('primary_context', 'balanced')
    recommended_markets = betting_context.get('recommended_markets', [])
    expected_outcome = betting_context.get('expected_outcome', 'balanced')
    
    narrative = safe_get(predictions, 'match_narrative') or {}
    
    context_emoji = get_context_emoji(primary_context)
    context_display = get_context_display_name(primary_context)
    
    # Professional calibration data
    pro_calibration = safe_get(predictions, 'professional_calibration') or {}
    final_pro_confidence = pro_calibration.get('final_professional', 0) * 100
    
    st.markdown(f'''
    <div style="text-align: center; font-size: 1.5rem; font-weight: 600; margin-bottom: 1rem;">
        {predictions.get("match", "Unknown Match")} 
        <span class="professional-tier-badge tier-{home_tier.lower() if home_tier else 'medium'}">{home_tier or 'MEDIUM'}</span> vs 
        <span class="professional-tier-badge tier-{away_tier.lower() if away_tier else 'medium'}">{away_tier or 'MEDIUM'}</span>
        {f'<span class="stability-bonus">Stability: +{stability_bonus:.1f}</span>' if stability_bonus > 0 else ''}
    </div>
    <div style="text-align: center; margin-top: 0.5rem;">
        <span class="professional-badge {league_badge_class}">{league_display_name}</span>
        <span class="production-feature">{context_emoji} {context_display}</span>
        <span class="production-feature">🎯 Pro: {final_pro_confidence:.1f}%</span>
        {f'<span class="championship-feature">🏠 Home Advantage</span>' if narrative.get('home_advantage_amplified') else ''}
        {f'<span class="championship-feature">✈️ Away Scoring Issues</span>' if narrative.get('away_scoring_issues') else ''}
    </div>
    ''', unsafe_allow_html=True)
    
    if recommended_markets:
        st.markdown('<div class="betting-priority">', unsafe_allow_html=True)
        st.markdown(f"**🎯 Recommended Betting Markets for {context_display}:**")
        for market in recommended_markets[:3]:
            st.markdown(f"- **{market}**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    xg = safe_get(predictions, 'expected_goals') or {'home': 0, 'away': 0}
    confidence_score = safe_get(predictions, 'confidence_score') or 0
    data_quality = safe_get(predictions, 'data_quality_score') or 0
    football_iq = safe_get(predictions, 'enhanced_intelligence', 'football_iq_score') or 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏠 Expected Goals", f"{xg.get('home', 0):.2f}")
    with col2:
        st.metric("✈️ Expected Goals", f"{xg.get('away', 0):.2f}")
    with col3:
        st.metric("Enhanced Context", f"{context_emoji} {context_display}")
    with col4:
        st.metric("Pro Confidence", f"{final_pro_confidence:.1f}%")
    
    # League-specific features display
    if league == 'championship':
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
    
    st.markdown('<div class="professional-section-title">📈 Enhanced Outcome Probabilities</div>', unsafe_allow_html=True)
    
    outcomes = safe_get(predictions, 'probabilities', 'match_outcomes') or {'home_win': 0, 'draw': 0, 'away_win': 0}
    col1, col2, col3 = st.columns(3)
    
    with col1:
        home_win_prob = outcomes.get('home_win', 0)
        st.markdown(f'''
        <div style="margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.8rem;">
                <span><strong>Home Win</strong></span>
                <span><strong>{home_win_prob:.1f}%</strong></span>
            </div>
            <div class="professional-probability-bar">
                <div class="professional-probability-fill" style="width: {home_win_prob}%;"></div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        draw_prob = outcomes.get('draw', 0)
        st.markdown(f'''
        <div style="margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.8rem;">
                <span><strong>Draw</strong></span>
                <span><strong>{draw_prob:.1f}%</strong></span>
            </div>
            <div class="professional-probability-bar">
                <div class="professional-probability-fill" style="width: {draw_prob}%; background: #FF9800;"></div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    with col3:
        away_win_prob = outcomes.get('away_win', 0)
        st.markdown(f'''
        <div style="margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.8rem;">
                <span><strong>Away Win</strong></span>
                <span><strong>{away_win_prob:.1f}%</strong></span>
            </div>
            <div class="professional-probability-bar">
                <div class="professional-probability-fill" style="width: {away_win_prob}%; background: #2196F3;"></div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('<div class="professional-section-title">⚽ Enhanced Goals Analysis</div>', unsafe_allow_html=True)
    
    btts_yes = safe_get(predictions, 'probabilities', 'both_teams_score', 'yes') or 0
    btts_no = safe_get(predictions, 'probabilities', 'both_teams_score', 'no') or 0
    
    over_25 = safe_get(predictions, 'probabilities', 'over_under', 'over_25') or 0
    under_25 = safe_get(predictions, 'probabilities', 'over_under', 'under_25') or 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if primary_context == 'offensive_showdown':
            recommendation = "YES"
            primary_prob = btts_yes
            secondary_prob = btts_no
            card_class = "risk-low"
            emoji = "🔥"
            context_note = "Offensive Context"
        elif primary_context == 'defensive_battle':
            recommendation = "NO"
            primary_prob = btts_no
            secondary_prob = btts_yes
            card_class = "risk-high"
            emoji = "🛡️"
            context_note = "Defensive Context"
        else:
            if btts_no > btts_yes:
                recommendation = "NO"
                primary_prob = btts_no
                secondary_prob = btts_yes
                card_class = "risk-high"
                emoji = "❌"
                context_note = "Enhanced Analysis"
            else:
                recommendation = "YES"
                primary_prob = btts_yes
                secondary_prob = btts_no
                card_class = "risk-low"
                emoji = "✅"
                context_note = "Enhanced Analysis"
        
        confidence = "HIGH" if abs(primary_prob - 50) > 20 else "MEDIUM" if abs(primary_prob - 50) > 10 else "LOW"
        if primary_context in ['offensive_showdown', 'defensive_battle'] and context_confidence > 70:
            confidence = "HIGH"
        
        st.markdown(f'''
        <div class="professional-card {card_class}">
            <h4>{emoji} Both Teams Score {f"({context_note})" if context_note else ""}</h4>
            <div style="font-size: 2rem; font-weight: bold; color: #333; margin: 0.8rem 0;">
                {recommendation}: {primary_prob:.1f}%
            </div>
            <div style="font-size: 1rem; color: #666; margin: 0.5rem 0;">
                {('NO' if recommendation == 'YES' else 'YES')}: {secondary_prob:.1f}%
            </div>
            <span class="professional-confidence-badge confidence-{confidence.lower()}">
                {confidence} CONFIDENCE
            </span>
        </div>
        ''', unsafe_allow_html=True)
        
        explanations = safe_get(predictions, 'explanations', 'btts') or []
        for explanation in explanations[:2]:
            if explanation:
                st.markdown(f'<div class="professional-explanation-card">💡 {explanation}</div>', unsafe_allow_html=True)
    
    with col2:
        if primary_context == 'offensive_showdown':
            recommendation = "OVER"
            primary_prob = over_25
            secondary_prob = under_25
            card_class = "risk-low"
            emoji = "🔥"
            context_note = "Offensive Context"
        elif primary_context == 'defensive_battle':
            recommendation = "UNDER"
            primary_prob = under_25
            secondary_prob = over_25
            card_class = "risk-high"
            emoji = "🛡️"
            context_note = "Defensive Context"
        else:
            if under_25 > over_25:
                recommendation = "UNDER"
                primary_prob = under_25
                secondary_prob = over_25
                card_class = "risk-high"
                emoji = "❌"
                context_note = "Enhanced Analysis"
            else:
                recommendation = "OVER"
                primary_prob = over_25
                secondary_prob = under_25
                card_class = "risk-low"
                emoji = "✅"
                context_note = "Enhanced Analysis"
        
        confidence = "HIGH" if abs(primary_prob - 50) > 20 else "MEDIUM" if abs(primary_prob - 50) > 10 else "LOW"
        if primary_context in ['offensive_showdown', 'defensive_battle'] and context_confidence > 70:
            confidence = "HIGH"
        
        st.markdown(f'''
        <div class="professional-card {card_class}">
            <h4>{emoji} Over/Under 2.5 {f"({context_note})" if context_note else ""}</h4>
            <div style="font-size: 2rem; font-weight: bold; color: #333; margin: 0.8rem 0;">
                {recommendation}: {primary_prob:.1f}%
            </div>
            <div style="font-size: 1rem; color: #666; margin: 0.5rem 0;">
                {('OVER' if recommendation == 'UNDER' else 'UNDER')}: {secondary_prob:.1f}%
            </div>
            <span class="professional-confidence-badge confidence-{confidence.lower()}">
                {confidence} CONFIDENCE
            </span>
        </div>
        ''', unsafe_allow_html=True)
        
        explanations = safe_get(predictions, 'explanations', 'over_under') or []
        for explanation in explanations[:2]:
            if explanation:
                st.markdown(f'<div class="professional-explanation-card">💡 {explanation}</div>', unsafe_allow_html=True)
    
    with col3:
        xg = safe_get(predictions, 'expected_goals') or {'home': 0, 'away': 0}
        total_xg = xg.get('home', 0) + xg.get('away', 0)
        
        if total_xg > 3.2:
            xg_context = "High Scoring"
            xg_emoji = "🔥"
        elif total_xg < 2.2:
            xg_context = "Low Scoring" 
            xg_emoji = "🛡️"
        else:
            xg_context = "Average"
            xg_emoji = "⚖️"
        
        st.markdown(f'''
        <div class="professional-card">
            <h4>🎯 Expected Goals</h4>
            <div style="font-size: 1.3rem; font-weight: bold; color: #333; margin: 0.8rem 0;">
                Home: {xg.get('home', 0):.2f}
            </div>
            <div style="font-size: 1.3rem; font-weight: bold; color: #333; margin: 0.8rem 0;">
                Away: {xg.get('away', 0):.2f}
            </div>
            <div style="font-size: 1.1rem; color: #666; margin: 0.5rem 0;">
                Total: {total_xg:.2f} {xg_emoji}
            </div>
            <div style="font-size: 0.9rem; color: #888; margin: 0.5rem 0;">
                {xg_context} Expected
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        context = safe_get(predictions, 'match_context') or 'balanced'
        narrative = safe_get(predictions, 'match_narrative') or {}
        quality_gap = narrative.get('quality_gap', 'even')
        expected_outcome = narrative.get('expected_outcome', 'balanced')
        
        context_emoji = get_context_emoji(context)
        
        quality_emoji = {
            'extreme': '🔥',
            'significant': '⭐', 
            'even': '⚖️'
        }.get(quality_gap, '⚖️')
        
        st.markdown(f'''
        <div class="professional-card">
            <h4>{context_emoji} Enhanced Context</h4>
            <div style="font-size: 1.2rem; font-weight: bold; color: #333; margin: 0.8rem 0;">
                {get_context_display_name(context)}
            </div>
            <div style="font-size: 1rem; color: #666; margin: 0.5rem 0;">
                {quality_emoji} Quality Gap: {quality_gap.title()}
            </div>
            <div style="font-size: 1rem; color: #666; margin: 0.5rem 0;">
                🎯 Expected: {expected_outcome.replace('_', ' ').title()}
            </div>
            <div style="font-size: 1rem; color: #666; margin: 0.5rem 0;">
                📊 Confidence: {context_confidence}%
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('<div class="professional-section-title">🎯 Most Likely Scores</div>', unsafe_allow_html=True)
    
    exact_scores = safe_get(predictions, 'probabilities', 'exact_scores') or {}
    top_scores = dict(list(exact_scores.items())[:6])
    
    if top_scores:
        score_cols = st.columns(6)
        for idx, (score, prob) in enumerate(top_scores.items()):
            with score_cols[idx]:
                st.metric(f"{score}", f"{prob*100:.1f}%")
    else:
        st.info("No exact score data available")
    
    risk = safe_get(predictions, 'risk_assessment') or {'risk_level': 'UNKNOWN', 'explanation': 'No data'}
    risk_class = f"risk-{risk.get('risk_level', 'unknown').lower()}"
    
    intelligence = safe_get(predictions, 'enhanced_intelligence') or {}
    
    st.markdown(f'''
    <div class="professional-card {risk_class}">
        <h3>📊 Enhanced Risk Assessment</h3>
        <strong>Risk Level:</strong> {risk.get("risk_level", "UNKNOWN")}<br>
        <strong>Enhanced Explanation:</strong> {risk.get("explanation", "No data available")}<br>
        <strong>Enhanced Recommendation:</strong> {risk.get("recommendation", "N/A")}<br>
        <strong>Certainty:</strong> {risk.get("certainty", "N/A")}<br>
        <strong>Context Confidence:</strong> {context_confidence}%<br>
        <strong>Narrative Coherence:</strong> {intelligence.get('narrative_coherence', 'N/A')}%<br>
        <strong>Prediction Alignment:</strong> {intelligence.get('prediction_alignment', 'N/A')}<br>
        <strong>Form Stability Bonus:</strong> +{intelligence.get('form_stability_bonus', 0):.1f}<br>
        <strong>Calibration Status:</strong> {intelligence.get('calibration_status', 'N/A')}
    </div>
    ''', unsafe_allow_html=True)
    
    # Professional calibration display
    display_professional_league_calibration(predictions, match_data)
    
    # League-specific insights
    if league == 'championship':
        with st.expander("🔍 Enhanced Championship Insights"):
            st.markdown("""
            **Championship-Specific Analysis:**
            
            **🏠 Home Advantage Patterns:**
            - 44% home win rate (higher than other leagues)
            - Recent home form can override team reputation
            - Strong home performances boost confidence significantly
            
            **✈️ Away Team Challenges:**
            - Away teams score 12% fewer goals on average
            - Scoring droughts trigger defensive context detection
            - Poor away form penalizes strong teams more heavily
            
            **📊 Enhanced Weighting:**
            - 35% weight on recent form (increased from 25%)
            - Form can override tier-based quality assessments
            - Recent home/away performance prioritized over season-long data
            
            **⚽ Scoring Patterns:**
            - 2.5 average goals per game (reduced from 2.6)
            - 48% BTTS rate (reduced from 51%)
            - Fewer high-scoring games than initially modeled
            """)
    
    st.markdown('<div class="professional-section-title">📝 Enhanced Match Summary</div>', unsafe_allow_html=True)
    summary = safe_get(predictions, 'summary') or "No enhanced summary available."
    st.info(summary)

def create_enhanced_input_form():
    """Create enhanced input form with professional features"""
    st.markdown('<p class="professional-header">🎯 Enhanced Professional Football Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="professional-subheader">Professional League-Aware Calibration with Dynamic Context Bonuses</p>', unsafe_allow_html=True)
    
    # Professional mode toggle in sidebar
    professional_mode = st.sidebar.checkbox(
        "🎯 PROFESSIONAL LEAGUE MODE", 
        value=True,
        help="Apply league-specific calibration, volatility adjustments, and dynamic context bonuses"
    )
    
    # League intelligence in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌍 League Intelligence")
    
    calibrator = ProfessionalLeagueCalibrator()
    
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
        index=0,
        key="enhanced_league_selection"
    )
    
    # Display league intelligence
    league_analysis = calibrator.get_league_analysis(selected_league)
    
    st.sidebar.markdown(f'''
    <div class="league-intelligence-panel">
        <h4>🎯 {league_analysis['league_name']}</h4>
        <strong>Volatility:</strong> {league_analysis['volatility'].upper()}<br>
        <strong>Adjustment:</strong> {league_analysis['adjustment']*100:+.1f}%<br>
        <strong>Min Edge:</strong> {league_analysis['min_edge']*100:.1f}%<br>
        <strong>Stake Multiplier:</strong> {league_analysis['volatility_multiplier']:.1f}x<br>
        <strong>Preferred Contexts:</strong><br>
        {', '.join([get_context_display_name(ctx) for ctx in league_analysis['preferred_contexts']])}
    </div>
    ''', unsafe_allow_html=True)
    
    league_badge_class = get_league_badge(selected_league)
    league_display_name = get_league_display_name(selected_league)
    st.markdown(f'<span class="professional-badge {league_badge_class}">{league_display_name}</span>', unsafe_allow_html=True)
    
    if professional_mode:
        st.markdown('<div class="pro-mode-active">PROFESSIONAL LEAGUE MODE ACTIVE • VOLATILITY-ADJUSTED STAKING • DYNAMIC CONTEXT BONUSES</div>', unsafe_allow_html=True)
    
    if selected_league == 'championship':
        st.markdown('<span class="championship-feature">🎯 ENHANCED CHAMPIONSHIP MODE ACTIVE</span>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🏠 Enhanced Data", "💰 Market Data", "⚙️ Professional Settings"])

    with tab1:
        st.markdown("### 🎯 Enhanced Football Data")
        
        calibrator = EnhancedTeamTierCalibrator()
        league_teams = calibrator.get_league_teams(selected_league)
        
        if not league_teams:
            st.error(f"❌ No teams found for {league_display_name}")
            return None, None
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏠 Home Team")
            home_team = st.selectbox(
                "Team Name", 
                options=league_teams,
                index=league_teams.index('Charlton Athletic') if 'Charlton Athletic' in league_teams else min(5, len(league_teams) - 1),
                key="enhanced_home_team"
            )
            
            home_goals = st.number_input("Total Goals (Last 6 Games)", min_value=0, value=8, key="enhanced_home_goals")
            home_conceded = st.number_input("Total Conceded (Last 6 Games)", min_value=0, value=6, key="enhanced_home_conceded")
            home_goals_home = st.number_input("Home Goals (Last 3 Home Games)", min_value=0, value=6, key="enhanced_home_goals_home")
            
        with col2:
            st.subheader("✈️ Away Team")
            away_team = st.selectbox(
                "Team Name",
                options=league_teams,
                index=league_teams.index('West Brom') if 'West Brom' in league_teams else 0,
                key="enhanced_away_team"
            )
            
            away_goals = st.number_input("Total Goals (Last 6 Games)", min_value=0, value=4, key="enhanced_away_goals")
            away_conceded = st.number_input("Total Conceded (Last 6 Games)", min_value=0, value=7, key="enhanced_away_conceded")
            away_goals_away = st.number_input("Away Goals (Last 3 Away Games)", min_value=0, value=1, key="enhanced_away_goals_away")
        
        home_tier = calibrator.get_team_tier(home_team, selected_league)
        away_tier = calibrator.get_team_tier(away_team, selected_league)
        
        st.markdown(f"""
        **Enhanced Team Assessment:** 
        <span class="professional-tier-badge tier-{home_tier.lower() if home_tier else 'medium'}">{home_tier or 'MEDIUM'}</span> vs 
        <span class="professional-tier-badge tier-{away_tier.lower() if away_tier else 'medium'}">{away_tier or 'MEDIUM'}</span>
        """, unsafe_allow_html=True)
        
        # Show league-specific insights
        if selected_league == 'championship':
            if home_goals_home >= 5:
                st.success(f"🏠 **Strong Home Form**: {home_team} scoring {home_goals_home} goals in last 3 home games")
            if away_goals_away <= 1:
                st.warning(f"✈️ **Away Scoring Issues**: {away_team} only {away_goals_away} goal(s) in last 3 away games")
        
        with st.expander("📊 Enhanced Head-to-Head Analysis"):
            h2h_col1, h2h_col2, h2h_col3 = st.columns(3)
            with h2h_col1:
                h2h_matches = st.number_input("Total H2H Matches", min_value=0, value=4, key="enhanced_h2h_matches")
                h2h_home_wins = st.number_input("Home Wins", min_value=0, value=0, key="enhanced_h2h_home_wins")
            with h2h_col2:
                h2h_away_wins = st.number_input("Away Wins", min_value=0, value=1, key="enhanced_h2h_away_wins")
                h2h_draws = st.number_input("Draws", min_value=0, value=3, key="enhanced_h2h_draws")
            with h2h_col3:
                h2h_home_goals = st.number_input("Home Goals in H2H", min_value=0, value=7, key="enhanced_h2h_home_goals")
                h2h_away_goals = st.number_input("Away Goals in H2H", min_value=0, value=9, key="enhanced_h2h_away_goals")

        with st.expander("📈 Enhanced Form Analysis"):
            st.info("Enhanced form points: Win=3, Draw=1, Loss=0 (Recent form weighted 35% in Championship)")
            form_col1, form_col2 = st.columns(2)
            with form_col1:
                st.write(f"**{home_team} Last 6 Matches**")
                home_form = st.multiselect(
                    f"{home_team} Recent Results",
                    options=["Win (3 pts)", "Draw (1 pt)", "Loss (0 pts)"],
                    default=["Win (3 pts)", "Draw (1 pt)", "Loss (0 pts)", "Win (3 pts)", "Draw (1 pt)", "Draw (1 pt)"],
                    key="enhanced_home_form"
                )
            with form_col2:
                st.write(f"**{away_team} Last 6 Matches**")
                away_form = st.multiselect(
                    f"{away_team} Recent Results", 
                    options=["Win (3 pts)", "Draw (1 pt)", "Loss (0 pts)"],
                    default=["Draw (1 pt)", "Win (3 pts)", "Draw (1 pt)", "Loss (0 pts)", "Win (3 pts)", "Draw (1 pt)"],
                    key="enhanced_away_form"
                )

    with tab2:
        st.markdown("### 💰 Enhanced Market Data") 
        
        odds_col1, odds_col2, odds_col3 = st.columns(3)
        
        with odds_col1:
            st.write("**1X2 Market**")
            home_odds = st.number_input("Home Win Odds", min_value=1.01, value=2.50, step=0.01, key="enhanced_home_odds")
            draw_odds = st.number_input("Draw Odds", min_value=1.01, value=2.95, step=0.01, key="enhanced_draw_odds")
            away_odds = st.number_input("Away Win Odds", min_value=1.01, value=2.85, step=0.01, key="enhanced_away_odds")
        
        with odds_col2:
            st.write("**Over/Under Markets**")
            over_15_odds = st.number_input("Over 1.5 Goals", min_value=1.01, value=1.45, step=0.01, key="enhanced_over_15_odds")
            over_25_odds = st.number_input("Over 2.5 Goals", min_value=1.01, value=2.63, step=0.01, key="enhanced_over_25_odds")
            over_35_odds = st.number_input("Over 3.5 Goals", min_value=1.01, value=3.50, step=0.01, key="enhanced_over_35_odds")
        
        with odds_col3:
            st.write("**Both Teams to Score**")
            btts_yes_odds = st.number_input("BTTS Yes", min_value=1.01, value=2.10, step=0.01, key="enhanced_btts_yes_odds")
            btts_no_odds = st.number_input("BTTS No", min_value=1.01, value=1.67, step=0.01, key="enhanced_btts_no_odds")

    with tab3:
        st.markdown("### ⚙️ Professional Configuration")
        
        model_col1, model_col2 = st.columns(2)
        
        with model_col1:
            st.write("**Enhanced Team Context**")
            home_injuries = st.slider("Home Key Absences", 0, 5, 2, key="enhanced_home_injuries")
            away_injuries = st.slider("Away Key Absences", 0, 5, 2, key="enhanced_away_injuries")
            
            home_absence_impact = st.select_slider(
                "Home Team Absence Impact",
                options=["Rotation Player", "Regular Starter", "Key Player", "Star Player", "Multiple Key Players"],
                value="Regular Starter",
                key="enhanced_home_absence_impact"
            )
            away_absence_impact = st.select_slider(
                "Away Team Absence Impact",
                options=["Rotation Player", "Regular Starter", "Key Player", "Star Player", "Multiple Key Players"],
                value="Regular Starter",
                key="enhanced_away_absence_impact"
            )
            
        with model_col2:
            st.write("**Enhanced Motivation Factors**")
            home_motivation = st.select_slider(
                "Home Team Motivation",
                options=["Low", "Normal", "High", "Very High"],
                value="Normal",
                key="enhanced_home_motivation"
            )
            away_motivation = st.select_slider(
                "Away Team Motivation", 
                options=["Low", "Normal", "High", "Very High"],
                value="Normal", 
                key="enhanced_away_motivation"
            )
            
            st.write("**Enhanced Simulation**")
            mc_iterations = st.select_slider(
                "Monte Carlo Iterations",
                options=[10000, 25000, 50000],
                value=25000,
                key="enhanced_mc_iterations"
            )
            
            bankroll = st.number_input("Enhanced Bankroll ($)", min_value=500, value=1000, step=100, key="enhanced_bankroll")
            kelly_fraction = st.slider("Enhanced Kelly Fraction", 0.1, 0.3, 0.2, key="enhanced_kelly_fraction")

    submitted = st.button("🎯 GENERATE PROFESSIONAL ANALYSIS", type="primary", use_container_width=True)
    
    if submitted:
        if not home_team or not away_team:
            st.error("❌ Please enter both team names")
            return None, None
        
        if home_team == away_team:
            st.error("❌ Home and away teams cannot be the same")
            return None, None
        
        form_map = {"Win (3 pts)": 3, "Draw (1 pt)": 1, "Loss (0 pts)": 0}
        home_form_points = [form_map[result] for result in home_form]
        away_form_points = [form_map[result] for result in away_form]
        
        motivation_map = {"Low": "Low", "Normal": "Normal", "High": "High", "Very High": "Very High"}
        
        absence_impact_map = {
            "Rotation Player": 1,
            "Regular Starter": 2,
            "Key Player": 3, 
            "Star Player": 4,
            "Multiple Key Players": 5
        }
        
        market_odds = {
            '1x2 Home': home_odds,
            '1x2 Draw': draw_odds,
            '1x2 Away': away_odds,
            'Over 1.5 Goals': over_15_odds,
            'Over 2.5 Goals': over_25_odds,
            'Over 3.5 Goals': over_35_odds,
            'BTTS Yes': btts_yes_odds,
            'BTTS No': btts_no_odds,
        }
        
        match_data = {
            'home_team': home_team,
            'away_team': away_team,
            'league': selected_league,
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
            'injuries': {
                'home': absence_impact_map[home_absence_impact],
                'away': absence_impact_map[away_absence_impact]
            },
            'motivation': {
                'home': motivation_map[home_motivation],
                'away': motivation_map[away_motivation]
            },
            'market_odds': market_odds,
            'bankroll': bankroll,
            'kelly_fraction': kelly_fraction
        }
        
        return match_data, mc_iterations
    
    return None, None

def main():
    """Main application function"""
    # Initialize session state
    if 'enhanced_predictions' not in st.session_state:
        st.session_state.enhanced_predictions = None
    
    if 'enhanced_prediction_history' not in st.session_state:
        st.session_state.enhanced_prediction_history = []
    
    if 'match_data' not in st.session_state:
        st.session_state.match_data = None
    
    # Display existing predictions if available
    if st.session_state.enhanced_predictions and st.session_state.match_data:
        display_enhanced_predictions(st.session_state.enhanced_predictions, st.session_state.match_data)
        
        # Professional betting analysis
        predictions = st.session_state.enhanced_predictions
        if predictions.get('professional_calibration'):
            with st.expander("🎯 Professional Betting Analysis"):
                narrative = predictions.get('match_narrative', {})
                pro_cal = predictions.get('professional_calibration', {})
                
                if narrative.get('home_advantage_amplified'):
                    st.success("""
                    **🏠 HOME ADVANTAGE BETTING OPPORTUNITY**
                    
                    Strong recent home form has overridden team reputation, creating value in:
                    - Home Win markets
                    - Home team to score first
                    - Home clean sheet possibilities
                    """)
                
                if narrative.get('away_scoring_issues'):
                    st.warning("""
                    **✈️ AWAY SCORING CONCERNS**
                    
                    Poor away scoring form suggests value in:
                    - BTTS No markets  
                    - Under 2.5 goals
                    - Home team clean sheet
                    """)
                
                # Professional stake calculation
                base_bankroll = st.session_state.match_data.get('bankroll', 1000)
                base_stake = base_bankroll * 0.02
                
                calibrator = ProfessionalLeagueCalibrator()
                professional_stake = calibrator.calculate_professional_stake(
                    pro_cal.get('final_professional', 0.7), 
                    base_stake,
                    predictions.get('league', 'premier_league'),
                    'protected_single'
                )
                
                st.info(f"""
                **💰 Professional Stake Sizing**
                - Base Stake: ${base_stake:.2f} (2% of bankroll)
                - Volatility Multiplier: {pro_cal.get('volatility_multiplier', 1.0):.1f}x
                - Professional Stake: **${professional_stake:.2f}** ({(professional_stake/base_bankroll)*100:.1f}% of bankroll)
                """)
        
        # Navigation buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 New Professional Analysis", use_container_width=True):
                st.session_state.enhanced_predictions = None
                st.session_state.match_data = None
                st.rerun()
        
        with col2:
            if st.button("📊 Professional History", use_container_width=True):
                if st.session_state.enhanced_prediction_history:
                    st.write("**Professional Prediction History:**")
                    for i, pred in enumerate(st.session_state.enhanced_prediction_history[-5:]):
                        with st.expander(f"Professional Analysis {i+1}: {pred.get('match', 'Unknown Match')} (Pro IQ: {pred.get('football_iq', 0):.1f})"):
                            st.write(f"Date: {pred.get('timestamp', 'N/A')}")
                            st.write(f"League: {get_league_display_name(pred.get('league', 'premier_league'))}")
                            st.write(f"Context: {get_context_display_name(pred.get('primary_context', 'balanced'))}")
                            st.write(f"Expected Goals: Home {pred['expected_goals'].get('home', 0):.2f} - Away {pred['expected_goals'].get('away', 0):.2f}")
                            st.write(f"Team Tiers: {pred.get('team_tiers', {}).get('home', 'N/A')} vs {pred.get('team_tiers', {}).get('away', 'N/A')}")
                            st.write(f"Professional IQ: {pred.get('football_iq', 0):.1f}/100")
                            st.write(f"Pro Confidence: {pred.get('pro_confidence', 0):.1f}%")
                else:
                    st.info("No professional prediction history yet.")
        
        with col3:
            if st.button("🎯 System Status", use_container_width=True):
                if st.session_state.enhanced_predictions:
                    league = st.session_state.enhanced_predictions.get('league', 'premier_league')
                    pro_cal = st.session_state.enhanced_predictions.get('professional_calibration', {})
                    
                    st.success(f"""
                    **Professional System Status: OPERATIONAL** 🟢
                    
                    **Active Features:**
                    - ✅ League-Aware Calibration ✅
                    - ✅ Dynamic Context Bonuses ✅  
                    - ✅ Volatility-Adjusted Staking ✅
                    - ✅ Professional Confidence Pipeline ✅
                    - ✅ Edge Verification ✅
                    - ✅ League Intelligence ✅
                    
                    **Current League:** {league.replace('_', ' ').title()}
                    **Volatility Multiplier:** {pro_cal.get('volatility_multiplier', 1.0):.1f}x
                    **Model Version:** 3.0.0_professional
                    **Calibration Level:** PROFESSIONAL_LEAGUE_MODE
                    """)
                else:
                    st.info("Professional system ready for analysis")
        
        return
    
    # Get new match data and generate predictions
    match_data, mc_iterations = create_enhanced_input_form()
    
    if match_data:
        with st.spinner("🔍 Running professional multi-league calibrated analysis..."):
            try:
                predictor = ApexEnhancedEngine(match_data)
                predictions = predictor.generate_enhanced_predictions(mc_iterations)
                
                if predictions:
                    predictions['league'] = match_data['league']
                    predictions['bankroll'] = match_data.get('bankroll', 1000)
                    predictions['kelly_fraction'] = match_data.get('kelly_fraction', 0.2)
                    
                    st.session_state.enhanced_predictions = predictions
                    st.session_state.match_data = match_data
                    
                    if 'enhanced_prediction_history' not in st.session_state:
                        st.session_state.enhanced_prediction_history = []
                    
                    # Get professional calibration for history
                    pro_cal = predictions.get('professional_calibration', {})
                    
                    prediction_record = {
                        'timestamp': datetime.now().isoformat(),
                        'match': predictions.get('match', 'Unknown Match'),
                        'league': predictions.get('league', 'premier_league'),
                        'primary_context': predictions.get('match_context', 'balanced'),
                        'expected_goals': predictions.get('expected_goals', {'home': 0, 'away': 0}),
                        'team_tiers': predictions.get('team_tiers', {}),
                        'probabilities': safe_get(predictions, 'probabilities', 'match_outcomes') or {},
                        'football_iq': safe_get(predictions, 'enhanced_intelligence', 'football_iq_score') or 0,
                        'pro_confidence': pro_cal.get('final_professional', 0) * 100,
                        'context_confidence': safe_get(predictions, 'enhanced_intelligence', 'context_confidence') or 0,
                        'stability_bonus': safe_get(predictions, 'enhanced_intelligence', 'form_stability_bonus') or 0,
                        'narrative_features': predictions.get('match_narrative', {}),
                        'volatility_multiplier': pro_cal.get('volatility_multiplier', 1.0)
                    }
                    
                    st.session_state.enhanced_prediction_history.append(prediction_record)
                    
                    # Professional success message
                    if predictions.get('professional_calibration'):
                        narrative = predictions.get('match_narrative', {})
                        pro_cal = predictions['professional_calibration']
                        st.success(f"""
                        ✅ **PROFESSIONAL ANALYSIS COMPLETE!**
                        
                        **Professional Features Activated:**
                        - 🎯 League Calibration: {pro_cal.get('league_adjusted', 0)*100:.1f}% → {pro_cal.get('final_professional', 0)*100:.1f}%
                        - 📊 Volatility Multiplier: {pro_cal.get('volatility_multiplier', 1.0):.1f}x
                        - 🧠 Context Bonus: {pro_cal.get('context_bonus_used', 0)*100:+.1f}%
                        - 💰 Professional Staking: Active
                        """)
                    else:
                        st.success("✅ Professional analysis complete!")
                    
                    st.rerun()
                else:
                    st.error("❌ Failed to generate professional predictions")
                
            except Exception as e:
                st.error(f"❌ Professional analysis error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                st.info("💡 Check professional input parameters and try again")

if __name__ == "__main__":
    main()
