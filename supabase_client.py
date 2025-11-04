# supabase_client.py
import os
from supabase import create_client, Client
from typing import Dict, Any, List
import json
from datetime import datetime

class SupabasePredictionSaver:
    def __init__(self):
        # Get credentials from environment variables (more secure)
        self.url = os.getenv('SUPABASE_URL')
        self.key = os.getenv('SUPABASE_KEY')
        
        if not self.url or not self.key:
            # Fallback to Streamlit secrets (if using Streamlit Cloud)
            try:
                import streamlit as st
                self.url = st.secrets["supabase"]["url"]
                self.key = st.secrets["supabase"]["key"]
            except:
                print("⚠️ Supabase credentials not found. Predictions will not be saved.")
                self.client = None
                return
        
        try:
            self.client: Client = create_client(self.url, self.key)
            print("✅ Supabase client initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Supabase client: {e}")
            self.client = None
    
    def save_prediction(self, prediction_data: Dict[str, Any]) -> bool:
        """Save prediction to Supabase"""
        if not self.client:
            print("⚠️ Supabase client not available - skipping save")
            return False
        
        try:
            # Prepare data for Supabase
            prediction_record = {
                'created_at': datetime.now().isoformat(),
                'match': prediction_data.get('match', 'Unknown'),
                'league': prediction_data.get('league', 'premier_league'),
                'home_team': prediction_data.get('match', '').split(' vs ')[0] if ' vs ' in prediction_data.get('match', '') else 'Unknown',
                'away_team': prediction_data.get('match', '').split(' vs ')[1] if ' vs ' in prediction_data.get('match', '') else 'Unknown',
                'expected_goals_home': prediction_data.get('expected_goals', {}).get('home', 0),
                'expected_goals_away': prediction_data.get('expected_goals', {}).get('away', 0),
                'home_win_prob': prediction_data.get('probabilities', {}).get('match_outcomes', {}).get('home_win', 0),
                'draw_prob': prediction_data.get('probabilities', {}).get('match_outcomes', {}).get('draw', 0),
                'away_win_prob': prediction_data.get('probabilities', {}).get('match_outcomes', {}).get('away_win', 0),
                'btts_yes_prob': prediction_data.get('probabilities', {}).get('both_teams_score', {}).get('yes', 0),
                'over_25_prob': prediction_data.get('probabilities', {}).get('over_under', {}).get('over_25', 0),
                'football_iq': prediction_data.get('apex_intelligence', {}).get('football_iq_score', 0),
                'risk_level': prediction_data.get('risk_assessment', {}).get('risk_level', 'UNKNOWN'),
                'data_quality': prediction_data.get('data_quality_score', 0),
                'team_tiers_home': prediction_data.get('team_tiers', {}).get('home', 'MEDIUM'),
                'team_tiers_away': prediction_data.get('team_tiers', {}).get('away', 'MEDIUM'),
                'match_context': prediction_data.get('match_context', 'unknown'),
                'stability_bonus': prediction_data.get('apex_intelligence', {}).get('form_stability_bonus', 0),
                'value_bets_count': len(prediction_data.get('betting_signals', [])),
                'full_prediction_data': json.dumps(prediction_data)  # Store complete data as JSON
            }
            
            # Insert into Supabase
            response = self.client.table('predictions').insert(prediction_record).execute()
            
            if hasattr(response, 'data') and response.data:
                print(f"✅ Prediction saved to Supabase with ID: {response.data[0]['id']}")
                return True
            else:
                print("❌ Failed to save prediction to Supabase")
                return False
                
        except Exception as e:
            print(f"❌ Error saving prediction to Supabase: {e}")
            return False
    
    def get_recent_predictions(self, limit: int = 10) -> List[Dict]:
        """Get recent predictions from Supabase"""
        if not self.client:
            return []
        
        try:
            response = self.client.table('predictions')\
                .select('*')\
                .order('created_at', desc=True)\
                .limit(limit)\
                .execute()
            
            return response.data if hasattr(response, 'data') else []
        except Exception as e:
            print(f"❌ Error fetching predictions from Supabase: {e}")
            return []

# Global instance
prediction_saver = SupabasePredictionSaver()
