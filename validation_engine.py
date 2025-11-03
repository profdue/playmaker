# validation_engine.py
import pandas as pd
from datetime import datetime, timedelta

class RollingValidationEngine:
    """Time-aware rolling validation to prevent lookahead bias"""
    
    def __init__(self, window_size: int = 90, step_size: int = 30):
        self.window_size = window_size  # days
        self.step_size = step_size      # days
    
    def time_aware_train_test_split(self, df: pd.DataFrame) -> List[Tuple]:
        """Create time-aware train/test splits"""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        min_date = df['timestamp'].min()
        max_date = df['timestamp'].max()
        
        splits = []
        current_start = min_date
        
        while current_start + timedelta(days=self.window_size) <= max_date:
            train_end = current_start + timedelta(days=self.window_size)
            test_start = train_end
            test_end = test_start + timedelta(days=self.step_size)
            
            if test_end > max_date:
                break
                
            train_mask = (df['timestamp'] >= current_start) & (df['timestamp'] < train_end)
            test_mask = (df['timestamp'] >= test_start) & (df['timestamp'] < test_end)
            
            if train_mask.sum() > 50 and test_mask.sum() > 10:  # Minimum samples
                splits.append((df[train_mask], df[test_mask]))
            
            current_start += timedelta(days=self.step_size)
        
        return splits
    
    def evaluate_time_stability(self, df: pd.DataFrame) -> Dict:
        """Evaluate model stability over time"""
        splits = self.time_aware_train_test_split(df)
        results = []
        
        for i, (train_df, test_df) in enumerate(splits):
            # For each period, we could retrain weights, but for now use current
            brier_over = self.brier_score(test_df['final_over25'].values, 
                                        test_df['actual_over25'].values)
            brier_btts = self.brier_score(test_df['final_btts_yes'].values, 
                                         test_df['actual_btts'].values)
            
            results.append({
                'period': i,
                'start_date': test_df['timestamp'].min(),
                'end_date': test_df['timestamp'].max(),
                'sample_size': len(test_df),
                'brier_over': brier_over,
                'brier_btts': brier_btts,
                'avg_goals': test_df['actual_total_goals'].mean()
            })
        
        return pd.DataFrame(results)
