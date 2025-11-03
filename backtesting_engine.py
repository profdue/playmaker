# backtesting_engine.py - ENHANCED WITH LEAGUE NORMALIZATION & VISUALIZATION
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import resample
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedBacktestingEngine:
    def __init__(self):
        self.results = {}
        
    def load_prediction_logs(self, jsonl_file: str) -> pd.DataFrame:
        """Load prediction logs from JSONL file"""
        records = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                record = json.loads(line)
                records.append(record)
        
        return self._create_analysis_dataframe(records)
    
    def _create_analysis_dataframe(self, records: List[Dict]) -> pd.DataFrame:
        """Create analysis-ready DataFrame from prediction logs"""
        data = []
        
        for record in records:
            debug = record['debug_data']
            actual = record.get('actual_result', {})
            
            # Skip records without actual results for backtesting
            if not actual or actual.get('home_goals') is None:
                continue
                
            row = {
                'match_id': f"{debug['match'].replace(' ', '_')}_{debug['timestamp']}",
                'timestamp': debug['timestamp'],
                'league': debug['league'],
                'home_team': debug['match'].split(' vs ')[0],
                'away_team': debug['match'].split(' vs ')[1],
                'home_tier': debug['team_tiers']['home'],
                'away_tier': debug['team_tiers']['away'],
                'tier_matchup': f"{debug['team_tiers']['home']}_vs_{debug['team_tiers']['away']}",
                'home_xg': debug['expected_goals']['home'],
                'away_xg': debug['expected_goals']['away'],
                'total_xg': debug['expected_goals']['total'],
                'poisson_over25': debug['poisson_probabilities']['over_25'],
                'historical_over25': debug['historical_factors']['over_25'],
                'final_over25': debug['final_probabilities']['over_25'],
                'base_btts': debug['poisson_probabilities']['btts_base'],
                'historical_btts': debug['historical_factors']['btts'],
                'final_btts_yes': debug['final_probabilities']['btts_yes'],
                'match_context': debug['match_context'],
                'confidence_score': debug['confidence_score'],
                'data_quality': debug['data_quality'],
                'actual_home_goals': actual.get('home_goals', 0),
                'actual_away_goals': actual.get('away_goals', 0),
                'actual_total_goals': actual.get('home_goals', 0) + actual.get('away_goals', 0),
                'actual_over25': 1 if (actual.get('home_goals', 0) + actual.get('away_goals', 0)) > 2.5 else 0,
                'actual_btts': 1 if actual.get('home_goals', 0) > 0 and actual.get('away_goals', 0) > 0 else 0
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def run_comprehensive_backtest(self, df: pd.DataFrame, n_bootstrap: int = 1000) -> Dict:
        """Run comprehensive backtest with bootstrap confidence intervals"""
        
        weights = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
        results = []
        
        print("🔬 Running comprehensive backtest...")
        print(f"📊 Analyzing {len(df)} matches across {df['league'].nunique()} leagues")
        
        for hw in weights:
            print(f"Testing weight: {hw:.2f}")
            metrics = self._evaluate_weight_comprehensive(df, hw, n_bootstrap)
            results.append({
                'historical_weight': hw,
                **metrics
            })
        
        results_df = pd.DataFrame(results)
        
        # Generate visualizations
        self._generate_calibration_plots(df, weights)
        self._generate_segment_analysis(df)
        
        return results_df
    
    def _evaluate_weight_comprehensive(self, df: pd.DataFrame, historical_weight: float, 
                                     n_bootstrap: int = 1000) -> Dict:
        """Comprehensive evaluation with bootstrap confidence intervals"""
        
        # Recalculate probabilities with new weight
        df_test = df.copy()
        df_test['test_over25'] = (df_test['poisson_over25'] * (1 - historical_weight) + 
                                 df_test['historical_over25'] * historical_weight)
        df_test['test_btts_yes'] = (df_test['base_btts'] * (1 - historical_weight) + 
                                   df_test['historical_btts'] * historical_weight)
        
        # Overall metrics
        overall_metrics = self._calculate_metrics(df_test, 'test_over25', 'test_btts_yes')
        
        # Bootstrap confidence intervals
        bootstrap_metrics = self._bootstrap_metrics(df_test, 'test_over25', 'test_btts_yes', n_bootstrap)
        
        # Segment analysis
        segment_metrics = self._calculate_segment_metrics(df_test, 'test_over25', 'test_btts_yes')
        
        # League-normalized metrics
        league_metrics = self._calculate_league_normalized_metrics(df_test, 'test_over25', 'test_btts_yes')
        
        return {
            **overall_metrics,
            'bootstrap_ci_lower': bootstrap_metrics['ci_lower'],
            'bootstrap_ci_upper': bootstrap_metrics['ci_upper'],
            'segment_metrics': segment_metrics,
            'league_metrics': league_metrics
        }
    
    def _calculate_metrics(self, df: pd.DataFrame, over_col: str, btts_col: str) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        
        # Basic probabilistic metrics
        brier_over = self.brier_score(df[over_col].values, df['actual_over25'].values)
        brier_btts = self.brier_score(df[btts_col].values, df['actual_btts'].values)
        
        logloss_over = self.log_loss(df[over_col].values, df['actual_over25'].values)
        logloss_btts = self.log_loss(df[btts_col].values, df['actual_btts'].values)
        
        # Sharpness (variance of predictions)
        sharpness_over = np.var(df[over_col].values)
        sharpness_btts = np.var(df[btts_col].values)
        
        # Calibration error
        cal_error_over = self.calibration_error(df[over_col].values, df['actual_over25'].values)
        cal_error_btts = self.calibration_error(df[btts_col].values, df['actual_btts'].values)
        
        # P&L simulation (simplified)
        roi_over = self.simulate_betting_roi(df, over_col, 'actual_over25')
        roi_btts = self.simulate_betting_roi(df, btts_col, 'actual_btts')
        
        return {
            'brier_over': brier_over,
            'brier_btts': brier_btts,
            'logloss_over': logloss_over,
            'logloss_btts': logloss_btts,
            'sharpness_over': sharpness_over,
            'sharpness_btts': sharpness_btts,
            'calibration_error_over': cal_error_over,
            'calibration_error_btts': cal_error_btts,
            'roi_over': roi_over,
            'roi_btts': roi_btts
        }
    
    def simulate_betting_roi(self, df: pd.DataFrame, prob_col: str, outcome_col: str, 
                           min_prob: float = 0.55, stake: float = 1.0) -> float:
        """Simulate betting ROI with Kelly-like strategy"""
        betting_matches = df[df[prob_col] >= min_prob].copy()
        
        if len(betting_matches) == 0:
            return 0.0
        
        # Simple ROI calculation
        total_stake = len(betting_matches) * stake
        total_return = betting_matches[outcome_col].sum() * stake * 2  # Assuming even odds
        
        roi = (total_return - total_stake) / total_stake if total_stake > 0 else 0.0
        return roi
    
    def _bootstrap_metrics(self, df: pd.DataFrame, over_col: str, btts_col: str, 
                          n_bootstrap: int = 1000) -> Dict:
        """Calculate bootstrap confidence intervals for Brier score"""
        brier_scores = []
        
        for _ in range(n_bootstrap):
            sample = resample(df)
            brier_over = self.brier_score(sample[over_col].values, sample['actual_over25'].values)
            brier_scores.append(brier_over)
        
        ci_lower = np.percentile(brier_scores, 2.5)
        ci_upper = np.percentile(brier_scores, 97.5)
        
        return {'ci_lower': ci_lower, 'ci_upper': ci_upper}
    
    def _calculate_segment_metrics(self, df: pd.DataFrame, over_col: str, btts_col: str) -> Dict:
        """Calculate metrics for different segments"""
        segments = {
            'elite_vs_elite': df[df['tier_matchup'] == 'ELITE_vs_ELITE'],
            'strong_vs_weak': df[(df['home_tier'] == 'STRONG') & (df['away_tier'] == 'WEAK')],
            'high_confidence': df[df['confidence_score'] >= 75],
            'bundesliga': df[df['league'] == 'bundesliga'],
            'premier_league': df[df['league'] == 'premier_league']
        }
        
        segment_results = {}
        for segment_name, segment_df in segments.items():
            if len(segment_df) > 10:  # Minimum sample size
                metrics = self._calculate_metrics(segment_df, over_col, btts_col)
                segment_results[segment_name] = {
                    'sample_size': len(segment_df),
                    'brier_over': metrics['brier_over'],
                    'roi_over': metrics['roi_over']
                }
        
        return segment_results
    
    def _calculate_league_normalized_metrics(self, df: pd.DataFrame, over_col: str, btts_col: str) -> Dict:
        """Calculate metrics normalized by league"""
        league_metrics = {}
        
        for league in df['league'].unique():
            league_df = df[df['league'] == league]
            if len(league_df) > 5:  # Minimum sample size per league
                metrics = self._calculate_metrics(league_df, over_col, btts_col)
                league_metrics[league] = {
                    'sample_size': len(league_df),
                    'brier_over': metrics['brier_over'],
                    'avg_goals': league_df['actual_total_goals'].mean()
                }
        
        return league_metrics
    
    def _generate_calibration_plots(self, df: pd.DataFrame, weights: List[float]):
        """Generate calibration plots for visual analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Calibration curve for current weights
        self._plot_calibration_curve(df, 'final_over25', 'actual_over25', 
                                   'Current Over 2.5 Calibration', axes[0, 0])
        self._plot_calibration_curve(df, 'final_btts_yes', 'actual_btts', 
                                   'Current BTTS Calibration', axes[0, 1])
        
        # Weight comparison
        self._plot_weight_comparison(df, weights, axes[1, 0])
        self._plot_segment_performance(df, axes[1, 1])
        
        plt.tight_layout()
        plt.savefig('calibration_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_calibration_curve(self, df: pd.DataFrame, pred_col: str, actual_col: str, 
                              title: str, ax):
        """Plot calibration curve"""
        from sklearn.calibration import calibration_curve
        
        prob_true, prob_pred = calibration_curve(df[actual_col], df[pred_col], n_bins=10)
        
        ax.plot(prob_pred, prob_true, 's-', label='Model')
        ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Actual Frequency')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_weight_comparison(self, df: pd.DataFrame, weights: List[float], ax):
        """Plot weight comparison"""
        brier_scores = []
        
        for hw in weights:
            test_over25 = (df['poisson_over25'] * (1 - hw) + df['historical_over25'] * hw)
            brier = self.brier_score(test_over25.values, df['actual_over25'].values)
            brier_scores.append(brier)
        
        ax.plot(weights, brier_scores, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Historical Weight')
        ax.set_ylabel('Brier Score (Lower is Better)')
        ax.set_title('Weight Optimization: Brier Score vs Historical Weight')
        ax.grid(True, alpha=0.3)
        
        # Mark the best weight
        best_idx = np.argmin(brier_scores)
        ax.plot(weights[best_idx], brier_scores[best_idx], 'ro', markersize=10, 
                label=f'Best: {weights[best_idx]:.2f}')
        ax.legend()
    
    def _plot_segment_performance(self, df: pd.DataFrame, ax):
        """Plot segment performance comparison"""
        segments = {
            'All Leagues': df,
            'Bundesliga': df[df['league'] == 'bundesliga'],
            'Premier League': df[df['league'] == 'premier_league'],
            'Elite vs Elite': df[df['tier_matchup'] == 'ELITE_vs_ELITE']
        }
        
        segment_names = []
        brier_scores = []
        
        for name, segment_df in segments.items():
            if len(segment_df) > 5:
                segment_names.append(name)
                brier_scores.append(self.brier_score(
                    segment_df['final_over25'].values, 
                    segment_df['actual_over25'].values
                ))
        
        bars = ax.bar(segment_names, brier_scores, color='skyblue', alpha=0.7)
        ax.set_ylabel('Brier Score')
        ax.set_title('Model Performance by Segment')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, brier_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{score:.3f}', ha='center', va='bottom')
    
    # Existing utility methods (brier_score, log_loss, calibration_error) remain the same
    def brier_score(self, probs, outcomes):
        return np.mean((np.array(probs) - np.array(outcomes)) ** 2)
    
    def log_loss(self, probs, outcomes):
        eps = 1e-15
        probs = np.clip(probs, eps, 1-eps)
        return -np.mean(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs))
    
    def calibration_error(self, probs, outcomes, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probs, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        calibration_error = 0
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_probs = np.array(probs)[mask]
                bin_outcomes = np.array(outcomes)[mask]
                avg_pred = np.mean(bin_probs)
                avg_actual = np.mean(bin_outcomes)
                calibration_error += np.abs(avg_pred - avg_actual) * len(bin_probs)
        
        return calibration_error / len(probs)
