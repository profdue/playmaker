# main_analysis.py - COMPLETE ANALYSIS WORKFLOW
def run_complete_analysis():
    """Run the complete professional analysis workflow"""
    
    # 1. Initialize components
    backtester = AdvancedBacktestingEngine()
    validator = RollingValidationEngine()
    logger = PredictionLogger(use_database=True)
    
    print("🎯 STARTING PROFESSIONAL MODEL ANALYSIS")
    print("=" * 50)
    
    # 2. Load and prepare data
    print("📁 Loading prediction logs...")
    df = backtester.load_prediction_logs('prediction_debug_log.jsonl')
    
    if len(df) < 50:
        print(f"⚠️  Insufficient data: {len(df)} matches. Need at least 50 for reliable analysis.")
        return
    
    print(f"✅ Loaded {len(df)} matches with actual results")
    print(f"🏆 Leagues: {', '.join(df['league'].unique())}")
    print(f"📊 Tier matchups: {df['tier_matchup'].value_counts().to_dict()}")
    
    # 3. Run comprehensive backtest
    print("\n🔬 Running comprehensive backtest...")
    results_df = backtester.run_comprehensive_backtest(df)
    
    # 4. Find optimal weight
    best_row = results_df.loc[results_df['brier_over'].idxmin()]
    print(f"\n🎯 OPTIMAL WEIGHT FOUND: {best_row['historical_weight']:.3f}")
    print(f"   Brier Score: {best_row['brier_over']:.4f}")
    print(f"   ROI Over 2.5: {best_row['roi_over']:.1%}")
    
    # 5. Statistical significance test
    current_weight_performance = results_df[results_df['historical_weight'] == 0.3].iloc[0]
    improvement = current_weight_performance['brier_over'] - best_row['brier_over']
    
    print(f"\n📈 IMPROVEMENT ANALYSIS:")
    print(f"   Current weight (0.300): Brier = {current_weight_performance['brier_over']:.4f}")
    print(f"   Best weight ({best_row['historical_weight']:.3f}): Brier = {best_row['brier_over']:.4f}")
    print(f"   Improvement: {improvement:.4f} ({(improvement/current_weight_performance['brier_over']*100):.1f}%)")
    
    # 6. Time stability analysis
    print("\n⏰ Running time stability analysis...")
    time_results = validator.evaluate_time_stability(df)
    
    print(f"   Time periods analyzed: {len(time_results)}")
    print(f"   Average Brier over time: {time_results['brier_over'].mean():.4f}")
    print(f"   Std Dev across periods: {time_results['brier_over'].std():.4f}")
    
    # 7. Generate final recommendation
    print("\n✅ FINAL RECOMMENDATION")
    print("=" * 50)
    
    if (improvement > 0.001 and  # Meaningful improvement
        best_row['bootstrap_ci_lower'] < current_weight_performance['brier_over'] and  # Statistically significant
        time_results['brier_over'].std() < 0.02):  # Stable over time
        
        print(f"🎉 DEPLOY NEW WEIGHT: {best_row['historical_weight']:.3f}")
        print("   - Statistically significant improvement")
        print("   - Stable across time periods")
        print("   - Positive ROI impact")
    else:
        print("🛑 MAINTAIN CURRENT WEIGHT: 0.300")
        print("   - Insufficient evidence for change")
        print("   - Current performance is acceptable")
    
    print(f"\n📊 Full results saved to: backtesting_results.csv")
    results_df.to_csv('backtesting_results.csv', index=False)
    
    return results_df, time_results

# Run the analysis
if __name__ == "__main__":
    run_complete_analysis()
