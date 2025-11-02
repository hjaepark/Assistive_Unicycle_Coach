# analyze_optuna_results.py

import optuna
import json
import pandas as pd
from pathlib import Path

# Load study
study = optuna.load_study(
    study_name="coach_baseline_sweep",
    storage="sqlite:///optuna_coach_baseline.db"
)

print("="*80)
print(f"📊 Optuna Study Analysis: {len(study.trials)} trials")
print("="*80)

# 1. Best trial info
print("\n🏆 BEST TRIAL")
print(f"Trial number: {study.best_trial.number}")
print(f"Best coach reward: {study.best_value:.6f}")
if 'student_success_rate' in study.best_trial.user_attrs:
    print(f"Student success rate: {study.best_trial.user_attrs['student_success_rate']:.4f}")

print("\n📋 Best parameters:")
for key, value in sorted(study.best_params.items()):
    if 'lr' in key:
        print(f"  {key:15s}: {value:.6f}")
    else:
        print(f"  {key:15s}: {value:.6f}")

# 2. Top 10 trials
print("\n🔝 TOP 10 TRIALS")
completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:10]

top10_data = []
for i, trial in enumerate(sorted_trials, 1):
    success_rate = trial.user_attrs.get('student_success_rate', 'N/A')
    top10_data.append({
        'Rank': i,
        'Trial': trial.number,
        'Coach Reward': f"{trial.value:.4f}",
        'Success Rate': f"{success_rate:.4f}" if success_rate != 'N/A' else 'N/A',
        'rho': f"{trial.params['rho']:.4f}",
        'beta': f"{trial.params['beta']:.6f}",
        'lambda_E': f"{trial.params['lambda_E']:.6f}",
        'kappa': f"{trial.params['kappa']:.4f}",
        'coach_lr': f"{trial.params['coach_lr']:.6f}",
    })

df_top10 = pd.DataFrame(top10_data)
print(df_top10.to_string(index=False))

# 3. Parameter importance
print("\n📈 PARAMETER IMPORTANCE")
try:
    importance = optuna.importance.get_param_importances(study)
    for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {param:15s}: {imp:.4f}")
except Exception as e:
    print(f"  Could not compute importance: {e}")

# 4. Statistics
print("\n📊 TRIAL STATISTICS")
print(f"Total trials: {len(study.trials)}")
print(f"Completed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
print(f"Pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
print(f"Failed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
print(f"Running: {len([t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING])}")

if completed_trials:
    values = [t.value for t in completed_trials]
    print(f"\nCoach Reward Statistics:")
    print(f"  Mean: {sum(values)/len(values):.4f}")
    print(f"  Std: {pd.Series(values).std():.4f}")
    print(f"  Min: {min(values):.4f}")
    print(f"  Max: {max(values):.4f}")
    print(f"  Median: {pd.Series(values).median():.4f}")

# 5. Save detailed results
output = {
    'best_trial': {
        'number': study.best_trial.number,
        'value': study.best_value,
        'params': study.best_params,
        'user_attrs': study.best_trial.user_attrs
    },
    'top_10_trials': [
        {
            'number': t.number,
            'value': t.value,
            'params': t.params,
            'user_attrs': t.user_attrs
        }
        for t in sorted_trials[:10]
    ],
    'statistics': {
        'total_trials': len(study.trials),
        'completed': len(completed_trials),
        'coach_reward_mean': sum(values)/len(values) if values else None,
        'coach_reward_std': pd.Series(values).std() if values else None,
    }
}

with open('optuna_analysis_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n💾 Detailed results saved: optuna_analysis_results.json")

# 6. Generate visualizations
print("\n📊 Generating visualizations...")
try:
    import optuna.visualization as vis
    
    fig = vis.plot_optimization_history(study)
    fig.write_html("optuna_history.html")
    print("  ✓ optuna_history.html")
    
    fig = vis.plot_param_importances(study)
    fig.write_html("optuna_importance.html")
    print("  ✓ optuna_importance.html")
    
    fig = vis.plot_slice(study)
    fig.write_html("optuna_slice.html")
    print("  ✓ optuna_slice.html")
    
    fig = vis.plot_parallel_coordinate(study)
    fig.write_html("optuna_parallel.html")
    print("  ✓ optuna_parallel.html")
    
    fig = vis.plot_contour(study)
    fig.write_html("optuna_contour.html")
    print("  ✓ optuna_contour.html")
    
except Exception as e:
    print(f"  Visualization error: {e}")

print("\n✅ Analysis complete!")