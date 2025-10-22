# train/unicycle_train/cotraining/tune.py

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import yaml
import subprocess
from pathlib import Path
import json
import sys

def create_config_from_trial(trial, base_config_path="train/unicycle_train/cotraining/configs/default.yaml"):
    
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Coach reward function (11 params)
    config['reward']['c_safe'] = trial.suggest_float('c_safe', 0.1, 2.0)
    config['reward']['beta_u'] = trial.suggest_float('beta_u', 0.5, 5.0)
    config['reward']['beta_p'] = trial.suggest_float('beta_p', 0.01, 0.2)
    config['reward']['lambda_E'] = trial.suggest_float('lambda_E', 0.05, 5.0, log=True)
    config['reward']['lambda_W0'] = trial.suggest_float('lambda_W0', 0.005, 0.1, log=True)
    config['reward']['rho'] = trial.suggest_float('rho', 1.0, 3.0)
    config['reward']['p'] = trial.suggest_float('p', 0.5, 2.0)
    config['reward']['c_fall'] = trial.suggest_float('c_fall', 5.0, 20.0)
    config['reward']['b_goal'] = trial.suggest_float('b_goal', 1.0, 10.0)
    config['reward']['resp_ema_alpha'] = trial.suggest_float('resp_ema_alpha', 0.05, 0.3)
    
    kappa = trial.suggest_float('kappa', 0.1, 0.5)
    config['reward']['kappa'] = [kappa, kappa]

    # Coach training schedule (3 params)
    config['coach']['update_freq'] = trial.suggest_int('coach_update_freq', 1, 5)
    config['coach']['gradient_steps'] = trial.suggest_int('coach_gradient_steps', 5, 20)
    config['coach']['learning_starts'] = trial.suggest_int('coach_learning_starts', 500, 2000, step=500)

    # Coach agent hyperparameters (3 params)
    config['coach']['learning_rate'] = trial.suggest_float('coach_lr', 1e-4, 1e-3, log=True)
    config['coach']['gamma'] = trial.suggest_float('coach_gamma', 0.95, 0.995)
    config['coach']['tau'] = trial.suggest_float('coach_tau', 0.001, 0.01, log=True)
    
    # Experiment name
    config['experiment']['name'] = f"optuna_trial_{trial.number:03d}"
    
    # Shorter training for faster feedback
    config['training']['total_timesteps'] = 200000 # Cutting down to 200k for tuning (reduced training length)
    config['training']['allow_early_stopping'] = True  # Early stop when reached target timesteps
    config['training']['eval_interval'] = 10000
    config['training']['eval_episodes'] = 50
    
    return config


def objective(trial):
    
    # 1) Create config
    config = create_config_from_trial(trial)
    
    # 2) Save config
    trial_config_path = f"train/unicycle_train/cotraining/configs/optuna_trial_{trial.number:03d}.yaml"
    Path(trial_config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(trial_config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"\n{'='*80}")
    print(f"Trial {trial.number} started")
    print(f"  lambda_E: {config['reward']['lambda_E']:.4f}")
    print(f"  lambda_W0: {config['reward']['lambda_W0']:.4f}")
    print(f"  beta_u: {config['reward']['beta_u']:.2f}")
    print(f"  kappa: {config['reward']['kappa'][0]:.2f}")
    print(f"  coach_lr: {config['coach']['learning_rate']:.5f}")
    print(f"  update_freq: {config['coach']['update_freq']}")
    print(f"{'='*80}")
    
    # 3) Run training
    try:
        result = subprocess.run([
            "python", "-m", "train.unicycle_train.cotraining.train",
            "--config", trial_config_path
        ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode != 0:
            print(f"Trial {trial.number} failed!")
            print(result.stderr[-500:])  # last 500 characters
            raise optuna.TrialPruned()
        
        # 4) Read results
        logs_dir = Path("train/unicycle_train/cotraining/logs_cotraining")
        run_dirs = sorted([d for d in logs_dir.iterdir() if d.is_dir()], 
                         key=lambda x: x.stat().st_mtime)
        latest_run = run_dirs[-1]
        
        result_file = latest_run / "evaluation_results.json"
        if not result_file.exists():
            print(f"Trial {trial.number}: No results file!")
            raise optuna.TrialPruned()
        
        with open(result_file, 'r') as f:
            results = json.load(f)
            success_rate = results.get('best_success_rate', 0.0)
        
        print(f"Trial {trial.number}: Success rate = {success_rate:.4f}")
        return success_rate
        
    except subprocess.TimeoutExpired:
        print(f"Trial {trial.number}: Timeout!")
        raise optuna.TrialPruned()
    except Exception as e:
        print(f"Trial {trial.number}: Error - {e}")
        raise optuna.TrialPruned()


def main():
    
    # Create study
    study = optuna.create_study(
        study_name="coach_hyperparameter_sweep",
        direction="maximize",
        sampler=TPESampler(seed=42, n_startup_trials=10),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        storage="sqlite:///optuna_coach.db",
        load_if_exists=True
    )
    
    print("="*80)
    print("Optuna Hyperparameter Tuning")
    print("="*80)
    print(f"Database: optuna_coach.db")
    print(f"Study name: coach_full_tuning")
    print(f"Tuning parameters:")
    print("  - Coach Reward Function (11 params)")
    print("  - Coach Training Schedule (3 params)")
    print("  - Coach Agent Hyperparameters (3 params)")
    print("="*80)
    
    # Optimize
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    study.optimize(objective, n_trials=n_trials, timeout=None)
    
    # Results
    print("\n" + "="*80)
    print("Optimization Complete!")
    print("="*80)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best success rate: {study.best_value:.4f}")
    print(f"\nBest parameters:")
    for key, value in sorted(study.best_params.items()):
        print(f"  {key:30s}: {value}")
    
    # Save best config
    best_config = create_config_from_trial(study.best_trial)
    best_config['experiment']['name'] = 'optuna_best'
    best_config['training']['total_timesteps'] = 400000
    best_config['training']['allow_early_stopping'] = False
    best_config['training']['eval_interval'] = 10000
    best_config['training']['eval_episodes'] = 100
    
    best_config_path = "train/unicycle_train/cotraining/configs/optuna_best.yaml"
    with open(best_config_path, 'w') as f:
        yaml.dump(best_config, f)
    print(f"\nBest config saved: {best_config_path}")
    
    # Visualizations
    try:
        import optuna.visualization as vis
        fig = vis.plot_optimization_history(study)
        fig.write_html("optuna_history.html")
        fig = vis.plot_param_importances(study)
        fig.write_html("optuna_importance.html")
        print("\nVisualizations: optuna_history.html, optuna_importance.html")
    except ImportError:
        print("\nFor visualizations: pip install plotly")


if __name__ == "__main__":
    main()