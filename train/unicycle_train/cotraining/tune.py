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
    
    # Coach reward hyperparameters
    config['reward']['rho'] = trial.suggest_float('rho', 0.35, 0.75)
    config['reward']['beta'] = trial.suggest_float('beta', 0.01, 0.1, log=True)
    config['reward']['lambda_E'] = trial.suggest_float('lambda_E', 0.05, 0.2, log=True)
    
    kappa = trial.suggest_float('kappa', 0.2, 0.5)
    config['reward']['kappa'] = [kappa, kappa]
    
    config['coach']['learning_rate'] = trial.suggest_float('coach_lr', 2e-4, 5e-4, log=True)

    # SAC hyperparameters (following rl-baselines3-zoo approach) (https://github.com/DLR-RM/rl-baselines3-zoo/blob/ab4aadb57c6c42abcf1016318c2bebe35c4c1270/rl_zoo3/hyperparams_opt.py#L257)
    one_minus_gamma = trial.suggest_float('one_minus_gamma', 0.0001, 0.03, log=True)
    config['student']['gamma'] = 1 - one_minus_gamma
    trial.set_user_attr('gamma', config['student']['gamma'])

    batch_size_pow = trial.suggest_int('batch_size_pow', 6, 10)
    config['student']['batch_size'] = 2 ** batch_size_pow
    trial.set_user_attr('batch_size', config['student']['batch_size'])

    config['student']['learning_rate'] = trial.suggest_float('student_lr', 1e-5, 0.002, log=True)
    config['student']['train_freq'] = trial.suggest_categorical('train_freq', [1, 4, 8, 16, 32, 64])
    config['student']['gradient_steps'] = config['student']['train_freq']
    config['student']['tau'] = trial.suggest_float('tau', 0.001, 0.08, log=True)

    # Fixed hyperparameters
    config['coach']['update_freq'] = 1
    config['coach']['gradient_steps'] = 10
    
    config['experiment']['name'] = f"optuna_baseline_trial_{trial.number:03d}"
    
    config['training']['total_timesteps'] = 400000
    config['training']['allow_early_stopping'] = True
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
    print(f"Coach hyperparameters:")
    print(f"  rho: {config['reward']['rho']:.3f}")
    print(f"  beta: {config['reward']['beta']:.4f}")
    print(f"  lambda_E: {config['reward']['lambda_E']:.4f}")
    print(f"  kappa: {config['reward']['kappa'][0]:.2f}")
    print(f"  coach_lr: {config['coach']['learning_rate']:.5f}")
    print(f"Student (SAC) hyperparameters:")
    print(f"  gamma: {config['student']['gamma']:.4f}")
    print(f"  batch_size: {config['student']['batch_size']}")
    print(f"  student_lr: {config['student']['learning_rate']:.5f}")
    print(f"  train_freq: {config['student']['train_freq']}")
    print(f"  gradient_steps: {config['student']['gradient_steps']}")
    print(f"  tau: {config['student']['tau']:.4f}")
    print(f"{'='*80}")
    
    # 3) Run training
    try:
        result = subprocess.run([
            "python", "-m", "train.unicycle_train.cotraining.train",
            "--config", trial_config_path
        ], capture_output=True, text=True, timeout=43200)
        
        if result.returncode != 0:
            print(f"Trial {trial.number} failed!")
            print(result.stderr[-500:])
            raise optuna.TrialPruned()
        
        # 4) Read coach reward results
        logs_dir = Path("train/unicycle_train/cotraining/logs_cotraining")
        run_dirs = sorted([d for d in logs_dir.iterdir() if d.is_dir()], 
                         key=lambda x: x.stat().st_mtime)
        latest_run = run_dirs[-1]
        
        # Read coach statistics
        coach_stats_file = latest_run / "coach_stats.json"
        if not coach_stats_file.exists():
            print(f"Trial {trial.number}: No coach stats file!")
            raise optuna.TrialPruned()
        
        with open(coach_stats_file, 'r') as f:
            coach_stats = json.load(f)
            # Use last 100 episodes mean as the objective
            coach_reward = coach_stats.get('last_100_mean', 
                                          coach_stats.get('mean_coach_reward', 0.0))
        
        # Optional: Also log student success rate for reference
        result_file = latest_run / "evaluation_results.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                results = json.load(f)
                success_rate = results.get('best_success_rate', 0.0)
                # Store as metadata (not used for optimization)
                trial.set_user_attr('student_success_rate', success_rate)
        
        print(f"Trial {trial.number}: Coach reward = {coach_reward:.4f}")
        return coach_reward  # Return coach reward, not success rate!
        
    except subprocess.TimeoutExpired:
        print(f"Trial {trial.number}: Timeout!")
        raise optuna.TrialPruned()
    except Exception as e:
        print(f"Trial {trial.number}: Error - {e}")
        raise optuna.TrialPruned()


def main():
    # Create study with retry logic for SQLite race condition
    import time
    import random
    
    max_retries = 10
    study = None
    
    for attempt in range(max_retries):
        try:
            study = optuna.create_study(
                study_name="coach_baseline_sweep",
                direction="maximize",
                sampler=TPESampler(seed=None, n_startup_trials=25),  # More random trials initially
                pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
                storage="sqlite:///optuna_coach_baseline.db",
                load_if_exists=True
            )
            break  # Success!
        except Exception as e:
            if attempt < max_retries - 1:
                # Random backoff between 0.5 to 3 seconds
                wait_time = random.uniform(0.5, 3.0)
                print(f"[Warning] DB initialization failed (attempt {attempt+1}/{max_retries}). "
                      f"Retrying in {wait_time:.2f}s... Error: {e}")
                time.sleep(wait_time)
            else:
                print(f"[Error] Failed to initialize study after {max_retries} attempts!")
                raise
    
    if study is None:
        raise RuntimeError("Failed to create Optuna study")
    
    print("="*80)
    print("Optuna Hyperparameter Tuning - Coach + Student")
    print("="*80)
    print(f"Database: optuna_coach_baseline.db")
    print(f"Study name: coach_baseline_sweep")
    print(f"\nCoach hyperparameters:")
    print("  1. rho: 0.35 ~ 0.75 (assistance discount threshold)")
    print("  2. beta: 0.01 ~ 0.1 (usage penalty)")
    print("  3. lambda_E: 0.05 ~ 0.2 (energy penalty)")
    print("  4. kappa: 0.2 ~ 0.5 (intervention scaling)")
    print("  5. coach_lr: 2e-4 ~ 5e-4 (coach learning rate)")
    print(f"\nStudent hyperparameters:")
    print("  6. gamma: 0.97 ~ 0.9999 (discount factor)")
    print("  7. batch_size: 64 ~ 1024 (powers of 2)")
    print("  8. student_lr: 1e-5 ~ 0.002 (learning rate)")
    print("  9. train_freq: [1, 4, 8, 16, 32, 64] (update frequency)")
    print(" 10. gradient_steps: = train_freq (gradient updates)")
    print(" 11. tau: 0.001 ~ 0.08 (Polyak coefficient)")
    print("="*80)
    
    # Optimize
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 100  # Default 100 trials
    print(f"Running {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, timeout=None)
    
    # Results
    print("\n" + "="*80)
    print("Optimization Complete!")
    print("="*80)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best coach reward: {study.best_value:.4f}")

    # Also show student success rate if available
    if 'student_success_rate' in study.best_trial.user_attrs:
        print(f"Student success rate: {study.best_trial.user_attrs['student_success_rate']:.4f}")

    print(f"\nBest parameters:")
    for key, value in sorted(study.best_params.items()):
        if 'lr' in key:
            print(f"  {key:30s}: {value:.5f}")
        else:
            print(f"  {key:30s}: {value:.4f}")
    
    # Save best config
    best_config = create_config_from_trial(study.best_trial)
    best_config['experiment']['name'] = 'optuna_best_baseline'
    best_config['training']['total_timesteps'] = 400000
    best_config['training']['allow_early_stopping'] = False
    best_config['training']['eval_interval'] = 10000
    best_config['training']['eval_episodes'] = 100
    
    best_config_path = "train/unicycle_train/cotraining/configs/optuna_best_baseline.yaml"
    with open(best_config_path, 'w') as f:
        yaml.dump(best_config, f)
    print(f"\nBest config saved: {best_config_path}")
    
    # Visualizations
    try:
        import optuna.visualization as vis
        fig = vis.plot_optimization_history(study)
        fig.write_html("optuna_baseline_history.html")
        fig = vis.plot_param_importances(study)
        fig.write_html("optuna_baseline_importance.html")
        fig = vis.plot_slice(study)
        fig.write_html("optuna_baseline_slice.html")
        print("\nVisualizations saved:")
        print("  - optuna_baseline_history.html")
        print("  - optuna_baseline_importance.html")
        print("  - optuna_baseline_slice.html")
    except ImportError:
        print("\nFor visualizations: pip install plotly")


if __name__ == "__main__":
    main()