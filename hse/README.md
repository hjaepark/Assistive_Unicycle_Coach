# Human Subject Experiment

Human subject experiments for unicycle balancing with/without coach assistance.

## Quick Start
```bash
# Pre-test (baseline)
python -m hse.experiment --participant_id P01 --condition without-coach --phase pre-test

# Training (with coach)
python -m hse.experiment --participant_id P01 --condition with-coach --phase training

# Post-test (retention)
python -m hse.experiment --participant_id P01 --condition without-coach --phase post-test
```

## Command Options
```bash
python -m hse.experiment \
    --participant_id P01 \
    --condition {with-coach|without-coach} \
    --phase {pre-test|training|post-test} \
    [--coach_model_path PATH]  # Auto-selects latest if omitted
    [--estimator_config PATH]   # Default: configs/optimized.yaml
    [--debug]                   # Print action details each step
```

## Data Output
```
data/
├── with-coach/
│   └── P01/
│       ├── P01_pre-test_trajectory.csv   # Step-by-step data
│       ├── P01_pre-test_summary.csv      # Aggregated metrics
│       └── ...
└── without-coach/
    └── P02/
        └── ...
```
