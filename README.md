




# Assistive Unicycle Coach

Safe Robotics Lab, 2024-25 | Georgia Tech

## Project Structure

```Text
Assistive_Unicycle_Coach/
├── data/              # Training data and saved models
├── envs/              # Environment definitions
└── train/             # Training scripts
```

## Setup

Download requirements with `pip install -r requirements.txt`

## Experiment

Invoke training as so: `python -m train.unicycle_train.student_train`

and human subject experiment as so: `python -m hse.experiment_runner`

### Logging

Activate TensorBoard with `tensorboard --logdir ./data/logs`

### Parameteres

4 modes, 4 strategies, alpha 0-1