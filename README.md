## Project Structure
Assistive_Unicycle_Coach/
<br>├── data/              # Training data and saved models
<br>├── envs/              # Environment definitions
<br>├── train/             # Training scripts
<br>└── wrappers/          # Environment wrappers

## Student Agent Training

### Training
To start the training process:

    python train/student_rl_v0.py -train

### Testing
To test a trained model:


    python train/student_rl_v0.py -test path/to/model.zip


## Coach - Student Agent Training

### Training
To start the training process:

    ex) python coach_student_rl_v0.py --train --coach reference_model.zip


### Testing
To test a trained model:

    ex) python coach_student_rl_v0.py --test path/to_trained_model.zip --coach reference_model.zip
