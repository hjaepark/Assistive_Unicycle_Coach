## Project Structure
Assistive_Unicycle_Coach/
<br>├── data/              # Training data and saved models
<br>├── envs/              # Environment definitions
<br>├── train/             # Training scripts
<br>└── wrappers/          # Environment wrappers

### Training
To start the training process:

    python train/student_rl_v0.py -train

### Testing
To test a trained model:


    python train/student_rl_v0.py -test path/to/model.zip

