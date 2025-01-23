# Project Structure
Assistive_Unicycle_Coach/
<br>├── data/              # Training data and saved models
<br>├── envs/              # Environment definitions
<br>└── train/             # Training scripts

## Core Components
### Coach-Student Learning with Reference Solution


### 1. Environment (`student_v0.py`)
This module defines the unicycle simulation environment. The environment implements:
- A 22-dimensional state space
- A 3-dimensional action space  
- Reward function and termination conditions for balancing and forward movement tasks

### 2. Coach-Student Wrapper (`coach_student_wrapper.py`) 
This wrapper manages the interaction between student and coach models with the following key functionalities:
- Loading of pre-trained coach models (fully-trained student = coach "expert")
- Definition of action combination methods (residual, partial_residual)
- Intervention control through alpha parameter adjustment

### 3. Training Implementation (`coach_student_rl_v0.py`)
The main training script orchestrates the entire learning process by:
- Configuring the environment and applying necessary wrappers
- Implementing various learning strategies (continuous, performance-based)
- Handling model persistence and evaluation procedures

### 4. Monitoring Tools
**coach_student_monitor.py**
- Tracks comprehensive learning metrics
- Monitors overall system performance

**student_monitor.py**
- Tracks student model performance metrics
- Monitors learning progression

**coach_monitor.py**
- Measures coach intervention frequency 
- Analyzes coach influence on student learning

# Experiment
<br>Activate TensorBoard with

        tensorboard --logdir ./data/logs
## Coach-Student Learning with Reference Solution
Implement a (partial) residual policy approach where the final action is computed as:
        
        final_action = student_action + α * (coach_action - student_action)
        ** Prove if residual policy and partial residual policy equal in computing **
        
where α represents the coach's influence level.

Adaptive Coaching Strategies

Investigate four different strategies for coach engagement to determine the most effective and efficient learning approach:
1. Continuous
<br>Coach maintains full engagement throughout the training process. This serves as our baseline for comparing other strategies, where α remains constant at its initial value.
2. Performance-Based 
<br>Coach remains active until student reaches predefined performance metrics. Performance thresholds are based on success rate and stability. Once these criteria are met, coach influence gradually decreases, allowing the student to operate more independently.
3. Progressive
<br>In this approach, coach influence (α) systematically decreases over training episodes. This implements a scheduled reduction in coach_action magnitude, enabling the student to gradually develop independence while maintaining learning stability.
4. Need-Based
<br>This strategy implements an adaptive intervention mechanism where the coach engages only when student performance drops below threshold. The α value dynamically adjusts based on recent performance metrics, providing support only when necessary.

<br>Each strategy aims to balance learning efficiency with the development of student autonomy. Results will be analyzed based on learning speed, final performance, and resource efficiency.
