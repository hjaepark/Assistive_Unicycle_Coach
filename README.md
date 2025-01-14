# Project Structure
Assistive_Unicycle_Coach/
<br>├── data/              # Training data and saved models
<br>├── envs/              # Environment definitions
<br>├── train/             # Training scripts
<br>└── wrappers/          # Environment wrappers


# Experiment
## Coach-Student Learning with Reference Solution
Implement a (partial) residual policy approach where the final action is computed as:
        
        final_action = student_action + α * (coach_action - student_action)
        
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
