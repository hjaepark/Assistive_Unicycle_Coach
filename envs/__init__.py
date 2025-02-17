from gymnasium.envs.registration import register

register(
    id='Student-v0',
    entry_point='envs.student:Student',
)

register(
    id='StudentHumanFactor-v0',
    entry_point='envs.student_human_factor:StudentHumanFactor',
)