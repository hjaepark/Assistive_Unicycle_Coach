from gymnasium.envs.registration import register

register(
    id='Student-v0',
    entry_point='envs.student_v0:Student',
)