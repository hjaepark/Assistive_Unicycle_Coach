from gymnasium.envs.registration import register

register(
    id='Student-v0',
    entry_point='envs.student_env:Student',
)

register(
    id='StudentHF-v0',
    entry_point='envs.student_hf_env:StudentHF',
)
