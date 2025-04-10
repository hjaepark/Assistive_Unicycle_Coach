from gymnasium.envs.registration import register

register(
    id='Student-v0',
    entry_point='envs.unicycle.student_env:Student',
)

register(
    id='StudentHF-v0',
    entry_point='envs.unicycle.student_hf_env:StudentHF',
)

register(
    id='Unicycle-v0',
    entry_point='envs.unicycle.unicycle_env:Unicycle'
)