from gymnasium.envs.registration import register

register(
    id='Unicycle-v0',
    entry_point='envs.unicycle.unicycle_env:Unicycle',
)

register(
    id='UnicyclePD-v0',
    entry_point='envs.unicycle.unicycle_env_pd_controller:UnicyclePD',
)

register(
    id='UnicyclePendulumPD-v0',
    entry_point='envs.unicycle.unicycle_pendulum_env_pd_controller:UnicyclePendulumPD',
)