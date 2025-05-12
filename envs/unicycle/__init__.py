from gymnasium.envs.registration import register

register(
    id="Unicycle-v0",
    entry_point="envs.unicycle.unicycle_env:Unicycle",
)
