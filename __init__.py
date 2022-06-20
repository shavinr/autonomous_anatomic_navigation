from gym.envs.registration import register

register(
    id='CTsim-v0',
    entry_point='simulus.environments:CTSimulatorEnv',
)