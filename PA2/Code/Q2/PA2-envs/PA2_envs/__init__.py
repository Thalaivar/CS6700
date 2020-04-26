from gym.envs.registration import register

register(
    id='chakra-v0',
    entry_point='PA2_envs.envs:Chakra',
)

register(
    id='vishamC-v0',
    entry_point='PA2_envs.envs:VishamC',
)