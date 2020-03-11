from gym.envs.registration import register

register(
    id='puddle_world-v0',
    entry_point='puddle_world.envs:PuddleWorld',
)