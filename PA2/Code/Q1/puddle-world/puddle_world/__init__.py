from gym.envs.registration import register

register(
    id='puddle-v0',
    entry_point='puddle_world.envs:PuddleWorld',
)