from gym.envs.registration import register

register(
    id='room-v0',
    entry_point='room_world.envs:RoomWorld',
)