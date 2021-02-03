from gym.envs.registration import register

register(
        id = 'uav-v0',
        entry_point='gym_uav.envs:UavEnv',
        )
## Add another register(...) for another environment
