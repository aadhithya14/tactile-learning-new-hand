# This is just a mock environment implementation that will get a dictionary of values and step will 
# just return a time step with the expected values

# from dm_env import TimeStep

import dm_env
import numpy as np
from dm_env import StepType, specs, TimeStep

# Returns similar outputs to a TimeStep
class MockEnv(dm_env.Environment):
    def __init__(self, episodes):
        self.episodes = episodes
        self.current_step = 0

        # Set the DM Env requirements
        self._action_spec = specs.BoundedArray(
            shape=(19,), # Should be tuple - or an iterable
            dtype=np.float32,
            minimum=-1,
            maximum=+1,
            name='action'
        )
        # Observation spec
        self._obs_spec = {}
        self._obs_spec['image_obs'] = specs.BoundedArray(
            shape=(480,480,3), # This is how we're transforming the image observations
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name='image_obs'
        )
        self._obs_spec['tactile_repr'] = specs.Array(
			shape =(1024,),
			dtype=np.float32, # NOTE: is this a problem?
			name ='tactile_repr' # We will receive the representation directly
		)

        # print('episodes: {}'.format(episodes['end_of_demos']))

        # print(f'EPISODES IN MOCKENV: {episodes['end_of_demos']}')

    def reset(self, **kwargs):
        # Find the step with the next demonstration
        self.current_step = self._find_closest_next_demo()

        obs = {}
        obs['image_obs'] = self.episodes['image_obs'][self.current_step]
        obs['tactile_repr'] = self.episodes['tactile_reprs'][self.current_step]
        
        # step_type = StepType.LAST if done else StepType.MID

        return TimeStep(
            step_type = StepType.FIRST,
            reward = 0, # Reward will always be calculated by the ot rewarder
            discount = 1.0, # Hardcoded for now
            observation = obs 
        )

    def _find_closest_next_demo(self):
        offset_step = 0
        if self.current_step == 0:
            return self.current_step
        
        while self.episodes['end_of_demos'][(self.current_step+offset_step) % len(self.episodes['end_of_demos'])] != 1:
            offset_step += 1

        next_demo_step = (self.current_step+offset_step+1) % len(self.episodes['end_of_demos'])
        return next_demo_step

    def step(self, action):
        # observation, reward, done, info = self._env.step(action)
        # obs = {}
        # obs['pixels'] = observation['pixels'].astype(np.uint8)
        # # We will be receiving 
        # obs['goal_achieved'] = info['is_success']
        # return obs, reward, done, info
        # self.current_step = (self.current_step+1) % len(self.episodes['end_of_demos'])
        self.current_step += 1

        obs = {}
        obs['image_obs'] = self.episodes['image_obs'][self.current_step]
        obs['tactile_repr'] = self.episodes['tactile_reprs'][self.current_step]

        step_type = StepType.LAST if self.episodes['end_of_demos'][self.current_step] == 1 else StepType.MID
    
        return TimeStep(
            step_type = step_type,
            reward = 0, # Reward will always be calculated by the ot rewarder
            discount = 1.0, # Hardcoded for now
            observation = obs 
        )

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec

    # def render(self, mode="rgb_array", width=256, height=256):
    #     return self._env.render(mode="rgb_array", width=width, height=height)

    # def __getattr__(self, name):
    #     return getattr(self._env, name)
