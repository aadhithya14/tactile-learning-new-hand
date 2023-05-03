# This is just a mock environment implementation that will get a dictionary of values and step will 
# just return a time step with the expected values

# from dm_env import TimeStep

import dm_env
import numpy as np

from typing import Any, NamedTuple
from dm_env import StepType, specs, TimeStep

from tactile_learning.utils import get_inverse_image_norm

class Spec:
    max_episode_steps = 70

class ExtendedTimeStep(NamedTuple):
	step_type: Any
	reward: Any
	discount: Any
	observation: Any
	action: Any
	base_action: Any

	def first(self):
		return self.step_type == StepType.FIRST

	def mid(self):
		return self.step_type == StepType.MID

	def last(self):
		return self.step_type == StepType.LAST

	def __getitem__(self, attr):
		return getattr(self, attr)

# Returns similar outputs to a TimeStep
class MockEnv(dm_env.Environment):
    def __init__(self, episodes):
        self.episodes = episodes
        self.current_step = 0

        self.__dict__['spec'] = Spec()

        # Set the DM Env requirements
        self._action_spec = specs.BoundedArray(
            shape=(23,), # Should be tuple - or an iterable
            dtype=np.float32,
            minimum=-1,
            maximum=+1,
            name='action'
        )
        # Observation spec
        self._obs_spec = {}
        self._obs_spec['pixels'] = specs.BoundedArray(
            shape=(480,480,3), # This is how we're transforming the image observations
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name='pixels'
        )
        self._obs_spec['tactile'] = specs.Array(
			shape =(1024,),
			dtype=np.float32, # NOTE: is this a problem?
			name ='tactile' # We will receive the representation directly
		)

        self.inv_image_transform = get_inverse_image_norm()

        # print('episodes: {}'.format(episodes['end_of_demos']))

        # print(f'EPISODES IN MOCKENV: {episodes['end_of_demos']}')

    def reset(self, **kwargs):
        # Find the step with the next demonstration
        self.current_step = self._find_closest_next_demo()

        obs = {}
        obs['pixels'] = self.episodes['image_obs'][self.current_step].detach().cpu().numpy()
        obs['tactile'] = self.episodes['tactile_reprs'][self.current_step].detach().cpu().numpy()

        # print('')

        # step_type = StepType.LAST if done else StepType.MID

        return ExtendedTimeStep(
            step_type = StepType.FIRST,
            reward = 0, # Reward will always be calculated by the ot rewarder
            discount = 1.0, # Hardcoded for now
            observation = obs,
            action = np.zeros(23),
            base_action = np.zeros(23)
        )

    def _find_closest_next_demo(self):
        offset_step = 0
        if self.current_step == 0:
            return self.current_step
        
        while self.episodes['end_of_demos'][(self.current_step+offset_step) % len(self.episodes['end_of_demos'])] != 1:
            offset_step += 1

        next_demo_step = (self.current_step+offset_step+1) % len(self.episodes['end_of_demos'])
        return next_demo_step

    def step(self, action, base_action):
        # observation, reward, done, info = self._env.step(action)
        # obs = {}
        # obs['pixels'] = observation['pixels'].astype(np.uint8)
        # # We will be receiving 
        # obs['goal_achieved'] = info['is_success']
        # return obs, reward, done, info
        # self.current_step = (self.current_step+1) % len(self.episodes['end_of_demos'])
        self.current_step += 1

        obs = {}
        obs['pixels'] = self.episodes['image_obs'][self.current_step].detach().cpu().numpy() # NOTE: These are added to imitate the actual environment - but not sure if works
        obs['tactile'] = self.episodes['tactile_reprs'][self.current_step].detach().cpu().numpy() # It should mock exactly the last 

        step_type = StepType.LAST if self.episodes['end_of_demos'][self.current_step] == 1 else StepType.MID
    
        return ExtendedTimeStep(
            step_type = step_type,
            reward = 0, # Reward will always be calculated by the ot rewarder
            discount = 1.0, # Hardcoded for now
            observation = obs,
            action = action, 
            base_action = base_action
        )

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec

    def render(self):
        return self.inv_image_transform(self.episodes['image_obs'][self.current_step]) # These should already be 

    # def __getattr__(self, name):
    #     return getattr(self._env, name)
