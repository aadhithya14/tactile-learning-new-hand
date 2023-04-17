# Gym implementation of our robotic setup

from collections import deque
from typing import Any, NamedTuple

import gym
# import hand_envs
from gym import  spaces

import dm_env
import numpy as np
from dm_env import StepType, specs, TimeStep
# from dm_control.utils import rewards

class RGBArrayAsObservationWrapper(dm_env.Environment):
	"""
	Use env.render(rgb_array) as observation
	rather than the observation environment provides

	From: https://github.com/hill-a/stable-baselines/issues/915
	"""
	def __init__(self, env, width=84, height=84):
		self._env = env
		self._width = width
		self._height = height
		self._env.reset()
		dummy_obs = self._env.render(mode="rgb_array", width=self._width, height=self._height)
		self.observation_space  = spaces.Box(low=0, high=255, shape=(height, width, 3), dtype=dummy_obs.dtype)
		self.action_space = self._env.action_space
		
		# Action spec
		wrapped_action_spec = self.action_space
		if not hasattr(wrapped_action_spec, 'minimum'):
			wrapped_action_spec.minimum = -np.ones(wrapped_action_spec.shape)
		if not hasattr(wrapped_action_spec, 'maximum'):
			wrapped_action_spec.maximum = np.ones(wrapped_action_spec.shape)
		self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
										np.float32,
										wrapped_action_spec.minimum,
										wrapped_action_spec.maximum,
										'action')
		#Observation spec
		self._obs_spec = {}
		self._obs_spec['pixels'] = specs.BoundedArray(shape=self.observation_space.shape,
													  dtype=np.uint8,
													  minimum=0,
													  maximum=255,
													  name='observation')


	def reset(self, **kwargs):
		obs = {}
		obs = self._env.reset(**kwargs)
		obs['pixels'] = obs['pixels'].astype(np.uint8)
		obs['goal_achieved'] = False
		return obs

	def step(self, action):
		observation, reward, done, info = self._env.step(action)
		obs = {}
		obs['pixels'] = observation['pixels'].astype(np.uint8)
		# We will be receiving 
		obs['goal_achieved'] = info['is_success']
		return obs, reward, done, info

	def observation_spec(self):
		return self._obs_spec

	def action_spec(self):
		return self._action_spec

	def render(self, mode="rgb_array", width=256, height=256):
		return self._env.render(mode="rgb_array", width=width, height=height)

	def __getattr__(self, name):
		return getattr(self._env, name)
	
# It should receive the tactile embeddings and add that as a spec to the observations
class TactileReprAsObservationWrapper(dm_env.Environment):
	def __init__(self, env, tactile_embedding_dim=1024):
		self._env = env 
		self._tactile_size = tactile_embedding_dim

		self._env.reset()

		# Add the tactile obs_spec
		self._obs_spec = self._env.observation_spec()
		self._obs_spec['tactile'] = specs.Array(
			shape = self._tactile_size,
			dtype=np.float32, # NOTE: is this a problem?
			name = 'tactile_repr' # We will receive the representation directly
		)

	def reset(self, **kwargs):
		obs = {}
		obs = self._env.reset(**kwargs)
		obs['tactile'] = obs['tactile'].astype(np.float32) # NOTE: you might need to change this after you implement the environment
		return obs

	def step(self, action):
		observation, reward, done, info = self._env.step(action)
		obs = observation
		obs['tactile'] = observation['tactile'].astype(np.float32)	
		return obs, reward, done, info

	def observation_spec(self):
		return self._obs_spec

	def action_spec(self):
		return self._env.action_spec()

	def render(self, mode="rgb_array", width=256, height=256): # NOTE: We could choose to render tactile as well 
		return self._env.render(mode="rgb_array", width=width, height=height)

	def __getattr__(self, name):
		return getattr(self._env, name)

class ExtendedTimeStep(NamedTuple):
	step_type: Any
	reward: Any
	discount: Any
	observation: Any
	action: Any
	expert_action: Any

	def first(self):
		return self.step_type == StepType.FIRST

	def mid(self):
		return self.step_type == StepType.MID

	def last(self):
		return self.step_type == StepType.LAST

	def __getitem__(self, attr):
		return getattr(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
	def __init__(self, env, num_repeats):
		self._env = env
		self._num_repeats = num_repeats
		
	def step(self, action):
		reward = 0.0
		discount = 1.0
		for i in range(self._num_repeats):
			time_step = self._env.step(action)
			reward += (time_step.reward or 0.0) * discount
			discount *= time_step.discount
			if time_step.last():
				break

		return time_step._replace(reward=reward, discount=discount)

	def observation_spec(self):
		return self._env.observation_spec()

	def action_spec(self):
		return self._env.action_spec()

	def reset(self):
		return self._env.reset()

	def __getattr__(self, name):
		return getattr(self._env, name)

class FrameStackWrapper(dm_env.Environment):
	def __init__(self, env, num_frames):
		self._env = env
		self._num_frames = num_frames
		self._frames = deque([], maxlen=num_frames)
		self._tactile_frames = deque([], maxlen=num_frames)

		wrapped_obs_spec = env.observation_spec()

		pixels_shape = wrapped_obs_spec['pixels'].shape
		if len(pixels_shape) == 4:
			pixels_shape = pixels_shape[1:]
		self._obs_spec = {}
		self._obs_spec['pixels'] = specs.BoundedArray(shape=np.concatenate(
			[[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
											dtype=np.uint8,
											minimum=0,
											maximum=255,
											name='observation')
		
		tactile_shape = wrapped_obs_spec['tactile'].shape
		self._obs_spec['tactile'] = specs.Array(
			shape = (num_frames, tactile_shape),
			dtype = np.float32, 
			name = 'tactile_repr'
		)

	def _transform_observation(self, time_step):
		assert len(self._frames) == self._num_frames
		assert len(self._tactile_frames) == self._num_frames
		obs = {}
		obs['pixels'] = np.concatenate(list(self._frames), axis=0)
		obs['tactile'] = np.concatenate(list(self._tactile_frames), axis=0)
		obs['goal_achieved'] = time_step.observation['goal_achieved']
		return time_step._replace(observation=obs)

	def _extract_pixels(self, time_step):
		pixels = time_step.observation['pixels']
		# remove batch dim
		if len(pixels.shape) == 4:
			pixels = pixels[0]
		return pixels.transpose(2, 0, 1).copy()
	
	def _extract_tactile_repr(self, time_step):
		tactile_repr = time_step.observation['tactile']
		return np.expand_dims(tactile_repr, 0) # Add a new dimension as the `batch` for frame stacking

	def reset(self):
		time_step = self._env.reset()
		pixels = self._extract_pixels(time_step)
		tactiles = self._extract_tactile_repr(time_step)
		for _ in range(self._num_frames):
			self._frames.append(pixels)
			self._tactile_frames.append(tactiles)

		return self._transform_observation(time_step)

	def step(self, action):
		time_step = self._env.step(action)
		pixels = self._extract_pixels(time_step)
		tactiles = self._extract_tactile_repr(time_step)
		self._frames.append(pixels)
		self._tactile_frames.append(tactiles)
		return self._transform_observation(time_step)

	def observation_spec(self):
		return self._obs_spec

	def action_spec(self):
		return self._env.action_spec()

	def __getattr__(self, name):
		return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
	def __init__(self, env, dtype):
		self._env = env
		self._discount = 1.0

		# Action spec
		wrapped_action_spec = env.action_space
		if not hasattr(wrapped_action_spec, 'minimum'):
			wrapped_action_spec.minimum = -np.ones(wrapped_action_spec.shape)
		if not hasattr(wrapped_action_spec, 'maximum'):
			wrapped_action_spec.maximum = np.ones(wrapped_action_spec.shape)
		self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
										np.float32,
										wrapped_action_spec.minimum,
										wrapped_action_spec.maximum,
										'action')
		# #Observation spec
		# self._obs_spec = {}
		# self._obs_spec['pixels'] = specs.BoundedArray(shape=env.observation_space.shape,
		# 								dtype=np.uint8,
		# 								minimum=0,
		# 								maximum=255,
		# 								name='observation')

	def step(self, action):
		action = action.astype(self._env.action_space.dtype)
		# Make time step for action space
		observation, reward, done, info = self._env.step(action)
		reward = reward + 1
		step_type = StepType.LAST if done else StepType.MID
		return TimeStep(
					step_type=step_type,
					reward=reward,
					discount=self._discount,
					observation=observation
				)

	def observation_spec(self):
		return self._env.observation_spec()

	def action_spec(self):
		return self._action_spec

	def reset(self):
		obs = self._env.reset()
		return TimeStep( # This only sort of wraps everything in a timestep
					step_type=StepType.FIRST,
					reward=0,
					discount=self._discount,
					observation=obs
				)

	def __getattr__(self, name):
		return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
	def __init__(self, env):
		self._env = env

	def reset(self):
		time_step = self._env.reset()
		return self._augment_time_step(time_step)

	def step(self, action, vinn_action):
		time_step = self._env.step(action)
		return self._augment_time_step(time_step, action, vinn_action)

	def _augment_time_step(self, time_step, action=None, vinn_action=None):
		if action is None:
			action_spec = self.action_spec()
			action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
			vinn_action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
		return ExtendedTimeStep(observation=time_step.observation,
								step_type=time_step.step_type,
								action=action,
								vinn_action=vinn_action,
								reward=time_step.reward or 0.0,
								discount=time_step.discount or 1.0)

	def _replace(self, time_step, observation=None, action=None, reward=None, discount=None):
		if observation is None:
			observation = time_step.observation
		if action is None:
			action = time_step.action
		if vinn_action is None:
			vinn_action = time_step.vinn_action
		if vinn_next_action is None:
			vinn_next_action = time_step.vinn_next_action
		if reward is None:
			reward = time_step.reward
		if discount is None:
			discount = time_step.discount
		return ExtendedTimeStep(observation=observation,
								step_type=time_step.step_type,
								action=action,
								reward=reward,
								discount=discount)


	def observation_spec(self):
		return self._env.observation_spec()

	def action_spec(self):
		return self._env.action_spec()

	def __getattr__(self, name):
		return getattr(self._env, name)


def make(name, robot_name, host_address, camera_num, height, width, tactile_dim, frame_stack, action_repeat, seed):
	env = gym.make(name, robot_name=robot_name, host_address=host_address, camera_num=camera_num)
	env.seed(seed)
	
	# add wrappers
	env = RGBArrayAsObservationWrapper(env, width=width, height=height)
	env = TactileReprAsObservationWrapper(env, tactile_embedding_dim=tactile_dim)
	env = ActionDTypeWrapper(env, np.float32)
	env = ActionRepeatWrapper(env, action_repeat)
	env = FrameStackWrapper(env, frame_stack)
	env = ExtendedTimeStepWrapper(env)
	return env