# Agent implementation in fish
"""
This takes potil_vinn_offset and compute q-filter on encoder_vinn and vinn_action_qfilter
"""
import glob
import os
import hydra
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision import transforms as T
import torch.nn.functional as F

# from agent.encoder import Encoder

from tactile_learning.models import *
from tactile_learning.utils import *
from tactile_learning.tactile_data import *

from holobot.robot.allegro.allegro_kdl import AllegroKDL

class FISHAgent:
	def __init__(self,
	 	data_path, demo_num, mock_demo_nums,
		image_out_dir, tactile_out_dir, # This is used to get the tactile representation size
		reward_representations, policy_representations,
		action_shape, device, lr, feature_dim,
		hidden_dim, critic_target_tau, num_expl_steps,
		update_every_steps, stddev_schedule, stddev_clip, augment,
		rewards, sinkhorn_rew_scale, update_target_every,
		auto_rew_scale, auto_rew_scale_factor,
		bc_weight_type, bc_weight_schedule, offset_scale_factor, offset_mask
	):
		
		self.device = device
		self.lr = lr
		self.critic_target_tau = critic_target_tau
		self.update_every_steps = update_every_steps
		# self.use_tb = use_tb
		self.num_expl_steps = num_expl_steps
		self.stddev_schedule = stddev_schedule
		self.stddev_clip = stddev_clip
		self.augment = augment
		self.rewards = rewards
		self.sinkhorn_rew_scale = sinkhorn_rew_scale
		self.update_target_every = update_target_every
		self.auto_rew_scale = auto_rew_scale
		self.auto_rew_scale_factor = auto_rew_scale_factor

		self.bc_weight_type = bc_weight_type
		self.bc_weight_schedule = bc_weight_schedule
		self.offset_scale_factor = offset_scale_factor
    
		# NOTE: What is the q-filter mentioned?
		# NOTE: We might need to lower the representation dimenstion at some point

        # TODO: Load the data for that one demonstration
		self.data_path = data_path
		self.roots = sorted(glob.glob(f'{data_path}/demonstration_*'))
		self.data = load_data(self.roots, demos_to_use=[demo_num])

		# Set the mock data
		self.mock_data = load_data(self.roots, demos_to_use=mock_demo_nums) # TODO: Delete this
		self.kdl_solver = AllegroKDL()

        # TODO: Load the encoders - both the normal ones and the target ones
		image_cfg, self.image_encoder, self.image_transform = init_encoder_info(self.device, image_out_dir, 'image')
		self.inv_image_transform = get_inverse_image_norm() 
		self.image_normalize = T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS)
		# self.tactile_normalize = T.Normalize(TACTILE_IMAGE_MEANS, TACTILE_IMAGE_STDS)
	
		tactile_cfg, self.tactile_encoder, _ = init_encoder_info(self.device, tactile_out_dir, 'tactile')
		tactile_img = TactileImage(
			tactile_image_size = tactile_cfg.tactile_image_size, 
			shuffle_type = None
		)
		self.tactile_repr = TactileRepresentation( # This will be used when calculating the reward - not getting the observations
			encoder_out_dim = tactile_cfg.encoder.out_dim,
			tactile_encoder = self.tactile_encoder,
			tactile_image = tactile_img,
			representation_type = 'tdex'
		)

		self.reward_representations = reward_representations
		self.policy_representations = policy_representations
		self.view_num = 1
	
		# Freeze the encoders
		self.image_encoder.eval()
		self.tactile_encoder.eval()
		for param in self.image_encoder.parameters():
			param.requires_grad = False 
		for param in self.tactile_encoder.parameters():
			param.requires_grad = False

		# Set the expert_demo - it will set the tactile representations and image observations and save them in a dictionary
		self._set_expert_demo()

		repr_dim = 0
		if 'tactile' in policy_representations:
			repr_dim += tactile_cfg.encoder.out_dim
		if 'image' in policy_representations:
			repr_dim += image_cfg.encoder.out_dim
		# repr_dim = image_cfg.encoder.out_dim + tactile_cfg.encoder.out_dim 
		self.actor = Actor(repr_dim, action_shape, feature_dim,
						   hidden_dim, offset_mask).to(device)

		self.critic = Critic(repr_dim, action_shape, feature_dim,
							 hidden_dim).to(device)
		self.critic_target = Critic(repr_dim, action_shape,
									feature_dim, hidden_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
			
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
		self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

		# Data augmentation
		self.image_aug = RandomShiftsAug(pad=4) # This is augmentation for the image

		self.train()
		self.critic_target.train()

		# Current step in episode
		self.count = 0
		# Openloop tracker
		self.curr_step = 0


	def __repr__(self):
		return "fish_agent"

	def train(self, training=True):
		self.training = training
		self.actor.train(training)
		self.critic.train(training)

	def _set_expert_demo(self):
		# We'll stack the tactile repr and the image observations
		for step_id in range(len(self.data['image']['indices'])): 
			demo_id, tactile_id = self.data['tactile']['indices'][step_id]

			# demo_id, _ = self.data['tactile']['indices'][0]
			# root = self.roots[demo_id]
			# demo_num = int(root.split('/')[-1].split('_')[-1])
			# print('expert demo_id: {}, demo_num: {}'.format(demo_id, demo_num))

			tactile_value = self.data['tactile']['values'][demo_id][tactile_id]
			tactile_repr = self.tactile_repr.get(tactile_value, detach=False)

			_, image_id = self.data['image']['indices'][step_id]
			image = load_dataset_image(
				data_path = self.data_path, 
				demo_id = demo_id, 
				image_id = image_id,
				view_num = self.view_num,
				transform = self.image_transform
			)

			if step_id == 0:
				tactile_reprs = tactile_repr.unsqueeze(0)
				image_obs = image.unsqueeze(0)
			else:
				image_obs = torch.concat([image_obs, image.unsqueeze(0)], dim=0)
				tactile_reprs = torch.concat([tactile_reprs, tactile_repr.unsqueeze(0)], dim=0)

		self.expert_demo = dict(
			image_obs = image_obs, 
			tactile_repr = tactile_reprs
		)
		
	def _load_dataset_image(self, demo_id, image_id):
		dset_img = load_dataset_image(self.data_path, demo_id, image_id, self.view_num)
		img = self.image_transform(dset_img)
		return torch.FloatTensor(img) 
		
	# Will give the next action in the step
	def base_act(self, obs): # Returns the action for the base policy
		# if self.count == 0:
			# self.curr_step = 0
		
		# Get the action in the current step
		# demo_id, action_id = self.data['allegro_actions']['indices'][self.curr_step]
		# allegro_action = self.data['allegro_actions']['values'][demo_id][action_id]

		# # NOTE: Set the thumb to the states so far - this should be removed as we fix the thumb calibration
		# _, allegro_state_id = self.data['allegro_joint_states']['indices'][self.curr_step]
		# allegro_state = self.data['allegro_joint_states']['values'][demo_id][allegro_state_id]
		# allegro_action[-4:] = allegro_state[-4:]
		
		# # Get the kinova action 
		# _, kinova_id = self.data['kinova']['indices'][self.curr_step]
		# kinova_action = self.data['kinova']['values'][demo_id][kinova_id]

		# # Concatenate the actions 
		# base_action = np.concatenate([allegro_action, kinova_action], axis=-1).unsqueeze(0)

		# return base_action

		# if step > 0 and step % max_step == 0:
		# 	self.curr_step = self._find_closest_mock_step(self.curr_step)
		
		# Get the action in the current step
		demo_id, action_id = self.data['allegro_actions']['indices'][self.curr_step]
		allegro_joint_action = self.data['allegro_actions']['values'][demo_id][action_id]
		allegro_fingertip_action = self.kdl_solver.get_fingertip_coords(allegro_joint_action)
		
		# Get the kinova action 
		_, kinova_id = self.data['kinova']['indices'][self.curr_step]
		kinova_action = self.data['kinova']['values'][demo_id][kinova_id]

		# Concatenate the actions 
		demo_action = np.concatenate([allegro_fingertip_action, kinova_action], axis=-1)

		self.curr_step = (self.curr_step+1) % (len(self.data['allegro_actions']['indices']))

		return demo_action # Base action will be 0s only for now

	def act(self, obs, step, eval_mode):
		obs = torch.as_tensor(obs, device=self.device).float()		
		if obs.ndim == 3:
			obs = obs.unsqueeze(0)

		with torch.no_grad():
			base_action = self.base_act(obs.clone())

		obs = self.encoder(obs) if self.use_encoder else obs

		stddev = utils.schedule(self.stddev_schedule, step)
		dist = self.actor(obs, base_action, stddev)
		if eval_mode:
			offset_action = dist.mean
		else:
			offset_action = dist.sample(clip=None)
			if step < self.num_expl_steps:
				offset_action.uniform_(-1.0, 1.0)

		action = base_action + offset_action * self.offset_scale_factor

		return action.cpu().numpy()[0], base_action.cpu().numpy()[0]
	
	def _find_closest_mock_step(self, curr_step):
		offset_step = 0
		if curr_step == 0:
			return curr_step

		curr_demo_id, _ = self.mock_data['allegro_actions']['indices'][curr_step]
		next_demo_id, _ = self.mock_data['allegro_actions']['indices'][curr_step+offset_step]
		while curr_demo_id == next_demo_id:
			offset_step += 1
			next_demo_id, _ = self.mock_data['allegro_actions']['indices'][(curr_step+offset_step) % len(self.mock_data['allegro_actions']['indices'])]
			

		next_demo_step = (curr_step+offset_step) % len(self.mock_data['allegro_actions']['indices'])
		return next_demo_step

	# Method that returns the next action in the mock data
	def mock_act(self, obs, step, max_step): # Returns the action for the base policy - TODO: This will be used after we have the environment
		# if self.count == 0:
		# 	self.curr_step = 0
		if step > 0 and step % max_step == 0:
			self.curr_step = self._find_closest_mock_step(self.curr_step)
		
		# Get the action in the current step
		demo_id, action_id = self.mock_data['allegro_actions']['indices'][self.curr_step]
		allegro_joint_action = self.mock_data['allegro_actions']['values'][demo_id][action_id]
		allegro_fingertip_action = self.kdl_solver.get_fingertip_coords(allegro_joint_action)


		# NOTE: Set the thumb to the states so far - this should be removed as we fix the thumb calibration
		# _, allegro_state_id = self.mock_data['allegro_joint_states']['indices'][self.curr_step]
		# allegro_state = self.mock_data['allegro_joint_states']['values'][demo_id][allegro_state_id]
		# allegro_action[-4:] = allegro_state[-4:]
		
		# Get the kinova action 
		_, kinova_id = self.mock_data['kinova']['indices'][self.curr_step]
		kinova_action = self.mock_data['kinova']['values'][demo_id][kinova_id]

		# Concatenate the actions 
		demo_action = np.concatenate([allegro_fingertip_action, kinova_action], axis=-1)

		self.curr_step = (self.curr_step+1) % (len(self.mock_data['allegro_actions']['indices']))

		return demo_action, np.zeros(19) # Base action will be 0s only for now

	def update_critic(self, obs, action, base_next_action, reward, discount, next_obs, step):
		metrics = dict()

		with torch.no_grad():
			stddev = utils.schedule(self.stddev_schedule, step)
			dist = self.actor(next_obs, base_next_action, stddev)

			next_action = base_next_action + dist.sample(clip=self.stddev_clip) * self.offset_scale_factor

			target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
			target_V = torch.min(target_Q1, target_Q2)
			target_Q = reward + (discount * target_V)

		Q1, Q2 = self.critic(obs, action)

		critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

		# optimize encoder and critic
		self.critic_opt.zero_grad(set_to_none=True)
		critic_loss.backward()
		self.critic_opt.step()

		# if self.use_tb:
		metrics['critic_target_q'] = target_Q.mean().item()
		metrics['critic_q1'] = Q1.mean().item()
		metrics['critic_q2'] = Q2.mean().item()
		metrics['critic_loss'] = critic_loss.item()
			
		return metrics

	def update_actor(self, obs, obs_expert, obs_qfilter, action_expert, base_action, base_action_expert, bc_regularize, step):
		metrics = dict()

		stddev = utils.schedule(self.stddev_schedule, step)

		# compute action offset
		dist = self.actor(obs, base_action, stddev)
		action_offset = dist.sample(clip=self.stddev_clip)
		log_prob = dist.log_prob(action_offset).sum(-1, keepdim=True)

		# compute action
		action = base_action + action_offset * self.offset_scale_factor
		Q1, Q2 = self.critic(obs, action)
		Q = torch.min(Q1, Q2)

		# Compute bc weight
		if not bc_regularize: # This will be false
			bc_weight = 0.0
		elif self.bc_weight_type == "linear":
			bc_weight = utils.schedule(self.bc_weight_schedule, step)
		elif self.bc_weight_type == "qfilter": # NOTE: This is for ROT but in FISH they had it set to false
			"""
			Soft Q-filtering inspired from 			
			Nair, Ashvin, et al. "Overcoming exploration in reinforcement 
			learning with demonstrations." 2018 IEEE international 
			conference on robotics and automation (ICRA). IEEE, 2018.
			"""
			with torch.no_grad():
				stddev = 0.1
				action_qf = base_action.clone()
				Q1_qf, Q2_qf = self.critic(obs_qfilter.clone(), action_qf)
				Q_qf = torch.min(Q1_qf, Q2_qf)
				bc_weight = (Q_qf>Q).float().mean().detach()

		actor_loss = - Q.mean() * (1-bc_weight)

		if bc_regularize:
			stddev = 0.1
			dist_expert = self.actor(obs_expert, base_action_expert, stddev)
			action_expert_offset = dist_expert.sample(clip=self.stddev_clip) * self.offset_scale_factor

			true_offset = torch.zeros(action_expert_offset.shape).to(self.device)
			log_prob_expert = dist_expert.log_prob(true_offset).sum(-1, keepdim=True)
			actor_loss += - log_prob_expert.mean()*bc_weight*0.03

		# optimize actor
		self.actor_opt.zero_grad(set_to_none=True)
		actor_loss.backward()
		self.actor_opt.step()
		
		# if self.use_tb:
		metrics['actor_loss'] = actor_loss.item()
		metrics['actor_logprob'] = log_prob.mean().item()
		metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
		metrics['actor_q'] = Q.mean().item()
		if bc_regularize and self.bc_weight_type == "qfilter":
			metrics['actor_qf'] = Q_qf.mean().item()
		metrics['bc_weight'] = bc_weight
		metrics['regularized_rl_loss'] = -Q.mean().item()* (1-bc_weight)
		metrics['rl_loss'] = -Q.mean().item()
		if bc_regularize:
			metrics['regularized_bc_loss'] = - log_prob_expert.mean().item()*bc_weight*0.03
			metrics['bc_loss'] = - log_prob_expert.mean().item()*0.03
			
		return metrics

	def update(self, replay_iter, expert_replay_iter, ssl_replay_iter, step, bc_regularize=False):
		metrics = dict()

		if step % self.update_every_steps != 0:
			return metrics

		batch = next(replay_iter)
		obs, action, base_action, reward, discount, next_obs, base_next_action = utils.to_torch(
			batch, self.device)

		# augment
		if self.augment:
			obs_qfilter = self.aug(obs.clone().float())
			obs = self.aug(obs.float())
			next_obs = self.aug(next_obs.float())
		else:
			obs_qfilter = obs.clone().float()
			obs = obs.float()
			next_obs = next_obs.float()

		# Get the representations
		reprs = []
		next_reprs = [] # These will be concatenated afterwards
		if 'image' in self.policy_representations:
			# Current representations
			image_obs = self.image_normalize(obs['image_obs']).to(self.device) # This will give all the image observations of one batch
			image_reprs = self.image_encoder(image_obs)
			reprs.append(image_reprs)

			# Next representations
			image_next_obs = self.image_normalize(next_obs['image_obs']).to(self.device)
			with torch.no_grad():
				image_next_reprs = self.image_encoder(image_next_obs)
			next_reprs.append(image_next_reprs)

		if 'tactile' in self.policy_representations:
			tactile_reprs = obs['tactile_repr'].to(self.device) # This will give all the representations of one batch
			reprs.append(tactile_reprs)

			tactile_next_reprs = next_obs['tactile_repr'].to(self.device)
			next_reprs.append(tactile_next_reprs)	

		obs = torch.concat([reprs], axis=-1) # Concatenate the representations to get the final representations
		next_obs = torch.concat([next_reprs], axis=-1)

		# encode
		# obs = self.normalize(obs/255.0)
		# next_obs = self.normalize(next_obs/255.0)
		# obs = self.encoder(obs)
		# with torch.no_grad():
		# 	next_obs = self.encoder(next_obs)

		# if bc_regularize:
		# 	batch = next(expert_replay_iter)
		# 	obs_expert, action_expert = utils.to_torch(batch, self.device)
		# 	action_expert = action_expert.float()
		# 	if self.k == 1:
		# 		base_action_expert = action_expert.clone()
		# 	else:
		# 		base_action_expert = self.vinn_act(obs_expert.clone())
		# 	# augment
		# 	if self.use_encoder and self.augment:
		# 		obs_expert = self.aug(obs_expert.float())
		# 	else:
		# 		obs_expert = obs_expert.float()
		# 	# encode
		# 	if bc_regularize and self.bc_weight_type=="qfilter":
		# 		if self.encoder_type != 'r3m':
		# 			obs_qfilter = self.normalize(obs_qfilter/255.0) if self.normalize else obs_qfilter
		# 		obs_qfilter = self.encoder_vinn(obs_qfilter) if self.use_encoder else obs_qfilter
		# 		obs_qfilter = obs_qfilter.detach()
		# 	else:
		# 		obs_qfilter = None
		# 		base_action_expert = None
		# 	if self.encoder_type != 'r3m':
		# 		obs_expert = self.normalize(obs_expert/255.0) if self.normalize else obs_expert
		# 	obs_expert = self.encoder(obs_expert) if self.use_encoder else obs_expert 
		# 	# Detach grads
		# 	obs_expert = obs_expert.detach()
		# else: # TODO: Fix this if you were to use bc_regularize
		obs_qfilter = None
		obs_expert = None 
		action_expert = None
		base_action_expert = None

		# if self.use_tb:
		metrics['batch_reward'] = reward.mean().item()

		# update critic
		metrics.update(
			self.update_critic(obs, action, base_next_action, reward, discount, next_obs, step))

		# update actor
		metrics.update(self.update_actor(obs.detach(), obs_expert, obs_qfilter, action_expert, base_action, base_action_expert, bc_regularize, step))

		# update critic target
		utils.soft_update_params(self.critic, self.critic_target,
								 self.critic_target_tau)

		return metrics

	def ot_rewarder(self, episode_obs):
		
		# TODO: Get the observations separately - instead of putting them directly we'll need
		# to have them separately - so the observations will be observations[pixels] and observations[tactile]
		
		# TODO: Get the image obs like this when we
		curr_reprs, exp_reprs = [], []
		if 'image' in self.reward_representations:
			image_obs = self.image_normalize(episode_obs['image_obs']).to(self.device) # This will give all the image observations of one episode
			image_reprs = self.image_encoder(image_obs)
			expert_image_reprs = self.image_encoder(self.expert_demo['image_obs'].to(self.device))
			curr_reprs.append(image_reprs)
			exp_reprs.append(expert_image_reprs)

		if 'tactile' in self.reward_representations:
			tactile_reprs = episode_obs['tactile_repr'].to(self.device) # This will give all the representations of one episode
			expert_tactile_reprs = self.expert_demo['tactile_repr'].to(self.device)
			curr_reprs.append(tactile_reprs)
			exp_reprs.append(expert_tactile_reprs)

		# print('image_reprs.shape: {}, tactile_reprs.shape: {}, expert_image_repre.shape: {}, expr_tactile.shape: {}'.format(
		# 	image_reprs.shape, tactile_reprs.shape, expert_image_reprs.shape, expert_tactile_reprs.shape
		# ))

		# Concatenate everything now
		obs = torch.concat(curr_reprs, dim=-1).detach()
		exp = torch.concat(exp_reprs, dim=-1).detach()

		print('obs.shape: {}, exp.shape: {}'.format(obs.shape, exp.shape))
			
		if self.rewards == 'sinkhorn_cosine':
			cost_matrix = cosine_distance(
				obs, exp)  # Get cost matrix for samples using critic network.
			transport_plan = optimal_transport_plan(
				obs, exp, cost_matrix, method='sinkhorn',
				niter=100).float()  # Getting optimal coupling
			ot_rewards = -self.sinkhorn_rew_scale * torch.diag(
				torch.mm(transport_plan,
							cost_matrix.T)).detach().cpu().numpy()
			
		elif self.rewards == 'sinkhorn_euclidean':
			cost_matrix = euclidean_distance(
				obs, exp)  # Get cost matrix for samples using critic network.
			transport_plan = optimal_transport_plan(
				obs, exp, cost_matrix, method='sinkhorn',
				niter=100).float()  # Getting optimal coupling
			ot_rewards = -self.sinkhorn_rew_scale * torch.diag(
				torch.mm(transport_plan,
							cost_matrix.T)).detach().cpu().numpy()
			
		elif self.rewards == 'cosine':
			exp = torch.cat((exp, exp[-1].unsqueeze(0)))
			ot_rewards = -(1. - F.cosine_similarity(obs, exp))
			ot_rewards *= self.sinkhorn_rew_scale
			ot_rewards = ot_rewards.detach().cpu().numpy()
			
		elif self.rewards == 'euclidean':
			exp = torch.cat((exp, exp[-1].unsqueeze(0)))
			ot_rewards = -(obs - exp).norm(dim=1)
			ot_rewards *= self.sinkhorn_rew_scale
			ot_rewards = ot_rewards.detach().cpu().numpy()
			
		else:
			raise NotImplementedError()

		return ot_rewards

	# TODO: implement these in the training time
	# def save_snapshot(self):
	# 	keys_to_save = ['actor', 'critic']
	# 	if self.use_encoder:
	# 		keys_to_save += ['encoder']
	# 	payload = {k: self.__dict__[k] for k in keys_to_save}
	# 	return payload

	# def load_snapshot(self, payload):
	# 	for k, v in payload.items():
	# 		if k == 'encoder':
	# 			self.encoder_vinn = v
	
	# 	self.critic_target.load_state_dict(self.critic.state_dict())
	# 	if self.use_encoder:
	# 		self.encoder.load_state_dict(self.encoder_vinn.state_dict())
	# 		self.encoder_target.load_state_dict(self.encoder.state_dict())

	# 	if self.use_encoder:
	# 		self.encoder_vinn.eval()
	# 		for param in self.encoder_vinn.parameters():
	# 			param.requires_grad = False
	
	# 		self.encoder.eval()
	# 		for param in self.encoder.parameters():
	# 			param.requires_grad = False
	# 		self.encoder_target.eval()
	# 		for param in self.encoder_target.parameters():
	# 			param.requires_grad = False

	# 	self.actor_opt = torch.optim.Adam(self.actor.policy.parameters(), lr=self.lr)
	# 	self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)


	# def load_snapshot_eval(self, payload, bc=False):
	# 	for k, v in payload.items():
	# 		self.__dict__[k] = v
	# 	self.critic_target.load_state_dict(self.critic.state_dict())
	# 	if self.use_encoder:
	# 		self.encoder_target.load_state_dict(self.encoder.state_dict())