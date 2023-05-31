from gym.envs.registration import register 


register(
	id='CupUnstacking-v1',
	entry_point='dexterous_env.cup_unstacking_env:CupUnstackingEnv',
	max_episode_steps=76,
)

register(
	id='BowlUnstacking-v1',
	entry_point='dexterous_env.bowl_unstacking_env:BowlUnstackingEnv',
	max_episode_steps=76,
)

register(
    id = 'PlierPicking-v1',
    entry_point='dexterous_env.plier_picking_env:PlierPicking',
	max_episode_steps=80, # For the 16th demo
)