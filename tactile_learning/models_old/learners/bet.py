import torch

from behavior_transformer import BehaviorTransformer, GPT, GPTConfig
from tactile_dexterity.learners import Learner


class BETLearner(Learner):
    def __init__(
        self,
        obs_dim,
        n_clusters,
        kmeans_fit_steps,
        weight_decay, 
        learning_rate, 
        betas
    ):
        self.bet = BehaviorTransformer(
            obs_dim = obs_dim, # Tactile and image together
            act_dim = 23, # Allegro and Kinova
            n_clusters = n_clusters,
            kmeans_fit_steps=kmeans_fit_steps,
            gpt_model = GPT(
                GPTConfig(
                    block_size=144,
                    input_dim=obs_dim,
                    n_lyer=6,
                    n_head=8,
                    n_embd=256,
                )
            )
        )

        self.optimizer = self.bet.configure_optimizers(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas
        )

        
    

# conditional = False
# obs_dim = 50
# act_dim = 8
# goal_dim = 50 if conditional else 0
# K = 32
# T = 16
# batch_size = 256

# cbet = BehaviorTransformer(
#     obs_dim=obs_dim,
#     act_dim=act_dim,
#     goal_dim=goal_dim,
#     gpt_model=GPT(
#         GPTConfig(
#             block_size=144,
#             input_dim=obs_dim,
#             n_layer=6,
#             n_head=8,
#             n_embd=256,
#         )
#     ),  # The sequence model to use.
#     n_clusters=K,  # Number of clusters to use for k-means discretization.
#     kmeans_fit_steps=5,  # The k-means discretization is done on the actions seen in the first kmeans_fit_steps.
# )

# optimizer = cbet.configure_optimizers(
#     weight_decay=2e-4,
#     learning_rate=1e-5,
#     betas=[0.9, 0.999],
# )

for i in range(10):
    obs_seq = torch.randn(batch_size, T, obs_dim)
    goal_seq = torch.randn(batch_size, T, goal_dim)
    action_seq = torch.randn(batch_size, T, act_dim)
    if i <= 7:
        # Training.
        train_action, train_loss, train_loss_dict = cbet(obs_seq, goal_seq=None, action_seq=action_seq)
    else:
        # Action inference
        eval_action, eval_loss, eval_loss_dict = cbet(obs_seq, goal_seq=None, action_seq=None)