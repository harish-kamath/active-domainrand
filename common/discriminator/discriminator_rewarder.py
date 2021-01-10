import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from common.models.discriminator import MLPDiscriminator
from common.utils.rollout_evaluation import evaluate_policy
import logging

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiscriminatorRewarder(object):
    def __init__(self, reference_env, randomized_env_id, discriminator_batchsz, reward_scale,
                 load_discriminator, discriminator_lr=3e-3, add_pz=True, use_new_discriminator='default', agent_policy=None, max_env_timesteps=0):
        self.discriminator = MLPDiscriminator(
            state_dim=reference_env.observation_space.shape[0],
            action_dim=reference_env.action_space.shape[0]).to(device)

        self.discriminator_criterion = nn.BCELoss()
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=discriminator_lr)
        self.reward_scale = reward_scale
        self.batch_size = discriminator_batchsz 
        self.add_pz = add_pz
        self.use_new_discriminator=use_new_discriminator
        self.agent_policy = agent_policy
        self.reference_env = reference_env
        self.max_env_timesteps=max_env_timesteps
        self.previous_ref_rew = 0

        if load_discriminator:
            self._load_discriminator(randomized_env_id)

    def calculate_rewards(self, randomized_trajectory):
        """Discriminator based reward calculation
        We want to use the negative of the adversarial calculation (Normally, -log(D)). We want to *reward*
        our simulator for making it easier to discriminate between the reference env + randomized onea
        """
        score, _, _ = self.get_score(randomized_trajectory)
        reward = np.log(score)

        if self.add_pz:
            reward -= np.log(0.5)

        return self.reward_scale * reward

    def get_score(self, trajectory):
        """Discriminator based reward calculation
        We want to use the negative of the adversarial calculation (Normally, -log(D)). We want to *reward*
        our simulator for making it easier to discriminate between the reference env + randomized onea
        """
        if self.use_new_discriminator == 'perfdiff':
            ref = self.ref_rewards
            rewards = trajectory[:,-1].squeeze()
            rew = np.sum(rewards)
            pd = abs(rew - ref)
            self.ref_rew = ref
            return pd,pd,0
        trajectory = trajectory[:,:-1]

        if self.use_new_discriminator == 'modeladv':
            ma_vals = []
            curr_state = trajectory[:,:self.reference_env.observation_space.shape[0]]
            next_state = trajectory[:,-self.reference_env.observation_space.shape[0]:]
            action = trajectory[:,self.reference_env.observation_space.shape[0]:-self.reference_env.observation_space.shape[0]]

            v_n = self.agent_policy.select_value(next_state, action)
            _, pred_ns, std_ns = self.agent_policy.select_next_state_dist(curr_state,action)
            pred_ns = pred_ns[0,:,:self.reference_env.observation_space.shape[0]]
            std_ns = std_ns[0,:,:self.reference_env.observation_space.shape[0]]
            temp_values = []
            for i in range(30):
                eps = np.random.randn(*std_ns.shape)
                sample = pred_ns + std_ns * eps
                temp_v_n = self.agent_policy.select_value(sample,action)
                temp_values.append(temp_v_n)
            temp_values = np.array(temp_values)
            temp_values = temp_values.squeeze().T
            ev_n = self.agent_policy.select_value(pred_ns,action)
            ma_vals = (v_n - ev_n).squeeze()
            rew = np.abs(ma_vals)/np.std(temp_values,axis=1)
            return np.amax(rew),np.median(rew),0

        traj_tensor = self._trajectory2tensor(trajectory).float()
        with torch.no_grad():
            score = (self.discriminator(traj_tensor).cpu().detach().numpy()+1e-8)
            return score.mean(), np.median(score), np.sum(score)

    def train_discriminator(self, reference_trajectory, randomized_trajectory, iterations):
        """Trains discriminator to distinguish between reference and randomized state action tuples
        """
        for _ in range(iterations):
            randind = np.random.randint(0, len(randomized_trajectory[0]), size=int(self.batch_size))
            refind = np.random.randint(0, len(reference_trajectory[0]), size=int(self.batch_size))

            randomized_batch = self._trajectory2tensor(randomized_trajectory[randind])
            reference_batch = self._trajectory2tensor(reference_trajectory[refind])

            g_o = self.discriminator(randomized_batch)
            e_o = self.discriminator(reference_batch)

            self.discriminator_optimizer.zero_grad()

            discrim_loss = self.discriminator_criterion(g_o, torch.ones((len(randomized_batch), 1), device=device)) + \
                           self.discriminator_criterion(e_o, torch.zeros((len(reference_batch), 1), device=device))
            discrim_loss.backward()

            self.discriminator_optimizer.step()
    
    def get_ref_reward(self):
        ref_rewards = []
        for i in range(5):
            trajectory = evaluate_policy(nagents=10,
                                env=self.reference_env,
                                agent_policy=self.agent_policy,
                                replay_buffer=None,
                                eval_episodes=10,
                                max_steps=self.max_env_timesteps,
                                freeze_agent=True,
                                add_noise=False,
                                log_distances=False)
            for roll in trajectory: ref_rewards.append(np.sum(roll[:,-1].squeeze()))
        self.ref_rewards = np.mean(ref_rewards)
            


    def _load_discriminator(self, name, path='saved-models/discriminator/discriminator_{}.pth'):
        self.discriminator.load_state_dict(torch.load(path.format(name), map_location=device))

    def _save_discriminator(self, name, path='saved-models/discriminator/discriminator_{}.pth'):
        torch.save(self.discriminator.state_dict(), path.format(name))

    def _trajectory2tensor(self, trajectory):
        return torch.from_numpy(trajectory).float().to(device)
