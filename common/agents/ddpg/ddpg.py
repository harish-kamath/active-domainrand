import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

from common.models.actor_critic import Actor, Critic, Dynamics
from .mbrl import ProbabilisticEnsemble, MBRLTrainer

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPG(object):
    def __init__(self, state_dim, action_dim, wandb_writer, agent_name='baseline', max_action=1.):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.dynamics = ProbabilisticEnsemble(ensemble_size=5, 
        obs_dim=state_dim, 
        action_dim=action_dim,
        hidden_sizes=[256,256])
        self.dynamics_trainer = MBRLTrainer(ensemble=self.dynamics, wandb_writer=wandb_writer)

        # self.dynamics = Dynamics(state_dim, action_dim).to(device)
        # self.dynamics_optimizer = torch.optim.Adam(self.dynamics.parameters())

        self.max_action = max_action

        self.agent_name = agent_name

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        return self.actor(state).cpu().data.numpy()
    
    def select_value(self, state, action):
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        return self.critic(state,action).cpu().data.numpy()
    
    def select_next_state_dist(self, state, action):
        dyn_inp = np.concatenate([state, action], axis=1)
        dyn_inp = torch.FloatTensor(dyn_inp).to(device)
        samples, mean, logstd = self.dynamics(dyn_inp)
        samples = samples.cpu().data.numpy()
        mean = mean.cpu().data.numpy()
        std = torch.exp(logstd).cpu().data.numpy()
        return samples, mean, std

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005):
        x,y,u,r,d = replay_buffer.sample(len(replay_buffer.storage))
        full_data = np.concatenate([x,u,r,d,y],axis=1)
        logger.info("Training Dynamics")
        self.dynamics_trainer.train_from_buffer(full_data)
        logger.info("Finished training dynamics!")
        for it in range(iterations):
            # Sample replay buffer 
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            # diff_state = next_state - state
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Train Dynamics
            # pred_diff_state = self.dynamics(state, action)
            # dynamics_loss = F.mse_loss(pred_diff_state, diff_state)
            # self.dynamics_optimizer.zero_grad()
            # dynamics_loss.backward()
            # self.dynamics_optimizer.step()

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename), map_location=device))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename), map_location=device))

    # To ensure backwards compatibility D:
    def load_model(self):
        cur_dir = os.getcwd()
        actor_path = 'common/agents/ddpg/saved_model/{}_{}.pth'.format(self.agent_name, 'actor')
        critic_path = 'common/agents/ddpg/saved_model/{}_{}.pth'.format(self.agent_name, 'critic')

        self.actor.load_state_dict(torch.load(os.path.join(cur_dir, actor_path), map_location=device))
        self.critic.load_state_dict(torch.load(os.path.join(cur_dir, critic_path), map_location=device))
