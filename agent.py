import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import gym
import numpy as np

from network import PolicyNet, ValueNet
from buffer import SampleBuffer

class Agent():

    clip_param = 0.2
    ppo_epoch = 10

    def __init__(self, env: gym.Env, epoch_steps = 2048, batch_size = 64):
        self.epoch_steps = epoch_steps
        self.batch_size = batch_size
        self.training_step = 0
        self.actor_net = PolicyNet(env.observation_space.shape[0], env.action_space.shape).float()
        self.critic_net = ValueNet(env.observation_space.shape[0]).float()
        self.counter = 0
        self.optimizer = optim.Adam([
            {'params': self.actor_net.parameters(), 'lr': 0.0003},
            {'params': self.critic_net.parameters(), 'lr': 0.001},
        ])

    def save(self, path: str):
        param_dict = {
            'actor_param': self.actor_net.state_dict(),
            'critic_param': self.critic_net.state_dict()
        }
        torch.save(param_dict, path)

    def select_action(self, state, action = None):
        state = torch.from_numpy(state).float()
        mu, log_std = self.actor_net(state)
        dist = torch.distributions.Normal(mu, torch.exp(log_std))
        if action is None:
            with torch.no_grad():
                action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, dist.entropy()

    def get_value(self, state):
        state = torch.from_numpy(state).float()
        state_value = self.critic_net(state)
        return state_value
    
    def update(self, buffer: SampleBuffer, epoch: int, writer: SummaryWriter):

        self.training_step += 1
        self.actor_net.train()
        self.critic_net.train()

        s = buffer.s
        a = buffer.a
        ret = torch.from_numpy(buffer.ret.reshape(-1, 1))
        old_log_probs = torch.from_numpy(buffer.a_lp)

        value_losses = []
        actor_losses = []

        for _ in range(self.ppo_epoch):

            value = self.get_value(s).squeeze(0).detach().numpy()
            advantage = ret - value
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            for index in BatchSampler(SubsetRandomSampler(range(buffer.size)), self.batch_size, False):

                _, log_probs, entropy = self.select_action(s[index], torch.from_numpy(a[index]))
                ratio = torch.exp(log_probs - old_log_probs[index])

                if torch.any(torch.isnan(log_probs)) or torch.any(torch.isnan(ratio)):
                    print('Encountered nan!')
                    continue

                surr1 = ratio * advantage[index]
                surr2 = torch.clamp(ratio, 1. - self.clip_param, 1. + self.clip_param) * advantage[index]
                surr = torch.min(surr1, surr2)
                
                if torch.any(torch.isnan(surr)):
                    print('Encountered nan!')
                    continue

                actor_loss = -surr.mean()
                value_loss = torch.mean(( ret[index] - self.get_value(s[index]) ) ** 2)

                loss = actor_loss + 0.5*value_loss - 0.00*entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                actor_losses.append(actor_loss.item())
                value_losses.append(value_loss.item())

                for param in self.actor_net.parameters():
                    if torch.any(torch.isnan(param)):
                        print('actor_net Has nan params!!')

                for param in self.critic_net.parameters():
                    if torch.any(torch.isnan(param)):
                        print('critic_net Has nan params!!')
            
        writer.add_scalar('loss/actor', np.average(actor_losses), epoch)
        writer.add_scalar('loss/critic', np.average(value_losses), epoch)
        writer.add_scalar('pi/entropy', entropy.mean(), epoch)
        writer.add_scalar('pi/mean', a.mean(), epoch)
        writer.add_scalar('pi/std', a.std().mean(), epoch)