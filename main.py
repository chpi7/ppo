import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
import gym
from tqdm import tqdm

from agent import Agent
from buffer import SampleBuffer

def main(env = 'LunarLanderContinuous-v2', steps = 1e6, steps_per_epoch = 1000, seed = 0, render = False, minibatch_size = 128, reward_gamma = .99):
    
    torch.manual_seed(seed)
    
    epochs = int(steps // steps_per_epoch)
    env = gym.make(env)

    env.seed(seed)

    writer = SummaryWriter()

    agent = Agent(env, steps_per_epoch, minibatch_size)

    # train for some number of epochs
    for epoch in tqdm(range(epochs), desc="epoch"):

        state = env.reset()
        rewards = []
        ep_returns = []
        ep_len = 0
        buffer = SampleBuffer(steps_per_epoch, env.observation_space.shape[0], env.action_space.shape[0], gamma=reward_gamma)

        agent.actor_net.eval()
        agent.critic_net.eval()

        # every epoch has a fixed number of timesteps (run and reset as needed)
        for t in range(steps_per_epoch):

            if render:
                env.render()
            
            with torch.no_grad():
                action, log_probs, _ = agent.select_action(np.array([state]))
            state_, reward, done, _ = env.step(torch.tanh(action[0]).numpy())

            buffer.store(state, action.numpy(), log_probs[0].numpy(), reward, state_)
            rewards.append(reward)
            
            state = state_
            ep_len += 1

            terminal = done or (ep_len == env.spec.max_episode_steps)
            if terminal or (t == steps_per_epoch - 1):
                
                if done and not (ep_len == env.spec.max_episode_steps):
                    buffer.finish_path(0)
                else:
                    state_val = agent.get_value(state)
                    buffer.finish_path(state_val)
                
                ep_returns.append(sum(rewards))
                state, ep_len, rewards = env.reset(), 0, []

        agent.update(buffer, epoch, writer)
        buffer.clear()
        
        avg_return = sum(ep_returns) / len(ep_returns)
        writer.add_scalar('ep_ret', avg_return, epoch)
        rewards = []

if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, default='LunarLanderContinuous-v2')
    p.add_argument('--steps', type = int, default = 1e6)
    p.add_argument('--steps_per_epoch', type = int, default = 1000)
    p.add_argument('--minibatch_size', type=int, default = 128)
    p.add_argument('--seed', type=int, default=2)
    p.add_argument('--reward_gamma', type=float, default=0.99)

    args = p.parse_args()

    main(**vars(args))
