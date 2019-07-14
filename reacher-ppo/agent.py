#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# Modified by Roee Schmidt (Jul 2019) for Udacity DRL project #2 in continuous control

import numpy as np
import torch
import torch.nn as nn

from utils import miniBatcher as mB

class PPOAgent(object):
    
    def __init__(self, environment, brain_name, policy_network, optimizier, config, train=True):
        self.config = config
        self.hyperparameters = config['hyperparameters']
        self.network = policy_network
        self.optimizier = optimizier
        self.total_steps = 0
        self.all_rewards = np.zeros(config['environment']['number_of_agents'])
        self.episode_rewards = []
        self.environment = environment
        self.brain_name = brain_name
        self.TrainMode = train
        env_info = environment.reset(train_mode=train)[brain_name]
        self.states = env_info.vector_observations              

    def step(self):
        rollout = []
        hyperparameters = self.hyperparameters

        ########################################################################################
        # Interact with environment for 'rollout_length'.
        # 1. For each time step Use state and actor-critic networks to obtain actions and values
        # 2. Use the predicted actions to advance environment. Obtaining next states, rewards and terminals
        # 3. Append [states, values, actions, log_probs, rewars, not(dones)] into rollout list
        # 4. Repeat with state = next state
        env_info = self.environment.reset(train_mode=self.TrainMode)[self.brain_name]
        self.states = env_info.vector_observations  
        states = self.states
        for _ in range(hyperparameters['rollout_length']):
            actions, log_probs, _, values = self.network(states)
            env_info = self.environment.step(actions.cpu().detach().numpy())[self.brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            terminals = np.array([1 if t else 0 for t in env_info.local_done])
            self.all_rewards += rewards

            # if episode finished then append tp list of accumulated episode_rewards
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.episode_rewards.append(self.all_rewards[i])
                    self.all_rewards[i] = 0
                    
            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), rewards, 1 - terminals])
            states = next_states

        self.states = states
        pending_value = self.network(states)[-1]
        # append last states and values into rollout list:
        rollout.append([states, pending_value, None, None, None, None])

        ########################################################################################
        # Process rollout to obtain advantages instead of values and returns instead of rewards:
        # Run on rollout from end-1 to start
        # 1. returns[i] = rewards[i] + gamma * returns[i+1] * not(dones)[i]
        # 2. td_error = rewards[i] + gamma * values[i+1] * not(dones)[i] - values[i]
        # 3. advantages[i] = advantages[i+1] * gae_tau * gamma * not(dones)[i] + td_error
        # 4. Append [states, actions, log_probs, returns, advantages] into processed_rollout
        processed_rollout = [None] * (len(rollout) - 1)
        advantages = torch.Tensor(np.zeros((self.config['environment']['number_of_agents'], 1)))
        returns = pending_value.detach()
        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, terminals = rollout[i]
            terminals = torch.Tensor(terminals).unsqueeze(1)
            rewards = torch.Tensor(rewards).unsqueeze(1)
            #actions = torch.Tensor(actions)
            states = torch.Tensor(states)
            next_value = rollout[i + 1][1]
            #returns = rewards + hyperparameters['discount_rate'] * terminals * returns
            returns = rewards.cuda() + hyperparameters['discount_rate'] * terminals.cuda() * returns

            #td_error = rewards + hyperparameters['discount_rate'] * terminals * next_value.detach() - value.detach()
            td_error = rewards.cuda() + hyperparameters[
                'discount_rate'] * terminals.cuda() * next_value.detach().cuda() - value.detach()
            advantages = advantages.cuda() * hyperparameters['tau'] * hyperparameters['discount_rate'] * \
                         terminals.cuda() + td_error
            processed_rollout[i] = [states, actions, log_probs, returns, advantages]

        states, actions, log_probs_old, returns, advantages = \
            map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        advantages = (advantages - advantages.mean()) / advantages.std()

        ########################################################################################
        # Go aver all processed_rollout in minibatches while optimizing the networks for minimum loss:
        # 1. Split all experience (processed_rollout) into 'mini_batch_number' minibatches.
        # 2. Use actor-critic network and (states, actions) to obtain (log_probs, entropy_loss, values)
        # 3. Calculate action probability ratio (and clipped ratio) using log_probs and previous log_probs.
        # 4. Calculate objective be multiplying ratio with advantages and multiplying by -1 (for minimization).
        # 5. Policy_Loss is basically the average of advantages * ratio
        # 6. Value_Loss is the average squared error between values and discounted returns.
        # 7. Optimization step towards reducing (policy_loss + value_loss)
        batch_size = states.size(0) // hyperparameters['mini_batch_number']
        mBtchr = mB(dataLength=states.size(0), batch_size=batch_size, shuffle=True, num_workers=1)
        for batch_indices in mBtchr.generator:
            batch_indices.tolist()
            sampled_states = states[batch_indices]
            sampled_actions = actions[batch_indices]
            sampled_log_probs_old = log_probs_old[batch_indices]
            sampled_returns = returns[batch_indices]
            sampled_advantages = advantages[batch_indices]

            _, log_probs, entropy_loss, values = self.network(sampled_states, sampled_actions)
            ratio = (log_probs - sampled_log_probs_old).exp()
            obj = ratio * sampled_advantages
            obj_clipped = ratio.clamp(1.0 - hyperparameters['ppo_clip'],
                                      1.0 + hyperparameters['ppo_clip']) * sampled_advantages
            policy_loss = -torch.min(obj, obj_clipped).mean(0) - hyperparameters['entropy_coefficent'] * \
                          entropy_loss.cuda().mean()

            value_loss = 0.5 * (sampled_returns - values).pow(2).mean()

            self.optimizier.zero_grad()
            (policy_loss + value_loss).backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), hyperparameters['gradient_clip'])
            self.optimizier.step()

        steps = hyperparameters['rollout_length'] * self.config['environment']['number_of_agents']
        self.total_steps += steps
