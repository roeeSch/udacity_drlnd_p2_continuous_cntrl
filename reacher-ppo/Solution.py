# Reacher - PPO
# Import packages

import time
import argparse
import pickle
from scipy import signal

import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from unityagents import UnityEnvironment
from IPython.display import clear_output

from model import PPOPolicyNetwork
from agent import PPOAgent


def initializeEnv():
    ## Create Unity environment
    env = UnityEnvironment(file_name="Reacher_Linux_multAgents/Reacher.x86_64")
    # env = UnityEnvironment(file_name="Reacher_Linux_1agent/Reacher.x86_64")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    # Set configurations:
    config = {
        'environment': {
            'state_size':  env_info.vector_observations.shape[1],
            'action_size': brain.vector_action_space_size,
            'number_of_agents': len(env_info.agents)
        }
    }

    return env, brain, brain_name, config

def merge_cnfg(config_trgt, config_ref=None):
    for key in config_ref.keys():
        config_trgt[key] = config_ref[key]
    return config_trgt

def play_round(env, brain_name, policy, config, train=True):
    env_info = env.reset(train_mode=train)[brain_name]
    states = env_info.vector_observations                 
    scores = np.zeros(config['environment']['number_of_agents'])                         
    while True:
        actions, _, _, _ = policy(states)
        env_info = env.step(actions.cpu().detach().numpy())[brain_name]
        next_states = env_info.vector_observations         
        rewards = env_info.rewards                         
        dones = env_info.local_done                     
        scores += env_info.rewards                      
        states = next_states                               
        if np.any(dones):                                  
            break
    
    return np.mean(scores)


def ppo(env, brain_name, policy, config, train):
    if train:
        optimizier = optim.Adam(policy.parameters(), config['hyperparameters']['adam_learning_rate'], 
                        eps=config['hyperparameters']['adam_epsilon'])
        agent = PPOAgent(env, brain_name, policy, optimizier, config)
        all_scores = []
        averages = []
        last_max = 30.0

        for i in tqdm.tqdm(range(config['hyperparameters']['episode_count'])):
            agent.step()
            last_mean_reward = play_round(env, brain_name, policy, config)
            if i == 0:
                last_average = last_mean_reward
            else:
                last_average = np.mean(np.array(all_scores[-100:])) if len(all_scores) > 100 else np.mean(np.array(all_scores))

            all_scores.append(last_mean_reward)
            averages.append(last_average)
            if last_average > last_max:
                torch.save(policy.state_dict(), f"reacher-ppo/models/ppo-max-hiddensize-{config['hyperparameters']['hidden_size']}.pth")
                last_max = last_average
            clear_output(True)
            print('Episode: {} Total score this episode: {} Last {} average: {}'.format(i + 1, last_mean_reward, min(i + 1, 100), last_average))
        return all_scores, averages
    else:
        all_scores = []
        for i in range(20):
            score = play_round(env, brain_name, policy, config, train)
            all_scores.append(score)
        return [score], [np.mean(score)]


def arg_parse():
    parser = argparse.ArgumentParser(description='Args for Reacher - PPO')
    parser.add_argument('--learn', help='learn new policy', action='store_true', default=False,
                        dest='learnNewPolicy')
    parser.add_argument('--pltLrn', help='plot progress of previous learning session', action='store_true',
                        default=False, dest='pltLrn')
    parser.add_argument('--playRound', help='play round of last learned policy', action='store_true',
                        default=False, dest='playRound')
    parser.add_argument('--numEpisodes', help='number of episodes for this session (default = 100)', action='store',
                        default=150, dest='numEpisodes', type=int)

    parser.add_argument('--gridSrchHidden', nargs='+', type=int, default=[])

    args = parser.parse_args()

    # Set configurations:
    config = {
        'pytorch': {
            'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        },
        'hyperparameters': {
            'discount_rate': 0.99,
            'tau': 0.95,
            'gradient_clip': 5,
            'rollout_length': 2048,
            'ppo_clip': 0.2,
            'log_interval': 2048,
            'max_steps': 1e5,
            'mini_batch_number': 32,
            'entropy_coefficent': 0.01,
            'episode_count': 100,
            'hidden_size': 512,
            'adam_learning_rate': 3e-4,
            'adam_epsilon': 1e-5
        }
    }

    if args.numEpisodes is not None:
        config['hyperparameters']['episode_count'] = args.numEpisodes


    return args, config


if __name__ == '__main__':



    args, config = arg_parse()

    if args.learnNewPolicy:
        print("Learning new policy...")
        env, brain, brain_name, config_env = initializeEnv()
        config = merge_cnfg(config, config_env)
        new_policy = PPOPolicyNetwork(config)
        all_scores, average_scores = ppo(env, brain_name, new_policy, config, train=True)

        dict_results = {'all_scores': all_scores, 'average_scores': average_scores}
        filename = 'results.pckl'
        with open(filename, 'wb') as outfile:
            pickle.dump(dict_results, outfile)

        plt.figure(1)
        plt.plot(all_scores)
        plt.plot(average_scores)
        plt.legend(('current score', 'average 100 epi'))
        plt.xlabel('episode')
        plt.savefig('learning_rates.png')
        plt.ion()
        plt.show()
        plt.pause(0.001)
        time.sleep(5)
        env.close()

    if args.pltLrn:
        print('plotting learning progression VS episode of saved session (results.pckl) ...')
        with open('results.pckl', 'rb') as fid:
            dictRes = pickle.load(fid)
        ff = lambda x, n: signal.filtfilt(np.ones(n), float(n), x)
        plt.figure(2)
        plt.plot(dictRes['all_scores'])
        plt.plot(ff(dictRes['all_scores'], 20))
        plt.legend(('epi score', 'avg 20 epi'))
        plt.savefig('learning_rates_filtfilt20.png')
        plt.ion()
        plt.show()
        plt.pause(0.001)
        time.sleep(5)

    if args.playRound:
        print('playing a few rounds with saved weights...')
        env, brain, brain_name, config_env = initializeEnv()
        config = merge_cnfg(config, config_env)
        config['hyperparameters']['hidden_size'] = 128
        policy = PPOPolicyNetwork(config)
        policy.load_state_dict(torch.load('reacher-ppo/models/ppo-max-hiddensize-128.pth'))
        _, _ = ppo(env, brain_name, policy, config, train=False)
        env.close()

    if args.gridSrchHidden and not args.learnNewPolicy:
        print("Grid Searching hidden size:")
        print(args.gridSrchHidden)
        env, brain, brain_name, config_env = initializeEnv()
        config = merge_cnfg(config, config_env)
        for hidden_size in args.gridSrchHidden:
            config['hyperparameters']['hidden_size'] = hidden_size
            new_policy = PPOPolicyNetwork(config)
            all_scores, average_scores = ppo(env, brain_name, new_policy, config, train=True)

            dict_results = {'all_scores': all_scores, 'average_scores': average_scores}
            filename = 'results_hidden' + str(hidden_size) + '_300episodes.pckl'
            with open(filename, 'wb') as outfile:
                pickle.dump(dict_results, outfile)

            plt.figure(1)
            plt.plot(all_scores)
            plt.plot(average_scores)
            plt.legend(('current score', 'average 100 epi'))
            plt.xlabel('episode')
            plt.savefig('learning_rates_hidden' + str(hidden_size) + '_' + str(config['hyperparameters']['episode_count']) + 'episodes.png')
            plt.ion()
            plt.show()
            plt.pause(0.001)
            time.sleep(5)
            plt.close(1)

        env.close()

