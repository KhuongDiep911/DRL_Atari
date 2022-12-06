import gym
from gym import wrappers
import numpy as np
from dueling_ddqn_agent import DuelingDDQNAgent
from utils import plot_learning_curve, make_env
import os ###
################################################################################
# Python program to store list to file using pickle module
import pickle

# write list to binary file
def write_list(list, filename):
    # store list in binary file so 'wb' mode
    with open(filename, 'w+') as fp:
        for item in list:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Finish writing')   

# Read list to memory
def read_list():
    # for reading also binary mode is important
    with open('sampleList', 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list
###################################################################################

if __name__ == '__main__':
    env = make_env('PongNoFrameskip-v4')
    best_score = -np.inf
    load_checkpoint = False
    n_games = 2500
    agent = DuelingDDQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001,
                     input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=50000, eps_min=0.15,
                     batch_size=32, replace=10000, eps_dec=1e-5,
                     chkpt_dir='models/', algo='DuelingDDQNAgent',
                     env_name='PongNoFrameskip-v4')

    if load_checkpoint:
        agent.load_models()

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' \
            + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    n_steps = 0
    # scores, eps_history, steps_array = [], [], []
    episodes, mean_scores, scores, eps_history, steps_array = [], [], [], [], []
    for i in range(n_games):
        done = False
        observation = env.reset()

        score = 0
        i_steps = 0 ##
        agent.tensorboard_step = i ##
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward

            if not load_checkpoint:
                agent.store_transition(observation, action,
                                     reward, observation_, int(done))
                agent.learn()
            observation = observation_
            i_steps += 1 ###########
            n_steps += 1
            env.render()
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        mean_scores.append(avg_score) #########
        episodes.append(i+1)
        min_reward = np.min(scores) ##########
        max_reward = np.max(scores) ##########

        print('episode: ', i+1,'score: ', score,
             ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
            'epsilon %.2f' % agent.epsilon, 'steps', i_steps, 'total steps', n_steps)

        agent.on_epsiode_end(reward_avg=avg_score, reward_min=min_reward, reward_max=max_reward, n_steps=n_steps, i_steps=i_steps)#########

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)
        if load_checkpoint and n_steps >= 18000:
            break

    write_list(mean_scores, "save_files/Pong/mean_scores.txt")#############
    # write_list(max_reward, "save_files/max_scores.txt")
    write_list(scores, "save_files/Pong/scores.txt")
    write_list(episodes, "save_files/Pong/episodes.txt")
    write_list(eps_history, "save_files/Pong/epsilons.txt")
    write_list(steps_array, "save_files/Pong/steps_array.txt")

    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(steps_array, scores, eps_history, figure_file)
