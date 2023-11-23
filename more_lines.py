import kins
import numpy as np
import matplotlib.pyplot as plt

num_lines = 4
max_len = 10
params = [.1, .2, .05, .05]
KinsEnv = kins.KinsEnv(num_lines, max_len, params)
sarsa = kins.SarsaAgent(KinsEnv, alpha=.1, gamma=.1, epsilon=.05, n=1, episode_length=10)
rewards, coverage = sarsa.learn(num_episodes=1000)

# create 2 subplots
fig, axs = plt.subplots(2, 1)
axs[0].plot(rewards, label='reward', alpha=.5, color='blue')
axs[0].plot(np.convolve(rewards, np.ones(10)/10, mode='valid'), label='moving average', color='blue')
axs[0].legend()
axs[0].set_title('Rewards')
axs[0].set_xlabel('Episode')
axs[0].set_ylabel('Reward')
axs[0].set_xticks(np.arange(0, len(rewards), len(rewards)//5))

axs[1].plot(coverage)
axs[1].set_title('Coverage')
axs[1].set_xlabel('Episode')
axs[1].set_ylabel('Coverage')
axs[1].set_xticks(np.arange(0, len(coverage), len(coverage)//5))


plt.show()

