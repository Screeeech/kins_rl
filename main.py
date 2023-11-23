import kins
import numpy as np
import matplotlib.pyplot as plt

num_lines = 2
max_len = 10
params = [.1, .9]
KinsEnv = kins.KinsEnv(num_lines, max_len, params)
sarsa = kins.SarsaAgent(KinsEnv, alpha=.1, gamma=0, epsilon=.05, n=5, episode_length=100)
rewards, coverage = sarsa.learn(num_episodes=10000)
pretty = sarsa.pretty_Q()

# create 4 subplots
fig, axs = plt.subplots(3, 2)
# plot rewards and overlay the moving average
axs[0, 0].plot(rewards, label='reward', alpha=.5, color='blue')
axs[0, 0].plot(np.convolve(rewards, np.ones(20)/20, mode='valid'), label='moving average', color='blue')
axs[0, 0].legend()


axs[0, 0].set_title('Rewards')
axs[0, 0].set_xlabel('Episode')
axs[0, 0].set_ylabel('Reward')
axs[0, 0].set_xticks(np.arange(0, len(rewards), len(rewards)//5))
i = 0
for ax in axs.flat[1:4]:
    im = ax.imshow(pretty[i], cmap='inferno', interpolation='nearest')
    ax.set_title('Q for adding to line {}'.format(i))
    ax.set_xlabel('line 1')
    ax.set_ylabel('line 0')
    ax.set_xticks(np.arange(0, max_len+1, 1))
    ax.set_yticks(np.arange(0, max_len+1, 1))
    i += 1

im = axs[2, 0].imshow(np.sign(pretty[0]-pretty[1]), cmap='inferno', interpolation='nearest')
axs[2, 0].set_title('line0-line1')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
fig.suptitle('SARSA Agent with lines of [.1, .2]')

axs[2, 1].plot(coverage)
axs[2, 1].set_title('Coverage')
axs[2, 1].set_xlabel('Episode')
axs[2, 1].set_ylabel('Coverage')
axs[2, 1].set_xticks(np.arange(0, len(coverage), len(coverage)//5))

fig.set_size_inches(8, 8)
plt.subplots_adjust(hspace=0.4, wspace=0.305)
plt.show()
